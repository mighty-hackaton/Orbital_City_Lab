import os
import socket
import json
import time

import joblib

import feature_utils
import route_utils as ru

# ==========================================
# 1. КОНФІГУРАЦІЯ
# ==========================================
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

TIMEOUT_THRESHOLD = 1.5
STATE_FILE = "dashboard_state.json"
# Синхронізовано з SEND_PERIOD_SEC у клієнті (0.5с) — частіші апдейти дають
# плавнішу анімацію на дашборді замість "смиканого" руху раз на секунду.
TICK_SEC = 0.5
MIN_MOVE_FOR_HEADING_M = 1.0  # нижче — вважаємо GPS-джиттером, heading не чіпаємо

# Модель (v3_heading) тренована на кроці 15 сек: 5 значень швидкості й курсу,
# узятих РАЗ на 15 секунд, а не щосекунди. Годування моделі посекундним
# вікном — той самий клас багів, що вже був з heading: формат входу не
# збігається з тим, на чому тренувались, і прогноз тихо псується.
MODEL_SAMPLE_INTERVAL_SEC = 15.0
MODEL_HISTORY_LEN = 5

print("⏳ Отримую маршрут (з кешу route_info.json або рахую сам)...")
route_coords = ru.load_or_build_route()
print(f"✅ Маршрут готовий: {len(route_coords)} точок. (граф OSM серверу під час роботи не потрібен)")

print("⏳ Завантаження ML-моделі HistGradientBoostingRegressor (v3, з курсом)...")
try:
    ai_model = joblib.load("model.pkl")
    print("✅ Справжню модель підключено успішно!")
except Exception as e:
    print(f"⚠️ Помилка завантаження моделі: {e}")
    ai_model = None


# ==========================================
# 2. ПРОГНОЗ ШВИДКОСТІ
# ==========================================
def start_reb_prediction(state, current_time):
    """Викликається ОДИН РАЗ у момент входу в РЕБ — а не щотіку.

    Модель тренована на кроці 15 сек, тож повторний виклик з тим самим
    вікном (адже під час РЕБ нових реальних даних не надходить) не дає нової
    інформації, лише даремно навантажує CPU і плутає логи.

    Модель повертає ДЕЛЬТУ швидкості на 15 сек вперед:
        ціль = остання_відома_швидкість + delta
    Цю ціль далі лінійно розтягуємо на 15 секунд наперед (sample_interp_speed),
    як і написано в model_metadata.json — а не застосовуємо миттєво.
    """
    last_speed = state["speed_buffer"][-1] if state["speed_buffer"] else ru.BASE_SPEED_KMH
    speed_window = state.get("model_speed_window", [])
    heading_window = state.get("model_heading_window", [])

    geometric_target = ru.target_speed_for_position(route_coords, state["route_idx"])

    if len(speed_window) >= MODEL_HISTORY_LEN and ai_model is not None:
        try:
            features = feature_utils.build_features(speed_window, heading_window)
            delta = float(ai_model.predict([features])[0])
            model_target = speed_window[-1] + delta
            # Змішуємо прогноз моделі (довга памʼять — тренд за останню
            # хвилину) з геометричним орієнтиром (знає МІСЦЕ на дорозі,
            # чого в чистій історії швидкостей немає).
            target_speed = 0.5 * model_target + 0.5 * geometric_target
            print(f"    ↳ модель: {speed_window[-1]:.1f}+({delta:+.1f})={model_target:.1f} | "
                  f"геометрія: {geometric_target:.1f} | ціль на 15с: {target_speed:.1f} км/год")
        except Exception as e:
            print(f"⚠️ Помилка прогнозу: {e}")
            target_speed = geometric_target
    else:
        # Перші ~75 сек роботи (5 семплів × 15 сек) історії для моделі ще
        # не вистачає — це очікувано, а не баг. До того часу орієнтуємось
        # лише на геометрію маршруту.
        target_speed = geometric_target
        print(f"    ↳ недостатньо історії для моделі ({len(speed_window)}/{MODEL_HISTORY_LEN}), "
              f"ціль лише за геометрією: {target_speed:.1f} км/год")

    return {"start_time": current_time, "start_speed": last_speed, "target_speed": target_speed}


def sample_interp_speed(interp, current_time):
    """Лінійна інтерполяція поточної швидкості до цілі на MODEL_SAMPLE_INTERVAL_SEC
    секунд вперед. Після завершення інтервалу тримає ціль сталою — інакше,
    якщо РЕБ триває довше 15 сек, довелось би раз у раз прогнозувати з тих
    самих застарілих даних і накопичувати помилку."""
    elapsed = current_time - interp["start_time"]
    ratio = min(1.0, elapsed / MODEL_SAMPLE_INTERVAL_SEC)
    return interp["start_speed"] + ratio * (interp["target_speed"] - interp["start_speed"])


def save_state_atomic(vehicles, max_retries=5, retry_delay=0.05):
    """Запис через тимчасовий файл + os.replace.

    На Windows os.replace() кидає PermissionError, якщо цільовий файл у цю
    мілісекунду відкритий іншим процесом (типово — Streamlit-дашборд, який
    читає dashboard_state.json раз на секунду, або антивірус/синхронізація
    теки). Робимо кілька коротких повторних спроб, а якщо і вони не вдались —
    пропускаємо цей кадр запису замість падіння всього сервера."""
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(vehicles, f)

    for attempt in range(max_retries):
        try:
            os.replace(tmp, STATE_FILE)
            return
        except PermissionError:
            time.sleep(retry_delay)

    print(f"⚠️ Не вдалося оновити {STATE_FILE} (файл зайнятий іншим процесом), пропускаю кадр")
    try:
        os.remove(tmp)
    except OSError:
        pass


# ==========================================
# 3. ОСНОВНИЙ ЦИКЛ СЕРВЕРА
# ==========================================
vehicles = {}
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(0.5)

print(f"🚀 Сервер запущено. Слухаю порт {UDP_PORT}...")

while True:
    loop_start = time.time()
    current_time = loop_start

    # --- ПРИЙОМ ДАНИХ (ОНЛАЙН) ---
    try:
        data, addr = sock.recvfrom(1024)
        packet = json.loads(data.decode("utf-8"))
        v_id = packet["id"]

        if v_id not in vehicles:
            vehicles[v_id] = {
                "speed_buffer": [],
                "heading": 0.0,
                "route_idx": 0,
                "model_speed_window": [],
                "model_heading_window": [],
                "next_model_sample_time": current_time,
                "reb_interp": None,
            }

        state = vehicles[v_id]
        prev_lat, prev_lon = state.get("lat"), state.get("lon")

        # Heading рахуємо САМІ з реального переміщення між двома останніми
        # точками, а не з поля пакета — клієнт його не рахує.
        heading = state.get("heading", 0.0)
        if prev_lat is not None:
            moved = ru.calculate_distance(prev_lat, prev_lon, packet["lat"], packet["lon"])
            if moved > MIN_MOVE_FOR_HEADING_M:
                heading = ru.calculate_bearing(prev_lat, prev_lon, packet["lat"], packet["lon"])

        # Прив'язуємо машину до найближчої точки ВІДОМОГО маршруту. Пошук
        # обмежений околом попереднього route_idx, щоб не "перескочити" на
        # схожу, але не ту ділянку дороги.
        route_idx = ru.nearest_route_index(
            route_coords, packet["lat"], packet["lon"], search_from=state.get("route_idx", 0)
        )

        state.update({
            "last_packet_time": current_time,   # для рішення "звʼязок є / нема"
            "last_update_time": current_time,   # для розрахунку dt фізичного кроку
            "lat": packet["lat"],
            "lon": packet["lon"],
            "heading": heading,
            "route_idx": route_idx,
            "is_predicted": False,
            "reb_interp": None,  # звʼязок відновився — старий прогноз більше не актуальний
        })

        state["speed_buffer"].append(packet["speed"])
        if len(state["speed_buffer"]) > 5:
            state["speed_buffer"].pop(0)

        # Окремий буфер для МОДЕЛІ: семплюємо РАЗ на MODEL_SAMPLE_INTERVAL_SEC
        # секунд, а не щопакета — інакше вікно не відповідатиме тому, на
        # чому тренувались (посекундні дані замість 15-секундних).
        if current_time >= state["next_model_sample_time"]:
            state["model_speed_window"].append(packet["speed"])
            state["model_heading_window"].append(heading)
            if len(state["model_speed_window"]) > MODEL_HISTORY_LEN:
                state["model_speed_window"].pop(0)
                state["model_heading_window"].pop(0)
            state["next_model_sample_time"] = current_time + MODEL_SAMPLE_INTERVAL_SEC

        print(f"[🟢 ONLINE] Авто {v_id} | Швидкість: {packet['speed']:.1f} км/год | idx={route_idx}")

    except socket.timeout:
        pass
    except Exception as e:
        print(f"⚠️ Помилка прийому пакета: {e}")

    # --- АВТОНОМНИЙ РУХ (РЕБ) ---
    for v_id, state in vehicles.items():
        if "lat" not in state:
            continue
        if current_time - state.get("last_packet_time", 0) > TIMEOUT_THRESHOLD:
            if state.get("reb_interp") is None:
                state["reb_interp"] = start_reb_prediction(state, current_time)

            model_speed_now = sample_interp_speed(state["reb_interp"], current_time)
            last_actual_speed = state["speed_buffer"][-1] if state["speed_buffer"] else ru.BASE_SPEED_KMH

            # Той самий фізичний крок (обмежений темп розгону/гальмування),
            # що й у клієнта — щоб прогноз змінювався так само плавно, як
            # реальне авто, а не стрибав до цілі за один тік.
            predicted_speed = max(15.0, ru.step_speed_toward(last_actual_speed, model_speed_now))

            # dt рахуємо ФАКТИЧНИЙ, а не берем фіксований TICK_SEC. Раніше
            # last_seen перезаписувався на current_time щотіку REB-блоку,
            # через що умова "current_time - last_seen > TIMEOUT_THRESHOLD"
            # одразу ставала хибною — сервер фактично оновлював позицію не
            # щосекунди, а раз на TIMEOUT_THRESHOLD секунд, і при цьому рухав
            # машину так, ніби минула лише 1 секунда. Тепер last_packet_time
            # (рішення онлайн/офлайн) і last_update_time (для dt) розведені.
            dt = current_time - state.get("last_update_time", current_time)
            dt = max(0.0, min(dt, 5.0))  # захист від аномально великого dt (пауза дебагера тощо)
            speed_mps = predicted_speed * (1000 / 3600)

            new_lat, new_lon, new_idx, reached_end = ru.advance_along_route(
                route_coords, state["route_idx"], state["lat"], state["lon"], speed_mps * dt
            )

            if new_idx != state["route_idx"]:
                state["heading"] = ru.heading_for_route_idx(route_coords, new_idx)

            state.update({
                "lat": new_lat,
                "lon": new_lon,
                "route_idx": new_idx,
                "last_update_time": current_time,
                "is_predicted": True,
            })

            state["speed_buffer"].append(predicted_speed)
            if len(state["speed_buffer"]) > 5:
                state["speed_buffer"].pop(0)

            tag = "🏁 КІНЕЦЬ МАРШРУТУ" if reached_end else "🔴 РЕБ"
            print(f"[{tag}] Авто {v_id} | Швидкість: {predicted_speed:.1f} км/год | idx={new_idx}")

    save_state_atomic(vehicles)

    elapsed = time.time() - loop_start
    time.sleep(max(0.0, TICK_SEC - elapsed))