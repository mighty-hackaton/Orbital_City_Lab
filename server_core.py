import os
import socket
import json
import time

import joblib

import feature_utils
import route_utils as ru
import wifi_fingerprint as wifi_fp

# ==========================================
# 1. КОНФІГУРАЦІЯ
# ==========================================
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

# Повне мовчання борта довше цього — це вже не "GNSS глушиться", а взагалі
# немає зв'язку (GSM/LTE теж пропав). Тоді ми не отримуємо навіть свіжого
# Wi-Fi-скану, і лишається чисте AI+геометрія dead reckoning, як і було.
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

# Максимальний "стрибок" за один тік, коли Wi-Fi-корекція підтягує показ до
# скоригованої точки. EMA-згладжування тут НЕ рятує (перевірено емпірично):
# shadow fading в scan_at() не seeded, тож RSSI-скан (а з ним і сам матч)
# шумить від тіку до тіку навіть при незмінній істинній позиції, і матч
# систематично тягне назад (усереднення топ-5 фінгерпринтів дає центроїд
# уже пройденої ділянки, а не крайню точку). EMA лише пом'якшує цю тягу, але
# не прибирає — показ все одно час від часу відкочувався в межах сегмента.
# Замість цього показ ЗАВЖДИ просувається вперед щонайменше на dr_step_m
# (тими самими geometry-фізикою, що й dr — гарантовано вздовж маршруту), а
# Wi-Fi дозволено лише "підтягувати" показ БЛИЖЧЕ до dr, ніж просте
# просування вперед — інакше ігнорується цей тік. Це структурно виключає
# і зависання (завжди є forward-крок), і рух назад (корекція ніколи не
# застосовується, якщо вона віддаляє від dr).
MAX_WIFI_SNAP_M = 15.0

print("⏳ Отримую маршрут (з кешу route_info.json або рахую сам)...")
route_coords = ru.load_or_build_route()
cum_dist_arr = ru.build_cumulative_distances(route_coords)
print(f"✅ Маршрут готовий: {len(route_coords)} точок. (граф OSM серверу під час роботи не потрібен)")

print("⏳ Завантаження ML-моделі HistGradientBoostingRegressor (v3, з курсом)...")
try:
    ai_model = joblib.load("model.pkl")
    print("✅ Справжню модель підключено успішно!")
except Exception as e:
    print(f"⚠️ Помилка завантаження моделі: {e}")
    ai_model = None

# Wi-Fi fingerprint база — на реальному сервері це таблиця в БД, спільна
# для ВСЬОГО флоту (кожен борт і поповнює, і користується нею). Тут —
# процес-локальний Python-об'єкт, семантика та сама.
fp_store = wifi_fp.FingerprintStore()
print("✅ Wi-Fi fingerprint база ініціалізована (порожня, наповнюється в реальному часі).")


# ==========================================
# 2. ПРОГНОЗ ШВИДКОСТІ (AI + геометрія)
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

    # --- ПРИЙОМ ПАКЕТА (може мати gps_fix=True АБО False — див. client_sim.py) ---
    try:
        data, addr = sock.recvfrom(2048)
        packet = json.loads(data.decode("utf-8"))
        v_id = packet["id"]
        gps_fix = packet.get("gps_fix", True) and "lat" in packet
        wifi_scan = packet.get("wifi") or {}

        if v_id not in vehicles:
            if not gps_fix:
                # Перший-ліпший пакет від нового борта без координат —
                # нема з чого стартувати стан, чекаємо першого GNSS-фіксу.
                raise KeyError("no initial fix yet")
            vehicles[v_id] = {
                "speed_buffer": [],
                "heading": 0.0,
                "route_idx": 0,
                "model_speed_window": [],
                "model_heading_window": [],
                "next_model_sample_time": current_time,
                "reb_interp": None,
                "last_wifi_scan": {},
                "gps_fix_live": True,
                "wifi_status": "ok",
                "wifi_confidence": 0.0,
            }

        state = vehicles[v_id]
        state["last_packet_time"] = current_time  # "будь-який пакет" — вартовий повного мовчання

        if gps_fix:
            # --- GNSS живий: маємо справжню координату ---
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
                "last_update_time": current_time,   # для розрахунку dt фізичного кроку
                "lat": packet["lat"],
                "lon": packet["lon"],
                "heading": heading,
                "route_idx": route_idx,
                "is_predicted": False,
                "gps_fix_live": True,
                "reb_interp": None,  # звʼязок відновився — старий прогноз більше не актуальний
                # Сирий DR-акумулятор синхронізуємо з підтвердженою позицією —
                # інакше він тягнув би за собою offset, накопичений під час
                # попереднього РЕБ.
                "dr_lat": packet["lat"],
                "dr_lon": packet["lon"],
                "dr_idx": route_idx,
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

            # GNSS живий — це достовірна позиція, тож саме зараз записуємо
            # Wi-Fi fingerprint (а не під час РЕБ, коли позиція вже оцінка).
            cum_dist_here = ru.cumulative_distance_at(route_coords, cum_dist_arr, route_idx, packet["lat"], packet["lon"])
            fp_store.maybe_record(v_id, packet["lat"], packet["lon"], route_idx, wifi_scan, cum_dist_here)
            state["wifi_status"] = "ok" if wifi_scan else "no_ap"
            state["wifi_confidence"] = 0.0

            print(f"[🟢 GNSS] Авто {v_id} | Швидкість: {packet['speed']:.1f} км/год | idx={route_idx} | Wi-Fi AP: {len(wifi_scan)}")

        else:
            # --- GNSS глушиться, але пакет ДІЙШОВ — маємо свіжий Wi-Fi-скан ---
            state["gps_fix_live"] = False
            state["last_wifi_scan"] = wifi_scan
            print(f"[📶 GNSS ВТРАЧЕНО, Wi-Fi живий] Авто {v_id} | AP видно: {len(wifi_scan)}")

    except socket.timeout:
        pass
    except KeyError:
        pass
    except Exception as e:
        print(f"⚠️ Помилка прийому пакета: {e}")

    # --- DEAD RECKONING: коли GNSS не живий (пакет каже gps_fix=False)
    # АБО коли взагалі немає пакетів довше TIMEOUT_THRESHOLD (повна тиша) ---
    for v_id, state in vehicles.items():
        if "lat" not in state:
            continue

        total_silence = current_time - state.get("last_packet_time", 0) > TIMEOUT_THRESHOLD
        gnss_jammed = not state.get("gps_fix_live", True)

        if gnss_jammed or total_silence:
            if state.get("reb_interp") is None:
                state["reb_interp"] = start_reb_prediction(state, current_time)

            model_speed_now = sample_interp_speed(state["reb_interp"], current_time)
            last_actual_speed = state["speed_buffer"][-1] if state["speed_buffer"] else ru.BASE_SPEED_KMH

            # Той самий фізичний крок (обмежений темп розгону/гальмування),
            # що й у клієнта — щоб прогноз змінювався так само плавно, як
            # реальне авто, а не стрибав до цілі за один тік.
            predicted_speed = max(15.0, ru.step_speed_toward(last_actual_speed, model_speed_now))

            dt = current_time - state.get("last_update_time", current_time)
            dt = max(0.0, min(dt, 5.0))  # захист від аномально великого dt (пауза дебагера тощо)
            speed_mps = predicted_speed * (1000 / 3600)

            # Просуваємо СИРИЙ dead reckoning від його ж попереднього кроку, а
            # не від state["lat"]/["lon"] — ті вже могли бути притягнуті
            # Wi-Fi-корекцією нижче. Якщо брати за старт відкориговану
            # позицію, corrected_lat/lon кожного тіку знову тягне ЦЮ Ж точку
            # назад до того самого якоря — точка збігається в нерухому
            # точку рівноваги за кілька тіків і "зависає" на весь час РЕБ,
            # а по відновленню GNSS реальна позиція (яка постійно рухалась)
            # підмінює її ривком.
            prev_dr_lat, prev_dr_lon = state["dr_lat"], state["dr_lon"]
            dr_lat, dr_lon, dr_idx, reached_end = ru.advance_along_route(
                route_coords, state["dr_idx"], state["dr_lat"], state["dr_lon"], speed_mps * dt
            )
            state["dr_lat"], state["dr_lon"], state["dr_idx"] = dr_lat, dr_lon, dr_idx
            dr_step_m = ru.calculate_distance(prev_dr_lat, prev_dr_lon, dr_lat, dr_lon)

            # Wi-Fi-корекція: якщо GNSS глушиться, але дані ще долітають
            # (найчастіший реальний випадок), у нас є свіжий скан з ІСТИННОЇ
            # позиції борта — навіть коли повна тиша (total_silence) і скан
            # застарілий, спробувати збіг все одно варто, просто шанс нижчий.
            match = fp_store.match(state.get("last_wifi_scan") or {})
            if match is not None:
                w = wifi_fp.correction_weight(match["confidence"])
                corrected_lat = (1 - w) * dr_lat + w * match["lat"]
                corrected_lon = (1 - w) * dr_lon + w * match["lon"]
                state["wifi_status"] = "matched" if w > 0 else "searching"
                state["wifi_confidence"] = match["confidence"]
            else:
                corrected_lat, corrected_lon = dr_lat, dr_lon
                state["wifi_status"] = "searching"
                state["wifi_confidence"] = 0.0

            # Forward-базова лінія: показ ЗАВЖДИ просувається вздовж маршруту
            # на dr_step_m (та сама фізика, що й dr) — гарантовано вперед,
            # ніколи не назад, ніколи не заморожено.
            fwd_lat, fwd_lon, _, _ = ru.advance_along_route(
                route_coords, state["route_idx"], state["lat"], state["lon"], dr_step_m
            )

            # Wi-Fi-корекцію застосовуємо, лише якщо вона робить показ
            # БЛИЖЧИМ до dr, ніж форвардна базова лінія — інакше цей тік
            # просто ігноруємо (лишаємо forward-крок як є). Навіть коли
            # корекція приймається, стрибок до неї обмежений MAX_WIFI_SNAP_M.
            dist_corrected_to_dr = ru.calculate_distance(corrected_lat, corrected_lon, dr_lat, dr_lon)
            dist_fwd_to_dr = ru.calculate_distance(fwd_lat, fwd_lon, dr_lat, dr_lon)

            if dist_corrected_to_dr < dist_fwd_to_dr:
                snap_step_m = ru.calculate_distance(fwd_lat, fwd_lon, corrected_lat, corrected_lon)
                ratio = min(1.0, MAX_WIFI_SNAP_M / snap_step_m) if snap_step_m > 1e-9 else 1.0
                display_lat = fwd_lat + (corrected_lat - fwd_lat) * ratio
                display_lon = fwd_lon + (corrected_lon - fwd_lon) * ratio
            else:
                display_lat, display_lon = fwd_lat, fwd_lon

            display_idx = ru.nearest_route_index(route_coords, display_lat, display_lon, search_from=state["route_idx"])
            display_idx = max(display_idx, state["route_idx"])  # запобіжник: ніколи не відкочуємось назад

            if display_idx != state["route_idx"]:
                state["heading"] = ru.heading_for_route_idx(route_coords, display_idx)

            state.update({
                "lat": display_lat,
                "lon": display_lon,
                "route_idx": display_idx,
                "last_update_time": current_time,
                "is_predicted": True,
            })

            state["speed_buffer"].append(predicted_speed)
            if len(state["speed_buffer"]) > 5:
                state["speed_buffer"].pop(0)

            tag = "🏁 КІНЕЦЬ МАРШРУТУ" if reached_end else "🔴 DEAD RECKONING"
            print(f"[{tag}] Авто {v_id} | Швидкість: {predicted_speed:.1f} км/год | idx={display_idx} | "
                  f"Wi-Fi: {state['wifi_status']} ({state['wifi_confidence']*100:.0f}%)")

    save_state_atomic(vehicles)

    elapsed = time.time() - loop_start
    time.sleep(max(0.0, TICK_SEC - elapsed))
