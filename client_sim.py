import socket
import json
import time
import random

import route_utils as ru
import wifi_fingerprint as wifi_fp

SERVER_IP = "127.0.0.1"
SERVER_PORT = 5005
VEHICLE_ID = "Борт-101"

# Було 1.0 (1 апдейт/сек) — звідси й "смикана" анімація на дашборді.
# 0.5 = 2 апдейти/сек, помітно плавніше при тій самій логіці руху.
SEND_PERIOD_SEC = 0.5

# --- Природний шум швидкості ---
# Реальне авто не тримає ідеально рівні 35.000 км/год навіть на прямій —
# трохи "гуляє" через ногу на педалі газу, нерівності, вітер тощо.
# Робимо це AR(1)-процесом (плавне блукання), а не незалежним шумом на
# кожному тіку — інакше це виглядало б як тремтіння/дрож, а не природний рух.
SPEED_NOISE_STD_KMH = 0.9    # амплітуда випадкового поштовху щотіку
SPEED_NOISE_DECAY = 0.85     # 0..1: більше = повільніша, "плавніша" зміна шуму
SPEED_NOISE_CLAMP_KMH = 3.0  # захист від рідкісного занадто великого викиду

# --- Цикл втрати GNSS-сигналу (РЕБ) ---
# ВАЖЛИВО: РЕБ тут = глушіння/спуфінг GNSS (GPS), а НЕ повне вимкнення
# зв'язку. Це відповідає реальним випадкам — глушник забиває вузьку смугу
# частот GPS (L1 ~1575 МГц), а GSM/LTE-модуль (інша смуга) і Wi-Fi-радіо
# (2.4/5 ГГц) продовжують працювати. Тому клієнт і під час "втрати сигналу"
# ПРОДОВЖУЄ надсилати пакети — просто без валідних lat/lon, зате разом зі
# свіжим Wi-Fi-сканом. Це і є технічна причина, чому Wi-Fi fingerprint
# взагалі можливий: не треба чекати, доки GPS "повернеться", сервер отримує
# альтернативний сигнал для корекції в реальному часі.
REB_CYCLE_SEC = 20        # повний цикл: онлайн+офлайн разом, секунд
REB_OFFLINE_START = 6     # з якої секунди циклу зникає GNSS-фікс
REB_OFFLINE_END = 14      # по яку секунду циклу GNSS відсутній


def is_signal_lost(seconds_elapsed):
    phase = seconds_elapsed % REB_CYCLE_SEC
    return REB_OFFLINE_START <= phase <= REB_OFFLINE_END


print("⏳ Симулятор: отримую маршрут (з кешу route_info.json або рахую сам)...")
route_coords = ru.load_or_build_route()
print(f"✅ Маршрут готовий: {len(route_coords)} точок.")

print("⏳ Генерую поле Wi-Fi точок доступу вздовж маршруту (детерміновано з координат)...")
ap_field = wifi_fp.build_ap_field(route_coords)
print(f"✅ Wi-Fi поле готове: {len(ap_field)} точок доступу.")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

current_idx = 0
current_lat, current_lon = route_coords[0]
current_speed_kmh = ru.BASE_SPEED_KMH  # стартуємо на базовій швидкості
speed_noise_kmh = 0.0
seconds_elapsed = 0.0

print("🚗 Рухаємось...")
while current_idx < len(route_coords) - 1:
    # "Чиста" цільова швидкість — з ПОСТУПОВИМ гальмуванням перед поворотом
    # (лінійна інтерполяція за відстанню до найближчого різкого повороту).
    clean_target_kmh = ru.target_speed_for_position(route_coords, current_idx)

    # Накладаємо природний шум: плавне випадкове блукання навколо цілі,
    # обмежене зверху/знизу, щоб не з'їхати в абсурдні значення на повороті.
    speed_noise_kmh = speed_noise_kmh * SPEED_NOISE_DECAY + random.gauss(0.0, SPEED_NOISE_STD_KMH)
    speed_noise_kmh = max(-SPEED_NOISE_CLAMP_KMH, min(SPEED_NOISE_CLAMP_KMH, speed_noise_kmh))
    noisy_target_kmh = max(0.0, clean_target_kmh + speed_noise_kmh)

    # Реальна швидкість все одно не стрибає до цілі миттєво, а рухається до
    # неї з обмеженим темпом розгону/гальмування — фізично правдоподібно,
    # і заразом згладжує сам шум (не дає йому виглядати як тремтіння).
    current_speed_kmh = ru.step_speed_toward(current_speed_kmh, noisy_target_kmh, dt_sec=SEND_PERIOD_SEC)
    speed_mps = current_speed_kmh * (1000 / 3600)

    heading = ru.heading_for_route_idx(route_coords, current_idx)

    current_lat, current_lon, current_idx, reached_end = ru.advance_along_route(
        route_coords, current_idx, current_lat, current_lon, speed_mps * SEND_PERIOD_SEC
    )

    # Wi-Fi-радіо сканує НЕЗАЛЕЖНО від того, чи є зараз GNSS-фікс — глушник
    # б'є по частоті GPS, а не по 2.4/5 ГГц. Тому скан завжди береться з
    # ІСТИННОЇ поточної позиції, навіть коли пакет нижче не понесе lat/lon.
    wifi_scan = wifi_fp.scan_at(current_lat, current_lon, ap_field)

    gps_fix = not is_signal_lost(seconds_elapsed)
    packet = {"id": VEHICLE_ID, "gps_fix": gps_fix, "wifi": wifi_scan}
    if gps_fix:
        packet.update({"lat": current_lat, "lon": current_lon, "speed": current_speed_kmh, "heading": heading})

    # Пакет летить ЗАВЖДИ (і з GNSS-фіксом, і без) — тільки повне глушіння
    # GSM/LTE (а не тільки GPS) означало б, що дані взагалі не долітають;
    # сервер окремо ловить і цей гірший випадок через TIMEOUT_THRESHOLD.
    sock.sendto(json.dumps(packet).encode("utf-8"), (SERVER_IP, SERVER_PORT))

    if gps_fix:
        print(f"🟢 Стан: {current_speed_kmh:.1f} км/год | idx={current_idx} | t={seconds_elapsed:.1f} | Wi-Fi AP: {len(wifi_scan)}")
    else:
        print(f"🔴 РЕБ (GNSS глушиться): позиція невідома | Wi-Fi AP видно: {len(wifi_scan)} | t={seconds_elapsed:.1f}")

    seconds_elapsed += SEND_PERIOD_SEC
    time.sleep(SEND_PERIOD_SEC)

    if reached_end:
        break

print("🏁 Маршрут завершено.")
