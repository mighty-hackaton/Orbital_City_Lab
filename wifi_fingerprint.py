"""Wi-Fi fingerprint layer — periodic correction for GPS/AI dead reckoning.

Ідея: поки є GPS, транспорт бачить не тільки координату, а й "краєвид" з
Wi-Fi точок доступу навколо (MAC + RSSI). Ми записуємо ці "відбитки"
(fingerprints) прив'язаними до місця на маршруті. Коли GPS зникає (РЕБ),
Wi-Fi-радіо телефону/бортового модему ПРОДОВЖУЄ бачити ті самі точки
доступу — глушник GPS/GSM не глушить 2.4/5 ГГц Wi-Fi. Порівнюючи поточний
скан із базою фінгерпринтів, можна зрозуміти "я вже проїжджав/проїжджала
це місце раніше" і підтягнути прогноз AI+геометрії ближче до правди,
замість того щоб некероване накопичення похибки dead reckoning росло
необмежено.

Для хакатон-демо реального Wi-Fi сканування (яке вимагає нативного
доступу до Wi-Fi-радіо — недоступно з браузера/Streamlit) НЕМАЄ.
Натомість build_ap_field()/scan_at() симулюють правдоподібне поле точок
доступу вздовж маршруту (детерміноване з координат — client і server
завжди згенерують ІДЕНТИЧНЕ поле, як і route_info.json синхронізує
маршрут). Усе, що йде ПІСЛЯ скану (FingerprintStore.match, корекція
позиції) — це реальний, продакшн-придатний алгоритм: та сама логіка
відпрацює однаково, якщо на вхід замість scan_at() подати справжній
Wi-Fi скан з телефону водія.
"""
import hashlib
import math
import random
import threading

from route_utils import calculate_distance

# ==========================================================================
# СИНТЕТИЧНЕ ПОЛЕ ТОЧОК ДОСТУПУ (заміна реального Wi-Fi сканування для демо)
# ==========================================================================
AP_CLUSTER_SPACING_M = 90.0       # середня відстань між кластерами точок доступу
APS_PER_CLUSTER = (3, 6)          # скільки AP у кожному кластері (під'їзди, кафе, офіси)
AP_JITTER_DEG = 0.00035           # розкид точок всередині кластера (~30-40м)

AP_TX_RSSI_AT_1M = -40.0          # калібрувальна константа моделі затухання, дБм
PATH_LOSS_EXPONENT = 2.8          # міська забудова/"вуличний каньйон" (> 2.0 вільного простору)
SHADOW_FADING_STD_DB = 4.0        # лог-нормальний шум (стіни, дерева, інші авто)
AP_MAX_RANGE_M = 130.0            # за межами цієї відстані AP вже не видно
RSSI_FLOOR_DBM = -95.0            # нижче цього рівня скан не вважається AP "видимою"

# ==========================================================================
# FINGERPRINT DB + MATCHING
# ==========================================================================
MIN_COMMON_APS = 3                 # менше спільних AP — збіг ненадійний, ігноруємо
TOPK_MATCHES = 5                   # скільки найближчих фінгерпринтів усереднюємо
RECORD_EVERY_M = 25.0              # як часто (по дистанції) записувати новий фінгерпринт
MATCH_CONFIDENCE_FLOOR = 0.15      # нижче цього confidence — корекцію не застосовуємо
MAX_CORRECTION_WEIGHT = 0.6        # навіть при ідеальному збігу не "стрибаємо" повністю на нього


def _seeded_rng(*parts):
    """RNG, детермінований координатами — щоб клієнт і сервер отримали
    ІДЕНТИЧНЕ поле точок доступу, не обмінюючись жодним пакетом (той самий
    трюк, що й з кешованим route_info.json для дорожньої полілінії)."""
    h = hashlib.sha256("|".join(f"{p:.6f}" if isinstance(p, float) else str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def build_ap_field(route_coords):
    """Розкидає синтетичні точки доступу кластерами вздовж маршруту.

    Повертає список {"mac", "ssid", "lat", "lon"}. Викликати ОДИН РАЗ на
    процес (клієнт і сервер/додаток) — результат детермінований відносно
    route_coords, тож кешувати можна як завгодно довго.
    """
    ssid_pool = ["Kyivstar_Home", "Lifecell-WiFi", "TP-LINK_2.4G", "FreeWifi_Kyiv",
                 "KV-", "ASUS_5G", "", "", "MTS-Router"]  # "" = прихований SSID, це нормально

    aps = []
    cum_dist = 0.0
    next_cluster_at = 0.0
    for i in range(len(route_coords) - 1):
        lat1, lon1 = route_coords[i]
        lat2, lon2 = route_coords[i + 1]
        seg_len = calculate_distance(lat1, lon1, lat2, lon2)

        while next_cluster_at <= cum_dist + seg_len and next_cluster_at < cum_dist + seg_len + 1e-6:
            ratio = 0.0 if seg_len < 1e-9 else (next_cluster_at - cum_dist) / seg_len
            ratio = max(0.0, min(1.0, ratio))
            clat = lat1 + (lat2 - lat1) * ratio
            clon = lon1 + (lon2 - lon1) * ratio

            rng = _seeded_rng(round(clat, 6), round(clon, 6))
            n_aps = rng.randint(*APS_PER_CLUSTER)
            for _ in range(n_aps):
                mac = ":".join(f"{rng.randint(0, 255):02X}" for _ in range(6))
                ssid = rng.choice(ssid_pool)
                aps.append({
                    "mac": mac,
                    "ssid": ssid,
                    "lat": clat + rng.uniform(-AP_JITTER_DEG, AP_JITTER_DEG),
                    "lon": clon + rng.uniform(-AP_JITTER_DEG, AP_JITTER_DEG),
                })
            next_cluster_at += rng.uniform(AP_CLUSTER_SPACING_M * 0.6, AP_CLUSTER_SPACING_M * 1.4)

        cum_dist += seg_len

    return aps


def scan_at(lat, lon, ap_field, rng=None):
    """Симулює те, що показав би Wi-Fi-скан телефону/бортового модему в цій
    точці: {mac: rssi_dBm} для всіх AP у межах досяжності, з реалістичним
    затуханням сигналу за відстанню (log-distance path loss) + шумом
    (shadow fading). Реальний клієнт замінить цю функцію на фактичний
    виклик Wi-Fi API — решта пайплайну не зміниться ні на рядок."""
    rng = rng or random
    scan = {}
    for ap in ap_field:
        d = calculate_distance(lat, lon, ap["lat"], ap["lon"])
        if d > AP_MAX_RANGE_M:
            continue
        d = max(1.0, d)
        path_loss = 10 * PATH_LOSS_EXPONENT * math.log10(d)
        shadow = rng.gauss(0.0, SHADOW_FADING_STD_DB) if hasattr(rng, "gauss") else random.gauss(0.0, SHADOW_FADING_STD_DB)
        rssi = AP_TX_RSSI_AT_1M - path_loss + shadow
        if rssi < RSSI_FLOOR_DBM:
            continue
        scan[ap["mac"]] = round(rssi, 1)
    return scan


class FingerprintStore:
    """База Wi-Fi фінгерпринтів, прив'язаних до позиції на маршруті.

    В реальній системі це — таблиця на сервері, що накопичується від
    УСЬОГО флоту (не одного авто): якщо маршрут 100 разів на день проїжджає
    5 різних автобусів, кожен з них поповнює ту саму базу. Тому нова
    машина без власної історії вже отримує користь від фінгерпринтів,
    записаних іншими бортами раніше. У цій демці екземпляр
    FingerprintStore навмисно спільний (кешується як ресурс процесу), а не
    створюється по одному на машину.
    """

    def __init__(self):
        self.records = []          # [{route_idx, lat, lon, scan: {mac: rssi}}]
        self._last_recorded_dist = {}   # vehicle_id -> остання cum_dist запису
        # Streamlit Community Cloud кешує цей об'єкт як ресурс ПРОЦЕСУ, тож
        # кілька відвідувачів (кожен зі своєю симуляцією флоту) пишуть у ту
        # саму базу одночасно з різних сесій — без лока append() з двох
        # потоків одночасно міг би пошкодити список.
        self._lock = threading.Lock()

    def clear(self):
        with self._lock:
            self.records.clear()
            self._last_recorded_dist.clear()

    def maybe_record(self, vehicle_id, lat, lon, route_idx, scan, cum_dist):
        if not scan:
            return
        last = self._last_recorded_dist.get(vehicle_id, -1e9)
        if cum_dist - last < RECORD_EVERY_M:
            return
        with self._lock:
            self._last_recorded_dist[vehicle_id] = cum_dist
            self.records.append({"route_idx": route_idx, "lat": lat, "lon": lon, "scan": dict(scan)})

    def match(self, live_scan):
        """Порівнює live_scan із базою через сигнальну відстань (стандартний
        підхід Wi-Fi indoor positioning: евклідова відстань RSSI по
        спільних MAC-адресах). Повертає найкращу оцінку позиції або None,
        якщо збігу з достатньою довірою немає."""
        if not live_scan or not self.records:
            return None

        with self._lock:
            records_snapshot = list(self.records)

        scored = []
        for rec in records_snapshot:
            common = set(rec["scan"]) & set(live_scan)
            if len(common) < MIN_COMMON_APS:
                continue
            sq_sum = sum((rec["scan"][m] - live_scan[m]) ** 2 for m in common)
            signal_dist = math.sqrt(sq_sum / len(common))
            weight = len(common) / (signal_dist + 1.0)
            scored.append((weight, rec, len(common)))

        if not scored:
            return None

        scored.sort(key=lambda x: -x[0])
        top = scored[:TOPK_MATCHES]
        total_w = sum(w for w, _, _ in top)

        idx_est = sum(w * r["route_idx"] for w, r, _ in top) / total_w
        lat_est = sum(w * r["lat"] for w, r, _ in top) / total_w
        lon_est = sum(w * r["lon"] for w, r, _ in top) / total_w
        best_common = max(c for _, _, c in top)

        # Евристика довіри: більше збігів + більше спільних AP на найкращому
        # записі => вища довіра. Обидва компоненти обмежені в [0, 1].
        confidence = min(1.0, (len(top) / TOPK_MATCHES) * 0.5 + min(1.0, best_common / 8) * 0.5)

        return {
            "route_idx": idx_est,
            "lat": lat_est,
            "lon": lon_est,
            "confidence": confidence,
            "n_matches": len(top),
            "n_common_best": best_common,
        }


def correction_weight(confidence):
    """Наскільки сильно тягнути прогноз AI+геометрії до Wi-Fi-оцінки
    позиції. Нижче MATCH_CONFIDENCE_FLOOR — не довіряємо збігу взагалі
    (могло бути 3 випадкові AP, що збіглись за рівнем сигналу). Вище —
    лінійно зростає, але ніколи не досягає 1.0: Wi-Fi лишається КОРЕКЦІЄЮ
    прогнозу, а не заміною його."""
    if confidence < MATCH_CONFIDENCE_FLOOR:
        return 0.0
    span = 1.0 - MATCH_CONFIDENCE_FLOOR
    return MAX_CORRECTION_WEIGHT * (confidence - MATCH_CONFIDENCE_FLOOR) / span
