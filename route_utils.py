"""Спільна логіка маршруту та геометрії для клієнта й сервера.

Раніше клієнт і сервер рахували маршрут (і навіть відстань між точками)
кожен по-своєму — це і було джерелом розсинхрону. Тут все в одному місці:
- один точний спосіб рахувати відстань і bearing;
- один спосіб отримати route_coords (з кешу route_info.json, якщо він вже є);
- рух точки вздовж ВІДОМОЇ полілінії маршруту (а не вгадування вузла графа) —
  саме це прибирає "зліт з траси" на поворотах під час dead reckoning.
"""
import json
import math
import os

ROUTE_FILE = "route_info.json"

# Єдине джерело правди для швидкісних констант. Раніше BASE_SPEED_KMH жив
# окремо і в клієнті, і (неявно) очікувався сервером — рано чи пізно вони б
# розійшлися, як уже сталось з heading. Тепер обидва читають звідси.
BASE_SPEED_KMH = 35.0
TURN_SLOW_SPEED_KMH = 20.0
TURN_ANGLE_THRESHOLD_DEG = 28.0


def calculate_distance(lat1, lon1, lat2, lon2):
    """Haversine, метри. На відміну від формули з acos() не падає з
    math domain error на малих відстанях через похибку округлення."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def angle_diff_deg(a, b):
    """Різниця кутів у градусах, коректно згорнута в [0, 180].
    Стара версія (abs(atan2()-atan2())) без згортання давала ~340° замість
    ~20° при переході через ±180° — звідси хибні спрацювання детектора повороту."""
    d = abs(a - b) % 360
    return min(d, 360 - d)


def build_route(place="Podilskyi District, Kyiv, Ukraine",
                 start=(50.4646, 30.5165), end=(50.4735, 30.4912)):
    """Рахує маршрут по графу доріг OSM. Важка операція (Overpass + граф) —
    викликати один раз, результат кешується через save_route()."""
    import osmnx as ox
    import networkx as nx

    G = ox.graph_from_place(place, network_type="drive")
    origin = ox.distance.nearest_nodes(G, X=start[1], Y=start[0])
    destination = ox.distance.nearest_nodes(G, X=end[1], Y=end[0])
    route = nx.shortest_path(G, origin, destination, weight="length")
    return [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]


def save_route(route_coords):
    data = {
        "path": [{"lat": lat, "lon": lon} for lat, lon in route_coords],
        "start": {"lat": route_coords[0][0], "lon": route_coords[0][1]},
        "end": {"lat": route_coords[-1][0], "lon": route_coords[-1][1]},
    }
    tmp = ROUTE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, ROUTE_FILE)


def load_or_build_route(place="Podilskyi District, Kyiv, Ukraine",
                         start=(50.4646, 30.5165), end=(50.4735, 30.4912)):
    """Бере маршрут з route_info.json, якщо його вже порахувала інша сторона
    (клієнт чи сервер, хто запустився першим) — інакше рахує сам і зберігає.
    Це гарантує, що клієнт і сервер завжди рухаються по ІДЕНТИЧНІЙ полілінії."""
    if os.path.exists(ROUTE_FILE):
        try:
            with open(ROUTE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            coords = [(p["lat"], p["lon"]) for p in data["path"]]
            if len(coords) >= 2:
                return coords
        except Exception:
            pass  # пошкоджений кеш — перерахуємо

    coords = build_route(place, start, end)
    save_route(coords)
    return coords


def nearest_route_index(route_coords, lat, lon, search_from=0, search_window=30):
    """Індекс точки маршруту, найближчої до (lat, lon). Пошук обмежений
    околом попереднього індексу — щоб машину не 'перекинуло' на схожу,
    але далеку ділянку дороги (буває на петлястих маршрутах)."""
    lo = max(0, search_from - 2)
    hi = min(len(route_coords), search_from + search_window)
    best_idx, best_dist = lo, float("inf")
    for i in range(lo, hi):
        d = calculate_distance(lat, lon, route_coords[i][0], route_coords[i][1])
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def advance_along_route(route_coords, idx, lat, lon, dist_to_move):
    """Рухає (lat, lon) вздовж ВІДОМОЇ полілінії маршруту на dist_to_move метрів,
    починаючи із сегмента idx. Координати завжди лежать точно на маршруті —
    це замінює графове 'вгадування наступного вузла за кутом', яке й спричиняло
    зліт машини з траси на поворотах під час втрати зв'язку.
    Повертає (нова lat, нова lon, новий idx, чи доїхали до кінця маршруту)."""
    while dist_to_move > 0.1 and idx < len(route_coords) - 1:
        t_lat, t_lon = route_coords[idx + 1]
        d = calculate_distance(lat, lon, t_lat, t_lon)
        if d > dist_to_move:
            ratio = dist_to_move / d
            lat += (t_lat - lat) * ratio
            lon += (t_lon - lon) * ratio
            dist_to_move = 0
        else:
            lat, lon = t_lat, t_lon
            dist_to_move -= d
            idx += 1
    reached_end = idx >= len(route_coords) - 1
    return lat, lon, idx, reached_end


def build_cumulative_distances(route_coords):
    """Кумулятивна відстань (у метрах) від початку маршруту до кожної
    точки. Використовується Wi-Fi fingerprint шаром (wifi_fingerprint.py),
    щоб вирішувати "чи проїхали ми вже достатньо, щоб записати новий
    фінгерпринт" — рахувати це наново на кожному тіку було б O(n) щоразу,
    а маршрут не змінюється, тож рахуємо один раз і кешуємо."""
    cum = [0.0]
    for i in range(len(route_coords) - 1):
        lat1, lon1 = route_coords[i]
        lat2, lon2 = route_coords[i + 1]
        cum.append(cum[-1] + calculate_distance(lat1, lon1, lat2, lon2))
    return cum


def cumulative_distance_at(route_coords, cum_dist_arr, idx, lat, lon):
    """Неперервна відстань уздовж маршруту до поточної (lat, lon), а не
    лише до найближчого кешованого вузла route_coords[idx]. route_coords
    з OSM буває дуже розрідженим (вузли графа, а не точки кожні N метрів),
    тож cum_dist_arr[idx] сам по собі змінюється рідко — Wi-Fi fingerprint
    шар (wifi_fingerprint.py) записував би фінгерпринти набагато рідше, ніж
    RECORD_EVERY_M, якби рахував відстань лише по індексу вузла."""
    idx = max(0, min(idx, len(route_coords) - 1))
    return cum_dist_arr[idx] + calculate_distance(*route_coords[idx], lat, lon)


def heading_for_route_idx(route_coords, idx):
    """Напрямок наступного сегмента маршруту за індексом — використовується
    сервером одразу після переходу на новий сегмент під час dead reckoning."""
    idx = max(0, min(idx, len(route_coords) - 2))
    (lat1, lon1), (lat2, lon2) = route_coords[idx], route_coords[idx + 1]
    return calculate_bearing(lat1, lon1, lat2, lon2)


# Наскільки заздалегідь (у метрах) починаємо гальмувати перед різким поворотом.
BRAKE_LOOKAHEAD_M = 60.0
# Реальне авто гальмує різкіше, ніж розганяється — тому пороги різні.
MAX_ACCEL_KMH_PER_SEC = 2.5
MAX_BRAKE_KMH_PER_SEC = 4.0


def find_next_turn_distance(route_coords, idx, angle_threshold_deg=TURN_ANGLE_THRESHOLD_DEG,
                             search_ahead_m=BRAKE_LOOKAHEAD_M):
    """Йде вперед по маршруту від idx і шукає найближчий різкий поворот
    (кут між сусідніми сегментами > angle_threshold_deg).
    Повертає (відстань_до_повороту_м, чи_знайдено). Якщо в межах
    search_ahead_m повороту немає — повертає (search_ahead_m, False)."""
    cum_dist = 0.0
    i = idx
    while i < len(route_coords) - 2 and cum_dist < search_ahead_m:
        p1, p2, p3 = route_coords[i], route_coords[i + 1], route_coords[i + 2]
        b1 = calculate_bearing(p1[0], p1[1], p2[0], p2[1])
        b2 = calculate_bearing(p2[0], p2[1], p3[0], p3[1])
        if angle_diff_deg(b1, b2) > angle_threshold_deg:
            return cum_dist, True
        cum_dist += calculate_distance(p1[0], p1[1], p2[0], p2[1])
        i += 1
    return search_ahead_m, False


def target_speed_for_position(route_coords, idx, base_speed_kmh=BASE_SPEED_KMH,
                               slow_speed_kmh=TURN_SLOW_SPEED_KMH,
                               angle_threshold_deg=TURN_ANGLE_THRESHOLD_DEG,
                               brake_distance_m=BRAKE_LOOKAHEAD_M):
    """Бажана швидкість у цій точці маршруту з ПОСТУПОВИМ гальмуванням:
    чим ближче попереду різкий поворот, тим нижча ціль — лінійна інтерполяція
    між base_speed_kmh і slow_speed_kmh в межах brake_distance_m. Замінює
    стару бінарну логіку 'зараз повертаємо чи ні', яка давала стрибок 35→20
    рівно в момент повороту замість плавного гальмування перед ним."""
    dist, found = find_next_turn_distance(route_coords, idx, angle_threshold_deg, brake_distance_m)
    if not found:
        return base_speed_kmh
    ratio = max(0.0, min(1.0, dist / brake_distance_m))
    return slow_speed_kmh + ratio * (base_speed_kmh - slow_speed_kmh)


def step_speed_toward(current_speed_kmh, target_speed_kmh, dt_sec=1.0,
                       max_accel_kmh_per_sec=MAX_ACCEL_KMH_PER_SEC,
                       max_brake_kmh_per_sec=MAX_BRAKE_KMH_PER_SEC):
    """Один крок фізично правдоподібного наближення до цільової швидкості:
    розгін і гальмування обмежені різними темпами. Використовується і
    клієнтом (реальний рух), і сервером (прогноз під час втрати зв'язку) —
    щоб обидві сторони змінювали швидкість однаково плавно, а не стрибками."""
    diff = target_speed_kmh - current_speed_kmh
    if diff >= 0:
        step = min(diff, max_accel_kmh_per_sec * dt_sec)
    else:
        step = max(diff, -max_brake_kmh_per_sec * dt_sec)
    return current_speed_kmh + step