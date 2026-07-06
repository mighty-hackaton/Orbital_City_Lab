import streamlit as st
import pydeck as pdk
import json
import time
import os

# ==========================================
# НАЛАШТУВАННЯ СТОРІНКИ
# ==========================================
st.set_page_config(page_title="Dead Reckoning Tracker", page_icon="🛰️", layout="wide")
st.title("🛰️ Зв'язок-Стійкий Навігатор | Orbital City Lab")

STATE_FILE = "dashboard_state.json"
ROUTE_FILE = "route_info.json"


def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Файл міг бути захоплений сервером у процесі запису —
        # просто пропускаємо цей кадр, а не падаємо (сервер тепер пише
        # атомарно через os.replace, тому це малоймовірно, але про всяк випадок).
        return None


def last_speed(state):
    buf = state.get("speed_buffer") or []
    return buf[-1] if buf else 0.0


# ==========================================
# ПОБУДОВА ІНТЕРФЕЙСУ
# ==========================================
col_stats, col_map = st.columns([1, 3])

vehicles_data = load_json(STATE_FILE)
route_data = load_json(ROUTE_FILE)

with col_stats:
    st.subheader("Статус автопарку")
    if not vehicles_data:
        st.info("Очікування даних від UDP-сервера...")
    else:
        for v_id, state in vehicles_data.items():
            is_predicted = state.get("is_predicted", False)
            status_text = "🔴 ВТРАТА ЗВ'ЯЗКУ (ШІ)" if is_predicted else "🟢 ЗВ'ЯЗОК СТАБІЛЬНИЙ"
            bg_color = "#ffcccc" if is_predicted else "#ccffcc"

            st.markdown(
                f"""
                <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px; color: black; margin-bottom: 10px;">
                    <b>ID: {v_id}</b><br>
                    Статус: {status_text}<br>
                    Швидкість: {last_speed(state):.1f} км/год
                </div>
                """,
                unsafe_allow_html=True,
            )

with col_map:
    layers = []

    # 1. ШАР МАРШРУТУ (Лінія + Старт/Фініш)
    if route_data:
        path_layer = pdk.Layer(
            "PathLayer",
            data=[{"path": [[p["lon"], p["lat"]] for p in route_data["path"]]}],
            get_path="path",
            get_color=[150, 150, 150, 150],
            width_scale=20,
            width_min_pixels=3,
        )
        layers.append(path_layer)

        endpoints = [
            {"coord": [route_data["start"]["lon"], route_data["start"]["lat"]], "color": [0, 200, 0, 200]},
            {"coord": [route_data["end"]["lon"], route_data["end"]["lat"]], "color": [200, 0, 0, 200]},
        ]
        endpoint_layer = pdk.Layer(
            "ScatterplotLayer",
            data=endpoints,
            get_position="coord",
            get_fill_color="color",
            get_radius=50,
            radius_min_pixels=6,
        )
        layers.append(endpoint_layer)

    # 2. ШАР ТРАНСПОРТУ (Динамічний маркер авто)
    if vehicles_data:
        v_list = []
        for v_id, state in vehicles_data.items():
            if "lat" not in state or "lon" not in state:
                continue
            is_pred = state.get("is_predicted", False)
            color = [255, 50, 50, 255] if is_pred else [50, 255, 50, 255]
            v_list.append({
                "coord": [state["lon"], state["lat"]],
                "color": color,
                "tooltip": f"Авто: {v_id}",
            })

        if v_list:
            vehicle_layer = pdk.Layer(
                "ScatterplotLayer",
                data=v_list,
                get_position="coord",
                get_fill_color="color",
                get_line_color=[0, 0, 0, 255],
                stroked=True,
                line_width_min_pixels=2,
                get_radius=60,
                radius_min_pixels=10,
            )
            layers.append(vehicle_layer)

    # 3. НАЛАШТУВАННЯ КАМЕРИ
    start_lat, start_lon = 50.462, 30.515
    if route_data:
        start_lat, start_lon = route_data["start"]["lat"], route_data["start"]["lon"]
    if vehicles_data:
        first_vehicle = next(iter(vehicles_data.values()))
        if "lat" in first_vehicle:
            start_lat, start_lon = first_vehicle["lat"], first_vehicle["lon"]

    view_state = pdk.ViewState(latitude=start_lat, longitude=start_lon, zoom=14, pitch=30)

    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, tooltip={"text": "{tooltip}"}))

# Синхронізовано з TICK_SEC на сервері — частіший рефреш дає плавнішу
# анімацію руху авто на карті замість "смиканого" оновлення раз на секунду.
time.sleep(0.5)
st.rerun()