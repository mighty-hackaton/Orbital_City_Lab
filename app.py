"""
Orbital City Lab — Signal-Resilient Transit Tracker
====================================================

Public transit vehicles keep sending GPS over the mobile network. During
electronic warfare (REB) activity that network can drop out for seconds at
a time. This app simulates a small bus/tram fleet on a real Kyiv route and
shows what riders would see on a live map:

  * green solid GPS fix — position comes straight from the "vehicle"
  * red dead reckoning — GPS is lost, so position is estimated from a
    trained speed model + route geometry until the signal comes back

Everything (fleet state, physics step, ML inference) runs inside this one
Streamlit session — no background sockets, no shared files on disk. That is
what makes it safe to deploy as a single public app on Streamlit Community
Cloud, where every visitor gets an independent in-memory simulation.
"""

import time

import joblib
import numpy as np
import pydeck as pdk
import streamlit as st

import feature_utils
import route_utils as ru

# ==========================================================================
# CONSTANTS
# ==========================================================================
TICK_SEC = 0.5                    # simulation step length (seconds of sim-time)
REFRESH_SEC = 0.5                 # how often the fragment redraws

REB_CYCLE_SEC = 20                # full online+offline cycle per vehicle
REB_OFFLINE_START = 6             # signal drops at this point in the cycle
REB_OFFLINE_END = 14              # signal returns at this point in the cycle

MODEL_SAMPLE_INTERVAL_SEC = 15.0  # matches training step of model.pkl
MODEL_HISTORY_LEN = 5

SPEED_NOISE_STD_KMH = 0.9
SPEED_NOISE_DECAY = 0.85
SPEED_NOISE_CLAMP_KMH = 3.0

VEHICLE_NAMES = ["Bus 101", "Bus 204", "Tram 7", "Trolley 12", "Bus 318"]

st.set_page_config(
    page_title="Orbital City Lab — Transit Tracker",
    page_icon="🛰️",
    layout="wide",
)


# ==========================================================================
# CACHED RESOURCES (loaded once per server process, shared read-only)
# ==========================================================================
@st.cache_resource(show_spinner="Loading speed-prediction model...")
def load_model():
    try:
        return joblib.load("model.pkl")
    except Exception as e:
        st.warning(f"Could not load model.pkl, falling back to route geometry only: {e}")
        return None


@st.cache_data(show_spinner="Loading route...")
def load_route():
    return ru.load_or_build_route()


ai_model = load_model()
route_coords = load_route()
ROUTE_LEN = len(route_coords)


# ==========================================================================
# SIMULATION — one simulated vehicle's worth of state + physics
# ==========================================================================
def new_vehicle(name, start_offset_idx, phase_offset_sec):
    idx = min(start_offset_idx, ROUTE_LEN - 2)
    lat, lon = route_coords[idx]
    return {
        "name": name,
        "idx": idx,
        "lat": lat,
        "lon": lon,
        "speed": ru.BASE_SPEED_KMH,
        "speed_noise": 0.0,
        "heading": ru.heading_for_route_idx(route_coords, idx),
        "speed_buffer": [],
        "model_speed_window": [],
        "model_heading_window": [],
        "next_model_sample_time": phase_offset_sec,
        "elapsed": phase_offset_sec,
        "phase_offset": phase_offset_sec,
        "reb_interp": None,
        "is_predicted": False,
        "reached_end": False,
    }


def is_signal_lost(elapsed_sec):
    phase = elapsed_sec % REB_CYCLE_SEC
    return REB_OFFLINE_START <= phase <= REB_OFFLINE_END


def start_reb_prediction(v, current_time):
    """Called once at the moment signal is lost, mirrors server_core.py logic."""
    last_speed = v["speed_buffer"][-1] if v["speed_buffer"] else ru.BASE_SPEED_KMH
    speed_window = v["model_speed_window"]
    heading_window = v["model_heading_window"]

    geometric_target = ru.target_speed_for_position(route_coords, v["idx"])

    if len(speed_window) >= MODEL_HISTORY_LEN and ai_model is not None:
        try:
            features = feature_utils.build_features(speed_window, heading_window)
            delta = float(ai_model.predict([features])[0])
            model_target = speed_window[-1] + delta
            target_speed = 0.5 * model_target + 0.5 * geometric_target
        except Exception:
            target_speed = geometric_target
    else:
        target_speed = geometric_target

    return {"start_time": current_time, "start_speed": last_speed, "target_speed": target_speed}


def sample_interp_speed(interp, current_time):
    elapsed = current_time - interp["start_time"]
    ratio = min(1.0, elapsed / MODEL_SAMPLE_INTERVAL_SEC)
    return interp["start_speed"] + ratio * (interp["target_speed"] - interp["start_speed"])


def step_vehicle(v, dt):
    if v["reached_end"]:
        return v

    v["elapsed"] += dt
    signal_lost = is_signal_lost(v["elapsed"])

    if not signal_lost:
        # --- normal GPS-driven movement (mirrors the old client_sim.py) ---
        clean_target = ru.target_speed_for_position(route_coords, v["idx"])
        v["speed_noise"] = v["speed_noise"] * SPEED_NOISE_DECAY + np.random.normal(0.0, SPEED_NOISE_STD_KMH)
        v["speed_noise"] = max(-SPEED_NOISE_CLAMP_KMH, min(SPEED_NOISE_CLAMP_KMH, v["speed_noise"]))
        noisy_target = max(0.0, clean_target + v["speed_noise"])
        v["speed"] = ru.step_speed_toward(v["speed"], noisy_target, dt_sec=dt)
        v["heading"] = ru.heading_for_route_idx(route_coords, v["idx"])

        speed_mps = v["speed"] * (1000 / 3600)
        v["lat"], v["lon"], v["idx"], reached = ru.advance_along_route(
            route_coords, v["idx"], v["lat"], v["lon"], speed_mps * dt
        )
        v["reached_end"] = reached
        v["is_predicted"] = False
        v["reb_interp"] = None

        v["speed_buffer"].append(v["speed"])
        v["speed_buffer"] = v["speed_buffer"][-5:]

        if v["elapsed"] >= v["next_model_sample_time"]:
            v["model_speed_window"].append(v["speed"])
            v["model_heading_window"].append(v["heading"])
            v["model_speed_window"] = v["model_speed_window"][-MODEL_HISTORY_LEN:]
            v["model_heading_window"] = v["model_heading_window"][-MODEL_HISTORY_LEN:]
            v["next_model_sample_time"] = v["elapsed"] + MODEL_SAMPLE_INTERVAL_SEC

    else:
        # --- signal lost: dead reckoning (mirrors the old server_core.py) ---
        if v["reb_interp"] is None:
            v["reb_interp"] = start_reb_prediction(v, v["elapsed"])

        model_speed_now = sample_interp_speed(v["reb_interp"], v["elapsed"])
        last_actual_speed = v["speed_buffer"][-1] if v["speed_buffer"] else ru.BASE_SPEED_KMH
        predicted_speed = max(15.0, ru.step_speed_toward(last_actual_speed, model_speed_now, dt_sec=dt))

        speed_mps = predicted_speed * (1000 / 3600)
        new_lat, new_lon, new_idx, reached = ru.advance_along_route(
            route_coords, v["idx"], v["lat"], v["lon"], speed_mps * dt
        )
        if new_idx != v["idx"]:
            v["heading"] = ru.heading_for_route_idx(route_coords, new_idx)

        v["lat"], v["lon"], v["idx"] = new_lat, new_lon, new_idx
        v["reached_end"] = reached
        v["is_predicted"] = True
        v["speed"] = predicted_speed

        v["speed_buffer"].append(predicted_speed)
        v["speed_buffer"] = v["speed_buffer"][-5:]

    return v


# ==========================================================================
# SESSION STATE
# ==========================================================================
def init_fleet(num_vehicles):
    spread = max(1, (ROUTE_LEN - 2) // max(1, num_vehicles))
    fleet = {}
    for i in range(num_vehicles):
        name = VEHICLE_NAMES[i % len(VEHICLE_NAMES)]
        fleet[name] = new_vehicle(
            name,
            start_offset_idx=i * spread,
            phase_offset_sec=i * (REB_CYCLE_SEC / max(1, num_vehicles)),
        )
    return fleet


if "num_vehicles" not in st.session_state:
    st.session_state.num_vehicles = 2
if "fleet" not in st.session_state:
    st.session_state.fleet = init_fleet(st.session_state.num_vehicles)
if "running" not in st.session_state:
    st.session_state.running = True
if "speed_multiplier" not in st.session_state:
    st.session_state.speed_multiplier = 1.0
if "last_tick_wall_time" not in st.session_state:
    st.session_state.last_tick_wall_time = time.time()


# ==========================================================================
# SIDEBAR — controls
# ==========================================================================
with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("▶️ Resume" if not st.session_state.running else "⏸️ Pause", use_container_width=True):
        st.session_state.running = not st.session_state.running

    if st.button("🔄 Reset simulation", use_container_width=True):
        st.session_state.fleet = init_fleet(st.session_state.num_vehicles)
        st.session_state.running = True

    num_vehicles = st.slider("Fleet size", min_value=1, max_value=5, value=st.session_state.num_vehicles)
    if num_vehicles != st.session_state.num_vehicles:
        st.session_state.num_vehicles = num_vehicles
        st.session_state.fleet = init_fleet(num_vehicles)

    st.session_state.speed_multiplier = st.slider(
        "Playback speed", min_value=0.5, max_value=4.0, value=st.session_state.speed_multiplier, step=0.5,
        help="Fast-forwards the simulation clock (does not affect realism of the physics model).",
    )

    st.divider()
    st.caption(
        "🟢 **Solid GPS fix** — live position from the vehicle.\n\n"
        "🔴 **Dead reckoning** — signal is jammed; position is estimated from "
        "a trained speed model and known route geometry until the GPS fix returns."
    )
    st.divider()
    st.caption(
        "This is a self-contained simulation for demonstration purposes: no live vehicles, "
        "no real jamming equipment, no personal data involved."
    )


# ==========================================================================
# MAIN CONTENT (auto-refreshing fragment)
# ==========================================================================
st.title("🛰️ Orbital City Lab")
st.caption("Signal-resilient public transit tracking during electronic-warfare (REB) interference")


@st.fragment(run_every=REFRESH_SEC)
def render():
    now = time.time()
    real_dt = now - st.session_state.last_tick_wall_time
    st.session_state.last_tick_wall_time = now
    real_dt = max(0.0, min(real_dt, 2.0))  # guard against long pauses/tab switches

    if st.session_state.running:
        sim_dt = real_dt * st.session_state.speed_multiplier
        # Step in fixed TICK_SEC increments so physics matches the tuned constants.
        remaining = sim_dt
        while remaining > 0:
            step = min(TICK_SEC, remaining)
            for v in st.session_state.fleet.values():
                step_vehicle(v, step)
            remaining -= step

    fleet = st.session_state.fleet
    all_done = all(v["reached_end"] for v in fleet.values())

    col_stats, col_map = st.columns([1, 2])

    with col_stats:
        st.subheader("Fleet status")
        for v in fleet.values():
            if v["reached_end"]:
                badge, bg = "🏁 Route complete", "#e0e0e0"
            elif v["is_predicted"]:
                badge, bg = "🔴 Signal lost — estimating position", "#ffd6d6"
            else:
                badge, bg = "🟢 GPS signal stable", "#d6ffd6"

            st.markdown(
                f"""
                <div style="background-color:{bg}; padding:12px 14px; border-radius:10px;
                            color:#111; margin-bottom:10px;">
                    <b>{v['name']}</b><br>
                    {badge}<br>
                    Speed: {v['speed']:.1f} km/h
                </div>
                """,
                unsafe_allow_html=True,
            )

        if all_done:
            st.success("All vehicles have reached the end of the route.")

    with col_map:
        layers = []

        path_layer = pdk.Layer(
            "PathLayer",
            data=[{"path": [[p[1], p[0]] for p in route_coords]}],
            get_path="path",
            get_color=[150, 150, 150, 150],
            width_scale=20,
            width_min_pixels=3,
        )
        layers.append(path_layer)

        endpoints = [
            {"coord": [route_coords[0][1], route_coords[0][0]], "color": [0, 150, 0, 200]},
            {"coord": [route_coords[-1][1], route_coords[-1][0]], "color": [150, 0, 0, 200]},
        ]
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=endpoints,
            get_position="coord",
            get_fill_color="color",
            get_radius=50,
            radius_min_pixels=6,
        ))

        v_list = []
        for v in fleet.values():
            color = [255, 60, 60, 255] if v["is_predicted"] else [50, 200, 60, 255]
            v_list.append({
                "coord": [v["lon"], v["lat"]],
                "color": color,
                "tooltip": f"{v['name']} — {v['speed']:.0f} km/h",
            })
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=v_list,
            get_position="coord",
            get_fill_color="color",
            get_line_color=[0, 0, 0, 255],
            stroked=True,
            line_width_min_pixels=2,
            get_radius=60,
            radius_min_pixels=10,
        ))

        first = next(iter(fleet.values()))
        view_state = pdk.ViewState(latitude=first["lat"], longitude=first["lon"], zoom=14, pitch=30)

        st.pydeck_chart(
            pdk.Deck(layers=layers, initial_view_state=view_state, tooltip={"text": "{tooltip}"}),
            use_container_width=True,
        )


render()
