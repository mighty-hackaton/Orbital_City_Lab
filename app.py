"""
Orbital City Lab — REBoot: Resilient Location Layer (Sensor Fusion)
====================================================================

Public transit vehicles keep sending GPS over the mobile network. During
electronic warfare (REB) activity that network can drop out for seconds at
a time. Earlier versions of this demo covered that gap with GPS-only dead
reckoning (a trained speed model + route geometry). That degrades the
longer the outage lasts — errors compound with nothing to correct them.

This version adds a second, independent signal: Wi-Fi fingerprinting.
While GPS/GSM is up, the vehicle also records which Wi-Fi access points it
sees (MAC + RSSI) at each point along the route — a "radio fingerprint" of
the street. A GPS/GSM jammer does not touch 2.4/5 GHz Wi-Fi, so during an
outage the vehicle's Wi-Fi radio keeps seeing the same APs. Matching the
live scan against the fingerprint database gives an independent position
estimate that is used to pull the AI + route-geometry prediction back
toward reality, instead of letting dead-reckoning error grow unchecked.

  * green solid GPS fix       — position comes straight from the vehicle
  * red dead reckoning        — GPS is lost; position = AI speed model +
                                 route geometry, corrected by Wi-Fi match
                                 whenever the fingerprint DB recognizes the
                                 area
  * uncertainty ring          — grows with time-since-signal, shrinks when
                                 a confident Wi-Fi match is found
  * gray last-GPS marker      — the last position we actually confirmed

Everything (fleet state, physics step, AI inference, Wi-Fi fingerprint DB)
runs inside this one Streamlit process. The fingerprint database is a
*shared, cached resource* rather than per-session state: every simulated
vehicle across every visitor contributes to and benefits from the same
route knowledge, which is how a real multi-vehicle fleet would work against
a central server DB.

Real Wi-Fi scanning requires native radio access unavailable from a browser
/ Streamlit session, so the scan itself (wifi_fingerprint.scan_at) is
simulated with a physically-motivated signal propagation model. Everything
downstream of that — the fingerprint DB, matching, and position correction
in wifi_fingerprint.py — is the real algorithm; feeding it genuine phone
Wi-Fi scans instead of scan_at() requires no change to this logic.
"""

import time

import joblib
import numpy as np
import pydeck as pdk
import streamlit as st

import feature_utils
import route_utils as ru
import wifi_fingerprint as wifi_fp

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

# Max single-tick "snap" distance when Wi-Fi correction pulls the display
# toward its target. EMA smoothing alone doesn't fix this (verified
# empirically): scan_at()'s shadow fading isn't seeded, so the RSSI scan (and
# therefore the match) is noisy tick to tick even at a near-constant true
# position, and the match is systematically biased backward too (averaging
# the top-5 fingerprints centroids toward the already-traveled stretch, not
# the current edge) — EMA only softens that pull, it doesn't remove it, so
# the display still drifted backward within a route segment periodically.
# Instead the display ALWAYS advances forward by at least dr_step_m (the
# same route-geometry physics dr uses), and Wi-Fi is only allowed to snap the
# display CLOSER to dr than that forward step already is — otherwise the
# correction is ignored for that tick. This structurally rules out both
# freezing (there's always a forward step) and backward motion (correction
# never applies if it would move the display away from dr).
MAX_WIFI_SNAP_M = 15.0

SPEED_NOISE_STD_KMH = 0.9
SPEED_NOISE_DECAY = 0.85
SPEED_NOISE_CLAMP_KMH = 3.0

# Скільки метрів "невизначеності" додається щосекунди без сигналу, і де це
# зростання стелиться (кільце на карті ніколи не росте нескінченно — після
# певної межі воно вже не несе нової інформації для пасажира).
UNCERTAINTY_GROWTH_M_PER_SEC = 7.0
MAX_UNCERTAINTY_RADIUS_M = 220.0
# Впевнений Wi-Fi-збіг звужує кільце (ми більше НЕ гадаємо наосліп), але не
# обнуляє його повністю — фінгерпринт теж не хірургічно точний.
WIFI_UNCERTAINTY_SHRINK = 0.55

VEHICLE_NAMES = ["Bus 101", "Bus 204", "Tram 7", "Trolley 12", "Bus 318"]

st.set_page_config(
    page_title="REBoot — трекер громадського транспорту",
    page_icon="📡",
    layout="wide",
)

# ==========================================================================
# STYLE — dark, "signal-resilience" visual language for REBoot
# ==========================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

    .stApp {
        background:
            radial-gradient(1200px 600px at 15% -10%, rgba(61,217,196,0.08), transparent 60%),
            radial-gradient(1000px 500px at 100% 0%, rgba(56,189,248,0.06), transparent 55%),
            #0A0E17;
    }

    section[data-testid="stSidebar"] {
        background: #0D1220;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }

    .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }

    /* ---- brand header ---- */
    .reboot-header {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 4px 0 6px 0;
        margin-bottom: 4px;
    }
    .reboot-logo {
        width: 52px; height: 52px; border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        font-size: 26px;
        background: linear-gradient(135deg, #3DD9C4 0%, #2563EB 100%);
        box-shadow: 0 0 24px rgba(61,217,196,0.35);
        flex-shrink: 0;
    }
    .reboot-title { font-size: 30px; font-weight: 800; color: #F3F6FB; letter-spacing: -0.02em; line-height: 1.1; }
    .reboot-title span { color: #3DD9C4; }
    .reboot-subtitle { font-size: 14.5px; color: #8B96AB; margin-top: 2px; }
    .reboot-badge {
        margin-left: auto; align-self: flex-start;
        background: rgba(61,217,196,0.12); color: #3DD9C4;
        border: 1px solid rgba(61,217,196,0.35);
        font-size: 11.5px; font-weight: 700; letter-spacing: 0.06em;
        padding: 5px 10px; border-radius: 999px; text-transform: uppercase;
        white-space: nowrap;
    }

    /* ---- legend strip ---- */
    .reboot-legend {
        display: flex; gap: 22px; flex-wrap: wrap;
        padding: 10px 16px; margin: 10px 0 18px 0;
        background: #121826; border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px; font-size: 13px; color: #C4CCDB;
    }
    .reboot-legend b { color: #F3F6FB; }
    .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:7px; vertical-align:middle; }
    .ring { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:7px; vertical-align:middle; border:2px solid #FB7185; }

    /* ---- section labels ---- */
    .reboot-section-label {
        font-size: 12px; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: #6C7890; margin: 2px 0 10px 2px;
    }

    /* ---- vehicle cards ---- */
    .vcard {
        border-radius: 14px; padding: 14px 16px; margin-bottom: 12px;
        background: #121826; border: 1px solid rgba(255,255,255,0.06);
        border-left: 3px solid var(--accent, #3DD9C4);
        transition: border-color .2s ease;
    }
    .vcard-top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
    .vcard-name { font-weight: 700; font-size: 15px; color: #F3F6FB; }
    .vcard-name .vicon { margin-right: 8px; opacity: 0.85; }
    .vcard-pill {
        font-size: 11px; font-weight: 700; letter-spacing: 0.03em;
        padding: 3px 9px; border-radius: 999px; white-space: nowrap;
    }
    .pill-live { background: rgba(52,211,153,0.14); color: #34D399; border: 1px solid rgba(52,211,153,0.35); }
    .pill-lost { background: rgba(251,113,133,0.14); color: #FB7185; border: 1px solid rgba(251,113,133,0.4); animation: pulse 1.4s ease-in-out infinite; }
    .pill-done { background: rgba(139,150,171,0.14); color: #8B96AB; border: 1px solid rgba(139,150,171,0.3); }
    .pill-info { background: rgba(56,189,248,0.14); color: #38BDF8; border: 1px solid rgba(56,189,248,0.35); }
    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.55; } }

    .vcard-row2 { display: flex; align-items: center; justify-content: space-between; margin-top: 8px; gap: 8px; }
    .vcard-speed { font-family: 'JetBrains Mono', monospace; font-size: 22px; font-weight: 600; color: #E7ECF5; }
    .vcard-speed span { font-size: 12px; font-weight: 500; color: #6C7890; margin-left: 4px; }

    .vcard-trust { text-align: right; }
    .vcard-trust-val { font-family: 'JetBrains Mono', monospace; font-size: 15px; font-weight: 700; }
    .vcard-trust-label { font-size: 10.5px; color: #6C7890; text-transform: uppercase; letter-spacing: 0.04em; }

    .vcard-errline { font-size: 11.5px; color: #6C7890; margin-top: 6px; }

    hr, div[data-testid="stDivider"] { border-color: rgba(255,255,255,0.08) !important; }

    .stButton > button {
        border-radius: 10px !important; font-weight: 600 !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================
# CACHED RESOURCES (loaded once per server process, shared read-only /
# shared-mutable in the case of the Wi-Fi fingerprint DB)
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


@st.cache_data(show_spinner=False)
def load_cum_dist(_route_coords):
    return ru.build_cumulative_distances(_route_coords)


@st.cache_resource(show_spinner="Генерація поля Wi-Fi точок доступу...")
def load_ap_field(_route_coords):
    return wifi_fp.build_ap_field(_route_coords)


@st.cache_resource(show_spinner=False)
def get_fingerprint_store():
    """Спільна для ВСІХ сесій/відвідувачів база фінгерпринтів — моделює
    центральну серверну БД, яку поповнює весь флот, а не один автобус."""
    return wifi_fp.FingerprintStore()


ai_model = load_model()
route_coords = load_route()
ROUTE_LEN = len(route_coords)
cum_dist_arr = load_cum_dist(route_coords)
ap_field = load_ap_field(route_coords)
fp_store = get_fingerprint_store()


# ==========================================================================
# SIMULATION — one simulated vehicle's worth of state + physics
# ==========================================================================
def new_vehicle(name, start_offset_idx, phase_offset_sec):
    idx = min(start_offset_idx, ROUTE_LEN - 2)
    lat, lon = route_coords[idx]
    heading = ru.heading_for_route_idx(route_coords, idx)
    return {
        "name": name,
        # --- reported/displayed position: what the passenger sees ---
        "idx": idx, "lat": lat, "lon": lon,
        # --- raw dead-reckoning accumulator: advances every REB tick on its
        # own, independent of the Wi-Fi-corrected displayed position (see
        # note in step_vehicle for why the two must not feed into each
        # other) ---
        "dr_idx": idx, "dr_lat": lat, "dr_lon": lon,
        "speed": ru.BASE_SPEED_KMH, "heading": heading,
        # --- hidden ground truth: the vehicle keeps moving for real even
        # when we can't hear from it. Only used inside this simulation to
        # (a) feed the Wi-Fi scan simulator a real physical location, and
        # (b) compute the "correction actually helped" diagnostic below. ---
        "true_idx": idx, "true_lat": lat, "true_lon": lon,
        "true_speed": ru.BASE_SPEED_KMH, "true_heading": heading, "true_speed_noise": 0.0,
        # --- naive baseline: "last known speed, held constant, straight
        # along the route" — no braking model, no AI, no Wi-Fi. Shown next
        # to the fused estimate to make the value of the fusion visible. ---
        "naive_idx": idx, "naive_lat": lat, "naive_lon": lon, "naive_speed": ru.BASE_SPEED_KMH,
        "speed_buffer": [],
        "model_speed_window": [],
        "model_heading_window": [],
        "next_model_sample_time": phase_offset_sec,
        "elapsed": phase_offset_sec,
        "phase_offset": phase_offset_sec,
        "reb_interp": None,
        "is_predicted": False,
        "reached_end": False,
        # --- Wi-Fi fusion state ---
        "wifi_status": "ok",       # ok | no_ap | matched | searching
        "wifi_confidence": 0.0,
        "signal_lost_since": None,
        "last_gps_lat": lat, "last_gps_lon": lon,
        "error_ai_m": 0.0, "error_fused_m": 0.0,
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


def uncertainty_radius_m(v):
    """Скільки метрів невизначеності показувати кільцем на карті: росте,
    поки немає GPS, і стискається, коли Wi-Fi впевнено впізнав місце."""
    if not v["is_predicted"] or v["signal_lost_since"] is None:
        return 0.0
    dt_lost = max(0.0, v["elapsed"] - v["signal_lost_since"])
    base = min(MAX_UNCERTAINTY_RADIUS_M, UNCERTAINTY_GROWTH_M_PER_SEC * dt_lost)
    shrink = 1.0 - WIFI_UNCERTAINTY_SHRINK * v.get("wifi_confidence", 0.0)
    return base * shrink


def trust_pct(v):
    """Єдине число 0-100%, що агрегує "наскільки вірити позиції на екрані"
    для пасажира — 100% на живому GPS, спадає з часом без сигналу, частково
    відновлюється Wi-Fi-корекцією."""
    if v["reached_end"]:
        return 100
    if not v["is_predicted"]:
        return 100
    r = uncertainty_radius_m(v)
    return int(round(max(5, 100 * (1 - r / MAX_UNCERTAINTY_RADIUS_M))))


def step_vehicle(v, dt):
    if v["reached_end"]:
        return v

    v["elapsed"] += dt
    signal_lost = is_signal_lost(v["elapsed"])

    # ---------------------------------------------------------------
    # GROUND TRUTH: the vehicle physically keeps moving with realistic
    # physics regardless of whether we can currently hear from it. This
    # hidden state is what a real vehicle would be doing; losing GPS/GSM
    # doesn't stop the bus, it just stops us from being told where it is.
    # ---------------------------------------------------------------
    clean_target = ru.target_speed_for_position(route_coords, v["true_idx"])
    v["true_speed_noise"] = v["true_speed_noise"] * SPEED_NOISE_DECAY + np.random.normal(0.0, SPEED_NOISE_STD_KMH)
    v["true_speed_noise"] = max(-SPEED_NOISE_CLAMP_KMH, min(SPEED_NOISE_CLAMP_KMH, v["true_speed_noise"]))
    noisy_target = max(0.0, clean_target + v["true_speed_noise"])
    v["true_speed"] = ru.step_speed_toward(v["true_speed"], noisy_target, dt_sec=dt)
    v["true_heading"] = ru.heading_for_route_idx(route_coords, v["true_idx"])
    true_speed_mps = v["true_speed"] * (1000 / 3600)
    v["true_lat"], v["true_lon"], v["true_idx"], reached = ru.advance_along_route(
        route_coords, v["true_idx"], v["true_lat"], v["true_lon"], true_speed_mps * dt
    )
    v["reached_end"] = reached

    if not signal_lost:
        # --- GPS/GSM link is up: what we report IS the truth ---
        v["idx"], v["lat"], v["lon"] = v["true_idx"], v["true_lat"], v["true_lon"]
        v["dr_idx"], v["dr_lat"], v["dr_lon"] = v["true_idx"], v["true_lat"], v["true_lon"]
        v["speed"], v["heading"] = v["true_speed"], v["true_heading"]
        v["is_predicted"] = False
        v["reb_interp"] = None
        v["signal_lost_since"] = None
        v["last_gps_lat"], v["last_gps_lon"] = v["lat"], v["lon"]
        v["error_ai_m"], v["error_fused_m"] = 0.0, 0.0

        v["speed_buffer"].append(v["speed"])
        v["speed_buffer"] = v["speed_buffer"][-5:]

        if v["elapsed"] >= v["next_model_sample_time"]:
            v["model_speed_window"].append(v["speed"])
            v["model_heading_window"].append(v["heading"])
            v["model_speed_window"] = v["model_speed_window"][-MODEL_HISTORY_LEN:]
            v["model_heading_window"] = v["model_heading_window"][-MODEL_HISTORY_LEN:]
            v["next_model_sample_time"] = v["elapsed"] + MODEL_SAMPLE_INTERVAL_SEC

        # Record a Wi-Fi fingerprint at this (known-true) position so future
        # dead-reckoning — by this vehicle OR any other one on the same
        # route — can be corrected against it.
        live_scan = wifi_fp.scan_at(v["lat"], v["lon"], ap_field)
        cum_dist_here = ru.cumulative_distance_at(route_coords, cum_dist_arr, v["idx"], v["lat"], v["lon"])
        fp_store.maybe_record(v["name"], v["lat"], v["lon"], v["idx"], live_scan, cum_dist_here)
        v["wifi_status"] = "ok" if live_scan else "no_ap"
        v["wifi_confidence"] = 0.0

        # Naive baseline resyncs to reality every time we actually know
        # where the vehicle is — it only diverges once signal is lost.
        v["naive_idx"], v["naive_lat"], v["naive_lon"] = v["idx"], v["lat"], v["lon"]
        v["naive_speed"] = v["speed"]

    else:
        # --- signal lost: AI + route-geometry dead reckoning ---
        if v["reb_interp"] is None:
            v["reb_interp"] = start_reb_prediction(v, v["elapsed"])
            v["signal_lost_since"] = v["elapsed"]

        model_speed_now = sample_interp_speed(v["reb_interp"], v["elapsed"])
        last_actual_speed = v["speed_buffer"][-1] if v["speed_buffer"] else ru.BASE_SPEED_KMH
        predicted_speed = max(15.0, ru.step_speed_toward(last_actual_speed, model_speed_now, dt_sec=dt))
        speed_mps = predicted_speed * (1000 / 3600)

        # Advance the RAW dr accumulator from its own previous step, not from
        # v["lat"]/["lon"] — those may already be pulled toward a Wi-Fi
        # match below. Starting from the corrected position would feed that
        # same correction back in every tick: the point converges to a fixed
        # offset from the matched fingerprint and stalls there for the rest
        # of the outage, then snaps to the real position once GPS returns.
        prev_dr_lat, prev_dr_lon = v["dr_lat"], v["dr_lon"]
        dr_lat, dr_lon, dr_idx, _ = ru.advance_along_route(
            route_coords, v["dr_idx"], v["dr_lat"], v["dr_lon"], speed_mps * dt
        )
        v["dr_idx"], v["dr_lat"], v["dr_lon"] = dr_idx, dr_lat, dr_lon
        dr_step_m = ru.calculate_distance(prev_dr_lat, prev_dr_lon, dr_lat, dr_lon)

        v["speed"] = predicted_speed
        v["speed_buffer"].append(predicted_speed)
        v["speed_buffer"] = v["speed_buffer"][-5:]

        # Naive baseline: keeps the speed frozen at whatever it was the
        # instant signal was lost, no braking-before-turn model at all —
        # this is "what you'd get with plain extrapolation, no AI, no Wi-Fi".
        naive_speed_mps = v["naive_speed"] * (1000 / 3600)
        v["naive_lat"], v["naive_lon"], v["naive_idx"], _ = ru.advance_along_route(
            route_coords, v["naive_idx"], v["naive_lat"], v["naive_lon"], naive_speed_mps * dt
        )

        # Wi-Fi correction: the phone's Wi-Fi radio isn't jammed, so it
        # keeps scanning APs at the vehicle's TRUE physical location. Match
        # that scan against the fingerprint DB to pull the AI+geometry
        # estimate back toward reality instead of letting it drift freely.
        live_scan = wifi_fp.scan_at(v["true_lat"], v["true_lon"], ap_field)
        match = fp_store.match(live_scan)
        if match is not None:
            w = wifi_fp.correction_weight(match["confidence"])
            corrected_lat = (1 - w) * dr_lat + w * match["lat"]
            corrected_lon = (1 - w) * dr_lon + w * match["lon"]
            v["wifi_status"] = "matched" if w > 0 else "searching"
            v["wifi_confidence"] = match["confidence"]
        else:
            corrected_lat, corrected_lon = dr_lat, dr_lon
            v["wifi_status"] = "searching"
            v["wifi_confidence"] = 0.0

        # Forward baseline: the display ALWAYS advances along the route by
        # dr_step_m (the same route-geometry physics dr uses) — guaranteed
        # forward, never frozen.
        fwd_lat, fwd_lon, _, _ = ru.advance_along_route(route_coords, v["idx"], v["lat"], v["lon"], dr_step_m)

        # Wi-Fi correction is only applied if it puts the display CLOSER to
        # dr than the forward baseline already is — otherwise this tick's
        # correction is ignored and the forward step stands. Even when
        # accepted, the jump toward it is capped at MAX_WIFI_SNAP_M.
        dist_corrected_to_dr = ru.calculate_distance(corrected_lat, corrected_lon, dr_lat, dr_lon)
        dist_fwd_to_dr = ru.calculate_distance(fwd_lat, fwd_lon, dr_lat, dr_lon)

        if dist_corrected_to_dr < dist_fwd_to_dr:
            snap_step_m = ru.calculate_distance(fwd_lat, fwd_lon, corrected_lat, corrected_lon)
            ratio = min(1.0, MAX_WIFI_SNAP_M / snap_step_m) if snap_step_m > 1e-9 else 1.0
            display_lat = fwd_lat + (corrected_lat - fwd_lat) * ratio
            display_lon = fwd_lon + (corrected_lon - fwd_lon) * ratio
        else:
            display_lat, display_lon = fwd_lat, fwd_lon

        display_idx = ru.nearest_route_index(route_coords, display_lat, display_lon, search_from=v["idx"])
        display_idx = max(display_idx, v["idx"])  # safety net: never regress

        if display_idx != v["idx"]:
            v["heading"] = ru.heading_for_route_idx(route_coords, display_idx)

        v["idx"], v["lat"], v["lon"] = display_idx, display_lat, display_lon
        v["is_predicted"] = True

        # Diagnostic only (this simulation controls ground truth, a real
        # deployment would not have it): how much did Wi-Fi actually help?
        v["error_ai_m"] = ru.calculate_distance(dr_lat, dr_lon, v["true_lat"], v["true_lon"])
        v["error_fused_m"] = ru.calculate_distance(display_lat, display_lon, v["true_lat"], v["true_lon"])

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
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:18px;">
            <div style="width:34px; height:34px; border-radius:10px; display:flex; align-items:center;
                        justify-content:center; font-size:17px;
                        background:linear-gradient(135deg,#3DD9C4 0%,#2563EB 100%);">📡</div>
            <div style="font-weight:800; font-size:17px; color:#F3F6FB;">RE<span style="color:#3DD9C4;">Boot</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="reboot-section-label">Керування симуляцією</div>', unsafe_allow_html=True)

    if st.button(
        "▶️  Відновити" if not st.session_state.running else "⏸️  Пауза",
        use_container_width=True,
    ):
        st.session_state.running = not st.session_state.running

    if st.button("🔄  Скинути симуляцію", use_container_width=True):
        st.session_state.fleet = init_fleet(st.session_state.num_vehicles)
        st.session_state.running = True

    st.write("")
    num_vehicles = st.slider("Розмір парку", min_value=1, max_value=5, value=st.session_state.num_vehicles)
    if num_vehicles != st.session_state.num_vehicles:
        st.session_state.num_vehicles = num_vehicles
        st.session_state.fleet = init_fleet(num_vehicles)

    st.session_state.speed_multiplier = st.slider(
        "Швидкість відтворення", min_value=0.5, max_value=4.0, value=st.session_state.speed_multiplier, step=0.5,
        help="Прискорює хід симуляції (на реалістичність фізичної моделі не впливає).",
    )

    st.divider()
    st.markdown('<div class="reboot-section-label">Wi-Fi fingerprint база</div>', unsafe_allow_html=True)
    st.caption(
        f"📶 Записано **{len(fp_store.records)}** фінгерпринтів вздовж маршруту "
        f"(спільна база для всього флоту, включно з іншими відвідувачами демо)."
    )
    if st.button("🧹  Очистити Wi-Fi базу", use_container_width=True):
        fp_store.clear()

    st.divider()
    st.markdown('<div class="reboot-section-label">Умовні позначення</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size:13px; line-height:1.6; color:#C4CCDB;">
            <span class="dot" style="background:#34D399;"></span>
            <b style="color:#F3F6FB;">GPS-сигнал стабільний</b> — позиція надходить напряму від транспорту,
            фінгерпринт локації записується в базу.<br><br>
            <span class="dot" style="background:#FB7185;"></span>
            <b style="color:#F3F6FB;">Втрата сигналу (РЕБ)</b> — позиція = AI-модель швидкості + геометрія
            маршруту, скоригована Wi-Fi-фінгерпринтом, якщо система впізнала місце.<br><br>
            <span class="ring"></span>
            <b style="color:#F3F6FB;">Кільце невизначеності</b> — росте з часом без сигналу, стискається
            при впевненому Wi-Fi-збігу.<br><br>
            <span class="dot" style="background:#8B96AB;"></span>
            <b style="color:#F3F6FB;">Остання відома GPS-точка</b> — де сигнал востаннє підтвердив позицію.<br><br>
            <span class="dot" style="background:#3B82F6;"></span>
            <b style="color:#F3F6FB;">Wi-Fi точка доступу</b> — синтетична AP вздовж маршруту; прозоре коло —
            дальність, у межах якої вона видима скану.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption(
        "Це автономна симуляція для демонстрації: реальних транспортних засобів, "
        "обладнання РЕБ, справжнього Wi-Fi сканування чи персональних даних тут немає. "
        "Wi-Fi-скан симулюється фізичною моделлю затухання сигналу; алгоритм зіставлення "
        "та корекції позиції — робочий і не зміниться при підключенні реального сканера."
    )


# ==========================================================================
# MAIN CONTENT (auto-refreshing fragment)
# ==========================================================================
st.markdown(
    """
    <div class="reboot-header">
        <div class="reboot-logo">📡</div>
        <div>
            <div class="reboot-title">RE<span>Boot</span></div>
            <div class="reboot-subtitle">Стійкий шар геолокації для міського транспорту — GPS + AI + Wi-Fi fingerprint</div>
        </div>
        <div class="reboot-badge">MVP · Sensor Fusion</div>
    </div>
    <div class="reboot-legend">
        <div><span class="dot" style="background:#34D399;"></span><b>GPS стабільний</b> — реальна позиція</div>
        <div><span class="dot" style="background:#FB7185;"></span><b>Втрата сигналу</b> — AI + Wi-Fi корекція</div>
        <div><span class="ring"></span><b>Невизначеність</b> — росте без сигналу, стискається Wi-Fi</div>
        <div><span class="dot" style="background:#8B96AB;"></span><b>Остання GPS-точка</b></div>
        <div><span class="dot" style="background:#3B82F6;"></span><b>Wi-Fi точка доступу</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)


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
        st.markdown('<div class="reboot-section-label">Стан парку</div>', unsafe_allow_html=True)

        def vehicle_icon(name):
            n = name.lower()
            if "tram" in n:
                return "🚊"
            if "trolley" in n:
                return "🚎"
            return "🚌"

        WIFI_LABELS = {
            "ok": ("pill-info", "📶 База поповнюється"),
            "no_ap": ("pill-info", "📶 Немає AP поруч"),
            "matched": ("pill-live", None),   # текст підставляється нижче з confidence
            "searching": ("pill-lost", "📶 Wi-Fi: збігу ще немає"),
        }

        for v in fleet.values():
            if v["reached_end"]:
                pill_cls, pill_text, accent = "pill-done", "🏁 Маршрут завершено", "#8B96AB"
            elif v["is_predicted"]:
                pill_cls, pill_text, accent = "pill-lost", "🔴 GPS втрачено", "#FB7185"
            else:
                pill_cls, pill_text, accent = "pill-live", "🟢 GPS стабільний", "#34D399"

            wifi_cls, wifi_text = WIFI_LABELS[v["wifi_status"]]
            if v["wifi_status"] == "matched":
                wifi_text = f"📶 Wi-Fi збіг · довіра {v['wifi_confidence'] * 100:.0f}%"

            trust = trust_pct(v)
            trust_color = "#34D399" if trust >= 70 else ("#FBBF24" if trust >= 35 else "#FB7185")

            err_line = ""
            if v["is_predicted"] and not v["reached_end"]:
                err_line = (
                    f'<div class="vcard-errline">Симуляція (діагностика): без Wi-Fi ~{v["error_ai_m"]:.0f} м '
                    f'від реального положення · з Wi-Fi-корекцією ~{v["error_fused_m"]:.0f} м</div>'
                )

            st.markdown(
                f"""
                <div class="vcard" style="--accent:{accent};">
                    <div class="vcard-top">
                        <div class="vcard-name"><span class="vicon">{vehicle_icon(v['name'])}</span>{v['name']}</div>
                        <div class="vcard-pill {pill_cls}">{pill_text}</div>
                    </div>
                    <div class="vcard-row2">
                        <div>
                            <div class="vcard-speed">{v['speed']:.1f}<span>км/год</span></div>
                            <div class="vcard-pill {wifi_cls}" style="margin-top:6px; display:inline-block;">{wifi_text}</div>
                        </div>
                        <div class="vcard-trust">
                            <div class="vcard-trust-val" style="color:{trust_color};">{trust}%</div>
                            <div class="vcard-trust-label">довіра до позиції</div>
                        </div>
                    </div>
                    {err_line}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if all_done:
            st.success("Усі транспортні засоби дісталися кінця маршруту.")

    with col_map:
        st.markdown('<div class="reboot-section-label">Карта маршруту</div>', unsafe_allow_html=True)

        SIGNAL_GREEN = [45, 212, 158]
        SIGNAL_RED = [251, 113, 133]
        LAST_GPS_GRAY = [148, 163, 184]
        AP_BLUE = [59, 130, 246]

        layers = []

        path_layer = pdk.Layer(
            "PathLayer",
            data=[{"path": [[p[1], p[0]] for p in route_coords]}],
            get_path="path",
            get_color=[93, 111, 145, 160],
            width_scale=20,
            width_min_pixels=3,
        )
        layers.append(path_layer)

        # Wi-Fi access points: faint transparent circle showing simulated
        # detection range (AP_MAX_RANGE_M — beyond this scan_at() no longer
        # sees the AP), plus a small solid dot marking the AP itself.
        ap_positions = [{"coord": [ap["lon"], ap["lat"]]} for ap in ap_field]
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=ap_positions,
            get_position="coord",
            get_radius=wifi_fp.AP_MAX_RANGE_M,
            get_fill_color=AP_BLUE + [18],
            stroked=False,
            filled=True,
        ))
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=ap_positions,
            get_position="coord",
            get_fill_color=AP_BLUE + [230],
            get_line_color=[10, 14, 23, 255],
            stroked=True,
            line_width_min_pixels=1,
            get_radius=8,
            radius_min_pixels=3,
        ))

        endpoints = [
            {"coord": [route_coords[0][1], route_coords[0][0]], "color": SIGNAL_GREEN + [210]},
            {"coord": [route_coords[-1][1], route_coords[-1][0]], "color": [148, 163, 184, 210]},
        ]
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=endpoints,
            get_position="coord",
            get_fill_color="color",
            get_radius=50,
            radius_min_pixels=6,
        ))

        # Uncertainty ring: grows with time-since-signal, shrinks with a
        # confident Wi-Fi match — an outline, not a filled halo, so it reads
        # as "region of doubt" rather than just a bigger dot.
        ring_list = [
            {"coord": [v["lon"], v["lat"]], "radius": uncertainty_radius_m(v)}
            for v in fleet.values() if v["is_predicted"] and not v["reached_end"] and uncertainty_radius_m(v) > 1
        ]
        if ring_list:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=ring_list,
                get_position="coord",
                get_radius="radius",
                filled=False,
                stroked=True,
                get_line_color=SIGNAL_RED + [180],
                line_width_min_pixels=2,
            ))

        # Last confirmed GPS fix — where the signal actually last agreed
        # with reality, as a fixed reference point next to the estimate.
        last_gps_list = [
            {"coord": [v["last_gps_lon"], v["last_gps_lat"]]}
            for v in fleet.values() if v["is_predicted"] and not v["reached_end"]
        ]
        if last_gps_list:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=last_gps_list,
                get_position="coord",
                get_fill_color=LAST_GPS_GRAY + [200],
                get_line_color=[10, 14, 23, 255],
                stroked=True,
                line_width_min_pixels=1,
                get_radius=35,
                radius_min_pixels=5,
            ))

        v_list = []
        for v in fleet.values():
            color = (SIGNAL_RED if v["is_predicted"] else SIGNAL_GREEN) + [255]
            wifi_note = ""
            if v["is_predicted"]:
                wifi_note = (
                    f" · Wi-Fi довіра {v['wifi_confidence'] * 100:.0f}%"
                    if v["wifi_status"] == "matched" else " · Wi-Fi: пошук збігу"
                )
            v_list.append({
                "coord": [v["lon"], v["lat"]],
                "color": color,
                "tooltip": f"{v['name']} — {v['speed']:.0f} км/год"
                           f"{' · GPS втрачено' + wifi_note if v['is_predicted'] else ''}",
            })
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=v_list,
            get_position="coord",
            get_fill_color="color",
            get_line_color=[10, 14, 23, 255],
            stroked=True,
            line_width_min_pixels=2,
            get_radius=60,
            radius_min_pixels=10,
        ))

        first = next(iter(fleet.values()))
        view_state = pdk.ViewState(latitude=first["lat"], longitude=first["lon"], zoom=14, pitch=30)

        st.pydeck_chart(
            pdk.Deck(
                layers=layers,
                initial_view_state=view_state,
                tooltip={
                    "text": "{tooltip}",
                    "style": {"backgroundColor": "#121826", "color": "#E7ECF5", "fontSize": "13px"},
                },
                map_style="dark",
            ),
            use_container_width=True,
        )


render()
