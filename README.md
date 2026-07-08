# Orbital City Lab 🛰️

**A simulation of signal-resilient public transit tracking during GPS/GSM jamming (REB).**

In areas affected by electronic warfare (REB), a transit vehicle's GPS/GSM
link can drop out for several seconds at a time. This project simulates a
small bus/tram fleet moving along a real street route in Kyiv and shows what
a passenger-facing live map would look like in that situation:

- 🟢 **Solid GPS fix** — the vehicle's position comes directly from its GPS.
- 🔴 **Dead reckoning** — the signal is lost, so the app estimates the
  vehicle's position from a trained speed-prediction model combined with the
  known route geometry, until the GPS fix returns.

This is a simulation for demonstration purposes only: there are no real
vehicles, no real jamming equipment, and no personal data involved.

## Try it

▶️ **Live demo:** https://huggingface.co/spaces/mariykart/REBoot

## How it works

| Piece | Role |
|---|---|
| `app.py` | The Streamlit app. Runs the whole simulation in memory, per visitor session, and renders a live map + fleet status panel. |
| `route_utils.py` | Shared geometry helpers: distance/bearing, moving a point along the route polyline, speed-vs-corner physics (braking before turns, acceleration/braking limits). |
| `feature_utils.py` | Builds the exact 20-feature vector the ML model expects from a 5-sample window of speed + heading. |
| `model.pkl` / `model_metadata.json` | A tuned `HistGradientBoostingRegressor` (scikit-learn) trained to predict the **change** in speed 15 seconds ahead, given the last 5 speed/heading samples. |
| `route_info.json` | A cached road route (Podilskyi district, Kyiv) computed once from OpenStreetMap via `osmnx`, shipped with the repo so the app never needs OSM/Overpass access at runtime. |
| `server_core.py`, `client_sim.py` | An alternative, **local-only** multi-process mode (see below) that talks over real UDP sockets — useful if you want to see the client/server split explicitly, but not used by the deployed app. |

### Dead reckoning logic

While a vehicle is "connected", the app tracks its real speed and heading.
The moment the (simulated) signal drops:

1. The model predicts a speed **delta** for 15 seconds ahead from the last 5
   samples of speed and heading.
2. That prediction is blended 50/50 with a purely geometric estimate (the
   route's known speed profile at the vehicle's current position — e.g.
   slowing for an upcoming turn).
3. The blended target speed is reached by linear interpolation over the
   15-second window, and the vehicle's position is advanced along the known
   route polyline accordingly — so it never "flies off" the road during a
   simulated jam.
4. As soon as signal returns, the app switches back to reporting the
   (simulated) real GPS position.

## Notes on the model

The shipped model is pinned to run with `scikit-learn==1.6.1` (the version
it was trained with); loading it with a different version may still work but
can print a compatibility warning or silently degrade prediction quality.
See `model_metadata.json` for feature names, training metrics
(`val_mae_kmh`, `val_r2`, etc.), and the exact input format.
