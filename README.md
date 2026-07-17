# Orbital City Lab 🛰️

**A resilient location layer for city transit: GPS + AI dead reckoning + Wi-Fi fingerprint sensor fusion, demonstrated on a simulated Kyiv bus/tram fleet.**

In areas affected by electronic warfare (REB), a transit vehicle's **GNSS
(GPS) fix** can be jammed or spoofed for seconds at a time — even while its
GSM/LTE data link and Wi-Fi radio (different frequency bands) keep working.
This project simulates a small bus/tram fleet moving along a real street
route in Kyiv and shows what a passenger-facing live map would look like:

- 🟢 **Solid GPS fix** — the vehicle's position comes directly from GNSS.
- 🔴 **Dead reckoning, Wi-Fi-corrected** — GNSS is jammed, so position is
  estimated from a trained speed model + route geometry, and then pulled
  back toward reality using a **Wi-Fi fingerprint match** whenever the
  vehicle passes somewhere the fleet has already mapped.
- ⭕ **Uncertainty ring** — grows the longer GNSS stays down, shrinks when a
  confident Wi-Fi match is found.
- ⚪ **Last known GPS marker** — the last position that was actually
  confirmed, kept on screen next to the live estimate.

This is a simulation for demonstration purposes only: there are no real
vehicles, no real jamming equipment, no real Wi-Fi scanning hardware, and no
personal data involved.

## Why Wi-Fi fingerprinting, not just GPS extrapolation

Pure dead reckoning (speed model + route geometry) degrades monotonically:
the longer the outage, the larger the error, with nothing to correct it.
That's fine for a 5-10 second gap, but real REB-affected corridors can stay
jammed far longer, and error compounds with no ceiling.

A GNSS jammer targets the narrow GPS frequency band (~1575 MHz L1). It
generally does **not** touch 2.4/5 GHz Wi-Fi. So a vehicle's Wi-Fi radio
keeps seeing the same access points regardless of what's happening to its
GPS fix. If the fleet has already recorded *which APs (MAC + RSSI) are
visible where* along its known routes, a live scan during a GNSS outage can
be matched against that fingerprint database to get an **independent**
position estimate — one that doesn't share the same failure mode as GPS.
Blending that into the AI+geometry prediction turns unbounded dead-reckoning
drift into a periodically-corrected estimate.

Three graceful-degradation tiers fall out of this naturally:

| Tier | What's jammed | What still works | Position source |
|---|---|---|---|
| 1 | Nothing | GNSS, data link, Wi-Fi | Real GPS fix |
| 2 | GNSS only (the common real-world case) | Data link + Wi-Fi radio | AI + geometry, corrected by Wi-Fi fingerprint match |
| 3 | GNSS + data link (full blackout) | Nothing reaches the server | AI + geometry only (no fresh Wi-Fi scan to correct with) |

Because the fingerprint database is shared across the whole fleet (not
per-vehicle), a bus that has never driven a given block before still
benefits from fingerprints recorded by *other* vehicles that drove it
earlier the same day — the more vehicles run a route, the better the
correction gets. Real Wi-Fi scanning requires native radio access
unavailable from a browser/Streamlit session, so the scan itself is
simulated with a physically-motivated signal-propagation model
(`wifi_fingerprint.scan_at`); everything downstream — the fingerprint DB,
signal-space matching, and position correction in `wifi_fingerprint.py` — is
the real, deployable algorithm. Feeding it genuine phone/router Wi-Fi scans
instead of `scan_at()` requires no change to that logic.

## How it works

| Piece | Role |
|---|---|
| `app.py` | The Streamlit app. Runs the whole simulation (fleet physics, AI inference, Wi-Fi fingerprint recording + matching) in memory, and renders a live map + fleet status panel. The fingerprint database is a *shared, cached resource* — every visitor's simulated fleet reads from and writes to the same one, modeling a real fleet-wide server DB. |
| `route_utils.py` | Shared geometry helpers: distance/bearing, moving a point along the route polyline, speed-vs-corner physics, and cumulative-distance-along-route (used to space out fingerprint recording). |
| `feature_utils.py` | Builds the exact 20-feature vector the speed-prediction model expects from a 5-sample window of speed + heading. |
| `wifi_fingerprint.py` | The sensor-fusion layer: generates a synthetic Wi-Fi AP field along the route, simulates a scan at any point, stores fingerprints tagged with position, and matches a live scan against the database (signal-space k-NN, the standard Wi-Fi indoor-positioning technique) to produce a correction weight. |
| `model.pkl` / `model_metadata.json` | A tuned `HistGradientBoostingRegressor` (scikit-learn) trained to predict the **change** in speed 15 seconds ahead, given the last 5 speed/heading samples. |
| `route_info.json` | A cached road route (Podilskyi district, Kyiv) computed once from OpenStreetMap via `osmnx`, shipped with the repo so the app never needs OSM/Overpass access at runtime. |
| `server_core.py`, `client_sim.py` | An alternative, **local-only** multi-process mode over real UDP sockets that makes the client/server split (and the GNSS-vs-data-link distinction) explicit — see below. |

### The pipeline

```
GNSS (may be jammed)
   │
   ▼
Transport Client  ──────────────►  Wi-Fi Scanner (always live, 2.4/5 GHz ≠ GPS L1)
   │  (lat/lon + speed, when fixed)         │  (MAC/RSSI scan, always)
   ▼                                        ▼
                    Server
                       │
        ┌──────────────┼───────────────────┐
        ▼               ▼                  ▼
  Fingerprint DB    AI Prediction     Map Matching
  (record when      (speed model,     (snap to known
   GNSS is live)      geometry)        route polyline)
        │               │                  │
        └──────► Position Correction ◄──────┘
              (blend AI+geometry estimate
               toward Wi-Fi-matched position,
               weighted by match confidence)
                       │
                       ▼
                    Frontend
        (GPS/estimate, confidence, last-known
         GPS marker, uncertainty ring)
```

### Dead reckoning + Wi-Fi correction logic

While a vehicle has a GNSS fix, the app tracks its real speed, heading, and
records a Wi-Fi fingerprint (position ↔ visible APs) every ~25 m. The
moment GNSS is jammed:

1. The speed model predicts a speed **delta** for 15 seconds ahead from the
   last 5 samples of speed and heading, blended 50/50 with a purely
   geometric estimate (the route's known speed profile — e.g. slowing for
   an upcoming turn).
2. Position advances along the known route polyline using that blended
   target speed — so it never "flies off" the road.
3. In parallel, the vehicle's (still-live) Wi-Fi radio scan is matched
   against the fingerprint database. If enough access points overlap with a
   past recording, that gives an independent position estimate with a
   confidence score.
4. The AI+geometry position is pulled toward the Wi-Fi-matched position, by
   an amount proportional to match confidence (never fully replacing the
   AI estimate — Wi-Fi is a **correction**, not a substitute for GPS).
5. As soon as GNSS returns, the app switches back to reporting the real
   fix, and the naive/AI/fused error metrics reset.

## Two ways to run it

### 1. Single self-contained app (what's deployed)

Everything — simulated fleet, physics, AI inference, Wi-Fi fingerprint DB —
runs inside one Streamlit process, with no sockets or shared files. Every
visitor gets their own independent fleet simulation, but they all share the
same underlying (in-memory) fingerprint database, which is what makes the
"the more vehicles drive a route, the better the correction gets" story
visible even from a single deployed instance.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the printed local URL in your browser. Use the sidebar to
pause/resume, change fleet size, speed up playback, or clear the Wi-Fi
fingerprint database to see the "cold start" (no correction yet) vs.
"warmed up" (correction kicks in) contrast live.

### 2. Local advanced mode (real UDP client/server)

This mode makes the GNSS-vs-data-link distinction explicit: the simulated
vehicle keeps sending UDP packets throughout — with `lat`/`lon` when GNSS is
live, and without them (but still with a Wi-Fi scan) when GNSS is jammed.
The server treats "no packets at all" and "packets arriving but GNSS-less"
as two different failure tiers (see the table above).

```bash
# Terminal 1 — the "server": receives packets, builds the Wi-Fi fingerprint
# DB, runs AI+geometry dead reckoning, and corrects it via Wi-Fi match
python server_core.py

# Terminal 2 — the "vehicle" sending simulated GNSS + Wi-Fi packets over UDP
python client_sim.py
```

This mode writes `dashboard_state.json` to disk (position, `is_predicted`,
`wifi_status`, `wifi_confidence` per vehicle) — kept for reference/local
experimentation and as a ready-made data source for a future dispatcher
dashboard, but **not** used by the deployed, self-contained app, since a
shared on-disk state file doesn't work safely with multiple concurrent
visitors on a public site.

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   your GitHub account.
3. Click **New app**, pick this repository/branch, and set the main file
   path to `app.py`.
4. Deploy. Streamlit Cloud installs `requirements.txt` automatically and
   gives you a public `*.streamlit.app` URL you can share with anyone.

No extra configuration or secrets are required — the route is cached in
`route_info.json`, the model is loaded from `model.pkl`, and the Wi-Fi AP
field is generated on first load, all already in the repo.

## Notes on the model

The shipped model is pinned to run with `scikit-learn==1.6.1` (the version
it was trained with); loading it with a different version may still work but
can print a compatibility warning or silently degrade prediction quality —
in that case the app falls back to the geometric (route-profile) speed
estimate automatically. See `model_metadata.json` for feature names,
training metrics (`val_mae_kmh`, `val_r2`, etc.), and the exact input
format.
