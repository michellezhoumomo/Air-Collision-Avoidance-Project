# Simulation Design

This document describes how the BlueSky simulation is structured for this project,
what state is available at each step, how actions are applied, and how to derive
the columns needed for pysda visualization.

---

## BlueSky Basics

BlueSky is an open-source ATC simulator developed at TU Delft. In this project it
runs in **GUI mode (`mode='sim'`) for visualization and debugging**, and in
**headless mode during bulk training** for speed.

```python
import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy

# GUI mode — real-time map view, pause/resume with HOLD/OP
bs.init(mode='sim')

# Headless mode — faster, no display (use during training)
# bs.init(mode='sim', detached=True)
# bs.scr = ScreenDummy()               # suppress display output

bs.stack.stack('DT 5;FF')            # set timestep to 5s, run fast-forward
```

`DT` is the simulation timestep in seconds. `FF` means fast-forward — the sim
runs as fast as the CPU allows, not in real time.

---

## Aircraft State Available at Each Step

All aircraft state lives in `bs.traf`. Each field is a numpy array indexed by
aircraft index (`idx`). Get the index from the callsign:

```python
idx = bs.traf.id2idx('KL001')
```

### Position and movement

| Field | Units | Description |
|---|---|---|
| `bs.traf.lat[idx]` | degrees | Latitude |
| `bs.traf.lon[idx]` | degrees | Longitude |
| `bs.traf.alt[idx]` | meters | Altitude (MSL) |
| `bs.traf.hdg[idx]` | degrees | True heading |
| `bs.traf.gs[idx]` | m/s | Ground speed |
| `bs.traf.vs[idx]` | m/s | Vertical speed (positive = climb) |
| `bs.traf.tas[idx]` | m/s | True airspeed |

### Autopilot selected values

| Field | Units | Description |
|---|---|---|
| `bs.traf.selalt[idx]` | meters | Target altitude |
| `bs.traf.selvs[idx]` | m/s | Target vertical speed |
| `bs.traf.selspd[idx]` | m/s | Target speed |
| `bs.traf.swvnav[idx]` | bool | VNAV (vertical nav) enabled |
| `bs.traf.swlnav[idx]` | bool | LNAV (lateral nav) enabled |

### Geometry helpers

```python
# bearing (degrees) and distance (NM) between two lat/lon points
qdr, dist_nm = bs.tools.geo.kwikqdrdist(lat1, lon1, lat2, lon2)

# distance only (NM)
dist_nm = bs.tools.geo.kwikdist(lat1, lon1, lat2, lon2)

# point at a given distance and bearing from a lat/lon
lat2, lon2 = fn.get_point_at_distance(lat1, lon1, dist_nm, bearing_deg)
```

---

## Creating Aircraft

```python
bs.traf.cre(
    acid='KL001',       # callsign
    actype='A320',      # aircraft type
    aclat=52.31,        # initial latitude
    aclon=4.77,         # initial longitude
    achdg=270,          # initial heading (degrees)
    acalt=3000,         # initial altitude (meters)
    acspd=180           # initial speed (knots)
)
```

To disable autopilot so the agent has full control:

```python
idx = bs.traf.id2idx('KL001')
bs.traf.swvnav[idx] = False   # disable vertical nav
bs.traf.swlnav[idx] = False   # disable lateral nav
```

---

## Applying Actions

Actions are sent via `bs.stack.stack()` — the same command interface a human
controller would use.

### Heading change (2D and 3D)

```python
new_hdg = current_hdg + action[0] * D_HEADING   # D_HEADING = 45 degrees max
bs.stack.stack(f'HDG KL001 {new_hdg}')
```

### Speed change (2D and 3D)

```python
new_spd = np.clip(current_spd + action[1] * D_SPEED, SPD_MIN, SPD_MAX)
bs.stack.stack(f'SPD KL001 {new_spd}')
```

### Vertical speed change (3D only)

BlueSky does not accept a direct vertical speed command. You must set a target
altitude and a vertical speed together:

```python
vs_ms = action[2] * MAX_VS_MS   # convert normalized action to m/s

if vs_ms >= 0:
    bs.traf.selalt[idx] = 100000    # arbitrarily high — climb toward it
else:
    bs.traf.selalt[idx] = 0         # ground — descend toward it

bs.traf.selvs[idx] = vs_ms
```

Unit conversions:
```python
FT_PER_MIN_TO_MS = 0.00508
MS_TO_FT_PER_MIN = 196.85
M_TO_FT = 3.28084
KT_TO_MS = 0.514444
```

---

## Simulation Step Loop

```
reset()
  ├── bs.traf.reset()                    # clear all aircraft
  ├── spawn ownship via bs.traf.cre()
  ├── spawn intruders via bs.traf.cre()
  └── disable autopilot nav

step(action)
  ├── _apply_action(action)
  │     ├── bs.stack.stack('HDG ...')
  │     ├── bs.stack.stack('SPD ...')
  │     └── bs.traf.selvs[idx] = ...     # Phase 2 only
  │
  ├── for _ in range(ACTION_FREQUENCY):  # default 10
  │     bs.sim.step()                    # advance by DT seconds (default 5s)
  │     _log_trajectory()               # capture state for visualization
  │
  ├── _get_obs()     → read bs.traf.* → normalized state vector
  ├── _get_reward()  → separation check + drift + speed deviation
  └── check termination
```

### Time budget per episode

| Parameter | Value |
|---|---|
| DT (sim timestep) | 5 seconds |
| ACTION_FREQUENCY | 10 steps |
| Sim time per agent decision | 50 seconds |
| Spawn distance from FAF | 30–50 km (16–27 NM) |
| Approach speed | ~180 kt (~333 km/h) |
| Approx time to FAF | 10–17 min sim time |
| Approx agent steps per episode | 12–20 |

---

## Intruder Spawning

Intruders are spawned inbound to the same FAF as the ownship, within an approach
cone of ±45° of the runway heading. They fly straight-line paths — no agent
controls them.

```python
# spawn intruder at random distance and angle within approach cone
dist = np.random.uniform(30, 50)                        # NM from FAF
angle = np.random.uniform(-45, 45)                      # degrees off runway heading
runway_hdg = 180                                        # runway heading (example)
spawn_bearing = runway_hdg + angle + 180                # direction away from FAF

lat, lon = fn.get_point_at_distance(faf_lat, faf_lon, dist, spawn_bearing)
spd = np.random.uniform(130, 230)                       # kt

bs.traf.cre(f'INT{i}', actype='A320',
            aclat=lat, aclon=lon,
            achdg=runway_hdg + angle,
            acspd=spd)
```

---

## Trajectory Logging

At each `bs.sim.step()` call, state is captured into a list of dicts. After the
episode, this converts to DataFrames for pysda.

The PPO helpers also expose `env.decision_trace`. It contains one frame per
agent decision (the completed `env.step(action)`), including aircraft
latitude/longitude, heading, speed, action, reward, cumulative reward, and
minimum separation. Use this trace for policy comparisons and Plotly playback;
do not use the per-tick `trajectory` list when the desired animation is one
frame per action.

### Phase 1 (2D) — birdseye only

```python
def _log_trajectory(self):
    for acid in bs.traf.id:
        idx = bs.traf.id2idx(acid)
        self.trajectory.append({
            'acid':      acid,
            'latitude':  bs.traf.lat[idx],
            'longitude': bs.traf.lon[idx],
        })

def get_trajectory_dataframes(self):
    import pandas as pd
    df = pd.DataFrame(self.trajectory)
    return {acid: grp[['latitude','longitude']].reset_index(drop=True)
            for acid, grp in df.groupby('acid')}
```

Pass the result directly to pysda:
```python
trajs = env.get_trajectory_dataframes()
plot_birdseye_view(ax, data=list(trajs.values()), title='Episode Trajectory')
```

### Phase 2 (3D) — birdseye + vertical profile

```python
FIELD_ELEVATION_FT = 11     # ft — set to actual airport field elevation
M_TO_FT = 3.28084
MS_TO_FT_PER_MIN = 196.85

def _log_trajectory(self):
    ac_idx = bs.traf.id2idx('KL001')
    _, dist_nm = bs.tools.geo.kwikqdrdist(
        bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
        self.faf_lat, self.faf_lon
    )
    self.trajectory.append({
        'latitude':              bs.traf.lat[ac_idx],
        'longitude':             bs.traf.lon[ac_idx],
        'distance_to_touchdown': dist_nm,
        'destination_afe':       bs.traf.alt[ac_idx] * M_TO_FT - FIELD_ELEVATION_FT,
        'vertical_speed':        bs.traf.vs[ac_idx] * MS_TO_FT_PER_MIN,
    })
```

`distance_to_touchdown` and `destination_afe` match the column names in the pysda
`pitch_deviations_from_optimal_glidepath` SPI config exactly — no renaming needed.

---

## Glideslope Reference (Phase 2)

The ideal altitude at any distance from the FAF on a 3° glideslope:

```python
import numpy as np

def ideal_altitude_ft(dist_nm_to_faf: float,
                      faf_alt_ft: float = 1500,
                      glideslope_deg: float = 3.0) -> float:
    """
    Returns the ideal altitude (ft AFE) at a given distance from the FAF.
    Aircraft above FAF distance are climbing away from the glideslope intercept.
    """
    dist_ft = dist_nm_to_faf * 6076.12    # NM to feet
    return faf_alt_ft + dist_ft * np.tan(np.deg2rad(glideslope_deg))

# glideslope error for reward
gs_error_ft = bs.traf.alt[ac_idx] * M_TO_FT - ideal_altitude_ft(dist_nm)
```

---

## BlueSky Stack Commands Reference

| Command | Effect |
|---|---|
| `HDG {acid} {degrees}` | Set heading |
| `SPD {acid} {knots}` | Set speed |
| `ALT {acid} {ft}` | Set target altitude |
| `VS {acid} {ft/min}` | Set vertical speed |
| `DEL {acid}` | Delete aircraft |
| `DT {seconds}` | Set simulation timestep |
| `FF` | Fast-forward (run at max CPU speed) |
| `RESET` | Reset entire simulation |

---

## Known Gotchas

- `bs.traf.gs` and `bs.traf.tas` are in **m/s**, not knots. Use `* 1/KT_TO_MS` to convert.
- `bs.traf.alt` is in **meters**, not feet.
- `bs.traf.vs` is in **m/s**, not ft/min.
- Vertical speed commands require setting both `selalt` and `selvs` together — setting `selvs` alone has no effect.
- `bs.traf.swvnav` must be set to `False` per aircraft after creation, otherwise the autopilot will override agent vertical commands.
- `bs.traf.id2idx` returns `-1` if the callsign does not exist — always check before indexing.
