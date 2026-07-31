# Approach Conflict Resolution — RL Project

## Overview

Train an RL agent to resolve conflicts between aircraft on approach to a runway.
The agent controls heading and speed to maintain separation while navigating toward
a runway fix. Visualization uses pysda birdseye (2D) and side profile (3D) plots
to compare agent behavior before and after training.

Two algorithms are explored across two phases:
- **Phase 1 (2D):** Both DQN (discrete) and PPO (continuous) are run on the same
  environment to get a direct apples-to-apples comparison. This answers whether
  continuous action precision is worth the added complexity before scaling up.
- **Phase 2 (3D):** PPO only, using the continuous action space. Adding a vertical
  dimension makes discrete bins impractical (5×5×5 = 125 actions), and Phase 1
  results justify the algorithm choice.

The narrative across phases: use Phase 1 to compare algorithms on equal footing,
pick the better performer, then extend it to 3D in Phase 2.

---

## Key Definitions

### Final Approach Fix (FAF)
The FAF is a defined geographic point (lat/lon) that marks where an aircraft
officially begins the final approach segment — the last "gate" before the runway.

In real operations it is defined by:
- A specific **lat/lon** published on approach charts
- A crossing **altitude** (e.g. 1500 ft)
- A distance from the runway threshold — typically **5–10 NM out**

```
                    FAF (e.g. 5NM, 1500ft)
                     |
                     |  3° glideslope
                     |
[30NM out]----------[FAF]----------[Runway threshold]
```

Aircraft cross the FAF at a defined altitude and from that point follow the
glideslope down to the runway. In real ops, once established past the FAF the
tower takes over and the approach is considered committed.

**In this project:**
- Phase 1 (2D): the FAF is a lat/lon target — episode ends when ownship reaches it
  in the horizontal plane. Altitude is not modeled.
- Phase 2 (3D): the FAF becomes a full 3D gate — ownship must cross it at the
  correct altitude on the correct heading, at which point the glideslope constraint
  activates.

---

## Assumptions

### What is included
- Multiple aircraft inbound to a single Final Approach Fix (FAF)
- Horizontal separation enforcement (minimum 5 NM)
- Agent controls heading and speed
- Realistic initial conditions seeded from QAR data (speeds, approach angles)
- Episodes end when ownship reaches the FAF

### What is NOT included and why

| Excluded | Why excluded |
|---|---|
| **Runway sequencing** | Sequencing decides the *order* aircraft land. That is a scheduling problem on top of conflict resolution. Including it would require the agent to reason about which aircraft goes first, manage holding patterns, and coordinate across multiple aircraft simultaneously. Out of scope for this project. |
| **Wake turbulence categories** | Directly tied to sequencing — separation minima differ by Heavy/Medium/Light. Without sequencing, a fixed separation standard is used. |
| **Go-around / missed approach** | Adds a recovery state that significantly complicates the episode structure. Assumed all aircraft complete the approach. |
| **Wind / weather** | BlueSky supports wind but adds stochasticity that complicates early training. Calm conditions assumed. |
| **Comms / ATC instructions** | Agent acts directly on aircraft state. No communication delay or instruction acknowledgement modeled. |
| **Terrain / obstacles** | Flat earth assumption. No terrain avoidance. |

### Simplifications
- All aircraft at the same altitude in 2D phase
- Fixed approach speed profile (not flap-schedule driven)
- Single runway, single FAF
- Ownship is the only controlled aircraft; intruders follow straight-line paths

---

## Phase 1 — 2D Horizontal Approach CR (DQN vs PPO)

### Goal
Agent resolves conflicts in the horizontal plane while navigating inbound aircraft
to the FAF. Both DQN and PPO are trained on the same environment and compared.

### Why both algorithms here
- DQN is a natural progression from Q-learning covered in the course, easier to
  interpret, and mirrors how real ATC instructions work (discrete commands)
- PPO handles continuous actions natively, giving finer heading/speed control
- Running both on the same env gives a direct comparison — same problem, same reward,
  different action representations
- The result informs the algorithm choice for Phase 2

### State space (shared by both algorithms)
| Variable | Description |
|---|---|
| `intruder_distance` | Distance to each intruder (NM) |
| `cos_bearing`, `sin_bearing` | Relative bearing to each intruder |
| `x_diff_speed`, `y_diff_speed` | Relative velocity components |
| `waypoint_distance` | Distance to FAF |
| `cos_drift`, `sin_drift` | Heading drift from FAF |
| `ownship_speed` | Current ownship speed (normalized) |

### Action space — DQN (discrete)
Heading and speed changes are binned into fixed increments. Agent picks one of
25 discrete actions per step:

| Dimension | Bins | Values |
|---|---|---|
| Heading change | 5 | -45°, -22.5°, 0°, +22.5°, +45° |
| Speed change | 5 | -50kt, -25kt, 0kt, +25kt, +50kt |
| Total actions | 25 | all heading × speed combinations |

### Action space — PPO (continuous)
| Action | Range | Description |
|---|---|---|
| `d_heading` | [-1, 1] → [-45°, +45°] | Heading change relative to current |
| `d_speed` | [-1, 1] → [-50kt, +50kt] | Speed change relative to current |

### Reward (shared by both algorithms)
| Component | Value | Trigger |
|---|---|---|
| Reach fix | +1 | Ownship within 5km of FAF |
| Drift penalty | -0.1 × drift | Each step, proportional to heading error |
| Intrusion penalty | -1 | Each step inside 5NM of any intruder |
| Speed deviation penalty | -0.05 × \|dv\| | Penalize large speed deviations from nominal |

### Episode
- Reset: ownship and intruders spawned inbound to FAF, seeded from QAR speed/angle distributions
- Terminate: ownship reaches FAF OR max steps exceeded
- Algorithms: DQN and PPO via Stable Baselines3, trained separately on same env
- Comparison metrics: total intrusions, average drift, reward per episode
- Visualization: pysda birdseye — DQN vs PPO trajectories side by side

---

## Phase 2 — 3D Approach CR with PPO (extends Phase 1)

### Goal
Agent also manages vertical profile — stay on or near a 3° glideslope while
maintaining horizontal separation, using PPO with a continuous action space.

### Why PPO only here
- Phase 1 comparison informs this choice — PPO is carried forward as the better
  or more scalable algorithm
- Adding a vertical dimension makes discrete bins impractical: 5×5×5 = 125 actions
- Continuous actions give the agent finer control over vertical rate, which matters
  for staying on a precise glideslope
- PPO is the standard algorithm in bluesky-gym examples and published CR research

### Additional state (on top of Phase 1)
| Variable | Description |
|---|---|
| `altitude` | Current altitude (ft) |
| `vertical_rate` | Current vertical rate (ft/min) |
| `glideslope_error` | Deviation from 3° glideslope (ft) |
| `intruder_altitude_diff` | Altitude difference to each intruder |

### Action space (continuous)
| Action | Range | Description |
|---|---|---|
| `d_heading` | [-1, 1] → [-45°, +45°] | Heading change relative to current |
| `d_speed` | [-1, 1] → [-50kt, +50kt] | Speed change relative to current |
| `d_vertical_rate` | [-1, 1] → [-1000, +1000] ft/min | Vertical rate change |

### Additional reward (on top of Phase 1)
| Component | Value | Trigger |
|---|---|---|
| Glideslope deviation penalty | -0.1 × \|gs_error\| | Each step off glideslope |
| Vertical intrusion penalty | -1 | Each step within 1000ft vertical of intruder |

### Episode
- Same as Phase 1 but aircraft initialized at altitude with descent profile
- Terminate: ownship reaches FAF at correct altitude band OR max steps exceeded
- Algorithm: PPO via Stable Baselines3 (`stable_baselines3.PPO`)
- Visualization:
  - pysda birdseye (horizontal plane — lat/lon trajectories)
  - pysda vertical profile (altitude AFE vs distance to threshold, with 3° glideslope brackets)
    - x-axis: distance to threshold (NM), inverted (6NM → 0NM)
    - y-axis: altitude above field elevation (ft)
    - 3° glideslope reference line with ±0.5° deviation brackets
    - color-coded by agent action state if needed (e.g. heading change active vs nominal)
    - configured via pysda SPI yaml — see `pitch_deviations_from_optimal_glidepath` as reference config

---

## Simulation Step

Each call to `env.step(action)` covers one agent decision cycle. Internally it runs
multiple BlueSky sim steps before returning the next observation. See
`SIMULATION.md` for full detail on how the simulation is structured.

At a high level:

```
reset()
  └── spawn ownship + intruders at approach geometry
  └── disable autopilot nav (agent has full control)

step(action)
  └── apply action  →  HDG / SPD / VS commands via bs.stack.stack()
  └── for _ in range(ACTION_FREQUENCY):   # 10 × DT=5s = 50s sim time per decision
        bs.sim.step()                      # advance physics
        log lat / lon / alt / spd          # capture trajectory
  └── read bs.traf.*  →  build observation
  └── compute reward
  └── check termination  →  reached FAF or max steps
```

Time per episode (approximate):
- Spawn distance: 30–50 NM from FAF
- Speed: ~180 kt
- Time to FAF: ~10–17 min sim time
- At 50s per decision: ~12–20 agent steps per episode

---

## Algorithm Summary

| | Phase 1 — DQN | Phase 1 — PPO | Phase 2 — PPO |
|---|---|---|---|
| Action space | Discrete (25 bins) | Continuous (2D vector) | Continuous (3D vector) |
| Dimensions | 2D horizontal | 2D horizontal | 3D horizontal + vertical |
| Library | `stable_baselines3.DQN` | `stable_baselines3.PPO` | `stable_baselines3.PPO` |
| Policy | `MultiInputPolicy` | `MultiInputPolicy` | `MultiInputPolicy` |
| Purpose | Baseline, interpretable | Compare vs DQN | Extend winner to 3D |
| Visualization | Birdseye | Birdseye | Birdseye + vertical profile |

---

## Project Structure

```
Project/
├── README.md               # This file
├── SIMULATION.md           # Detailed simulation design and BlueSky reference
├── envs/
│   ├── approach_cr_2d.py   # Phase 1 environment (discrete actions for DQN)
│   └── approach_cr_3d.py   # Phase 2 environment (continuous actions for PPO)
├── train_dqn.py            # Phase 1 training — DQN (discrete)
├── train_ppo.py            # Phase 1 + 2 training — PPO (continuous)
└── visualize.py            # Trajectory logging + pysda plotting
```

---

## References
- [BlueSky-Gym](https://github.com/TUDelft-CNS-ATM/bluesky-gym)
- [BlueSky ATC Simulator](https://github.com/TUDelft-CNS-ATM/bluesky)
- pysda `plot_birdseye_view` — see `pysda/qar/plots.py`
- pysda vertical profile — see `pitch_deviations_from_optimal_glidepath` SPI config
  - x: `distance_to_touchdown - ground_distance_runway_threshold_to_touchdown * 0.000164579` (NM)
  - y: `destination_afe` (ft above field elevation)
  - glideslope brackets: 3° ± 0.5° overlaid via `plot_slope_brackets`
