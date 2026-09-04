import numpy as np
import pygame

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn

import gymnasium as gym
from gymnasium import spaces

# Separation standards
INTRUSION_DISTANCE = 5      # NM — minimum horizontal separation
DISTANCE_MARGIN = 5         # km — radius to consider fix reached

# Spawn geometry
NUM_INTRUDERS = 3
NUM_WAYPOINTS = 1
APPROACH_DISTANCE_MIN = 30  # km from fix — realistic final approach range
APPROACH_DISTANCE_MAX = 50
APPROACH_CONE_DEG = 45      # aircraft spawned within +/- this angle of runway heading
INTRUDER_BEHAVIORS = ("straight", "curved")
INTRUDER_ROUTE_WAYPOINT_RADIUS_NM = 1.5
INTRUDER_CURVE_MAX_OFFSET_NM = 8.0

# Aircraft performance
AC_SPD_NOMINAL = 180        # kt — typical final approach speed
AC_SPD_MIN = 130
AC_SPD_MAX = 230
D_HEADING = 45              # max heading change per action (degrees)
D_SPEED = 50                # max speed change per action (kt)

# Reward shaping
REACH_REWARD = 1.0
DRIFT_PENALTY = -0.1
INTRUSION_PENALTY = -1.0
SPEED_DEVIATION_PENALTY = -0.05
PROGRESS_REWARD         =  0.1   # per km closed toward fix each step
PROXIMITY_PENALTY       = -0.5   # max penalty inside 2x separation zone

# Reward profiles keep the original baseline reproducible while allowing
# safety-oriented alternatives to be compared as separate experiments.
REWARD_PROFILES = {
    "baseline": {
        "intrusion_penalty": INTRUSION_PENALTY,
        "proximity_penalty": PROXIMITY_PENALTY,
    },
    "safety_shaping": {
        "intrusion_penalty": -5.0,
        "proximity_penalty": -2.0,
    },
}

ACTION_FREQUENCY = 10       # sim steps per agent action
MAX_STEPS        = 200      # max agent steps per episode before truncation
NM2KM = 1.852

# Runway fix — default to a placeholder; override in reset() options if needed
RUNWAY_FIX_LAT = 52.3105
RUNWAY_FIX_LON = 4.7683     # roughly Amsterdam Schiphol area as default

# BlueSky owns global command/plugin registries. Re-initialising it for every
# Gym environment produces harmless "Attempt to reimplement" warnings and can
# overwrite singleton state, so initialize it once per Python process.
_BLUESKY_INITIALIZED = False


def _ensure_bluesky_initialized(dt: float) -> None:
    """Initialize the process-wide BlueSky singleton and set the timestep."""
    global _BLUESKY_INITIALIZED
    if not _BLUESKY_INITIALIZED:
        bs.init(mode="sim", detached=True)
        _BLUESKY_INITIALIZED = True
    bs.scr = ScreenDummy()
    bs.stack.stack(f"DT {dt};FF")


class ApproachCREnv2D(gym.Env):
    """
    2D Horizontal Approach Conflict Resolution Environment.

    Agent controls heading and speed of ownship (KL001) inbound to a runway fix.
    Intruders default to straight-line paths toward the same fix. An optional
    curved mode gives them randomized, smooth arrival paths that still end at
    the fix for a separate robustness experiment.
    Episode ends when ownship reaches the fix or max steps exceeded.

    Extends the concept of HorizontalCREnv (bluesky-gym) with:
    - Speed control action (heading + speed vs heading only)
    - Approach-oriented spawn geometry (inbound cone vs random)
    - Speed deviation penalty to discourage erratic speed changes
    - Intruders also inbound to fix (vs random headings)
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 120}

    def __init__(self, render_mode=None, runway_lat=RUNWAY_FIX_LAT, runway_lon=RUNWAY_FIX_LON,
                 action_freq=None, dt=None, reward_profile="baseline",
                 intruder_behavior="straight",
                 spawn_distance_min=APPROACH_DISTANCE_MIN,
                 spawn_distance_max=APPROACH_DISTANCE_MAX):
        """
        Initialise the environment, observation/action spaces, BlueSky sim, and rendering state.

        Args:
            render_mode: 'human' for live pygame window, 'rgb_array' for headless, None for no render.
            runway_lat: Latitude of the runway fix (default: Schiphol area).
            runway_lon: Longitude of the runway fix.
            reward_profile: ``baseline`` or ``safety_shaping``.
            intruder_behavior: ``straight`` for the baseline or ``curved`` for
                randomized routed intruder arrivals.
            spawn_distance_min: Minimum aircraft spawn distance from the FAF in km.
            spawn_distance_max: Maximum aircraft spawn distance from the FAF in km.
        """
        if reward_profile not in REWARD_PROFILES:
            raise ValueError(
                f"Unknown reward_profile={reward_profile!r}; "
                f"choose one of {sorted(REWARD_PROFILES)}"
            )
        if intruder_behavior not in INTRUDER_BEHAVIORS:
            raise ValueError(
                f"Unknown intruder_behavior={intruder_behavior!r}; "
                f"choose one of {INTRUDER_BEHAVIORS}"
            )
        if spawn_distance_min <= 0 or spawn_distance_max < spawn_distance_min:
            raise ValueError("spawn distances must satisfy 0 < min <= max")
        self.reward_profile = reward_profile
        self.intruder_behavior = intruder_behavior
        self.spawn_distance_min = float(spawn_distance_min)
        self.spawn_distance_max = float(spawn_distance_max)
        reward_settings = REWARD_PROFILES[reward_profile]
        self._intrusion_penalty = reward_settings["intrusion_penalty"]
        self._proximity_penalty = reward_settings["proximity_penalty"]

        self.window_width = 512
        self.window_height = 512
        self.window_size = (self.window_width, self.window_height)

        self.runway_lat = runway_lat
        self.runway_lon = runway_lon

        self.observation_space = spaces.Dict({
            "intruder_distance":   spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            "cos_bearing":         spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            "sin_bearing":         spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            "x_difference_speed":  spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            "y_difference_speed":  spaces.Box(-np.inf, np.inf, shape=(NUM_INTRUDERS,), dtype=np.float64),
            "waypoint_distance":   spaces.Box(-np.inf, np.inf, shape=(NUM_WAYPOINTS,), dtype=np.float64),
            "cos_drift":           spaces.Box(-np.inf, np.inf, shape=(NUM_WAYPOINTS,), dtype=np.float64),
            "sin_drift":           spaces.Box(-np.inf, np.inf, shape=(NUM_WAYPOINTS,), dtype=np.float64),
            "ownship_speed":       spaces.Box(-np.inf, np.inf, shape=(1,),             dtype=np.float64),
        })

        # [heading_delta, speed_delta] both normalized to [-1, 1]
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        _dt = dt if dt is not None else 5
        self._action_freq = action_freq if action_freq is not None else ACTION_FREQUENCY
        self._action_freq_max_steps = MAX_STEPS
        _ensure_bluesky_initialized(_dt)

        self.total_reward = 0
        self.total_intrusions = 0
        self.minimum_separation_nm = np.inf
        self.average_drift = np.array([])

        # trajectory logging for birdseye visualization
        self.trajectory = []
        # One frame per agent decision, used by static and interactive plots.
        self.decision_trace = []
        self._intruder_routes = {}
        self._intruder_route_indices = {}

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state and return the first observation.

        Clears BlueSky traffic, resets reward/intrusion counters, spawns ownship and
        intruders, and sets the runway waypoint.

        Returns:
            observation (dict): Initial observation.
            info (dict): Auxiliary diagnostic info.
        """
        super().reset(seed=seed)

        bs.traf.reset()
        self.total_reward = 0
        self.total_intrusions = 0
        self.minimum_separation_nm = np.inf
        self.average_drift = np.array([])
        self.trajectory = []
        self.decision_trace = []
        self._intruder_routes = {}
        self._intruder_route_indices = {}
        self.step_count = 0
        self.reached_fix = False
        self._prev_dist = None   # for dense progress reward

        self._spawn_ownship()
        self._spawn_intruders()
        self._set_waypoint()

        observation = self._get_obs()
        info = self._get_info()
        self._log_decision(action=None, reward=0.0, info=info)

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """
        Advance the simulation by one agent step (ACTION_FREQUENCY sim ticks).

        Applies the heading/speed action, steps BlueSky, computes reward, and checks
        for episode termination (ownship reached fix or all waypoints reached).

        Args:
            action (np.ndarray): [heading_delta, speed_delta] each normalised to [-1, 1].

        Returns:
            observation (dict), reward (float), terminated (bool), truncated (bool), info (dict)
        """
        self._apply_action(action)

        for _ in range(self._action_freq):
            self._guide_curved_intruders()
            bs.sim.step()
            if not self.reached_fix:   # stop logging once fix is reached
                self._log_trajectory()
            if self.render_mode == "human":
                self._render_frame()

        observation = self._get_obs()
        self.step_count += 1
        reward, terminated = self._get_reward()  # sets self.reached_fix if fix reached
        truncated = self.step_count >= self._action_freq_max_steps
        info = self._get_info()
        self._log_decision(action=action, reward=reward, info=info)

        if terminated or truncated:
            for acid in bs.traf.id:
                bs.traf.delete(bs.traf.id2idx(acid))

        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Spawn helpers
    # ------------------------------------------------------------------

    def _spawn_ownship(self):
        """Spawn ownship inbound to runway fix within approach cone."""
        dist = self.np_random.uniform(self.spawn_distance_min, self.spawn_distance_max)
        # runway heading assumed 0 (north) — offset ownship behind the fix
        hdg_to_fix = 180  # ownship is south of fix, heading north
        angle_offset = self.np_random.uniform(-APPROACH_CONE_DEG, APPROACH_CONE_DEG)
        spawn_hdg = hdg_to_fix + angle_offset + 180  # direction away from fix

        lat, lon = fn.get_point_at_distance(self.runway_lat, self.runway_lon, dist, spawn_hdg)
        bs.traf.cre('KL001', actype="A320", aclat=lat, aclon=lon,
                    achdg=hdg_to_fix + angle_offset, acspd=AC_SPD_NOMINAL)

    def _spawn_intruders(self):
        """Spawn intruders inbound to the same fix, spread around the cone."""
        ac_idx = bs.traf.id2idx('KL001')
        for i in range(NUM_INTRUDERS):
            dist = self.np_random.uniform(self.spawn_distance_min, self.spawn_distance_max)
            angle_offset = self.np_random.uniform(-APPROACH_CONE_DEG, APPROACH_CONE_DEG)
            hdg_to_fix = 180
            spawn_hdg = hdg_to_fix + angle_offset + 180

            lat, lon = fn.get_point_at_distance(self.runway_lat, self.runway_lon, dist, spawn_hdg)
            spd = self.np_random.uniform(AC_SPD_MIN, AC_SPD_MAX)
            route = (
                self._build_curved_intruder_route(lat, lon)
                if self.intruder_behavior == "curved" else []
            )
            initial_heading = (
                bs.tools.geo.kwikqdrdist(lat, lon, *route[0])[0]
                if route else hdg_to_fix + angle_offset
            )
            bs.traf.cre(f'INT{i}', actype="A320", aclat=lat, aclon=lon,
                        achdg=initial_heading, acspd=spd)
            if route:
                self._intruder_routes[f"INT{i}"] = route
                self._intruder_route_indices[f"INT{i}"] = 0

    def _build_curved_intruder_route(self, start_lat, start_lon):
        """Build a randomized smooth local route ending exactly at the FAF."""
        bearing, distance = bs.tools.geo.kwikqdrdist(
            self.runway_lat, self.runway_lon, start_lat, start_lon
        )
        bearing_rad = np.deg2rad(bearing)
        start_x = distance * np.sin(bearing_rad)
        start_y = distance * np.cos(bearing_rad)
        length = max(np.hypot(start_x, start_y), 1.0)
        perpendicular = np.array((-start_y, start_x)) / length
        bend = self.np_random.uniform(-INTRUDER_CURVE_MAX_OFFSET_NM,
                                      INTRUDER_CURVE_MAX_OFFSET_NM)
        wiggle = self.np_random.uniform(-2.0, 2.0)

        route = []
        for fraction in (0.25, 0.5, 0.75, 1.0):
            base = np.array((start_x, start_y)) * (1.0 - fraction)
            offset = bend * np.sin(np.pi * fraction) + wiggle * np.sin(2 * np.pi * fraction)
            point = base + perpendicular * offset
            if fraction == 1.0:
                point = np.zeros(2)
            point_distance = np.hypot(*point)
            point_bearing = np.rad2deg(np.arctan2(point[0], point[1])) % 360
            route.append(fn.get_point_at_distance(
                self.runway_lat, self.runway_lon, point_distance, point_bearing
            ))
        return route

    def _guide_curved_intruders(self):
        """Turn curved intruders toward their next route point each sim tick."""
        if self.intruder_behavior != "curved":
            return
        for acid, route in self._intruder_routes.items():
            route_index = self._intruder_route_indices[acid]
            ac_idx = bs.traf.id2idx(acid)
            target_lat, target_lon = route[route_index]
            bearing, distance = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], target_lat, target_lon
            )
            while (distance < INTRUDER_ROUTE_WAYPOINT_RADIUS_NM
                   and route_index < len(route) - 1):
                route_index += 1
                target_lat, target_lon = route[route_index]
                bearing, distance = bs.tools.geo.kwikqdrdist(
                    bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], target_lat, target_lon
                )
            self._intruder_route_indices[acid] = route_index
            bs.stack.stack(f"HDG {acid} {bearing}")

    def _set_waypoint(self):
        """
        Set the single runway fix as the ownship's target waypoint.

        Stores lat/lon lists and a reached-flag list (0 = not reached, 1 = reached).
        """
        self.wpt_lat = [self.runway_lat]
        self.wpt_lon = [self.runway_lon]
        self.wpt_reach = [0]

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self):
        """
        Build and return the normalised observation dict for the current sim state.

        Computes relative bearing, distance, and speed differences to each intruder,
        plus bearing and distance to the runway fix, all normalised to [-1, 1] or [0, 1].

        Returns:
            dict matching self.observation_space.
        """
        ac_idx = bs.traf.id2idx('KL001')
        self.ac_hdg = bs.traf.hdg[ac_idx]
        self.ac_spd = bs.traf.gs[ac_idx]

        intruder_distance, cos_bearing, sin_bearing = [], [], []
        x_diff_spd, y_diff_spd = [], []

        for i in range(NUM_INTRUDERS):
            int_idx = bs.traf.id2idx(f'INT{i}')
            qdr, dis = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
                bs.traf.lat[int_idx], bs.traf.lon[int_idx]
            )
            intruder_distance.append(dis * NM2KM)

            bearing = fn.bound_angle_positive_negative_180(self.ac_hdg - qdr)
            cos_bearing.append(np.cos(np.deg2rad(bearing)))
            sin_bearing.append(np.sin(np.deg2rad(bearing)))

            hdg_diff = self.ac_hdg - bs.traf.hdg[int_idx]
            x_diff_spd.append(-np.cos(np.deg2rad(hdg_diff)) * bs.traf.gs[int_idx])
            y_diff_spd.append(self.ac_spd - np.sin(np.deg2rad(hdg_diff)) * bs.traf.gs[int_idx])

        waypoint_distance, cos_drift, sin_drift = [], [], []
        self.drift = []

        for lat, lon in zip(self.wpt_lat, self.wpt_lon):
            wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], lat, lon
            )
            waypoint_distance.append(wpt_dis * NM2KM)
            drift = fn.bound_angle_positive_negative_180(self.ac_hdg - wpt_qdr)
            self.drift.append(drift)
            cos_drift.append(np.cos(np.deg2rad(drift)))
            sin_drift.append(np.sin(np.deg2rad(drift)))

        self.intruder_distance = intruder_distance

        return {
            "intruder_distance":  np.array(intruder_distance) / (APPROACH_DISTANCE_MAX * NM2KM),
            "cos_bearing":        np.array(cos_bearing),
            "sin_bearing":        np.array(sin_bearing),
            "x_difference_speed": np.array(x_diff_spd) / AC_SPD_MAX,
            "y_difference_speed": np.array(y_diff_spd) / AC_SPD_MAX,
            "waypoint_distance":  np.array(waypoint_distance) / (APPROACH_DISTANCE_MAX * NM2KM),
            "cos_drift":          np.array(cos_drift),
            "sin_drift":          np.array(sin_drift),
            "ownship_speed":      np.array([self.ac_spd / AC_SPD_MAX]),
        }

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _get_reward(self):
        """
        Compute the total step reward and check for episode termination.

        Combines waypoint reach reward, drift penalty, intrusion penalty, and
        speed deviation penalty. Sets terminated=True when all waypoints are reached.

        Returns:
            total_reward (float), terminated (bool)
        """
        reach_reward = self._check_waypoint()
        drift_reward = self._check_drift()
        intrusion_reward = self._check_intrusion()
        speed_reward = self._check_speed()
        progress_reward = self._check_progress()
        proximity_reward = self._check_proximity()

        total_reward = (reach_reward + drift_reward + intrusion_reward
                        + speed_reward + progress_reward + proximity_reward)
        self.total_reward += total_reward

        terminated = all(r == 1 for r in self.wpt_reach)
        if terminated:
            self.reached_fix = True
        return total_reward, terminated

    def _check_waypoint(self):
        """
        Check whether ownship has entered the DISTANCE_MARGIN radius of each waypoint.

        Awards REACH_REWARD once per waypoint on first entry and marks it as reached.

        Returns:
            reward (float): Sum of reach rewards earned this step.
        """
        reward = 0
        for i, dist in enumerate(self.intruder_distance):
            # reuse waypoint_distance from last obs
            pass
        ac_idx = bs.traf.id2idx('KL001')
        for i, (lat, lon) in enumerate(zip(self.wpt_lat, self.wpt_lon)):
            _, dis = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], lat, lon
            )
            if dis * NM2KM < DISTANCE_MARGIN and self.wpt_reach[i] != 1:
                self.wpt_reach[i] = 1
                reward += REACH_REWARD
        return reward

    def _check_drift(self):
        """
        Penalise angular drift between ownship heading and the bearing to the runway fix.

        Appends the current drift magnitude to the running average and returns a
        scaled negative reward proportional to the drift angle.

        Returns:
            reward (float): Negative drift penalty for this step.
        """
        drift = abs(np.deg2rad(self.drift[0]))
        self.average_drift = np.append(self.average_drift, drift)
        return drift * DRIFT_PENALTY

    def _check_intrusion(self):
        """
        Check for separation violations between ownship and each intruder.

        Increments total_intrusions and applies INTRUSION_PENALTY for every
        intruder closer than INTRUSION_DISTANCE NM.

        Returns:
            reward (float): Sum of intrusion penalties earned this step.
        """
        ac_idx = bs.traf.id2idx('KL001')
        reward = 0
        for i in range(NUM_INTRUDERS):
            int_idx = bs.traf.id2idx(f'INT{i}')
            _, dis = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
                bs.traf.lat[int_idx], bs.traf.lon[int_idx]
            )
            self.minimum_separation_nm = min(self.minimum_separation_nm, float(dis))
            if dis < INTRUSION_DISTANCE:
                self.total_intrusions += 1
                reward += self._intrusion_penalty
        return reward

    def _check_speed(self):
        """
        Penalise deviation of ownship speed from AC_SPD_NOMINAL.

        Returns:
            reward (float): Negative penalty proportional to normalised speed deviation.
        """
        deviation = abs(self.ac_spd - AC_SPD_NOMINAL) / AC_SPD_MAX
        return deviation * SPEED_DEVIATION_PENALTY

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def _check_progress(self):
        """Dense reward: +PROGRESS_REWARD per km closed toward fix this step."""
        ac_idx = bs.traf.id2idx('KL001')
        _, dis = bs.tools.geo.kwikqdrdist(
            bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
            self.wpt_lat[0], self.wpt_lon[0]
        )
        curr_dist = dis * NM2KM
        if self._prev_dist is None:
            self._prev_dist = curr_dist
            return 0.0
        delta = self._prev_dist - curr_dist
        self._prev_dist = curr_dist
        return delta * PROGRESS_REWARD

    def _check_proximity(self):
        """Soft ramp penalty inside 2x separation distance — same as numpy env."""
        ac_idx = bs.traf.id2idx('KL001')
        reward = 0.0
        warn_nm = INTRUSION_DISTANCE * 2
        for i in range(NUM_INTRUDERS):
            int_idx = bs.traf.id2idx(f'INT{i}')
            _, dis = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
                bs.traf.lat[int_idx], bs.traf.lon[int_idx]
            )
            if dis < warn_nm:
                reward += self._proximity_penalty * (1.0 - dis / warn_nm)
        return reward

    def _apply_action(self, action):
        """
        Translate normalised agent action into BlueSky HDG and SPD commands.

        Heading delta is applied relative to current heading; speed is clipped to
        [AC_SPD_MIN, AC_SPD_MAX] before issuing the command.

        Args:
            action (np.ndarray): [heading_delta, speed_delta] each in [-1, 1].
        """
        # read current state fresh from BlueSky — do NOT use self.ac_hdg/ac_spd
        # because _apply_action is called before _get_obs() updates those caches
        ac_idx = bs.traf.id2idx('KL001')
        cur_hdg = bs.traf.hdg[ac_idx]
        cur_spd = bs.traf.gs[ac_idx]
        new_hdg = cur_hdg + action[0] * D_HEADING
        new_spd = np.clip(cur_spd + action[1] * D_SPEED, AC_SPD_MIN, AC_SPD_MAX)
        bs.stack.stack(f"HDG KL001 {new_hdg}")
        bs.stack.stack(f"SPD KL001 {new_spd}")

    # ------------------------------------------------------------------
    # Trajectory logging (for birdseye visualization)
    # ------------------------------------------------------------------

    def _log_trajectory(self):
        """Log lat/lon of all aircraft at current sim step."""
        step = {}
        for acid in bs.traf.id:
            idx = bs.traf.id2idx(acid)
            step[acid] = {
                "latitude": bs.traf.lat[idx],
                "longitude": bs.traf.lon[idx],
            }
        self.trajectory.append(step)

    def _log_decision(self, action, reward, info):
        """Log aircraft state once per agent decision for research plots."""
        frame = {
            "decision_step": int(self.step_count),
            "reward": float(reward),
            "cumulative_reward": float(info["total_reward"]),
            "minimum_separation_nm": info["minimum_separation_nm"],
            "total_intrusions": int(info["total_intrusions"]),
            "sim_time_s": float(getattr(bs.sim, "simt", np.nan)),
            "action": None if action is None else np.asarray(action, dtype=float).tolist(),
        }
        for acid in bs.traf.id:
            idx = bs.traf.id2idx(acid)
            frame[acid] = {
                "latitude": float(bs.traf.lat[idx]),
                "longitude": float(bs.traf.lon[idx]),
                "heading": float(bs.traf.hdg[idx]),
                "speed": float(bs.traf.gs[idx]),
            }
        self.decision_trace.append(frame)

    def get_trajectory_dataframes(self):
        """
        Convert logged trajectory to a dict of DataFrames keyed by aircraft ID.
        Each DataFrame has columns [latitude, longitude] — compatible with
        pysda plot_birdseye_view.

        Returns:
            dict[str, pd.DataFrame]
        """
        import pandas as pd
        if not self.trajectory:
            return {}
        acids = self.trajectory[0].keys()
        return {
            acid: pd.DataFrame([step[acid] for step in self.trajectory])
            for acid in acids
        }

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def _get_info(self):
        """
        Return auxiliary diagnostic info for the current episode state.

        Returns:
            dict with keys: total_reward, total_intrusions, average_drift.
        """
        return {
            "total_reward": self.total_reward,
            "total_intrusions": self.total_intrusions,
            "reward_profile": self.reward_profile,
            "minimum_separation_nm": (
                None if np.isinf(self.minimum_separation_nm) else self.minimum_separation_nm
            ),
            "average_drift": self.average_drift.mean() if len(self.average_drift) else 0.0,
            "reached_fix": self.reached_fix,
            "step": self.step_count,
        }

    # ------------------------------------------------------------------
    # Rendering (pygame — ownship centered, same style as HorizontalCREnv)
    # ------------------------------------------------------------------

    def _render_frame(self):
        """
        Render the current sim state to a pygame window (human mode).

        Draws ownship centred on screen, intruders with separation rings coloured
        red on intrusion, and the runway fix as a white circle with reach-margin ring.
        """
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        max_distance = 200  # km — display radius
        canvas = pygame.Surface(self.window_size)
        canvas.fill((135, 206, 235))

        ac_idx = bs.traf.id2idx('KL001')
        self._draw_aircraft(canvas, ac_idx, max_distance, size=8, color=(0, 0, 0))

        for i in range(NUM_INTRUDERS):
            int_idx = bs.traf.id2idx(f'INT{i}')
            qdr, dis = bs.tools.geo.kwikqdrdist(
                bs.traf.lat[ac_idx], bs.traf.lon[ac_idx],
                bs.traf.lat[int_idx], bs.traf.lon[int_idx]
            )
            color = (220, 20, 60) if dis < INTRUSION_DISTANCE else (80, 80, 80)
            x_pos = self.window_width / 2 + (np.cos(np.deg2rad(qdr)) * dis * NM2KM / max_distance) * self.window_width
            y_pos = self.window_height / 2 - (np.sin(np.deg2rad(qdr)) * dis * NM2KM / max_distance) * self.window_height
            self._draw_aircraft(canvas, int_idx, max_distance, size=3, color=color,
                                 x_pos=x_pos, y_pos=y_pos)
            pygame.draw.circle(canvas, color, (int(x_pos), int(y_pos)),
                                int((INTRUSION_DISTANCE * NM2KM / max_distance) * self.window_width), 2)

        # draw runway fix
        for qdr, dis in zip(
            [bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], lat, lon)[0]
             for lat, lon in zip(self.wpt_lat, self.wpt_lon)],
            [bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], lat, lon)[1]
             for lat, lon in zip(self.wpt_lat, self.wpt_lon)]
        ):
            cx = self.window_width / 2 + (np.cos(np.deg2rad(qdr)) * dis * NM2KM / max_distance) * self.window_width
            cy = self.window_height / 2 - (np.sin(np.deg2rad(qdr)) * dis * NM2KM / max_distance) * self.window_height
            pygame.draw.circle(canvas, (255, 255, 255), (int(cx), int(cy)), 5, 0)
            pygame.draw.circle(canvas, (255, 255, 255), (int(cx), int(cy)),
                                int((DISTANCE_MARGIN / max_distance) * self.window_width), 2)

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def _draw_aircraft(self, canvas, idx, max_distance, size, color, x_pos=None, y_pos=None):
        """
        Draw a single aircraft as a heading line on the pygame canvas.

        Args:
            canvas: pygame.Surface to draw on.
            idx: BlueSky traffic index of the aircraft.
            max_distance: Display radius in km used for coordinate scaling.
            size: Length of the aircraft body line in display units.
            color: RGB tuple for the aircraft colour.
            x_pos: Screen x position (defaults to window centre for ownship).
            y_pos: Screen y position (defaults to window centre for ownship).
        """
        hdg = bs.traf.hdg[idx]
        end_x = (np.cos(np.deg2rad(hdg)) * size / max_distance) * self.window_width
        end_y = (np.sin(np.deg2rad(hdg)) * size / max_distance) * self.window_width
        cx = x_pos if x_pos is not None else self.window_width / 2
        cy = y_pos if y_pos is not None else self.window_height / 2
        pygame.draw.line(canvas, color,
                         (cx - end_x / 2, cy + end_y / 2),
                         (cx + end_x / 2, cy - end_y / 2), width=4)
        hdg_len = 50 if x_pos is None else 10
        hx = (np.cos(np.deg2rad(hdg)) * hdg_len / max_distance) * self.window_width
        hy = (np.sin(np.deg2rad(hdg)) * hdg_len / max_distance) * self.window_width
        pygame.draw.line(canvas, color, (cx, cy), (cx + hx, cy - hy), width=1)

    def close(self):
        """Clean up environment resources (pygame window if open)."""
        pass
