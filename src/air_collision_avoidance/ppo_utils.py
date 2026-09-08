"""Shared PPO training, evaluation, metrics, and plotting helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from air_collision_avoidance.envs.approach_cr_2d import (
    APPROACH_DISTANCE_MAX,
    APPROACH_DISTANCE_MIN,
    INTRUSION_DISTANCE,
    ApproachCREnv2D,
)


def make_env(action_freq: int = 1, dt: float = 10.0,
             reward_profile: str = "baseline",
             intruder_behavior: str = "straight",
             spawn_distance_min: float = APPROACH_DISTANCE_MIN,
             spawn_distance_max: float = APPROACH_DISTANCE_MAX) -> Monitor:
    """Create one monitored BlueSky environment.

    BlueSky is a singleton simulator, so experiments deliberately use one
    environment at a time. ``action_freq=1`` and ``dt=10`` are the fast
    exploration settings used by the reference notebook; use ``10`` and ``5``
    for the nominal simulation settings.
    """

    return Monitor(ApproachCREnv2D(
        action_freq=action_freq, dt=dt, reward_profile=reward_profile,
        intruder_behavior=intruder_behavior,
        spawn_distance_min=spawn_distance_min,
        spawn_distance_max=spawn_distance_max,
    ))


def make_ppo_model(env: Monitor, seed: int = 42, verbose: int = 0, **overrides: Any) -> PPO:
    """Build the shared PPO configuration with optional experiment overrides."""

    settings: dict[str, Any] = {
        "learning_rate": 3e-4,
        "n_steps": 128,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "seed": seed,
        "verbose": verbose,
    }
    settings.update(overrides)
    return PPO("MultiInputPolicy", env, **settings)


class PPOEpisodeCallback(BaseCallback):
    """Collect episode-level metrics while Stable-Baselines3 trains PPO."""

    def __init__(self) -> None:
        super().__init__()
        self.episodes: list[dict[str, Any]] = []
        self._episode_reward = 0.0

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals["rewards"]).reshape(-1)
        dones = np.asarray(self.locals["dones"]).reshape(-1)
        infos = self.locals["infos"]

        for index, reward in enumerate(rewards):
            self._episode_reward += float(reward)
            if not dones[index]:
                continue

            info = infos[index]
            self.episodes.append(
                {
                    "episode": len(self.episodes) + 1,
                    "training_timesteps": int(self.num_timesteps),
                    "reward": self._episode_reward,
                    "reached_fix": bool(info.get("reached_fix", False)),
                    "total_intrusions": int(info.get("total_intrusions", 0)),
                    "minimum_separation_nm": float(info.get("minimum_separation_nm", np.nan)),
                    "average_drift": float(info.get("average_drift", 0.0)),
                    "steps": int(info.get("step", 0)),
                }
            )
            self._episode_reward = 0.0

        return True


def run_episode(
    model: Any | None,
    seed: int,
    action_freq: int = 1,
    dt: float = 10.0,
    reward_profile: str = "baseline",
    intruder_behavior: str = "straight",
    spawn_distance_min: float = APPROACH_DISTANCE_MIN,
    spawn_distance_max: float = APPROACH_DISTANCE_MAX,
    dqn_action_map: list[tuple[float, float]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one episode using PPO, DQN, straight-line, or a seeded random policy.

    Pass ``dqn_action_map`` when evaluating a DQN model. The model predicts a
    discrete action index which is mapped to a continuous [hdg, spd] pair before
    being passed to the environment, matching the training wrapper exactly.

    Returns episode metrics and the raw trajectory captured by the environment.
    Episodes are created and closed one at a time because BlueSky is singleton.
    """

    env = ApproachCREnv2D(
        action_freq=action_freq, dt=dt, reward_profile=reward_profile,
        intruder_behavior=intruder_behavior,
        spawn_distance_min=spawn_distance_min,
        spawn_distance_max=spawn_distance_max,
    )
    if model is None:
        env.action_space.seed(seed)

    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    terminated = truncated = False
    info: dict[str, Any] = {}

    while not (terminated or truncated):
        if model is None:
            action = env.action_space.sample()
        elif model == "straight":
            action = np.zeros(2, dtype=np.float64)
        elif dqn_action_map is not None:
            action_idx, _ = model.predict(observation, deterministic=True)
            action = np.array(dqn_action_map[int(action_idx)], dtype=np.float64)
        else:
            action, _ = model.predict(observation, deterministic=True)

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

    metrics = {
        "seed": seed,
        "reward": total_reward,
        "reached_fix": bool(info.get("reached_fix", False)),
        "total_intrusions": int(info.get("total_intrusions", 0)),
        "minimum_separation_nm": float(info.get("minimum_separation_nm", np.nan)),
        "average_drift": float(info.get("average_drift", 0.0)),
        "steps": int(info.get("step", 0)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "reward_profile": reward_profile,
    }
    trajectory = env.decision_trace
    env.close()
    return metrics, trajectory


def evaluate_policy(
    model: Any | None,
    seeds: list[int],
    action_freq: int = 1,
    dt: float = 10.0,
    reward_profile: str = "baseline",
    intruder_behavior: str = "straight",
    spawn_distance_min: float = APPROACH_DISTANCE_MIN,
    spawn_distance_max: float = APPROACH_DISTANCE_MAX,
) -> list[dict[str, Any]]:
    """Evaluate a policy on a fixed list of episode seeds."""

    return [
        run_episode(
            model,
            seed=seed,
            action_freq=action_freq,
            dt=dt,
            reward_profile=reward_profile,
            intruder_behavior=intruder_behavior,
            spawn_distance_min=spawn_distance_min,
            spawn_distance_max=spawn_distance_max,
        )[0]
        for seed in seeds
    ]


def summarize_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return aggregate research metrics for a list of episodes."""

    if not records:
        return {"episodes": 0}

    rewards = np.array([record["reward"] for record in records], dtype=float)
    intrusions = np.array([record["total_intrusions"] for record in records], dtype=float)
    separations = np.array([record["minimum_separation_nm"] for record in records], dtype=float)
    drifts = np.array([record["average_drift"] for record in records], dtype=float)
    steps = np.array([record["steps"] for record in records], dtype=float)

    return {
        "episodes": len(records),
        "success_rate": float(np.mean([record["reached_fix"] for record in records])),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_intrusions": float(np.mean(intrusions)),
        "mean_minimum_separation_nm": float(np.nanmean(separations)),
        "unsafe_episode_rate": float(np.mean(separations < 5.0)),
        "mean_average_drift": float(np.mean(drifts)),
        "mean_steps": float(np.mean(steps)),
    }


def save_metrics(records: list[dict[str, Any]], summary: dict[str, Any], directory: Path, name: str) -> None:
    """Save episode rows as CSV and aggregate metrics as JSON."""

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{name}.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys() if records else ["episodes"])
        writer.writeheader()
        writer.writerows(records)

    with (directory / f"{name}_summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")


def plot_training_progress(
    records: list[dict[str, Any]], output_path: Path, title: str = "PPO training progress",
) -> None:
    """Save episode-level learning and safety metrics over training timesteps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not records:
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.text(
            0.5, 0.5,
            "No complete episodes were observed in this training budget",
            ha="center", va="center", fontsize=12,
        )
        axis.set_axis_off()
        figure.suptitle(title, fontsize=14)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160)
        plt.close(figure)
        return

    x = np.asarray([
        record.get("training_timesteps", record.get("episode", index + 1))
        for index, record in enumerate(records)
    ], dtype=float)
    rewards = np.asarray([record["reward"] for record in records], dtype=float)
    success = np.asarray([record["reached_fix"] for record in records], dtype=float)
    intrusions = np.asarray([record["total_intrusions"] for record in records], dtype=float)
    separation = np.asarray([
        record["minimum_separation_nm"] for record in records
    ], dtype=float)

    window = max(1, min(25, len(records)))
    kernel = np.ones(window) / window
    rolling_reward = np.convolve(rewards, kernel, mode="valid")
    rolling_success = np.convolve(success, kernel, mode="valid")
    rolling_intrusions = np.convolve(intrusions, kernel, mode="valid")
    rolling_x = x[window - 1:]

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    figure.suptitle(title, fontsize=14)
    axes[0, 0].plot(x, rewards, color="#9ecae1", alpha=0.6, label="episode")
    axes[0, 0].plot(rolling_x, rolling_reward, color="#1769aa", linewidth=2, label=f"rolling {window}")
    axes[0, 0].set_title("Reward")
    axes[0, 0].set_ylabel("Episode reward")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(rolling_x, rolling_success, color="#2ca02c", linewidth=2)
    axes[0, 1].set_title("Fix success")
    axes[0, 1].set_ylabel("Rolling success rate")
    axes[0, 1].set_ylim(-0.05, 1.05)

    axes[1, 0].plot(x, separation, color="#d62728", alpha=0.45, label="episode")
    axes[1, 0].axhline(INTRUSION_DISTANCE, color="black", linestyle="--", label="5 NM standard")
    axes[1, 0].set_title("Minimum separation")
    axes[1, 0].set_xlabel("Training timesteps")
    axes[1, 0].set_ylabel("Minimum separation (NM)")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(x, intrusions, color="#9467bd", alpha=0.4, label="episode")
    axes[1, 1].plot(rolling_x, rolling_intrusions, color="#7b3fb6", linewidth=2, label=f"rolling {window}")
    axes[1, 1].set_title("Intrusions")
    axes[1, 1].set_xlabel("Training timesteps")
    axes[1, 1].set_ylabel("Intrusions per episode")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def latlon_to_xy_km(lat: float, lon: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    """Convert local latitude/longitude to a flat-earth kilometre offset."""

    x = (lon - ref_lon) * np.cos(np.deg2rad(ref_lat)) * 111.32
    y = (lat - ref_lat) * 111.32
    return float(x), float(y)


AIRCRAFT_COLORS = {
    "KL001": "#1769aa",
    "INT0": "#d62728",
    "INT1": "#e67e22",
    "INT2": "#7b3fb6",
}
PLANE_GLYPH = "✈"
FAF_RING_KM = 5.0 * 1.852
SEP_RING_KM = 5.0 * 1.852   # 5 NM separation radius in km


def _plane_text_angle(heading: float) -> float:
    """Convert BlueSky heading to the forward-facing plane glyph rotation."""
    return float(270.0 - heading)


def _path_heading(points: list[tuple[float, float]], fallback: float) -> float:
    """Estimate aircraft heading from the latest movement segment."""
    if len(points) < 2:
        return fallback
    previous_x, previous_y = points[-2]
    current_x, current_y = points[-1]
    delta_x = current_x - previous_x
    delta_y = current_y - previous_y
    if np.isclose(delta_x, 0.0) and np.isclose(delta_y, 0.0):
        return fallback
    return float(np.rad2deg(np.arctan2(delta_x, delta_y)) % 360)


def _trace_plane_heading(
    trace: list[dict[str, Any]], acid: str, frame_index: int,
    reference_lat: float, reference_lon: float,
) -> float:
    """Get a display heading from path movement, with a first-frame fallback."""
    end = min(frame_index, len(trace) - 1)
    points = _trace_points(trace, acid, reference_lat, reference_lon, end=end)
    fallback = float(trace[end][acid]["heading"])
    if len(points) >= 2:
        return _path_heading(points, fallback)
    if end < len(trace) - 1:
        next_points = _trace_points(
            trace, acid, reference_lat, reference_lon, end=end + 1
        )
        if len(next_points) >= 2:
            return _path_heading(next_points, fallback)
    return fallback


def _trace_points(
    trace: list[dict[str, Any]],
    acid: str,
    reference_lat: float,
    reference_lon: float,
    end: int | None = None,
) -> list[tuple[float, float]]:
    """Return local x/y points for one aircraft from a decision trace."""
    frames = trace if end is None else trace[: end + 1]
    return [
        latlon_to_xy_km(
            frame[acid]["latitude"], frame[acid]["longitude"], reference_lat, reference_lon
        )
        for frame in frames
        if acid in frame
    ]


def _metric_title(label: str, metrics: dict[str, Any]) -> str:
    minimum = metrics.get("minimum_separation_nm")
    minimum_text = "n/a" if minimum is None or not np.isfinite(minimum) else f"{minimum:.1f} NM"
    return (
        f"{label}  |  fix={bool(metrics.get('reached_fix'))}  "
        f"| intrusions={metrics.get('total_intrusions', 0)}  "
        f"| min sep={minimum_text}"
    )


def _set_equal_limits(axes: Any, traces: list[list[dict[str, Any]]],
                      reference_lat: float, reference_lon: float) -> None:
    """Set equal, shared local bounds around all plotted aircraft and the FAF."""
    coordinates = [(0.0, 0.0)]
    for trace in traces:
        for acid in AIRCRAFT_COLORS:
            coordinates.extend(_trace_points(trace, acid, reference_lat, reference_lon))
    values = np.asarray(coordinates, dtype=float)
    x_min, y_min = values.min(axis=0)
    x_max, y_max = values.max(axis=0)
    span = max(x_max - x_min, y_max - y_min, FAF_RING_KM * 2)
    pad = max(span * 0.08, FAF_RING_KM * 0.35)
    x_mid, y_mid = (x_min + x_max) / 2, (y_min + y_max) / 2
    half = span / 2 + pad
    for axis in axes:
        axis.set_xlim(x_mid - half, x_mid + half)
        axis.set_ylim(y_mid - half, y_mid + half)
        axis.set_aspect("equal", adjustable="box")


def _set_ownship_focus_limits(axes: Any, traces: list[list[dict[str, Any]]],
                              reference_lat: float, reference_lon: float) -> None:
    """Frame plots around the ownship path and final approach fix.

    Intruder trajectories can extend far beyond the approach area. Including
    their full extent in the shared limits makes the ownship path unreadable,
    so distant intruder segments are intentionally clipped in comparison plots.
    """
    coordinates = [(0.0, 0.0)]
    for trace in traces:
        coordinates.extend(_trace_points(trace, "KL001", reference_lat, reference_lon))
    values = np.asarray(coordinates, dtype=float)
    x_min, y_min = values.min(axis=0)
    x_max, y_max = values.max(axis=0)
    span = max(x_max - x_min, y_max - y_min, FAF_RING_KM * 2)
    pad = max(span * 0.12, FAF_RING_KM * 0.5)
    x_mid, y_mid = (x_min + x_max) / 2, (y_min + y_max) / 2
    half = span / 2 + pad
    for axis in axes:
        axis.set_xlim(x_mid - half, x_mid + half)
        axis.set_ylim(y_mid - half, y_mid + half)
        axis.set_aspect("equal", adjustable="box")


def plot_trajectory(trajectory: list[dict[str, Any]], reference_lat: float,
                    reference_lon: float, output_path: Path, title: str) -> None:
    """Save a reference-style birdseye plot sampled at agent decisions."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 7))
    axis.set_facecolor("#d0e8f0")
    for acid, color in AIRCRAFT_COLORS.items():
        points = _trace_points(trajectory, acid, reference_lat, reference_lon)
        if not points:
            continue
        xs, ys = zip(*points)
        style = "-" if acid == "KL001" else "--"
        axis.plot(xs, ys, color=color, linestyle=style, linewidth=2 if acid == "KL001" else 1.4,
                  label=acid)
        axis.scatter(xs[0], ys[0], color=color, marker="o", s=48, edgecolor="white", zorder=4)
        axis.annotate(
            PLANE_GLYPH, (xs[-1], ys[-1]), color=color,
            fontsize=15 if acid == "KL001" else 12,
            rotation=_plane_text_angle(_path_heading(
                list(zip(xs, ys)), trajectory[-1][acid]["heading"]
            )),
            rotation_mode="anchor", ha="center", va="center", zorder=6,
        )
        axis.annotate(acid, (xs[-1], ys[-1]), xytext=(5, 5), textcoords="offset points",
                      fontsize=8, color=color)

    axis.add_patch(plt.Circle((0, 0), FAF_RING_KM, color="white", alpha=0.8, zorder=1,
                              label="5 NM FAF zone"))
    axis.scatter([0], [0], color="black", marker="X", s=90, zorder=6, label="FAF")
    axis.set_xlabel("East-west offset from FAF (km)")
    axis.set_ylabel("North-south offset from FAF (km)")
    axis.set_title(title)
    axis.grid(alpha=0.25, color="white")
    _set_equal_limits([axis], [trajectory], reference_lat, reference_lon)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_comparison(
    first: tuple[dict[str, Any], list[dict[str, Any]]],
    second: tuple[dict[str, Any], list[dict[str, Any]]],
    output_path: Path,
    reference_lat: float = 52.3105,
    reference_lon: float = 4.7683,
    labels: tuple[str, str] = ("PPO", "Random"),
) -> None:
    """Save an ownship-focused side-by-side comparison of two episodes."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)
    figure.patch.set_facecolor("white")

    for axis, (metrics, trajectory), label in zip(axes, (first, second), labels):
        axis.set_facecolor("#d0e8f0")
        for acid, color in AIRCRAFT_COLORS.items():
            points = _trace_points(trajectory, acid, reference_lat, reference_lon)
            if not points:
                continue
            xs, ys = zip(*points)
            style = "-" if acid == "KL001" else "--"
            axis.plot(xs, ys, color=color, linestyle=style,
                      linewidth=2.4 if acid == "KL001" else 1.2,
                      alpha=1.0 if acid == "KL001" else 0.75, label=acid)
            axis.scatter(xs[0], ys[0], color=color, marker="o",
                         s=50 if acid == "KL001" else 34, edgecolor="white", zorder=4)
            axis.annotate(
                PLANE_GLYPH, (xs[-1], ys[-1]), color=color,
                fontsize=16 if acid == "KL001" else 13,
                rotation=_plane_text_angle(_path_heading(
                    list(zip(xs, ys)), trajectory[-1][acid]["heading"]
                )),
                rotation_mode="anchor", ha="center", va="center", zorder=6,
            )
            if acid == "KL001":
                axis.annotate("ownship end", (xs[-1], ys[-1]), xytext=(5, 5),
                              textcoords="offset points", fontsize=8, color=color)
        axis.add_patch(plt.Circle((0, 0), FAF_RING_KM, color="white", alpha=0.8, zorder=1))
        axis.scatter([0], [0], color="black", marker="X", s=80, zorder=6, label="FAF")
        axis.set_title(_metric_title(label, metrics), fontsize=10)
        axis.set_xlabel("East-west offset from FAF (km)")
        axis.grid(alpha=0.25, color="white")

    axes[0].set_ylabel("North-south offset from FAF (km)")
    _set_ownship_focus_limits(axes, [first[1], second[1]], reference_lat, reference_lon)
    axes[1].legend(fontsize=8, loc="best", ncol=2)
    figure.suptitle("Ownship-focused trajectory comparison", fontsize=14, y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_interactive_comparison(
    first: tuple[dict[str, Any], list[dict[str, Any]]],
    second: tuple[dict[str, Any], list[dict[str, Any]]],
    output_path: Path,
    reference_lat: float = 52.3105,
    reference_lon: float = 4.7683,
    labels: tuple[str, str] = ("PPO", "Random"),
) -> None:
    """Write an ownship-focused Plotly animation per agent decision."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    episodes = (first, second)
    max_frames = max(len(trace) for _, trace in episodes)
    figure = make_subplots(rows=1, cols=2, shared_xaxes=True, shared_yaxes=True,
                           subplot_titles=[_metric_title(label, metrics)
                                           for (metrics, _), label in zip(episodes, labels)])
    subplot_annotations = [annotation.to_plotly_json() for annotation in figure.layout.annotations]

    def frame_points(trace: list[dict[str, Any]], acid: str, frame_index: int):
        end = min(frame_index, len(trace) - 1)
        return _trace_points(trace, acid, reference_lat, reference_lon, end=end)

    def plane_annotations(frame_index: int):
        annotations = list(subplot_annotations)
        for col, (_, trace) in enumerate(episodes, start=1):
            end = min(frame_index, len(trace) - 1)
            xref = "x" if col == 1 else "x2"
            yref = "y" if col == 1 else "y2"
            for acid, color in AIRCRAFT_COLORS.items():
                points = frame_points(trace, acid, frame_index)
                if not points or acid not in trace[end]:
                    continue
                x, y = points[-1]
                annotations.append({
                    "x": x,
                    "y": y,
                    "xref": xref,
                    "yref": yref,
                    "text": PLANE_GLYPH,
                    "textangle": _plane_text_angle(_trace_plane_heading(
                        trace, acid, frame_index, reference_lat, reference_lon
                    )),
                    "showarrow": False,
                    "xanchor": "center",
                    "yanchor": "middle",
                    "font": {"color": color, "size": 18, "family": "Arial"},
                })
        return annotations

    ring_t = np.linspace(0, 2 * np.pi, 60)
    ring_cos, ring_sin = np.cos(ring_t), np.sin(ring_t)

    def sep_ring_trace(cx: float, cy: float, color: str, show_legend: bool, name: str):
        return go.Scatter(
            x=(cx + SEP_RING_KM * ring_cos).tolist(),
            y=(cy + SEP_RING_KM * ring_sin).tolist(),
            mode="lines",
            name=name, legendgroup=f"sep_{name}", showlegend=show_legend,
            line={"color": color, "width": 1, "dash": "dot"},
            opacity=0.5,
        )

    for col, (_, trace) in enumerate(episodes, start=1):
        for acid, color in AIRCRAFT_COLORS.items():
            points = frame_points(trace, acid, 0)
            if not points:
                points = [(None, None)]
            xs, ys = zip(*points)
            line_style = "solid" if acid == "KL001" else "dash"
            figure.add_trace(go.Scatter(x=list(xs), y=list(ys), mode="lines",
                                        name=acid, legendgroup=acid, showlegend=(col == 1),
                                        line={"color": color, "width": 3 if acid == "KL001" else 2,
                                              "dash": line_style}), row=1, col=col)
        # one separation ring trace per aircraft per panel (animated in frames)
        for acid, color in AIRCRAFT_COLORS.items():
            points = frame_points(trace, acid, 0)
            cx, cy = points[-1] if points else (0.0, 0.0)
            figure.add_trace(
                sep_ring_trace(cx, cy, color, show_legend=(col == 1),
                               name=f"{acid} 5NM"),
                row=1, col=col,
            )

    # Static FAF markers/rings
    for col in (1, 2):
        figure.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text", text=["FAF"],
                                    textposition="bottom center", name="FAF", showlegend=(col == 1),
                                    marker={"color": "black", "size": 12, "symbol": "x"}),
                         row=1, col=col)
        ring = np.linspace(0, 2 * np.pi, 100)
        figure.add_trace(go.Scatter(x=(FAF_RING_KM * np.cos(ring)).tolist(),
                                    y=(FAF_RING_KM * np.sin(ring)).tolist(), mode="lines",
                                    fill="toself", fillcolor="rgba(255,255,255,0.65)",
                                    name="5 NM FAF zone", showlegend=(col == 1),
                                    line={"color": "white", "width": 2, "dash": "dot"}),
                         row=1, col=col)

    frames = []
    for frame_index in range(max_frames):
        data = []
        for _, trace in episodes:
            # path traces
            for acid in AIRCRAFT_COLORS:
                points = frame_points(trace, acid, frame_index)
                if not points:
                    points = [(None, None)]
                xs, ys = zip(*points)
                data.append(go.Scatter(x=list(xs), y=list(ys)))
            # separation ring traces — centred on current aircraft position
            for acid, color in AIRCRAFT_COLORS.items():
                points = frame_points(trace, acid, frame_index)
                cx, cy = points[-1] if points else (0.0, 0.0)
                data.append(go.Scatter(
                    x=(cx + SEP_RING_KM * ring_cos).tolist(),
                    y=(cy + SEP_RING_KM * ring_sin).tolist(),
                ))
        frames.append(go.Frame(
            data=data, name=str(frame_index),
            layout={"annotations": plane_annotations(frame_index)},
        ))
    figure.frames = frames

    slider_steps = [
        {"label": str(index), "method": "animate", "args": [[str(index)],
         {"mode": "immediate", "frame": {"duration": 0, "redraw": True},
          "transition": {"duration": 0}}]}
        for index in range(max_frames)
    ]
    figure.update_layout(
        title=f"Ownship-focused: {labels[0]} versus {labels[1]} — one frame per agent decision",
        template="plotly_white",
        plot_bgcolor="#d0e8f0",
        paper_bgcolor="white",
        legend={"title": {"text": "Aircraft (color)"}},
        annotations=plane_annotations(0),
        height=700,
        hovermode="closest",
        updatemenus=[{"type": "buttons", "showactive": False, "x": 0.1, "y": 1.12,
                      "buttons": [
                          {"label": "▶ Play", "method": "animate", "args": [None, {
                              "fromcurrent": True, "frame": {"duration": 500, "redraw": True},
                              "transition": {"duration": 200}}]},
                          {"label": "⏸ Pause", "method": "animate", "args": [[None], {
                              "mode": "immediate", "frame": {"duration": 0, "redraw": False},
                              "transition": {"duration": 0}}]},
                      ]}],
        sliders=[{"active": 0, "currentvalue": {"prefix": "Agent decision: "},
                  "pad": {"t": 35}, "steps": slider_steps}],
    )
    _set_ownship_focus_plotly_limits(
        figure, [first[1], second[1]], reference_lat, reference_lon
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs=True, full_html=True, auto_play=False)


def plot_safety_diagnostics(
    records_by_policy: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> None:
    """Save aggregate safety diagnostics for matched policy evaluations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    policies = list(records_by_policy)
    colors = ["#1769aa", "#d62728", "#2ca02c", "#9467bd"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    figure.suptitle("PPO Phase 1 safety diagnostics", fontsize=14)

    for index, policy in enumerate(policies):
        records = records_by_policy[policy]
        color = colors[index % len(colors)]
        episodes = np.arange(1, len(records) + 1)
        separations = np.asarray(
            [record["minimum_separation_nm"] for record in records], dtype=float
        )
        intrusions = np.asarray(
            [record["total_intrusions"] for record in records], dtype=float
        )
        rewards = np.asarray([record["reward"] for record in records], dtype=float)
        axes[0, 0].scatter(episodes, separations, label=policy, color=color, s=42)
        axes[0, 1].bar(
            index,
            float(np.mean(separations < INTRUSION_DISTANCE)),
            color=color,
            label=policy,
        )
        axes[1, 0].bar(index, float(np.mean(intrusions)), color=color, label=policy)
        axes[1, 1].scatter(separations, rewards, label=policy, color=color, s=42)

    axes[0, 0].axhline(INTRUSION_DISTANCE, color="black", linestyle="--", label="5 NM standard")
    axes[0, 0].set_title("Minimum separation by episode")
    axes[0, 0].set_xlabel("Evaluation episode")
    axes[0, 0].set_ylabel("Minimum separation (NM)")
    axes[0, 1].set_title("Unsafe episode rate")
    axes[0, 1].set_ylabel("Fraction below 5 NM")
    axes[0, 1].set_xticks(range(len(policies)), policies, rotation=20, ha="right")
    axes[0, 1].set_ylim(0, 1)
    axes[1, 0].set_title("Mean intrusion count")
    axes[1, 0].set_ylabel("Intrusions per episode")
    axes[1, 0].set_xticks(range(len(policies)), policies, rotation=20, ha="right")
    axes[1, 1].set_title("Reward versus minimum separation")
    axes[1, 1].set_xlabel("Minimum separation (NM)")
    axes[1, 1].set_ylabel("Episode reward")
    axes[0, 0].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_action_and_separation(
    trace: list[dict[str, Any]], output_path: Path, title: str,
) -> None:
    """Save a decision-step plot of safety margin and PPO actions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.asarray([frame["decision_step"] for frame in trace], dtype=int)
    separation = np.asarray([
        np.nan if frame["minimum_separation_nm"] is None
        else frame["minimum_separation_nm"] for frame in trace
    ], dtype=float)
    actions = np.asarray([
        [np.nan, np.nan] if frame["action"] is None else frame["action"]
        for frame in trace
    ], dtype=float)

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(steps, separation, color="#1769aa", linewidth=2, marker="o", markersize=3)
    axes[0].axhline(INTRUSION_DISTANCE, color="red", linestyle="--", label="5 NM standard")
    axes[0].set_ylabel("Minimum separation (NM)")
    axes[0].set_title(title)
    axes[0].legend()
    axes[1].plot(steps, actions[:, 0], label="heading action", color="#d62728")
    axes[1].plot(steps, actions[:, 1], label="speed action", color="#2ca02c")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Agent decision")
    axes[1].set_ylabel("Normalized action")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _set_ownship_focus_plotly_limits(
    figure: Any, traces: list[list[dict[str, Any]]],
    reference_lat: float, reference_lon: float,
) -> None:
    """Apply ownship-focused equal local-coordinate ranges to Plotly panels."""
    coordinates = [(0.0, 0.0)]
    for trace in traces:
        coordinates.extend(_trace_points(trace, "KL001", reference_lat, reference_lon))
    values = np.asarray(coordinates, dtype=float)
    x_min, y_min = values.min(axis=0)
    x_max, y_max = values.max(axis=0)
    span = max(x_max - x_min, y_max - y_min, FAF_RING_KM * 2)
    pad = max(span * 0.12, FAF_RING_KM * 0.5)
    x_mid, y_mid = (x_min + x_max) / 2, (y_min + y_max) / 2
    half = span / 2 + pad
    bounds = [x_mid - half, x_mid + half]
    figure.update_xaxes(range=bounds, scaleanchor="y", scaleratio=1)
    figure.update_yaxes(range=[y_mid - half, y_mid + half])
