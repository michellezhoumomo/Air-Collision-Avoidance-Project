"""DQN-specific evaluation and episode helpers for Phase 1."""

from __future__ import annotations

from typing import Any

import numpy as np

from air_collision_avoidance.envs.approach_cr_2d import (
    APPROACH_DISTANCE_MAX,
    APPROACH_DISTANCE_MIN,
    ApproachCREnv2D,
)


def run_episode_dqn(
    model: Any | None,
    seed: int,
    action_freq: int = 1,
    dt: float = 10.0,
    reward_profile: str = "dqn_phase1",
    intruder_behavior: str = "curved",
    spawn_distance_min: float = APPROACH_DISTANCE_MIN,
    spawn_distance_max: float = APPROACH_DISTANCE_MAX,
    dqn_action_map: list[tuple[float, float]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one episode with a DQN, straight-line, or random policy.

    Unlike ``ppo_utils.run_episode``, this function:
    - defaults to ``reward_profile='dqn_phase1'`` so eval matches training
    - accepts ``dqn_action_map`` to convert discrete action indices to
      continuous [hdg, spd] pairs without a wrapper class
    - defaults to ``intruder_behavior='curved'`` to match Phase B training

    Pass ``model='straight'`` for the straight-line baseline or ``None`` for
    a seeded random policy.
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
        "intrusion_seconds": float(info.get("intrusion_seconds", 0.0)),
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
