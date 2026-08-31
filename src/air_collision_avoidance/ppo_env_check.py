"""Run BlueSky environment checks and establish a seeded random baseline."""

from __future__ import annotations

import argparse
import json

import numpy as np
from stable_baselines3.common.env_checker import check_env

from air_collision_avoidance.envs.approach_cr_2d import ApproachCREnv2D
from air_collision_avoidance.ppo_utils import evaluate_policy, make_env, summarize_metrics


def seeded_reset_matches(seed: int, action_freq: int, dt: float,
                         intruder_behavior: str = "straight") -> bool:
    """Verify that the same Gymnasium seed recreates the initial observation."""

    env = ApproachCREnv2D(
        action_freq=action_freq, dt=dt, intruder_behavior=intruder_behavior
    )
    first, _ = env.reset(seed=seed)
    second, _ = env.reset(seed=seed)
    env.close()
    return all(np.array_equal(first[key], second[key]) for key in first)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-freq", type=int, default=1)
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--intruder-behavior", choices=("straight", "curved"),
                        default="straight")
    args = parser.parse_args()

    env = make_env(
        action_freq=args.action_freq, dt=args.dt,
        intruder_behavior=args.intruder_behavior,
    )
    check_env(env, warn=True)
    env.close()

    seeds = list(range(args.seed, args.seed + args.episodes))
    records = evaluate_policy(
        None, seeds, action_freq=args.action_freq, dt=args.dt,
        intruder_behavior=args.intruder_behavior,
    )
    print(json.dumps({
        "seeded_reset_matches": seeded_reset_matches(
            args.seed, args.action_freq, args.dt, args.intruder_behavior
        ),
        "random_baseline": summarize_metrics(records),
        "episodes": records,
    }, indent=2))


if __name__ == "__main__":
    main()
