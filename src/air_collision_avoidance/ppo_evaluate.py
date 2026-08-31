"""Evaluate a saved PPO model against fixed BlueSky episode seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from air_collision_avoidance.ppo_utils import evaluate_policy, save_metrics, summarize_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--action-freq", type=int, default=1)
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--reward-profile", choices=("baseline", "safety_shaping"),
                        default="baseline")
    parser.add_argument("--intruder-behavior", choices=("straight", "curved"),
                        default="straight")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ppo_evaluation"))
    args = parser.parse_args()

    model = PPO.load(args.model)
    seeds = [args.seed + index for index in range(args.episodes)]
    records = evaluate_policy(
        model, seeds, action_freq=args.action_freq, dt=args.dt,
        reward_profile=args.reward_profile,
        intruder_behavior=args.intruder_behavior,
    )
    summary = summarize_metrics(records)
    save_metrics(records, summary, args.output_dir, "ppo_evaluation")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
