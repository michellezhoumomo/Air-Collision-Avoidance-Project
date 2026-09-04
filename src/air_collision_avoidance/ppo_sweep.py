"""Run a small one-factor PPO sensitivity study on fixed BlueSky seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from air_collision_avoidance.ppo_utils import (
    PPOEpisodeCallback,
    evaluate_policy,
    make_env,
    make_ppo_model,
    save_metrics,
    summarize_metrics,
)


CONFIGS = {
    "baseline": {},
    "no_entropy": {"ent_coef": 0.0},
    "long_rollout": {"n_steps": 256, "batch_size": 128},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--action-freq", type=int, default=1)
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--reward-profile", choices=("baseline", "safety_shaping"),
                        default="baseline")
    parser.add_argument("--intruder-behavior", choices=("straight", "curved"),
                        default="straight")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ppo_sweep"))
    args = parser.parse_args()

    seeds = [args.eval_seed + index for index in range(args.episodes)]
    summaries = {}
    for name, overrides in CONFIGS.items():
        directory = args.output_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        env = make_env(
            action_freq=args.action_freq, dt=args.dt,
            reward_profile=args.reward_profile,
            intruder_behavior=args.intruder_behavior,
        )
        env.reset(seed=args.seed)
        callback = PPOEpisodeCallback()
        model = make_ppo_model(env, seed=args.seed, **overrides)
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save(str(directory / "model"))
        env.close()

        records = evaluate_policy(
            model, seeds, action_freq=args.action_freq, dt=args.dt,
            reward_profile=args.reward_profile,
            intruder_behavior=args.intruder_behavior,
        )
        save_metrics(records, summarize_metrics(records), directory, "evaluation")
        save_metrics(callback.episodes, summarize_metrics(callback.episodes), directory, "training_episodes")
        summaries[name] = summarize_metrics(records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "sweep_summary.json").open("w") as file:
        json.dump(summaries, file, indent=2)
        file.write("\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
