"""Evaluate available PPO checkpoints across matched Phase 1 scenarios."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from stable_baselines3 import PPO

from air_collision_avoidance.ppo_utils import evaluate_policy, summarize_metrics


DEFAULT_MODELS = {
    "full_baseline": (
        Path("runs/ppo_phase1_full/ppo_phase1_bluesky.zip"), "baseline"
    ),
    "smoke_baseline": (
        Path("runs/ppo_safety_smoke/baseline/ppo_phase1_bluesky.zip"), "baseline"
    ),
    "smoke_safety": (
        Path("runs/ppo_safety_smoke/safety_shaping/ppo_phase1_bluesky.zip"),
        "safety_shaping",
    ),
}

SCENARIOS = {
    "fast": {"action_freq": 1, "dt": 10.0},
    "nominal": {"action_freq": 10, "dt": 5.0},
}


def _parse_model_specs(values: list[str]) -> dict[str, tuple[Path, str]]:
    """Parse repeated ``NAME=MODEL_PATH[,PROFILE]`` options."""
    if not values:
        return DEFAULT_MODELS
    models = {}
    for value in values:
        name, spec = value.split("=", 1)
        path, _, profile = spec.rpartition(",")
        if not path or not profile:
            raise ValueError(
                "--model must look like NAME=MODEL_PATH,REWARD_PROFILE"
            )
        models[name] = (Path(path), profile)
    return models


def run_matrix(args: argparse.Namespace) -> dict[str, object]:
    models = _parse_model_specs(args.model)
    missing = [str(path) for path, _ in models.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model checkpoint(s): {', '.join(missing)}")

    seeds = [args.seed + index for index in range(args.episodes)]
    all_records: list[dict[str, object]] = []
    summaries: dict[str, object] = {}

    for scenario, settings in SCENARIOS.items():
        for policy, (model_path, reward_profile) in models.items():
            model = PPO.load(model_path)
            records = evaluate_policy(
                model,
                seeds,
                action_freq=settings["action_freq"],
                dt=settings["dt"],
                reward_profile=reward_profile,
                intruder_behavior=args.intruder_behavior,
            )
            label = f"{policy} — {scenario}"
            summaries[label] = summarize_metrics(records)
            all_records.extend({
                "policy": policy,
                "scenario": scenario,
                **record,
            } for record in records)

        for reward_profile in ("baseline", "safety_shaping"):
            records = evaluate_policy(
                None,
                seeds,
                action_freq=settings["action_freq"],
                dt=settings["dt"],
                reward_profile=reward_profile,
                intruder_behavior=args.intruder_behavior,
            )
            policy = f"random_{reward_profile}"
            label = f"{policy} — {scenario}"
            summaries[label] = summarize_metrics(records)
            all_records.extend({
                "policy": policy,
                "scenario": scenario,
                **record,
            } for record in records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "scenario_records.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=all_records[0].keys())
        writer.writeheader()
        writer.writerows(all_records)

    result = {
        "config": {
            "episodes": args.episodes,
            "seed": args.seed,
            "scenarios": SCENARIOS,
            "intruder_behavior": args.intruder_behavior,
            "models": {
                name: {"path": str(path), "reward_profile": profile}
                for name, (path, profile) in models.items()
            },
        },
        "summaries": summaries,
    }
    with (args.output_dir / "scenario_summary.json").open("w") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--intruder-behavior", choices=("straight", "curved"),
                        default="straight")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ppo_scenario_matrix"))
    parser.add_argument(
        "--model", action="append", default=[],
        help="NAME=MODEL_PATH,REWARD_PROFILE; may be repeated",
    )
    args = parser.parse_args()
    run_matrix(args)


if __name__ == "__main__":
    main()
