"""Compare baseline and safety-shaped PPO rewards on matched Phase 1 seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from air_collision_avoidance.ppo_utils import (
    PPOEpisodeCallback,
    evaluate_policy,
    make_env,
    make_ppo_model,
    plot_action_and_separation,
    plot_comparison,
    plot_interactive_comparison,
    plot_safety_diagnostics,
    plot_training_progress,
    run_episode,
    save_metrics,
    summarize_metrics,
)


PROFILES = ("baseline", "safety_shaping")


def run_profile(args: argparse.Namespace, profile: str, seeds: list[int]) -> dict[str, object]:
    """Train and evaluate one reward profile using the shared PPO settings."""
    directory = args.output_dir / profile
    directory.mkdir(parents=True, exist_ok=True)

    train_env = make_env(
        action_freq=args.action_freq, dt=args.dt, reward_profile=profile,
        intruder_behavior=args.intruder_behavior,
    )
    train_env.reset(seed=args.seed)
    callback = PPOEpisodeCallback()
    model = make_ppo_model(train_env, seed=args.seed, verbose=0)
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model_path = directory / "ppo_phase1_bluesky"
    model.save(str(model_path))
    train_env.close()

    records = evaluate_policy(
        model, seeds, action_freq=args.action_freq, dt=args.dt,
        reward_profile=profile,
        intruder_behavior=args.intruder_behavior,
    )
    save_metrics(records, summarize_metrics(records), directory, "ppo_evaluation")
    save_metrics(callback.episodes, summarize_metrics(callback.episodes), directory,
                 "training_episodes")
    plot_training_progress(
        callback.episodes, directory / "training_progress.png",
        title=f"PPO training progress — {profile}",
    )
    return {
        "profile": profile,
        "model": str(model_path.with_suffix(".zip")),
        "records": records,
        "training_summary": summarize_metrics(callback.episodes),
        "evaluation_summary": summarize_metrics(records),
    }


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    """Run the matched safety comparison and save its complete local bundle."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [args.eval_seed + index for index in range(args.episodes)]
    results = {
        profile: run_profile(args, profile, seeds) for profile in PROFILES
    }

    records_by_policy = {
        f"PPO {profile}": result["records"]
        for profile, result in results.items()
    }
    comparison_records = [
        {"policy": policy, **record}
        for policy, records in records_by_policy.items()
        for record in records
    ]
    comparison_summary = {
        policy: summarize_metrics(records)
        for policy, records in records_by_policy.items()
    }
    save_metrics(comparison_records, comparison_summary, args.output_dir, "comparison")

    baseline_episode = run_episode(
        PPO.load(results["baseline"]["model"]), seeds[0],
        args.action_freq, args.dt, "baseline", args.intruder_behavior,
    )
    safety_episode = run_episode(
        PPO.load(results["safety_shaping"]["model"]), seeds[0],
        args.action_freq, args.dt, "safety_shaping", args.intruder_behavior,
    )
    plot_comparison(
        baseline_episode, safety_episode,
        args.output_dir / "ppo_baseline_vs_safety.png",
        labels=("PPO baseline", "PPO safety shaping"),
    )
    plot_interactive_comparison(
        baseline_episode, safety_episode,
        args.output_dir / "ppo_baseline_vs_safety_interactive.html",
        labels=("PPO baseline", "PPO safety shaping"),
    )
    plot_safety_diagnostics(
        records_by_policy, args.output_dir / "safety_diagnostics.png"
    )
    plot_action_and_separation(
        safety_episode[1], args.output_dir / "safety_action_trace.png",
        "Safety-shaped PPO — action and separation by decision",
    )

    summary = {
        "config": {
            "timesteps": args.timesteps,
            "seed": args.seed,
            "eval_seed": args.eval_seed,
            "episodes": args.episodes,
            "action_freq": args.action_freq,
            "dt": args.dt,
            "profiles": list(PROFILES),
            "intruder_behavior": args.intruder_behavior,
        },
        "evaluation": comparison_summary,
        "models": {profile: results[profile]["model"] for profile in PROFILES},
    }
    with (args.output_dir / "config.json").open("w") as file:
        json.dump(summary["config"], file, indent=2)
        file.write("\n")
    with (args.output_dir / "safety_comparison_summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
    return summary


def log_mlflow(summary: dict[str, object], output_dir: Path) -> None:
    """Log the safety comparison bundle to the active local MLflow run."""
    import mlflow

    config = summary["config"]
    assert isinstance(config, dict)
    mlflow.log_params({key: value for key, value in config.items() if not isinstance(value, list)})
    evaluation = summary["evaluation"]
    assert isinstance(evaluation, dict)
    for policy, metrics in evaluation.items():
        assert isinstance(metrics, dict)
        prefix = policy.lower().replace(" ", "_")
        mlflow.log_metrics({
            f"{prefix}_{key}": float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        })
    mlflow.log_artifacts(str(output_dir), artifact_path="safety_research_bundle")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=2_048)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--action-freq", type=int, default=1)
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--intruder-behavior", choices=("straight", "curved"),
                        default="straight")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ppo_safety_smoke"))
    parser.add_argument("--mlflow-tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--mlflow-experiment", default="air-collision-avoidance-ppo")
    parser.add_argument("--mlflow-run-name", default="ppo-safety-smoke")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    if args.no_mlflow:
        summary = run_experiment(args)
    else:
        import mlflow

        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run(run_name=args.mlflow_run_name):
            summary = run_experiment(args)
            log_mlflow(summary, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
