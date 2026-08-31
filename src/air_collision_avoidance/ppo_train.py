"""Train and evaluate a reproducible PPO Phase 1 BlueSky experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from air_collision_avoidance.ppo_utils import (
    PPOEpisodeCallback,
    evaluate_policy,
    make_env,
    make_ppo_model,
    plot_comparison,
    plot_interactive_comparison,
    plot_training_progress,
    run_episode,
    save_metrics,
    summarize_metrics,
)


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    """Train, evaluate, and write the complete local research bundle."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_seeds = [args.eval_seed + index for index in range(args.episodes)]

    random_records = evaluate_policy(
        None, evaluation_seeds, action_freq=args.action_freq, dt=args.dt,
        reward_profile=args.reward_profile,
        intruder_behavior=args.intruder_behavior,
    )
    save_metrics(random_records, summarize_metrics(random_records), args.output_dir, "random_baseline")

    train_env = make_env(
        action_freq=args.action_freq, dt=args.dt,
        reward_profile=args.reward_profile,
        intruder_behavior=args.intruder_behavior,
    )
    train_env.reset(seed=args.seed)
    callback = PPOEpisodeCallback()
    model = make_ppo_model(train_env, seed=args.seed, verbose=1)
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model_path = args.output_dir / "ppo_phase1_bluesky"
    model.save(str(model_path))
    train_env.close()

    save_metrics(callback.episodes, summarize_metrics(callback.episodes), args.output_dir, "training_episodes")
    plot_training_progress(
        callback.episodes, args.output_dir / "training_progress.png",
        title=f"PPO training progress — {args.reward_profile}",
    )

    trained_records = evaluate_policy(
        model, evaluation_seeds, action_freq=args.action_freq, dt=args.dt,
        reward_profile=args.reward_profile,
        intruder_behavior=args.intruder_behavior,
    )
    save_metrics(trained_records, summarize_metrics(trained_records), args.output_dir, "ppo_evaluation")

    ppo_episode = run_episode(
        model, evaluation_seeds[0], args.action_freq, args.dt, args.reward_profile,
        args.intruder_behavior,
    )
    random_episode = run_episode(
        None, evaluation_seeds[0], args.action_freq, args.dt, args.reward_profile,
        args.intruder_behavior,
    )
    plot_comparison(ppo_episode, random_episode, args.output_dir / "ppo_vs_random.png")
    plot_interactive_comparison(
        ppo_episode, random_episode, args.output_dir / "ppo_vs_random_interactive.html"
    )

    trajectory_examples_dir = args.output_dir / "trajectory_examples"
    trajectory_examples = [(ppo_episode, random_episode)]
    for seed in evaluation_seeds[1:3]:
        trajectory_examples.append((
            run_episode(
                model, seed, args.action_freq, args.dt, args.reward_profile,
                args.intruder_behavior,
            ),
            run_episode(
                None, seed, args.action_freq, args.dt, args.reward_profile,
                args.intruder_behavior,
            ),
        ))
    for index, (ppo_example, random_example) in enumerate(trajectory_examples, start=1):
        plot_comparison(
            ppo_example, random_example,
            trajectory_examples_dir / f"example_{index}.png",
        )
        plot_interactive_comparison(
            ppo_example, random_example,
            trajectory_examples_dir / f"example_{index}.html",
        )

    summary = {
        "config": {
            "timesteps": args.timesteps,
            "seed": args.seed,
            "eval_seed": args.eval_seed,
            "episodes": args.episodes,
            "action_freq": args.action_freq,
            "dt": args.dt,
            "reward_profile": args.reward_profile,
            "intruder_behavior": args.intruder_behavior,
        },
        "random_baseline": summarize_metrics(random_records),
        "ppo": summarize_metrics(trained_records),
        "model": str(model_path.with_suffix(".zip")),
    }
    with (args.output_dir / "config.json").open("w") as file:
        json.dump(summary["config"], file, indent=2)
        file.write("\n")
    with (args.output_dir / "experiment_summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
    return summary


def log_mlflow(summary: dict[str, object], output_dir: Path) -> None:
    """Log metrics and every generated research artifact to the active run."""
    import mlflow

    config = summary["config"]
    assert isinstance(config, dict)
    mlflow.log_params(config)
    for prefix in ("random_baseline", "ppo"):
        metrics = summary[prefix]
        assert isinstance(metrics, dict)
        mlflow.log_metrics({
            f"{prefix}_{key}": float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        })
    # This includes the model ZIP, config, CSV/JSON metrics, PNG, and HTML.
    mlflow.log_artifacts(str(output_dir), artifact_path="research_bundle")


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
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ppo_phase1"))
    parser.add_argument("--mlflow-tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--mlflow-experiment", default="air-collision-avoidance-ppo")
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Skip local MLflow logging for a script-only run.")
    args = parser.parse_args()

    if args.no_mlflow:
        summary = run_experiment(args)
    else:
        import mlflow

        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        run_name = args.mlflow_run_name or (
            f"ppo-{args.intruder_behavior}-{args.reward_profile}-"
            f"{args.timesteps}steps-f{args.action_freq}-dt{args.dt:g}"
        )
        with mlflow.start_run(run_name=run_name):
            summary = run_experiment(args)
            log_mlflow(summary, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
