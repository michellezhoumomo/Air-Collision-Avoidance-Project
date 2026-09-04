# Reusable PPO Experiment Scripts

These scripts keep training and evaluation reproducible for the group while the notebook remains focused on explanation and research discussion. They use the real `ApproachCREnv2D` BlueSky environment and run one episode at a time because BlueSky is a singleton simulator.

Set SRES credentials in the shell before installing:

```bash
export UV_INDEX_SRES_USERNAME="<your-user>"
export UV_INDEX_SRES_PASSWORD="<your-password>"
uv sync
```

Run the environment check and seeded random baseline:

```bash
uv run ppo-env-check --episodes 5 --seed 1000
```

Run an initial PPO experiment:

```bash
uv run ppo-train --timesteps 10000 --episodes 10 --eval-seed 1000
```

Open the presentation notebook through the same UV-managed environment:

```bash
uv run jupyter notebook playground/ppo_phase1_bluesky.ipynb
```

The default `action_freq=1, dt=10` is the fast BlueSky exploration setting used by the reference notebook. For nominal timing, use `--action-freq 10 --dt 5`. Training writes the model, episode CSV files, JSON summaries, a training-progress plot, a reference-style PNG, and a self-contained step-by-step Plotly HTML under `runs/ppo_phase1/`. The training CSV includes the PPO timestep at which each episode ended, so reward and safety can be tracked over learning.

The default intruder behavior is the reproducible straight-line baseline. For a
separate robustness experiment with smooth randomized intruder routes that still
end at the FAF, add `--intruder-behavior curved` to training or evaluation.

Training also logs the complete bundle to a local SQLite MLflow run by default:

```bash
uv run ppo-train --timesteps 10000 --episodes 10 \
  --output-dir runs/ppo_phase1
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db \
  --host 127.0.0.1 --port 5000
```

Open the run in the UI to download `research_bundle`, or open
`runs/ppo_phase1/ppo_vs_random_interactive.html` directly. The animation advances
one frame per PPO/environment decision, not once per internal BlueSky tick. Use
`--no-mlflow` when only local files are needed.

When no `--mlflow-run-name` is provided, the training command creates a
descriptive name from the intruder behavior, reward profile, training budget,
and timing. Each run also logs three matched trajectory examples under
`research_bundle/trajectory_examples/`, using the same filenames across runs so
MLflow can compare their artifacts.

Evaluate a saved model on a separate seed range:

```bash
uv run ppo-evaluate runs/ppo_phase1/ppo_phase1_bluesky.zip
```

Run the initial one-factor sensitivity study:

```bash
uv run ppo-sweep --timesteps 10000 --episodes 10
```

The sweep compares the baseline, no entropy bonus, and a longer PPO rollout. It keeps the seed list and BlueSky timing identical across configurations.

Run the Phase 1 safety-shaped reward comparison:

```bash
uv run ppo-safety-compare \
  --timesteps 2048 --episodes 10 --output-dir runs/ppo_safety_smoke
```

This preserves the baseline and compares it with `safety_shaping` using stronger
proximity and intrusion penalties. It writes per-profile models and metrics,
aggregate safety diagnostics, action/separation plots, a labeled birdseye PNG,
and a self-contained Plotly HTML artifact. Use `--no-mlflow` to skip local
MLflow logging.

Evaluate more unseen scenarios across fast and nominal timing:

```bash
uv run ppo-scenario-matrix \
  --episodes 50 --seed 2000 --output-dir runs/ppo_scenario_matrix
```

The matrix evaluates the full baseline checkpoint and both safety-smoke
checkpoints on the same 50 seeds, plus matched random policies.

The primary metrics are fix-reached rate, mean reward, intrusion count, minimum separation, average drift, and episode length. Training and evaluation use separate seed ranges by default; compare policies only on the same held-out seed list and simulation settings.
