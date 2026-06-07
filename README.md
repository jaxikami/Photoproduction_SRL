# Safe Policy Reinforcement Learning (SPRL) for Bioreactor Control

This repository implements a Safe Reinforcement Learning pipeline for phycocyanin production control in a continuous multi-stage photobioreactor. It compares:

- `Standard RL`: A baseline Proximal Policy Optimization (PPO) agent using a Lagrangian multiplier approach (reward penalties) for constraint handling.
- `safe_RL agent`: A PPO agent equipped with a learned Action Projection Network (APN) that projects unsafe action intents onto a safe action manifold.

## Overview

The environment is a nonlinear, volume-tracked photobioreactor with Runge-Kutta integration and stage-dependent control.

- **Normalized observation (12D):**
  - `[Cx, CN, Cq, V]` physical states normalized
  - stage one-hot vector (4 dims)
  - remaining stage credit
  - normalized episode time
  - normalized nitrate supply
  - operation time left
- **Action space (4D, bounded to `[-1, 1]`):**
  - stage duration time multiplier
  - light intensity `I`
  - nitrate feed `Fn`
  - outstream flow `Fout`
- **Objective:**
  - Maximize phycocyanin production while strictly respecting process, volume, and terminal constraints.

## Safety Constraints

Implemented constraints in the environment (`env_core.py`):

- `G1` (Path Nitrate): `CN <= 800 mg/L`
- `G2` (Quality Ratio): `Cq/Cx <= 0.011`
- `G3` (Terminal Nitrate): `CN <= 150 mg/L` at episode end
- `G4` (Reactor Overflow): `V <= 50 L`
- `G5` (Terminal Stage): Episode must end in Idle stage (stage 3)

*Note: APN pretraining covers instantaneous constraints (G1, G2, G4). Constraints G3 and G5 are handled via Lagrangian multipliers and temporal policy behavior.*

## Architecture

### SPRL Agent (`safe_agent.py`)
1. **Actor-Critic Policy**: Feedforward MLP with stage-aware action masking.
2. **Safety Filter (APN)**: Loaded from `policy/action_projection_network.pth`. Uses gradient ascent on the margin surface to project action intents to safe proxies before execution.

### Baseline Agent (`lag_agent.py`)
1. **Actor-Critic Policy**: Standard PPO architecture without the projection filter. Relies on reward shaping to learn safety behaviors.

## Project Structure

| File | Purpose |
| :--- | :--- |
| `main.py` | Training and evaluation orchestrator for the agents. |
| `env_core.py` | Base photobioreactor environment, state representation, and RK4 dynamics. |
| `env_bench.py` | Benchmark environment for Standard RL using Lagrangian penalties. |
| `env_safe.py` | Safe environment specifically integrated with the APN agent. |
| `safe_agent.py` | SPRL implementation featuring the APN projection safety filter. |
| `lag_agent.py` | Baseline Standard RL (Lagrangian) implementation. |
| `pretrain.py` | Generative dataset training loop for the APN classifier. |
| `data_gen.py` | Physics-based offline dataset generation for APN pretraining. |
| `validation.py` | Constraint validation suite for the APN model. |
| `utils.py` | Logging and plotting utilities for visualizing trajectories and violations. |
| `policy/` | Checkpoint directory for the APN and RL weights. |
| `plot/` | Directory where generated training and evaluation plots are saved. |

## Setup

Use Python 3.10+ (recommended) and install dependencies:

```bash
pip install torch numpy matplotlib pandas tqdm
```

## How To Run

1. **Pretrain APN** (if you need to generate a new safety filter from scratch)
```bash
python pretrain.py
```

2. **Validate APN Safety Behavior**
```bash
python validation.py
```

3. **Train and/or Evaluate RL Agents**
Open `main.py` and set:
- `EVALUATE_ONLY = False` to train and then evaluate
- `EVALUATE_ONLY = True` to evaluate existing checkpoints only
- `RUN_BENCHMARK = True/False` to toggle between Standard RL and Safe RL.

Then run:
```bash
python main.py
```

## Outputs

Generated plots are saved in the `plot/` directory, including:
- `training_Standard RL.png` / `training_safe_RL agent.png`
- `training_violations.png`
- Detailed Evaluation Trajectories: `plot_nitrate.png`, `plot_phycocyanin.png`, `plot_light.png`, `plot_nitrate_feed.png`, `plot_volume.png`, `plot_ratio.png`, and `plot_violations.png`.
- `am_loss.png` (APN pretraining convergence)
