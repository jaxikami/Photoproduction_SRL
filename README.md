# Safe Policy Reinforcement Learning (SPRL) for Bioreactor Control

This repository implements a Safe Reinforcement Learning pipeline for phycocyanin production control in a multi-stage photobioreactor. It compares:

- `Standard RL`: PPO with reward penalties for constraint handling.
- `SPRL`: PPO with a learned Action Projection Network (APN) that projects intents toward a safe action manifold.

## Overview

The environment (`env.py`) is a nonlinear, volume-tracked photobioreactor with Runge-Kutta integration and stage-dependent control.

- Normalized observation (11D):
	- `[Cx, CN, Cq, V]` normalized
	- stage one-hot (4 dims)
	- remaining stage credit
	- normalized time
	- normalized nitrate supply
- Action space (4D, normalized to `[-1, 1]`):
	- time multiplier
	- light intensity `I`
	- nitrate feed `Fn`
	- outstream flow `Fout`
- Objective:
	- maximize phycocyanin production while respecting process and terminal constraints.

## Safety Constraints

Implemented constraints in `env.py`:

- `g1` (path nitrate): `CN <= 800 mg/L`
- `g2` (quality ratio): `Cq/Cx <= 0.011`
- `g3` (terminal nitrate): `CN <= 150 mg/L` at episode end
- `g4` (overflow): `V <= 50 L`
- `g5` (underflow): `V >= 5 L`

Notes:

- APN pretraining covers `g1`, `g2`, `g4`, `g5` (instantaneous constraints).
- `g3` is handled in environment terminal logic and temporal policy behavior.

## Architecture

### SPRL agent (`safe_agent.py`)

1. Actor-Critic policy:
	 - dual encoder (GRU temporal context + skip encoder)
	 - stage-aware action masking
2. Safety filter (APN):
	 - loaded from `policy/action_projection_network.pth`
	 - gradient projection of action intent onto safer region

### Baseline agent (`lag_agent.py`)

- standard PPO actor-critic with no projection filter.
- relies on reward shaping and penalty terms for safety behavior.

## Project Structure

| File | Purpose |
| :--- | :--- |
| `main.py` | Training/evaluation entry point for `Standard RL` and `SPRL`. |
| `env.py` | Multi-stage photobioreactor environment, constraints, rewards, RK4 dynamics. |
| `safe_agent.py` | SPRL implementation (GRU actor-critic + APN projection safety filter). |
| `lag_agent.py` | Baseline PPO implementation. |
| `pretrain.py` | APN training loop. |
| `data_gen.py` | Synthetic dataset generation for APN pretraining. |
| `validation.py` | APN validation suite (boundary and identity tests). |
| `utils.py` | Logging and plotting utilities. |
| `policy/action_projection_network.pth` | APN checkpoint used by `SPRL`. |
| `plot/` | Generated figures. |

## Setup

Use Python 3.10+ (recommended) and install dependencies:

```bash
pip install torch numpy numba matplotlib pandas tqdm
```

## How To Run

1. Pretrain APN (if you need to regenerate the safety filter)

```bash
python pretrain.py
```

2. Validate APN safety behavior

```bash
python validation.py
```

3. Train and/or evaluate RL agents

- Open `main.py` and set:
	- `EVALUATE_ONLY = False` to train then evaluate
	- `EVALUATE_ONLY = True` to evaluate existing checkpoints only

Then run:

```bash
python main.py
```

## Outputs

- Policy checkpoints:
	- `policy/Standard RL_final_weights.pth`
	- `policy/SPRL_final_weights.pth`
- Plots:
	- `plot/training_Standard RL.png`
	- `plot/training_SPRL.png`
	- `plot/training_violations.png`
	- `plot/comprehensive_evaluation.png`
	- `plot/am_loss.png` (APN pretraining)

## Current Simulation Configuration

From `env.py` defaults:

- total simulated horizon: `500 h`
- control interval: `10 h`
- max RL steps per episode: `50`

Adjust constants in `env.py` and `main.py` for your experiment design.
