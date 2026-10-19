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

Implemented constraints in the environment (`envs/`):

- `G1` (Path Nitrate): `CN <= 800 mg/L`
- `G2` (Quality Ratio): `Cq/Cx <= 0.011`
- `G3` (Terminal Nitrate): `CN <= 150 mg/L` at episode end
- `G4` (Reactor Overflow): `V <= 50 L`
- `G5` (Idle Stage Minimum Volume): `V >= 5 L`

*Note: APN pretraining covers instantaneous constraints (G1, G2, G4). Constraints G3 and G5 are handled via Lagrangian multipliers and temporal policy behavior.*

## Architecture

### Safe RL Agent (`agents/safe.py`)
A recurrent PPO agent (GRU + raw-state skip connection, `agents/networks.py`) wrapped in a layered safety stack, laid out like the Fermentation PPO project's safe agent:

*   **L0 SERL Stage Mask**: locks the action dims a stage does not allow, in the policy distribution and again on the action sent.
*   **L1 APN** (`safety/apn.py`): one learned margin logit per instantaneous constraint (G1, G2, G4), loaded from `policy/action_projection_network.pth`.
*   **L2 Projection**: sign-gradient ascent on the smooth minimum of the margins until the action is accepted (Inoculation and Growth only).
*   **Safety gradient**: during the PPO update the APN's hinge on the actor's own mean action is backpropagated into the actor, plus a mapping penalty towards the projected action, so it learns to propose actions the projection no longer corrects.
*   **Lagrangian Constraint Shaping**: adaptive penalty multipliers in the environment (`PhycocyaninEnvSafe`) for the temporal constraints G3 and G5.

Training is Deep Recurrent Policy Optimisation: GAE over recurrent value estimates, Truncated Backpropagation Through Time over 32-step chunks, and Welford running reward normalisation. PPO is trained on the raw intent the actor sampled; `use_safeguard`, `use_projection` and `use_safety_grad` switch the safety components off for ablations.

### Standard RL Agent (`agents/bench.py`)
A feedforward PPO actor-critic without the projection filter. Relies on reward shaping to learn safety behaviors (`PhycocyaninEnvBench`).

## Project Structure

```
main.py                   run settings + dispatch (python main.py)
envs/
  kinetics.py             Numba rate equations and RK4 integrator
  plant.py                PhycocyaninEnvCore: state, stage logic, action mask
  safe.py                 PhycocyaninEnvSafe: Lagrangian multipliers
  bench.py                PhycocyaninEnvBench: static penalties (baseline)
agents/
  networks.py             GRU actor-critic (+ device)
  memory.py               rollout buffer and reward normalisation
  safe.py                 Safe PPO agent: safety stack and safety gradient
  bench.py                Standard PPO baseline
  checkpoint.py           save / load policies
safety/                   the Action Projection Network
  constants.py            physical limits and projection settings
  apn.py                  network, projection, safety loss
  data.py                 physics-based dataset generation
  train.py                pretraining (python -m safety.train)
  validate.py             checks against the plant (python -m safety.validate)
experiments/
  config.py               RunConfig and PPO hyperparameters
  train.py / evaluate.py  main.py's training loop and noisy evaluation
reporting/
  logger.py               DataLogger
  plots.py                training and evaluation plots, comparison table
  gantt.py                stage-schedule Gantt chart (python -m reporting.gantt)
policy/                   checkpoints and evaluation trajectories
plot/                     generated plots
```

## Setup

Use Python 3.10+ (recommended) and install dependencies:

```bash
pip install torch numpy matplotlib pandas tqdm
```

## How To Run

Run every command from the repository root.

1. **Pretrain APN** (if you need to generate a new safety filter from scratch)
```bash
python -m safety.train
```

2. **Validate APN Safety Behavior**
```bash
python -m safety.validate
```

3. **Train and/or Evaluate RL Agents**
Open `main.py` and set:
- `EVALUATE_ONLY = False` to train and then evaluate
- `EVALUATE_ONLY = True` to evaluate existing checkpoints only
- `RUN_BENCHMARK = True/False` to toggle between Standard RL and Safe RL.

PPO hyperparameters live in `experiments/config.py`. Then run:
```bash
python main.py
```

4. **Compare stage schedules** (after evaluating both agents)
```bash
python -m reporting.gantt
```

## Outputs

Generated plots are saved in the `plot/` directory, including:
- `training_Standard RL.png` / `training_safe_RL agent.png`
- `training_violations.png`
- Detailed Evaluation Trajectories: `plot_nitrate.png`, `plot_phycocyanin.png`, `plot_light.png`, `plot_nitrate_feed.png`, `plot_volume.png`, `plot_ratio.png`, and `plot_violations.png`.
- `am_loss.png` (APN pretraining convergence)
- `comparison_table.png` and `gantt_comparison.png` (both agents)
