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
- `G3` (Harvest-start Nitrate): `CN <= 150 mg/L` when Growth ends and Harvesting begins
- `G4` (Reactor Overflow): `V <= 50 L`
- `G5` (Idle Stage Minimum Volume): `V >= 5 L`

*Note: the APN covers G1, G2, G3 and G4. G1, G2 and G4 are one control step ahead, so their heads are exact: the plant's own kinetics over the step. G3 is decided over many steps: the action must leave it reachable under the fastest possible nitrate draw-down (no feed, slowest clock, growth-optimal light) until Growth ends. A small network (KineticNet) predicts that draw-down step by step from biomass and nitrate, and the step on which Growth ends is the plant's credit arithmetic, computed exactly. G5 is handled by the Lagrangian multipliers and the policy.*

## Architecture

### Safe RL Agent (`agents/safe.py`)
A recurrent PPO agent (GRU + raw-state skip connection, `agents/networks.py`) wrapped in the Fermentation PPO project's safety stack:

*   **L0 SERL Stage Mask**: locks the action dims a stage does not allow, in the policy distribution and again on the action sent.
*   **L1 APN** (`safety/apn.py`, a multi-head APN like Fermentation's): one margin per constraint as a fraction of its limit. G1, G2 and G4 are exact heads, the plant's own one-step kinetics in torch (`safety/dynamics.py`). G3 is the learned head: KineticNet (`safety/kinetic.py`, weights in `policy/apn_kinetic.pth`) predicts the fallback's nitrate draw-down, and the credit arithmetic picks the step where Growth ends.
*   **L2 Projection**: a closed-form, priority-weighted minimum-norm correction of the heads that are short (Inoculation and Growth only), re-linearised a few times, then the containment heads (G1, G4) re-checked alone. A row G3 is still short on takes the closest accepted point toward the G3 fallback.
*   **State estimate** (`safety/estimator.py`): the APN projects on a Kalman-filtered estimate of the measured features, not on the raw reading. The nominal plant (`safety/nominal.py`, the environment's own step) predicts the next reading from the action sent; sensor noise is identified online from the lag-1 correlation of the prediction errors (actuator noise leaves it at 0) and the estimate is moved two of its own standard deviations toward every constraint. The process noise it identifies (an actuator error) widens the G1, G3 and G4 buffers by three of its standard deviations. Without noise the estimate is the reading and the buffers stay static. The actor and the APN's training data stay noise-free.
*   **Safety gradient**: during the PPO update the APN's hinge on the actor's own mean action is backpropagated into the actor, plus a mapping penalty towards the projected action, so it learns to propose actions the projection no longer corrects.
*   **Lagrangian Constraint Shaping**: adaptive penalty multipliers in the environment (`PhycocyaninEnvSafe`) on top, and alone for G5.

Training is Deep Recurrent Policy Optimisation: GAE over recurrent value estimates, Truncated Backpropagation Through Time over 32-step chunks, and Welford running reward normalisation. PPO is trained on the raw intent the actor sampled; `use_safeguard`, `use_projection`, `use_safety_grad` and `use_g3` switch the safety components off for ablations.

### Standard RL Agent (`agents/bench.py`)
A feedforward PPO actor-critic without the projection filter. Relies on reward shaping to learn safety behaviors (`PhycocyaninEnvBench`).

## Project Structure

```
main.py                   run settings + dispatch (python main.py)
ablation.py               ablation study entry point (python ablation.py --help)
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
  apn.py                  multi-head APN: margins, projection, safety loss
  dynamics.py             the plant's one-step kinetics in torch (exact heads)
  kinetic.py              KineticNet, the G3 fallback draw-down (learned head)
  estimator.py            Kalman state estimate the APN projects on
  nominal.py              nominal plant model the estimator predicts with
  data.py                 KineticNet's dataset
  train.py                KineticNet training (python -m safety.train)
experiments/
  config.py               RunConfig and PPO hyperparameters
  train.py / evaluate.py  main.py's training loop and noisy evaluation
  ablation/               study config, ablated agent, runs, summary, plots, CLI
reporting/
  logger.py               DataLogger
  plots.py                training and evaluation plots, comparison table
  gantt.py                stage-schedule Gantt chart (python -m reporting.gantt)
  paths.py                plot output folders
policy/                   checkpoints and evaluation trajectories
plot/                     generated plots
  plot_bench/             Standard RL agent
  plot_safe/              Safe RL agent and APN pretraining
  plot_compare/           both agents side by side
```

## Setup

Use Python 3.10+ (recommended) and install dependencies:

```bash
pip install torch numpy matplotlib pandas tqdm
```

## How To Run

Run every command from the repository root.

1. **Train the APN's G3 head** (KineticNet, a few minutes; the exact heads need no training, and without `policy/apn_kinetic.pth` G3 is not covered)
```bash
python -m safety.train
```

2. **Train and/or Evaluate RL Agents**
Open `main.py` and set:
- `EVALUATE_ONLY = False` to train and then evaluate
- `EVALUATE_ONLY = True` to evaluate existing checkpoints only
- `RUN_BENCHMARK = True/False` to toggle between Standard RL and Safe RL.

PPO hyperparameters live in `experiments/config.py`. Then run:
```bash
python main.py
```

3. **Ablation study** (optional)
```bash
python ablation.py
```
Leave-one-out over the GRU, Lagrangian, stage mask, APN projection, safety gradient and G3 head, seeds 0-2, 3,000 episodes per run. Each policy is evaluated without noise, with actuator noise and with sensor noise (no disturbances). The Standard RL baseline is trained and evaluated alongside (`--no-baseline` leaves it out), and `ablation/ablation_vs_baseline.png` compares it with the full Safe RL agent under all three noise modes. Results land in `ablation/`; `--summarize-only` re-plots and `--help` lists the options.

4. **Compare stage schedules** (after evaluating both agents)
```bash
python -m reporting.gantt
```

## Outputs

Generated plots are saved in sub folders of `plot/`:
- `plot/plot_bench/` (Standard RL): `training_Standard RL.png` and the evaluation trajectories `plot_nitrate_standard.png`, `plot_phycocyanin_standard.png`, `plot_light_standard.png`, `plot_nitrate_feed_standard.png`, `plot_volume_standard.png`, `plot_ratio_standard.png`, `plot_violations_standard.png`
- `plot/plot_safe/` (Safe RL): `training_safe_RL agent.png`, the same evaluation trajectories with the `_safe` suffix, and `am_loss.png` (KineticNet training loss)
- `plot/plot_compare/` (both agents): `training_violations.png`, `comparison_table.png` and `gantt_comparison.png`
