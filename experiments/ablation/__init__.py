"""Ablation study of the Safe RL agent (run with: python ablation.py).

Trains and evaluates variants of the Safe RL agent (agents/safe.py with
envs/safe.py) in which individual components are switched off, to measure what
each one contributes to production and constraint satisfaction.

Components:
    gru         Recurrent memory.  Ablated by zeroing the GRU hidden state
                before every timestep, so the policy is purely reactive while
                keeping the same network, parameter count and initial weights.
    lagrangian  Adaptive Lagrange multipliers in PhycocyaninEnvSafe.  Ablated
                by freezing every multiplier at 0 (no dual ascent).  The static
                shaping (base spikes, buffer zones, G3 anticipation, G5 guiding
                and idle spike) is kept, so only the adaptive part is removed.
    mask        L0, the SERL stage mask in the policy distribution.  Ablated
                by sampling every action dimension unmasked.  select_action()
                still overwrites stage-locked dimensions with their defaults
                before the APN and the plant, which then acts as a plant-side
                interlock.
    proj        L2, the APN projection of every action.  Ablated by sending
                the actor's (stage-masked) action unprojected; the APN can
                still train the actor through the safety gradient.
    grad        The APN's safety gradient in the PPO update (safety loss on
                the actor's mean and mapping penalty).  Ablated by training
                the actor on the reward alone.

    Without proj and grad the APN is gone altogether.

Why the mask ablation keeps the interlock:
    The plant decodes only the dimensions its stage allows (envs/plant.py), and
    the APN is trained on stage-masked actions.  Without the interlock the
    unmasked policy would explore and be credited for dimensions with no
    effect, so the variant would be solving a different control problem.  With
    it, every variant controls the same plant and the comparison is the
    standard invalid-action masking vs. no-op one (Huang & Ontanon, 2020).
    illegal_cmd_rate reports how often the unmasked policy still commands
    stage-illegal actions that only the interlock blocked.

Constraints:
    PhycocyaninEnvSafe's five constraints: G1 path nitrate (CN <= 800 mg/L),
    G2 product ratio (Cq/Cx <= 0.011), G3 nitrate at the Growth -> Harvesting
    transition (CN <= 150 mg/L), G4 reactor overflow (V <= V_MAX) and G5 the
    episode ending in Idle.  Each is tracked by occurrence, the number of
    violation events per episode as the environment counts them and the share
    of episodes with one; G1-G4 also by severity: the worst exceedance over
    all evaluation episodes and the mean exceedance per event, both as a
    percentage of the limit.

Noise (no disturbances):
    Training runs on the nominal plant.  Each trained policy is evaluated on
    the nominal plant three times: without noise, with multiplicative Gaussian
    actuator noise on the light, nitrate feed and outflow sent to the plant
    (after the safety stack, so the projection cannot correct it), and with
    multiplicative Gaussian sensor noise on the measured observations
    (Cx, CN, Cq, V and the nitrate supply), --noise-std, default 5%.  Every
    mode reseeds identically, so the comparison is paired.

Protocol:
    Runs use experiments.config's hyperparameters and LR/entropy schedules with
    a fixed episode budget instead of early stopping, so variants are compared
    at equal sample counts.  For a given seed every variant starts from
    identical initial weights.  Trained policies are evaluated on a fresh
    PhycocyaninEnvSafe with all multipliers frozen at 0, so shaped returns are
    comparable with and without the Lagrangian; the physical KPIs (harvested
    phycocyanin, constraint violations) are the primary metrics.

Outputs (under --out, default ablation/):
    runs/<config>__seed<k>/train_log.csv   per-episode training metrics
    runs/<config>__seed<k>/eval.json       evaluation metrics and run metadata
    runs/<config>__seed<k>/policy.pth      final policy weights
    summary.csv                            mean and std across seeds per
                                           (config, noise mode)
    ablation_curves.png                    learning curves against the full model
    ablation_final.png                     final evaluation metrics per noise mode
    ablation_occurrence.png                violation events per episode by constraint
    ablation_severity.png                  worst and mean exceedance by constraint

Runs with an eval.json are skipped, so re-running the same command resumes an
interrupted study; a run that was interrupted part-way restarts from scratch.
A completed run whose eval.json lacks a noise mode (or was evaluated with a
different --noise-std or --eval-episodes) is re-evaluated from its saved
policy.pth without retraining.

Usage:
    python ablation.py                                # leave-one-out, seeds 0-2
    python ablation.py --configs full no_proj --seeds 0 1 2 3 4
    python ablation.py --design factorial --episodes 5000
    python ablation.py --workers 1                    # one process, progress bars
    python ablation.py --summarize-only               # re-aggregate and re-plot
"""
