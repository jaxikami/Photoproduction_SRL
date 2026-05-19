"""
Pretraining loop for the Photoproduction Safety Boundary Classifier.

Trains the ActionProjectionNetwork as a margin regressor covering:
  G1 — Nitrate path constraint  (cN + Fn*10h ≤ 800 mg/L)
  G2 — Product/biomass ratio    (cq/cx ≤ 0.011)
  G4 — Reactor overflow         (V ≤ 50 L)
  G5 — Reactor underflow        (V ≥ 5 L)

Note: G3 (terminal nitrate ≤ 150 mg/L) is handled by the GRU's temporal
context and the Lagrangian multiplier, not by the APN.

Architecture (11D state + 4D action = 15 input):
  Linear(15→64) → LayerNorm → Mish
  → 1× Residual block (64→64)
  → Linear(64→1) [raw margin]
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from data_gen import get_fresh_batch_dataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ActionProjectionNetwork(nn.Module):
    """
    Safety Proximity Regressor for the multi-stage photobioreactor.

    Given a (state, action) pair, outputs a continuous signed margin score:
      margin > 0  →  safe
      margin < 0  →  unsafe
      margin = 0  →  on the safety boundary

    Architecture: wider (128-dim) with 2 residual blocks for better
    boundary discrimination across all 4 constraints.
    """

    def __init__(self, state_dim: int = 11, action_dim: int = 4,
                 latent_dim: int = 128):
        super(ActionProjectionNetwork, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Mish()
        )

        # Two residual blocks for depth
        self.res_block1 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Mish(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.res_block2 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Mish(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.res_act = nn.Mish()

        self.margin_head = nn.Linear(latent_dim, 1)

        nn.init.zeros_(self.margin_head.weight)
        nn.init.zeros_(self.margin_head.bias)

    def forward(self, state_norm, action_norm):
        x_in = torch.cat([state_norm, action_norm], dim=-1)
        x_enc = self.encoder(x_in)
        x_enc = self.res_act(x_enc + self.res_block1(x_enc))
        x_enc = self.res_act(x_enc + self.res_block2(x_enc))
        margin = self.margin_head(x_enc).squeeze(-1)
        return margin

    def classify(self, state_norm, action_norm):
        return torch.sigmoid(self.forward(state_norm, action_norm))


def run_pretraining(epochs=100000, batch_size=32768, buffer_size=1000000,
                    refresh_interval=100, load=False):
    """
    Trains the safety boundary classifier via Binary Cross-Entropy on
    mass-balance simulation labels.

    The training loop continuously generates fresh randomized datasets
    (biased toward boundary states) so the classifier generalizes across
    the full state-action space rather than memorizing a fixed dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ActionProjectionNetwork(state_dim=11, action_dim=4).to(device)

    policy_path = os.path.join("policy", "action_projection_network.pth")
    if load and os.path.exists(policy_path):
        model.load_state_dict(torch.load(policy_path, map_location=device,
                                         weights_only=True))
        print(f"[Load] Resumed from checkpoint: {policy_path}")

    # Optimizer settings tuned for long (~20k epoch) runs:
    # lower start LR, gentler decay envelope, and stronger L2 regularization.
    initial_lr = 5e-4
    weight_decay = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=initial_lr,
                           weight_decay=weight_decay)

    # ── Loss function design ────────────────────────────────────────────────
    # Physical margins from data_gen are normalised to the constraint limit.
    # Training on raw margins gives SmoothL1 residuals that stall learning and
    # prevent the model output from reaching the 0.95 inference threshold used
    # in _project_to_safe.
    #
    # Two complementary components solve this:
    #
    #   BCE component: uses margin_pred directly as a logit.
    #     sign(b_m) → 1.0 (safe) / 0.0 (unsafe) target.
    #     Scale-invariant: gradient is always O(1) regardless of margin magnitude.
    #     Explicitly maximises gradient at margin=0, the decision boundary.
    #     Forces outputs into sigmoid-useful range so classify() works correctly.
    #
    #   Scaled regression component: SmoothL1(margin_pred, b_m * MARGIN_SCALE).
    #     Multiplying targets by MARGIN_SCALE maps small physical margins to a
    #     range giving meaningful SmoothL1 residuals and calibrating the output
    #     magnitude so sigmoid(margin_pred) >= 0.95 for comfortable safe actions.
    #
    #   Boundary-emphasis weights: 1 / (|b_m| * MARGIN_SCALE + 0.5)
    #     Upweights samples near margin=0 where the decision is uncertain and the
    #     gradient signal is most informative. Deep safe/unsafe samples are already
    #     easy and get lower weight.

    MARGIN_SCALE = 30.0

    steps_per_epoch = buffer_size // batch_size

    # Applies a CosineAnnealingWarmRestarts scheduler to inject periodic learning
    # rate spikes aligned with dataset refresh intervals to aggressively hyper-explore
    # newly loaded sample topologies.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=refresh_interval, T_mult=3, eta_min=1e-5
    )

    # Apply a mild long-horizon decay envelope before warm restarts.
    decay_factor = 0.995
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor
    scheduler.base_lrs = [group['lr'] for group in optimizer.param_groups]

    loss_history = []
    accuracy_history = []

    # Stopping thresholds: accuracy here = fraction of samples where the predicted
    # margin sign matches the ground-truth margin sign (safe vs unsafe agreement).
    early_stop_threshold = 0.993    # Target 99.3% sign-accuracy
    required_success_per_buffer = 3000     # For 3000 consecutive epochs
    buffer_success_count = 0

    # Anti-stall plateau tracking
    best_moving_avg = float('inf')
    plateau_counter = 0
    patience = 3000
    window_size = 200
    improvement_threshold = 1e-4
    min_training = 10000

    pbar = tqdm(range(epochs), desc="Training Safety Classifier")

    # AMP (Automatic Mixed Precision) for GPU throughput
    if device.type == 'cuda':
        scaler = GradScaler('cuda')
    else:
        scaler = None

    # Asynchronous data generation background worker
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future_dataset = None

    # Initial explicit dataset spawn
    b_s_raw, b_a_raw, b_labels, b_margins = get_fresh_batch_dataset(
        buffer_size, bias=0.7)
    b_s_raw = b_s_raw.to(device)
    b_a_raw = b_a_raw.to(device)
    b_labels = b_labels.to(device)
    b_margins = b_margins.to(device)

    # Print class balance diagnostics for the initial dataset
    safe_ratio = b_labels.mean().item()
    print(
        f"[Data] Initial class balance: {safe_ratio:.2%} safe, {1-safe_ratio:.2%} unsafe")

    for epoch in pbar:
        # Dynamic Dataset Replacement Phase
        if epoch % refresh_interval == 0:
            # Retains `buffer_success_count` persistently across background dataset refreshes.
            # This rigorously validates genuine generalization by demanding >99.3% classification
            # accuracy continuously across completely distinct batches of randomized sample points.

            if epoch > 0 and future_dataset is not None:
                b_s_raw, b_a_raw, b_labels, b_margins = future_dataset.result()
                b_s_raw = b_s_raw.to(device)
                b_a_raw = b_a_raw.to(device)
                b_labels = b_labels.to(device)
                b_margins = b_margins.to(device)
                safe_ratio = b_labels.mean().item()
                print(
                    f"\n[Data] Refreshed class balance at epoch {epoch}: {safe_ratio:.2%} safe, {1-safe_ratio:.2%} unsafe")

            # Spawn the next required dataset batch asynchronously
            future_dataset = executor.submit(
                get_fresh_batch_dataset, buffer_size, 0.7)

        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        indices = torch.randperm(buffer_size, device=device)

        for i in range(0, buffer_size, batch_size):
            batch_idx = indices[i: i + batch_size]
            b_s = b_s_raw[batch_idx]
            b_a = b_a_raw[batch_idx]
            b_y = b_labels[batch_idx]      # [batch] — 1.0 safe, 0.0 unsafe
            # [batch] — continuous physical margin
            b_m = b_margins[batch_idx]

            optimizer.zero_grad()

            def calculate_loss():
                """
                Composite boundary-learning loss with two components:

                1. BCE (boundary signal, primary, 60% weight):
                   Uses margin_pred as a logit against the hard safe/unsafe class.
                   Gradient is always O(1) — no stalling from tiny margin values.
                   Boundary-emphasis weights concentrate learning at margin=0.

                2. Scaled regression (magnitude calibration, secondary, 40% weight):
                   SmoothL1(margin_pred, b_m * MARGIN_SCALE).
                   Calibrates the output range so sigmoid(margin_pred) maps to
                   the 0.95 inference threshold correctly for comfortable safe actions.

                Sign-accuracy: fraction where predicted sign matches ground truth.
                """
                margin_pred = model(b_s, b_a)

                # Hard class target from physical margin sign
                sign_target = (b_m >= 0.0).float()

                # Boundary-emphasis: upweight samples near margin=0 (most uncertain).
                # Normalise so the mean weight is 1, preserving effective batch size.
                boundary_weight = 1.0 / (b_m.abs() * MARGIN_SCALE + 0.5)
                boundary_weight = boundary_weight / \
                    (boundary_weight.mean() + 1e-8)

                # 1. BCE component: trains the safe/unsafe boundary
                l_bce = F.binary_cross_entropy_with_logits(
                    margin_pred, sign_target, weight=boundary_weight)

                # 2. Scaled regression: calibrates margin magnitude
                l_reg = F.smooth_l1_loss(margin_pred, b_m * MARGIN_SCALE)

                loss = 0.6 * l_bce + 0.4 * l_reg
                pred_safe = (margin_pred.detach() >= 0.0)
                truth_safe = (b_m >= 0.0)
                correct = (pred_safe == truth_safe).sum().item()
                total = b_m.size(0)
                return loss, correct, total

            # Execute backward with FP16 scaling where hardware-available
            if scaler is not None:
                with autocast(device_type='cuda'):
                    loss, correct, total = calculate_loss()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, correct, total = calculate_loss()
                loss.backward()
                optimizer.step()

            scheduler.step(epoch + i / steps_per_epoch)

            epoch_loss += loss.item()
            epoch_correct += correct
            epoch_total += total

        # Finalize epoch metrics
        avg_loss = epoch_loss / (buffer_size / batch_size)
        accuracy = epoch_correct / max(epoch_total, 1)
        loss_history.append(avg_loss)
        accuracy_history.append(accuracy)

        # Plateau Logic: Only check after minimum warmup phase
        if len(loss_history) >= window_size and epoch >= min_training:
            current_moving_avg = np.mean(loss_history[-window_size:])

            if current_moving_avg < best_moving_avg * (1 - improvement_threshold):
                best_moving_avg = current_moving_avg
                plateau_counter = 0
            else:
                plateau_counter += 1

        # Convergence check (sustained high accuracy)
        if epoch >= min_training and accuracy >= early_stop_threshold:
            buffer_success_count += 1
        else:
            buffer_success_count = 0

        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Acc':  f'{accuracy:.4f}',
            'Patience': f'{plateau_counter}/{patience}',
        })

        if buffer_success_count >= required_success_per_buffer:
            print(
                f"\n[Success] Safety Classifier reached accuracy threshold. Running full validation suite...")
            model.eval()
            from validation import run_validation
            all_passed = run_validation(model)

            # Unfreeze for potential further training
            for p in model.parameters():
                p.requires_grad_(True)

            if all_passed:
                print(
                    f"\n[Success] All validation tests passed at epoch {epoch}! Terminating training.")
                break
            else:
                print(f"\n[Validation] Tests failed. Continuing training...")
                buffer_success_count = 0

        # Also periodically run validation to catch successes even if accuracy threshold isn't fully met
        elif epoch > min_training and epoch % 1000 == 0:
            print(
                f"\n[Validation] Periodic validation check at epoch {epoch}...")
            model.eval()
            from validation import run_validation
            all_passed = run_validation(model)

            for p in model.parameters():
                p.requires_grad_(True)

            if all_passed:
                print(
                    f"\n[Success] All validation tests passed at epoch {epoch}! Terminating training.")
                break
            else:
                print(f"\n[Validation] Tests failed. Continuing training...")

        if epoch >= min_training and plateau_counter >= patience:
            print(
                f"\n[Plateau] No 0.1% improvement for {patience} epochs. Terminating.")
            break

    os.makedirs("policy", exist_ok=True)
    torch.save(model.state_dict(), policy_path)
    print(f"[Saved] Weights → {policy_path}")

    # ── Convergence plot ─────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    def moving_average(values, window):
        values_np = np.asarray(values, dtype=np.float64)
        cumulative = np.cumsum(values_np)
        smoothed = np.empty_like(values_np)
        for idx in range(len(values_np)):
            start = max(0, idx - window + 1)
            total = cumulative[idx] - \
                (cumulative[start - 1] if start > 0 else 0.0)
            smoothed[idx] = total / (idx - start + 1)
        return smoothed

    smoothed_loss = moving_average(loss_history,     window_size)
    smoothed_acc = moving_average(accuracy_history, window_size)
    epochs_range = range(len(loss_history))

    line1 = ax1.plot(epochs_range, smoothed_loss, color='blue',
                     label=f'BCE Loss ({window_size}-Ep MA)')
    line2 = ax2.plot(epochs_range, smoothed_acc,  color='green',
                     label=f'Accuracy ({window_size}-Ep MA)')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("BCE Loss", color='blue')
    ax2.set_ylabel("Accuracy", color='green')
    ax1.set_yscale('log')
    plt.title("Safety Classifier Convergence")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    os.makedirs("plot", exist_ok=True)
    plt.savefig(os.path.join("plot", "am_loss.png"))
    plt.close()


if __name__ == "__main__":
    run_pretraining(load=False)
