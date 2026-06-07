"""
Pretraining loop for the Photoproduction Safety Boundary Classifier.

Trains the ActionProjectionNetwork as a margin regressor covering:
  G1 — Nitrate path constraint  (cN + Fn*10h ≤ 800 mg/L)
  G2 — Product/biomass ratio    (cq/cx ≤ 0.011)
  G4 — Reactor overflow         (V ≤ 50 L)

Note: G3 (terminal nitrate ≤ 150 mg/L) is handled by the GRU's temporal
context and the Lagrangian multiplier, not by the APN.

Architecture (FiLM-conditioned, spectral-normalized):
  State encoder: Linear(12→256) → LN → Mish → Linear(256→256) → LN → Mish
  FiLM generators: 3× Linear(256→512) producing (γ,β) pairs
  Action trunk: Linear(4→256) → 3× [FC → LN → FiLM → Mish] (spectral-normed)
  → Linear(256→1) [raw margin]
  + 3× per-constraint auxiliary heads (train-only)
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
    """Safety Proximity Regressor for the multi-stage photobioreactor.

    Given a (state, action) pair, outputs a continuous signed margin score:
      - margin > 0: Action is safe.
      - margin < 0: Action is unsafe.
      - margin = 0: Action is precisely on the safety boundary.

    Architecture (FiLM-conditioned with spectral normalization):
        State encoder: Linear(12→256) → Mish → Linear(256→256) → Mish
        State → FiLM params: 3× (γ, β) pairs for action trunk modulation
        Action trunk: Linear(4→256) → [FiLM block]×3 → Linear(256→1)
        Spectral norm on action-path layers for Lipschitz-bounded ∂margin/∂action.
        Also includes 3 per-constraint auxiliary heads used for APN pretraining.
    """

    def __init__(self, state_dim: int = 12, action_dim: int = 4, latent_dim: int = 256):
        """Initializes the Action Projection Network with FiLM conditioning.

        The state is encoded separately and produces scale/shift (γ, β) parameters
        that modulate the action processing trunk. This yields cleaner ∂margin/∂action
        gradients for the projection step. Spectral normalization ensures Lipschitz
        continuity for stable gradient ascent convergence.

        Args:
            state_dim (int, optional): Dimension of observation space. Defaults to 12.
            action_dim (int, optional): Dimension of action space. Defaults to 4.
            latent_dim (int, optional): Width of hidden layers. Defaults to 256.
        """
        super(ActionProjectionNetwork, self).__init__()
        self.latent_dim = latent_dim

        # ── State encoder (no spectral norm — not differentiated during projection)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Mish(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Mish(),
        )

        # FiLM generators: state → (γ, β) for each of 3 action trunk blocks.
        # Zero-initialised so all generators start as identity (γ=0, β=0),
        # preventing activation explosions before the first gradient update.
        self.film1 = nn.Linear(latent_dim, latent_dim * 2)
        self.film2 = nn.Linear(latent_dim, latent_dim * 2)
        self.film3 = nn.Linear(latent_dim, latent_dim * 2)
        for _film in (self.film1, self.film2, self.film3):
            nn.init.zeros_(_film.weight)
            nn.init.zeros_(_film.bias)

        # ── Action trunk (spectral-normed for Lipschitz-bounded action gradients)
        self.action_lift = nn.utils.spectral_norm(nn.Linear(action_dim, latent_dim))

        self.trunk_fc1 = nn.utils.spectral_norm(nn.Linear(latent_dim, latent_dim))
        self.trunk_ln1 = nn.LayerNorm(latent_dim)
        self.trunk_fc2 = nn.utils.spectral_norm(nn.Linear(latent_dim, latent_dim))
        self.trunk_ln2 = nn.LayerNorm(latent_dim)
        self.trunk_fc3 = nn.utils.spectral_norm(nn.Linear(latent_dim, latent_dim))
        self.trunk_ln3 = nn.LayerNorm(latent_dim)

        self.act = nn.Mish()

        # ── Output heads
        # spectral_norm wraps weight_orig — do NOT zero-init weight_orig as
        # σ = u^T W v → 0 causes NaN (division by zero in the SN forward hook).
        self.margin_head = nn.utils.spectral_norm(nn.Linear(latent_dim, 1))
        nn.init.zeros_(self.margin_head.bias)

        # Per-constraint auxiliary heads (train-only, direct gradient signal)
        self.g1_head = nn.Linear(latent_dim, 1)
        self.g2_head = nn.Linear(latent_dim, 1)
        self.g4_head = nn.Linear(latent_dim, 1)

    def _encode_state(self, state_norm):
        """Encodes state into FiLM conditioning parameters."""
        h_s = self.state_encoder(state_norm)
        film1 = self.film1(h_s)
        film2 = self.film2(h_s)
        film3 = self.film3(h_s)
        return film1, film2, film3

    def _apply_film(self, x, film_params):
        """Applies FiLM: x * (1 + γ) + β."""
        gamma, beta = film_params.chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta

    def _action_trunk(self, action_norm, film1, film2, film3):
        """Processes action through FiLM-modulated trunk."""
        h = self.act(self.action_lift(action_norm))

        h = self.trunk_fc1(h)
        h = self.trunk_ln1(h)
        h = self._apply_film(h, film1)
        h = self.act(h)

        h = self.trunk_fc2(h)
        h = self.trunk_ln2(h)
        h = self._apply_film(h, film2)
        h = self.act(h)

        h = self.trunk_fc3(h)
        h = self.trunk_ln3(h)
        h = self._apply_film(h, film3)
        h = self.act(h)

        return h

    def forward(self, state_norm, action_norm):
        """Computes the continuous signed safety margin for the given state and action.

        Args:
            state_norm (torch.Tensor): Normalized observation vector.
            action_norm (torch.Tensor): Action vector.

        Returns:
            torch.Tensor: Continuous margin score.
        """
        film1, film2, film3 = self._encode_state(state_norm)
        h = self._action_trunk(action_norm, film1, film2, film3)
        margin = self.margin_head(h).squeeze(-1)
        return margin

    def forward_aux(self, state_norm, action_norm):
        """Forward pass returning main margin and per-constraint auxiliary margins.

        Args:
            state_norm (torch.Tensor): Normalized observation vector.
            action_norm (torch.Tensor): Action vector.

        Returns:
            tuple: (margin, g1_margin, g2_margin, g4_margin)
        """
        film1, film2, film3 = self._encode_state(state_norm)
        h = self._action_trunk(action_norm, film1, film2, film3)
        margin = self.margin_head(h).squeeze(-1)
        g1 = self.g1_head(h).squeeze(-1)
        g2 = self.g2_head(h).squeeze(-1)
        g4 = self.g4_head(h).squeeze(-1)
        return margin, g1, g2, g4

    def classify(self, state_norm, action_norm):
        """Returns the probability that the state-action pair is safe (margin > 0).

        Args:
            state_norm (torch.Tensor): Normalized observation vector.
            action_norm (torch.Tensor): Action vector.

        Returns:
            torch.Tensor: Safety probability [0, 1].
        """
        return torch.sigmoid(self.forward(state_norm, action_norm))

    @classmethod
    def from_checkpoint(cls, path: str, device: torch.device, **kwargs) -> "ActionProjectionNetwork":
        """Instantiate an APN and load a checkpoint with shape-safe partial loading.

        Keys present in the checkpoint but with a shape mismatch against the
        current architecture are silently skipped, so weights always load without
        crashing across architecture changes (e.g. latent_dim 160 → 256).

        Args:
            path (str): Path to the .pth checkpoint file.
            device (torch.device): Target device.
            **kwargs: Forwarded to __init__ (e.g. state_dim, action_dim, latent_dim).

        Returns:
            ActionProjectionNetwork: Model with compatible weights loaded.
        """
        model = cls(**kwargs).to(device)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        current = model.state_dict()
        compatible = {k: v for k, v in ckpt.items()
                      if k in current and current[k].shape == v.shape}
        skipped = len(ckpt) - len(compatible)
        model.load_state_dict(compatible, strict=False)
        print(f"[APN] Loaded '{path}' — {len(compatible)} tensors restored, "
              f"{skipped} skipped (shape mismatch / new layers).")
        return model


# Keep as a module-level alias so existing call sites in validation/safe_agent
# that import load_compatible_checkpoint still work without changes.
def load_compatible_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    current = model.state_dict()
    compatible = {k: v for k, v in ckpt.items()
                  if k in current and current[k].shape == v.shape}
    skipped = len(ckpt) - len(compatible)
    model.load_state_dict(compatible, strict=False)
    return len(compatible), skipped


def run_pretraining(epochs=100000, batch_size=32768, buffer_size=1000000,
                    refresh_interval=100, load=False):
    """Trains the safety boundary classifier APN on generated offline datasets.

    The training loop continuously generates fresh randomized datasets biased 
    toward the boundary states so the classifier learns the safety manifold.
    Optimizes a combination of BCE focal loss, smooth L1 regression, and auxiliary
    constraint losses.

    Args:
        epochs (int, optional): Max number of epochs to train. Defaults to 100000.
        batch_size (int, optional): Mini-batch size. Defaults to 32768.
        buffer_size (int, optional): Size of the background dataset buffer. Defaults to 1000000.
        refresh_interval (int, optional): Epochs between dataset refreshes. Defaults to 100.
        load (bool, optional): Whether to resume from an existing checkpoint. Defaults to False.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ActionProjectionNetwork(state_dim=12, action_dim=4).to(device)

    policy_path = os.path.join("policy", "action_projection_network.pth")
    if load and os.path.exists(policy_path):
        model = ActionProjectionNetwork.from_checkpoint(
            policy_path, device, state_dim=12, action_dim=4)

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
    patience = 2000
    window_size = 200
    improvement_threshold = 1e-4
    min_training = 5000

    pbar = tqdm(range(epochs), desc="Training Safety Classifier")

    # AMP (Automatic Mixed Precision) for GPU throughput
    if device.type == 'cuda':
        scaler = GradScaler('cuda')
    else:
        scaler = None

    # Asynchronous data generation background worker
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future_dataset = None
    latest_pass_rates = None

    # Initial explicit dataset spawn
    b_s_raw, b_a_raw, b_labels, b_margins, b_mg1, b_mg2, b_mg4 = \
        get_fresh_batch_dataset(buffer_size, bias=0.7, pass_rates=latest_pass_rates)
    b_s_raw = b_s_raw.to(device)
    b_a_raw = b_a_raw.to(device)
    b_labels = b_labels.to(device)
    b_margins = b_margins.to(device)
    b_mg1 = b_mg1.to(device)
    b_mg2 = b_mg2.to(device)
    b_mg4 = b_mg4.to(device)

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
                b_s_raw, b_a_raw, b_labels, b_margins, b_mg1, b_mg2, b_mg4 = \
                    future_dataset.result()
                b_s_raw = b_s_raw.to(device)
                b_a_raw = b_a_raw.to(device)
                b_labels = b_labels.to(device)
                b_margins = b_margins.to(device)
                b_mg1 = b_mg1.to(device)
                b_mg2 = b_mg2.to(device)
                b_mg4 = b_mg4.to(device)
                safe_ratio = b_labels.mean().item()
                print(
                    f"\n[Data] Refreshed class balance at epoch {epoch}: {safe_ratio:.2%} safe, {1-safe_ratio:.2%} unsafe")

            # Spawn the next required dataset batch asynchronously
            future_dataset = executor.submit(
                get_fresh_batch_dataset, buffer_size, 0.7, latest_pass_rates)

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
            b_m_g1 = b_mg1[batch_idx]
            b_m_g2 = b_mg2[batch_idx]
            b_m_g4 = b_mg4[batch_idx]

            optimizer.zero_grad()

            def calculate_loss():
                """
                Composite boundary-learning loss with three components:

                1. BCE with focal modulation (boundary signal, primary):
                   Uses margin_pred as a logit against the hard safe/unsafe class.
                   Focal weighting concentrates on hard-to-classify boundary samples.

                2. Scaled regression (magnitude calibration):
                   SmoothL1(margin_pred, b_m * MARGIN_SCALE).

                3. Per-constraint auxiliary BCE (G2/G5 upweighted 1.5×):
                   Each aux head gets direct gradient from its own constraint.
                """
                margin_pred, aux_g1, aux_g2, aux_g4 = model.forward_aux(b_s, b_a)

                # Hard class target from physical margin sign
                sign_target = (b_m >= 0.0).float()

                # Boundary-emphasis weights
                boundary_weight = 1.0 / (b_m.abs() * MARGIN_SCALE + 0.5)
                boundary_weight = boundary_weight / \
                    (boundary_weight.mean() + 1e-8)

                # Focal-loss modulation: (1 - p_t)^gamma downweights easy samples
                with torch.no_grad():
                    p_pred = torch.sigmoid(margin_pred)
                    p_t = sign_target * p_pred + (1.0 - sign_target) * (1.0 - p_pred)
                    focal_weight = (1.0 - p_t) ** 2.0  # gamma=2
                    focal_weight = focal_weight / (focal_weight.mean() + 1e-8)

                combined_weight = boundary_weight * focal_weight

                # 1. Focal BCE component
                l_bce = F.binary_cross_entropy_with_logits(
                    margin_pred, b_y, weight=combined_weight)

                # 2. Scaled regression
                l_reg = F.smooth_l1_loss(margin_pred, b_m * MARGIN_SCALE)

                # 3. Per-constraint auxiliary BCE losses
                target_g1 = (b_m_g1 >= 0.0).float()
                target_g2 = (b_m_g2 >= 0.0).float()
                target_g4 = (b_m_g4 >= 0.0).float()

                l_aux_g1 = F.binary_cross_entropy_with_logits(aux_g1, target_g1)
                l_aux_g2 = F.binary_cross_entropy_with_logits(aux_g2, target_g2)
                l_aux_g4 = F.binary_cross_entropy_with_logits(aux_g4, target_g4)

                l_aux = (1.5 * l_aux_g1 + 2.0 * l_aux_g2 + 1.5 * l_aux_g4) / 5.0

                loss = 0.4 * l_bce + 0.25 * l_reg + 0.35 * l_aux

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
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, correct, total = calculate_loss()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            all_passed, latest_pass_rates = run_validation(model)

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
            all_passed, latest_pass_rates = run_validation(model, num_test_samples=500)

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
    run_pretraining(load=True)
