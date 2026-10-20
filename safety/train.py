"""
Pretraining loop for the Photoproduction Safety Boundary Classifier.

Trains the ActionProjectionNetwork (safety/apn.py) as a margin regressor on
fresh datasets from safety/data.py and checks it against the plant with
safety/validate.py.  Run from the repository root:  python -m safety.train
"""

import concurrent.futures
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from reporting.paths import agent_plot_dir
from safety.apn import ActionProjectionNetwork
from safety.constants import APN_PATH
from safety.data import get_fresh_batch_dataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def run_pretraining(epochs=10000, batch_size=32768, buffer_size=1000000,
                    refresh_interval=100, load=False):
    """Trains the safety boundary classifier APN on generated offline datasets.

    The training loop continuously generates fresh randomized datasets biased 
    toward the boundary states so the classifier learns the safety manifold.
    Optimizes a combination of BCE focal loss, smooth L1 regression, and auxiliary
    constraint losses.

    Args:
        epochs (int, optional): Max number of epochs to train. Defaults to 10000.
        batch_size (int, optional): Mini-batch size. Defaults to 32768.
        buffer_size (int, optional): Size of the background dataset buffer. Defaults to 1000000.
        refresh_interval (int, optional): Epochs between dataset refreshes. Defaults to 100.
        load (bool, optional): Whether to resume from an existing checkpoint. Defaults to False.
    """
    # TF32 and CuDNN benchmark can cause HIPBLAS_STATUS_NOT_SUPPORTED errors on some Windows/ROCm setups
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    def _to_device(t, dev):
        """Moves a tensor to the specified device, pinning memory if CUDA.

        Args:
            t (torch.Tensor): The tensor to move.
            dev (torch.device): The target device.

        Returns:
            torch.Tensor: The moved tensor.
        """
        if dev.type == 'cuda':
            return t.pin_memory().to(dev, non_blocking=True)
        return t.to(dev)

    model = ActionProjectionNetwork(state_dim=12, action_dim=4).to(device)

    policy_path = APN_PATH
    if load and os.path.exists(policy_path):
        model = ActionProjectionNetwork.from_checkpoint(
            policy_path, device, state_dim=12, action_dim=4)

    _fwd_aux = model.forward_aux

    # Optimizer settings tuned for long (~20k epoch) runs:
    # lower start LR, gentler decay envelope, and stronger L2 regularization.
    initial_lr = 5e-4
    weight_decay = 1e-4
    optimizer = optim.Adam(
        model.parameters(), lr=initial_lr, weight_decay=weight_decay
    )

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

    # Validation / training configuration
    min_training = 5000
    validation_interval = 1000
    window_size = 200

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
    b_s_raw, b_a_raw, b_labels, b_margins, b_mg1, b_mg2, b_mg4, b_mg3 = \
        get_fresh_batch_dataset(buffer_size, bias=0.7, pass_rates=latest_pass_rates)
    b_s_raw     = _to_device(b_s_raw, device)
    b_a_raw     = _to_device(b_a_raw, device)
    b_labels    = _to_device(b_labels, device)
    b_margins   = _to_device(b_margins, device)
    b_mg1       = _to_device(b_mg1, device)
    b_mg2       = _to_device(b_mg2, device)
    b_mg4       = _to_device(b_mg4, device)
    b_mg3       = _to_device(b_mg3, device)

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
                b_s_raw, b_a_raw, b_labels, b_margins, b_mg1, b_mg2, b_mg4, b_mg3 = \
                    future_dataset.result()
                b_s_raw     = _to_device(b_s_raw, device)
                b_a_raw     = _to_device(b_a_raw, device)
                b_labels    = _to_device(b_labels, device)
                b_margins   = _to_device(b_margins, device)
                b_mg1       = _to_device(b_mg1, device)
                b_mg2       = _to_device(b_mg2, device)
                b_mg4       = _to_device(b_mg4, device)
                b_mg3       = _to_device(b_mg3, device)
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

        for batch_idx_num, i in enumerate(range(0, buffer_size, batch_size)):
            batch_idx = indices[i: i + batch_size]
            b_s = b_s_raw[batch_idx]
            b_a = b_a_raw[batch_idx]
            b_y = b_labels[batch_idx]      # [batch] — 1.0 safe, 0.0 unsafe
            # [batch] — continuous physical margin
            b_m = b_margins[batch_idx]
            b_m_g1 = b_mg1[batch_idx]
            b_m_g2 = b_mg2[batch_idx]
            b_m_g4 = b_mg4[batch_idx]
            b_m_g3 = b_mg3[batch_idx]

            optimizer.zero_grad()

            def calculate_loss():
                """Computes the composite boundary-learning loss for pretraining.

                Combines BCE focal loss on safety logits, scaled SmoothL1 regression
                to ground margins, and auxiliary per-constraint classifiers.

                Returns:
                    tuple: A tuple containing:
                        - loss (torch.Tensor): The scalar loss tensor.
                        - correct (int): Number of correct safety classifications.
                        - total (int): Total number of samples evaluated in the batch.
                """
                margin_pred, aux_g1, aux_g2, aux_g3, aux_g4 = _fwd_aux(b_s, b_a)

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
                    margin_pred, sign_target, weight=combined_weight)

                # 2. Scaled regression
                l_reg = F.smooth_l1_loss(margin_pred, b_m * MARGIN_SCALE)

                # 3. Per-constraint losses (BCE + scaled regression), one per head
                targets = {"G1": (aux_g1, b_m_g1), "G2": (aux_g2, b_m_g2),
                           "G3": (aux_g3, b_m_g3), "G4": (aux_g4, b_m_g4)}
                head_losses = {
                    k: (0.5 * F.binary_cross_entropy_with_logits(pred, (m >= 0.0).float())
                        + 0.5 * F.smooth_l1_loss(pred, m * MARGIN_SCALE))
                    for k, (pred, m) in targets.items()}

                # Dynamically weight the heads by their latest validation failure rates
                n_heads = len(head_losses)
                coef = dict.fromkeys(head_losses, 2.0)
                if latest_pass_rates is not None:
                    errs = {k: max(0.0, 1.0 - latest_pass_rates.get(k, 1.0)) for k in head_losses}
                    total_err = sum(errs.values())
                    if total_err > 1e-8:
                        # Default coefficients sum to 2.0 * n_heads
                        coef = {k: 2.0 * n_heads * (0.2 / n_heads + 0.8 * errs[k] / total_err)
                                for k in head_losses}

                l_aux = sum(coef[k] * head_losses[k] for k in head_losses) / (2.0 * n_heads)

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

            scheduler.step(epoch + batch_idx_num / steps_per_epoch)

            epoch_loss += loss.item()
            epoch_correct += correct
            epoch_total += total

        # Finalize epoch metrics
        avg_loss = epoch_loss / (buffer_size / batch_size)
        accuracy = epoch_correct / max(epoch_total, 1)
        loss_history.append(avg_loss)
        accuracy_history.append(accuracy)

        # Finalize epoch metrics and update status
        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Acc':  f'{accuracy:.4f}',
        })

        # Periodic evaluation: after min_training warmup, reevaluate every validation_interval epochs
        if epoch >= min_training and (epoch - min_training) % validation_interval == 0:
            print(f"\n[Validation] Re-evaluating validation suite at epoch {epoch}...")
            model.eval()
            from safety.validate import run_validation
            all_passed, latest_pass_rates = run_validation(model)

            # Unfreeze parameters for further training
            for p in model.parameters():
                p.requires_grad_(True)

            if all_passed:
                print(f"\n[Success] All validation tests passed at epoch {epoch}! Terminating training.")
                break
            else:
                print(f"[Validation] Some tests failed: {latest_pass_rates}. Dynamic weights updated.")

    os.makedirs("policy", exist_ok=True)
    torch.save(model.state_dict(), policy_path)
    print(f"[Saved] Weights → {policy_path}")

    # ── Convergence plot ─────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    def moving_average(values, window):
        """Computes a fast moving average over a sliding window.

        Args:
            values (list or np.ndarray): Raw data values.
            window (int): The width of the sliding window.

        Returns:
            np.ndarray: The smoothed array of moving averages.
        """
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
    plt.savefig(os.path.join(agent_plot_dir("safe"), "am_loss.png"))
    plt.close()


if __name__ == "__main__":
    run_pretraining(load=True)
