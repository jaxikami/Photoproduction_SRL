"""Training of the APN's learned G3 head (apn.KineticNet).

The APN's exact heads are the plant's own kinetics and are not trained.  The
learned head regresses the G3 fallback's nitrate draw-down (safety/data.py):
nitrate at the end of each of the next DRAWDOWN_STEPS control steps, from
biomass and nitrate.  The loss is a Huber loss on nitrate scaled by the G3
limit, weighted towards the G3 target, where the head's sign is decided.

Usage (from the repository root):
    python -m safety.train      # fit KineticNet -> policy/apn_kinetic.pth
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os

# ─── Third-Party ──────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ─── Local ────────────────────────────────────────────────────────────────────
from reporting.paths import agent_plot_dir
from safety.constants import G3_BUFFER, KINETIC_PATH, N_LIMIT_TERM
from safety.data import make_dataset
from safety.kinetic import KineticNet

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Nitrate within a few mg/L of the G3 target weighs up to 3x more.
TARGET_EMPHASIS = 2.0
TARGET_WIDTH = 10.0
_TARGET = N_LIMIT_TERM * G3_BUFFER


def kinetic_loss(model, cx, cn, profile):
    """Huber loss of KineticNet on one batch (nitrate scaled by the G3 limit)."""
    pred = model(cx, cn)
    w = 1.0 + TARGET_EMPHASIS * torch.exp(-(profile - _TARGET).abs() / TARGET_WIDTH)
    loss = F.smooth_l1_loss(pred / N_LIMIT_TERM, profile / N_LIMIT_TERM, beta=0.02,
                            reduction="none")
    return (loss * w).sum() / w.sum()


@torch.no_grad()
def evaluate(model, cx, cn, profile, batch=65536):
    """Validation metrics: nitrate MAE and max error (mg/L), and G3 sign accuracy."""
    model.eval()
    pred = torch.cat([model(cx[i:i + batch], cn[i:i + batch]) for i in range(0, len(cx), batch)])
    err = (pred - profile).abs()
    near = (profile - _TARGET).abs() < TARGET_WIDTH
    sign = (pred <= _TARGET) == (profile <= _TARGET)
    return {
        "mae": float(err.mean()),
        "max_err": float(err.max()),
        "mae_near_target": float(err[near].mean()) if near.any() else None,
        "g3_sign_acc": float(sign.float().mean()),
        "g3_sign_acc_near_target": float(sign[near].float().mean()) if near.any() else None,
    }


def fit(model, train, steps, lr, batch=4096, weight_decay=1e-5, val=None, log_every=0):
    """Fit KineticNet for ``steps`` minibatch steps (cosine-decayed AdamW).

    Args:
        model: KineticNet (trained in place).
        train: Tuple (cx, cn, profile) of tensors.
        steps: Minibatch steps.
        lr: Peak learning rate.
        batch: Minibatch size.
        weight_decay: AdamW weight decay.
        val: Optional validation tuple, evaluated every ``log_every`` steps.
        log_every: Logging period (0: never).

    Returns:
        A tuple of the last validation metrics (or None) and the loss history.
    """
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.02)
    n = len(train[0])
    metrics, history = None, []
    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, n, (batch,), device=train[0].device)
        loss = kinetic_loss(model, *(t[idx] for t in train))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        history.append(loss.item())
        if log_every and val is not None and (step % log_every == 0 or step == steps):
            metrics = evaluate(model, *val)
            print(f"  step {step:6d} | loss {loss.item():.6f} | nitrate MAE {metrics['mae']:.3f} "
                  f"mg/L (near target {metrics['mae_near_target']:.3f}, max {metrics['max_err']:.2f})"
                  f" | G3 sign {metrics['g3_sign_acc']:.5f} "
                  f"(near target {metrics['g3_sign_acc_near_target']:.4f})", flush=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return metrics, history


def _plot(history, path):
    """Loss curve of the fit."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history, color="#2a78d6", alpha=0.35, linewidth=0.8, label="minibatch loss")
    k = max(len(history) // 100, 1)
    ax.plot(np.convolve(history, np.ones(k) / k, mode="valid"), color="#2a78d6",
            label=f"{k}-step mean")
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Huber loss (nitrate / 150 mg/L)")
    ax.set_title("KineticNet (G3 head) training")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_pretraining(out_path=KINETIC_PATH, n_samples=400000, steps=20000, lr=1e-3,
                    val_frac=0.1, seed=0):
    """Generate the draw-down dataset, train KineticNet on it and save it.

    Args:
        out_path: Checkpoint path.
        n_samples: (Cx, CN) pairs.
        steps: Minibatch steps.
        lr: Peak learning rate.
        val_frac: Held-out share.
        seed: Seed.

    Returns:
        Validation metrics of the saved model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    data = make_dataset(n_samples, seed)
    tensors = tuple(data[k].to(device) for k in ("cx", "cn", "profile"))
    n_val = int(n_samples * val_frac)
    val = tuple(t[:n_val] for t in tensors)
    train = tuple(t[n_val:] for t in tensors)
    print(f"Training KineticNet on {n_samples - n_val:,} pairs ({n_val:,} held out), "
          f"device {device}")

    model = KineticNet().to(device)
    metrics, history = fit(model, train, steps, lr, val=val, log_every=max(steps // 20, 1))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metrics": metrics}, out_path)
    print(f"Saved {out_path}")
    _plot(history, os.path.join(agent_plot_dir("safe"), "am_loss.png"))
    return metrics


if __name__ == "__main__":
    run_pretraining()
