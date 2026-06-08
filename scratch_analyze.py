import torch
import numpy as np
import os
import sys

sys.path.append(os.getcwd())

from pretrain import ActionProjectionNetwork
from env_core import PhycocyaninEnvCore
from validation import _make_state

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = "policy/action_projection_network.pth"
if not os.path.exists(ckpt):
    ckpt = "action_projection_network.pth"
if not os.path.exists(ckpt):
    print("No checkpoint found.")
    sys.exit(1)

apn = ActionProjectionNetwork.from_checkpoint(ckpt, device, state_dim=12, action_dim=4)
apn.eval()

V_MAX = 20.0
SAFE_BUFFER = 0.98
FN_MAX_GROWTH = 40.0
C_N_STOCK = 3000.0
CONTROL_INTERVAL = 10.0
RATIO_LIMIT = 0.011

def test_projection(threshold, max_steps=5):
    # Standard sign-based projection with the dynamic multiplier we just added
    g4_passes = 0
    num_samples = 1000
    
    for _ in range(num_samples):
        V  = float(torch.empty(1).uniform_(V_MAX * 0.85, V_MAX * SAFE_BUFFER))
        cx = float(torch.empty(1).uniform_(0.5, 5.0))
        cN = float(torch.empty(1).uniform_(0.0, 400.0))
        cq = float(torch.empty(1).uniform_(0.0, cx * RATIO_LIMIT * 0.8))
        t  = float(torch.empty(1).uniform_(0.0, 0.5))

        s_t = _make_state(cx, cN, cq, V, stage_idx=0, credit_norm=0.5,
                          t_norm=t, supply_norm=0.5, device=device)
        a_t = torch.ones(1, 4, device=device)  # max feed

        # Run projection loop locally with specified threshold
        state_fixed = s_t.detach()
        stage_mask = PhycocyaninEnvCore.get_action_mask(state_fixed)
        default_squashed = torch.tensor([-0.333, -1.0, -1.0, -1.0], device=device)
        
        a = a_t.clone().detach() * stage_mask + default_squashed * (1 - stage_mask)
        
        with torch.no_grad():
            p = apn.classify(state_fixed, a)
            if p.item() >= threshold:
                a_proj = a
            else:
                best_a = a.clone()
                best_margin = apn(state_fixed, a).item()
                
                with torch.enable_grad():
                    for step in range(max_steps):
                        a_var = a.clone().requires_grad_(True)
                        margin = apn(state_fixed, a_var)
                        
                        p_val = torch.sigmoid(margin)
                        if p_val.item() >= threshold:
                            best_a = a_var.detach().clone()
                            break
                            
                        m_val = margin.item()
                        if m_val > best_margin:
                            best_margin = m_val
                            best_a = a_var.detach().clone()
                            
                        grad = torch.autograd.grad(margin.sum(), a_var)[0]
                        with torch.no_grad():
                            grad = grad.clone() * stage_mask
                            at_lower = (a_var.data <= -0.9999) & (grad < 0)
                            at_upper = (a_var.data >=  0.9999) & (grad > 0)
                            grad[at_lower | at_upper] = 0.0
                            
                            step_size = 0.5 / (1.0 + step * 0.03)
                            
                            lr_mult = torch.ones_like(grad)
                            is_high_vol = state_fixed[..., 3] > 0.75
                            lr_mult[..., 2] = torch.where(is_high_vol, torch.tensor(3.0, device=grad.device), torch.tensor(1.0, device=grad.device))
                            
                            a = a + step_size * grad.sign() * lr_mult
                            a = a.clamp(-1.0, 1.0)
                
                a_proj = best_a * stage_mask + default_squashed * (1 - stage_mask)

        # Physics
        a_scaled = (a_proj[0] + 1.0) / 2.0
        Fn_phys = a_scaled[2].item() * FN_MAX_GROWTH
        F_in_vol = Fn_phys * V / C_N_STOCK
        V_next = V + F_in_vol * CONTROL_INTERVAL

        if V_next <= V_MAX * SAFE_BUFFER:
            g4_passes += 1
            
    return g4_passes / num_samples

for thresh in [0.5, 0.711, 0.80, 0.85, 0.90, 0.92, 0.95]:
    rate = test_projection(thresh)
    print(f"Threshold: {thresh:.3f} | Pass rate: {rate:.1%}")
