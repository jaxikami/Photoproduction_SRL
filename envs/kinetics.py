"""Photobioreactor kinetics: Numba rate equations and the RK4 integrator."""

import numpy as np
from numba import njit

# =============================================================================
# KINETIC ENGINE (Photobioreactor with Volume Tracking)
# =============================================================================

# Feed stock concentration (mg/L) — concentrated nitrate solution
C_N_STOCK = 3000.0

@njit
def calculate_rates_numba(state, I, Fn, F_out, um_scale=1.0, ynx_scale=1.0, km_scale=1.0):
    """Computes instantaneous kinetic rates for the bioreactor state.

    The formulation accounts for dilution effects from feed inflow
    and product removal via outstream. This function is compiled with Numba
    for fast execution during simulation.

    Args:
        state (np.ndarray): Current physical state [Biomass Cx (g/L),
            Nitrate CN (mg/L), Phycocyanin Cq (mg/L), Volume V (L)].
        I (float): Light intensity (umol/m^2/s).
        Fn (float): Nitrate feed concentration rate (mg/L/h).
        F_out (float): Outstream volumetric flow (L/h).
        um_scale (float, optional): Multiplier on the maximum growth rate um.
        ynx_scale (float, optional): Multiplier on the nitrate yield YNX.
        km_scale (float, optional): Multiplier on the product synthesis rate km.

    Returns:
        np.ndarray: Array of instantaneous rates of change 
            [dCx/dt, dCN/dt, dCq/dt, dV/dt].
    """
    Cx = state[0]
    CN = state[1]
    Cq = state[2]
    V  = max(state[3], 0.1)

    # Static Kinetic Parameters
    um  = 0.0572 * um_scale     # h^-1
    ud  = 0.0        # h^-1 (Death rate)
    KN  = 393.1      # mg/L
    YNX = 504.5 * ynx_scale     # mg/g
    km  = 0.00016 * km_scale    # mg/g/h
    kd  = 0.281      # h^-1
    ks  = 178.9      # umol/m^2/s
    ki  = 447.1      # umol/m^2/s
    ksq = 23.51      # umol/m^2/s
    kiq = 800.0      # umol/m^2/s
    KNp = 16.89      # mg/L

    # Volumetric inflow from concentrated nitrate feed
    F_in_vol = max(0.0, Fn * V / C_N_STOCK)   # L/h

    # Growth photolimitation (Aiba model)
    phi_I  = I / (I + ks + (I**2 / ki))
    # Nutrient availability
    phi_N  = CN / (CN + KN)
    # Product photolimitation
    phi_Iq = I / (I + ksq + (I**2 / kiq))

    growth = um * phi_I * Cx * phi_N

    R = np.zeros(4)
    # dCx/dt: growth – death – dilution from inflow (no biomass in feed)
    R[0] = growth - ud * Cx - F_in_vol * Cx / V
    # dCN/dt: consumption + feed term
    R[1] = -YNX * growth + F_in_vol * (C_N_STOCK - CN) / V
    # dCq/dt: synthesis – degradation – dilution from inflow
    R[2] = km * phi_Iq * Cx - (kd * Cq) / (CN + KNp) - F_in_vol * Cq / V
    # dV/dt: inflow – outflow
    R[3] = F_in_vol - F_out

    return R


@njit
def integrate_rk4(state_init, I, Fn, F_out, dt, n_steps,
                  um_scale=1.0, ynx_scale=1.0, km_scale=1.0):
    """Executes a 4th-order Runge-Kutta integration step for the bioreactor system.

    Advances the physical state of the bioreactor [Cx, CN, Cq, V] over a given
    time period using the defined instantaneous rates.

    Args:
        state_init (np.ndarray): The initial physical state before integration.
        I (float): Light intensity applied during the step.
        Fn (float): Nitrate feed rate applied during the step.
        F_out (float): Outstream flow rate during the step.
        dt (float): Inner integration time step duration (hours).
        n_steps (int): Number of integration steps to execute.
        um_scale, ynx_scale, km_scale (float, optional): Kinetic multipliers
            (calculate_rates_numba); 1.0 is the nominal plant.

    Returns:
        np.ndarray: The updated physical state after integration.
    """
    state = state_init.copy()
    for _ in range(n_steps):
        k1 = calculate_rates_numba(state, I, Fn, F_out, um_scale, ynx_scale, km_scale)
        k2 = calculate_rates_numba(state + 0.5 * dt * k1, I, Fn, F_out,
                                   um_scale, ynx_scale, km_scale)
        k3 = calculate_rates_numba(state + 0.5 * dt * k2, I, Fn, F_out,
                                   um_scale, ynx_scale, km_scale)
        k4 = calculate_rates_numba(state + dt * k3, I, Fn, F_out,
                                   um_scale, ynx_scale, km_scale)
        state += (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        # Clamp non-physical values
        state[0] = max(state[0], 0.0)
        state[1] = max(state[1], 0.0)
        state[2] = max(state[2], 0.0)
        state[3] = max(state[3], 0.1)
    return state
