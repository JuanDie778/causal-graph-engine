"""Kermack-McKendrick SEIR continuous ODE simulation."""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any


def seir_ode(t: float, y: np.ndarray, N: float, beta: float, sigma: float, gamma: float) -> np.ndarray:
    """System of ordinary differential equations for SEIR model:

        dS/dt = - beta * S * I / N
        dE/dt =   beta * S * I / N - sigma * E
        dI/dt =   sigma * E - gamma * I
        dR/dt =   gamma * I
    """
    S, E, I, R = y
    dS_dt = -(beta * S * I) / N
    dE_dt = (beta * S * I) / N - sigma * E
    dI_dt = sigma * E - gamma * I
    dR_dt = gamma * I
    return np.array([dS_dt, dE_dt, dI_dt, dR_dt])


def simulate_seir(
    N: float,
    S0: float,
    E0: float,
    I0: float,
    R0_init: float = 0.0,
    beta: float = 0.5,
    sigma: float = 0.2,
    gamma: float = 0.1,
    t_max: float = 160.0,
    steps: int = 1000,
) -> Dict[str, Any]:
    """Simulate the Kermack-McKendrick SEIR model using adaptive Runge-Kutta (RK45).

    Verifies the population conservation invariant:
        S(t) + E(t) + I(t) + R(t) == N  for all t.
    """
    if N <= 0 or S0 < 0 or E0 < 0 or I0 < 0 or R0_init < 0:
        raise ValueError("Population compartments and N must be non-negative, with N > 0.")
    if beta < 0 or sigma < 0 or gamma < 0:
        raise ValueError("Transmission rates must be non-negative.")

    t_eval = np.linspace(0.0, t_max, steps)
    y0 = [S0, E0, I0, R0_init]

    sol = solve_ivp(
        fun=lambda t, y: seir_ode(t, y, N, beta, sigma, gamma),
        t_span=(0.0, t_max),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
    )

    S = sol.y[0]
    E = sol.y[1]
    I = sol.y[2]
    R = sol.y[3]

    # Population conservation check across all time steps
    total_pop = S + E + I + R
    conservation_residuals = np.abs(total_pop - N)
    max_conservation_violation = float(np.max(conservation_residuals))

    # Peak infection and time of peak
    peak_idx = int(np.argmax(I))
    peak_infection = float(I[peak_idx])
    t_peak = float(sol.t[peak_idx])

    return {
        "time": sol.t.tolist(),
        "S": S.tolist(),
        "E": E.tolist(),
        "I": I.tolist(),
        "R": R.tolist(),
        "peak_infection": peak_infection,
        "time_of_peak": t_peak,
        "final_recovered": float(R[-1]),
        "final_susceptible": float(S[-1]),
        "max_conservation_violation": max_conservation_violation,
        "conservation_verified": max_conservation_violation < 1e-3,
    }
