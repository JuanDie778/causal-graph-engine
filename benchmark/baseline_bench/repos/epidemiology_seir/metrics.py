"""Epidemiological metrics and counterfactual intervention analysis."""
from __future__ import annotations
from typing import Dict, Any
from .seir_model import simulate_seir


def calculate_r0(beta: float, gamma: float) -> float:
    """Calculate the basic reproduction number R_0.

    Formula:
        R_0 = beta / gamma
    """
    if gamma <= 0:
        raise ValueError("Recovery rate gamma must be strictly positive.")
    return beta / gamma


def calculate_herd_immunity_threshold(r0: float) -> float:
    """Calculate the theoretical herd immunity threshold (HIT).

    Formula:
        HIT = max(0.0, 1.0 - 1.0 / R_0)
    """
    if r0 <= 0:
        raise ValueError("R_0 must be strictly positive.")
    return max(0.0, 1.0 - (1.0 / r0))


def calculate_intervention_impact(
    base_beta: float,
    intervention_beta: float,
    sigma: float,
    gamma: float,
    N: float,
    S0: float,
    E0: float,
    I0: float,
    t_intervention: float = 30.0,
    t_max: float = 160.0,
) -> Dict[str, Any]:
    """Evaluate the impact of a surgical counterfactual intervention do(beta <- beta_intervention).

    Simulates the unmitigated baseline, then simulates phase 1 (pre-intervention)
    and continues with phase 2 under the modified beta.
    """
    # 1. Unmitigated baseline
    base_sim = simulate_seir(
        N=N, S0=S0, E0=E0, I0=I0, beta=base_beta, sigma=sigma, gamma=gamma, t_max=t_max
    )

    # 2. Phase 1 up to t_intervention
    phase1 = simulate_seir(
        N=N, S0=S0, E0=E0, I0=I0, beta=base_beta, sigma=sigma, gamma=gamma, t_max=t_intervention, steps=500
    )
    s_int = phase1["S"][-1]
    e_int = phase1["E"][-1]
    i_int = phase1["I"][-1]
    r_int = phase1["R"][-1]

    # 3. Phase 2 with reduced transmission
    remaining_time = max(1.0, t_max - t_intervention)
    phase2 = simulate_seir(
        N=N, S0=s_int, E0=e_int, I0=i_int, R0_init=r_int,
        beta=intervention_beta, sigma=sigma, gamma=gamma, t_max=remaining_time, steps=500
    )

    peak_intervened = max(phase1["peak_infection"], phase2["peak_infection"])
    peak_reduction = (base_sim["peak_infection"] - peak_intervened) / base_sim["peak_infection"]

    return {
        "baseline_peak_infection": base_sim["peak_infection"],
        "intervened_peak_infection": peak_intervened,
        "peak_reduction_ratio": peak_reduction,
        "baseline_r0": calculate_r0(base_beta, gamma),
        "intervened_r0": calculate_r0(intervention_beta, gamma),
    }
