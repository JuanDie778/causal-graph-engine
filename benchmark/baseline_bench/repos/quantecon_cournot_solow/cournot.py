"""Cournot Oligopoly Nash Equilibrium solver."""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Optional


def solve_cournot_nash(
    a: float,
    b: float,
    marginal_costs: List[float],
    capacities: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Solve the Cournot-Nash equilibrium for N competing firms with linear inverse demand.

    Inverse market demand:
        P(Q) = max(0, a - b * Q), where Q = sum(q_i)

    Cost for firm i:
        C_i(q_i) = c_i * q_i

    Profit for firm i:
        pi_i(q_i, Q_{-i}) = P(q_i + Q_{-i}) * q_i - c_i * q_i

    First-Order Necessary Conditions (FONC) with active firms (q_i > 0):
        a - b * Q - b * q_i - c_i = 0
        summing over all M active firms:
        M * a - M * b * Q - b * Q - sum(c_i) = 0
        ==> Q* = (M * a - sum(c_i)) / (b * (M + 1))
        ==> q_i* = (a - c_i - b * Q*) / b = (a - c_i) / b - Q*

    If some firm has q_i < 0, an active set algorithm eliminates uncompetitive firms.
    """
    n = len(marginal_costs)
    if n == 0:
        raise ValueError("Must provide at least one firm's marginal cost.")
    if a <= 0 or b <= 0:
        raise ValueError("Demand choke price 'a' and slope 'b' must be strictly positive.")

    # Sort firms by marginal cost to handle entry/exit deterministically
    c = np.array(marginal_costs, dtype=float)
    active = np.ones(n, dtype=bool)

    # Iterative active set to guarantee non-negative quantities q_i >= 0
    while True:
        m = int(np.sum(active))
        if m == 0:
            q = np.zeros(n)
            break
        
        sum_c = float(np.sum(c[active]))
        q_total = (m * a - sum_c) / (b * (m + 1))

        if q_total <= 0:
            q = np.zeros(n)
            break

        q_active = (a - c[active] - b * q_total) / b

        # Check if any active firm produces negative quantity
        if np.any(q_active < 0):
            # Deactivate the firm with most negative output (highest cost)
            worst_active_idx = np.where(active)[0][np.argmin(q_active)]
            active[worst_active_idx] = False
        else:
            q = np.zeros(n)
            q[active] = q_active
            break

    # Apply capacity upper bounds if provided
    if capacities is not None:
        caps = np.array(capacities, dtype=float)
        q = np.minimum(q, caps)

    total_q = float(np.sum(q))
    market_price = max(0.0, float(a - b * total_q))
    profits = [float(market_price * q[i] - c[i] * q[i]) for i in range(n)]

    return {
        "quantities": [float(val) for val in q],
        "total_quantity": total_q,
        "price": market_price,
        "profits": profits,
        "active_firms_count": int(np.sum(q > 1e-6)),
        "market_cleared": True,
    }
