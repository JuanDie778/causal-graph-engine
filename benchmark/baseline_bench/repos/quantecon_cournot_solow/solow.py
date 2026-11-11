"""Solow-Swan Neoclassical Growth Model steady state and golden rule."""
from __future__ import annotations
import math
from typing import Dict, Any


def compute_steady_state(
    s: float,
    alpha: float,
    n: float,
    g: float,
    delta: float,
) -> Dict[str, Any]:
    """Calculate the steady-state equilibrium of the Solow-Swan growth model.

    Production function per effective worker:
        y = f(k) = k^alpha

    Capital accumulation law:
        dot(k) = s * k^alpha - (n + g + delta) * k = 0 at steady state

    Steady state capital per effective worker:
        k* = (s / (n + g + delta))^(1 / (1 - alpha))

    Conservation invariant:
        y* = c* + i*
    """
    if not (0.0 < s < 1.0):
        raise ValueError("Savings rate 's' must be strictly between 0 and 1.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("Capital share 'alpha' must be strictly between 0 and 1.")
    effective_deprec = n + g + delta
    if effective_deprec <= 0:
        raise ValueError("Effective depreciation (n + g + delta) must be strictly positive.")

    k_star = math.pow(s / effective_deprec, 1.0 / (1.0 - alpha))
    y_star = math.pow(k_star, alpha)
    i_star = s * y_star
    c_star = (1.0 - s) * y_star

    conservation_residual = abs(y_star - (c_star + i_star))

    return {
        "k_star": k_star,
        "y_star": y_star,
        "c_star": c_star,
        "i_star": i_star,
        "savings_rate": s,
        "alpha": alpha,
        "effective_depreciation": effective_deprec,
        "conservation_residual": conservation_residual,
        "conservation_verified": conservation_residual < 1e-6,
    }


def compute_golden_rule(
    alpha: float,
    n: float,
    g: float,
    delta: float,
) -> Dict[str, Any]:
    """Calculate the Golden Rule savings rate and consumption-maximizing capital stock.

    In the Solow model, the Golden Rule requires:
        f'(k_gold) = n + g + delta
        alpha * k^(alpha - 1) = n + g + delta
        ==> s_gold = alpha
        ==> k_gold = (alpha / (n + g + delta))^(1 / (1 - alpha))
        ==> c_gold = (1 - alpha) * (k_gold)^alpha
    """
    effective_deprec = n + g + delta
    s_gold = alpha
    k_gold = math.pow(alpha / effective_deprec, 1.0 / (1.0 - alpha))
    y_gold = math.pow(k_gold, alpha)
    c_gold = (1.0 - alpha) * y_gold
    i_gold = alpha * y_gold

    return {
        "s_golden_rule": s_gold,
        "k_gold": k_gold,
        "y_gold": y_gold,
        "c_gold_max": c_gold,
        "i_gold": i_gold,
    }
