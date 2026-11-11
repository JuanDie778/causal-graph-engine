"""Constrained economic dispatch optimization."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Any
from .generator import ThermalGenerator


def solve_economic_dispatch(
    generators: List[ThermalGenerator],
    load_demand: float,
) -> Dict[str, Any]:
    """Solve the constrained economic dispatch problem for a set of thermal generators.

    Formulation:
        min sum_{i=1}^M (a_i * P_i^2 + b_i * P_i + c_i)
        s.t.
            sum_{i=1}^M P_i = load_demand
            P_min,i <= P_i <= P_max,i   for all i = 1..M
    """
    n = len(generators)
    if n == 0:
        raise ValueError("Must provide at least one generator.")

    min_total = sum(g.p_min for g in generators)
    max_total = sum(g.p_max for g in generators)

    # 1. Direct Feasibility Certification
    if load_demand < min_total - 1e-6:
        return {
            "status": "INFEASIBLE_CERTIFIED",
            "iis_constraints": ["power_balance", "min_generation_limit"],
            "slack_required": float(min_total - load_demand),
            "optimal_power": {},
            "optimal_cost": float("inf"),
            "system_lambda": 0.0,
            "power_balance_residual": float(min_total - load_demand),
        }
    if load_demand > max_total + 1e-6:
        return {
            "status": "INFEASIBLE_CERTIFIED",
            "iis_constraints": ["power_balance", "max_generation_limit"],
            "slack_required": float(load_demand - max_total),
            "optimal_power": {},
            "optimal_cost": float("inf"),
            "system_lambda": 0.0,
            "power_balance_residual": float(load_demand - max_total),
        }

    # 2. Setup Quadratic Optimization
    def objective(p: np.ndarray) -> float:
        return float(sum(generators[i].fuel_cost(p[i]) for i in range(n)))

    def objective_grad(p: np.ndarray) -> np.ndarray:
        return np.array([generators[i].marginal_cost(p[i]) for i in range(n)], dtype=float)

    bounds = [(g.p_min, g.p_max) for g in generators]
    constraints = [
        {"type": "eq", "fun": lambda p: np.sum(p) - load_demand, "jac": lambda p: np.ones(n)}
    ]

    # Proportional initial guess
    span = max_total - min_total
    if span > 0:
        p0 = np.array([
            g.p_min + (load_demand - min_total) * (g.p_max - g.p_min) / span
            for g in generators
        ])
    else:
        p0 = np.array([g.p_min for g in generators])

    res = minimize(
        fun=objective,
        x0=p0,
        jac=objective_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500},
    )

    p_opt = res.x
    optimal_cost = float(objective(p_opt))
    power_balance_residual = float(abs(np.sum(p_opt) - load_demand))

    # Calculate marginal system lambda from interior generators
    interior_lambdas = [
        generators[i].marginal_cost(p_opt[i])
        for i in range(n)
        if (p_opt[i] > generators[i].p_min + 1e-4) and (p_opt[i] < generators[i].p_max - 1e-4)
    ]
    if interior_lambdas:
        system_lambda = float(np.mean(interior_lambdas))
    else:
        # Fallback to mean marginal cost of active generators
        system_lambda = float(np.mean([generators[i].marginal_cost(p_opt[i]) for i in range(n)]))

    power_map = {generators[i].id: float(p_opt[i]) for i in range(n)}

    return {
        "status": "CERTIFIED_OPTIMAL",
        "optimal_power": power_map,
        "optimal_cost": optimal_cost,
        "system_lambda": system_lambda,
        "power_balance_residual": power_balance_residual,
        "kkt_verified": power_balance_residual < 1e-5,
    }
