"""Economic Order Quantity (EOQ) calculations and volume discount optimization."""
from __future__ import annotations
import math
from typing import List, Dict, Any, Optional


def calculate_eoq(annual_demand: float, order_cost: float, holding_cost_per_unit: float) -> float:
    """Calculate the classical Wilson Economic Order Quantity (EOQ).

    Formula:
        Q* = sqrt((2 * D * S) / H)

    Args:
        annual_demand: Total annual demand (units/year). Must be > 0.
        order_cost: Fixed setup/order cost per replenishment order (S). Must be > 0.
        holding_cost_per_unit: Annual inventory carrying cost per unit (H). Must be > 0.

    Returns:
        Optimal order quantity Q*.
    """
    if annual_demand <= 0 or order_cost <= 0 or holding_cost_per_unit <= 0:
        raise ValueError("Annual demand, order cost, and holding cost must be strictly positive.")
    return math.sqrt((2.0 * annual_demand * order_cost) / holding_cost_per_unit)


def calculate_total_cost(
    annual_demand: float,
    order_cost: float,
    holding_cost_per_unit: float,
    unit_cost: float,
    order_quantity: float,
) -> float:
    """Calculate the total annual inventory cost for a given order quantity Q.

    Formula:
        TC(Q) = D * C + (D / Q) * S + (Q / 2) * H
    """
    if order_quantity <= 0:
        raise ValueError("Order quantity must be strictly positive.")
    purchase_cost = annual_demand * unit_cost
    ordering_cost = (annual_demand / order_quantity) * order_cost
    holding_cost = (order_quantity / 2.0) * holding_cost_per_unit
    return purchase_cost + ordering_cost + holding_cost


def calculate_eoq_volume_discount(
    annual_demand: float,
    order_cost: float,
    holding_rate: float,
    discount_brackets: List[Dict[str, float]],
) -> Dict[str, Any]:
    """Calculate the optimal order quantity under all-units volume discount schedule.

    Each bracket contains:
        {"min_qty": float, "max_qty": float, "unit_cost": float}

    Holding cost per unit in bracket i is: H_i = holding_rate * unit_cost_i.

    Returns:
        Dict with "optimal_quantity", "optimal_cost", "bracket_evaluated", and "all_evaluated".
    """
    best_qty = 0.0
    best_cost = float("inf")
    best_bracket = -1
    evaluated = []

    for idx, bracket in enumerate(discount_brackets):
        c_i = bracket["unit_cost"]
        h_i = holding_rate * c_i
        q_raw = calculate_eoq(annual_demand, order_cost, h_i)
        min_q = bracket.get("min_qty", 0.0)
        max_q = bracket.get("max_qty", float("inf"))

        # Feasibility check for raw EOQ in bracket
        if min_q <= q_raw <= max_q:
            q_candidate = q_raw
        elif q_raw < min_q:
            # Step threshold kink: evaluate at breakpoint min_qty
            q_candidate = min_q
        else:
            # Above max_qty - dominated or infeasible for this bracket
            q_candidate = max_q

        cost_candidate = calculate_total_cost(annual_demand, order_cost, h_i, c_i, q_candidate)
        evaluated.append({
            "bracket_index": idx,
            "unit_cost": c_i,
            "raw_eoq": q_raw,
            "candidate_qty": q_candidate,
            "total_cost": cost_candidate,
        })

        if cost_candidate < best_cost:
            best_cost = cost_candidate
            best_qty = q_candidate
            best_bracket = idx

    return {
        "optimal_quantity": best_qty,
        "optimal_cost": best_cost,
        "best_bracket_index": best_bracket,
        "evaluations": evaluated,
    }
