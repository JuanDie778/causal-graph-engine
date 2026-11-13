"""Progressive tax calculation and piecewise kink detection."""
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional

DEFAULT_BRACKETS: List[Tuple[float, float, float]] = [
    (0.0, 12450.0, 0.19),
    (12450.0, 20200.0, 0.24),
    (20200.0, 35200.0, 0.30),
    (35200.0, 60000.0, 0.37),
    (60000.0, float("inf"), 0.45),
]


def compute_progressive_tax(
    taxable_income: float,
    custom_brackets: Optional[List[Tuple[float, float, float]]] = None,
) -> Dict[str, Any]:
    """Compute progressive income tax over piecewise linear tax brackets.

    Formula:
        T(Y) = sum_j tau_j * max(0, min(Y, B_j) - B_{j-1})

    Invariants:
        tax_total + net_income == taxable_income
        effective_rate == tax_total / taxable_income
    """
    if taxable_income < 0:
        raise ValueError("Taxable income must be non-negative.")

    brackets = custom_brackets if custom_brackets is not None else DEFAULT_BRACKETS
    tax_total = 0.0
    active_marginal_rate = 0.0
    bracket_details = []

    for lower, upper, rate in brackets:
        if taxable_income > lower:
            taxable_in_bracket = min(taxable_income, upper) - lower
            bracket_tax = taxable_in_bracket * rate
            tax_total += bracket_tax
            active_marginal_rate = rate
            bracket_details.append({
                "lower": lower,
                "upper": upper,
                "rate": rate,
                "taxable_amount": taxable_in_bracket,
                "tax": bracket_tax,
            })
        else:
            bracket_details.append({
                "lower": lower,
                "upper": upper,
                "rate": rate,
                "taxable_amount": 0.0,
                "tax": 0.0,
            })

    effective_rate = (tax_total / taxable_income) if taxable_income > 0 else 0.0
    net_income = taxable_income - tax_total

    return {
        "taxable_income": taxable_income,
        "tax_total": tax_total,
        "effective_rate": effective_rate,
        "marginal_rate": active_marginal_rate,
        "net_income": net_income,
        "brackets": bracket_details,
        "conservation_verified": abs((tax_total + net_income) - taxable_income) < 1e-6,
    }


def detect_tax_kinks(
    income_range: Tuple[float, float] = (0.0, 100000.0),
    custom_brackets: Optional[List[Tuple[float, float, float]]] = None,
) -> List[float]:
    """Return the threshold points (kinks) where the marginal tax rate has a jump discontinuity."""
    brackets = custom_brackets if custom_brackets is not None else DEFAULT_BRACKETS
    min_inc, max_inc = income_range
    kinks = []
    for lower, upper, _ in brackets:
        if lower > 0 and min_inc <= lower <= max_inc:
            kinks.append(lower)
    return kinks
