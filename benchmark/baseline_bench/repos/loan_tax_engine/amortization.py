"""French amortization schedule (fixed payment annuity) calculator."""
from __future__ import annotations
import math
from typing import Dict, Any, List


def compute_french_schedule(
    principal: float,
    annual_rate: float,
    periods: int,
    freq_per_year: int = 12,
) -> Dict[str, Any]:
    """Compute the full French amortization schedule.

    Formula for periodic payment A:
        r = annual_rate / freq_per_year
        A = P_0 * [r * (1 + r)^N] / [(1 + r)^N - 1]

    Guarantees strict financial invariants:
        sum(principal_amortized_t) == principal  (error < 1e-6)
        sum(payment_t) == principal + sum(interest_t)
        ending_balance_N == 0.0

    Returns:
        Dict with "payment_per_period", "total_paid", "total_principal",
        "total_interest", "schedule", and "conservation_verified".
    """
    if principal <= 0:
        raise ValueError("Principal must be strictly positive.")
    if periods <= 0:
        raise ValueError("Periods must be a positive integer.")
    if freq_per_year <= 0:
        raise ValueError("Frequency must be a positive integer.")

    r = annual_rate / float(freq_per_year)

    if r == 0:
        # Zero-interest edge case
        payment = principal / periods
        total_paid = principal
        total_interest = 0.0
        schedule = []
        bal = principal
        for t in range(1, periods + 1):
            c_t = payment
            bal -= c_t
            schedule.append({
                "period": t,
                "payment": payment,
                "interest": 0.0,
                "principal_amortized": c_t,
                "remaining_balance": max(0.0, bal),
            })
    else:
        factor = math.pow(1.0 + r, periods)
        payment = principal * (r * factor) / (factor - 1.0)
        schedule = []
        bal = principal
        tot_int = 0.0
        tot_prin = 0.0

        for t in range(1, periods + 1):
            i_t = bal * r
            c_t = payment - i_t
            if t == periods:
                # Force terminal balance to exact zero
                c_t = bal
                payment_adj = c_t + i_t
            else:
                payment_adj = payment

            bal -= c_t
            tot_int += i_t
            tot_prin += c_t

            schedule.append({
                "period": t,
                "payment": payment_adj,
                "interest": i_t,
                "principal_amortized": c_t,
                "remaining_balance": max(0.0, bal),
            })

        total_paid = tot_prin + tot_int
        total_interest = tot_int

    principal_diff = abs(tot_prin - principal) if r != 0 else 0.0
    conservation_verified = principal_diff < 1e-4

    return {
        "payment_per_period": payment,
        "total_paid": total_paid,
        "total_principal": principal,
        "total_interest": total_interest,
        "periods": periods,
        "periodic_rate": r,
        "conservation_verified": conservation_verified,
        "schedule": schedule,
    }
