"""Safety stock and reorder point determination under stochastic demand."""
from __future__ import annotations
import math
from scipy.stats import norm


def calculate_lead_time_demand(daily_demand: float, lead_time_days: float) -> float:
    """Calculate expected demand over the replenishment lead time.

    Formula:
        mu_L = d_daily * L
    """
    if daily_demand < 0 or lead_time_days < 0:
        raise ValueError("Daily demand and lead time must be non-negative.")
    return daily_demand * lead_time_days


def calculate_safety_stock(
    daily_demand_std: float,
    lead_time_days: float,
    service_level: float = 0.95,
) -> float:
    """Calculate the safety stock needed to satisfy a target cycle service level.

    Formula:
        Z = norm.ppf(service_level)
        SS = Z * sigma_d * sqrt(L)

    Args:
        daily_demand_std: Standard deviation of daily demand (sigma_d).
        lead_time_days: Replenishment lead time in days (L).
        service_level: Target cycle service level (e.g. 0.95 for 95%).

    Returns:
        Safety stock in units.
    """
    if daily_demand_std < 0 or lead_time_days < 0:
        raise ValueError("Standard deviation and lead time must be non-negative.")
    if not (0.0 < service_level < 1.0):
        raise ValueError("Service level must be strictly between 0 and 1.")
    z = float(norm.ppf(service_level))
    return z * daily_demand_std * math.sqrt(lead_time_days)


def calculate_reorder_point(
    daily_demand: float,
    daily_demand_std: float,
    lead_time_days: float,
    service_level: float = 0.95,
) -> float:
    """Calculate the Reorder Point (ROP).

    Formula:
        ROP = mu_L + SS = (d_daily * L) + (Z * sigma_d * sqrt(L))
    """
    mu_l = calculate_lead_time_demand(daily_demand, lead_time_days)
    ss = calculate_safety_stock(daily_demand_std, lead_time_days, service_level)
    return mu_l + ss
