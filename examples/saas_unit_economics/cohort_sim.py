"""Simulador imperativo multi-mes (L2) de retención de cohorte y flujo de caja acumulado."""
from .tiers import compute_cloud_delivery_cost


def simulate_cohort_retention(
    initial_seats: int = 40,
    seat_price: float = 65.0,
    monthly_churn: float = 0.03,
    expansion_rate: float = 0.015,
    horizon_months: int = 12,
) -> dict:
    """Simula mes a mes la evolución neta de asientos y el margen acumulado de la cohorte."""
    if initial_seats <= 0 or seat_price < 15.0:
        raise ValueError("initial_seats > 0 and seat_price >= 15.0 required")
    if monthly_churn < 0.0 or monthly_churn > 0.30:
        raise ValueError("monthly_churn out of bounds")
    if expansion_rate < 0.0 or expansion_rate > 0.20:
        raise ValueError("expansion_rate out of bounds")

    active_seats = float(initial_seats)
    cumulative_margin = 0.0
    for _m in range(int(horizon_months)):
        monthly_bookings = active_seats * seat_price
        monthly_cost = compute_cloud_delivery_cost(active_seats, monthly_bookings)
        cumulative_margin += (monthly_bookings - monthly_cost)
        active_seats = active_seats * (1.0 - monthly_churn + expansion_rate)

    return {
        "ending_seats": active_seats,
        "cumulative_margin": cumulative_margin,
    }
