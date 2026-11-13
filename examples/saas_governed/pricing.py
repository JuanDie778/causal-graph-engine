"""Motor de pricing y margen operativo para contratos SaaS gobernados por actordb.yaml."""
from __future__ import annotations

from .infrastructure import compute_infra_spend


def evaluate_governed_contract(
    seats: int = 50,
    seat_price: float = 80.0,
    annual_discount: float = 0.10,
    sla_reserve_rate: float = 0.02,
) -> dict[str, float]:
    """Evalúa ingresos brutos, gasto de infraestructura y margen operativo neto."""
    if seats <= 0:
        raise ValueError("seats must be strictly positive")
    if seat_price <= 0.0:
        raise ValueError("seat_price must be strictly positive")
    if annual_discount < 0.0:
        raise ValueError("annual_discount cannot be negative")
    if annual_discount > 0.50:
        raise ValueError("annual_discount cannot exceed 50%")
    if sla_reserve_rate < 0.0:
        raise ValueError("sla_reserve_rate cannot be negative")
    if sla_reserve_rate > 0.25:
        raise ValueError("sla_reserve_rate cannot exceed 25%")

    effective_price = seat_price * (1.0 - annual_discount)
    gross_revenue = seats * effective_price
    infra_spend = compute_infra_spend(seats, gross_revenue)
    sla_reserve = gross_revenue * sla_reserve_rate
    net_operating_margin = gross_revenue - infra_spend - sla_reserve
    return {
        "gross_revenue": gross_revenue,
        "infra_spend": infra_spend,
        "net_operating_margin": net_operating_margin,
    }
