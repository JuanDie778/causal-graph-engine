"""Cálculo de costo de infraestructura cloud para contratos B2B gobernados."""
from __future__ import annotations

BASE_CLUSTER_COST_USD: float = 320.0
PER_SEAT_INFRA_USD: float = 6.5
REVENUE_CLOUD_EGRESS_RATE: float = 0.04


def compute_infra_spend(seats: int, gross_revenue: float) -> float:
    """Calcula el gasto de infraestructura en función de asientos y facturación bruta."""
    if seats <= 0:
        raise ValueError("seats must be strictly positive")
    if gross_revenue < 0.0:
        raise ValueError("gross_revenue must be non-negative")
    variable_compute = seats * PER_SEAT_INFRA_USD
    egress_overhead = gross_revenue * REVENUE_CLOUD_EGRESS_RATE
    return BASE_CLUSTER_COST_USD + variable_compute + egress_overhead
