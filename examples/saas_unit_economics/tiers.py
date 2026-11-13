"""Constantes de infraestructura cloud y helper cross-módulo de costes por asiento."""
from dataclasses import dataclass

BASE_CLUSTER_COST: float = 180.0
GPU_COST_PER_SEAT: float = 7.50
SUPPORT_OVERHEAD_RATE: float = 0.04


@dataclass
class EnterprisePlanDefaults:
    seats: int = 40
    seat_price: float = 65.0
    annual_discount: float = 0.12
    sla_credit_reserve: float = 0.03
    partner_rebate_rate: float = 0.08


def compute_cloud_delivery_cost(seats: int, gross_bookings: float) -> float:
    """Calcula el coste mensual total de entrega cloud + soporte técnico."""
    if seats <= 0:
        raise ValueError("seats must be strictly positive")
    return BASE_CLUSTER_COST + (GPU_COST_PER_SEAT * seats) + (SUPPORT_OVERHEAD_RATE * gross_bookings)
