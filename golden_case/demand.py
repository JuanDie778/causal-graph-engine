"""Mecanismo de generación de demanda de mercado."""
import math


def calculate_demand(base_demand: float, price: float, marketing_budget: float) -> float:
    """Calcula la demanda esperada en función del precio y presupuesto de marketing.
    
    d = D_0 * (1 - 0.15 * (price - 3)) + 8 * sqrt(marketing_budget)
    """
    price_factor = 1.0 - 0.15 * (price - 3.0)
    marketing_lift = 8.0 * math.sqrt(max(0.0, marketing_budget))
    raw_demand = base_demand * price_factor + marketing_lift
    return max(0.0, raw_demand)
