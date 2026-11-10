"""Pipeline integral de ejecución para la Tostadora de Café."""
from typing import Dict, Any
from .config import PRICE, MARKETING_BUDGET, ORDER_QTY
from .environment import get_nominal_base_demand, sample_base_demand
from .demand import calculate_demand
from .inventory import calculate_unit_cost, calculate_sales, calculate_holding_cost
from .finance import calculate_revenue, calculate_fees, calculate_profit
from .marketing import calculate_brand_awareness


def run_pipeline(
    price: float = PRICE,
    marketing_budget: float = MARKETING_BUDGET,
    order_qty: int = ORDER_QTY,
    base_demand: float = None
) -> Dict[str, Any]:
    """Ejecuta el grafo causal completo de la Tostadora de Café."""
    if base_demand is None:
        base_demand = get_nominal_base_demand()

    # Mecanismos de propagación causal F
    brand_awareness = calculate_brand_awareness(marketing_budget)
    demand = calculate_demand(base_demand, price, marketing_budget)
    sales = calculate_sales(demand, order_qty)
    unit_cost = calculate_unit_cost(order_qty)
    holding_cost = calculate_holding_cost(order_qty, sales)
    revenue = calculate_revenue(price, sales)
    fees = calculate_fees(revenue)
    profit = calculate_profit(revenue, fees, unit_cost, order_qty, marketing_budget, holding_cost)

    return {
        "base_demand": base_demand,
        "price": price,
        "marketing_budget": marketing_budget,
        "order_qty": order_qty,
        "brand_awareness": brand_awareness,
        "demand": demand,
        "sales": sales,
        "unit_cost": unit_cost,
        "holding_cost": holding_cost,
        "revenue": revenue,
        "fees": fees,
        "profit": profit
    }
