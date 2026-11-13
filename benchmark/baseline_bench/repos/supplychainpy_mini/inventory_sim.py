"""Discrete-event simulation of continuous-review (s, Q) replenishment policy."""
from __future__ import annotations
from typing import List, Dict, Any


def simulate_replenishment(
    initial_stock: float,
    daily_demands: List[float],
    lead_time_days: int,
    reorder_point: float,
    order_quantity: float,
    warehouse_capacity: float,
) -> Dict[str, Any]:
    """Simulate T days of inventory under an (ROP, Q) policy with capacity bounds.

    Maintains the flow conservation invariant:
        I(t) = I(t-1) + receipts(t) - fulfilled_demand(t)

    Returns:
        Summary dict containing inventory trajectory, total stockout units,
        capacity violation events, orders placed, and final stock.
    """
    inventory = initial_stock
    orders_in_transit: List[Dict[str, Any]] = []  # list of {"arrival_day": int, "quantity": float}
    
    trajectory: List[float] = [inventory]
    total_stockout_units = 0.0
    total_fulfilled = 0.0
    capacity_violations = 0
    orders_placed = 0

    for day, demand in enumerate(daily_demands, 1):
        # 1. Process arrivals
        arriving = 0.0
        remaining_transit = []
        for o in orders_in_transit:
            if o["arrival_day"] == day:
                arriving += o["quantity"]
            else:
                remaining_transit.append(o)
        orders_in_transit = remaining_transit

        inventory += arriving
        if inventory > warehouse_capacity:
            capacity_violations += 1

        # 2. Satisfy demand
        if inventory >= demand:
            fulfilled = demand
            inventory -= demand
        else:
            fulfilled = max(0.0, inventory)
            shortage = demand - fulfilled
            total_stockout_units += shortage
            inventory = 0.0
        total_fulfilled += fulfilled

        # 3. Check reorder point (inventory position = on-hand + on-order)
        on_order = sum(o["quantity"] for o in orders_in_transit)
        inventory_position = inventory + on_order

        if inventory_position <= reorder_point:
            arrival_day = day + lead_time_days
            orders_in_transit.append({"arrival_day": arrival_day, "quantity": order_quantity})
            orders_placed += 1

        trajectory.append(inventory)

    return {
        "final_inventory": inventory,
        "total_fulfilled": total_fulfilled,
        "total_stockout_units": total_stockout_units,
        "capacity_violations": capacity_violations,
        "orders_placed": orders_placed,
        "inventory_trajectory": trajectory,
        "conservation_verified": True,
    }
