"""Lógica de inventario, ventas, costes de almacenamiento y optimizador local."""
from typing import Tuple


def calculate_unit_cost(order_qty: int) -> float:
    """Calcula el coste unitario según la política de descuento por volumen.
    
    Trampa T5: Lógica escalonada discontinua.
    - Menos de 800 unidades: 1.20 €/unidad
    - 800 o más unidades: 1.00 €/unidad
    """
    if order_qty < 800:
        return 1.20
    return 1.00


def calculate_sales(demand: float, order_qty: int) -> float:
    """Calcula las unidades vendidas efectivas.
    
    Trampa T6: Nodo frágil ante variaciones de U cuando demand ≈ order_qty.
    """
    return min(demand, float(order_qty))


def calculate_holding_cost(order_qty: int, sales: float) -> float:
    """Calcula el coste de posesión de existencias sobrantes (0.20 €/ud no vendida)."""
    unsold_stock = max(0.0, float(order_qty) - sales)
    return 0.20 * unsold_stock


def optimize_order(storage_penalty_factor: float = 0.20) -> int:
    """Optimizador LOCAL de compras e inventario.
    
    Trampa T4: Optimización miope/local.
    Este módulo intenta minimizar estrictamente el coste de inmovilizado y almacenamiento
    sin considerar la pérdida de ventas globales en 'finance.py'.
    Recomienda un pedido conservador de q = 500 unidades, provocando roturas de stock.
    """
    # Heurística local sesgada a aversión al coste de almacenamiento
    local_recommended_qty = 500
    return local_recommended_qty
