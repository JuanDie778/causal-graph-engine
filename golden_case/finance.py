"""Módulo financiero, cálculo de beneficio neto y libro mayor (Ledger)."""
from typing import List, Dict, Any


def calculate_revenue(price: float, sales: float) -> float:
    """Calcula los ingresos brutos totales."""
    return price * sales


def calculate_fees(revenue: float) -> float:
    """Calcula las comisiones de pasarela y procesamiento.
    
    Trampa T3: Doc vs Código.
    El README afirma expresamente que la comisión es del 2.0% (0.02),
    pero el código aplica efectivamente un 3.0% (0.03).
    """
    fee_rate = 0.03
    return fee_rate * revenue


def calculate_profit(
    revenue: float,
    fees: float,
    unit_cost: float,
    order_qty: int,
    marketing_budget: float,
    holding_cost: float
) -> float:
    """Calcula el beneficio neto operativo J (métrica objetivo target).
    
    J = revenue - fees - (unit_cost * order_qty) - marketing_budget - holding_cost
    """
    total_goods_cost = unit_cost * float(order_qty)
    profit = revenue - fees - total_goods_cost - marketing_budget - holding_cost
    return profit


class Ledger:
    """Libro mayor contable para verificación de invariante de suma cero (c_ledger)."""

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []

    def record_credit(self, amount: float, description: str = ""):
        self.entries.append({"type": "credit", "amount": float(amount), "desc": description})

    def record_debit(self, amount: float, description: str = ""):
        self.entries.append({"type": "debit", "amount": float(amount), "desc": description})

    @property
    def total_credits(self) -> float:
        return sum(e["amount"] for e in self.entries if e["type"] == "credit")

    @property
    def total_debits(self) -> float:
        return sum(e["amount"] for e in self.entries if e["type"] == "debit")

    def is_balanced(self) -> bool:
        """Comprueba el invariante c_ledger: sum(credits) - sum(debits) == 0."""
        return abs(self.total_credits - self.total_debits) < 1e-6


def process_refund(ledger: Ledger, amount: float):
    """Procesa una devolución a cliente en el libro mayor.
    
    Trampa T2: Violación de invariante contable.
    El flujo registra un crédito de cliente de 'amount', pero anota erróneamente
    el débito dos veces (duplicación en payout y adjustment), violando c_ledger.
    """
    ledger.record_credit(amount, "refund_issued_to_customer")
    # ERROR / BUG T2: Se anota el débito dos veces rompiendo sum(credits) - sum(debits) == 0
    ledger.record_debit(amount, "refund_cash_outflow")
    ledger.record_debit(amount, "refund_tax_adjustment")
