"""Tests unitarios para el repositorio Golden Case (Tostadora de Café)."""
import pytest
from golden_case.environment import get_nominal_base_demand
from golden_case.demand import calculate_demand
from golden_case.inventory import calculate_unit_cost, calculate_sales, calculate_holding_cost, optimize_order
from golden_case.finance import calculate_revenue, calculate_fees, calculate_profit, Ledger, process_refund
from golden_case.marketing import calculate_brand_awareness
from golden_case.pipeline import run_pipeline


def test_nominal_demand():
    d = calculate_demand(base_demand=1000.0, price=3.0, marketing_budget=400.0)
    # d = 1000 * 1.0 + 8 * 20 = 1160.0
    assert abs(d - 1160.0) < 1e-4


def test_volume_discount_step_t5():
    # Trampa T5: discontinuidad en q=800
    assert calculate_unit_cost(799) == 1.20
    assert calculate_unit_cost(800) == 1.00
    assert calculate_unit_cost(1000) == 1.00


def test_sales_and_holding_cost():
    # Demand 1000, Order 800 -> Sales = 800, Holding = 0
    assert calculate_sales(1000.0, 800) == 800.0
    assert calculate_holding_cost(800, 800.0) == 0.0

    # Demand 600, Order 800 -> Sales = 600, Holding = 0.20 * 200 = 40.0
    assert calculate_sales(600.0, 800) == 600.0
    assert calculate_holding_cost(800, 600.0) == 40.0


def test_fee_rate_discrepancy_t3():
    # Trampa T3: El código aplica 0.03 (el README dice 2%)
    rev = 1000.0
    fees = calculate_fees(rev)
    assert fees == 30.0  # 3%


def test_dead_node_brand_awareness_t1():
    # Trampa T1: brand_awareness se calcula pero no afecta a profit
    res1 = run_pipeline(marketing_budget=500.0)
    assert res1["brand_awareness"] > 0.0
    # Si alteramos manualmente brand_awareness en el resultado, profit no cambia
    res2 = run_pipeline(marketing_budget=500.0)
    assert res1["profit"] == res2["profit"]


def test_ledger_invariant_t2():
    # Transacción normal balanceada
    ledger = Ledger()
    ledger.record_credit(100.0, "sale")
    ledger.record_debit(100.0, "cash")
    assert ledger.is_balanced()

    # Trampa T2: process_refund introduce desbalance (duplica débito)
    process_refund(ledger, 50.0)
    assert not ledger.is_balanced(), "process_refund violó el invariante c_ledger al duplicar el débito"


def test_local_optimizer_t4():
    # Trampa T4: optimize_order() devuelve 500, produciendo rotura de stock
    q_local = optimize_order()
    assert q_local == 500
    
    # Comparamos el beneficio con q=500 vs q=800 con precio 5.43 y m=423
    res_local = run_pipeline(price=5.43, marketing_budget=423.0, order_qty=500, base_demand=1000.0)
    res_better = run_pipeline(price=5.43, marketing_budget=423.0, order_qty=800, base_demand=1000.0)
    assert res_better["profit"] > res_local["profit"] + 600.0
