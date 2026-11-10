"""10 Problemas Públicos de Investigación Operativa y Modelado de Decisiones (OR-Bench Público, Fase 10).

Cada problema está escrito como código Python natural (funciones escalares o pipelines),
con cero solapamiento con la plantilla de la tostadora de café.
"""
from __future__ import annotations
from typing import Dict, Any, List
from .oracles import ORACLE_REGISTRY


PUBLIC_PROBLEMS: List[Dict[str, Any]] = [
    {
        "id": "p01_newsvendor",
        "name": "P01 - Classical Single-Period Newsvendor with Salvage & Shortage Penalty",
        "category": "NLP_NONSMOOTH",
        "sense": "max",
        "expected_status": "CERTIFIED_OPTIMAL",
        "entrypoint": "simulate_newsvendor",
        "source_code": '''"""P01: Classical single-period Newsvendor model under 3 equiprobable demand scenarios."""
SENSE = "max"
STOCK_LEVEL: float = 100.0
STOCK_LEVEL_BOUNDS = (40.0, 200.0)

def simulate_newsvendor(stock_level: float = 100.0) -> float:
    sold_low = min(stock_level, 80.0)
    salv_low = max(0.0, stock_level - 80.0)
    short_low = max(0.0, 80.0 - stock_level)
    pi_low = 25.0 * sold_low + 4.0 * salv_low - 5.0 * short_low - 10.0 * stock_level

    sold_mid = min(stock_level, 120.0)
    salv_mid = max(0.0, stock_level - 120.0)
    short_mid = max(0.0, 120.0 - stock_level)
    pi_mid = 25.0 * sold_mid + 4.0 * salv_mid - 5.0 * short_mid - 10.0 * stock_level

    sold_high = min(stock_level, 160.0)
    salv_high = max(0.0, stock_level - 160.0)
    short_high = max(0.0, 160.0 - stock_level)
    pi_high = 25.0 * sold_high + 4.0 * salv_high - 5.0 * short_high - 10.0 * stock_level

    expected_return = (pi_low + pi_mid + pi_high) / 3.0
    return expected_return
''',
    },
    {
        "id": "p02_eoq",
        "name": "P02 - Economic Order Quantity (EOQ) with Warehouse Space Constraint",
        "category": "NLP_SMOOTH",
        "sense": "min",
        "expected_status": "CERTIFIED_OPTIMAL",
        "entrypoint": "compute_eoq_cost",
        "source_code": '''"""P02: Economic Order Quantity (EOQ) with setup, holding cost and warehouse capacity."""
SENSE = "min"
BATCH_QTY: float = 400.0
BATCH_QTY_BOUNDS = (100.0, 2000.0)
CONSTRAINTS = {
    "c_warehouse_space": "2.5 * batch_qty <= 1800.0",
}

def compute_eoq_cost(batch_qty: float = 400.0) -> float:
    ordering_cost = (120.0 * 10000.0) / batch_qty
    holding_expense = 0.5 * 3.0 * batch_qty
    total_inventory_cost = ordering_cost + holding_expense
    return total_inventory_cost
''',
    },
    {
        "id": "p03_prod_plan_lp",
        "name": "P03 - 6-Product Production Planning LP with 4 Shared Resource Capacities",
        "category": "LP",
        "sense": "max",
        "expected_status": "CERTIFIED_OPTIMAL",
        "entrypoint": "evaluate_production_plan",
        "source_code": '''"""P03: Multi-product linear production planning with 6 products and 4 resource constraints."""
SENSE = "max"
PROD_1: float = 10.0
PROD_2: float = 10.0
PROD_3: float = 10.0
PROD_4: float = 10.0
PROD_5: float = 10.0
PROD_6: float = 10.0

PROD_1_BOUNDS = (0.0, 50.0)
PROD_2_BOUNDS = (0.0, 50.0)
PROD_3_BOUNDS = (0.0, 50.0)
PROD_4_BOUNDS = (0.0, 50.0)
PROD_5_BOUNDS = (0.0, 50.0)
PROD_6_BOUNDS = (0.0, 50.0)

CONSTRAINTS = {
    "c_labor": "2.0 * prod_1 + 3.0 * prod_2 + 1.5 * prod_3 + 4.0 * prod_4 + 2.5 * prod_5 + 3.5 * prod_6 <= 240.0",
    "c_machine": "1.5 * prod_1 + 2.0 * prod_2 + 2.5 * prod_3 + 3.0 * prod_4 + 1.0 * prod_5 + 2.5 * prod_6 <= 180.0",
    "c_raw_mat": "3.0 * prod_1 + 1.5 * prod_2 + 2.0 * prod_3 + 2.5 * prod_4 + 3.0 * prod_5 + 2.0 * prod_6 <= 210.0",
    "c_energy": "1.0 * prod_1 + 2.5 * prod_2 + 1.0 * prod_3 + 3.5 * prod_4 + 2.0 * prod_5 + 3.0 * prod_6 <= 190.0",
}

def evaluate_production_plan(
    prod_1: float = 10.0,
    prod_2: float = 10.0,
    prod_3: float = 10.0,
    prod_4: float = 10.0,
    prod_5: float = 10.0,
    prod_6: float = 10.0,
) -> float:
    total_contribution = (
        45.0 * prod_1 + 60.0 * prod_2 + 35.0 * prod_3
        + 80.0 * prod_4 + 50.0 * prod_5 + 70.0 * prod_6
    )
    return total_contribution
''',
    },
    {
        "id": "p04_knapsack_01",
        "name": "P04 - 8-Item 0/1 Binary Knapsack Problem",
        "category": "MILP",
        "sense": "max",
        "expected_status": "CERTIFIED_OPTIMAL",
        "entrypoint": "evaluate_knapsack",
        "source_code": '''"""P04: 0/1 Binary Knapsack with 8 items and weight capacity 95."""
SENSE = "max"
ITEM_1: int = 1
ITEM_2: int = 0
ITEM_3: int = 1
ITEM_4: int = 0
ITEM_5: int = 0
ITEM_6: int = 0
ITEM_7: int = 1
ITEM_8: int = 0

ITEM_1_BOUNDS = (0, 1)
ITEM_2_BOUNDS = (0, 1)
ITEM_3_BOUNDS = (0, 1)
ITEM_4_BOUNDS = (0, 1)
ITEM_5_BOUNDS = (0, 1)
ITEM_6_BOUNDS = (0, 1)
ITEM_7_BOUNDS = (0, 1)
ITEM_8_BOUNDS = (0, 1)

CONSTRAINTS = {
    "c_weight_limit": "18.0 * item_1 + 15.0 * item_2 + 28.0 * item_3 + 12.0 * item_4 + 23.0 * item_5 + 31.0 * item_6 + 14.0 * item_7 + 21.0 * item_8 <= 95.0",
}

def evaluate_knapsack(
    item_1: int = 1,
    item_2: int = 0,
    item_3: int = 1,
    item_4: int = 0,
    item_5: int = 0,
    item_6: int = 0,
    item_7: int = 1,
    item_8: int = 0,
) -> float:
    packed_value = (
        42.0 * item_1 + 38.0 * item_2 + 65.0 * item_3 + 29.0 * item_4
        + 54.0 * item_5 + 71.0 * item_6 + 33.0 * item_7 + 49.0 * item_8
    )
    return packed_value
''',
    },
    {
        "id": "p05_blending_eq",
        "name": "P05 - Chemical Blending LP with Exact Equality Composition & Quality Bounds",
        "category": "LP",
        "sense": "min",
        "expected_status": "CERTIFIED_OPTIMAL",
        "entrypoint": "evaluate_blend_cost",
        "source_code": '''"""P05: Chemical blending LP with exact equality constraint sum(ing_i) == 100 and potency/impurity limits."""
SENSE = "min"
ING_1: float = 25.0
ING_2: float = 25.0
ING_3: float = 25.0
ING_4: float = 25.0

ING_1_BOUNDS = (0.0, 70.0)
ING_2_BOUNDS = (0.0, 70.0)
ING_3_BOUNDS = (0.0, 70.0)
ING_4_BOUNDS = (0.0, 70.0)

CONSTRAINTS = {
    "c_total_mass_eq": "ing_1 + ing_2 + ing_3 + ing_4 == 100.0",
    "c_min_potency": "5.0 * ing_1 + 8.0 * ing_2 + 3.0 * ing_3 + 10.0 * ing_4 >= 680.0",
    "c_max_impurity": "3.0 * ing_1 + 1.5 * ing_2 + 4.0 * ing_3 + 0.8 * ing_4 <= 210.0",
}

def evaluate_blend_cost(
    ing_1: float = 25.0,
    ing_2: float = 25.0,
    ing_3: float = 25.0,
    ing_4: float = 25.0,
) -> float:
    blend_cost = 12.0 * ing_1 + 18.0 * ing_2 + 9.0 * ing_3 + 24.0 * ing_4
    return blend_cost
''',
    },
    {
        "id": "p06_serial_bottlenecks_6",
        "name": "P06 - 6-Stage Serial Manufacturing Line Bottleneck Expansion (K=5 Min Atoms)",
        "category": "NLP_NONSMOOTH",
        "sense": "max",
        "expected_status": "CERTIFIED_OPTIMAL",
        "entrypoint": "evaluate_serial_line",
        "source_code": '''"""P06: 6-stage serial manufacturing line y = min(c1, c2, c3, c4, c5, c6) with expansion budget."""
SENSE = "max"
ADD_CAP_1: float = 5.0
ADD_CAP_2: float = 5.0
ADD_CAP_3: float = 5.0
ADD_CAP_4: float = 5.0
ADD_CAP_5: float = 5.0
ADD_CAP_6: float = 5.0

ADD_CAP_1_BOUNDS = (0.0, 60.0)
ADD_CAP_2_BOUNDS = (0.0, 60.0)
ADD_CAP_3_BOUNDS = (0.0, 60.0)
ADD_CAP_4_BOUNDS = (0.0, 60.0)
ADD_CAP_5_BOUNDS = (0.0, 60.0)
ADD_CAP_6_BOUNDS = (0.0, 60.0)

CONSTRAINTS = {
    "c_expansion_budget": "3.0 * add_cap_1 + 4.0 * add_cap_2 + 2.5 * add_cap_3 + 3.5 * add_cap_4 + 3.0 * add_cap_5 + 4.5 * add_cap_6 <= 260.0",
}

def evaluate_serial_line(
    add_cap_1: float = 5.0,
    add_cap_2: float = 5.0,
    add_cap_3: float = 5.0,
    add_cap_4: float = 5.0,
    add_cap_5: float = 5.0,
    add_cap_6: float = 5.0,
) -> float:
    stage_1 = 50.0 + add_cap_1
    stage_2 = 42.0 + add_cap_2
    stage_3 = 58.0 + add_cap_3
    stage_4 = 45.0 + add_cap_4
    stage_5 = 52.0 + add_cap_5
    stage_6 = 40.0 + add_cap_6

    line_throughput = min(stage_1, stage_2, stage_3, stage_4, stage_5, stage_6)
    expansion_spend = (
        3.0 * add_cap_1 + 4.0 * add_cap_2 + 2.5 * add_cap_3
        + 3.5 * add_cap_4 + 3.0 * add_cap_5 + 4.5 * add_cap_6
    )
    net_line_value = 50.0 * line_throughput - expansion_spend
    return net_line_value
''',
    },
    {
        "id": "p07_piecewise_tax",
        "name": "P07 - 4-Bracket Progressive Piecewise Tax & Concave Return Optimization",
        "category": "NLP_NONSMOOTH",
        "sense": "max",
        "expected_status": "CERTIFIED_OPTIMAL",
        "entrypoint": "evaluate_after_tax_surplus",
        "source_code": '''"""P07: 4-bracket progressive tax schedule with concave operational revenue."""
SENSE = "max"
ACTIVITY_LEVEL: float = 60.0
ACTIVITY_LEVEL_BOUNDS = (10.0, 220.0)

def evaluate_after_tax_surplus(activity_level: float = 60.0) -> float:
    gross_income = 18.0 * activity_level - 0.04 * (activity_level ** 2)
    if activity_level < 50.0:
        tax_charge = 0.10 * activity_level
    elif activity_level < 100.0:
        tax_charge = 5.0 + 0.20 * (activity_level - 50.0)
    elif activity_level < 150.0:
        tax_charge = 15.0 + 0.32 * (activity_level - 100.0)
    else:
        tax_charge = 31.0 + 0.45 * (activity_level - 150.0)
    overhead = 1.5 * activity_level
    after_tax_surplus = gross_income - tax_charge - overhead
    return after_tax_surplus
''',
    },
    {
        "id": "p08_infeasible_bounds_iis",
        "name": "P08 - Over-Constrained Resource Allocation with 3-Constraint Minimal IIS",
        "category": "LP",
        "sense": "max",
        "expected_status": "INFEASIBLE_CERTIFIED",
        "entrypoint": "evaluate_alloc_score",
        "source_code": '''"""P08: Over-constrained allocation where 3 out of 7 constraints form a minimal IIS."""
SENSE = "max"
ALLOC_A: float = 30.0
ALLOC_B: float = 30.0
ALLOC_C: float = 30.0

ALLOC_A_BOUNDS = (0.0, 100.0)
ALLOC_B_BOUNDS = (0.0, 100.0)
ALLOC_C_BOUNDS = (0.0, 100.0)

CONSTRAINTS = {
    "c_min_a": "alloc_a >= 45.0",
    "c_min_b": "alloc_b >= 50.0",
    "c_pair_cap": "alloc_a + alloc_b <= 80.0",
    "c_min_c": "alloc_c >= 10.0",
    "c_max_c": "alloc_c <= 90.0",
    "c_total_cap": "alloc_a + alloc_b + alloc_c <= 240.0",
    "c_weighted_cap": "0.5 * alloc_a + alloc_c <= 150.0",
}

def evaluate_alloc_score(alloc_a: float = 30.0, alloc_b: float = 30.0, alloc_c: float = 30.0) -> float:
    alloc_score = 4.0 * alloc_a + 5.0 * alloc_b + 3.0 * alloc_c
    return alloc_score
''',
    },
    {
        "id": "p09_unsupported_random",
        "name": "P09 - Unseeded Stochastic RNG in Causal Path (Must Reject as UNSUPPORTED)",
        "category": "UNSUPPORTED",
        "sense": "max",
        "expected_status": "UNSUPPORTED",
        "entrypoint": "evaluate_noisy_yield",
        "source_code": '''"""P09: Function invoking unseeded random.uniform() in causal path of target."""
import random

SENSE = "max"
INPUT_DOSE: float = 20.0
INPUT_DOSE_BOUNDS = (5.0, 80.0)

def evaluate_noisy_yield(input_dose: float = 20.0) -> float:
    stochastic_jitter = random.uniform(0.8, 1.2)
    noisy_yield = (15.0 * input_dose - 0.1 * (input_dose ** 2)) * stochastic_jitter
    return noisy_yield
''',
    },
    {
        "id": "p10_unsupported_io_loop",
        "name": "P10 - Opaque Stateful Mutation / I/O Side-Effect (Must Reject as UNSUPPORTED)",
        "category": "UNSUPPORTED",
        "sense": "max",
        "expected_status": "UNSUPPORTED",
        "entrypoint": "evaluate_stateful_accumulator",
        "source_code": '''"""P10: Function mutating external module state across invocations."""
SENSE = "max"
CONTROL_RATE: float = 10.0
CONTROL_RATE_BOUNDS = (1.0, 50.0)

_EXTERNAL_LEDGER = [0.0]

def evaluate_stateful_accumulator(control_rate: float = 10.0) -> float:
    _EXTERNAL_LEDGER.append(control_rate)
    accumulated_output = sum(_EXTERNAL_LEDGER) * 2.5 - 0.05 * (control_rate ** 2)
    return accumulated_output
''',
    },
]


def get_public_problems() -> List[Dict[str, Any]]:
    out = []
    for p in PUBLIC_PROBLEMS:
        item = dict(p)
        item["oracle"] = ORACLE_REGISTRY[p["id"]]()
        out.append(item)
    return out
