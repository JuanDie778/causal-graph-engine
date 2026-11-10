"""10 Problemas Sellados Holdout de Investigación Operativa y Modelado de Decisiones (OR-Bench Sellado, Fase 10).

Cada problema está escrito como código Python natural o pipeline DataFrame de Pandas,
con cero solapamiento con la plantilla de la tostadora de café ni con los problemas públicos.
"""
from __future__ import annotations
from typing import Dict, Any, List
from .oracles import ORACLE_REGISTRY


SEALED_PROBLEMS: List[Dict[str, Any]] = [
    {
        "id": "s01_transportation_3x4",
        "name": "S01 - 3x4 Hitchcock Transportation LP (12 Flow Variables)",
        "category": "LP",
        "sense": "min",
        "expected_status": "CERTIFIED_OPTIMAL",
        "entrypoint": "evaluate_transport_cost",
        "source_code": '''"""S01: 3-supply x 4-demand Hitchcock transportation LP."""
SENSE = "min"
FLOW_1_1: float = 20.0
FLOW_1_2: float = 20.0
FLOW_1_3: float = 20.0
FLOW_1_4: float = 20.0
FLOW_2_1: float = 20.0
FLOW_2_2: float = 20.0
FLOW_2_3: float = 20.0
FLOW_2_4: float = 20.0
FLOW_3_1: float = 20.0
FLOW_3_2: float = 20.0
FLOW_3_3: float = 20.0
FLOW_3_4: float = 20.0

FLOW_1_1_BOUNDS = (0.0, 200.0)
FLOW_1_2_BOUNDS = (0.0, 200.0)
FLOW_1_3_BOUNDS = (0.0, 200.0)
FLOW_1_4_BOUNDS = (0.0, 200.0)
FLOW_2_1_BOUNDS = (0.0, 200.0)
FLOW_2_2_BOUNDS = (0.0, 200.0)
FLOW_2_3_BOUNDS = (0.0, 200.0)
FLOW_2_4_BOUNDS = (0.0, 200.0)
FLOW_3_1_BOUNDS = (0.0, 200.0)
FLOW_3_2_BOUNDS = (0.0, 200.0)
FLOW_3_3_BOUNDS = (0.0, 200.0)
FLOW_3_4_BOUNDS = (0.0, 200.0)

CONSTRAINTS = {
    "c_sup_1": "flow_1_1 + flow_1_2 + flow_1_3 + flow_1_4 <= 150.0",
    "c_sup_2": "flow_2_1 + flow_2_2 + flow_2_3 + flow_2_4 <= 200.0",
    "c_sup_3": "flow_3_1 + flow_3_2 + flow_3_3 + flow_3_4 <= 150.0",
    "c_dem_1": "flow_1_1 + flow_2_1 + flow_3_1 >= 110.0",
    "c_dem_2": "flow_1_2 + flow_2_2 + flow_3_2 >= 130.0",
    "c_dem_3": "flow_1_3 + flow_2_3 + flow_3_3 >= 120.0",
    "c_dem_4": "flow_1_4 + flow_2_4 + flow_3_4 >= 100.0",
}

def evaluate_transport_cost(
    flow_1_1: float = 20.0, flow_1_2: float = 20.0, flow_1_3: float = 20.0, flow_1_4: float = 20.0,
    flow_2_1: float = 20.0, flow_2_2: float = 20.0, flow_2_3: float = 20.0, flow_2_4: float = 20.0,
    flow_3_1: float = 20.0, flow_3_2: float = 20.0, flow_3_3: float = 20.0, flow_3_4: float = 20.0,
) -> float:
    total_shipping_cost = (
        8.0 * flow_1_1 + 6.0 * flow_1_2 + 10.0 * flow_1_3 + 9.0 * flow_1_4
        + 9.0 * flow_2_1 + 12.0 * flow_2_2 + 13.0 * flow_2_3 + 7.0 * flow_2_4
        + 14.0 * flow_3_1 + 9.0 * flow_3_2 + 16.0 * flow_3_3 + 5.0 * flow_3_4
    )
    return total_shipping_cost
''',
    },
    {
        "id": "s02_markowitz_qp",
        "name": "S02 - 4-Asset Markowitz Portfolio Variance Minimization QP",
        "category": "QP",
        "sense": "min",
        "expected_status": "CERTIFIED_OPTIMAL",
        "entrypoint": "evaluate_portfolio_risk",
        "source_code": '''"""S02: 4-asset Markowitz portfolio allocation QP."""
SENSE = "min"
W_1: float = 0.25
W_2: float = 0.25
W_3: float = 0.25
W_4: float = 0.25

W_1_BOUNDS = (0.0, 1.0)
W_2_BOUNDS = (0.0, 1.0)
W_3_BOUNDS = (0.0, 1.0)
W_4_BOUNDS = (0.0, 1.0)

CONSTRAINTS = {
    "c_full_investment": "w_1 + w_2 + w_3 + w_4 == 1.0",
    "c_target_return": "0.08 * w_1 + 0.12 * w_2 + 0.15 * w_3 + 0.18 * w_4 >= 0.135",
}

def evaluate_portfolio_risk(
    w_1: float = 0.25,
    w_2: float = 0.25,
    w_3: float = 0.25,
    w_4: float = 0.25,
) -> float:
    var_diag = 0.04 * (w_1 ** 2) + 0.09 * (w_2 ** 2) + 0.16 * (w_3 ** 2) + 0.25 * (w_4 ** 2)
    cov_cross = 2.0 * (
        0.01 * w_1 * w_2 + 0.005 * w_1 * w_3 + 0.002 * w_1 * w_4
        + 0.02 * w_2 * w_3 + 0.01 * w_2 * w_4 + 0.03 * w_3 * w_4
    )
    scaled_portfolio_risk = 1000.0 * (var_diag + cov_cross)
    return scaled_portfolio_risk
''',
    },
    {
        "id": "s03_facility_location_milp",
        "name": "S03 - Capacitated Facility Location MILP (3 Binary + 6 Continuous)",
        "category": "MILP",
        "sense": "min",
        "expected_status": "CERTIFIED_OPTIMAL",
        "entrypoint": "evaluate_facility_network",
        "source_code": '''"""S03: Capacitated Facility Location MILP with 3 binary facilities and 2 demand zones."""
SENSE = "min"
OPEN_1: int = 1
OPEN_2: int = 1
OPEN_3: int = 1
SHIP_1_1: float = 40.0
SHIP_1_2: float = 40.0
SHIP_2_1: float = 40.0
SHIP_2_2: float = 50.0
SHIP_3_1: float = 30.0
SHIP_3_2: float = 40.0

OPEN_1_BOUNDS = (0, 1)
OPEN_2_BOUNDS = (0, 1)
OPEN_3_BOUNDS = (0, 1)
SHIP_1_1_BOUNDS = (0.0, 120.0)
SHIP_1_2_BOUNDS = (0.0, 120.0)
SHIP_2_1_BOUNDS = (0.0, 160.0)
SHIP_2_2_BOUNDS = (0.0, 160.0)
SHIP_3_1_BOUNDS = (0.0, 140.0)
SHIP_3_2_BOUNDS = (0.0, 140.0)

CONSTRAINTS = {
    "c_cap_1": "ship_1_1 + ship_1_2 <= 120.0 * open_1",
    "c_cap_2": "ship_2_1 + ship_2_2 <= 160.0 * open_2",
    "c_cap_3": "ship_3_1 + ship_3_2 <= 140.0 * open_3",
    "c_zone_1": "ship_1_1 + ship_2_1 + ship_3_1 >= 110.0",
    "c_zone_2": "ship_1_2 + ship_2_2 + ship_3_2 >= 130.0",
}

def evaluate_facility_network(
    open_1: int = 1, open_2: int = 1, open_3: int = 1,
    ship_1_1: float = 40.0, ship_1_2: float = 40.0,
    ship_2_1: float = 40.0, ship_2_2: float = 50.0,
    ship_3_1: float = 30.0, ship_3_2: float = 40.0,
) -> float:
    fixed_facility_cost = 400.0 * open_1 + 500.0 * open_2 + 450.0 * open_3
    variable_transit_cost = (
        4.0 * ship_1_1 + 7.0 * ship_1_2
        + 6.0 * ship_2_1 + 3.0 * ship_2_2
        + 5.0 * ship_3_1 + 4.0 * ship_3_2
    )
    total_network_cost = fixed_facility_cost + variable_transit_cost
    return total_network_cost
''',
    },
    {
        "id": "s04_staff_scheduling_ilp",
        "name": "S04 - 6-Period Cyclic Shift Covering Integer Linear Program",
        "category": "MILP",
        "sense": "min",
        "expected_status": "CERTIFIED_OPTIMAL",
        "entrypoint": "evaluate_shift_roster",
        "source_code": '''"""S04: Cyclic shift covering ILP with 6 integer shift headcount variables."""
SENSE = "min"
SHIFT_1: int = 12
SHIFT_2: int = 12
SHIFT_3: int = 12
SHIFT_4: int = 12
SHIFT_5: int = 12
SHIFT_6: int = 12

SHIFT_1_BOUNDS = (0, 30)
SHIFT_2_BOUNDS = (0, 30)
SHIFT_3_BOUNDS = (0, 30)
SHIFT_4_BOUNDS = (0, 30)
SHIFT_5_BOUNDS = (0, 30)
SHIFT_6_BOUNDS = (0, 30)

CONSTRAINTS = {
    "c_cov_p1": "shift_1 + shift_6 >= 14",
    "c_cov_p2": "shift_2 + shift_1 >= 18",
    "c_cov_p3": "shift_3 + shift_2 >= 22",
    "c_cov_p4": "shift_4 + shift_3 >= 20",
    "c_cov_p5": "shift_5 + shift_4 >= 16",
    "c_cov_p6": "shift_6 + shift_5 >= 12",
}

def evaluate_shift_roster(
    shift_1: int = 12,
    shift_2: int = 12,
    shift_3: int = 12,
    shift_4: int = 12,
    shift_5: int = 12,
    shift_6: int = 12,
) -> float:
    roster_payroll = (
        180.0 * shift_1 + 195.0 * shift_2 + 210.0 * shift_3
        + 225.0 * shift_4 + 240.0 * shift_5 + 200.0 * shift_6
    )
    return roster_payroll
''',
    },
    {
        "id": "s05_logit_pricing_nlp",
        "name": "S05 - 2-Product Multinomial Logit Smooth Nonlinear Pricing",
        "category": "NLP_SMOOTH",
        "sense": "max",
        "expected_status": "CERTIFIED_OPTIMAL",
        "entrypoint": "evaluate_logit_margin",
        "source_code": '''"""S05: Smooth non-convex 2-product Multinomial Logit choice pricing."""
import math

SENSE = "max"
TARIFF_A: float = 15.0
TARIFF_B: float = 20.0

TARIFF_A_BOUNDS = (8.0, 35.0)
TARIFF_B_BOUNDS = (11.0, 40.0)

CONSTRAINTS = {
    "c_bundle_cap": "tariff_a + tariff_b <= 52.0",
}

def evaluate_logit_margin(tariff_a: float = 15.0, tariff_b: float = 20.0) -> float:
    util_a = math.exp(3.2 - 0.18 * tariff_a)
    util_b = math.exp(3.8 - 0.15 * tariff_b)
    denom = 1.0 + util_a + util_b
    share_a = 1000.0 * (util_a / denom)
    share_b = 1000.0 * (util_b / denom)
    logit_contribution = (tariff_a - 8.0) * share_a + (tariff_b - 11.0) * share_b
    return logit_contribution
''',
    },
    {
        "id": "s06_multiperiod_inventory",
        "name": "S06 - 4-Period Dynamic Inventory-Production Balance (Pandas Pipeline)",
        "category": "LP",
        "sense": "min",
        "expected_status": "CERTIFIED_OPTIMAL",
        "entrypoint": "run_inventory_pipeline",
        "source_code": '''"""S06: 4-period dynamic inventory balance modeled as a Pandas DataFrame pipeline."""
SENSE = "min"
LOT_1: float = 80.0
LOT_2: float = 85.0
LOT_3: float = 85.0
LOT_4: float = 90.0

LOT_1_BOUNDS = (0.0, 110.0)
LOT_2_BOUNDS = (0.0, 110.0)
LOT_3_BOUNDS = (0.0, 110.0)
LOT_4_BOUNDS = (0.0, 110.0)

CONSTRAINTS = {
    "c_inv1_nonneg": "20.0 + lot_1 - 70.0 >= 0.0",
    "c_inv1_cap": "20.0 + lot_1 - 70.0 <= 60.0",
    "c_inv2_nonneg": "20.0 + lot_1 + lot_2 - 160.0 >= 0.0",
    "c_inv2_cap": "20.0 + lot_1 + lot_2 - 160.0 <= 60.0",
    "c_inv3_nonneg": "20.0 + lot_1 + lot_2 + lot_3 - 240.0 >= 0.0",
    "c_inv3_cap": "20.0 + lot_1 + lot_2 + lot_3 - 240.0 <= 60.0",
    "c_inv4_nonneg": "20.0 + lot_1 + lot_2 + lot_3 + lot_4 - 340.0 >= 0.0",
    "c_inv4_cap": "20.0 + lot_1 + lot_2 + lot_3 + lot_4 - 340.0 <= 60.0",
}

def run_inventory_pipeline(df):
    df["inv_1"] = 20.0 + df["lot_1"] - 70.0
    df["inv_2"] = df["inv_1"] + df["lot_2"] - 90.0
    df["inv_3"] = df["inv_2"] + df["lot_3"] - 80.0
    df["inv_4"] = df["inv_3"] + df["lot_4"] - 100.0
    df["prod_outlay"] = 10.0 * df["lot_1"] + 12.0 * df["lot_2"] + 11.0 * df["lot_3"] + 13.0 * df["lot_4"]
    df["carry_outlay"] = 1.5 * (df["inv_1"] + df["inv_2"] + df["inv_3"] + df["inv_4"])
    df["horizon_total_cost"] = df["prod_outlay"] + df["carry_outlay"]
    return df
''',
    },
    {
        "id": "s07_deep_nested_minmax",
        "name": "S07 - 10-Atom Deep Nested Min/Max Network Reliability Tree (K=10 > 8)",
        "category": "NLP_NONSMOOTH",
        "sense": "max",
        "expected_status": "CERTIFIED_OPTIMAL",
        "entrypoint": "evaluate_network_tree",
        "source_code": '''"""S07: Deeply nested 10-atom min/max network reliability tree (K=10 > 8 nonsmooth atoms)."""
SENSE = "max"
LINK_1: float = 20.0
LINK_2: float = 20.0
LINK_3: float = 20.0
LINK_4: float = 20.0
LINK_5: float = 20.0
LINK_6: float = 20.0

LINK_1_BOUNDS = (0.0, 80.0)
LINK_2_BOUNDS = (0.0, 80.0)
LINK_3_BOUNDS = (0.0, 80.0)
LINK_4_BOUNDS = (0.0, 80.0)
LINK_5_BOUNDS = (0.0, 80.0)
LINK_6_BOUNDS = (0.0, 80.0)

CONSTRAINTS = {
    "c_link_budget": "2.0 * link_1 + 2.5 * link_2 + 1.8 * link_3 + 2.2 * link_4 + 3.0 * link_5 + 1.5 * link_6 <= 300.0",
}

def evaluate_network_tree(
    link_1: float = 20.0,
    link_2: float = 20.0,
    link_3: float = 20.0,
    link_4: float = 20.0,
    link_5: float = 20.0,
    link_6: float = 20.0,
) -> float:
    c1 = 20.0 + link_1
    c2 = 15.0 + link_2
    c3 = 25.0 + link_3
    c4 = 18.0 + link_4
    c5 = 12.0 + link_5
    c6 = 22.0 + link_6

    b1 = min(c1, c2)
    b2 = min(c3, c4)
    b3 = min(c5, c6)
    b4 = min(b1, b2)
    flow_core = min(b4, b3)

    surge_1 = max(0.0, c1 - 45.0)
    surge_2 = max(0.0, c3 - 50.0)
    surge_cap = min(surge_1, surge_2)

    pen_1 = max(0.0, 35.0 - c2)
    pen_2 = max(0.0, 35.0 - c5)
    pen_tot = max(pen_1, pen_2)

    spend = 2.0 * link_1 + 2.5 * link_2 + 1.8 * link_3 + 2.2 * link_4 + 3.0 * link_5 + 1.5 * link_6
    net_network_score = 40.0 * flow_core + 5.0 * surge_cap - 8.0 * pen_tot - 0.5 * spend
    return net_network_score
''',
    },
    {
        "id": "s08_infeasible_equality_iis",
        "name": "S08 - Infeasible Blending Equality + Upper Sub-Capacities IIS",
        "category": "LP",
        "sense": "min",
        "expected_status": "INFEASIBLE_CERTIFIED",
        "entrypoint": "evaluate_flow_mix",
        "source_code": '''"""S08: Infeasible equality and sub-capacity system with minimal 3-constraint IIS."""
SENSE = "min"
FLOW_A: float = 30.0
FLOW_B: float = 30.0
FLOW_C: float = 30.0

FLOW_A_BOUNDS = (0.0, 100.0)
FLOW_B_BOUNDS = (0.0, 100.0)
FLOW_C_BOUNDS = (0.0, 100.0)

CONSTRAINTS = {
    "c_sum_eq": "flow_a + flow_b + flow_c == 100.0",
    "c_cap_ab": "flow_a + flow_b <= 55.0",
    "c_cap_c": "flow_c <= 30.0",
    "c_wide_bound": "flow_a + 2.0 * flow_b + flow_c <= 250.0",
    "c_min_a_zero": "flow_a >= 5.0",
}

def evaluate_flow_mix(flow_a: float = 30.0, flow_b: float = 30.0, flow_c: float = 30.0) -> float:
    mix_expense = 3.0 * flow_a + 4.0 * flow_b + 2.0 * flow_c
    return mix_expense
''',
    },
    {
        "id": "s09_infeasible_discrete_iis",
        "name": "S09 - Infeasible Discrete Integer Capacity System IIS",
        "category": "MILP",
        "sense": "max",
        "expected_status": "INFEASIBLE_CERTIFIED",
        "entrypoint": "evaluate_discrete_modules",
        "source_code": '''"""S09: Infeasible integer problem with conflicting discrete module bounds."""
SENSE = "max"
UNITS_X: int = 4
UNITS_Y: int = 4
UNITS_Z: int = 2

UNITS_X_BOUNDS = (0, 10)
UNITS_Y_BOUNDS = (0, 10)
UNITS_Z_BOUNDS = (0, 10)

CONSTRAINTS = {
    "c_min_units": "units_x + units_y >= 12",
    "c_max_x": "units_x <= 5",
    "c_max_y": "units_y <= 5",
    "c_aux_z": "units_z <= 8",
}

def evaluate_discrete_modules(units_x: int = 4, units_y: int = 4, units_z: int = 2) -> float:
    module_output = 25.0 * units_x + 30.0 * units_y + 10.0 * units_z
    return module_output
''',
    },
    {
        "id": "s10_unsupported_external_state",
        "name": "S10 - Global Dictionary State Mutation Across Calls (Must Reject as UNSUPPORTED)",
        "category": "UNSUPPORTED",
        "sense": "max",
        "expected_status": "UNSUPPORTED",
        "entrypoint": "evaluate_cached_dispatch",
        "source_code": '''"""S10: Function mutating a global cache dictionary across calls."""
SENSE = "max"
DISPATCH_VOL: float = 25.0
DISPATCH_VOL_BOUNDS = (5.0, 100.0)

_GLOBAL_STATE_CACHE = {"calls": 0, "drift": 0.0}

def evaluate_cached_dispatch(dispatch_vol: float = 25.0) -> float:
    _GLOBAL_STATE_CACHE["calls"] += 1
    _GLOBAL_STATE_CACHE["drift"] += 0.25 * dispatch_vol
    effective_gain = 14.0 * dispatch_vol - 0.08 * (dispatch_vol ** 2) + _GLOBAL_STATE_CACHE["drift"]
    return effective_gain
''',
    },
]


def get_sealed_problems() -> List[Dict[str, Any]]:
    out = []
    for p in SEALED_PROBLEMS:
        item = dict(p)
        item["oracle"] = ORACLE_REGISTRY[p["id"]]()
        out.append(item)
    return out
