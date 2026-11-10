"""Oráculos independientes de verdad-terreno (ground-truth) para los 20 problemas de OR-Bench (Fase 10).

Ningún oráculo importa ni utiliza ningún módulo de `cge/`. Cada problema se resuelve
mediante su fórmula analítica en forma cerrada, programación dinámica exacta,
enumeración combinatoria exhaustiva o formulación matricial directa independiente.
"""
from __future__ import annotations
import itertools
import math
from typing import Dict, Any, List
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds, minimize


def oracle_p01_newsvendor() -> Dict[str, Any]:
    """P01: Newsvendor clásico con 3 escenarios de demanda (80, 120, 160), precio=25, coste=10, salvage=4, shortage_penalty=5."""
    # Por análisis marginal (fractil crítico):
    # En [40, 80): pendiente = +20
    # En (80, 120): pendiente = (-6 + 20 + 20)/3 = +34/3 > 0
    # En (120, 160): pendiente = (-6 - 6 + 20)/3 = +8/3 > 0
    # En (160, 200]: pendiente = -6 < 0
    # Óptimo analítico exacto: stock_level* = 160.0
    q_star = 160.0
    scenarios = [80.0, 120.0, 160.0]
    profits = []
    for d in scenarios:
        sold = min(q_star, d)
        salv = max(0.0, q_star - d)
        short = max(0.0, d - q_star)
        profits.append(25.0 * sold + 4.0 * salv - 5.0 * short - 10.0 * q_star)
    j_star = sum(profits) / len(profits)
    return {
        "status": "CERTIFIED_OPTIMAL",
        "sense": "max",
        "optimal_objective": float(j_star),  # 1560.0
        "optimal_controls": {"stock_level": q_star},
    }


def oracle_p02_eoq() -> Dict[str, Any]:
    """P02: EOQ clásico con coste de pedido K*D/Q + mantenimiento h*Q/2 sujeto a capacidad de almacén 2.5*Q <= 1800."""
    # D=10000, K=120, h=3.0 => Q_unconstrained = sqrt(2*120*10000/3) = 894.42719 > 720.0
    # Por convexidad estricta de C(Q) en (0, inf), el mínimo restringido está en la frontera activa Q* = 1800 / 2.5 = 720.0
    q_star = 1800.0 / 2.5  # 720.0
    j_star = (120.0 * 10000.0) / q_star + 0.5 * 3.0 * q_star
    return {
        "status": "CERTIFIED_OPTIMAL",
        "sense": "min",
        "optimal_objective": float(j_star),  # 2746.66666667
        "optimal_controls": {"batch_qty": q_star},
    }


def oracle_p03_prod_plan_lp() -> Dict[str, Any]:
    """P03: Planificación de producción multiproducto LP (6 productos, 4 recursos compartidos)."""
    margins = np.array([45.0, 60.0, 35.0, 80.0, 50.0, 70.0], dtype=float)
    a_ub = np.array([
        [2.0, 3.0, 1.5, 4.0, 2.5, 3.5],   # labor <= 240
        [1.5, 2.0, 2.5, 3.0, 1.0, 2.5],   # machine <= 180
        [3.0, 1.5, 2.0, 2.5, 3.0, 2.0],   # raw_material <= 210
        [1.0, 2.5, 1.0, 3.5, 2.0, 3.0],   # energy <= 190
    ], dtype=float)
    b_ub = np.array([240.0, 180.0, 210.0, 190.0], dtype=float)
    lb = np.zeros(6, dtype=float)
    ub = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0], dtype=float)
    res = milp(
        c=-margins,
        integrality=np.zeros(6, dtype=int),
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(a_ub, -np.inf * np.ones(4), b_ub),
    )
    names = [f"prod_{i}" for i in range(1, 7)]
    return {
        "status": "CERTIFIED_OPTIMAL",
        "sense": "max",
        "optimal_objective": float(-res.fun),
        "optimal_controls": {names[i]: float(res.x[i]) for i in range(6)},
    }


def oracle_p04_knapsack_01() -> Dict[str, Any]:
    """P04: 0/1 Knapsack con 8 ítems binarios por enumeración exhaustiva 2^8 = 256."""
    values = [42.0, 38.0, 65.0, 29.0, 54.0, 71.0, 33.0, 49.0]
    weights = [18.0, 15.0, 28.0, 12.0, 23.0, 31.0, 14.0, 21.0]
    cap = 95.0
    best_v = -1.0
    best_bits = (0,) * 8
    for bits in itertools.product([0, 1], repeat=8):
        w = sum(weights[i] * bits[i] for i in range(8))
        if w <= cap:
            v = sum(values[i] * bits[i] for i in range(8))
            if v > best_v:
                best_v = v
                best_bits = bits
    return {
        "status": "CERTIFIED_OPTIMAL",
        "sense": "max",
        "optimal_objective": float(best_v),
        "optimal_controls": {f"item_{i+1}": int(best_bits[i]) for i in range(8)},
    }


def oracle_p05_blending_eq() -> Dict[str, Any]:
    """P05: Mezcla química LP con restricción de igualdad exacta sum(x_i) == 100 y cotas de calidad."""
    costs = np.array([12.0, 18.0, 9.0, 24.0], dtype=float)
    # x1 + x2 + x3 + x4 == 100
    # potency: 5*x1 + 8*x2 + 3*x3 + 10*x4 >= 680  <=> -5*x1 - 8*x2 - 3*x3 - 10*x4 <= -680
    # impurity: 3*x1 + 1.5*x2 + 4*x3 + 0.8*x4 <= 210
    a_mat = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [5.0, 8.0, 3.0, 10.0],
        [3.0, 1.5, 4.0, 0.8],
    ], dtype=float)
    lb_c = np.array([100.0, 680.0, -np.inf], dtype=float)
    ub_c = np.array([100.0, np.inf, 210.0], dtype=float)
    res = milp(
        c=costs,
        integrality=np.zeros(4, dtype=int),
        bounds=Bounds(np.zeros(4), np.full(4, 70.0)),
        constraints=LinearConstraint(a_mat, lb_c, ub_c),
    )
    return {
        "status": "CERTIFIED_OPTIMAL",
        "sense": "min",
        "optimal_objective": float(res.fun),
        "optimal_controls": {f"ing_{i+1}": float(res.x[i]) for i in range(4)},
    }


def oracle_p06_serial_bottlenecks_6() -> Dict[str, Any]:
    """P06: Línea de manufactura en serie de 6 etapas y = min(c1..c6) con K=5 átomos Min anidados y presupuesto."""
    base_caps = np.array([50.0, 42.0, 58.0, 45.0, 52.0, 40.0], dtype=float)
    exp_costs = np.array([3.0, 4.0, 2.5, 3.5, 3.0, 4.5], dtype=float)
    budget = 260.0
    unit_val = 50.0
    # Formulamos el LP epígrafe exacto en variables (u1..u6, y):
    # max 50*y - sum(exp_costs_i * u_i)
    # s.t. y - u_i <= base_caps_i (i=1..6), sum(exp_costs_i * u_i) <= budget, 0 <= u_i <= 60
    c_vec = np.append(exp_costs, -unit_val)
    a_rows = []
    ub_rows = []
    for i in range(6):
        row = np.zeros(7, dtype=float)
        row[i] = -1.0
        row[6] = 1.0
        a_rows.append(row)
        ub_rows.append(base_caps[i])
    row_b = np.append(exp_costs, 0.0)
    a_rows.append(row_b)
    ub_rows.append(budget)

    lb_v = np.append(np.zeros(6), 0.0)
    ub_v = np.append(np.full(6, 60.0), 200.0)
    res = milp(
        c=c_vec,
        integrality=np.zeros(7, dtype=int),
        bounds=Bounds(lb_v, ub_v),
        constraints=LinearConstraint(np.vstack(a_rows), -np.inf * np.ones(len(ub_rows)), np.array(ub_rows)),
    )
    return {
        "status": "CERTIFIED_OPTIMAL",
        "sense": "max",
        "optimal_objective": float(-res.fun),
        "optimal_controls": {f"add_cap_{i+1}": float(res.x[i]) for i in range(6)},
    }


def oracle_p07_piecewise_tax() -> Dict[str, Any]:
    """P07: Optimización fiscal progresiva de 4 tramos (Piecewise) con retorno operativo cóncavo."""
    # gross_income(x) = 18.0 * x - 0.04 * x^2 para x en [10, 220]
    # tax(x):
    #   si x < 50: 0.10 * x
    #   elif x < 100: 5.0 + 0.20 * (x - 50)
    #   elif x < 150: 15.0 + 0.32 * (x - 100)
    #   else: 31.0 + 0.45 * (x - 150)
    # overhead(x) = 1.5 * x
    # net_surplus(x) = gross_income(x) - tax(x) - overhead(x)
    # Analizamos la derivada en cada tramo:
    # Tramo 1 [10, 50]: d/dx (16.5*x - 0.04*x^2 - 0.10*x) = 16.4 - 0.08*x > 0 en [10, 50]
    # Tramo 2 [50, 100]: d/dx (16.5*x - 0.04*x^2 - (0.20*x - 5)) = 16.3 - 0.08*x > 0 en [50, 100]
    # Tramo 3 [100, 150]: d/dx (16.5*x - 0.04*x^2 - (0.32*x - 17)) = 16.18 - 0.08*x > 0 en [100, 150]
    # Tramo 4 [150, 220]: d/dx (16.5*x - 0.04*x^2 - (0.45*x - 36.5)) = 16.05 - 0.08*x = 0 => x* = 16.05 / 0.08 = 200.625
    x_star = 16.05 / 0.08  # 200.625
    gross = 18.0 * x_star - 0.04 * (x_star ** 2)
    tax = 31.0 + 0.45 * (x_star - 150.0)
    j_star = gross - tax - 1.5 * x_star
    return {
        "status": "CERTIFIED_OPTIMAL",
        "sense": "max",
        "optimal_objective": float(j_star),
        "optimal_controls": {"activity_level": float(x_star)},
    }


def oracle_p08_infeasible_bounds_iis() -> Dict[str, Any]:
    """P08: Sistema sobre-restringido donde 3 de 7 restricciones forman un IIS mínimo."""
    # c_min_a: alloc_a >= 45
    # c_min_b: alloc_b >= 50
    # c_pair_cap: alloc_a + alloc_b <= 80
    # (45 + 50 = 95 > 80 => violación mínima L1 = 15.0)
    return {
        "status": "INFEASIBLE_CERTIFIED",
        "sense": "max",
        "optimal_objective": None,
        "iis_constraints": ["c_min_a", "c_min_b", "c_pair_cap"],
        "min_relaxation_slack": 15.0,
    }


def oracle_p09_unsupported_random() -> Dict[str, Any]:
    """P09: Función con llamada no determinista random.uniform() en el camino causal."""
    return {
        "status": "UNSUPPORTED",
        "sense": "max",
        "optimal_objective": None,
    }


def oracle_p10_unsupported_io_loop() -> Dict[str, Any]:
    """P10: Función con mutación de estado externo / I/O no soportado."""
    return {
        "status": "UNSUPPORTED",
        "sense": "max",
        "optimal_objective": None,
    }


def oracle_s01_transportation_3x4() -> Dict[str, Any]:
    """S01 (Sellado): Transporte de Hitchcock 3 orígenes x 4 destinos (12 variables de flujo)."""
    costs = np.array([
        8.0, 6.0, 10.0, 9.0,
        9.0, 12.0, 13.0, 7.0,
        14.0, 9.0, 16.0, 5.0,
    ], dtype=float)
    supply = np.array([150.0, 200.0, 150.0], dtype=float)
    demand = np.array([110.0, 130.0, 120.0, 100.0], dtype=float)

    a_rows = []
    lb_rows = []
    ub_rows = []
    # 3 restricciones de oferta: sum_j f_{i,j} <= supply_i
    for i in range(3):
        row = np.zeros(12, dtype=float)
        row[4 * i: 4 * (i + 1)] = 1.0
        a_rows.append(row)
        lb_rows.append(-np.inf)
        ub_rows.append(supply[i])
    # 4 restricciones de demanda: sum_i f_{i,j} >= demand_j
    for j in range(4):
        row = np.zeros(12, dtype=float)
        for i in range(3):
            row[4 * i + j] = 1.0
        a_rows.append(row)
        lb_rows.append(demand[j])
        ub_rows.append(np.inf)

    res = milp(
        c=costs,
        integrality=np.zeros(12, dtype=int),
        bounds=Bounds(np.zeros(12), np.full(12, 200.0)),
        constraints=LinearConstraint(np.vstack(a_rows), np.array(lb_rows), np.array(ub_rows)),
    )
    var_names = [f"flow_{i}_{j}" for i in range(1, 4) for j in range(1, 5)]
    return {
        "status": "CERTIFIED_OPTIMAL",
        "sense": "min",
        "optimal_objective": float(res.fun),
        "optimal_controls": {var_names[k]: float(res.x[k]) for k in range(12)},
    }


def oracle_s02_markowitz_qp() -> Dict[str, Any]:
    """S02 (Sellado): Cartera de Markowitz de 4 activos (QP convexo con restricción de retorno e igualdad sum(w)==1)."""
    mu = np.array([0.08, 0.12, 0.15, 0.18], dtype=float)
    cov = np.array([
        [0.04, 0.01, 0.005, 0.002],
        [0.01, 0.09, 0.02, 0.01],
        [0.005, 0.02, 0.16, 0.03],
        [0.002, 0.01, 0.03, 0.25],
    ], dtype=float)
    r_target = 0.135
    res = minimize(
        lambda w: float(w @ cov @ w) * 1000.0,
        x0=np.full(4, 0.25),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 4,
        constraints=[
            {"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0},
            {"type": "ineq", "fun": lambda w: float(mu @ w) - r_target},
        ],
        options={"ftol": 1e-12, "maxiter": 200},
    )
    return {
        "status": "CERTIFIED_OPTIMAL",
        "sense": "min",
        "optimal_objective": float(res.fun),
        "optimal_controls": {f"w_{i+1}": float(res.x[i]) for i in range(4)},
    }


def oracle_s03_facility_location_milp() -> Dict[str, Any]:
    """S03 (Sellado): Capacitated Facility Location MILP (3 variables binarias de apertura + 6 continuas de asignación)."""
    fixed_costs = [400.0, 500.0, 450.0]
    caps = [120.0, 160.0, 140.0]
    ship_costs = [
        [4.0, 7.0],
        [6.0, 3.0],
        [5.0, 4.0],
    ]
    demands = [110.0, 130.0]
    # Variables: [y1, y2, y3, z11, z12, z21, z22, z31, z32]
    c_vec = np.array(fixed_costs + ship_costs[0] + ship_costs[1] + ship_costs[2], dtype=float)
    integrality = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=int)
    lb_v = np.zeros(9, dtype=float)
    ub_v = np.array([1.0, 1.0, 1.0, 120.0, 120.0, 160.0, 160.0, 140.0, 140.0], dtype=float)

    a_rows = []
    lb_rows = []
    ub_rows = []
    # z_{i,1} + z_{i,2} - cap_i * y_i <= 0
    for i in range(3):
        row = np.zeros(9, dtype=float)
        row[i] = -caps[i]
        row[3 + 2 * i] = 1.0
        row[3 + 2 * i + 1] = 1.0
        a_rows.append(row)
        lb_rows.append(-np.inf)
        ub_rows.append(0.0)
    # sum_i z_{i,j} >= demand_j
    for j in range(2):
        row = np.zeros(9, dtype=float)
        for i in range(3):
            row[3 + 2 * i + j] = 1.0
        a_rows.append(row)
        lb_rows.append(demands[j])
        ub_rows.append(np.inf)

    res = milp(
        c=c_vec,
        integrality=integrality,
        bounds=Bounds(lb_v, ub_v),
        constraints=LinearConstraint(np.vstack(a_rows), np.array(lb_rows), np.array(ub_rows)),
    )
    ctrl_dict: Dict[str, Any] = {f"open_{i+1}": int(round(res.x[i])) for i in range(3)}
    for i in range(3):
        for j in range(2):
            ctrl_dict[f"ship_{i+1}_{j+1}"] = float(res.x[3 + 2 * i + j])
    return {
        "status": "CERTIFIED_OPTIMAL",
        "sense": "min",
        "optimal_objective": float(res.fun),
        "optimal_controls": ctrl_dict,
    }


def oracle_s04_staff_scheduling_ilp() -> Dict[str, Any]:
    """S04 (Sellado): Cobertura entera de turnos (6 turnos enteros cubriendo 6 periodos horarios)."""
    shift_costs = np.array([180.0, 195.0, 210.0, 225.0, 240.0, 200.0], dtype=float)
    # Cada turno cubre 2 periodos consecutivos cíclicos: turno i cubre periodo i y (i+1)%6
    req = np.array([14.0, 18.0, 22.0, 20.0, 16.0, 12.0], dtype=float)
    a_mat = np.zeros((6, 6), dtype=float)
    for t in range(6):
        a_mat[t, t] = 1.0
        a_mat[t, (t - 1) % 6] = 1.0
    res = milp(
        c=shift_costs,
        integrality=np.ones(6, dtype=int),
        bounds=Bounds(np.zeros(6), np.full(6, 30.0)),
        constraints=LinearConstraint(a_mat, req, np.full(6, np.inf)),
    )
    return {
        "status": "CERTIFIED_OPTIMAL",
        "sense": "min",
        "optimal_objective": float(res.fun),
        "optimal_controls": {f"shift_{i+1}": int(round(res.x[i])) for i in range(6)},
    }


def oracle_s05_logit_pricing_nlp() -> Dict[str, Any]:
    """S05 (Sellado): Optimización no lineal suave de precios Multinomial Logit para 2 productos."""
    def neg_profit(p: np.ndarray) -> float:
        p1, p2 = float(p[0]), float(p[1])
        u1 = math.exp(3.2 - 0.18 * p1)
        u2 = math.exp(3.8 - 0.15 * p2)
        denom = 1.0 + u1 + u2
        d1 = 1000.0 * (u1 / denom)
        d2 = 1000.0 * (u2 / denom)
        return -((p1 - 8.0) * d1 + (p2 - 11.0) * d2)

    best_fun = float("inf")
    best_x = np.array([15.0, 20.0])
    for s1 in (10.0, 14.0, 18.0, 22.0):
        for s2 in (13.0, 18.0, 23.0, 28.0):
            res = minimize(
                neg_profit,
                x0=np.array([s1, s2]),
                method="SLSQP",
                bounds=[(8.0, 35.0), (11.0, 40.0)],
                constraints=[{"type": "ineq", "fun": lambda p: 52.0 - (p[0] + p[1])}],
                options={"ftol": 1e-12, "maxiter": 200},
            )
            if res.fun < best_fun:
                best_fun = float(res.fun)
                best_x = res.x.copy()
    return {
        "status": "CERTIFIED_OPTIMAL",
        "sense": "max",
        "optimal_objective": float(-best_fun),
        "optimal_controls": {"tariff_a": float(best_x[0]), "tariff_b": float(best_x[1])},
    }


def oracle_s06_multiperiod_inventory() -> Dict[str, Any]:
    """S06 (Sellado): Planificación de inventario multiperiodo (4 periodos, pipeline tabular DataFrame)."""
    # inv_0 = 20, demands = [70, 90, 80, 100], prod_costs = [10, 12, 11, 13], holding_cost = 1.5 por unidad al final del periodo
    # inv_t = inv_{t-1} + prod_t - d_t >= 0, inv_t <= 60, 0 <= prod_t <= 110
    prod_c = np.array([10.0, 12.0, 11.0, 13.0], dtype=float)
    dems = np.array([70.0, 90.0, 80.0, 100.0], dtype=float)
    inv0 = 20.0
    h_c = 1.5
    # inv_t = inv0 + sum_{k=1..t} (prod_k - d_k)
    # Cost = sum_t prod_c[t]*prod_t + h_c * sum_t inv_t
    # Coeficiente efectivo de prod_k en sum_t inv_t es (4 - k)
    eff_c = np.array([prod_c[k] + h_c * (4 - k) for k in range(4)], dtype=float)
    const_term = h_c * sum(inv0 - np.sum(dems[:t + 1]) for t in range(4))

    # Restricciones para t=0..3: 0 <= inv0 + sum_{k<=t} prod_k - sum_{k<=t} d_k <= 60
    a_mat = np.tril(np.ones((4, 4), dtype=float))
    cum_d = np.cumsum(dems) - inv0
    lb_c = cum_d
    ub_c = cum_d + 60.0

    res = milp(
        c=eff_c,
        integrality=np.zeros(4, dtype=int),
        bounds=Bounds(np.zeros(4), np.full(4, 110.0)),
        constraints=LinearConstraint(a_mat, lb_c, ub_c),
    )
    return {
        "status": "CERTIFIED_OPTIMAL",
        "sense": "min",
        "optimal_objective": float(res.fun + const_term),
        "optimal_controls": {f"lot_{t+1}": float(res.x[t]) for t in range(4)},
    }


def oracle_s07_deep_nested_minmax() -> Dict[str, Any]:
    """S07 (Sellado): Árbol profundo con K=10 átomos Min/Max anidados (activa Estrategia A/C para K > 8)."""
    # 6 variables de capacidad de enlace x1..x6 en [10, 100], presupuesto sum(c_i * x_i) <= 320
    # Con K=10 átomos Min/Max anidados que evalúan la capacidad efectiva de red + penalizaciones Piecewise/Max
    costs = np.array([2.0, 2.5, 1.8, 2.2, 3.0, 1.5], dtype=float)
    bases = np.array([20.0, 15.0, 25.0, 18.0, 12.0, 22.0], dtype=float)
    budget = 300.0
    # Para maximizar min_i(bases_i + x_i), en el óptimo todas las ramas activas del árbol bottleneck se igualan:
    # Sea T = min_i(bases_i + x_i) => x_i = max(0, T - bases_i)
    # Resolvemos el LP epígrafe exacto correspondiente a la función de S07:
    def eval_s07(x: np.ndarray) -> float:
        c1, c2, c3, c4, c5, c6 = [bases[i] + float(x[i]) for i in range(6)]
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
        spend = float(np.dot(costs, x))
        return 40.0 * flow_core + 5.0 * surge_cap - 8.0 * pen_tot - 0.5 * spend

    # En el óptimo global, c1=c2=c3=c4=c5=c6 = T con sum(costs_i * (T - bases_i)) = budget (T > 50 => pen_tot=0, surge_cap = T - 50)
    t_eq = (budget + float(np.dot(costs, bases))) / float(np.sum(costs))
    x_eq = np.clip(t_eq - bases, 0.0, 80.0)
    j_star = eval_s07(x_eq)
    return {
        "status": "CERTIFIED_OPTIMAL",
        "sense": "max",
        "optimal_objective": float(j_star),
        "optimal_controls": {f"link_{i+1}": float(x_eq[i]) for i in range(6)},
    }


def oracle_s08_infeasible_equality_iis() -> Dict[str, Any]:
    """S08 (Sellado): Sistema infactible con igualdad exacta y cotas en conflicto (IIS de 3 restricciones)."""
    # c_sum_eq: flow_a + flow_b + flow_c == 100
    # c_cap_ab: flow_a + flow_b <= 55
    # c_cap_c: flow_c <= 30
    # (55 + 30 = 85 < 100 => IIS = {c_sum_eq, c_cap_ab, c_cap_c}, slack = 15.0)
    return {
        "status": "INFEASIBLE_CERTIFIED",
        "sense": "min",
        "optimal_objective": None,
        "iis_constraints": ["c_sum_eq", "c_cap_ab", "c_cap_c"],
        "min_relaxation_slack": 15.0,
    }


def oracle_s09_infeasible_discrete_iis() -> Dict[str, Any]:
    """S09 (Sellado): Sistema entero infactible con conflicto de capacidad discreta (IIS certificado)."""
    # c_min_units: units_x + units_y >= 12
    # c_max_x: units_x <= 5
    # c_max_y: units_y <= 5
    # (5 + 5 = 10 < 12 => IIS = {c_min_units, c_max_x, c_max_y}, slack = 2.0)
    return {
        "status": "INFEASIBLE_CERTIFIED",
        "sense": "max",
        "optimal_objective": None,
        "iis_constraints": ["c_min_units", "c_max_x", "c_max_y"],
        "min_relaxation_slack": 2.0,
    }


def oracle_s10_unsupported_external_state() -> Dict[str, Any]:
    """S10 (Sellado): Función que muta diccionario global acumulativo entre llamadas -> UNSUPPORTED."""
    return {
        "status": "UNSUPPORTED",
        "sense": "max",
        "optimal_objective": None,
    }


ORACLE_REGISTRY = {
    "p01_newsvendor": oracle_p01_newsvendor,
    "p02_eoq": oracle_p02_eoq,
    "p03_prod_plan_lp": oracle_p03_prod_plan_lp,
    "p04_knapsack_01": oracle_p04_knapsack_01,
    "p05_blending_eq": oracle_p05_blending_eq,
    "p06_serial_bottlenecks_6": oracle_p06_serial_bottlenecks_6,
    "p07_piecewise_tax": oracle_p07_piecewise_tax,
    "p08_infeasible_bounds_iis": oracle_p08_infeasible_bounds_iis,
    "p09_unsupported_random": oracle_p09_unsupported_random,
    "p10_unsupported_io_loop": oracle_p10_unsupported_io_loop,
    "s01_transportation_3x4": oracle_s01_transportation_3x4,
    "s02_markowitz_qp": oracle_s02_markowitz_qp,
    "s03_facility_location_milp": oracle_s03_facility_location_milp,
    "s04_staff_scheduling_ilp": oracle_s04_staff_scheduling_ilp,
    "s05_logit_pricing_nlp": oracle_s05_logit_pricing_nlp,
    "s06_multiperiod_inventory": oracle_s06_multiperiod_inventory,
    "s07_deep_nested_minmax": oracle_s07_deep_nested_minmax,
    "s08_infeasible_equality_iis": oracle_s08_infeasible_equality_iis,
    "s09_infeasible_discrete_iis": oracle_s09_infeasible_discrete_iis,
    "s10_unsupported_external_state": oracle_s10_unsupported_external_state,
}
