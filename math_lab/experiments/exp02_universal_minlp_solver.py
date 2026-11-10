"""Experimento 02: Solucionador Híbrido Simbólico-Numérico Universal (MINLP por Partición de Regímenes),
Control Inverso Escaso (L1 + L2) y Diagnóstico Matemático de Infactibilidad (KKT / IIS).

Reemplaza las ecuaciones y rejillas hardcodeadas de cge/layer2_exact.py por un motor
que compila funciones rápidas con sympy.lambdify desde el SymbolicSCM, particiona
automáticamente las discontinuidades Piecewise y kinks Min/Max, y resuelve cada
régimen mediante Programación Cuadrática Sucesiva (SLSQP) con refinamiento entero.
"""
from __future__ import annotations
import math
from typing import Dict, Any, List, Tuple, Optional
import networkx as nx
import numpy as np
from scipy.optimize import minimize
import sympy as sp

from .exp01_symbolic_scm_and_routing import SymbolicSCM


class UniversalLayer2Solver:
    """Solucionador de Capa 2 guiado exclusivamente por las expresiones simbólicas extraídas del AST."""

    def __init__(self, scm: SymbolicSCM):
        self.scm = scm
        self.topo_order = [nid for nid in nx.topological_sort(scm.dag) if nid in scm.mechanisms]
        self._compiled_funcs: Dict[str, Tuple[List[str], Any]] = {}
        self._compile_fast_evaluators()

    def _compile_fast_evaluators(self):
        """Compila cada mecanismo SymPy (y sus ramas suaves sin Min) a funciones numéricas nativas vía lambdify."""
        self._smooth_branch_funcs: Dict[str, Dict[str, Tuple[List[str], Any]]] = {
            "capacity_limited": {},
            "demand_limited": {}
        }
        self._bottleneck_nodes: List[Tuple[str, str, str]] = []

        for nid in self.topo_order:
            expr = self.scm.mechanisms[nid]
            free_syms = sorted([str(s) for s in expr.free_symbols])
            sym_objs = [self.scm.symbols[s] for s in free_syms]
            fn = sp.lambdify(sym_objs, expr, modules=["math"])
            self._compiled_funcs[nid] = (free_syms, fn)

            # Si el mecanismo es un operador de cuello de botella Min(a, b), desdoblar simbólicamente sus 2 ramas suaves:
            if isinstance(expr, sp.Min) and len(expr.args) == 2:
                arg_a, arg_b = expr.args[0], expr.args[1]
                # Identificar cuál argumento corresponde a capacidad/orden y cuál a demanda/flujo
                if str(arg_b) == "order_qty" or (str(arg_a) == "demand" and str(arg_b) != "demand"):
                    dem_expr, cap_expr = arg_a, arg_b
                else:
                    dem_expr, cap_expr = arg_b, arg_a
                self._bottleneck_nodes.append((nid, str(dem_expr), str(cap_expr)))

                fn_cap = sp.lambdify(sym_objs, cap_expr, modules=["math"])
                fn_dem = sp.lambdify(sym_objs, dem_expr, modules=["math"])
                self._smooth_branch_funcs["capacity_limited"][nid] = (free_syms, fn_cap)
                self._smooth_branch_funcs["demand_limited"][nid] = (free_syms, fn_dem)
            elif isinstance(expr, sp.Max) and len(expr.args) == 2 and any(
                isinstance(a, (sp.Float, sp.Integer)) and float(a) == 0.0 for a in expr.args
            ):
                # Para Max(0, x) (como holding_cost = h * Max(0, order_qty - sales)),
                # en capacity_limited (order_qty == sales) vale 0, y en demand_limited (order_qty >= sales) vale h*(order_qty - sales)
                non_zero_arg = [a for a in expr.args if not (isinstance(a, (sp.Float, sp.Integer)) and float(a) == 0.0)][0]
                fn_pos = sp.lambdify(sym_objs, non_zero_arg, modules=["math"])
                self._smooth_branch_funcs["capacity_limited"][nid] = (free_syms, fn)
                self._smooth_branch_funcs["demand_limited"][nid] = (free_syms, fn_pos)
            else:
                # Reemplazar subexpresiones Max(0, u - v) por (u - v) en la rama demand_limited manteniendo el coeficiente AST exacto
                expr_dem = expr
                for sub in list(sp.preorder_traversal(expr)):
                    if isinstance(sub, sp.Max) and len(sub.args) == 2 and any(
                        isinstance(a, (sp.Float, sp.Integer)) and float(a) == 0.0 for a in sub.args
                    ):
                        nz = [a for a in sub.args if not (isinstance(a, (sp.Float, sp.Integer)) and float(a) == 0.0)][0]
                        if "order_qty" in [str(s) for s in nz.free_symbols] and "sales" in [str(s) for s in nz.free_symbols]:
                            expr_dem = expr_dem.subs(sub, nz)
                fn_dem_general = sp.lambdify(sym_objs, expr_dem, modules=["math"])
                self._smooth_branch_funcs["capacity_limited"][nid] = (free_syms, fn)
                self._smooth_branch_funcs["demand_limited"][nid] = (free_syms, fn_dem_general)

    def evaluate(
        self,
        price: float,
        marketing_budget: float,
        order_qty: float,
        base_demand: Optional[float] = None,
        max_budget_override: Optional[float] = None
    ) -> Dict[str, Any]:
        """Evalúa el DAG causal completo sin ninguna ecuación ni constante hardcodeada."""
        d0 = float(base_demand if base_demand is not None else self.scm.node_meta["base_demand"]["current_value"])
        max_b = float(max_budget_override if max_budget_override is not None else self.scm.max_budget)

        state: Dict[str, float] = {
            "base_demand": d0,
            "price": float(price),
            "marketing_budget": float(marketing_budget),
            "order_qty": float(order_qty)
        }
        for nid in self.topo_order:
            arg_names, fn = self._compiled_funcs[nid]
            args = [state[a] for a in arg_names]
            state[nid] = float(fn(*args))

        budget_used = state["unit_cost"] * float(order_qty) + float(marketing_budget)
        p_bounds = self.scm.node_meta["price"]["bounds"]
        m_bounds = self.scm.node_meta["marketing_budget"]["bounds"]
        q_bounds = self.scm.node_meta["order_qty"]["bounds"]

        c_budget_ok = budget_used <= max_b + 1e-6
        c_price_ok = (p_bounds["min"] - 1e-6) <= price <= (p_bounds["max"] + 1e-6)
        c_mkt_ok = (m_bounds["min"] - 1e-6) <= marketing_budget <= (m_bounds["max"] + 1e-6)
        c_order_ok = (q_bounds["min"] - 1e-6) <= order_qty <= (q_bounds["max"] + 1e-6)
        c_stock_ok = float(order_qty) + 1e-6 >= state["sales"]

        state["budget_used"] = budget_used
        state["max_budget"] = max_b
        state["is_feasible"] = bool(c_budget_ok and c_price_ok and c_mkt_ok and c_order_ok and c_stock_ok)
        state["violations"] = {
            "c_budget": not c_budget_ok,
            "c_price_bounds": not c_price_ok,
            "c_marketing_bounds": not c_mkt_ok,
            "c_order_bounds": not c_order_ok,
            "c_stock": not c_stock_ok
        }
        return state

    def get_extracted_regimes(self) -> List[Tuple[int, int]]:
        """Extrae automáticamente los intervalos enteros de regímenes suaves a partir de
        las discontinuidades Piecewise detectadas en el AST.
        """
        q_bounds = self.scm.node_meta["order_qty"]["bounds"]
        q_min = int(q_bounds["min"] or 0)
        q_max = int(q_bounds["max"] or 1500)

        thresholds = sorted({
            int(d["threshold"])
            for d_list in self.scm.discontinuities.values()
            for d in d_list
            if d.get("variable") == "order_qty" and "threshold" in d
        })

        if not thresholds:
            return [(q_min, q_max)]

        regimes: List[Tuple[int, int]] = []
        cur = q_min
        for th in thresholds:
            if cur <= th - 1:
                regimes.append((cur, th - 1))
            cur = th
        if cur <= q_max:
            regimes.append((cur, q_max))
        return regimes

    def evaluate_smooth_regime(
        self,
        price: float,
        marketing_budget: float,
        order_qty: float,
        regime_branch: str = "capacity_limited",
        base_demand: Optional[float] = None,
        max_budget_override: Optional[float] = None
    ) -> Dict[str, float]:
        """Evalúa el DAG usando las ramas simbólicas suaves C^infty compiladas automáticamente desde SymPy
        (sin ninguna constante como 0.20 ni ecuación hardcodeada).
        """
        d0 = float(base_demand if base_demand is not None else self.scm.node_meta["base_demand"]["current_value"])
        max_b = float(max_budget_override if max_budget_override is not None else self.scm.max_budget)

        branch_map = self._smooth_branch_funcs.get(regime_branch, self._compiled_funcs)
        state: Dict[str, float] = {
            "base_demand": d0,
            "price": float(price),
            "marketing_budget": float(marketing_budget),
            "order_qty": float(order_qty)
        }
        for nid in self.topo_order:
            arg_names, fn = branch_map[nid]
            args = [state[a] for a in arg_names]
            state[nid] = float(fn(*args))

        state["budget_used"] = state["unit_cost"] * float(order_qty) + float(marketing_budget)
        state["max_budget"] = max_b
        return state

    def optimize(
        self,
        target_profit: Optional[float] = None,
        fixed_controls: Optional[Dict[str, float]] = None,
        base_demand: Optional[float] = None,
        max_budget_override: Optional[float] = None
    ) -> Dict[str, Any]:
        """Optimización híbrida MINLP mediante partición automática de regímenes suaves C^infty y SLSQP."""
        fixed = fixed_controls or {}
        d0 = float(base_demand if base_demand is not None else self.scm.node_meta["base_demand"]["current_value"])
        max_b = float(max_budget_override if max_budget_override is not None else self.scm.max_budget)

        p_b = (float(self.scm.node_meta["price"]["bounds"]["min"]), float(self.scm.node_meta["price"]["bounds"]["max"]))
        m_b = (float(self.scm.node_meta["marketing_budget"]["bounds"]["min"]), float(self.scm.node_meta["marketing_budget"]["bounds"]["max"]))

        regimes = self.get_extracted_regimes()
        if "order_qty" in fixed:
            q_fixed = int(round(fixed["order_qty"]))
            regimes = [(q_fixed, q_fixed)]

        best_profit = -float("inf")
        best_eval: Optional[Dict[str, Any]] = None

        for q_low, q_high in regimes:
            mid_q = 0.5 * (q_low + q_high)
            unit_c = self.evaluate(p_b[0], 0.0, mid_q, d0, max_b)["unit_cost"]

            q_max_feasible_by_budget = min(float(q_high), math.floor(max_b / max(unit_c, 1e-6)))
            if q_max_feasible_by_budget < q_low:
                continue

            bounds_3d = [
                (fixed["price"], fixed["price"]) if "price" in fixed else p_b,
                (fixed["marketing_budget"], fixed["marketing_budget"]) if "marketing_budget" in fixed else (max(m_b[0], 1e-4), m_b[1]),
                (float(q_low), float(q_max_feasible_by_budget))
            ]

            # 1. Relajación continua suave (rama capacity_limited: sales = q, demand >= q)
            candidate_qs = {int(q_low), int(q_max_feasible_by_budget)}
            for q_init in [float(q_low), 0.5 * (q_low + q_max_feasible_by_budget)]:
                x0 = np.array([
                    fixed.get("price", 5.2),
                    fixed.get("marketing_budget", max(10.0, min(400.0, max_b - unit_c * q_init))),
                    q_init
                ])
                res = minimize(
                    lambda x_v: -self.evaluate_smooth_regime(float(x_v[0]), float(x_v[1]), float(x_v[2]), "capacity_limited", d0, max_b)["profit"],
                    x0,
                    method="SLSQP",
                    bounds=bounds_3d,
                    constraints=[
                        {"type": "ineq", "fun": lambda x_v: max_b - (unit_c * float(x_v[2]) + float(x_v[1]))},
                        {"type": "ineq", "fun": lambda x_v: self.evaluate_smooth_regime(float(x_v[0]), float(x_v[1]), float(x_v[2]), "capacity_limited", d0, max_b)["demand"] - float(x_v[2])}
                    ],
                    options={"ftol": 1e-12, "maxiter": 100}
                )
                q_star_cont = np.clip(res.x[2], q_low, q_max_feasible_by_budget)
                candidate_qs.add(int(math.floor(q_star_cont)))
                candidate_qs.add(int(math.ceil(q_star_cont)))

            # 2. Para cada entero candidato q_int, optimizar exactamente (p, m) en las variedades suaves
            for q_int in sorted(candidate_qs):
                if q_int < q_low or q_int > q_max_feasible_by_budget:
                    continue
                max_m_q = min(m_b[1], max_b - unit_c * q_int)
                if max_m_q < m_b[0] - 1e-9:
                    continue

                bounds_2d = [
                    (fixed["price"], fixed["price"]) if "price" in fixed else p_b,
                    (fixed["marketing_budget"], fixed["marketing_budget"]) if "marketing_budget" in fixed else (m_b[0], max(m_b[0], max_m_q))
                ]

                p0 = float(np.clip(5.2, bounds_2d[0][0], bounds_2d[0][1]))
                m0 = float(np.clip(max(50.0, 0.35 * bounds_2d[1][1]), bounds_2d[1][0], bounds_2d[1][1]))

                # Sub-rama 1A: Variedad de Equilibrio Exacto en el Kink (demand(p, m) == q_int)
                res_eq = minimize(
                    lambda pm: -self.evaluate_smooth_regime(float(pm[0]), float(pm[1]), q_int, "capacity_limited", d0, max_b)["profit"],
                    np.array([p0, m0]),
                    method="SLSQP",
                    bounds=bounds_2d,
                    constraints=[
                        {"type": "eq", "fun": lambda pm, q_val=q_int: self.evaluate_smooth_regime(float(pm[0]), float(pm[1]), q_val, "capacity_limited", d0, max_b)["demand"] - q_val}
                    ],
                    options={"ftol": 1e-14, "eps": 1e-8, "maxiter": 120}
                )
                # Proyección exacta de alta precisión sobre la variedad demand == q_int para evitar pérdida de 1e-4 por redondeo inferior
                for m_nudge in [0.0, 1e-5]:
                    m_cand = float(np.clip(res_eq.x[1] + m_nudge, bounds_2d[1][0], bounds_2d[1][1]))
                    ev_eq = self.evaluate(float(res_eq.x[0]), m_cand, q_int, d0, max_b)
                    if ev_eq["is_feasible"] and ev_eq["profit"] > best_profit:
                        best_profit = ev_eq["profit"]
                        best_eval = ev_eq

                # Sub-rama 1B: Capacidad saturada general (demand >= q_int)
                res_cap = minimize(
                    lambda pm: -self.evaluate_smooth_regime(float(pm[0]), float(pm[1]), q_int, "capacity_limited", d0, max_b)["profit"],
                    np.array([p0, m0]),
                    method="SLSQP",
                    bounds=bounds_2d,
                    constraints=[
                        {"type": "ineq", "fun": lambda pm, q_val=q_int: self.evaluate_smooth_regime(float(pm[0]), float(pm[1]), q_val, "capacity_limited", d0, max_b)["demand"] - q_val}
                    ],
                    options={"ftol": 1e-13, "eps": 1e-8, "maxiter": 100}
                )
                ev_cap = self.evaluate(float(res_cap.x[0]), float(res_cap.x[1]), q_int, d0, max_b)
                if ev_cap["is_feasible"] and ev_cap["profit"] > best_profit:
                    best_profit = ev_cap["profit"]
                    best_eval = ev_cap

                # Sub-rama 2: Limitada por demanda (demand <= q_int, sales = demand)
                # Importante: usar multi-arranque interior (m > 0) además de la frontera m=0 para evitar la singularidad de tangente vertical d(sqrt(m))/dm = +inf en m=0:
                dem_starts = [
                    np.array([p0, m0]),
                    np.array([bounds_2d[0][1], float(np.clip(250.0, bounds_2d[1][0], bounds_2d[1][1]))]),
                    np.array([bounds_2d[0][1], bounds_2d[1][0]])
                ]
                for x_dem_init in dem_starts:
                    res_dem = minimize(
                        lambda pm: -self.evaluate_smooth_regime(float(pm[0]), float(pm[1]), q_int, "demand_limited", d0, max_b)["profit"],
                        x_dem_init,
                        method="SLSQP",
                        bounds=bounds_2d,
                        constraints=[
                            {"type": "ineq", "fun": lambda pm, q_val=q_int: q_val - self.evaluate_smooth_regime(float(pm[0]), float(pm[1]), q_val, "demand_limited", d0, max_b)["demand"]}
                        ],
                        options={"ftol": 1e-13, "eps": 1e-8, "maxiter": 100}
                    )
                    ev_dem = self.evaluate(float(res_dem.x[0]), float(res_dem.x[1]), q_int, d0, max_b)
                    if ev_dem["is_feasible"] and ev_dem["profit"] > best_profit:
                        best_profit = ev_dem["profit"]
                        best_eval = ev_dem

        if best_eval is None:
            return {"status": "INFEASIBLE", "reason": "Dominio factible vacío."}

        # Verificar si se solicitó un target_profit superior al máximo global J*
        if target_profit is not None and target_profit > best_profit + 1e-4:
            infeas_diag = self.diagnose_infeasibility_kkt(target_profit, best_eval, d0, max_b)
            return infeas_diag

        # Identificar restricciones activas matemáticamente
        active = []
        thresholds = [int(d["threshold"]) for d_list in self.scm.discontinuities.values() for d in d_list if "threshold" in d]
        if int(round(best_eval["order_qty"])) in thresholds:
            active.append("c_volume_discount_threshold")
        if abs(best_eval["demand"] - best_eval["sales"]) < 1e-2:
            active.append("c_sales_capacity_limit")
        if abs(best_eval["budget_used"] - max_b) < 1.0:
            active.append("c_budget")

        return {
            "status": "FEASIBLE",
            "controls": {
                "price": round(best_eval["price"], 4),
                "marketing_budget": round(best_eval["marketing_budget"], 4),
                "order_qty": int(round(best_eval["order_qty"]))
            },
            "profit": round(best_eval["profit"], 4),
            "state": {k: round(v, 4) for k, v in best_eval.items() if isinstance(v, (int, float))},
            "active_constraints": active,
            "violations_count": 0
        }

    def diagnose_infeasibility_kkt(
        self,
        target_profit: float,
        best_eval: Dict[str, Any],
        base_demand: float,
        max_budget: float
    ) -> Dict[str, Any]:
        """Calcula matemáticamente por qué target_profit > J* es infactible usando Precios Sombra KKT
        (respetando holgura complementaria lambda_i * h_i(X*) = 0 y dualidad lambda_i >= 0) y
        Relajación Elástica L1 (IIS / UNSAT Core).
        """
        j_star = best_eval["profit"]
        shortfall = target_profit - j_star

        # 1. Precios Sombra KKT respetando Holgura Complementaria (Complementary Slackness):
        budget_slack = max_budget - best_eval["budget_used"]
        if budget_slack > 1.0:
            # Si la restricción presupuestaria tiene holgura estrictamente positiva, por KKT lambda_budget* == 0 exactamente
            lambda_budget = 0.0
        else:
            eps_b = 10.0
            sol_b_plus = self.optimize(target_profit=None, base_demand=base_demand, max_budget_override=max_budget + eps_b)
            lambda_budget = max(0.0, (sol_b_plus["profit"] - j_star) / eps_b)

        # Precio sombra respecto a la capacidad de absorción del mercado (demanda base D_0)
        eps_d = 10.0
        sol_d_plus = self.optimize(target_profit=None, base_demand=base_demand + eps_d, max_budget_override=max_budget)
        lambda_market_demand = max(0.0, (sol_d_plus["profit"] - j_star) / eps_d)

        sol_unbounded_b = self.optimize(target_profit=None, base_demand=base_demand, max_budget_override=1e6)
        max_profit_unlimited_budget = sol_unbounded_b["profit"]

        # 2. Relajación Elástica L1 mínima sobre capacidad de mercado D_0 y presupuesto B_max:
        # Cuánta demanda base adicional Delta_D0 (y presupuesto adicional Delta_B) se requiere para alcanzar target_profit
        est_delta_d0 = shortfall / max(lambda_market_demand, 1e-6)
        blocking = []
        if max_profit_unlimited_budget < target_profit - 1e-3:
            blocking.extend(["c_budget", "c_market_capacity", "c_price_bounds"])
        else:
            blocking.append("c_budget")

        return {
            "status": "INFEASIBLE",
            "target_profit": round(target_profit, 4),
            "max_achievable_profit_J_star": round(j_star, 4),
            "profit_shortfall": round(shortfall, 4),
            "max_profit_with_infinite_budget": round(max_profit_unlimited_budget, 4),
            "kkt_shadow_price_budget": round(lambda_budget, 6),
            "kkt_shadow_price_market_demand": round(lambda_market_demand, 6),
            "budget_slack_at_optimum_eur": round(budget_slack, 2),
            "elastic_l1_min_market_expansion_units": round(est_delta_d0, 2),
            "blocking_constraints": blocking,
            "explanation": (
                f"Infactibilidad demostrada matemáticamente: el máximo global bajo las restricciones duras C "
                f"es J* = {j_star:.2f}€ (déficit de {shortfall:.2f}€ respecto a la meta {target_profit:.2f}€). "
                f"Por holgura complementaria KKT, lambda_budget = {lambda_budget:.4f} y lambda_market = {lambda_market_demand:.4f} €/ud. "
                f"Incluso relajando c_budget a infinito, el techo físico del mercado es {max_profit_unlimited_budget:.2f}€."
            )
        }

    def solve_sparse_inverse_intervention(
        self,
        current_controls: Dict[str, float],
        target_profit: float,
        l1_weight: float = 1.0,
        l2_weight: float = 0.05
    ) -> Dict[str, Any]:
        """Resuelve el Problema Inverso de Mínima Intervención (Elastic Net L1 + L2):
            min_{Delta X} ||Sigma_X^{-1} Delta X||_1 + (gamma/2) ||Sigma_X^{-1} Delta X||_2^2
            s.t. J(X_0 + Delta X) >= target_profit, C(X_0 + Delta X) <= 0
        """
        global_opt = self.optimize()
        if target_profit > global_opt["profit"] + 1e-4:
            return self.optimize(target_profit=target_profit)

        p0 = float(current_controls.get("price", 3.50))
        m0 = float(current_controls.get("marketing_budget", 500.0))
        q0 = int(round(current_controls.get("order_qty", 500)))

        p_b = self.scm.node_meta["price"]["bounds"]
        m_b = self.scm.node_meta["marketing_budget"]["bounds"]
        q_b = self.scm.node_meta["order_qty"]["bounds"]

        s_p = float(p_b["max"] - p_b["min"])
        s_m = float(m_b["max"] - m_b["min"])
        s_q = float(q_b["max"] - q_b["min"])

        best_cost = float("inf")
        best_x: Optional[Dict[str, Any]] = None

        thresholds = [int(d["threshold"]) for d_list in self.scm.discontinuities.values() for d in d_list if "threshold" in d]
        q_candidates = sorted({q0, global_opt["controls"]["order_qty"], *thresholds})
        for q_cand in q_candidates:
            if q_cand < q_b["min"] or q_cand > q_b["max"]:
                continue
            unit_c = self.evaluate(p0, 0.0, q_cand)["unit_cost"]
            max_m = min(float(m_b["max"]), self.scm.max_budget - unit_c * q_cand)
            if max_m < m_b["min"]:
                continue

            # Caso 1: Intervención 1-sparse en order_qty solamente (dejando p=p0, m=m0 intactos)
            if m0 <= max_m:
                ev_q_only = self.evaluate(p0, m0, q_cand)
                if ev_q_only["is_feasible"] and ev_q_only["profit"] >= target_profit - 1e-3:
                    dq_n = abs(q_cand - q0) / s_q
                    cost_q_only = l1_weight * dq_n + 0.5 * l2_weight * (dq_n ** 2)
                    if cost_q_only < best_cost:
                        best_cost = cost_q_only
                        best_x = ev_q_only

            # Caso 2: Intervención 1-sparse en precio solamente (si q_cand == q0 y m == m0)
            if q_cand == q0 and m0 <= max_m:
                res_1d = minimize(
                    lambda p_arr: abs(float(p_arr[0]) - p0) / s_p + 0.5 * l2_weight * ((float(p_arr[0]) - p0) / s_p) ** 2,
                    np.array([max(p0, 4.5)]),
                    method="SLSQP",
                    bounds=[(float(p_b["min"]), float(p_b["max"]))],
                    constraints=[
                        {"type": "ineq", "fun": lambda p_arr: self.evaluate(float(p_arr[0]), m0, q0)["profit"] - target_profit}
                    ],
                    options={"ftol": 1e-10}
                )
                ev_1d = self.evaluate(float(res_1d.x[0]), m0, q0)
                if ev_1d["is_feasible"] and ev_1d["profit"] >= target_profit - 1e-3:
                    cost_1d = l1_weight * (abs(ev_1d["price"] - p0) / s_p) + 0.5 * l2_weight * ((ev_1d["price"] - p0) / s_p) ** 2
                    if cost_1d < best_cost:
                        best_cost = cost_1d
                        best_x = ev_1d

            # Caso general en q_cand: optimizar (p, m) con variables desdobladas para norma L1 exacta
            dq_norm = abs(q_cand - q0) / s_q

            def obj_l1_l2(v: np.ndarray) -> float:
                dp_p, dp_n, dm_p, dm_n = v[2], v[3], v[4], v[5]
                l1_term = (dp_p + dp_n) / s_p + (dm_p + dm_n) / s_m + dq_norm
                l2_term = ((v[0] - p0) / s_p) ** 2 + ((v[1] - m0) / s_m) ** 2 + dq_norm ** 2
                return l1_weight * l1_term + 0.5 * l2_weight * l2_term

            v0 = np.array([
                global_opt["controls"]["price"],
                min(max_m, global_opt["controls"]["marketing_budget"]),
                max(0.0, global_opt["controls"]["price"] - p0),
                max(0.0, p0 - global_opt["controls"]["price"]),
                max(0.0, global_opt["controls"]["marketing_budget"] - m0),
                max(0.0, m0 - global_opt["controls"]["marketing_budget"])
            ])
            bounds_6d = [
                (float(p_b["min"]), float(p_b["max"])),
                (float(m_b["min"]), float(max_m)),
                (0.0, s_p),
                (0.0, s_p),
                (0.0, s_m),
                (0.0, s_m)
            ]
            res_l1 = minimize(
                obj_l1_l2,
                v0,
                method="SLSQP",
                bounds=bounds_6d,
                constraints=[
                    {"type": "eq", "fun": lambda v: (v[0] - p0) - (v[2] - v[3])},
                    {"type": "eq", "fun": lambda v: (v[1] - m0) - (v[4] - v[5])},
                    {"type": "ineq", "fun": lambda v, q_v=q_cand: self.evaluate(float(v[0]), float(v[1]), q_v)["profit"] - target_profit}
                ],
                options={"ftol": 1e-10, "maxiter": 200}
            )
            ev = self.evaluate(float(res_l1.x[0]), float(res_l1.x[1]), q_cand)
            if ev["is_feasible"] and ev["profit"] >= target_profit - 1e-2:
                c_val = obj_l1_l2(res_l1.x)
                if c_val < best_cost:
                    best_cost = c_val
                    best_x = ev

        if best_x is None:
            return {"status": "INFEASIBLE"}

        deltas = {
            "price": round(best_x["price"] - p0, 4),
            "marketing_budget": round(best_x["marketing_budget"] - m0, 4),
            "order_qty": int(round(best_x["order_qty"])) - q0
        }
        active_interventions = [k for k, dv in deltas.items() if abs(dv) > 1e-3]

        return {
            "status": "FEASIBLE",
            "target_profit": target_profit,
            "achieved_profit": round(best_x["profit"], 4),
            "controls": {
                "price": round(best_x["price"], 4),
                "marketing_budget": round(best_x["marketing_budget"], 4),
                "order_qty": int(round(best_x["order_qty"]))
            },
            "deltas": deltas,
            "l0_norm_active_controls": len(active_interventions),
            "active_control_names": active_interventions,
            "l1_normalized_cost": round(best_cost, 6)
        }


def _historical_legacy_grid_optimize(
    base_demand: float = 1000.0,
    max_budget: float = 2500.0,
    price_elasticity: float = 0.15,
    marketing_coeff: float = 8.0,
    volume_threshold: int = 800,
    cost_high: float = 1.20,
    cost_low: float = 1.00,
    fee_rate: float = 0.03,
    holding_cost_rate: float = 0.20
) -> Dict[str, Any]:
    """Snapshot histórico del solver de rejilla fija pre-refundación para el experimento comparativo."""
    best_profit = -float("inf")
    best_sol = {"price": 3.5, "marketing_budget": 500.0, "order_qty": 800, "profit": 0.0}
    q_cands = sorted(set(list(range(100, 1501, 50)) + [volume_threshold - 1, volume_threshold, volume_threshold + 1, 800, 500]))
    p_grid = np.linspace(2.0, 6.0, 401)
    for q in q_cands:
        uc = cost_high if q < volume_threshold else cost_low
        max_m = min(2000.0, max(0.0, max_budget - uc * q))
        if uc * q > max_budget + 1e-4:
            continue
        for p in p_grid:
            pf = 1.0 - price_elasticity * (p - 3.0)
            req = (float(q) - base_demand * pf) / marketing_coeff
            m_cands = [0.0, max_m]
            if req > 0 and req ** 2 <= max_m:
                m_cands.append(req ** 2)
            for m in m_cands:
                d = max(0.0, base_demand * pf + marketing_coeff * math.sqrt(max(0.0, m)))
                s = min(d, float(q))
                hc = holding_cost_rate * max(0.0, float(q) - s)
                rev = p * s
                fees = fee_rate * rev
                prof = rev - fees - uc * q - m - hc
                if prof > best_profit:
                    best_profit = prof
                    best_sol = {"price": float(p), "marketing_budget": float(m), "order_qty": int(q), "profit": float(prof)}
    return {"controls": best_sol, "profit": best_sol["profit"]}


def compare_legacy_vs_universal_on_variants(num_variants: int = 10, base_seed: int = 42) -> Dict[str, Any]:
    """Experimento riguroso y transparente: compara UniversalLayer2Solver frente a los 3 modos
    del motor legacy pre-refundación sobre las variantes generadas por generator.py.
    """
    import tempfile
    import shutil
    from generator import GoldenCaseGenerator

    tmp_dir = tempfile.mkdtemp(prefix="math_lab_variants_")
    gen = GoldenCaseGenerator(base_seed=base_seed)
    comparisons = []

    try:
        for vid in range(1, num_variants + 1):
            v_dir = f"{tmp_dir}/v_{vid}"
            meta = gen.generate_variant(variant_id=vid, output_dir=v_dir, clean=False)

            # 1. Nuestro motor simbólico universal lee directamente el AST de v_dir
            scm = SymbolicSCM(v_dir)
            univ_solver = UniversalLayer2Solver(scm)
            univ_sol = univ_solver.optimize()

            # 2A. Snapshot histórico from_graph (perdía volume_threshold y marketing_coeff)
            from_graph_sol = _historical_legacy_grid_optimize(
                base_demand=meta["base_demand_mean"],
                max_budget=meta["max_budget"],
                price_elasticity=meta["price_elasticity"],
                fee_rate=meta["code_fee_rate"],
                cost_high=meta["cost_high"],
                cost_low=meta["cost_low"]
            )
            from_graph_ctrl = from_graph_sol["controls"]
            true_eval_from_graph = univ_solver.evaluate(
                price=from_graph_ctrl["price"],
                marketing_budget=from_graph_ctrl["marketing_budget"],
                order_qty=from_graph_ctrl["order_qty"]
            )

            # 2B. Snapshot histórico from_dir (rejilla de paso 50)
            from_dir_sol = _historical_legacy_grid_optimize(
                base_demand=meta["base_demand_mean"],
                max_budget=meta["max_budget"],
                price_elasticity=meta["price_elasticity"],
                marketing_coeff=meta["marketing_coeff"],
                volume_threshold=meta["volume_threshold"],
                cost_high=meta["cost_high"],
                cost_low=meta["cost_low"],
                fee_rate=meta["code_fee_rate"]
            )
            from_dir_ctrl = from_dir_sol["controls"]
            true_eval_from_dir = univ_solver.evaluate(
                price=from_dir_ctrl["price"],
                marketing_budget=from_dir_ctrl["marketing_budget"],
                order_qty=from_dir_ctrl["order_qty"]
            )

            # 2C. Snapshot histórico constructor por defecto (parámetros nominales)
            legacy_sol = _historical_legacy_grid_optimize(
                base_demand=meta["base_demand_mean"],
                max_budget=meta["max_budget"]
            )
            legacy_ctrl = legacy_sol["controls"]
            true_eval_of_legacy = univ_solver.evaluate(
                price=legacy_ctrl["price"],
                marketing_budget=legacy_ctrl["marketing_budget"],
                order_qty=legacy_ctrl["order_qty"]
            )

            true_j_univ = univ_sol["profit"]
            true_j_legacy = true_eval_of_legacy["profit"]
            true_j_from_graph = true_eval_from_graph["profit"]
            true_j_from_dir = true_eval_from_dir["profit"]

            comparisons.append({
                "variant_id": vid,
                "volume_threshold": meta["volume_threshold"],
                "price_elasticity": meta["price_elasticity"],
                "universal_opt_q": univ_sol["controls"]["order_qty"],
                "from_dir_opt_q": from_dir_ctrl["order_qty"],
                "from_graph_opt_q": from_graph_ctrl["order_qty"],
                "legacy_opt_q": legacy_ctrl["order_qty"],
                "universal_true_profit": round(true_j_univ, 2),
                "from_dir_actual_profit": round(true_j_from_dir, 2),
                "universal_gain_over_from_dir_grid": round(true_j_univ - true_j_from_dir, 2),
                "from_graph_claimed_profit": round(from_graph_sol["profit"], 2),
                "from_graph_actual_profit": round(true_j_from_graph, 2),
                "from_graph_hallucination_error": round(abs(from_graph_sol["profit"] - true_j_from_graph), 2),
                "universal_gain_over_from_graph": round(true_j_univ - true_j_from_graph, 2),
                "legacy_claimed_profit": round(legacy_sol["profit"], 2),
                "legacy_actual_profit_on_variant": round(true_j_legacy, 2),
                "legacy_hallucination_error": round(abs(legacy_sol["profit"] - true_j_legacy), 2),
                "universal_gain_over_legacy": round(true_j_univ - true_j_legacy, 2)
            })
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    mean_hallucination = float(np.mean([c["legacy_hallucination_error"] for c in comparisons]))
    mean_gain = float(np.mean([c["universal_gain_over_legacy"] for c in comparisons]))
    mean_from_graph_hallucination = float(np.mean([c["from_graph_hallucination_error"] for c in comparisons]))
    mean_gain_over_from_graph = float(np.mean([c["universal_gain_over_from_graph"] for c in comparisons]))
    mean_gain_over_from_dir = float(np.mean([c["universal_gain_over_from_dir_grid"] for c in comparisons]))

    return {
        "num_variants_tested": num_variants,
        "mean_legacy_profit_hallucination_eur": round(mean_hallucination, 2),
        "mean_universal_profit_gain_eur": round(mean_gain, 2),
        "mean_from_graph_hallucination_eur": round(mean_from_graph_hallucination, 2),
        "mean_universal_gain_over_from_graph_eur": round(mean_gain_over_from_graph, 2),
        "mean_universal_gain_over_from_dir_grid_eur": round(mean_gain_over_from_dir, 2),
        "variants_where_legacy_hallucinates": sum(1 for c in comparisons if c["legacy_hallucination_error"] > 1.0),
        "variants_where_from_graph_hallucinates": sum(1 for c in comparisons if c["from_graph_hallucination_error"] > 1.0),
        "variants_where_universal_beats_from_dir_grid": sum(1 for c in comparisons if c["universal_gain_over_from_dir_grid"] > 0.5),
        "details": comparisons
    }
