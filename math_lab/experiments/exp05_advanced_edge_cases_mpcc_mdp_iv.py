"""Experimento 05: Resolución Matemática de los 3 Casos Extremos Avanzados (Edge Cases):
1. Explosión Combinatoria de Kinks (2^M Regímenes) -> Reformulación Epígrafe/MPCC + Homotopía Log-Sum-Exp en O(1) solves.
2. Grafos con Ciclos Temporales (Inventario Inter-Temporal I_t = max(0, I_{t-1} + q_t - sales_t)) -> Desenrollado en DAG Temporal Expandido (Time-Expanded DAG) y Control Predictivo Basado en Modelos (MPC).
3. Sesgo de Endogeneidad / Confusión en Econometría Causal (T7) -> Test de Hausman-Wu, Ajuste Back-Door y Variables Instrumentales en Dos Etapas (2SLS / IV).
"""
from __future__ import annotations
import math
import time
from typing import Dict, Any, List, Tuple
import networkx as nx
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


# =============================================================================
# 1. EXPLOSIÓN COMBINATORIA DE KINKS (2^M) VS HOMOTOPÍA SUAVE + MPCC/EPÍGRAFE
# =============================================================================

class MultiKinkSupplyChainBenchmark:
    """Red causal de M etapas en serie/paralelo con operadores no diferenciables Min(flujo, capacidad):
        stage_k = min(inflow_k(x_k), cap_k(x_k))   para k = 1, ..., M.
    El número de ortantes suaves posibles es 2^M.
    Demuestra cómo la continuación homotópica Log-Sum-Exp + reformulación de epígrafe suave (MPCC)
    halla el óptimo global exacto en O(1) resoluciones NLP frente a las 2^M de la enumeración exhaustiva.
    """

    def __init__(self, num_kinks: int = 6):
        self.M = num_kinks
        # Parámetros heterogéneos por etapa k = 0, ..., M-1
        self.alpha = np.array([18.0 + 2.0 * k for k in range(num_kinks)], dtype=float)
        self.beta = np.array([12.0 + 1.5 * k for k in range(num_kinks)], dtype=float)
        self.cost_a = np.array([1.1 + 0.1 * k for k in range(num_kinks)], dtype=float)
        self.cost_b = np.array([0.9 + 0.08 * k for k in range(num_kinks)], dtype=float)
        self.price_per_unit = 8.0

    def evaluate_stage_branches(self, u_vec: np.ndarray, v_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna los dos argumentos (A_k, B_k) de cada kink k = 1..M:
        A_k(u_k) = alpha_k * sqrt(u_k),   B_k(v_k) = beta_k * sqrt(v_k).
        """
        u_safe = np.maximum(1e-9, u_vec)
        v_safe = np.maximum(1e-9, v_vec)
        A = self.alpha * np.sqrt(u_safe)
        B = self.beta * np.sqrt(v_safe)
        return A, B

    def evaluate_exact_nonsmooth(self, x: np.ndarray) -> float:
        """Evalúa el beneficio exacto no suave:
            throughput_k = min(A_k(u_k), B_k(v_k))
            J = sum_{k=1}^M [ p_k * min(A_k, B_k) - c_{a,k} * u_k - c_{b,k} * v_k ]
        """
        u = x[:self.M]
        v = x[self.M:2 * self.M]
        A, B = self.evaluate_stage_branches(u, v)
        throughputs = np.minimum(A, B)
        weights = np.array([1.0 + 0.15 * k for k in range(self.M)])
        revenue = float(np.sum(self.price_per_unit * weights * throughputs))
        cost = float(np.sum(self.cost_a * u + self.cost_b * v))
        return revenue - cost

    def analytical_optimum(self) -> Dict[str, Any]:
        """En el óptimo de cada cuello de botella convexo-cóncavo, el equilibrio exacto ocurre en el kink
        A_k(u_k*) == B_k(v_k*), lo que permite despejar analíticamente la solución exacta para verificar precisión.
        """
        weights = np.array([1.0 + 0.15 * k for k in range(self.M)])
        # Como s_k = alpha_k * sqrt(u_k) = beta_k * sqrt(v_k) => u_k = (s_k / alpha_k)^2, v_k = (s_k / beta_k)^2
        # Coste marginal de producir s_k: (c_{a,k}/alpha_k^2 + c_{b,k}/beta_k^2) * s_k^2
        # Maximizando p_k * s_k - gamma_k * s_k^2 => s_k* = p_k / (2 * gamma_k)
        gamma = (self.cost_a / (self.alpha ** 2)) + (self.cost_b / (self.beta ** 2))
        p_eff = self.price_per_unit * weights
        s_star = p_eff / (2.0 * gamma)
        u_star = (s_star / self.alpha) ** 2
        v_star = (s_star / self.beta) ** 2
        x_star = np.concatenate([u_star, v_star])
        j_star = self.evaluate_exact_nonsmooth(x_star)
        return {
            "x_star": x_star,
            "s_star": s_star,
            "j_star": float(j_star)
        }

    def solve_by_exhaustive_2m_regimes(self) -> Dict[str, Any]:
        """Enumera los 2^M ortantes de régimen (eligiendo A_k <= B_k o B_k <= A_k en cada etapa)."""
        t0 = time.perf_counter()
        num_orthants = 1 << self.M
        weights = np.array([1.0 + 0.15 * k for k in range(self.M)])
        best_j = -float("inf")
        best_x = np.ones(2 * self.M)
        nlp_calls = 0

        # En cada ortante bitmask b in {0, ..., 2^M - 1}, la etapa k es limitada por A_k si bit_k=0 o por B_k si bit_k=1
        for mask in range(num_orthants):
            bits = np.array([(mask >> k) & 1 for k in range(self.M)], dtype=int)

            def smooth_orthant_neg_obj(x_v: np.ndarray, b_vec: np.ndarray = bits) -> float:
                u, v = x_v[:self.M], x_v[self.M:]
                A, B = self.evaluate_stage_branches(u, v)
                s = np.where(b_vec == 0, A, B)
                return -float(np.sum(self.price_per_unit * weights * s - self.cost_a * u - self.cost_b * v))

            def orthant_ineq(x_v: np.ndarray, b_vec: np.ndarray = bits) -> np.ndarray:
                u, v = x_v[:self.M], x_v[self.M:]
                A, B = self.evaluate_stage_branches(u, v)
                return np.where(b_vec == 0, B - A, A - B)

            res = minimize(
                smooth_orthant_neg_obj,
                np.full(2 * self.M, 50.0),
                method="SLSQP",
                bounds=[(1.0, 6000.0)] * (2 * self.M),
                constraints=[{"type": "ineq", "fun": orthant_ineq}],
                options={"ftol": 1e-8, "maxiter": 45}
            )
            nlp_calls += 1
            j_val = self.evaluate_exact_nonsmooth(res.x)
            if j_val > best_j:
                best_j = j_val
                best_x = res.x

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "method": f"Exhaustive_2^M_Regimes (2^{self.M}={num_orthants})",
            "num_nlp_subproblems": nlp_calls,
            "optimal_profit": round(float(best_j), 4),
            "elapsed_ms": round(elapsed_ms, 2)
        }

    def solve_by_homotopy_mpcc_epigraph(self) -> Dict[str, Any]:
        """Resuelve el DAG con M kinks en O(1) subproblemas (2 pasos de homotopía Log-Sum-Exp + 1 pulido
        exacto de epígrafe con variables auxiliares de flujo s_k <= A_k, s_k <= B_k).
        """
        t0 = time.perf_counter()
        weights = np.array([1.0 + 0.15 * k for k in range(self.M)])
        nlp_calls = 0

        # Paso 1: Continuación Homotópica Suave Log-Sum-Exp: softmin_tau(A, B) = -1/tau * log(exp(-tau*A) + exp(-tau*B))
        x_cur = np.full(2 * self.M, 20.0)
        for tau in [0.5, 5.0]:
            def smooth_lse_neg_obj(x_v: np.ndarray, t_val: float = tau) -> float:
                u, v = x_v[:self.M], x_v[self.M:]
                A, B = self.evaluate_stage_branches(u, v)
                m_min = np.minimum(A, B)
                soft_min = m_min - (1.0 / t_val) * np.log(np.exp(-t_val * (A - m_min)) + np.exp(-t_val * (B - m_min)))
                return -float(np.sum(self.price_per_unit * weights * soft_min - self.cost_a * u - self.cost_b * v))

            res_h = minimize(
                smooth_lse_neg_obj,
                x_cur,
                method="L-BFGS-B",
                bounds=[(1.0, 6000.0)] * (2 * self.M),
                options={"ftol": 1e-10, "maxiter": 80}
            )
            x_cur = res_h.x
            nlp_calls += 1

        # Paso 2: Pulido Exacto C^infty mediante Elevación de Epígrafe / Complementariedad (u, v, s)
        # s_k <= A_k(u_k) y s_k <= B_k(v_k)
        u0, v0 = x_cur[:self.M], x_cur[self.M:]
        A0, B0 = self.evaluate_stage_branches(u0, v0)
        s0 = np.minimum(A0, B0)
        z0 = np.concatenate([u0, v0, s0])

        def epigraph_neg_obj(z: np.ndarray) -> float:
            u, v, s = z[:self.M], z[self.M:2 * self.M], z[2 * self.M:]
            return -float(np.sum(self.price_per_unit * weights * s - self.cost_a * u - self.cost_b * v))

        def epigraph_ineq(z: np.ndarray) -> np.ndarray:
            u, v, s = z[:self.M], z[self.M:2 * self.M], z[2 * self.M:]
            A, B = self.evaluate_stage_branches(u, v)
            return np.concatenate([A - s, B - s])

        res_epi = minimize(
            epigraph_neg_obj,
            z0,
            method="SLSQP",
            bounds=[(1.0, 6000.0)] * (2 * self.M) + [(0.0, 6000.0)] * self.M,
            constraints=[{"type": "ineq", "fun": epigraph_ineq}],
            options={"ftol": 1e-13, "maxiter": 150}
        )
        nlp_calls += 1
        x_final = res_epi.x[:2 * self.M]
        j_final = self.evaluate_exact_nonsmooth(x_final)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "method": "Homotopy_LSE_Plus_Epigraph_MPCC",
            "num_nlp_subproblems": nlp_calls,
            "optimal_profit": round(float(j_final), 4),
            "elapsed_ms": round(elapsed_ms, 2)
        }


def run_multi_kink_scalability_experiment(num_kinks: int = 6) -> Dict[str, Any]:
    """Compara la enumeración combinatoria 2^M frente a Homotopía + Epígrafe MPCC para M=6 kinks (64 regímenes)."""
    bench = MultiKinkSupplyChainBenchmark(num_kinks=num_kinks)
    exact = bench.analytical_optimum()
    res_2m = bench.solve_by_exhaustive_2m_regimes()
    res_mpcc = bench.solve_by_homotopy_mpcc_epigraph()

    return {
        "num_kinks_M": num_kinks,
        "combinatorial_regimes_2_pow_M": 1 << num_kinks,
        "analytical_global_optimum_profit": round(exact["j_star"], 4),
        "exhaustive_2m_solver": res_2m,
        "homotopy_epigraph_mpcc_solver": res_mpcc,
        "mpcc_error_vs_analytical": round(abs(res_mpcc["optimal_profit"] - exact["j_star"]), 4),
        "subproblem_reduction_factor": round(res_2m["num_nlp_subproblems"] / max(1, res_mpcc["num_nlp_subproblems"]), 2),
        "speedup_factor": round(res_2m["elapsed_ms"] / max(0.01, res_mpcc["elapsed_ms"]), 2)
    }


# =============================================================================
# 2. GRAFOS CON CICLOS TEMPORALES (FEEDBACK LOOPS / INVENTARIO INTER-TEMPORAL)
# =============================================================================

class TemporalCyclicSCMAndMPCSolver:
    """Resuelve grafos causales con bucles de retroalimentación temporal (ej. carry-over de inventario):
        I_t = max(0, I_{t-1} + q_t - sales_t)
        available_stock_t = I_{t-1} + q_t
        sales_t = min(demand_t, available_stock_t)
    En representación estática, el ciclo `inventory -> sales -> inventory` viola la aciclicidad DAG.
    Al desenrollar el grafo sobre un horizonte T (Time-Expanded DAG G_T), el sistema se vuelve
    acíclico de tamaño T*|V| y permite optimizar políticas multi-periodo (Batch Ordering / MPC).
    """

    def __init__(self, horizon_T: int = 2, base_demand_per_period: float = 420.0):
        self.T = horizon_T
        self.base_demand = base_demand_per_period

    @staticmethod
    def build_static_cyclic_graph() -> nx.DiGraph:
        G = nx.DiGraph()
        G.add_edges_from([
            ("price", "demand"),
            ("marketing_budget", "demand"),
            ("demand", "sales"),
            ("order_qty", "available_stock"),
            ("inventory_stock", "available_stock"),
            ("available_stock", "sales"),
            ("sales", "inventory_stock"),  # Ciclo de retroalimentación temporal: inventory -> available -> sales -> inventory
            ("sales", "revenue"),
            ("inventory_stock", "holding_cost"),
            ("revenue", "profit"),
            ("holding_cost", "profit")
        ])
        return G

    def build_time_expanded_dag(self) -> nx.DiGraph:
        """Desenrolla el grafo cíclico en un DAG acíclico expandido en el tiempo G_T sobre t = 1..T."""
        G_T = nx.DiGraph()
        G_T.add_node("I_0", period=0, category="initial_state")
        for t in range(1, self.T + 1):
            for v in ["p", "m", "q", "d", "avail", "sales", "I", "c_unit", "hold", "profit"]:
                G_T.add_node(f"{v}_{t}", period=t)
            G_T.add_edge(f"I_{t-1}", f"avail_{t}", lag=1)
            G_T.add_edge(f"q_{t}", f"avail_{t}", lag=0)
            G_T.add_edge(f"p_{t}", f"d_{t}", lag=0)
            G_T.add_edge(f"m_{t}", f"d_{t}", lag=0)
            G_T.add_edge(f"d_{t}", f"sales_{t}", lag=0)
            G_T.add_edge(f"avail_{t}", f"sales_{t}", lag=0)
            G_T.add_edge(f"avail_{t}", f"I_{t}", lag=0)
            G_T.add_edge(f"sales_{t}", f"I_{t}", lag=0)
            G_T.add_edge(f"I_{t}", f"hold_{t}", lag=0)
            G_T.add_edge(f"q_{t}", f"c_unit_{t}", lag=0)
            G_T.add_edge(f"sales_{t}", f"profit_{t}", lag=0)
            G_T.add_edge(f"c_unit_{t}", f"profit_{t}", lag=0)
            G_T.add_edge(f"hold_{t}", f"profit_{t}", lag=0)
            G_T.add_edge(f"profit_{t}", "J_total", lag=0)
        return G_T

    def evaluate_period(self, p: float, m: float, q: int, inv_prev: float) -> Dict[str, float]:
        unit_cost = 1.35 if q < 800 else 1.00
        demand = max(0.0, self.base_demand * (1.0 - 0.15 * (p - 3.0)) + 8.0 * math.sqrt(max(0.0, m)))
        avail = inv_prev + float(q)
        sales = min(demand, avail)
        inv_next = max(0.0, avail - sales)
        holding_cost = 0.15 * inv_next
        revenue = 0.97 * p * sales
        profit = revenue - unit_cost * float(q) - m - holding_cost
        return {
            "demand": demand,
            "available": avail,
            "sales": sales,
            "inventory_end": inv_next,
            "holding_cost": holding_cost,
            "profit": profit
        }

    def solve_myopic_vs_mpc_multi_period(self) -> Dict[str, Any]:
        """Compara la política Miope (optimiza cada periodo t por separado maximizando profit_t sin valorar
        el inventario transferido I_t) frente al Optimizador Multi-Periodo sobre el Time-Expanded DAG (MPC).
        Con demanda base moderada (D0 = 420), un optimizador miope de 1 periodo rechaza pedir q_1 = 800
        porque el stock sobrante I_1 se contabiliza como gasto puro en t=1, pagando c=1.35€/ud en ambos periodos.
        El optimizador MPC sobre el DAG expandido G_T pide q_1 = 800 en t=1 (desbloqueando c=1.00€/ud),
        transfiere I_1 > 0 hacia t=2 (coste de almacenaje 0.15€/ud < ahorro 0.35€/ud) y pide q_2 reducido en t=2.
        """
        static_g = self.build_static_cyclic_graph()
        expanded_dag = self.build_time_expanded_dag()
        q_grid = [0, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 800]

        # 1. Política Miope (resuelve t=1 maximizando profit_1(p1, m1, q1 | I0=0) y luego t=2 maximizando profit_2(p2, m2, q2 | I1)):
        best_myopic_1 = -float("inf")
        best_ctrl_1 = (5.0, 100.0, 400)
        for q1 in q_grid:
            res1 = minimize(
                lambda pm, qv=q1: -self.evaluate_period(float(pm[0]), float(pm[1]), qv, 0.0)["profit"],
                np.array([5.3, 120.0]),
                method="SLSQP",
                bounds=[(2.0, 6.0), (1.0, 1500.0)],
                options={"ftol": 1e-10, "maxiter": 100}
            )
            ev1 = self.evaluate_period(float(res1.x[0]), float(res1.x[1]), q1, 0.0)
            if ev1["profit"] > best_myopic_1:
                best_myopic_1 = ev1["profit"]
                best_ctrl_1 = (float(res1.x[0]), float(res1.x[1]), q1)

        ev1_myopic = self.evaluate_period(best_ctrl_1[0], best_ctrl_1[1], best_ctrl_1[2], 0.0)
        best_myopic_2 = -float("inf")
        for q2 in q_grid:
            res2 = minimize(
                lambda pm, qv=q2: -self.evaluate_period(float(pm[0]), float(pm[1]), qv, ev1_myopic["inventory_end"])["profit"],
                np.array([5.3, 120.0]),
                method="SLSQP",
                bounds=[(2.0, 6.0), (1.0, 1500.0)],
                options={"ftol": 1e-10, "maxiter": 100}
            )
            ev2 = self.evaluate_period(float(res2.x[0]), float(res2.x[1]), q2, ev1_myopic["inventory_end"])
            if ev2["profit"] > best_myopic_2:
                best_myopic_2 = ev2["profit"]

        myopic_total_profit = best_myopic_1 + best_myopic_2

        # 2. Política Dinámica Multi-Periodo sobre el DAG Expandido G_T (optimiza J_total = profit_1 + profit_2):
        best_mpc_total = -float("inf")
        best_mpc_plan = None
        for q1 in [400, 450, 500, 800]:
            for q2 in [0, 50, 100, 150, 200, 400, 450]:
                def mpc_neg_obj(x4: np.ndarray, q1_v: int = q1, q2_v: int = q2) -> float:
                    p1, m1, p2, m2 = float(x4[0]), float(x4[1]), float(x4[2]), float(x4[3])
                    ev_t1 = self.evaluate_period(p1, m1, q1_v, 0.0)
                    ev_t2 = self.evaluate_period(p2, m2, q2_v, ev_t1["inventory_end"])
                    return -(ev_t1["profit"] + ev_t2["profit"])

                res_mpc = minimize(
                    mpc_neg_obj,
                    np.array([5.3, 120.0, 5.3, 120.0]),
                    method="SLSQP",
                    bounds=[(2.0, 6.0), (1.0, 1500.0), (2.0, 6.0), (1.0, 1500.0)],
                    options={"ftol": 1e-11, "maxiter": 150}
                )
                if -res_mpc.fun > best_mpc_total:
                    best_mpc_total = -res_mpc.fun
                    p1, m1, p2, m2 = [float(v) for v in res_mpc.x]
                    ev1 = self.evaluate_period(p1, m1, q1, 0.0)
                    ev2 = self.evaluate_period(p2, m2, q2, ev1["inventory_end"])
                    best_mpc_plan = {
                        "period_1": {"price": round(p1, 4), "marketing_budget": round(m1, 2), "order_qty": q1, "carry_over_inventory_I1": round(ev1["inventory_end"], 2), "profit_t1": round(ev1["profit"], 2)},
                        "period_2": {"price": round(p2, 4), "marketing_budget": round(m2, 2), "order_qty": q2, "ending_inventory_I2": round(ev2["inventory_end"], 2), "profit_t2": round(ev2["profit"], 2)},
                        "total_2period_profit": round(best_mpc_total, 2)
                    }

        return {
            "static_graph_is_dag": nx.is_directed_acyclic_graph(static_g),
            "time_expanded_graph_is_dag": nx.is_directed_acyclic_graph(expanded_dag),
            "time_expanded_num_nodes": expanded_dag.number_of_nodes(),
            "myopic_single_period_policy": {
                "controls_per_period": {"price": round(best_ctrl_1[0], 4), "marketing_budget": round(best_ctrl_1[1], 2), "order_qty": best_ctrl_1[2]},
                "total_2period_profit": round(myopic_total_profit, 2)
            },
            "mpc_time_expanded_policy": best_mpc_plan,
            "mpc_gain_over_myopic_eur": round(best_mpc_total - myopic_total_profit, 2)
        }


# =============================================================================
# 3. ENDOGENEIDAD Y VARIABLES CONFUSORAS EN T7 (BACK-DOOR & 2SLS IV + HAUSMAN)
# =============================================================================

class CausalEndogeneityIV2SLSEstimator:
    """Resuelve el Caso Extremo 3 (T7 con Endogeneidad / Confusión Observacional):
    Cuando en datos históricos el presupuesto de marketing M_t fue decidido endógenamente en respuesta a un
    shock de demanda no observado W_t (M_t <- W_t -> D_t) pero existe un instrumento válido Z_t (ej. coste de
    puja publicitaria exógeno) o la variable confusora W_t se mide (Ajuste Back-Door de Pearl):
    1. Demuestra que OLS clásico sufre un sesgo severo e inconsistente (Cov(sqrt(M_t), eps_t) != 0).
    2. Detecta matemáticamente la endogeneidad mediante el Test de Especificación de Hausman-Wu.
    3. Recupera consistentemente los parámetros estructurales verdaderos (alpha=0.15, beta=8.0)
       mediante Mínimos Cuadrados en Dos Etapas (2SLS / IV) y Ajuste Back-Door sobre el DAG.
    """

    @staticmethod
    def run_endogeneity_and_iv_experiment(n_samples: int = 500, seed: int = 42) -> Dict[str, Any]:
        rng = np.random.RandomState(seed)
        alpha_true = 0.15
        beta_true = 8.0

        # Generamos un proceso estructural con confusor latente W_t ~ N(0, 1) (ej. campaña estacional / fiebre de consumo)
        # e instrumento exógeno Z_t ~ U(100, 900) (ej. asignación presupuestaria base / subasta publicitaria exógena):
        D0 = rng.normal(1000.0, 80.0, n_samples)
        P = rng.uniform(2.5, 5.5, n_samples)
        Z_inst = rng.uniform(150.0, 850.0, n_samples)
        W_confounder = rng.normal(0.0, 1.0, n_samples)
        eps_noise = rng.normal(0.0, 2.0, n_samples)

        # El gestor histórico subía el marketing M_t cuando veía alta demanda latente W_t > 0 (endogeneidad):
        sqrt_M = np.maximum(5.0, np.sqrt(Z_inst) + 4.5 * W_confounder + rng.normal(0.0, 1.0, n_samples))
        M = sqrt_M ** 2

        # Ecuación estructural verdadera de demanda: D_t = D0_t*(1 - 0.15*(P_t - 3)) + 8.0*sqrt(M_t) + 18.0*W_t + eps_t
        structural_error = 18.0 * W_confounder + eps_noise
        Y_demand = D0 * (1.0 - alpha_true * (P - 3.0)) + beta_true * sqrt_M + structural_error

        # 1. Estimador OLS ingenuo (ignora la endogeneidad de sqrt_M con W_t):
        X_ols = np.column_stack([D0, D0 * P, sqrt_M])
        theta_ols, _, _, _ = np.linalg.lstsq(X_ols, Y_demand, rcond=None)
        alpha_ols = float(-theta_ols[1])
        beta_ols = float(theta_ols[2])

        # 2. Test de Endogeneidad de Hausman-Wu (Función de Control / Primera Etapa):
        Z_mat = np.column_stack([D0, D0 * P, np.sqrt(Z_inst)])
        pi_hat, _, _, _ = np.linalg.lstsq(Z_mat, sqrt_M, rcond=None)
        sqrt_M_hat = Z_mat @ pi_hat
        v_first_stage_resid = sqrt_M - sqrt_M_hat

        X_hausman = np.column_stack([D0, D0 * P, sqrt_M, v_first_stage_resid])
        theta_h, _, _, _ = np.linalg.lstsq(X_hausman, Y_demand, rcond=None)
        res_h = Y_demand - X_hausman @ theta_h
        s2_h = float(np.sum(res_h ** 2) / (n_samples - X_hausman.shape[1]))
        cov_h = s2_h * np.linalg.inv(X_hausman.T @ X_hausman)
        hausman_t_stat = float(theta_h[3] / math.sqrt(max(cov_h[3, 3], 1e-12)))
        hausman_p_value = float(2.0 * (1.0 - norm.cdf(abs(hausman_t_stat))))
        endogeneity_detected = abs(hausman_t_stat) > 1.96

        # 3. Estimador de Variables Instrumentales en Dos Etapas (2SLS / IV):
        # Proyección ortogonal sobre el subespacio de instrumentos P_Z X = [D0, D0*P, sqrt_M_hat]
        X_hat = np.column_stack([D0, D0 * P, sqrt_M_hat])
        theta_2sls = np.linalg.inv(X_hat.T @ X_ols) @ (X_hat.T @ Y_demand)
        alpha_2sls = float(-theta_2sls[1])
        beta_2sls = float(theta_2sls[2])

        # 4. Estimador con Ajuste Back-Door de Pearl (condicionando sobre el confusor W_t si es observable):
        X_backdoor = np.column_stack([D0, D0 * P, sqrt_M, W_confounder])
        theta_bd, _, _, _ = np.linalg.lstsq(X_backdoor, Y_demand, rcond=None)
        alpha_bd = float(-theta_bd[1])
        beta_bd = float(theta_bd[2])

        return {
            "ground_truth": {"alpha": alpha_true, "beta": beta_true},
            "naive_ols_under_confounding": {
                "estimated_alpha": round(alpha_ols, 4),
                "estimated_beta": round(beta_ols, 4),
                "beta_bias_error": round(abs(beta_ols - beta_true), 4),
                "beta_relative_error_pct": round(100.0 * abs(beta_ols - beta_true) / beta_true, 2)
            },
            "hausman_wu_endogeneity_test": {
                "t_statistic": round(hausman_t_stat, 4),
                "p_value": round(hausman_p_value, 6),
                "endogeneity_detected": bool(endogeneity_detected)
            },
            "iv_2sls_estimator": {
                "estimated_alpha": round(alpha_2sls, 4),
                "estimated_beta": round(beta_2sls, 4),
                "beta_bias_error": round(abs(beta_2sls - beta_true), 4),
                "beta_relative_error_pct": round(100.0 * abs(beta_2sls - beta_true) / beta_true, 2)
            },
            "pearl_backdoor_adjustment_estimator": {
                "estimated_alpha": round(alpha_bd, 4),
                "estimated_beta": round(beta_bd, 4),
                "beta_bias_error": round(abs(beta_bd - beta_true), 4),
                "beta_relative_error_pct": round(100.0 * abs(beta_bd - beta_true) / beta_true, 2)
            }
        }


if __name__ == "__main__":
    import json
    out = {
        "multi_kink_mpcc_vs_2m": run_multi_kink_scalability_experiment(num_kinks=6),
        "temporal_cyclic_dag_mpc": TemporalCyclicSCMAndMPCSolver().solve_myopic_vs_mpc_multi_period(),
        "causal_endogeneity_iv_2sls": CausalEndogeneityIV2SLSEstimator.run_endogeneity_and_iv_experiment()
    }
    print(json.dumps(out, indent=2))

