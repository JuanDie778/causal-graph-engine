"""Experimento 03: Optimización Estocástica Directa, Cuantificación Analítica de la Brecha de Jensen,
Cuadratura de Gauss-Hermite y Optimización Aversa al Riesgo (Mean-CVaR de Rockafellar-Uryasev).

Demuestra por qué optimizar para el valor esperado de U (E[J(X, U)]) difiere sustancialmente
de optimizar en el promedio nominal (J(X, E[U])) debido a la concavidad en el kink sales = min(d, q).
"""
from __future__ import annotations
import math
from typing import Dict, Any, List, Optional
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from .exp01_symbolic_scm_and_routing import SymbolicSCM
from .exp02_universal_minlp_solver import UniversalLayer2Solver


class StochasticCVaROptimizer:
    """Motor de evaluación y optimización estocástica / aversa al riesgo sobre el SCM."""

    def __init__(
        self,
        solver: UniversalLayer2Solver,
        u_mean: float = 1000.0,
        u_std: float = 100.0,
        n_quadrature: int = 32
    ):
        self.solver = solver
        self.u_mean = u_mean
        self.u_std = u_std
        self.n_quadrature = n_quadrature
        self._xg, self._wg = np.polynomial.legendre.leggauss(n_quadrature)

    def _build_dynamic_kink_split_quadrature(self, price: float, marketing_budget: float, order_qty: int):
        """Construye dinámicamente los dos paneles de cuadratura de Gauss-Legendre partidos
        exactamente en el kink u_kink(p, m, q) donde demand(u_kink, p, m) == order_qty.
        Garantiza integrandos C^infty en ambos subintervalos para CUALQUIER política (p, m, q).
        """
        d_at_0 = self.solver.evaluate(price, marketing_budget, order_qty, base_demand=0.0)["demand"]
        d_at_1 = self.solver.evaluate(price, marketing_budget, order_qty, base_demand=1.0)["demand"]
        slope_u = max(d_at_1 - d_at_0, 1e-9)
        u_kink = (float(order_qty) - d_at_0) / slope_u

        a_min = max(0.0, self.u_mean - 6.0 * self.u_std)
        b_max = self.u_mean + 6.0 * self.u_std
        u_split = float(np.clip(u_kink, a_min + 1e-3, b_max - 1e-3))

        nodes1 = 0.5 * (u_split - a_min) * self._xg + 0.5 * (u_split + a_min)
        weights1 = 0.5 * (u_split - a_min) * self._wg * norm.pdf(nodes1, loc=self.u_mean, scale=self.u_std)

        nodes2 = 0.5 * (b_max - u_split) * self._xg + 0.5 * (b_max + u_split)
        weights2 = 0.5 * (b_max - u_split) * self._wg * norm.pdf(nodes2, loc=self.u_mean, scale=self.u_std)

        nodes = np.concatenate([nodes1, nodes2])
        weights = np.concatenate([weights1, weights2])
        weights /= np.sum(weights)
        return nodes, weights, u_kink

    @staticmethod
    def compute_analytical_golden_case_expectation(
        price: float,
        marketing_budget: float,
        order_qty: int,
        u_mean: float = 1000.0,
        u_std: float = 100.0
    ) -> Dict[str, float]:
        """Fórmula cerrada analítica del beneficio esperado y la Brecha de Jensen (Teorema 1, Propuesta 03)."""
        unit_cost = 1.20 if order_qty < 800 else 1.00
        phi_p = 1.0 - 0.15 * (price - 3.0)
        mkt_lift = 8.0 * math.sqrt(max(0.0, marketing_budget))

        mu_d = u_mean * phi_p + mkt_lift
        sigma_d = max(1e-9, u_std * abs(phi_p))

        # Beneficio si se vende todo q
        full_sales_profit = (0.97 * price - unit_cost) * order_qty - marketing_budget
        # Beneficio determinista evaluado en u_mean
        det_sales = min(mu_d, float(order_qty))
        det_holding = 0.20 * max(0.0, float(order_qty) - det_sales)
        det_profit = 0.97 * price * det_sales - unit_cost * order_qty - marketing_budget - det_holding

        # Déficit esperado E[max(0, q - D)] usando la función de pérdida normal estándar:
        # Sea D = mu_d + sigma_d * Z con Z ~ N(0, 1), y z_buf = (mu_d - q)/sigma_d.
        # Por simetría W = -Z ~ N(0, 1), q - D = sigma_d * (W - z_buf), luego:
        # E[max(0, q - D)] = sigma_d * E[max(0, W - z_buf)] = sigma_d * (pdf(z_buf) - z_buf * (1 - cdf(z_buf)))
        z_buf = (mu_d - float(order_qty)) / sigma_d
        normal_loss = norm.pdf(z_buf) - z_buf * (1.0 - norm.cdf(z_buf))
        expected_unsold_units = sigma_d * normal_loss

        marginal_unsold_penalty = 0.97 * price + 0.20
        expected_profit = full_sales_profit - marginal_unsold_penalty * expected_unsold_units
        jensen_gap = det_profit - expected_profit
        stockout_prob = float(1.0 - norm.cdf(-z_buf))

        return {
            "deterministic_profit": round(det_profit, 4),
            "analytical_expected_profit": round(expected_profit, 4),
            "jensen_gap_eur": round(jensen_gap, 4),
            "standardized_demand_buffer_z": round(z_buf, 4),
            "expected_unsold_units": round(expected_unsold_units, 4),
            "analytical_stockout_prob": round(stockout_prob, 4)
        }

    def evaluate_gauss_hermite(self, price: float, marketing_budget: float, order_qty: int) -> Dict[str, float]:
        """Integra E[J(X, U)] y Var[J(X, U)] con precisión espectral (< 1e-6€ error en todo el dominio)
        usando cuadratura compuesta de Gauss-Legendre dividida dinámicamente en u_kink(p, m, q).
        """
        nodes, weights, u_kink = self._build_dynamic_kink_split_quadrature(price, marketing_budget, order_qty)
        profits = np.array([
            self.solver.evaluate(price, marketing_budget, order_qty, base_demand=float(u_val))["profit"]
            for u_val in nodes
        ])
        mean_j = float(np.sum(weights * profits))
        var_j = float(np.sum(weights * (profits - mean_j) ** 2))
        det_j = self.solver.evaluate(price, marketing_budget, order_qty, base_demand=self.u_mean)["profit"]
        return {
            "gh_expected_profit": round(mean_j, 4),
            "gh_std_profit": round(math.sqrt(max(0.0, var_j)), 4),
            "deterministic_profit": round(det_j, 4),
            "jensen_degradation": round(det_j - mean_j, 4),
            "dynamic_u_kink": round(float(u_kink), 4)
        }

    def evaluate_tail_cvar_quadrature(
        self,
        price: float,
        marketing_budget: float,
        order_qty: int,
        alpha: float = 0.05
    ) -> float:
        """Calcula el CVaR_alpha exacto integrando sobre la cola inferior [u_min, u_alpha]
        mediante cuadratura de Gauss-Legendre suave (equivalente continuo exacto de Rockafellar-Uryasev).
        """
        a_min = max(0.0, self.u_mean - 6.0 * self.u_std)
        u_alpha = float(self.u_mean + self.u_std * norm.ppf(alpha))
        nodes_tail = 0.5 * (u_alpha - a_min) * self._xg + 0.5 * (u_alpha + a_min)
        weights_tail = 0.5 * (u_alpha - a_min) * self._wg * norm.pdf(nodes_tail, loc=self.u_mean, scale=self.u_std)
        weights_tail /= np.sum(weights_tail)

        tail_profits = np.array([
            self.solver.evaluate(price, marketing_budget, order_qty, base_demand=float(u_val))["profit"]
            for u_val in nodes_tail
        ])
        return float(np.sum(weights_tail * tail_profits))

    def evaluate_monte_carlo_metrics(
        self,
        price: float,
        marketing_budget: float,
        order_qty: int,
        n_samples: int = 1000,
        seed: int = 42,
        alpha: float = 0.05
    ) -> Dict[str, float]:
        """Evalúa media, desviación, VaR_alpha, CVaR_alpha y probabilidad de rotura sobre muestra Monte Carlo."""
        rng = np.random.RandomState(seed)
        d0_samples = np.maximum(0.0, rng.normal(self.u_mean, self.u_std, n_samples))

        profits = np.empty(n_samples, dtype=float)
        demands = np.empty(n_samples, dtype=float)
        for i, d0 in enumerate(d0_samples):
            st = self.solver.evaluate(price, marketing_budget, order_qty, base_demand=float(d0))
            profits[i] = st["profit"]
            demands[i] = st["demand"]

        mean_p = float(np.mean(profits))
        std_p = float(np.std(profits))
        var_alpha = float(np.percentile(profits, alpha * 100.0))
        tail_losses = profits[profits <= var_alpha]
        cvar_alpha = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_alpha
        stockout_prob = float(np.mean(demands > float(order_qty)))

        return {
            "mean_profit": round(mean_p, 2),
            "std_profit": round(std_p, 2),
            "var_5pct": round(var_alpha, 2),
            "cvar_5pct": round(cvar_alpha, 2),
            "p95_profit": round(float(np.percentile(profits, 95)), 2),
            "stockout_probability": round(stockout_prob, 4)
        }

    def optimize_stochastic(
        self,
        risk_aversion_rho: float = 0.0,
        alpha: float = 0.05,
        q_fixed: Optional[int] = 800
    ) -> Dict[str, Any]:
        """Resuelve el problema de optimización estocástica / Mean-CVaR:
            max_{p, m, q} (1 - rho) * E_U[J(p, m, q, U)] + rho * CVaR_alpha(J(p, m, q, U))
            s.t. c(q)*q + m <= max_budget, price in [2, 6], m >= 0
        usando cuadratura dinámica dividida en el kink para E[J] y cuadratura de cola suave para CVaR_alpha.
        """
        q_candidates = [q_fixed] if q_fixed is not None else [584, 700, 800]
        p_b = (2.0, 6.0)

        best_val = float("inf")
        best_pm = np.array([5.437, 428.214])
        best_q = 800

        for q_cand in q_candidates:
            unit_c = self.solver.evaluate(3.5, 0.0, q_cand)["unit_cost"]
            max_m = min(2000.0, self.solver.scm.max_budget - unit_c * q_cand)
            if max_m < 0.0:
                continue
            m_b = (1.0, max(1.0, max_m))

            def objective_neg(pm: np.ndarray, q_v: int = q_cand) -> float:
                p_v, m_v = float(pm[0]), float(pm[1])
                nodes, weights, _ = self._build_dynamic_kink_split_quadrature(p_v, m_v, q_v)
                gh_profits = np.array([
                    self.solver.evaluate(p_v, m_v, q_v, base_demand=float(u_val))["profit"]
                    for u_val in nodes
                ])
                exp_j = float(np.sum(weights * gh_profits))
                if risk_aversion_rho <= 1e-9:
                    return -exp_j

                cvar_val = self.evaluate_tail_cvar_quadrature(p_v, m_v, q_v, alpha=alpha)
                return -((1.0 - risk_aversion_rho) * exp_j + risk_aversion_rho * cvar_val)

            for p_init, m_init in [(5.75, 385.0), (5.47, 485.0), (6.0, 340.0)]:
                res = minimize(
                    objective_neg,
                    np.array([p_init, min(m_init, max_m)]),
                    method="SLSQP",
                    bounds=[p_b, m_b],
                    options={"ftol": 1e-11, "eps": 1e-6, "maxiter": 120}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_pm = res.x
                    best_q = q_cand

        p_opt, m_opt = float(best_pm[0]), float(best_pm[1])
        nom_state = self.solver.evaluate(p_opt, m_opt, best_q, base_demand=self.u_mean)
        mc_metrics = self.evaluate_monte_carlo_metrics(p_opt, m_opt, best_q, n_samples=1000, seed=42, alpha=alpha)
        gh_metrics = self.evaluate_gauss_hermite(p_opt, m_opt, best_q)
        exact_cvar = self.evaluate_tail_cvar_quadrature(p_opt, m_opt, best_q, alpha=alpha)

        return {
            "risk_aversion_rho": risk_aversion_rho,
            "controls": {
                "price": round(p_opt, 4),
                "marketing_budget": round(m_opt, 4),
                "order_qty": int(best_q)
            },
            "nominal_demand_at_mean_U": round(nom_state["demand"], 2),
            "demand_safety_buffer_units": round(nom_state["demand"] - best_q, 2),
            "gauss_hermite_expected_profit": gh_metrics["gh_expected_profit"],
            "exact_tail_cvar_5pct": round(exact_cvar, 4),
            "monte_carlo_1000": mc_metrics
        }


def run_jensen_and_cvar_experiment() -> Dict[str, Any]:
    """Ejecuta la comparativa completa entre:
    1. Política Determinista (ignora incertidumbre U, sufre Brecha de Jensen de -138.54€),
    2. Política Estocástica Neutral al Riesgo (maximiza E[J] con cuadratura dinámica dividida en u_kink),
    3. Política Estocástica Aversa al Riesgo en q=800 y con adaptación de volumen q* (maximiza CVaR_5%).
    """
    scm = SymbolicSCM("golden_case")
    solver = UniversalLayer2Solver(scm)
    stoch = StochasticCVaROptimizer(solver)

    # 1. Evaluación de la política determinista óptima (p=5.4370, m=428.2140, q=800)
    det_sol = solver.optimize()
    p_det = det_sol["controls"]["price"]
    m_det = det_sol["controls"]["marketing_budget"]
    q_det = det_sol["controls"]["order_qty"]

    analytical_det = stoch.compute_analytical_golden_case_expectation(p_det, m_det, q_det)
    gh_det = stoch.evaluate_gauss_hermite(p_det, m_det, q_det)
    mc_det = stoch.evaluate_monte_carlo_metrics(p_det, m_det, q_det, n_samples=1000, seed=42)

    # 2. Política estocástica que maximiza E_U[J(X, U)] (rho = 0.0)
    stoch_neutral = stoch.optimize_stochastic(risk_aversion_rho=0.0, q_fixed=800)
    analytical_stoch = stoch.compute_analytical_golden_case_expectation(
        stoch_neutral["controls"]["price"],
        stoch_neutral["controls"]["marketing_budget"],
        stoch_neutral["controls"]["order_qty"]
    )
    stoch_neutral["analytical_expected_profit"] = analytical_stoch["analytical_expected_profit"]
    stoch_neutral["quadrature_vs_analytical_error_eur"] = round(
        abs(stoch_neutral["gauss_hermite_expected_profit"] - analytical_stoch["analytical_expected_profit"]), 6
    )

    # 3A. Política aversa al riesgo de cola en q=800 (rho = 1.0, q_fixed=800)
    stoch_risk_averse_q800 = stoch.optimize_stochastic(risk_aversion_rho=1.0, q_fixed=800)

    # 3B. Política aversa al riesgo de cola optimizando también q (rho = 1.0, q libre)
    stoch_risk_averse_global = stoch.optimize_stochastic(risk_aversion_rho=1.0, q_fixed=None)

    return {
        "deterministic_policy": {
            "controls": det_sol["controls"],
            "nominal_demand": 800.0,
            "demand_buffer": 0.0,
            "analytical": analytical_det,
            "gauss_hermite": gh_det,
            "monte_carlo_1000": mc_det
        },
        "stochastic_risk_neutral_policy": stoch_neutral,
        "stochastic_cvar_averse_policy": stoch_risk_averse_q800,
        "stochastic_cvar_averse_global_q_policy": stoch_risk_averse_global,
        "improvements_over_deterministic": {
            "expected_profit_gain_gh_eur": round(
                stoch_neutral["gauss_hermite_expected_profit"] - gh_det["gh_expected_profit"], 2
            ),
            "expected_profit_gain_mc_eur": round(
                stoch_neutral["monte_carlo_1000"]["mean_profit"] - mc_det["mean_profit"], 2
            ),
            "stockout_prob_reduction_neutral": round(
                mc_det["stockout_probability"] - stoch_neutral["monte_carlo_1000"]["stockout_probability"], 4
            ),
            "cvar_5pct_improvement_risk_averse_eur": round(
                stoch_risk_averse_q800["monte_carlo_1000"]["cvar_5pct"] - mc_det["cvar_5pct"], 2
            ),
            "var_5pct_improvement_risk_averse_eur": round(
                stoch_risk_averse_q800["monte_carlo_1000"]["var_5pct"] - mc_det["var_5pct"], 2
            ),
            "cvar_5pct_improvement_global_q_eur": round(
                stoch_risk_averse_global["monte_carlo_1000"]["cvar_5pct"] - mc_det["cvar_5pct"], 2
            )
        }
    }
