"""Simulación estocástica Monte Carlo, SAA y optimización Mean-CVaR sobre el CausalIR (R-10, F5)."""
from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
from scipy.optimize import minimize

from .layer2_exact import Layer2Solver


def run_monte_carlo_simulation(
    controls: Optional[Dict[str, float]] = None,
    n_samples: int = 1000,
    seed: int = 42,
    solver: Optional[Layer2Solver] = None,
    **kwargs
) -> Dict[str, Any]:
    """Evalúa la distribución del objetivo J y la probabilidad de saturación de capacidad
    muestreando las distribuciones exógenas U del CausalIR sin ecuaciones ni nombres cableados.
    """
    if solver is None:
        solver = Layer2Solver()

    target_id = solver.target_id
    ctrl_map: Dict[str, float] = {}
    if controls:
        ctrl_map.update(controls)
    for k, v in kwargs.items():
        if k in solver.control_ids and v is not None:
            ctrl_map[k] = float(v)

    if not ctrl_map:
        opt_sol = solver.optimize()
        ctrl_map = {k: float(v) for k, v in opt_sol["controls"].items()}
        nominal_j = float(opt_sol["objective_value"])
    else:
        nom_ev = solver.evaluate(**ctrl_map)
        nominal_j = float(nom_ev[target_id])

    rng = np.random.RandomState(seed)
    u_samples_map: Dict[str, np.ndarray] = {}
    for uid in solver.exogenous_ids:
        u_node = solver.graph.nodes[uid]
        dist = u_node.distribution or {}
        u_mean = float(dist.get("mean", u_node.current_value if u_node.current_value is not None else 0.0))
        u_std = float(dist.get("std", 0.1 * abs(u_mean)))
        b_min = u_node.bounds.get("min")
        if dist.get("family") == "uniform":
            half_span = np.sqrt(3.0) * u_std
            draws = rng.uniform(u_mean - half_span, u_mean + half_span, n_samples)
        else:
            draws = rng.normal(u_mean, u_std, n_samples)
        if b_min is not None:
            draws = np.maximum(float(b_min), draws)
        u_samples_map[uid] = draws

    j_values = np.empty(n_samples, dtype=float)
    saturation_events = np.zeros(n_samples, dtype=bool)
    min_nodes = getattr(solver, "_min_nodes_compiled", [])

    for i in range(n_samples):
        step_inputs = dict(ctrl_map)
        for uid, arr in u_samples_map.items():
            step_inputs[uid] = float(arr[i])
        st = solver.evaluate(**step_inputs)
        j_values[i] = float(st[target_id])
        for min_nid, min_syms, fn_a, fn_b in min_nodes:
            try:
                va = float(fn_a(*[st[s] for s in min_syms]))
                vb = float(fn_b(*[st[s] for s in min_syms]))
                if abs(va - vb) > 1e-6 and st.get(min_nid, 0.0) < max(va, vb) - 1e-6:
                    # Detectar cuando el argumento estocástico supera la cota discreta/capacidad
                    if any(s in solver.discrete_controls for s in min_syms):
                        disc_v = next(st[s] for s in min_syms if s in solver.discrete_controls)
                        if max(va, vb) > disc_v + 1e-6:
                            saturation_events[i] = True
            except Exception:
                pass

    mean_j = float(np.mean(j_values))
    std_j = float(np.std(j_values))
    p5_j = float(np.percentile(j_values, 5))
    p95_j = float(np.percentile(j_values, 95))
    tail_vals = j_values[j_values <= p5_j]
    cvar_5 = float(np.mean(tail_vals)) if len(tail_vals) > 0 else p5_j
    stockout_prob = float(np.mean(saturation_events))

    base_ev = solver.evaluate()
    base_j = float(base_ev[target_id])

    return {
        "n_samples": n_samples,
        f"mean_{target_id}": round(mean_j, 2),
        f"std_{target_id}": round(std_j, 2),
        f"p5_{target_id}": round(p5_j, 2),
        f"p95_{target_id}": round(p95_j, 2),
        "cvar_5pct": round(cvar_5, 2),
        "stockout_probability": round(stockout_prob, 4),
        f"nominal_{target_id}": round(nominal_j, 2),
        f"{target_id}_degradation_due_to_noise": round(nominal_j - mean_j, 2),
        "mean_objective": round(mean_j, 2),
        "std_objective": round(std_j, 2),
        "nominal_objective": round(nominal_j, 2),
        "jensen_gap": round(nominal_j - mean_j, 2),
        "expected_improvement": round(mean_j - base_j, 2),
        "optimal_policy": {
            "expected_objective": round(mean_j, 2),
            "objective_std": round(std_j, 2),
            "cvar_95": round(cvar_5, 2)
        }
    }


def optimize_stochastic_saa_cvar(
    solver: Optional[Layer2Solver] = None,
    risk_aversion_rho: float = 0.0,
    alpha: float = 0.05,
    n_samples: int = 250,
    seed: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """Optimiza directamente bajo incertidumbre U mediante Sample Average Approximation (SAA) y Mean-CVaR."""
    if solver is None:
        solver = Layer2Solver()
    if "n_scenarios" in kwargs and kwargs["n_scenarios"] is not None:
        n_samples = int(kwargs["n_scenarios"])
    if "risk_aversion" in kwargs and kwargs["risk_aversion"] is not None:
        risk_aversion_rho = float(kwargs["risk_aversion"])
    if "beta" in kwargs and kwargs["beta"] is not None:
        alpha = max(0.01, 1.0 - float(kwargs["beta"]))

    det_sol = solver.optimize()
    cont_ids = list(solver.continuous_controls)
    if det_sol.get("status") != "FEASIBLE" or not cont_ids:
        return det_sol

    n_cont = len(cont_ids)
    disc_ids = list(solver.discrete_controls)

    rng = np.random.RandomState(seed)
    u_draws: Dict[str, np.ndarray] = {}
    for uid in solver.exogenous_ids:
        u_node = solver.graph.nodes[uid]
        dist = u_node.distribution or {}
        u_mean = float(dist.get("mean", u_node.current_value or 1000.0))
        u_std = float(dist.get("std", 0.1 * abs(u_mean)))
        b_min = u_node.bounds.get("min")
        if dist.get("family") == "uniform":
            half_span = np.sqrt(3.0) * u_std
            arr = rng.uniform(u_mean - half_span, u_mean + half_span, n_samples)
        else:
            arr = rng.normal(u_mean, u_std, n_samples)
        if b_min is not None:
            arr = np.maximum(float(b_min), arr)
        u_draws[uid] = arr

    upfront_syms = set(solver._upfront_cost_compiled[0]) if solver._upfront_cost_compiled else set()
    budget_cont_ids = {c for c in cont_ids if c in upfront_syms}

    # Generar candidatos discretos para optimización conjunta de controles discretos y continuos
    disc_candidates: List[List[int]] = []
    for d in disc_ids:
        b = solver.graph.nodes[d].bounds
        b_lo = int(b.get("min", 0) or 0)
        b_hi = int(b.get("max", 5000) or 5000)
        q_det = int(round(det_sol["controls"].get(d, 0)))
        cand_set = {q_det, b_lo, b_hi}
        span = max(50, int(0.15 * max(abs(q_det), 1)))
        for offset in (-2 * span, -span, int(-0.5 * span), int(0.5 * span), span, 2 * span):
            qv = q_det + offset
            if b_lo <= qv <= b_hi:
                cand_set.add(qv)
        for n in solver.graph.nodes.values():
            for dc in n.discontinuities:
                if dc.get("variable") == d and "threshold" in dc:
                    th = int(dc["threshold"])
                    if b_lo <= th <= b_hi:
                        cand_set.add(th)
        disc_candidates.append(sorted(c for c in cand_set if b_lo <= c <= b_hi))

    import itertools
    all_disc_tuples = list(itertools.product(*disc_candidates)) if disc_candidates else [()]

    best_score = -float("inf")
    best_stoch_ctrls = dict(det_sol["controls"])

    for q_tuple in all_disc_tuples:
        q_dict = {disc_ids[k]: q_tuple[k] for k in range(len(disc_ids))}
        probe_in = {
            c: float(solver.graph.nodes[c].bounds.get("min", 0.0) or 0.0)
            for c in cont_ids
        }
        probe_in.update(q_dict)
        probe = solver.evaluate(**probe_in)
        if probe["budget_used"] > solver.max_budget + 1e-4:
            continue
        rem_budget = solver.max_budget - float(probe["budget_used"])

        bounds_saa = []
        x0_list = []
        for c in cont_ids:
            b = solver.graph.nodes[c].bounds
            b_lo = float(b["min"]) if b.get("min") is not None else 0.0
            b_hi = float(b["max"]) if b.get("max") is not None else 1000.0
            if c in budget_cont_ids:
                b_hi = max(b_lo, min(b_hi, rem_budget))
            bounds_saa.append((b_lo, b_hi))
            x0_list.append(float(np.clip(det_sol["controls"][c], b_lo, b_hi)))

        def obj_saa(xv: np.ndarray, qd=q_dict) -> float:
            vals = np.empty(n_samples, dtype=float)
            base_in = {cont_ids[i]: float(xv[i]) for i in range(n_cont)}
            base_in.update(qd)
            for i in range(n_samples):
                for uid, arr in u_draws.items():
                    base_in[uid] = float(arr[i])
                vals[i] = float(solver.evaluate(**base_in)[solver.target_id])
            mean_v = float(np.mean(vals))
            if risk_aversion_rho <= 1e-9:
                return -mean_v
            q_alpha = float(np.percentile(vals, alpha * 100.0))
            tail = vals[vals <= q_alpha]
            cvar_v = float(np.mean(tail)) if len(tail) > 0 else q_alpha
            return -((1.0 - risk_aversion_rho) * mean_v + risk_aversion_rho * cvar_v)

        saa_cons = []
        if solver.max_budget < 1e12:
            saa_cons.append({
                "type": "ineq",
                "fun": lambda xv, qd=q_dict: solver.max_budget - float(
                    solver.evaluate(**{**{cont_ids[i]: float(xv[i]) for i in range(n_cont)}, **qd})["budget_used"]
                )
            })

        res = minimize(
            obj_saa,
            np.array(x0_list, dtype=float),
            method="SLSQP",
            bounds=bounds_saa,
            constraints=saa_cons,
            options={"ftol": 1e-7, "eps": 1e-3, "maxiter": 60}
        )
        cand_score = -float(res.fun) if res.success or hasattr(res, "fun") else -float("inf")
        if cand_score > best_score:
            best_score = cand_score
            cur_ctrls = dict(det_sol["controls"])
            cur_ctrls.update(q_dict)
            for i, c in enumerate(cont_ids):
                cur_ctrls[c] = round(float(np.clip(res.x[i], bounds_saa[i][0], bounds_saa[i][1])), 4)
            best_stoch_ctrls = cur_ctrls

    mc_eval = run_monte_carlo_simulation(controls=best_stoch_ctrls, n_samples=1000, seed=seed, solver=solver)

    return {
        "status": "FEASIBLE",
        "risk_aversion_rho": risk_aversion_rho,
        "controls": best_stoch_ctrls,
        "expected_objective": mc_eval["mean_objective"],
        "cvar_95": mc_eval["cvar_5pct"],
        "monte_carlo_metrics": mc_eval
    }
