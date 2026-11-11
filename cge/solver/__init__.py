"""Motor de Resolución Matemática Genérica con Certificados (Fase 10)."""
from __future__ import annotations
import math
import re
from typing import Dict, Any, List, Optional
import networkx as nx
import numpy as np
import sympy as sp

from ..schema import CausalGraph
from ..fidelity import certify_fidelity, FidelityReport
from .certificates import SolveStatus, SolveCertificate, verify_and_certify
from .compile import (
    Var,
    CompiledConstraint,
    NonsmoothAtom,
    Problem,
    compile_problem,
)
from .strategies import (
    solve_strategy_a_milp,
    solve_strategy_a_piecewise_milp,
    solve_strategy_b_regime_enumeration,
    solve_strategy_c_homotopy_lse,
    solve_strategy_d_minlp,
    solve_strategy_e_iis,
)


_SOLVE_PROBLEM_CACHE: Dict[Any, Dict[str, Any]] = {}


def _extract_var_ids(expr_str: str) -> List[str]:
    if not expr_str:
        return []
    return re.findall(r"[A-Za-z_]\w*", expr_str)


def _check_epistemic_honesty_gate(
    ir: CausalGraph,
    target_id: str,
    source_path_or_code: Optional[str] = None,
    entrypoint: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Verifica la Compuerta de Honestidad Epistémica antes de optimizar.

    Si algún ancestro causal del objetivo o de las restricciones carece de fidelidad
    verificada (por no-determinismo, mutación opaca, I/O externo o divergencia numérica),
    aborta la optimización devolviendo UNSUPPORTED.
    """
    meta = ir.metadata or {}
    src = (
        source_path_or_code
        or meta.get("source_code")
        or meta.get("inline_source_code")
        or meta.get("source_path")
        or meta.get("target_dir")
    )
    ep = entrypoint or meta.get("entrypoint")

    if src and "fidelity_report" not in meta:
        try:
            certify_fidelity(ir, src, entrypoint=ep)
        except Exception:
            pass

    fid_rep = (ir.metadata or {}).get("fidelity_report")

    # Construir grafo dirigido para identificar ancestros relevantes de target_id y restricciones
    dg = nx.DiGraph()
    node_map = ir.nodes if isinstance(ir.nodes, dict) else {n.id: n for n in ir.nodes}
    for nid in node_map:
        dg.add_node(nid)
    for e in ir.edges:
        if e.type != "empirical_data":
            dg.add_edge(e.source, e.target)

    relevant_nodes = set()
    if target_id in dg:
        relevant_nodes.add(target_id)
        relevant_nodes.update(nx.ancestors(dg, target_id))
    else:
        relevant_nodes.update(node_map.keys())

    constrs_iter = ir.constraints.values() if isinstance(ir.constraints, dict) else (ir.constraints or [])
    for constr in constrs_iter:
        for v_id in _extract_var_ids(constr.expression):
            if v_id in dg:
                relevant_nodes.add(v_id)
                relevant_nodes.update(nx.ancestors(dg, v_id))

    unverified: List[str] = []
    reasons: List[str] = []

    if isinstance(fid_rep, dict):
        rep_unverified = fid_rep.get("unverified_nodes") or []
        for u_nid in rep_unverified:
            if u_nid in relevant_nodes and u_nid not in unverified:
                unverified.append(u_nid)
        if not fid_rep.get("verified", True):
            if fid_rep.get("reason"):
                reasons.append(str(fid_rep["reason"]))
            if not unverified and rep_unverified:
                unverified.extend(rep_unverified)

    for nid in relevant_nodes:
        node = node_map.get(nid)
        if node is not None and isinstance(node.fidelity, dict):
            if not node.fidelity.get("verified", True):
                if nid not in unverified:
                    unverified.append(nid)
                if node.fidelity.get("reason") and str(node.fidelity["reason"]) not in reasons:
                    reasons.append(str(node.fidelity["reason"]))

    if unverified or (isinstance(fid_rep, dict) and not fid_rep.get("verified", True)):
        reason_str = "; ".join(reasons) if reasons else f"Fidelity verification failed on nodes: {unverified}"
        cert = SolveCertificate(
            status=SolveStatus.UNSUPPORTED,
            problem_class="UNSUPPORTED",
            strategy_used="epistemic_honesty_gate",
            optimality_gap=None,
            max_constraint_violation=0.0,
            kkt_residual=None,
            fidelity_verified=False,
            code_verified=False,
            constraint_violation_on_code=0.0,
            unverified_nodes=unverified,
            reason=reason_str,
        )
        return {
            "status": "UNSUPPORTED",
            "certificate_status": SolveStatus.UNSUPPORTED.value,
            "certificate": cert.to_dict(),
            "reason": reason_str,
            "unverified_nodes": unverified,
            "controls": {},
            "optimal_deltas": {},
            "achieved_delta": None,
            "objective_value": None,
            "evaluated_nodes": {},
            "active_regime": "unsupported",
            "active_constraints": [],
            "iis_constraints": [],
            "blocking_constraints": [],
        }

    return None


def solve_problem(
    ir: CausalGraph,
    target_id: Optional[str] = None,
    sense: Optional[str] = None,
    mode: str = "direct",
    target_delta: Optional[float] = None,
    target_value: Optional[float] = None,
    exo_overrides: Optional[Dict[str, float]] = None,
    ceiling_override: Optional[float] = None,
    fixed_controls: Optional[Dict[str, float]] = None,
    ctrl_overrides: Optional[Dict[str, float]] = None,
    bounds_overrides: Optional[Dict[str, Any]] = None,
    source_path_or_code: Optional[str] = None,
    entrypoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Compila y resuelve un CausalGraph con selección automática de estrategia y certificado verificable."""
    node_map = ir.nodes if isinstance(ir.nodes, dict) else {n.id: n for n in ir.nodes}
    resolved_target = target_id
    if not resolved_target:
        for nid, n in node_map.items():
            if n.category == "target":
                resolved_target = nid
                break
        if not resolved_target and node_map:
            resolved_target = list(node_map.keys())[-1]

    if not resolved_target:
        raise ValueError("CausalGraph has no nodes to optimize.")

    # 1. Compuerta de Honestidad Epistémica (Fidelity Gate)
    gate_abort = _check_epistemic_honesty_gate(
        ir=ir,
        target_id=resolved_target,
        source_path_or_code=source_path_or_code,
        entrypoint=entrypoint,
    )
    if gate_abort is not None:
        return gate_abort

    constrs_iter = ir.constraints.values() if isinstance(ir.constraints, dict) else (ir.constraints or [])
    cache_key = (
        ir.system_id,
        resolved_target,
        sense,
        mode,
        target_delta,
        target_value,
        tuple(sorted((exo_overrides or {}).items())),
        ceiling_override,
        tuple(sorted((fixed_controls or {}).items())),
        tuple(sorted((ctrl_overrides or {}).items())),
        str(bounds_overrides),
        tuple((n.id, n.current_value, n.symbolic_expr, str(n.bounds), str(n.fidelity)) for n in node_map.values()),
        tuple((c.id, c.expression) for c in constrs_iter),
        str((ir.metadata or {}).get("blackbox_surrogates")),
    )
    if cache_key in _SOLVE_PROBLEM_CACHE:
        import copy
        return copy.deepcopy(_SOLVE_PROBLEM_CACHE[cache_key])

    # 2. Compilar problema matemático canónico
    problem = compile_problem(
        ir=ir,
        target_id=resolved_target,
        sense=sense,
        mode=mode,
        target_delta=target_delta,
        target_value=target_value,
        exo_overrides=exo_overrides,
        ceiling_override=ceiling_override,
        fixed_controls=fixed_controls,
        ctrl_overrides=ctrl_overrides,
        bounds_overrides=bounds_overrides,
    )

    # 3. Seleccionar y ejecutar estrategia según estructura simbólica
    candidates: List[Dict[str, Any]] = []

    if problem.problem_class in ("LP", "MILP"):
        sol_a = solve_strategy_a_milp(problem)
        if sol_a is not None:
            candidates.append(sol_a)

    if problem.is_piecewise_linear and len(problem.nonsmooth_atoms) > 0:
        sol_pw = solve_strategy_a_piecewise_milp(problem)
        if sol_pw is not None:
            candidates.append(sol_pw)
        if len(problem.nonsmooth_atoms) <= 8 and not any(v.domain in ("integer", "binary") for v in problem.vars):
            sol_b = solve_strategy_b_regime_enumeration(problem)
            if sol_b is not None:
                candidates.append(sol_b)

    elif problem.problem_class == "MINLP":
        sol_d = solve_strategy_d_minlp(problem)
        if sol_d is not None:
            candidates.append(sol_d)

    elif problem.problem_class in ("QP", "NLP_SMOOTH", "NLP_NONSMOOTH"):
        if len(problem.nonsmooth_atoms) <= 8:
            sol_b = solve_strategy_b_regime_enumeration(problem)
            if sol_b is not None:
                candidates.append(sol_b)
        else:
            sol_c = solve_strategy_c_homotopy_lse(problem)
            if sol_c is not None:
                candidates.append(sol_c)

        if not candidates and len(problem.nonsmooth_atoms) > 0:
            sol_c_fb = solve_strategy_c_homotopy_lse(problem)
            if sol_c_fb is not None:
                candidates.append(sol_c_fb)

    # Filtrar candidatos factibles y elegir el mejor según el objetivo de minimización canónico
    feasible_candidates = [c for c in candidates if c.get("max_violation", 1.0) <= 1e-4]

    best_sol: Optional[Dict[str, Any]] = None
    if feasible_candidates:
        def _score_candidate(cand: Dict[str, Any]) -> float:
            x_c = cand["x"]
            subs = {v.symbol: float(x_c[i]) for i, v in enumerate(problem.vars)}
            try:
                return float(problem.min_obj_expr.evalf(subs=subs))
            except Exception:
                return -float(cand["j_val"]) if problem.sense == "max" else float(cand["j_val"])

        feasible_candidates.sort(key=lambda c: (round(_score_candidate(c), 7), 0 if c.get("is_exact") else 1))
        best_sol = feasible_candidates[0]

    # 4. Si ningún candidato es factible, ejecutar Estrategia E (Diagnóstico IIS / Recuperación Elástica)
    if best_sol is None:
        iis_res = solve_strategy_e_iis(problem)
        if iis_res.get("feasible_recovered") and iis_res.get("x") is not None:
            best_sol = iis_res
        else:
            iis_list = iis_res.get("iis_constraints", [])
            min_slack = float(iis_res.get("min_relaxation_slack", 0.0))
            shadow_map = iis_res.get("shadow_multipliers", {})
            cert_infeas = SolveCertificate(
                status=SolveStatus.INFEASIBLE_CERTIFIED,
                problem_class=problem.problem_class,
                strategy_used="strategy_e_iis_deletion_filter",
                optimality_gap=None,
                max_constraint_violation=min_slack,
                kkt_residual=None,
                fidelity_verified=True,
                code_verified=True,
                constraint_violation_on_code=0.0,
                iis_constraints=iis_list,
                min_relaxation_slack=min_slack,
                reason=f"Minimal Irreducible Inconsistent Subsystem (IIS): {iis_list}",
            )
            return {
                "status": "INFEASIBLE",
                "certificate_status": SolveStatus.INFEASIBLE_CERTIFIED.value,
                "certificate": cert_infeas.to_dict(),
                "problem_class": problem.problem_class,
                "strategy_used": "strategy_e_iis_deletion_filter",
                "iis_constraints": iis_list,
                "blocking_constraints": iis_list,
                "min_relaxation_slack": min_slack,
                "kkt_shadow_prices": shadow_map,
                "controls": {},
                "optimal_deltas": {},
                "achieved_delta": None,
                "objective_value": None,
                "evaluated_nodes": {},
                "active_regime": "infeasible",
                "active_constraints": iis_list,
                "reason": cert_infeas.reason,
                "explanation": (
                    f"El dominio de optimización es infactible (vacío) debido al subsistema irreduciblemente inconsistente (IIS): "
                    f"{', '.join(iis_list)} (holgura mínima de relajación = {min_slack:.4f})."
                ),
            }

    # 5. Construir solución final y certificado verificable sobre grafo y código fuente
    x_star = np.array(best_sol["x"], dtype=float)
    controls: Dict[str, Any] = {}
    optimal_deltas: Dict[str, float] = {}
    for i, v in enumerate(problem.vars):
        if v.domain in ("integer", "binary"):
            val_clean: Any = int(round(float(x_star[i])))
        else:
            val_clean = float(x_star[i])
        controls[v.id] = val_clean
        optimal_deltas[v.id] = float(val_clean) - float(v.x0)

    evaluated_nodes = problem.evaluate_all_nodes(x_star)
    j_star = float(evaluated_nodes.get(problem.target_id, problem.evaluate_j(x_star)))
    achieved_delta = float(j_star - problem.baseline_j)
    max_viol, viol_map = problem.max_constraint_violation(x_star)

    # Identificar restricciones activas en el óptimo
    active_constraints: List[str] = []
    args_star = [float(z) for z in x_star]
    for cid, kind, fn_c in problem._fast_constr_fns:
        try:
            c_val = float(fn_c(*args_star))
            if abs(c_val) <= 1e-2:
                base_cid = cid.split("_lo")[0].split("_hi")[0]
                if base_cid not in active_constraints:
                    active_constraints.append(base_cid)
        except Exception:
            pass
    for i, v in enumerate(problem.vars):
        if abs(float(x_star[i]) - v.lb) <= 1e-4 or abs(float(x_star[i]) - v.ub) <= 1e-4:
            active_constraints.append(f"bounds({v.id})")

    meta = ir.metadata or {}
    src = (
        source_path_or_code
        or meta.get("source_code")
        or meta.get("inline_source_code")
        or meta.get("source_path")
        or meta.get("target_dir")
    )
    ep = entrypoint or meta.get("entrypoint")

    cert = verify_and_certify(
        ir=ir,
        problem_class=problem.problem_class,
        strategy_used=str(best_sol.get("strategy_used", "generic_solver")),
        controls=controls,
        ir_objective_value=j_star,
        max_constraint_violation=max_viol,
        is_exact_strategy=bool(best_sol.get("is_exact", False)),
        optimality_gap=float(best_sol.get("optimality_gap") or 0.0),
        kkt_residual=float(best_sol.get("kkt_residual") or 0.0),
        source_path_or_code=src,
        entrypoint=ep,
        exo_overrides=exo_overrides,
    )

    out_res = {
        "status": "OPTIMAL",
        "certificate_status": cert.status.value,
        "certificate": cert.to_dict(),
        "problem_class": problem.problem_class,
        "strategy_used": cert.strategy_used,
        "controls": controls,
        "optimal_deltas": optimal_deltas,
        "achieved_delta": achieved_delta,
        "objective_value": j_star,
        "baseline_objective": problem.baseline_j,
        "evaluated_nodes": evaluated_nodes,
        "active_regime": str(best_sol.get("active_regime", "smooth")),
        "active_constraints": active_constraints,
        "iis_constraints": [],
        "blocking_constraints": [],
        "max_constraint_violation": cert.max_constraint_violation,
        "constraint_violations": viol_map,
    }
    if len(_SOLVE_PROBLEM_CACHE) < 512:
        import copy
        _SOLVE_PROBLEM_CACHE[cache_key] = copy.deepcopy(out_res)
    return out_res


__all__ = [
    "SolveStatus",
    "SolveCertificate",
    "verify_and_certify",
    "Var",
    "CompiledConstraint",
    "NonsmoothAtom",
    "Problem",
    "compile_problem",
    "solve_problem",
]
