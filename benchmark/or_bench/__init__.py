"""Suite Externa de Benchmark OR-Bench (20 Problemas Clásicos de Investigación Operativa, Fase 10)."""
from __future__ import annotations
import math
from typing import Dict, Any, List, Optional, Tuple

from cge.ingest import compile_source_to_ir
from cge.fidelity import certify_fidelity, evaluate_recipe_on_source
from cge.solver import solve_problem, SolveStatus
from .oracles import ORACLE_REGISTRY
from .problems_public import PUBLIC_PROBLEMS, get_public_problems
from .problems_sealed import SEALED_PROBLEMS, get_sealed_problems


def get_all_or_bench_problems() -> List[Dict[str, Any]]:
    return get_public_problems() + get_sealed_problems()


_OR_EVAL_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}


def evaluate_single_or_problem(prob: Dict[str, Any]) -> Dict[str, Any]:
    """Ejecuta extremo a extremo un problema de OR-Bench: ingesta -> fidelidad -> solver -> verificación en código vs oráculo."""
    pid = prob["id"]
    src = prob["source_code"]
    cache_key = (pid, src)
    if cache_key in _OR_EVAL_CACHE:
        return dict(_OR_EVAL_CACHE[cache_key])
    ep = prob.get("entrypoint")
    sense = prob.get("sense", "max")
    oracle = prob["oracle"]
    exp_status = prob["expected_status"]

    ir = compile_source_to_ir(src, entrypoint=ep, system_id=pid)
    fid_dict = (ir.metadata or {}).get("fidelity_report")
    if not fid_dict:
        fid_rep = certify_fidelity(ir, src, entrypoint=ep)
        fid_dict = fid_rep.to_dict()

    sol = solve_problem(
        ir=ir,
        sense=sense,
        mode="direct",
        source_path_or_code=src,
        entrypoint=ep,
    )

    cert = sol.get("certificate") or {}
    cert_status = sol.get("certificate_status") or cert.get("status") or sol.get("status")

    passed = False
    rel_gap = None
    code_viol = float(cert.get("constraint_violation_on_code") or 0.0)
    false_confidence = False

    if exp_status == "UNSUPPORTED":
        if cert_status == "UNSUPPORTED" or sol.get("status") == "UNSUPPORTED":
            passed = True
        else:
            passed = False
            false_confidence = True

    elif exp_status == "INFEASIBLE_CERTIFIED":
        if cert_status == "INFEASIBLE_CERTIFIED" or sol.get("status") == "INFEASIBLE":
            exp_iis = set(oracle.get("iis_constraints", []))
            got_iis = set(sol.get("iis_constraints", []))
            passed = (exp_iis == got_iis)
        else:
            passed = False
            false_confidence = True

    elif exp_status == "CERTIFIED_OPTIMAL":
        if sol.get("status") == "OPTIMAL" and cert_status in ("CERTIFIED_OPTIMAL", "BEST_FOUND"):
            j_engine = float(sol["objective_value"])
            j_oracle = float(oracle["optimal_objective"])
            rel_gap = abs(j_engine - j_oracle) / max(1.0, abs(j_oracle))
            fid_ok = bool(fid_dict.get("verified", False)) and float(fid_dict.get("max_rel_error", 1.0)) <= 1e-6
            code_ok = bool(cert.get("code_verified", False)) and code_viol <= 1e-5
            passed = (rel_gap <= 0.01) and fid_ok and code_ok
            if not code_ok and cert_status == "CERTIFIED_OPTIMAL":
                false_confidence = True

    out = {
        "id": pid,
        "name": prob["name"],
        "split": "public" if pid.startswith("p") else "sealed",
        "expected_status": exp_status,
        "engine_status": cert_status,
        "problem_class": sol.get("problem_class"),
        "strategy_used": sol.get("strategy_used"),
        "fidelity_verified": bool(fid_dict.get("verified", False)),
        "fidelity_max_rel_error": fid_dict.get("max_rel_error"),
        "engine_objective": sol.get("objective_value"),
        "oracle_objective": oracle.get("optimal_objective"),
        "rel_optimality_gap": rel_gap,
        "code_verified": cert.get("code_verified"),
        "constraint_violation_on_code": code_viol,
        "iis_constraints": sol.get("iis_constraints", []),
        "oracle_iis": oracle.get("iis_constraints", []),
        "unverified_nodes": sol.get("unverified_nodes", []),
        "passed": passed,
        "false_confidence": false_confidence,
    }
    _OR_EVAL_CACHE[cache_key] = out
    return dict(out)


def evaluate_or_bench_suite(split: str = "all") -> Dict[str, Any]:
    """Evalúa la suite OR-Bench ('public', 'sealed' o 'all') y devuelve métricas agregadas y tabla detallada."""
    if split == "public":
        probs = get_public_problems()
    elif split == "sealed":
        probs = get_sealed_problems()
    else:
        probs = get_all_or_bench_problems()

    results = [evaluate_single_or_problem(p) for p in probs]
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    false_conf_count = sum(1 for r in results if r["false_confidence"])

    pub_res = [r for r in results if r["split"] == "public"]
    seal_res = [r for r in results if r["split"] == "sealed"]

    pub_rate = (100.0 * sum(1 for r in pub_res if r["passed"]) / len(pub_res)) if pub_res else 100.0
    seal_rate = (100.0 * sum(1 for r in seal_res if r["passed"]) / len(seal_res)) if seal_res else 100.0
    overall_rate = (100.0 * passed_count / total) if total else 100.0

    supported_res = [r for r in results if r["expected_status"] != "UNSUPPORTED"]
    fid_rate = (
        sum(1 for r in supported_res if r["fidelity_verified"]) / len(supported_res)
        if supported_res
        else 1.0
    )
    max_code_viol = max((float(r["constraint_violation_on_code"] or 0.0) for r in supported_res), default=0.0)

    return {
        "total_problems": total,
        "passed_problems": passed_count,
        "pass_rate": round(overall_rate / 100.0, 4),
        "pass_rate_pct": round(overall_rate, 2),
        "public_pass_rate": round(pub_rate / 100.0, 4),
        "public_pass_rate_pct": round(pub_rate, 2),
        "sealed_pass_rate": round(seal_rate / 100.0, 4),
        "sealed_pass_rate_pct": round(seal_rate, 2),
        "fidelity_verified_rate": round(fid_rate, 4),
        "max_code_constraint_violation": max_code_viol,
        "false_confidence_count": false_conf_count,
        "zero_false_confidence": (false_conf_count == 0),
        "results": results,
    }


def format_or_bench_report_table(summary: Dict[str, Any]) -> str:
    """Formatea una tabla Markdown legible con el resultado detallado de los problemas de OR-Bench."""
    lines = [
        "| ID | Split | Clase | Estrategia | Estado Esperado | Estado Motor | J Motor | J Oráculo | Gap Rel | Fidelidad | Viol Código | Pasa |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary.get("results", []):
        j_eng = f"{r['engine_objective']:.4f}" if isinstance(r.get("engine_objective"), (int, float)) else "-"
        j_ora = f"{r['oracle_objective']:.4f}" if isinstance(r.get("oracle_objective"), (int, float)) else "-"
        gap_s = f"{r['rel_optimality_gap']:.2e}" if isinstance(r.get("rel_optimality_gap"), (int, float)) else "-"
        fid_s = "OK" if r.get("fidelity_verified") else "NO"
        viol_s = f"{float(r.get('constraint_violation_on_code') or 0.0):.2e}"
        ok_s = "PASS" if r.get("passed") else "FAIL"
        lines.append(
            f"| {r['id']} | {r['split']} | {r.get('problem_class') or '-'} | {r.get('strategy_used') or '-'} | "
            f"{r['expected_status']} | {r['engine_status']} | {j_eng} | {j_ora} | {gap_s} | {fid_s} | {viol_s} | {ok_s} |"
        )
    return "\n".join(lines)


__all__ = [
    "ORACLE_REGISTRY",
    "PUBLIC_PROBLEMS",
    "SEALED_PROBLEMS",
    "get_public_problems",
    "get_sealed_problems",
    "get_all_or_bench_problems",
    "evaluate_single_or_problem",
    "evaluate_or_bench_suite",
    "format_or_bench_report_table",
]
