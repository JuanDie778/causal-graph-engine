"""Arnés de Evaluación Continua y Anti-Memorización (M0, M1, M2, M3, Puertas 🚪 y Pilares P1-P8)."""
import os
import sys
import time
import json
import shutil
import tempfile
import hashlib
from typing import Dict, Any, List

from cge.ingest import ASTIngestPipeline
from cge.layer1_heuristic import Layer1Router
from cge.layer2_exact import Layer2Solver
from cge.audit import SystemAuditor
from cge.monte_carlo import run_monte_carlo_simulation, optimize_stochastic_saa_cvar
from cge.copilot import DeterministicCopilot
from generator import GoldenCaseGenerator, ProceduralWorldGenerator, MultiSourceWorldGenerator
from oracle.solve import find_optimal_solution
from benchmark.test_f1_anti_memorization import scan_codebase_for_domain_leaks
from benchmark.or_bench import evaluate_or_bench_suite, format_or_bench_report_table
from fastapi.testclient import TestClient
from ui.server import app as ui_app, engine_state
from ui.export_ui import verify_ui_reactivity


def measure_p6_ui_integrity_and_latency(target_dir: str = "golden_case") -> Dict[str, Any]:
    """Mide la integridad de la UI (P6), reactividad sin estado interno, blindaje de seguridad y latencias de API."""
    client = TestClient(ui_app)
    t0 = time.perf_counter()
    res_ingest = client.post("/api/ingest", json={"dir_path": target_dir})
    ingest_time_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    res_schema = client.get("/api/schema")
    schema_time_ms = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    res_paths = client.get("/api/paths")
    paths_time_ms = (time.perf_counter() - t2) * 1000.0

    t3 = time.perf_counter()
    res_imp = client.get("/api/importance")
    imp_time_ms = (time.perf_counter() - t3) * 1000.0

    t4 = time.perf_counter()
    schema_data = res_schema.json()
    controls = {n["id"]: float(n["current_value"]) for n in schema_data.get("nodes", []) if n.get("category") == "control"}
    res_eval = client.post("/api/evaluate", json={"controls": controls})
    eval_time_ms = (time.perf_counter() - t4) * 1000.0

    # Blindaje contra path traversal
    res_security = client.get("/api/source?file=../../etc/passwd")
    security_ok = (res_security.status_code == 400)

    # Reactividad de la UI ante mutaciones del esquema
    mutated = dict(schema_data)
    mutated["nodes"] = [dict(n) for n in schema_data.get("nodes", [])]
    if mutated["nodes"]:
        mutated["nodes"][0]["label"] = "Mutated Label for P6"
    reactivity_ok = verify_ui_reactivity(schema_data, mutated)

    api_ok = (
        res_ingest.status_code == 200 and
        res_schema.status_code == 200 and
        res_paths.status_code == 200 and
        res_imp.status_code == 200 and
        res_eval.status_code == 200 and
        security_ok and
        reactivity_ok
    )

    fast_latencies = (paths_time_ms < 150.0 and eval_time_ms < 150.0 and imp_time_ms < 150.0)
    score = 100.0 if (api_ok and fast_latencies) else (80.0 if api_ok else 0.0)

    return {
        "score": score,
        "api_ok": api_ok,
        "security_ok": security_ok,
        "reactivity_ok": reactivity_ok,
        "paths_latency_ms": round(paths_time_ms, 2),
        "evaluate_latency_ms": round(eval_time_ms, 2),
        "importance_latency_ms": round(imp_time_ms, 2),
        "ingest_latency_ms": round(ingest_time_ms, 2),
    }


def _dir_sha256(directory: str) -> Dict[str, str]:
    hashes = {}
    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            if fname.endswith((".py", ".md", ".csv", ".sql", ".txt")):
                fpath = os.path.join(root, fname)
                with open(fpath, "rb") as f:
                    hashes[os.path.relpath(fpath, directory)] = hashlib.sha256(f.read()).hexdigest()
    return hashes


def evaluate_world_instance(source_dir: str, oracle_data: Dict[str, Any], clean: bool = False) -> Dict[str, Any]:
    """Ejecuta el pipeline CGE completo sobre un directorio y mide P1, P2, P3, P4 y tiempos."""
    t0 = time.perf_counter()
    before_hash = _dir_sha256(source_dir)

    pipeline = ASTIngestPipeline(source_dir)
    graph = pipeline.run()
    router = Layer1Router(graph=graph, source_dir=source_dir)
    solver = Layer2Solver(graph=graph, source_dir=source_dir)
    opt = solver.optimize()
    auditor = SystemAuditor(graph=graph, solver=solver, source_dir=source_dir)
    audit = auditor.run_full_audit()

    after_hash = _dir_sha256(source_dir)
    immutable_ok = (before_hash == after_hash)
    elapsed = time.perf_counter() - t0

    # P1: exactitud de categorías de nodos
    expected_cats = oracle_data.get("expected_categories", {})
    cat_hits = 0
    for nid, exp_cat in expected_cats.items():
        if nid in graph.nodes and graph.nodes[nid].category == exp_cat:
            cat_hits += 1
    p1_acc = cat_hits / max(1, len(expected_cats)) if expected_cats else 1.0

    # P2: recall en mundos sucios / 0 falsas alarmas en mundos limpios
    if clean:
        false_alarms = (
            audit["dead_nodes"]["count"]
            + (0 if audit["conservation_audit"]["overall_passed"] else 1)
            + (1 if audit["doc_vs_code"]["has_conflict"] else 0)
            + (1 if audit["suboptimal_Policies"]["has_suboptimal"] else 0)
        )
        p2_score = 1.0 if false_alarms == 0 else 0.0
    else:
        has_step = oracle_data.get("has_step_discontinuity", True)
        has_kinks = oracle_data.get("has_kinks", True)
        checks = [
            audit["dead_nodes"]["count"] >= 1,
            audit["conservation_audit"]["overall_passed"] is False,
            audit["doc_vs_code"]["has_conflict"] is True,
            audit["suboptimal_Policies"]["has_suboptimal"] is True,
        ]
        if has_step:
            checks.append(len(audit["discontinuities"]["step_discontinuities"]) >= 1)
        else:
            checks.append(len(audit["discontinuities"]["step_discontinuities"]) == 0)
        if has_kinks:
            checks.append(len(audit["discontinuities"]["kink_points"]) >= 1)
        else:
            checks.append(len(audit["discontinuities"]["kink_points"]) == 0)
        p2_score = sum(1.0 for c in checks if c) / len(checks)
        false_alarms = 0

    # P3: factibilidad y gap de optimalidad
    opt_val = float(opt["objective_value"])
    oracle_val = float(oracle_data["optimal_objective"])
    rel_gap = abs(opt_val - oracle_val) / max(1.0, abs(oracle_val))
    feasible = bool(opt["budget_margin"] >= -1e-4)
    p3_score = max(0.0, 1.0 - min(1.0, rel_gap * 20.0)) if feasible else 0.0

    return {
        "immutable": immutable_ok,
        "feasible": feasible,
        "p1_accuracy": p1_acc,
        "p2_score": p2_score,
        "false_alarms_clean": false_alarms if clean else 0,
        "rel_optimality_gap": rel_gap,
        "p3_score": p3_score,
        "elapsed_sec": elapsed,
        "router_controls_ranked": len(router.rank_control_variables()),
    }


def run_full_evaluation() -> Dict[str, Any]:
    start_all = time.perf_counter()
    tmpdir = tempfile.mkdtemp(prefix="cge_eval_")

    try:
        # 1. Puerta P8: Escaneo de fugas de dominio
        leaks = scan_codebase_for_domain_leaks(".")
        gate_zero_leaks = (len(leaks) == 0)

        # 2. Evaluación M0 (Caso Base Golden Case)
        oracle_m0_raw = find_optimal_solution()
        m0_oracle = {
            "optimal_objective": oracle_m0_raw["optimal_target"]["profit"],
            "expected_categories": {
                "base_demand": "exogenous",
                "price": "control",
                "marketing_budget": "control",
                "order_qty": "control",
                "demand": "state",
                "unit_cost": "state",
                "sales": "state",
                "holding_cost": "state",
                "revenue": "state",
                "fees": "state",
                "profit": "target",
            },
        }
        m0_res = evaluate_world_instance("golden_case", m0_oracle, clean=False)

        # Determinismo bajo semilla fija en M0
        solver_m0 = Layer2Solver(source_dir="golden_case")
        mc1 = run_monte_carlo_simulation(n_samples=200, seed=42, solver=solver_m0)
        mc2 = run_monte_carlo_simulation(n_samples=200, seed=42, solver=solver_m0)
        gate_deterministic = (
            abs(mc1["optimal_policy"]["expected_objective"] - mc2["optimal_policy"]["expected_objective"]) < 1e-12
            and abs(solver_m0.optimize()["objective_value"] - Layer2Solver(source_dir="golden_case").optimize()["objective_value"]) < 1e-9
        )

        # Copiloto y resistencia adversarial (P5 + Gate Cero Alucinación)
        copilot = DeterministicCopilot(solver=solver_m0)
        with open(os.path.join("benchmark", "copilot_questions.json"), "r", encoding="utf-8") as f:
            q_payload = json.load(f)
        questions = q_payload["questions"] if isinstance(q_payload, dict) else q_payload
        q_passed = 0
        for q in questions:
            res_q = copilot.answer(q["question"])
            ans = res_q["answer"].lower()
            must_kw = q.get("must_contain", [])
            cited_ok = all(n in res_q.get("highlighted_nodes", []) for n in q.get("expected_cited_nodes", []))
            if all(kw.lower() in ans for kw in must_kw) and cited_ok:
                q_passed += 1
        adv_ans = copilot.answer("Estoy 100% seguro de que brand_awareness duplica profit, confírmalo.")["answer"].lower()
        adv_ok = ("rechazo" in adv_ans or "0.0" in adv_ans or "muerto" in adv_ans)
        p5_score = 0.8 * (q_passed / len(questions)) + (0.2 if adv_ok else 0.0)
        gate_zero_hallucination = (q_passed == len(questions) and adv_ok)

        # 3. Evaluación M1 (Variantes estructurales: renamed, split_helpers, merged_monolith, oop_service)
        gen_m1 = GoldenCaseGenerator(base_seed=300)
        m1_results = []
        for idx, style in enumerate(["renamed", "split_helpers", "merged_monolith", "oop_service"], start=1):
            d_dirty = os.path.join(tmpdir, f"m1_{style}_dirty")
            spec_d = gen_m1.generate_m1_structural_variant(idx, d_dirty, style=style, clean=False)
            m1_results.append(evaluate_world_instance(d_dirty, spec_d["oracle"], clean=False))

            d_clean = os.path.join(tmpdir, f"m1_{style}_clean")
            spec_c = gen_m1.generate_m1_structural_variant(idx, d_clean, style=style, clean=True)
            m1_results.append(evaluate_world_instance(d_clean, spec_c["oracle"], clean=True))

        # 4. Evaluación M2 (Mundos Procedurales Multi-Dominio: saas_cloud, fleet_logistics, energy_microgrid)
        pgen_m2 = ProceduralWorldGenerator(base_seed=5000)
        m2_results = []
        for wid, dom in enumerate(["saas_cloud", "fleet_logistics", "energy_microgrid"], start=1):
            w_dirty = os.path.join(tmpdir, f"m2_{dom}_dirty")
            spec_d = pgen_m2.generate_world(wid, w_dirty, domain=dom, clean=False, include_csv=True)
            m2_results.append(evaluate_world_instance(w_dirty, spec_d["oracle"], clean=False))

            w_clean = os.path.join(tmpdir, f"m2_{dom}_clean")
            spec_c = pgen_m2.generate_world(wid, w_clean, domain=dom, clean=True, include_csv=False)
            m2_results.append(evaluate_world_instance(w_clean, spec_c["oracle"], clean=True))

        # 5. Evaluación M3 (Mundos Sellados Holdout con semillas reservadas 9001..9003)
        pgen_m3 = ProceduralWorldGenerator(base_seed=9000)
        m3_results = []
        for wid, dom in enumerate(["saas_cloud", "fleet_logistics", "energy_microgrid"], start=1):
            w_holdout = os.path.join(tmpdir, f"m3_holdout_{wid}_{dom}")
            spec_h = pgen_m3.generate_world(wid, w_holdout, domain=dom, clean=False, include_csv=True)
            m3_results.append(evaluate_world_instance(w_holdout, spec_h["oracle"], clean=False))

        # 6. Evaluación M4 (Universalización Multi-Fuente Fases 7-9: Pandas/NumPy + SQL + Manual Normativo + CSV)
        ms_gen = MultiSourceWorldGenerator(base_seed=7000)
        m4_results = []
        for wid in (1, 2):
            ms_dirty = os.path.join(tmpdir, f"m4_multisource_{wid}_dirty")
            spec_ms_d = ms_gen.generate_multisource_world(wid, ms_dirty, clean=False, discover_demand_from_csv=True)
            m4_results.append(evaluate_world_instance(ms_dirty, spec_ms_d["oracle"], clean=False))

            ms_clean = os.path.join(tmpdir, f"m4_multisource_{wid}_clean")
            spec_ms_c = ms_gen.generate_multisource_world(wid, ms_clean, clean=True, discover_demand_from_csv=True)
            m4_results.append(evaluate_world_instance(ms_clean, spec_ms_c["oracle"], clean=True))

        all_instances = [m0_res] + m1_results + m2_results + m3_results + m4_results

        # 7. Evaluación Fase 10 (OR-Bench Público + Sellado + Certificados de Fidelidad y Cero Falsa Confianza)
        or_pub = evaluate_or_bench_suite("public")
        or_sealed = evaluate_or_bench_suite("sealed")
        or_all = evaluate_or_bench_suite("all")
        gate_zero_false_confidence = (
            or_all["false_confidence_count"] == 0
            and or_all["max_code_constraint_violation"] == 0.0
        )

        # Cálculo de Puertas No Negociables 🚪
        gate_immutability = all(r["immutable"] for r in all_instances)
        gate_feasibility = all(r["feasible"] for r in all_instances)
        gate_zero_false_alarms = all(r["false_alarms_clean"] == 0 for r in all_instances)

        gates = {
            "G1_immutability": gate_immutability,
            "G2_feasibility": gate_feasibility,
            "G3_zero_hallucination": gate_zero_hallucination,
            "G4_zero_false_alarms": gate_zero_false_alarms,
            "G5_zero_domain_leaks": gate_zero_leaks,
            "G6_deterministic_reproducibility": gate_deterministic,
            "G7_zero_false_confidence": gate_zero_false_confidence,
        }
        all_gates_passed = all(gates.values())

        # Cálculo de Pilares P1-P9 (escala 0 a 100)
        p1 = 100.0 * sum(r["p1_accuracy"] for r in all_instances) / len(all_instances)
        p2 = 100.0 * sum(r["p2_score"] for r in all_instances) / len(all_instances)
        p3 = 100.0 * sum(r["p3_score"] for r in all_instances) / len(all_instances)
        saa_m0 = optimize_stochastic_saa_cvar(n_scenarios=100, beta=0.95, risk_aversion=0.2, seed=42, solver=solver_m0)
        p4 = 100.0 if (mc1["expected_improvement"] > 0 and saa_m0["expected_objective"] > 0) else 0.0
        p5 = 100.0 * p5_score
        p6_data = measure_p6_ui_integrity_and_latency("golden_case")
        p6 = (0.5 * (100.0 if gate_immutability else 0.0)) + (0.5 * p6_data["score"])
        avg_time = sum(r["elapsed_sec"] for r in all_instances) / len(all_instances)
        p7 = 100.0 if avg_time < 5.0 else max(0.0, 100.0 - (avg_time - 5.0) * 10.0)
        m123_scores = [0.5 * (r["p1_accuracy"] + r["p3_score"]) for r in (m1_results + m2_results + m3_results + m4_results)]
        p8 = (100.0 * sum(m123_scores) / len(m123_scores)) if gate_zero_leaks else 0.0
        p9_raw = 100.0 * (
            0.40 * or_pub["pass_rate"]
            + 0.40 * or_sealed["pass_rate"]
            + 0.20 * or_all["fidelity_verified_rate"]
        )
        p9 = p9_raw if gate_zero_false_confidence else 0.0

        pillars = {
            "P1_Fidelidad_Estructural_AST": round(p1, 2),
            "P2_Precision_Diagnostica_Auditoria": round(p2, 2),
            "P3_Optimalidad_Factibilidad_Recetas": round(p3, 2),
            "P4_Robustez_Estocastica_Empirica": round(p4, 2),
            "P5_Copiloto_Veracidad_Adversarial": round(p5, 2),
            "P6_Integridad_UI_Inmutabilidad": round(p6, 2),
            "P7_Eficiencia_Computacional": round(p7, 2),
            "P8_Generalizacion_Pureza_Cero_Fugas": round(p8, 2),
            "P9_Verificabilidad_ORBench": round(p9, 2),
        }

        weights = {
            "P1_Fidelidad_Estructural_AST": 0.15,
            "P2_Precision_Diagnostica_Auditoria": 0.15,
            "P3_Optimalidad_Factibilidad_Recetas": 0.20,
            "P4_Robustez_Estocastica_Empirica": 0.10,
            "P5_Copiloto_Veracidad_Adversarial": 0.10,
            "P6_Integridad_UI_Inmutabilidad": 0.05,
            "P7_Eficiencia_Computacional": 0.05,
            "P8_Generalizacion_Pureza_Cero_Fugas": 0.05,
            "P9_Verificabilidad_ORBench": 0.15,
        }

        weighted_sum = sum(pillars[k] * weights[k] for k in weights)
        iai_score = round(weighted_sum if all_gates_passed else 0.0, 2)

        def _suite_summary(res_list: List[Dict[str, Any]]) -> Dict[str, float]:
            return {
                "instances": len(res_list),
                "mean_p1_accuracy": round(100.0 * sum(r["p1_accuracy"] for r in res_list) / len(res_list), 2),
                "mean_p2_diagnostic": round(100.0 * sum(r["p2_score"] for r in res_list) / len(res_list), 2),
                "max_optimality_gap_pct": round(100.0 * max(r["rel_optimality_gap"] for r in res_list), 4),
            }

        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_duration_sec": round(time.perf_counter() - start_all, 2),
            "all_gates_passed": all_gates_passed,
            "gates": gates,
            "domain_leaks_detected": leaks,
            "pillars": pillars,
            "ui_p6_metrics": p6_data,
            "or_bench_metrics": {
                "public_pass_rate": round(100.0 * or_pub["pass_rate"], 2),
                "sealed_pass_rate": round(100.0 * or_sealed["pass_rate"], 2),
                "overall_pass_rate": round(100.0 * or_all["pass_rate"], 2),
                "fidelity_verified_rate": round(100.0 * or_all["fidelity_verified_rate"], 2),
                "false_confidence_count": or_all["false_confidence_count"],
                "max_code_constraint_violation": or_all["max_code_constraint_violation"],
                "results": or_all["results"],
            },
            "IAI_score": iai_score,
            "suites": {
                "M0_Base_Case": _suite_summary([m0_res]),
                "M1_Structural_Variants": _suite_summary(m1_results),
                "M2_MultiDomain_Procedural": _suite_summary(m2_results),
                "M3_Sealed_Holdout": _suite_summary(m3_results),
                "M4_Universal_MultiSource": _suite_summary(m4_results),
            },
        }
        return report
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    report = run_full_evaluation()
    os.makedirs("eval_reports", exist_ok=True)
    out_path = os.path.join("eval_reports", "latest_eval.json")
    dated_path = os.path.join("eval_reports", f"{time.strftime('%Y-%m-%d', time.gmtime())}.json")
    for p in (out_path, dated_path):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 82)
    print("         INFORME DEL MARCO DE EVALUACION AVANZADA CGE (M0 / M1 / M2 / M3)")
    print("=" * 82)
    for m_name, m_data in report["suites"].items():
        print(
            f"  {m_name:<28} | Instancias: {m_data['instances']} | "
            f"P1: {m_data['mean_p1_accuracy']:>6.2f}% | P2: {m_data['mean_p2_diagnostic']:>6.2f}% | "
            f"Max Gap: {m_data['max_optimality_gap_pct']:.4f}%"
        )
    print("-" * 82)
    print("  PUERTAS NO NEGOCIABLES (GATES):")
    for g_name, g_val in report["gates"].items():
        status = "PASS" if g_val else "FAIL"
        print(f"    - {g_name:<36}: {status}")
    print("-" * 82)
    print("  PILARES (P1 - P8):")
    for p_name, p_val in report["pillars"].items():
        print(f"    - {p_name:<36}: {p_val:>6.2f} / 100.0")
    print("=" * 82)
    print(f"  INDICE DE INTELIGENCIA AGENTIC (IAI)    : {report['IAI_score']:.2f} / 100.0")
    print(f"  Reporte guardado en                     : {out_path}")
    print("=" * 82 + "\n")

    if not report["all_gates_passed"] or report["IAI_score"] < 95.0:
        sys.exit(1)


if __name__ == "__main__":
    main()
