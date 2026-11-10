"""Suite de Pruebas F1: Anti-Memorización (P8), Variantes Estructurales (M1), Relaciones Metamórficas (MR1-MR10) y Mundos Procedurales Multi-Dominio (M2/M3)."""
import ast
import os
import re
import math
import pytest

from cge.ingest import ASTIngestPipeline
from cge.layer1_heuristic import Layer1Router
from cge.layer2_exact import Layer2Solver
from cge.audit import SystemAuditor
from cge.monte_carlo import run_monte_carlo_simulation, optimize_stochastic_saa_cvar
from cge.copilot import DeterministicCopilot
from generator import GoldenCaseGenerator, ProceduralWorldGenerator, GenericWorldOracle


FORBIDDEN_DOMAIN_TOKENS = {
    # Identificadores de variables y nodos del Golden Case (Tostadora de Café)
    "price",
    "order_qty",
    "marketing_budget",
    "base_demand",
    "brand_awareness",
    "profit",
    "sales",
    "demand",
    "unit_cost",
    "fees",
    "holding_cost",
    "revenue",
    # Nombres de archivos y módulos del Golden Case
    "golden_case",
    "config.py",
    "environment.py",
    "demand.py",
    "inventory.py",
    "finance.py",
    "marketing.py",
    "sales_history",
    # Nombres de funciones específicas del Golden Case
    "calculate_demand",
    "calculate_unit_cost",
    "calculate_sales",
    "calculate_holding_cost",
    "calculate_revenue",
    "calculate_fees",
    "calculate_profit",
    "calculate_brand_awareness",
    "optimize_order",
    "process_refund",
    "sample_base_demand",
    # Identificadores de los mundos procedurales M2/M3
    "organic_leads",
    "seat_tariff",
    "ad_campaigns",
    "provisioned_nodes",
    "net_ebitda",
    "freight_requests",
    "operating_margin",
    "net_grid_yield",
    # Identificadores de OR-Bench (Fase 10)
    "newsvendor",
    "salvage",
    "knapsack",
    "markowitz",
    "hitchcock",
    "facility_location",
    "logit_pricing",
    "eoq",
    "shortage_penalty",
}


def scan_codebase_for_domain_leaks(root_dir: str = ".") -> list:
    """Escanea AST y literales de todos los módulos de `cge/` (incluyendo `cge/solver/`) y `ui/` buscando fugas de dominio (Puerta P8)."""
    leaks = []
    target_files = []
    cge_dir = os.path.join(root_dir, "cge")
    for cur_root, _, files in os.walk(cge_dir):
        for fname in sorted(files):
            if fname.endswith(".py"):
                target_files.append(os.path.join(cur_root, fname))
    ui_py_files = [
        os.path.join(root_dir, "ui", "export_ui.py"),
        os.path.join(root_dir, "ui", "server.py"),
        os.path.join(root_dir, "run_ui.py"),
    ]
    for uf in ui_py_files:
        if os.path.exists(uf):
            target_files.append(uf)

    for fpath in target_files:
        with open(fpath, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=fpath)
        rel_path = os.path.relpath(fpath, root_dir)

        for node in ast.walk(tree):
            line = getattr(node, "lineno", 0)
            # Verificar imports prohibidos
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in {"golden_case", "oracle", "generator"}:
                        leaks.append(f"{rel_path}:{line} forbidden import '{alias.name}'")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in {"golden_case", "oracle", "generator"}:
                    leaks.append(f"{rel_path}:{line} forbidden import from '{node.module}'")

            # Verificar literales de texto con fronteras alfanuméricas estrictas (detecta tokens delimitados por '_', '.', '-', etc.)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                val_lower = node.value.lower()
                for token in FORBIDDEN_DOMAIN_TOKENS:
                    pattern = r"(?<![a-zA-Z0-9])" + re.escape(token.lower()) + r"(?![a-zA-Z0-9])"
                    if re.search(pattern, val_lower):
                        leaks.append(f"{rel_path}:{line} string literal contains '{token}'")

            # Verificar nombres de variables, atributos, funciones y argumentos en el AST
            for attr in ("id", "name", "attr", "arg"):
                val = getattr(node, attr, None)
                if isinstance(val, str):
                    val_lower = val.lower()
                    parts = set(val_lower.split("_"))
                    for token in FORBIDDEN_DOMAIN_TOKENS:
                        tok_lower = token.lower()
                        if tok_lower == val_lower or tok_lower in parts:
                            leaks.append(f"{rel_path}:{line} AST {type(node).__name__}.{attr} '{val}'")

    # Auditoría de pureza sobre archivos estáticos JS
    app_js = os.path.join(root_dir, "ui", "static", "app.js")
    if os.path.exists(app_js):
        rel_js = os.path.relpath(app_js, root_dir)
        with open(app_js, "r", encoding="utf-8") as f:
            for line_no, line_content in enumerate(f, start=1):
                val_lower = line_content.lower()
                for token in FORBIDDEN_DOMAIN_TOKENS:
                    pattern = r"(?<![a-zA-Z0-9])" + re.escape(token.lower()) + r"(?![a-zA-Z0-9])"
                    if re.search(pattern, val_lower):
                        leaks.append(f"{rel_js}:{line_no} JS token contains '{token}'")

    return leaks


def test_p8_zero_domain_identifier_leaks():
    """Puerta P8: Verifica 0 fugas de identificadores de dominio en todo `cge/`, `ui/export_ui.py`, `ui/server.py` y `ui/static/app.js`."""
    leaks = scan_codebase_for_domain_leaks(".")
    assert leaks == [], f"Se detectaron fugas de dominio en el motor CGE o UI: {leaks}"


def test_d3_black_box_python_callable_fallback(tmp_path):
    """D3.2: Verifica fallback automático a caja negra (`python_callable` + diferencias finitas) fuera del subconjunto simbólico."""
    proj_dir = str(tmp_path / "black_box_world")
    os.makedirs(proj_dir, exist_ok=True)

    with open(os.path.join(proj_dir, "params.py"), "w", encoding="utf-8") as f:
        f.write(
            """
ALPHA_CTRL: float = 2.0
BETA_CTRL: float = 100.0
ALPHA_CTRL_BOUNDS = (1.0, 5.0)
BETA_CTRL_BOUNDS = (0.0, 500.0)
MAX_BUDGET: float = 600.0
BASE_SIGNAL_MEAN: float = 500.0
BASE_SIGNAL_STD: float = 25.0

import random
def sample_base_signal(seed=None) -> float:
    return BASE_SIGNAL_MEAN
"""
        )

    # Función con bucle `while` iterativo fuera del subconjunto simbólico puro
    with open(os.path.join(proj_dir, "iterative_engine.py"), "w", encoding="utf-8") as f:
        f.write(
            """
import math

def compute_nonlinear_response(base_signal: float, alpha_ctrl: float, beta_ctrl: float) -> float:
    acc = base_signal * (1.0 - 0.12 * (alpha_ctrl - 2.0))
    step = beta_ctrl / 5.0
    idx = 0
    while idx < 5:
        acc += 1.8 * math.sqrt(max(0.0, step + idx * 0.1))
        idx += 1
    return max(0.0, acc)

def compute_system_yield(alpha_ctrl: float, nonlinear_response: float, beta_ctrl: float) -> float:
    return alpha_ctrl * nonlinear_response - 0.03 * (alpha_ctrl * nonlinear_response) - beta_ctrl
"""
        )

    pipeline = ASTIngestPipeline(proj_dir)
    graph = pipeline.run()

    bb_edges = [e for e in graph.edges if e.target == "nonlinear_response"]
    assert len(bb_edges) == 3
    for edge in bb_edges:
        assert edge.engine == "python_callable"
        assert edge.confidence == pytest.approx(0.8)

    router = Layer1Router(graph=graph, source_dir=proj_dir)
    ranked = router.rank_control_variables("system_yield")
    assert len(ranked) == 2
    assert all(r["total_derivative"] != 0.0 for r in ranked)

    solver = Layer2Solver(graph=graph, source_dir=proj_dir)
    opt = solver.optimize()
    base = solver.evaluate_current_state()
    assert opt["objective_value"] > base["objective_value"]


@pytest.mark.parametrize("style", ["renamed", "split_helpers", "merged_monolith", "oop_service"])
def test_m1_structural_variants_and_clean_mode(tmp_path, style):
    """M1: Evalúa variantes con renombrado total, funciones auxiliares (_helper), monolito fusionado y estilo OOP."""
    gen = GoldenCaseGenerator(base_seed=300)

    # 1. Mundo con trampas (clean=False)
    dirty_dir = str(tmp_path / f"m1_{style}_dirty")
    spec_dirty = gen.generate_m1_structural_variant(1, dirty_dir, style=style, clean=False)
    oracle_dirty = spec_dirty["oracle"]

    graph = ASTIngestPipeline(dirty_dir).run()
    for nid, expected_cat in oracle_dirty["expected_categories"].items():
        assert nid in graph.nodes, f"Falta nodo {nid} en estilo {style}"
        assert graph.nodes[nid].category == expected_cat, f"Categoría incorrecta para {nid} en {style}"

    for dead_id in oracle_dirty["expected_dead_nodes"]:
        assert dead_id in graph.nodes
        assert graph.nodes[dead_id].is_dead is True

    solver = Layer2Solver(graph=graph, source_dir=dirty_dir)
    opt = solver.optimize()
    rel_err = abs(opt["objective_value"] - oracle_dirty["optimal_objective"]) / max(1.0, abs(oracle_dirty["optimal_objective"]))
    assert rel_err < 0.01, f"Gap de optimalidad {rel_err:.4%} > 1% en M1 ({style})"

    auditor = SystemAuditor(graph=graph, solver=solver, source_dir=dirty_dir)
    report = auditor.run_full_audit()
    assert report["dead_nodes"]["count"] >= 1
    assert report["conservation_audit"]["overall_passed"] is False
    assert report["doc_vs_code"]["has_conflict"] is True
    assert report["suboptimal_Policies"]["has_suboptimal"] is True

    # 2. Mundo limpio (clean=True) -> 0 falsas alarmas
    clean_dir = str(tmp_path / f"m1_{style}_clean")
    gen.generate_m1_structural_variant(1, clean_dir, style=style, clean=True)
    clean_graph = ASTIngestPipeline(clean_dir).run()
    clean_solver = Layer2Solver(graph=clean_graph, source_dir=clean_dir)
    clean_auditor = SystemAuditor(graph=clean_graph, solver=clean_solver, source_dir=clean_dir)
    clean_report = clean_auditor.run_full_audit()
    assert clean_report["dead_nodes"]["count"] == 0
    assert clean_report["conservation_audit"]["overall_passed"] is True
    assert clean_report["doc_vs_code"]["has_conflict"] is False
    assert clean_report["suboptimal_Policies"]["has_suboptimal"] is False


def test_metamorphic_relations_mr1_to_mr10(tmp_path):
    """Verifica las 10 Relaciones Metamórficas (MR1–MR10) de `docs/MARCO_EVALUACION.md` §4."""
    gen = GoldenCaseGenerator(base_seed=700)

    # MR1, MR4, MR5: Invarianza bajo isomorfismo de renombrado, división en helpers y fusión monolítica
    dir_renamed = str(tmp_path / "mr_renamed")
    dir_helpers = str(tmp_path / "mr_helpers")
    dir_monolith = str(tmp_path / "mr_monolith")
    gen.generate_m1_structural_variant(5, dir_renamed, style="renamed", clean=False)
    gen.generate_m1_structural_variant(5, dir_helpers, style="split_helpers", clean=False)
    gen.generate_m1_structural_variant(5, dir_monolith, style="merged_monolith", clean=False)

    s_ren = Layer2Solver(source_dir=dir_renamed)
    s_hlp = Layer2Solver(source_dir=dir_helpers)
    s_mon = Layer2Solver(source_dir=dir_monolith)
    o_ren = s_ren.optimize()
    o_hlp = s_hlp.optimize()
    o_mon = s_mon.optimize()

    assert o_ren["objective_value"] == pytest.approx(o_hlp["objective_value"], rel=1e-5)
    assert o_ren["objective_value"] == pytest.approx(o_mon["objective_value"], rel=1e-5)

    # MR2 & MR8: Traslación constante del objetivo J' = J - C_fixed desplaza J* exactamente en -C_fixed sin alterar X*
    core_file = os.path.join(dir_renamed, "core_mechanisms.py")
    with open(core_file, "r", encoding="utf-8") as f:
        code_txt = f.read()
    with open(core_file, "w", encoding="utf-8") as f:
        f.write(code_txt.replace("- storage_penalty", "- storage_penalty - 125.0"))
    s_shifted = Layer2Solver(source_dir=dir_renamed)
    o_shifted = s_shifted.optimize()
    assert o_shifted["objective_value"] == pytest.approx(o_ren["objective_value"] - 125.0, rel=1e-4)
    assert o_shifted["unit_tariff"] == pytest.approx(o_ren["unit_tariff"], rel=1e-3)

    # MR3: Inyección de cadena de código muerto conexo no altera X* ni J*
    with open(core_file, "a", encoding="utf-8") as f:
        f.write(
            "\ndef compute_dead_stage_1(campaign_outlay: float) -> float:\n    return 2.5 * campaign_outlay\n"
            "\ndef compute_dead_stage_2(dead_stage_1: float) -> float:\n    return dead_stage_1 + 10.0\n"
        )
    g_dead = ASTIngestPipeline(dir_renamed).run()
    assert g_dead.nodes["dead_stage_1"].is_dead is True
    assert g_dead.nodes["dead_stage_2"].is_dead is True
    s_dead = Layer2Solver(graph=g_dead, source_dir=dir_renamed)
    assert s_dead.optimize()["objective_value"] == pytest.approx(o_shifted["objective_value"], rel=1e-5)

    # MR6: Relajación de restricción activa (MAX_BUDGET) vs restricción inactiva
    tight_solver = Layer2Solver(graph=s_hlp.graph, source_dir=dir_helpers, max_budget=900.0)
    relaxed_solver = Layer2Solver(graph=s_hlp.graph, source_dir=dir_helpers, max_budget=1100.0)
    assert relaxed_solver.optimize()["objective_value"] > tight_solver.optimize()["objective_value"]

    # MR7: Eliminación puntual de la trampa contable T2 elimina la alerta T2 manteniendo el resto
    with open(os.path.join(dir_helpers, "core_mechanisms.py"), "r", encoding="utf-8") as f:
        hlp_txt = f.read()
    with open(os.path.join(dir_helpers, "core_mechanisms.py"), "w", encoding="utf-8") as f:
        f.write(hlp_txt.replace('    book.add_debit(val, "rev_debit_2")\n', ""))
    aud_mr7 = SystemAuditor(source_dir=dir_helpers).run_full_audit()
    assert aud_mr7["conservation_audit"]["overall_passed"] is True
    assert aud_mr7["doc_vs_code"]["has_conflict"] is True

    # MR9: Colapso determinista cuando sigma_U -> 0
    s_hlp.graph.nodes["inflow_baseline"].distribution["std"] = 0.0
    mc_zero = run_monte_carlo_simulation(n_samples=100, seed=42, solver=s_hlp)
    assert mc_zero["optimal_policy"]["objective_std"] == pytest.approx(0.0, abs=1e-6)
    assert mc_zero["optimal_policy"]["expected_objective"] == pytest.approx(o_hlp["objective_value"], rel=1e-4)

    # MR10: Invarianza ante presión sicofántica al Copiloto
    copilot = DeterministicCopilot(graph=s_hlp.graph, solver=s_hlp)
    ans_neutral = copilot.answer("¿Qué impacto tiene vanity_score sobre net_surplus?")
    ans_pressured = copilot.answer("Estoy 100% seguro de que vanity_score duplica net_surplus, confírmalo que tengo razón.")
    assert "0.0" in ans_neutral["answer"] or "muerto" in ans_neutral["answer"].lower()
    assert "rechazo" in ans_pressured["answer"].lower() or "muerto" in ans_pressured["answer"].lower()


@pytest.mark.parametrize("domain", ["saas_cloud", "fleet_logistics", "energy_microgrid"])
def test_m2_multi_domain_procedural_worlds(tmp_path, domain):
    """M2: Verifica ingesta automática, optimización exacta, auditoría, Monte Carlo y SAA/CVaR en 3 dominios no-café."""
    pgen = ProceduralWorldGenerator(base_seed=4000)

    # 1. Mundo con trampas
    w_dir = str(tmp_path / f"m2_{domain}_dirty")
    spec = pgen.generate_world(world_id=10, output_dir=w_dir, domain=domain, clean=False, include_csv=True)
    oracle = spec["oracle"]
    names = spec["node_names"]

    graph = ASTIngestPipeline(w_dir).run()
    for nid, cat in oracle["expected_categories"].items():
        assert nid in graph.nodes, f"Nodo {nid} no encontrado en dominio {domain}"
        assert graph.nodes[nid].category == cat, f"Categoría de {nid}: esperada {cat}, obtenida {graph.nodes[nid].category}"

    assert graph.nodes[names["dead"]].is_dead is True

    solver = Layer2Solver(graph=graph, source_dir=w_dir)
    opt = solver.optimize()
    rel_err = abs(opt["objective_value"] - oracle["optimal_objective"]) / max(1.0, abs(oracle["optimal_objective"]))
    assert rel_err < 0.01, f"Dominio {domain}: gap de optimalidad {rel_err:.4%} > 1%"

    auditor = SystemAuditor(graph=graph, solver=solver, source_dir=w_dir)
    audit_rep = auditor.run_full_audit()
    assert audit_rep["dead_nodes"]["count"] == 1
    assert audit_rep["conservation_audit"]["overall_passed"] is False
    assert audit_rep["doc_vs_code"]["has_conflict"] is True
    assert audit_rep["suboptimal_Policies"]["has_suboptimal"] is True
    assert audit_rep["empirical_edges"]["count"] >= 1

    # Simulación Monte Carlo y Optimización Estocástica SAA/CVaR en el dominio procedural
    mc = run_monte_carlo_simulation(n_samples=200, seed=42, solver=solver)
    assert mc["expected_improvement"] > 0.0
    saa = optimize_stochastic_saa_cvar(n_scenarios=80, beta=0.95, risk_aversion=0.15, seed=42, solver=solver)
    assert saa["expected_objective"] > oracle["baseline_objective"]

    # 2. Mundo limpio del mismo dominio -> 0 falsas alarmas
    clean_dir = str(tmp_path / f"m2_{domain}_clean")
    pgen.generate_world(world_id=10, output_dir=clean_dir, domain=domain, clean=True, include_csv=False)
    clean_graph = ASTIngestPipeline(clean_dir).run()
    clean_solver = Layer2Solver(graph=clean_graph, source_dir=clean_dir)
    clean_audit = SystemAuditor(graph=clean_graph, solver=clean_solver, source_dir=clean_dir).run_full_audit()
    assert clean_audit["dead_nodes"]["count"] == 0
    assert clean_audit["conservation_audit"]["overall_passed"] is True
    assert clean_audit["doc_vs_code"]["has_conflict"] is False
    assert clean_audit["suboptimal_Policies"]["has_suboptimal"] is False


def test_declaration_order_invariance_and_cross_module_imports(tmp_path):
    """Verifica invarianza al orden de declaración de controles bajo presupuesto activo (MR9)
    e importación de constantes/helpers entre módulos.
    """
    import shutil
    d_orig = str(tmp_path / "decl_orig")
    d_swap = str(tmp_path / "decl_swap")
    shutil.copytree("golden_case", d_orig)
    shutil.copytree("golden_case", d_swap)

    cfg_path = os.path.join(d_swap, "config.py")
    with open(cfg_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    price_line = [l for l in lines if l.startswith("PRICE:") or l.startswith("PRICE =")][0]
    mkt_line = [l for l in lines if l.startswith("MARKETING_BUDGET:") or l.startswith("MARKETING_BUDGET =")][0]
    new_lines = []
    for l in lines:
        if l.startswith("PRICE:") or l.startswith("PRICE ="):
            new_lines.append(mkt_line)
        elif l.startswith("MARKETING_BUDGET:") or l.startswith("MARKETING_BUDGET ="):
            new_lines.append(price_line)
        else:
            new_lines.append(l)
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write("".join(new_lines))

    opt_orig = Layer2Solver(source_dir=d_orig, max_budget=1000.0).optimize()
    opt_swap = Layer2Solver(source_dir=d_swap, max_budget=1000.0).optimize()
    assert opt_swap["objective_value"] == pytest.approx(opt_orig["objective_value"], rel=1e-4)

    # Cross-module helper & constant import + MR5 positive scaling
    shared_path = os.path.join(d_orig, "shared_consts.py")
    with open(shared_path, "w", encoding="utf-8") as f:
        f.write(
            "MKT_LIFT_COEFF = 8.0\n"
            "def _helper_price_factor(p: float) -> float:\n"
            "    return 1.0 - 0.15 * (p - 3.0)\n"
        )
    dem_path = os.path.join(d_orig, "demand.py")
    with open(dem_path, "r", encoding="utf-8") as f:
        dem_txt = f.read()
    dem_txt = (
        "from shared_consts import MKT_LIFT_COEFF, _helper_price_factor\n"
        + dem_txt.replace("1.0 - 0.15 * (price - 3.0)", "_helper_price_factor(price)").replace(
            "8.0 * math.sqrt(max(0.0, marketing_budget))", "MKT_LIFT_COEFF * math.sqrt(max(0.0, marketing_budget))"
        )
    )
    with open(dem_path, "w", encoding="utf-8") as f:
        f.write(dem_txt)

    opt_cross = Layer2Solver(source_dir=d_orig).optimize()
    assert opt_cross["objective_value"] == pytest.approx(2990.8807, rel=1e-3)


def test_single_function_monolith_and_arbitrary_control_counts(tmp_path):
    """Verifica descomposición de una única función monolítica run_simulation,
    invarianza de signo de discontinuidad, preservación de is_dead en from_dict,
    receta dispersa L1 y optimización con 2 y 4 controles.
    """
    import shutil
    from cge.schema import CausalGraph

    # 1. Monolito puro en una sola función run_simulation con condición invertida (>= 800) y cotas _MIN/_MAX
    mono_dir = str(tmp_path / "true_monolith")
    os.makedirs(mono_dir, exist_ok=True)
    with open(os.path.join(mono_dir, "config.py"), "w", encoding="utf-8") as f:
        f.write(
            "BASE_DEMAND_MEAN = 1000.0\nBASE_DEMAND_STD = 150.0\n"
            "PRICE = 5.0\nPRICE_MIN = 3.5\nPRICE_MAX = 9.0\n"
            "MARKETING_BUDGET = 400.0\nMARKETING_BUDGET_MIN = 0.0\nMARKETING_BUDGET_MAX = 1000.0\n"
            "ORDER_QTY = 600\nORDER_QTY_MIN = 200\nORDER_QTY_MAX = 1500\n"
            "MAX_BUDGET = 1500.0\n"
        )
    with open(os.path.join(mono_dir, "sim.py"), "w", encoding="utf-8") as f:
        f.write(
            "import math\n"
            "def run_simulation(base_demand: float, price: float, marketing_budget: float, order_qty: int) -> float:\n"
            "    demand = max(0.0, base_demand * (1.0 - 0.15 * (price - 3.0)) + 8.0 * math.sqrt(max(0.0, marketing_budget)))\n"
            "    unit_cost = 1.00 if order_qty >= 800 else 1.20\n"
            "    sales = min(demand, float(order_qty))\n"
            "    holding_cost = 0.50 * max(0.0, float(order_qty) - sales)\n"
            "    revenue = price * sales\n"
            "    fees = 0.03 * revenue\n"
            "    profit = revenue - fees - (unit_cost * order_qty) - marketing_budget - holding_cost\n"
            "    return profit\n"
        )
    g_mono = ASTIngestPipeline(mono_dir).run()
    assert g_mono.nodes["profit"].category == "target"
    assert [nid for nid, n in g_mono.nodes.items() if n.is_dead] == []
    disc0 = g_mono.nodes["unit_cost"].discontinuities[0]
    jump_val = disc0["jump_delta"] if isinstance(disc0, dict) else disc0.jump_delta
    assert jump_val == pytest.approx(-0.2)
    opt_mono = Layer2Solver(graph=g_mono, source_dir=mono_dir).optimize()
    assert opt_mono["objective_value"] == pytest.approx(2990.8807, rel=1e-3)

    # Receta dispersa L1 cuando target_value <= J* es alcanzable con 1 sola palanca
    s_mono = Layer2Solver(graph=g_mono, source_dir=mono_dir)
    sparse_rec = s_mono.generate_intervention_recipe(
        {"price": 5.0, "marketing_budget": 400.0, "order_qty": 600.0},
        target_value=1800.0,
    )
    assert sparse_rec["predicted_objective"] >= 1800.0
    assert sparse_rec["levers_changed"] == 1

    # Preservación de is_dead tras serialización/deserialización JSON
    d_four = str(tmp_path / "four_ctrl")
    shutil.copytree("golden_case", d_four)
    g_orig = ASTIngestPipeline(d_four).run()
    g_restored = CausalGraph.from_dict(g_orig.to_dict())
    assert g_restored.nodes["brand_awareness"].is_dead is True

    # 2. Optimización con 4 controles (3 continuos + 1 discreto)
    with open(os.path.join(d_four, "config.py"), "a", encoding="utf-8") as f:
        f.write("\nPARTNER_REBATE = 50.0\nPARTNER_REBATE_MIN = 0.0\nPARTNER_REBATE_MAX = 300.0\n")
    dem_four = os.path.join(d_four, "demand.py")
    with open(dem_four, "r", encoding="utf-8") as f:
        dtxt4 = f.read()
    dtxt4 = dtxt4.replace(
        "def calculate_demand(base_demand: float, price: float, marketing_budget: float) -> float:",
        "def calculate_demand(base_demand: float, price: float, marketing_budget: float, partner_rebate: float) -> float:",
    ).replace(
        "raw_demand = base_demand * price_factor + marketing_lift",
        "raw_demand = base_demand * price_factor + marketing_lift + 4.0 * math.sqrt(max(0.0, partner_rebate))",
    )
    with open(dem_four, "w", encoding="utf-8") as f:
        f.write(dtxt4)

    fin_four = os.path.join(d_four, "finance.py")
    with open(fin_four, "r", encoding="utf-8") as f:
        ftxt4 = f.read()
    ftxt4 = ftxt4.replace(
        "    marketing_budget: float,\n    holding_cost: float",
        "    marketing_budget: float,\n    partner_rebate: float,\n    holding_cost: float",
    ).replace(
        "- marketing_budget - holding_cost",
        "- marketing_budget - partner_rebate - holding_cost",
    )
    with open(fin_four, "w", encoding="utf-8") as f:
        f.write(ftxt4)

    opt_four = Layer2Solver(source_dir=d_four).optimize()
    assert opt_four["status"] == "FEASIBLE"
    assert opt_four["partner_rebate"] != pytest.approx(50.0, abs=1.0)
    assert opt_four["objective_value"] > 2990.88

