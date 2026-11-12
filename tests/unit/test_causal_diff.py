"""Suite completa de pruebas unitarias y de integración Git para `cge/diff/` (Motor de Causal Diff)."""
from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
import pytest

from cge.diff import (
    CausalDiffEngine,
    CausalDiffError,
    CausalDiffReport,
    DiffVerdict,
    diff_git_revisions,
    diff_graphs,
    diff_sources,
    load_causal_graph,
)
from cge.diff.__main__ import main as diff_cli_main
from cge.ingest import compile_source_to_ir
from cge.schema import CausalGraph, Constraint, Edge, Node, SourceReference


# ==============================================================================
# Helpers de Construcción de Grafos y Repositorios Git Temporales
# ==============================================================================

def _make_clean_pricing_code(
    margin_mult: float = 1.0,
    budget_limit: float = 100.0,
    include_dead_node: bool = False,
    include_cancel_node: bool = False,
    disconnect_control: bool = False,
) -> str:
    dead_fn = ""
    if include_dead_node:
        dead_fn = """
def compute_vanity_impressions(ad_spend: float = 20.0) -> float:
    return 50.0 * ad_spend + 10.0
"""
    cancel_line = ""
    cancel_ret = ""
    if include_cancel_node:
        dead_fn += """
def compute_neutral_rebate(price: float = 5.0) -> float:
    return 3.0 * price
"""
        cancel_line = ", neutral_rebate: float = 15.0"
        cancel_ret = " + neutral_rebate - neutral_rebate"

    ad_term = "0.0" if disconnect_control else "4.0 * ad_spend"
    return f'''
SENSE = "max"
PRICE: float = 5.0
AD_SPEND: float = 20.0
PRICE_BOUNDS = (1.0, 12.0)
AD_SPEND_BOUNDS = (0.0, 80.0)
CONSTRAINTS = {{
    "c_budget": "5.0 * price + ad_spend <= {budget_limit}",
}}
{dead_fn}
def compute_revenue(price: float = 5.0, ad_spend: float = 20.0) -> float:
    return {margin_mult} * (20.0 * price - (price ** 2) + {ad_term})

def compute_net_profit(revenue: float = 75.0, ad_spend: float = 20.0{cancel_line}) -> float:
    return revenue - ad_spend{cancel_ret}
'''


def _git(cwd: Path, *args: str) -> None:
    res = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)
    assert res.returncode == 0, f"git {' '.join(args)} failed: {res.stderr}"


# ==============================================================================
# 1. Equivalencia Semántica e Inmunidad ante Refactorización Sintáctica
# ==============================================================================

class TestSemanticEquivalenceAndRefactoring:
    def test_identical_graphs_and_serialization_formats_pass(self, tmp_path: Path):
        base_code = _make_clean_pricing_code()
        g_base = compile_source_to_ir(base_code, system_id="pricing_v1")
        g_copy = CausalGraph.from_dict(copy.deepcopy(g_base.to_dict()))

        report = diff_graphs(g_base, g_copy)
        assert isinstance(report, CausalDiffReport)
        assert report.verdict == DiffVerdict.PASS.value
        assert report.has_regressions is False
        assert report.is_semantically_equivalent is True
        assert report.invariant_regressions == []
        assert report.dead_nodes["added_dead_nodes"] == []
        assert report.policy_optimality["is_regression"] is False
        assert report.policy_optimality["policy_changed"] is False

        # Serialización JSON y Markdown válidas + protocolo diccionario
        as_dict = json.loads(report.to_json())
        assert as_dict["verdict"] == "PASS"
        assert "verdict" in report
        assert report["verdict"] == "PASS"
        assert report.get("nonexistent_key", "fallback") == "fallback"
        md = report.to_markdown()
        assert "# Causal Diff Report" in md
        assert "`PASS`" in md

        # Carga desde archivo .json en disco
        json_path = tmp_path / "base_schema.json"
        g_base.save_to_file(str(json_path))
        rep_from_json = diff_sources(json_path, g_base)
        assert rep_from_json.verdict == "PASS"
        assert rep_from_json.is_semantically_equivalent is True

    def test_algebraic_refactoring_is_semantically_equivalent(self):
        code_v1 = '''
SENSE = "max"
X: float = 2.0
Y: float = 3.0
X_BOUNDS = (0.0, 10.0)
Y_BOUNDS = (0.0, 10.0)
CONSTRAINTS = {"c_cap": "x + 2.0 * y <= 12.0"}

def compute_profit(x: float = 2.0, y: float = 3.0) -> float:
    return 4.0 * x + 6.0 * y - 10.0
'''
        # Reescritura algebraicamente idéntica: términos conmutados, factorizados y desigualdad invertida
        code_v2 = '''
SENSE = "max"
X: float = 2.0
Y: float = 3.0
X_BOUNDS = (0.0, 10.0)
Y_BOUNDS = (0.0, 10.0)
CONSTRAINTS = {"c_cap": "12.0 >= 2.0 * y + x"}

def compute_profit(x: float = 2.0, y: float = 3.0) -> float:
    # Refactorización cosmética sin cambio causal
    return 2.0 * (3.0 * y + 2.0 * x) - 10.0
'''
        report = diff_sources(code_v1, code_v2, entrypoint="compute_profit")
        assert report.verdict == DiffVerdict.PASS.value
        assert report.has_regressions is False
        assert report.is_semantically_equivalent is True
        assert report.structural_diff["nodes_semantically_modified"] == []
        assert report.structural_diff["constraints_semantically_modified"] == []

    def test_trigonometric_refactoring_equivalence_and_transcendental_diff(self):
        sr = SourceReference(file="osc.py", entity="f", line_range=[1, 4])

        def _make_trig_graph(expr_y: str, cap: float) -> CausalGraph:
            g = CausalGraph(system_id="trig_sys")
            g.add_node(Node("theta", "Theta", "control", "continuous", sr, current_value=0.5, bounds={"min": 0.0, "max": 3.0}))
            g.add_node(Node("energy", "Energy", "target", "continuous", sr, symbolic_expr=expr_y))
            g.add_edge(Edge("theta", "energy", "exact_code"))
            g.add_constraint(Constraint("c_wave", "hard", f"sin(theta) <= {cap}"))
            return g

        # `sin(theta)**2` vs `1 - cos(theta)**2` son idénticas trigonométricamente -> PASS semántico
        g_sin = _make_trig_graph("sin(theta)**2 + tanh(theta)", cap=0.8)
        g_cos = _make_trig_graph("1 - cos(theta)**2 + tanh(theta)", cap=0.8)
        rep_eq = diff_graphs(g_sin, g_cos)
        assert rep_eq.verdict == DiffVerdict.PASS.value
        assert rep_eq.has_regressions is False
        assert rep_eq.structural_diff["nodes_semantically_modified"] == []

        # Relajar `sin(theta) <= 0.5` -> `sin(theta) <= 0.95` admite theta=pi/2 (~1.5708) que violaba la base -> FAIL
        g_tight = _make_trig_graph("sin(theta)**2", cap=0.5)
        g_loose = _make_trig_graph("sin(theta)**2", cap=0.95)
        rep_weak = diff_graphs(g_tight, g_loose)
        assert rep_weak.verdict == DiffVerdict.FAIL.value
        assert any(r["code"] == "HARD_CONSTRAINT_WEAKENED" for r in rep_weak.invariant_regressions)

        # Endurecer restricciones con constante simbólica `pi` (`theta <= pi / 3` -> `theta <= 1.0`)
        # y función `tan(theta)` en dominio libre de polos `[0, 1]` (`energy <= 1.5` -> `energy <= 1.2`)
        # NO debe tratar `pi` como variable libre ni cruzar polos fuera de las cotas declaradas.
        def _make_pi_tan_graph(angle_cap: str, tan_cap: float) -> CausalGraph:
            gp = CausalGraph(system_id="pi_tan_sys")
            gp.add_node(Node("theta", "Theta", "control", "continuous", sr, current_value=0.5, bounds={"min": 0.0, "max": 1.0}))
            gp.add_node(Node("energy", "Energy", "target", "continuous", sr, symbolic_expr="tan(theta)"))
            gp.add_edge(Edge("theta", "energy", "exact_code"))
            gp.add_constraint(Constraint("c_angle", "hard", f"theta <= {angle_cap}"))
            gp.add_constraint(Constraint("c_tan", "hard", f"energy <= {tan_cap}"))
            return gp

        rep_pi_tight = diff_graphs(
            _make_pi_tan_graph("pi / 3", 1.5),
            _make_pi_tan_graph("1.0", 1.2),
        )
        assert not any(r["code"] == "HARD_CONSTRAINT_WEAKENED" for r in rep_pi_tight.invariant_regressions)
        mod_rels = {m["constraint_id"]: m["relation"] for m in rep_pi_tight.structural_diff["constraints_modified"]}
        assert mod_rels == {"c_angle": "tightened", "c_tan": "tightened"}


# ==============================================================================
# 2. Regresiones de Invariantes (T2, Restricciones, Dominio, IIS, T3, Fidelidad, Ciclos)
# ==============================================================================

class TestInvariantRegressions:
    def test_conservation_invariant_t2_regression_and_restoration(self, tmp_path: Path):
        # Construir versión limpia de golden_case sin el bug de process_refund vs versión con bug
        clean_dir = tmp_path / "ledger_clean"
        buggy_dir = tmp_path / "ledger_buggy"
        clean_dir.mkdir()
        buggy_dir.mkdir()

        common_Ledger = '''
PRICE: float = 4.0
PRICE_BOUNDS = (1.0, 10.0)

class Ledger:
    def __init__(self):
        self.credits = 0.0
        self.debits = 0.0
    def record_credit(self, amount: float):
        self.credits += amount
    def record_debit(self, amount: float):
        self.debits += amount

def calculate_profit(price: float = 4.0) -> float:
    return 100.0 * price - 10.0 * (price ** 2)
'''
        balanced_op = '''
def process_refund(ledger: Ledger, amount: float) -> None:
    ledger.record_credit(-amount)
    ledger.record_debit(-amount)
'''
        unbalanced_op = '''
def process_refund(ledger: Ledger, amount: float) -> None:
    ledger.record_credit(-amount)
    ledger.record_debit(-amount)
    ledger.record_debit(-amount)
'''
        (clean_dir / "finance.py").write_text(common_Ledger + balanced_op, encoding="utf-8")
        (buggy_dir / "finance.py").write_text(common_Ledger + unbalanced_op, encoding="utf-8")

        rep_reg = diff_sources(clean_dir, buggy_dir)
        assert rep_reg.verdict == DiffVerdict.FAIL.value
        assert rep_reg.has_regressions is True
        codes = [r["code"] for r in rep_reg.invariant_regressions]
        assert "CONSERVATION_INVARIANT_BROKEN" in codes

        # Dirección inversa: arreglar el invariante contable pasa en limpio y registra invariants_fixed
        rep_fix = diff_sources(buggy_dir, clean_dir)
        assert rep_fix.verdict == DiffVerdict.PASS.value
        assert rep_fix.has_regressions is False
        fixed_codes = [f["code"] for f in rep_fix.invariants_fixed]
        assert "CONSERVATION_INVARIANT_RESTORED" in fixed_codes

    def test_hard_constraint_removed_downgraded_weakened_vs_tightened(self):
        base_code = _make_clean_pricing_code(budget_limit=80.0)
        g_base = compile_source_to_ir(base_code)

        # Caso A: Eliminación de restricción dura
        g_removed = CausalGraph.from_dict(copy.deepcopy(g_base.to_dict()))
        g_removed.constraints.pop("c_budget")
        rep_rem = diff_graphs(g_base, g_removed)
        assert rep_rem.verdict == "FAIL"
        assert any(r["code"] == "HARD_CONSTRAINT_REMOVED" for r in rep_rem.invariant_regressions)

        # Caso B: Degradación de 'hard' a 'soft'
        g_soft = CausalGraph.from_dict(copy.deepcopy(g_base.to_dict()))
        g_soft.constraints["c_budget"].type = "soft"
        rep_soft = diff_graphs(g_base, g_soft)
        assert rep_soft.verdict == "FAIL"
        assert any(r["code"] == "HARD_CONSTRAINT_DOWNGRADED" for r in rep_soft.invariant_regressions)

        # Caso C: Relajación / debilitamiento silencioso (80 -> 120 admite estados antes prohibidos)
        relaxed_code = _make_clean_pricing_code(budget_limit=120.0)
        rep_weak = diff_sources(base_code, relaxed_code)
        assert rep_weak.verdict == "FAIL"
        weak_items = [r for r in rep_weak.invariant_regressions if r["code"] == "HARD_CONSTRAINT_WEAKENED"]
        assert len(weak_items) == 1
        assert weak_items[0]["relation"] == "weakened"
        assert weak_items[0]["witness"] is not None

        # Caso D: Endurecimiento legítimo (120 -> 115, donde el óptimo 5*9 + 60 = 105 sigue intacto)
        tight_base = _make_clean_pricing_code(budget_limit=120.0)
        tight_head = _make_clean_pricing_code(budget_limit=115.0)
        rep_tight = diff_sources(tight_base, tight_head)
        assert not any(r["code"] == "HARD_CONSTRAINT_WEAKENED" for r in rep_tight.invariant_regressions)
        mod_c = rep_tight.structural_diff["constraints_modified"][0]
        assert mod_c["relation"] == "tightened"
        assert mod_c["is_weakened"] is False

    def test_nominal_violation_and_state_invariant_falsified_across_shocks(self):
        # Construir grafo donde un invariante de estado se cumple en el punto nominal pero falla ante shock exógeno U
        sr = SourceReference(file="sim.py", entity="fn", line_range=[1, 5])

        def _build_reserve_graph(buffer_val: float) -> CausalGraph:
            g = CausalGraph(system_id="reserve_sys")
            g.add_node(Node(
                id="shock_u", label="Shock", category="exogenous", data_type="continuous",
                source_reference=sr, current_value=10.0,
                bounds={"min": 0.0, "max": 50.0},
                distribution={"type": "normal", "mean": 10.0, "std": 10.0},
            ))
            g.add_node(Node(
                id="rate_x", label="Rate", category="control", data_type="continuous",
                source_reference=sr, current_value=5.0, bounds={"min": 1.0, "max": 10.0},
            ))
            g.add_node(Node(
                id="reserve", label="Reserve", category="state", data_type="continuous",
                source_reference=sr, symbolic_expr=f"{buffer_val} - shock_u",
            ))
            g.add_node(Node(
                id="utility", label="Utility", category="target", data_type="continuous",
                source_reference=sr, symbolic_expr="10.0 * rate_x + 0.01 * reserve",
            ))
            g.add_edge(Edge(source="shock_u", target="reserve", type="exact_code"))
            g.add_edge(Edge(source="reserve", target="utility", type="exact_code"))
            g.add_edge(Edge(source="rate_x", target="utility", type="exact_code"))
            g.add_constraint(Constraint(id="c_solvency", type="hard", expression="reserve >= 0"))
            return g

        # En base (buffer=60), para todo shock_u en [0, 50], reserve >= 10 >= 0 (PROVEN)
        # En head (buffer=25), en el punto nominal shock_u=10 reserve=15 >= 0, pero en shock_u=40 reserve=-15 < 0!
        g_safe = _build_reserve_graph(60.0)
        g_fragile = _build_reserve_graph(25.0)
        rep = diff_graphs(g_safe, g_fragile)
        assert rep.verdict == "FAIL"
        falsified = [r for r in rep.invariant_regressions if r["code"] == "STATE_INVARIANT_FALSIFIED"]
        assert len(falsified) == 1
        assert falsified[0]["constraint_id"] == "c_solvency"
        assert falsified[0]["counterexample"]["shock_u"] > 25.0

        # Violación directa en el estado nominal (buffer=5 -> nominal reserve = -5 < 0)
        g_nom_bad = _build_reserve_graph(5.0)
        rep_nom = diff_graphs(g_safe, g_nom_bad)
        assert rep_nom.verdict == "FAIL"
        assert any(r["code"] == "NOMINAL_INVARIANT_VIOLATION" for r in rep_nom.invariant_regressions)

    def test_feasible_space_collapse_reports_minimal_iis(self):
        base_code = '''
SENSE = "max"
X1: float = 5.0
X2: float = 5.0
X1_BOUNDS = (0.0, 20.0)
X2_BOUNDS = (0.0, 20.0)
CONSTRAINTS = {"c_upper": "x1 <= 8.0"}
def evaluate_obj(x1: float = 5.0, x2: float = 5.0) -> float:
    return 3.0 * x1 + 2.0 * x2
'''
        head_code = '''
SENSE = "max"
X1: float = 5.0
X2: float = 5.0
X1_BOUNDS = (0.0, 20.0)
X2_BOUNDS = (0.0, 20.0)
CONSTRAINTS = {
    "c_upper": "x1 <= 8.0",
    "c_contradict": "x1 >= 12.0",
}
def evaluate_obj(x1: float = 5.0, x2: float = 5.0) -> float:
    return 3.0 * x1 + 2.0 * x2
'''
        rep = diff_sources(base_code, head_code, entrypoint="evaluate_obj")
        assert rep.verdict == "FAIL"
        collapsed = [r for r in rep.invariant_regressions if r["code"] == "FEASIBLE_SPACE_COLLAPSED"]
        assert len(collapsed) == 1
        assert set(collapsed[0]["iis_constraints"]) == {"c_upper", "c_contradict"}

    def test_epistemic_fidelity_regression_and_doc_discrepancy_t3(self, tmp_path: Path):
        pure_code = '''
SENSE = "max"
X_VAL: float = 5.0
X_VAL_BOUNDS = (1.0, 20.0)
def compute_score(x_val: float = 5.0) -> float:
    return 4.0 * x_val
'''
        impure_code = '''
import random
SENSE = "max"
X_VAL: float = 5.0
X_VAL_BOUNDS = (1.0, 20.0)
def compute_score(x_val: float = 5.0) -> float:
    return 4.0 * x_val * random.random()
'''
        rep_fid = diff_sources(pure_code, impure_code, entrypoint="compute_score")
        assert rep_fid.verdict == "FAIL"
        assert any(r["code"] == "EPISTEMIC_FIDELITY_REGRESSION" for r in rep_fid.invariant_regressions)

        # Discrepancia T3 entre Markdown y código AST
        d_base = tmp_path / "doc_ok"
        d_head = tmp_path / "doc_bad"
        d_base.mkdir()
        d_head.mkdir()
        (d_base / "README.md").write_text("Platform fee rate is 2%.", encoding="utf-8")
        (d_head / "README.md").write_text("Platform fee rate is 2%.", encoding="utf-8")
        code_template = '''
REV: float = 100.0
REV_BOUNDS = (10.0, 500.0)
def calc_fee(rev: float = 100.0) -> float:
    return {rate} * rev
def calc_net(rev: float = 100.0, fee: float = 2.0) -> float:
    return rev - fee
'''
        (d_base / "fees.py").write_text(code_template.format(rate="0.02"), encoding="utf-8")
        (d_head / "fees.py").write_text(code_template.format(rate="0.05"), encoding="utf-8")
        rep_t3 = diff_sources(d_base, d_head)
        assert rep_t3.verdict == "FAIL"
        assert any(r["code"] == "DOC_CODE_DISCREPANCY_INTRODUCED" for r in rep_t3.invariant_regressions)

    def test_ill_posed_simultaneous_cycle_handled_gracefully(self):
        sr = SourceReference(file="cycle.py", entity="f", line_range=[1, 2])
        g_dag = CausalGraph(system_id="dag_ok")
        g_dag.add_node(Node("x", "x", "control", "continuous", sr, current_value=2.0, bounds={"min": 0.0, "max": 10.0}))
        g_dag.add_node(Node("a", "a", "state", "continuous", sr, symbolic_expr="x + 1"))
        g_dag.add_node(Node("b", "b", "target", "continuous", sr, symbolic_expr="2 * a"))
        g_dag.add_edge(Edge("x", "a", "exact_code"))
        g_dag.add_edge(Edge("a", "b", "exact_code"))

        # Introducir ciclo simultáneo contradictorio Lambda=0: a = b + 1, b = a + 1
        g_cyc = CausalGraph(system_id="dag_cyclic")
        g_cyc.add_node(Node("x", "x", "control", "continuous", sr, current_value=2.0, bounds={"min": 0.0, "max": 10.0}))
        g_cyc.add_node(Node("a", "a", "state", "continuous", sr, symbolic_expr="b + 1"))
        g_cyc.add_node(Node("b", "b", "target", "continuous", sr, symbolic_expr="a + 1"))
        g_cyc.add_edge(Edge("a", "b", "exact_code"))
        g_cyc.add_edge(Edge("b", "a", "exact_code"))

        rep = diff_graphs(g_dag, g_cyc)
        assert rep.verdict == "FAIL"
        cyc_regs = [r for r in rep.invariant_regressions if r["code"] == "ILL_POSED_CYCLE_INTRODUCED"]
        assert len(cyc_regs) == 1
        assert cyc_regs[0]["classification"] == "ILL_POSED_CYCLE"

    def test_nonlinear_and_logical_constraint_relaxation_vs_equivalence(self):
        nl_template = '''
SENSE = "max"
X: float = 1.0
Y: float = 1.0
X_BOUNDS = (-10.0, 10.0)
Y_BOUNDS = (-10.0, 10.0)
CONSTRAINTS = {{"c_circle": "{expr}"}}
def compute_obj(x: float = 1.0, y: float = 1.0) -> float:
    return x + y
'''
        # 1. Relajación no lineal cuadrática (radio^2: 25 -> 10000) -> FAIL con testigo concreto
        rep_nl_weak = diff_sources(
            nl_template.format(expr="x**2 + y**2 <= 25.0"),
            nl_template.format(expr="x**2 + y**2 <= 10000.0"),
            entrypoint="compute_obj",
        )
        assert rep_nl_weak.verdict == "FAIL"
        weak_nl = [r for r in rep_nl_weak.invariant_regressions if r["code"] == "HARD_CONSTRAINT_WEAKENED"]
        assert len(weak_nl) == 1
        assert weak_nl[0]["relation"] == "weakened"
        w = weak_nl[0]["witness"]
        assert w is not None and (w["x"] ** 2 + w["y"] ** 2) > 25.0

        # 2. Endurecimiento no lineal cuadrático (100 -> 25) -> relación 'tightened', is_weakened=False
        rep_nl_tight = diff_sources(
            nl_template.format(expr="x**2 + y**2 <= 100.0"),
            nl_template.format(expr="x**2 + y**2 <= 25.0"),
            entrypoint="compute_obj",
        )
        assert not any(r["code"] == "HARD_CONSTRAINT_WEAKENED" for r in rep_nl_tight.invariant_regressions)
        assert rep_nl_tight.structural_diff["constraints_modified"][0]["relation"] == "tightened"

        # 3. Equivalencia lógica proposicional por Ley de De Morgan -> PASS y semánticamente equivalente
        rep_demorgan = diff_sources(
            nl_template.format(expr="not (x > 5.0 and y > 5.0)"),
            nl_template.format(expr="x <= 5.0 or y <= 5.0"),
            entrypoint="compute_obj",
        )
        assert rep_demorgan.verdict == "PASS"
        assert rep_demorgan.is_semantically_equivalent is True
        assert rep_demorgan.structural_diff["constraints_modified"][0]["relation"] == "equivalent"

        # 4. Relajación de guardia implicacional (`amount >= 100 => kyc >= 1` -> `amount >= 10000 => kyc >= 1`)
        sr = SourceReference(file="kyc.py", entity="fn", line_range=[1, 3])

        def _kyc_graph(thresh: float) -> CausalGraph:
            g = CausalGraph(system_id="kyc_gate")
            g.add_node(Node("amount", "amount", "control", "continuous", sr, current_value=50.0, bounds={"min": 0.0, "max": 5000.0}))
            g.add_node(Node("kyc", "kyc", "control", "boolean", sr, current_value=1, bounds={"min": 0, "max": 1}))
            g.add_node(Node("rev", "rev", "target", "continuous", sr, symbolic_expr="0.02 * amount - 0.1 * kyc"))
            g.add_edge(Edge("amount", "rev", "exact_code"))
            g.add_edge(Edge("kyc", "rev", "exact_code"))
            g.add_constraint(Constraint("c_kyc_gate", "hard", f"amount >= {thresh} => kyc >= 1"))
            return g

        rep_imp_weak = diff_graphs(_kyc_graph(100.0), _kyc_graph(10000.0))
        assert rep_imp_weak.verdict == "FAIL"
        weak_imp = [r for r in rep_imp_weak.invariant_regressions if r["code"] == "HARD_CONSTRAINT_WEAKENED"]
        assert len(weak_imp) == 1
        assert weak_imp[0]["witness"]["amount"] >= 100.0
        assert weak_imp[0]["witness"]["kyc"] == 0.0

    def test_control_box_bounds_relaxation_and_removal_flagged(self):
        sr = SourceReference(file="bnds.py", entity="fn", line_range=[1, 2])

        def _make_ctrl_graph(bnds: dict | None) -> CausalGraph:
            g = CausalGraph(system_id="ctrl_bounds")
            g.add_node(Node("x", "x", "control", "continuous", sr, current_value=3.0, bounds=bnds))
            g.add_node(Node("obj", "obj", "target", "continuous", sr, symbolic_expr="100.0 - (x - 5.0)**2"))
            g.add_edge(Edge("x", "obj", "exact_code"))
            return g

        g_base = _make_ctrl_graph({"min": 0.0, "max": 10.0})
        g_widened = _make_ctrl_graph({"min": -500.0, "max": 500.0})
        g_removed = _make_ctrl_graph({"min": 0.0, "max": None})
        g_tightened = _make_ctrl_graph({"min": 1.0, "max": 9.0})

        rep_wide = diff_graphs(g_base, g_widened)
        assert rep_wide.verdict == "FAIL"
        assert any(r["code"] == "CONTROL_BOUNDS_WEAKENED" and r["node"] == "x" for r in rep_wide.invariant_regressions)

        rep_rem = diff_graphs(g_base, g_removed)
        assert rep_rem.verdict == "FAIL"
        assert any(r["code"] == "CONTROL_BOUNDS_WEAKENED" and r["node"] == "x" for r in rep_rem.invariant_regressions)

        rep_tight = diff_graphs(g_base, g_tightened)
        assert not any(r["code"] == "CONTROL_BOUNDS_WEAKENED" for r in rep_tight.invariant_regressions)


# ==============================================================================
# 3. Alteración de Optimalidad de Políticas (Max, Min, Shift, Regret Cruzado)
# ==============================================================================

class TestPolicyOptimalityAlterations:
    def test_maximization_degradation_and_improvement(self):
        base_code = _make_clean_pricing_code(margin_mult=1.0, budget_limit=100.0)
        worse_code = _make_clean_pricing_code(margin_mult=0.6, budget_limit=100.0)
        better_code = _make_clean_pricing_code(margin_mult=1.4, budget_limit=100.0)

        rep_deg = diff_sources(base_code, worse_code)
        assert rep_deg.verdict == "FAIL"
        assert rep_deg.policy_optimality["direction"] == "degraded"
        assert rep_deg.policy_optimality["is_regression"] is True
        assert rep_deg.policy_optimality["objective_delta"] < -10.0

        rep_imp = diff_sources(base_code, better_code)
        assert rep_imp.has_regressions is False
        assert rep_imp.policy_optimality["direction"] == "improved"
        assert rep_imp.policy_optimality["is_regression"] is False
        assert rep_imp.policy_optimality["objective_delta"] > 10.0

    def test_minimization_sense_correctly_inverts_degradation_polarity(self):
        min_template = '''
SENSE = "min"
W1: float = 0.5
W2: float = 0.5
W1_BOUNDS = (0.0, 1.0)
W2_BOUNDS = (0.0, 1.0)
CONSTRAINTS = {{"c_sum": "w1 + w2 == 1.0"}}
def compute_total_cost(w1: float = 0.5, w2: float = 0.5) -> float:
    return {c_base} + 2.0 * (w1 ** 2) + 6.0 * (w2 ** 2)
'''
        code_low_cost = min_template.format(c_base="10.0")
        code_high_cost = min_template.format(c_base="25.0")

        # En sentido 'min', aumentar el coste óptimo (11.5 -> 26.5) ES una degradación crítica
        rep_worse = diff_sources(code_low_cost, code_high_cost, entrypoint="compute_total_cost")
        assert rep_worse.verdict == "FAIL"
        assert rep_worse.policy_optimality["sense"] == "min"
        assert rep_worse.policy_optimality["direction"] == "degraded"
        assert rep_worse.policy_optimality["is_regression"] is True
        assert rep_worse.policy_optimality["objective_delta"] == pytest.approx(15.0, abs=1e-2)

        # En sentido 'min', reducir el coste óptimo (26.5 -> 11.5) ES una mejora
        rep_better = diff_sources(code_high_cost, code_low_cost, entrypoint="compute_total_cost")
        assert rep_better.has_regressions is False
        assert rep_better.policy_optimality["direction"] == "improved"
        assert rep_better.policy_optimality["is_regression"] is False

    def test_optimal_lever_shift_regret_and_previous_policy_infeasibility(self):
        # Mismo valor óptimo J* = 100.0, pero el óptimo se desplaza de x=3 a x=7 -> WARN con regret > 0
        code_opt3 = '''
SENSE = "max"
X: float = 3.0
X_BOUNDS = (0.0, 10.0)
def eval_parabola(x: float = 3.0) -> float:
    return 100.0 - (x - 3.0) ** 2
'''
        code_opt7 = '''
SENSE = "max"
X: float = 3.0
X_BOUNDS = (0.0, 10.0)
def eval_parabola(x: float = 3.0) -> float:
    return 100.0 - (x - 7.0) ** 2
'''
        rep_shift = diff_sources(code_opt3, code_opt7, entrypoint="eval_parabola")
        assert rep_shift.verdict == DiffVerdict.WARN.value
        assert rep_shift.has_regressions is False
        assert rep_shift.policy_optimality["direction"] == "unchanged"
        assert rep_shift.policy_optimality["policy_shifted"] is True
        assert rep_shift.policy_optimality["control_deltas"]["x"] == pytest.approx(4.0, abs=1e-3)
        assert rep_shift.policy_optimality["base_policy_in_head"]["regret_vs_head_optimum"] == pytest.approx(16.0, abs=1e-2)

        # Ahora endurecemos una restricción de forma que X*_base (x=8) pasa a ser INFACTIBLE en head (x <= 6),
        # aunque J* mejore por una bonificación constante -> debe fallar por old_policy_became_infeasible!
        code_prev = '''
SENSE = "max"
X: float = 2.0
X_BOUNDS = (0.0, 10.0)
CONSTRAINTS = {"c_limit": "x <= 8.0"}
def eval_lin(x: float = 2.0) -> float:
    return 10.0 * x
'''
        code_infeas_prev = '''
SENSE = "max"
X: float = 2.0
X_BOUNDS = (0.0, 10.0)
CONSTRAINTS = {"c_limit": "x <= 6.0"}
def eval_lin(x: float = 2.0) -> float:
    return 10.0 * x + 50.0
'''
        rep_old_infeas = diff_sources(code_prev, code_infeas_prev, entrypoint="eval_lin")
        assert rep_old_infeas.verdict == "FAIL"
        assert rep_old_infeas.policy_optimality["old_policy_became_infeasible"] is True
        assert "c_limit" in rep_old_infeas.policy_optimality["base_policy_in_head"]["violated_constraints"]

        # Cuando tanto J* degrada (80 -> 60) COMO X*_base (x=8) pasa a ser infactible en head (x <= 6),
        # `diagnostics` debe emitir AMBOS códigos sin suprimir `PREVIOUS_POLICY_RENDERED_INFEASIBLE`
        code_both_deg = '''
SENSE = "max"
X: float = 2.0
X_BOUNDS = (0.0, 10.0)
CONSTRAINTS = {"c_limit": "x <= 6.0"}
def eval_lin(x: float = 2.0) -> float:
    return 10.0 * x
'''
        rep_both = diff_sources(code_prev, code_both_deg, entrypoint="eval_lin")
        assert rep_both.verdict == "FAIL"
        diag_codes = {d["code"] for d in rep_both.diagnostics}
        assert "POLICY_OPTIMALITY_REGRESSION" in diag_codes
        assert "PREVIOUS_POLICY_RENDERED_INFEASIBLE" in diag_codes


# ==============================================================================
# 4. Adición de Nodos Muertos (T1-A, T1-B, T1-C e Inmunidad de Guardas en C)
# ==============================================================================

class TestDeadNodesDetection:
    def test_structural_orphan_algebraic_cancellation_and_disconnected_control(self):
        base_code = _make_clean_pricing_code()

        # T1-A: Nuevo nodo estado huérfano
        orphan_code = _make_clean_pricing_code(include_dead_node=True)
        rep_orphan = diff_sources(base_code, orphan_code)
        assert rep_orphan.verdict == "FAIL"
        assert "vanity_impressions" in rep_orphan.dead_nodes["added_dead_node_ids"]
        dn0 = rep_orphan.dead_nodes["added_dead_nodes"][0]
        assert dn0["subtype"] == "T1-A_STRUCTURAL_ORPHAN"
        assert dn0["origin"] == "newly_added_actor"

        # Limpieza del nodo muerto en dirección inversa
        rep_clean = diff_sources(orphan_code, base_code)
        assert rep_clean.verdict == "PASS"
        assert "vanity_impressions" in rep_clean.dead_nodes["removed_dead_node_ids"]

        # T1-B: Cancelación algebraica exacta (+neutral_rebate - neutral_rebate => dJ/d_node == 0)
        cancel_code = _make_clean_pricing_code(include_cancel_node=True)
        rep_cancel = diff_sources(base_code, cancel_code)
        assert rep_cancel.verdict == "FAIL"
        assert "neutral_rebate" in rep_cancel.dead_nodes["added_dead_node_ids"]
        dn_c = next(d for d in rep_cancel.dead_nodes["added_dead_nodes"] if d["node"] == "neutral_rebate")
        assert dn_c["subtype"] == "T1-B_ALGEBRAIC_PATH_CANCELLATION"

        # T1-C: Palanca de control desconectada tanto del objetivo como de restricciones
        sr = SourceReference(file="ctrl.py", entity="f", line_range=[1, 2])
        g_live = CausalGraph(system_id="live_ctrl")
        g_live.add_node(Node("k", "k", "control", "continuous", sr, current_value=2.0, bounds={"min": 0.0, "max": 5.0}))
        g_live.add_node(Node("j", "j", "target", "continuous", sr, symbolic_expr="10 * k"))
        g_live.add_edge(Edge("k", "j", "exact_code"))

        g_dead_ctrl = CausalGraph(system_id="dead_ctrl")
        g_dead_ctrl.add_node(Node("k", "k", "control", "continuous", sr, current_value=2.0, bounds={"min": 0.0, "max": 5.0}))
        g_dead_ctrl.add_node(Node("j", "j", "target", "continuous", sr, symbolic_expr="20.0"))

        rep_dc = diff_graphs(g_live, g_dead_ctrl)
        assert rep_dc.verdict == "FAIL"
        assert "k" in rep_dc.dead_nodes["added_dead_node_ids"]
        assert rep_dc.dead_nodes["added_dead_nodes"][0]["subtype"] == "T1-C_DISCONNECTED_CONTROL"

    def test_state_node_guarding_hard_constraint_is_not_falsely_flagged_dead(self):
        sr = SourceReference(file="risk.py", entity="f", line_range=[1, 4])
        g = CausalGraph(system_id="guarded_state")
        g.add_node(Node("leverage", "Leverage", "control", "continuous", sr, current_value=2.0, bounds={"min": 1.0, "max": 5.0}))
        g.add_node(Node("risk_index", "Risk", "state", "continuous", sr, symbolic_expr="0.15 * leverage"))
        g.add_node(Node("profit", "Profit", "target", "continuous", sr, symbolic_expr="50.0 * leverage"))
        g.add_edge(Edge("leverage", "risk_index", "exact_code"))
        g.add_edge(Edge("leverage", "profit", "exact_code"))
        g.add_constraint(Constraint("c_risk_cap", "hard", "risk_index <= 0.60"))

        # `risk_index` no va a `profit`, pero gobierna la invariante dura `c_risk_cap` -> NO está muerto
        rep_same = diff_graphs(g, g)
        assert rep_same.dead_nodes["head_dead_nodes"] == []

        # Si alguien elimina `c_risk_cap`, entonces `risk_index` pierde todo rol causal y pasa a ser huérfano T1-A!
        g_no_constr = CausalGraph.from_dict(copy.deepcopy(g.to_dict()))
        g_no_constr.constraints.clear()
        rep_dropped = diff_graphs(g, g_no_constr)
        assert "risk_index" in rep_dropped.dead_nodes["added_dead_node_ids"]
        dn_r = next(d for d in rep_dropped.dead_nodes["added_dead_nodes"] if d["node"] == "risk_index")
        assert dn_r["origin"] == "disconnected_existing_actor"


# ==============================================================================
# 5. Integración Real con Repositorios Git y CLI CI/CD
# ==============================================================================

class TestGitIntegrationAndCLI:
    def test_multi_commit_git_workflow_and_worktree_diff(self, tmp_path: Path):
        repo = tmp_path / "mini_repo"
        pkg = repo / "pricing_core"
        pkg.mkdir(parents=True)

        _git(repo, "init")
        _git(repo, "config", "user.email", "ci@actordb.test")
        _git(repo, "config", "user.name", "ActorDB CI")

        # Commit 1: Línea base limpia
        (pkg / "model.py").write_text(_make_clean_pricing_code(margin_mult=1.0, budget_limit=100.0), encoding="utf-8")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "feat: clean initial pricing model")

        # Commit 2: Refactorización equivalente (con comentarios adicionales)
        (pkg / "model.py").write_text(
            "# Refactored header\n" + _make_clean_pricing_code(margin_mult=1.0, budget_limit=100.0),
            encoding="utf-8",
        )
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "refactor: add header comment")

        rep_1_2 = diff_git_revisions(repo, base_ref="HEAD~1", head_ref="HEAD", subpath="pricing_core")
        assert rep_1_2.verdict == "PASS"
        assert rep_1_2.is_semantically_equivalent is True
        assert rep_1_2.git_metadata is not None
        assert len(rep_1_2.git_metadata["base_commit"]) == 40
        assert rep_1_2.git_metadata["base_commit"] != rep_1_2.git_metadata["head_commit"]

        # Commit 3: PR regresivo (introduce nodo muerto + relaja presupuesto + degrada margen)
        (pkg / "model.py").write_text(
            _make_clean_pricing_code(margin_mult=0.5, budget_limit=130.0, include_dead_node=True),
            encoding="utf-8",
        )
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "chore: risky changes")

        # Invocando directamente sobre el subdirectorio `pkg` (resolución implícita de subpath!)
        rep_2_3 = CausalDiffEngine().diff_git(pkg, base_ref="HEAD~1", head_ref="HEAD")
        assert rep_2_3.verdict == "FAIL"
        assert rep_2_3.has_regressions is True
        assert "vanity_impressions" in rep_2_3.dead_nodes["added_dead_node_ids"]
        assert any(r["code"] == "HARD_CONSTRAINT_WEAKENED" for r in rep_2_3.invariant_regressions)
        assert rep_2_3.policy_optimality["is_regression"] is True

        # Modificación sin commitear en WORKTREE que restaura el modelo limpio y mejora el margen
        (pkg / "model.py").write_text(
            _make_clean_pricing_code(margin_mult=1.0, budget_limit=100.0, include_dead_node=False),
            encoding="utf-8",
        )
        rep_wt = diff_git_revisions(repo, base_ref="HEAD~1", head_ref="WORKTREE", subpath="pricing_core")
        assert rep_wt.verdict == "PASS"
        assert rep_wt.git_metadata["head_commit"] == "WORKTREE"

        # CLI `--check` devuelve código 1 ante regresiones (HEAD~1 vs HEAD) y 0 ante PASS (HEAD~2 vs HEAD~1)
        rc_fail = diff_cli_main([
            "--git-repo", str(repo),
            "--base-ref", "HEAD~1",
            "--head-ref", "HEAD",
            "--subpath", "pricing_core",
            "--format", "markdown",
            "--check",
        ])
        assert rc_fail == 1

        rc_pass = diff_cli_main([
            "--git-repo", str(repo),
            "--base-ref", "HEAD~2",
            "--head-ref", "HEAD~1",
            "--subpath", "pricing_core",
            "--format", "json",
            "--check",
        ])
        assert rc_pass == 0

    def test_error_handling_for_invalid_inputs_and_git_refs(self, tmp_path: Path):
        non_git = tmp_path / "not_a_git_repo"
        non_git.mkdir()

        with pytest.raises(CausalDiffError, match="not inside a valid Git repository"):
            diff_git_revisions(non_git, base_ref="HEAD~1", head_ref="HEAD")

        with pytest.raises(CausalDiffError, match="does not exist"):
            diff_git_revisions(tmp_path / "missing_folder")

        with pytest.raises(CausalDiffError, match="Invalid Git revision"):
            diff_git_revisions(".", base_ref="nonexistent_branch_999999", head_ref="HEAD", subpath="golden_case")

        with pytest.raises(CausalDiffError):
            load_causal_graph("")

        with pytest.raises(CausalDiffError):
            load_causal_graph({"not_nodes": []})

        rc_err = diff_cli_main(["--git-repo", str(non_git)])
        assert rc_err == 2

    def test_empty_causal_graphs_and_real_repo_git_diff(self):
        # 1. Grafos vacíos (0 actores) no deben lanzar excepciones en ningún sub-analizador
        g_empty_a = CausalGraph(system_id="empty_a")
        g_empty_b = CausalGraph(system_id="empty_b")
        rep_empty = diff_graphs(g_empty_a, g_empty_b)
        assert rep_empty.verdict == "PASS"
        assert rep_empty.is_semantically_equivalent is True
        assert "PASS" in rep_empty.to_markdown()

        # 2. Diff real sobre el historial Git de este repositorio (`golden_case` entre HEAD~1 y HEAD)
        repo_root = Path(__file__).resolve().parents[2]
        rep_real = diff_git_revisions(repo_root, base_ref="HEAD~1", head_ref="HEAD", subpath="golden_case")
        assert rep_real.verdict == "PASS"
        assert rep_real.is_semantically_equivalent is True
        assert rep_real.git_metadata is not None
        assert len(rep_real.git_metadata["head_commit"]) >= 7
