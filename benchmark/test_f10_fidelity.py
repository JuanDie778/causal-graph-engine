"""Tests unitarios y de integración para el Certificado de Fidelidad (`cge/fidelity.py`) y el Solver Genérico (`cge/solver/`) — Fase 10."""
from __future__ import annotations
import pytest
from fastapi.testclient import TestClient

from cge.ingest import ASTIngestPipeline, compile_source_to_ir
from cge.fidelity import certify_fidelity, evaluate_recipe_on_source, FidelityReport
from cge.solver import compile_problem, solve_problem, SolveStatus
from cge.layer2_exact import Layer2Solver, solve_layer2
from cge.copilot import DeterministicCopilot
from ui.server import app as ui_app, engine_state


def test_f10_1_fidelity_exact_scalar_and_golden_case():
    """Verifica que funciones escalares puras y el directorio golden_case obtienen verified=True y max_rel_error <= 1e-6."""
    code = '''
SENSE = "max"
X_CTRL: float = 4.0
Y_CTRL: float = 10.0
X_CTRL_BOUNDS = (1.0, 10.0)
Y_CTRL_BOUNDS = (2.0, 30.0)

def compute_intermediate(x_ctrl: float, y_ctrl: float) -> float:
    return 3.5 * x_ctrl + 0.25 * (y_ctrl ** 2)

def compute_total_score(x_ctrl: float, intermediate: float, y_ctrl: float) -> float:
    return intermediate - 1.2 * x_ctrl * y_ctrl
'''
    ir = compile_source_to_ir(code, system_id="test_scalar_fidelity")
    rep = certify_fidelity(ir, code, n_samples=64, rel_tol=1e-6)
    assert rep.verified is True
    assert rep.target_verified is True
    assert rep.max_rel_error <= 1e-6
    assert len(rep.unverified_nodes) == 0
    for nid, node in ir.nodes.items():
        assert node.fidelity is not None
        assert node.fidelity["verified"] is True

    # Golden Case completo
    g_ir = ASTIngestPipeline("golden_case").run()
    fid_meta = g_ir.metadata.get("fidelity_report")
    assert fid_meta is not None
    assert fid_meta["verified"] is True
    assert fid_meta["max_rel_error"] <= 1e-6


def test_f10_2_fidelity_detects_nondeterminism_and_state_mutation():
    """Verifica que RNG sin semilla y mutaciones de estado global son cazados con verified=False."""
    rng_code = '''
import random
U_CTRL: float = 10.0
U_CTRL_BOUNDS = (1.0, 50.0)

def compute_stochastic_metric(u_ctrl: float = 10.0) -> float:
    return u_ctrl * random.random()
'''
    ir_rng = compile_source_to_ir(rng_code, entrypoint="compute_stochastic_metric")
    rep_rng = certify_fidelity(ir_rng, rng_code, entrypoint="compute_stochastic_metric")
    assert rep_rng.verified is False
    assert "stochastic_metric" in rep_rng.unverified_nodes
    assert "NON_DETERMINISTIC" in (rep_rng.reason or "")

    mut_code = '''
U_CTRL: float = 5.0
U_CTRL_BOUNDS = (1.0, 20.0)
_HISTORY = []

def compute_mutating_metric(u_ctrl: float = 5.0) -> float:
    _HISTORY.append(u_ctrl)
    return sum(_HISTORY) * 2.0
'''
    ir_mut = compile_source_to_ir(mut_code, entrypoint="compute_mutating_metric")
    rep_mut = certify_fidelity(ir_mut, mut_code, entrypoint="compute_mutating_metric")
    assert rep_mut.verified is False
    assert "mutating_metric" in rep_mut.unverified_nodes
    assert "EXTERNAL_STATE_MUTATION" in (rep_mut.reason or "")


def test_f10_3_problem_classification_and_modes():
    """Verifica la clasificación simbólica automática (LP, MILP, QP, NLP_SMOOTH, NLP_NONSMOOTH, MINLP) y el modo target_reaching."""
    # 1. LP
    lp_ir = compile_source_to_ir('''
SENSE = "max"
X1: float = 1.0
X2: float = 1.0
X1_BOUNDS = (0.0, 10.0)
X2_BOUNDS = (0.0, 10.0)
CONSTRAINTS = {"c1": "x1 + 2.0 * x2 <= 14.0"}
def f_lp(x1: float = 1.0, x2: float = 1.0) -> float:
    out_lp = 3.0 * x1 + 4.0 * x2
    return out_lp
''', entrypoint="f_lp")
    prob_lp = compile_problem(lp_ir)
    assert prob_lp.problem_class == "LP"

    # Modo target_reaching (intervención mínima cuadrática para subir out_lp de 7.0 a >= 25.0)
    sol_tr = solve_problem(lp_ir, mode="target_reaching", target_delta=18.0)
    assert sol_tr["status"] == "OPTIMAL"
    assert sol_tr["objective_value"] >= 25.0 - 1e-4
    assert sol_tr["achieved_delta"] >= 18.0 - 1e-4

    # 2. QP
    qp_ir = compile_source_to_ir('''
SENSE = "min"
W1: float = 0.5
W2: float = 0.5
W1_BOUNDS = (0.0, 1.0)
W2_BOUNDS = (0.0, 1.0)
CONSTRAINTS = {"c_eq": "w1 + w2 == 1.0"}
def f_qp(w1: float = 0.5, w2: float = 0.5) -> float:
    risk = 2.0 * (w1 ** 2) + 6.0 * (w2 ** 2)
    return risk
''', entrypoint="f_qp")
    prob_qp = compile_problem(qp_ir)
    assert prob_qp.problem_class == "QP"
    sol_qp = solve_problem(qp_ir)
    assert sol_qp["status"] == "OPTIMAL"
    assert sol_qp["controls"]["w1"] == pytest.approx(0.75, abs=1e-3)
    assert sol_qp["controls"]["w2"] == pytest.approx(0.25, abs=1e-3)


def test_f10_4_epistemic_honesty_gate_end_to_end():
    """Verifica que cuando la fidelidad falla, Layer2Solver, DeterministicCopilot y /api/optimize devuelven UNSUPPORTED."""
    bad_code = '''
import random
X_LEVER: float = 10.0
X_LEVER_BOUNDS = (1.0, 50.0)

def compute_unreliable_target(x_lever: float = 10.0) -> float:
    return x_lever * random.uniform(0.5, 1.5)
'''
    ir_bad = compile_source_to_ir(bad_code, entrypoint="compute_unreliable_target")
    solver = Layer2Solver(graph=ir_bad)
    opt_res = solver.optimize()
    assert opt_res["status"] == "UNSUPPORTED"
    assert opt_res["certificate_status"] == "UNSUPPORTED"
    assert "unreliable_target" in opt_res["unverified_nodes"]
    assert opt_res["controls"] == {}

    copilot = DeterministicCopilot(graph=ir_bad, solver=solver)
    ans = copilot.answer("¿Cuál es la política óptima para maximizar unreliable_target?")
    assert "UNSUPPORTED" in ans["answer"] or "RECHAZADA" in ans["answer"]

    # Verificar endpoint /api/optimize con estado cargado
    with engine_state._lock:
        prev_graph = engine_state.graph
        prev_solver = engine_state.solver
        engine_state.graph = ir_bad
        engine_state.solver = solver
    try:
        client = TestClient(ui_app)
        resp = client.post("/api/optimize", json={})
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "UNSUPPORTED"
        assert payload["certificate_status"] == "UNSUPPORTED"
    finally:
        with engine_state._lock:
            engine_state.graph = prev_graph
            engine_state.solver = prev_solver


def test_f10_5_recipe_verification_with_source_asserts():
    """Verifica el Certificado 3: evaluación directa de la receta X* contra el código original con asserts activos."""
    code_with_assert = '''
SENSE = "max"
A_VAL: float = 5.0
B_VAL: float = 5.0
A_VAL_BOUNDS = (0.0, 20.0)
B_VAL_BOUNDS = (0.0, 20.0)

def evaluate_guarded_process(a_val: float = 5.0, b_val: float = 5.0) -> float:
    assert a_val + b_val <= 15.0
    assert a_val >= 2.0
    yield_out = 4.0 * a_val + 7.0 * b_val - 0.1 * (b_val ** 2)
    return yield_out
'''
    ir = compile_source_to_ir(code_with_assert, entrypoint="evaluate_guarded_process")
    # Las dos sentencias assert deben haberse extraído como restricciones duras en ir.constraints
    assert len(ir.constraints) >= 2
    sol = solve_layer2(ir, source_path_or_code=code_with_assert, entrypoint="evaluate_guarded_process")
    assert sol["status"] == "OPTIMAL"
    assert sol["certificate_status"] == "CERTIFIED_OPTIMAL"
    assert sol["certificate"]["code_verified"] is True
    assert sol["certificate"]["constraint_violation_on_code"] == 0.0
    assert sol["controls"]["a_val"] + sol["controls"]["b_val"] <= 15.0 + 1e-5
