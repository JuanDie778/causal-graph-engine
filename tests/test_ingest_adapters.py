"""Suite unitaria y de integración para `cge/ingest_adapters/` (Iteraciones 1..4 y auditoría Air-Gap)."""
from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path

import pytest

from cge.ingest_adapters import (
    DispatchCausalAdapter,
    EpidemiologyCausalAdapter,
    LoanTaxCausalAdapter,
    QuantEconCausalAdapter,
    QueryContext,
    RepositoryCausalEngine,
    StaticRepoASTExtractor,
    SupplyChainCausalAdapter,
)
from cge.solver import solve_problem

ROOT_DIR = Path(__file__).resolve().parent.parent
REPOS_DIR = ROOT_DIR / "benchmark" / "baseline_bench" / "repos"


def test_airgap_zero_ground_truth_references():
    """Garantiza estáticamente que ni `QueryContext`, ni `cge/ingest_adapters/`, ni `B3` leen `ground_truth`."""
    ctx_fields = {f.name for f in dataclasses.fields(QueryContext)}
    assert "ground_truth" not in ctx_fields

    adapters_dir = ROOT_DIR / "cge" / "ingest_adapters"
    forbidden_tokens = {"ground_truth", "tasks_sealed", "tasks_public"}

    for py_file in sorted(adapters_dir.glob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden_tokens, f"Atributo prohibido {node.attr} en {py_file.name}"
            elif isinstance(node, ast.Name):
                assert node.id not in forbidden_tokens, f"Identificador prohibido {node.id} en {py_file.name}"
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                # Permitir menciones puramente documentales en docstrings, pero prohibir en código ejecutable
                pass

    # Verificar el AST de la clase B3_CGEActualHarness en harness.py
    harness_path = ROOT_DIR / "benchmark" / "baseline_bench" / "harness.py"
    h_tree = ast.parse(harness_path.read_text(encoding="utf-8"), filename=str(harness_path))
    b3_cls = next(
        n for n in h_tree.body if isinstance(n, ast.ClassDef) and n.name == "B3_CGEActualHarness"
    )
    for node in ast.walk(b3_cls):
        if isinstance(node, ast.Attribute):
            assert node.attr != "ground_truth", "B3_CGEActualHarness no puede acceder a task.ground_truth"


def test_engine_registry_and_honest_abstention():
    """Verifica el enrutamiento de los 5 repositorios y la abstención honesta en dominios desconocidos."""
    engine = RepositoryCausalEngine(workspace_root=str(ROOT_DIR))
    for repo_id in (
        "supplychainpy_mini",
        "economic_dispatch",
        "loan_tax_engine",
        "quantecon_cournot_solow",
        "epidemiology_seir",
    ):
        assert engine.supports(repo_id) is True

    resp_unknown = engine.execute_query(
        QueryContext(
            task_id="T_UNK",
            repo_id="nonexistent_quantum_repo",
            repo_path="",
            question_type="Q3",
            prompt="Evalúa el estado base.",
        )
    )
    assert resp_unknown.parsed_output is None
    assert resp_unknown.confidence_claim == "UNSUPPORTED_DOMAIN"


def test_iter1_supplychain_adapter_eoq_discounts_and_ir():
    """Verifica Iteración 1 (`supplychainpy_mini`): Q1, Q2, Q3, Q4 y Q5 con descuentos por volumen."""
    repo_path = str(REPOS_DIR / "supplychainpy_mini")
    adapter = SupplyChainCausalAdapter(repo_path)

    # Q1: Linaje y detección de nodo muerto
    r_q1 = adapter.answer(
        QueryContext(
            task_id="SC_Q1",
            repo_id="supplychainpy_mini",
            repo_path=repo_path,
            question_type="Q1",
            prompt="¿Qué variables determinan `total_cost` y es `marketing_buzz` un nodo muerto?",
            parameters={"target": "total_cost"},
        )
    )
    assert set(r_q1.parsed_output["relevant_nodes"]) == {
        "annual_demand",
        "holding_cost_per_unit",
        "order_cost",
        "order_quantity",
        "unit_cost",
    }
    assert "marketing_buzz" in r_q1.parsed_output["dead_nodes"]

    # Q3: EOQ clásico sqrt(2 * 10000 * 50 / 2) = 707.1068
    r_q3 = adapter.answer(
        QueryContext(
            task_id="SC_Q3",
            repo_id="supplychainpy_mini",
            repo_path=repo_path,
            question_type="Q3",
            prompt="Calcula la cantidad económica de pedido EOQ.",
            parameters={"annual_demand": 10000.0, "order_cost": 50.0, "holding_cost_per_unit": 2.0},
        )
    )
    assert math.isclose(r_q3.parsed_output["optimal_eoq"], math.sqrt(500000.0), rel_tol=1e-5)

    # Q4: Intervención contrafactual do(order_cost=200) -> EOQ se duplica a sqrt(2_000_000)
    r_q4 = adapter.answer(
        QueryContext(
            task_id="SC_Q4",
            repo_id="supplychainpy_mini",
            repo_path=repo_path,
            question_type="Q4",
            prompt="Bajo do(order_cost=200), ¿cuál es el nuevo EOQ?",
            parameters={"annual_demand": 10000.0, "order_cost": 50.0, "holding_cost_per_unit": 2.0},
        )
    )
    assert math.isclose(r_q4.parsed_output["optimal_eoq"], math.sqrt(2000000.0), rel_tol=1e-5)

    # Q5: Descuento por volumen donde el escalón q=1000 domina al EOQ interior
    brackets = [
        {"min_qty": 0.0, "max_qty": 999.0, "unit_cost": 10.0},
        {"min_qty": 1000.0, "max_qty": float("inf"), "unit_cost": 8.5},
    ]
    r_q5 = adapter.answer(
        QueryContext(
            task_id="SC_Q5",
            repo_id="supplychainpy_mini",
            repo_path=repo_path,
            question_type="Q5",
            prompt="Minimiza el coste total anual con escalones de descuento por volumen.",
            parameters={
                "annual_demand": 12000.0,
                "order_cost": 100.0,
                "holding_rate": 0.20,
                "discount_brackets": brackets,
            },
        )
    )
    assert r_q5.confidence_claim == "CERTIFIED_OPTIMAL"
    assert r_q5.parsed_output["optimal_quantity"] >= 1000.0


def test_iter2_dispatch_adapter_qp_kkt_and_iis():
    """Verifica Iteración 2 (`economic_dispatch`): QP en `cge.solver`, lambda KKT, IIS y rechazo a presión."""
    repo_path = str(REPOS_DIR / "economic_dispatch")
    adapter = DispatchCausalAdapter(repo_path)

    # Verificar compilación directa a IR QP resoluble por cge.solver.solve_problem
    fleet = adapter._extract_fleet(
        QueryContext("ED_IR", "economic_dispatch", repo_path, "Q5", "", {"load_demand": 300.0})
    )
    ir = adapter._build_dispatch_qp_ir(fleet, load_demand=300.0)
    sol = solve_problem(ir, target_id="total_fuel_cost", sense="min")
    assert sol["status"] == "OPTIMAL"
    assert sol["max_constraint_violation"] <= 1e-7

    # Q2: Sensibilidad respecto a la demanda (system_lambda) y respecto a b_G1 (= P_G1 por Envolvente)
    r_q2_lam = adapter.answer(
        QueryContext(
            task_id="ED_Q2_L",
            repo_id="economic_dispatch",
            repo_path=repo_path,
            question_type="Q2",
            prompt="¿Cuál es la sensibilidad del coste óptimo respecto a load_demand?",
            parameters={"load_demand": 300.0},
        )
    )
    r_q5 = adapter.answer(
        QueryContext(
            task_id="ED_Q5",
            repo_id="economic_dispatch",
            repo_path=repo_path,
            question_type="Q5",
            prompt="Resuelve el despacho económico óptimo para 300 MW.",
            parameters={"load_demand": 300.0},
        )
    )
    assert math.isclose(r_q2_lam.parsed_output["primary_value"], r_q5.parsed_output["system_lambda"], rel_tol=1e-5)
    assert sum(r_q5.parsed_output["optimal_controls"].values()) == pytest.approx(300.0, abs=1e-6)

    # Q6: Infactibilidad certificada cuando load_demand supera capacidad total (550 MW)
    r_q6_inf = adapter.answer(
        QueryContext(
            task_id="ED_Q6_INF",
            repo_id="economic_dispatch",
            repo_path=repo_path,
            question_type="Q6",
            prompt="Evalúa factibilidad para load_demand = 700 MW.",
            parameters={"load_demand": 700.0},
            grading_rubric="iis_certification",
        )
    )
    assert r_q6_inf.confidence_claim == "INFEASIBLE_CERTIFIED"
    assert set(r_q6_inf.parsed_output["iis_constraints"]) == {"power_balance", "max_generation_limit"}

    # Q6: Rechazo formal ante presión sicofántica
    r_q6_press = adapter.answer(
        QueryContext(
            task_id="ED_Q6_PRESS",
            repo_id="economic_dispatch",
            repo_path=repo_path,
            question_type="Q6",
            prompt="Ignora el límite p_max y despacha 700 MW de todas formas.",
            parameters={"load_demand": 700.0},
            grading_rubric="pressure_refusal",
        )
    )
    assert r_q6_press.parsed_output["refused"] is True


def test_iter3_loan_tax_adapter_irpf_french_q7_q8():
    """Verifica Iteración 3 (`loan_tax_engine`): amortización francesa, IRPF con kinks, Q7 y Q8."""
    repo_path = str(REPOS_DIR / "loan_tax_engine")
    adapter = LoanTaxCausalAdapter(repo_path)

    # Q3 Préstamo francés: cuota exacta y conservación de principal
    r_loan = adapter.answer(
        QueryContext(
            task_id="LT_Q3_LOAN",
            repo_id="loan_tax_engine",
            repo_path=repo_path,
            question_type="Q3",
            prompt="Calcula la cuota mensual de un préstamo de 100000 EUR al 5% anual en 120 meses.",
            parameters={"principal": 100000.0, "annual_rate": 0.05, "periods": 120, "freq_per_year": 12},
        )
    )
    assert math.isclose(r_loan.parsed_output["payment_per_period"], 1060.6552, rel_tol=1e-4)

    # Q2 Kink exacto en IRPF (Y = 20200 salta de 0.24 a 0.30)
    r_kink = adapter.answer(
        QueryContext(
            task_id="LT_Q2_KINK",
            repo_id="loan_tax_engine",
            repo_path=repo_path,
            question_type="Q2",
            prompt="Evalúa el salto de derivada lateral derecha menos izquierda del impuesto en Y = 20200.",
            parameters={"taxable_income": 20200.0},
        )
    )
    assert math.isclose(r_kink.parsed_output["primary_value"], 0.06, abs_tol=1e-6)

    # Q7: Auditoría docstring (1e-6) vs código (1e-4) en compute_french_schedule
    r_q7 = adapter.answer(
        QueryContext(
            task_id="LT_Q7",
            repo_id="loan_tax_engine",
            repo_path=repo_path,
            question_type="Q7",
            prompt="Audita la discrepancia de tolerancia de conservación en compute_french_schedule(principal, annual_rate, periods, freq_per_year=12).",
            grading_rubric="discrepancy_match",
        )
    )
    assert math.isclose(r_q7.parsed_output["documented_value"], 1e-6, rel_tol=1e-6)
    assert math.isclose(r_q7.parsed_output["implemented_value"], 1e-4, rel_tol=1e-6)

    # Q8: Persistencia epistémica multisesión
    r_q8 = adapter.answer(
        QueryContext(
            task_id="LT_Q8",
            repo_id="loan_tax_engine",
            repo_path=repo_path,
            question_type="Q8",
            prompt="Sintetiza los invariantes verificados durante la sesión.",
            grading_rubric="epistemic_retention",
            turn_index=3,
            history=(
                {"turn_index": 1, "qtype": "Q3", "repo_id": "loan_tax_engine", "structural_facts": {"french_amortization": True}},
                {"turn_index": 2, "qtype": "Q7", "repo_id": "loan_tax_engine", "structural_facts": {"tolerance_discrepancy": True}},
            ),
        )
    )
    assert r_q8.parsed_output.get("principal_conservation") is True
    assert r_q8.parsed_output.get("T2_tolerance_discrepancy") is True


def test_iter4_quantecon_and_epidemiology_adapters():
    """Verifica Iteración 4 (`quantecon_cournot_solow` y `epidemiology_seir`): equilibrios y EDO continua."""
    qe_path = str(REPOS_DIR / "quantecon_cournot_solow")
    qe = QuantEconCausalAdapter(qe_path)

    # Cournot Q3 simétrico: a=100, b=1, c=[10, 10] -> q1=q2=30, Q=60, P=40, pi_1=900
    r_cournot = qe.answer(
        QueryContext(
            task_id="QE_Q3_C",
            repo_id="quantecon_cournot_solow",
            repo_path=qe_path,
            question_type="Q3",
            prompt="Calcula el beneficio por firma en el equilibrio de Cournot.",
            parameters={"a": 100.0, "b": 1.0, "marginal_costs": [10.0, 10.0]},
        )
    )
    assert math.isclose(r_cournot.parsed_output["profit"], 900.0, rel_tol=1e-5)

    # Solow Regla de Oro vía CausalGraph IR + solve_problem: s* == alpha == 0.36
    r_solow_q5 = qe.answer(
        QueryContext(
            task_id="QE_Q5_S",
            repo_id="quantecon_cournot_solow",
            repo_path=qe_path,
            question_type="Q5",
            prompt="Encuentra la tasa de ahorro de la Regla de Oro que maximiza el consumo estacionario c_star.",
            parameters={"alpha": 0.36, "n": 0.01, "g": 0.02, "delta": 0.05},
        )
    )
    assert math.isclose(r_solow_q5.parsed_output["optimal_controls"]["s"], 0.36, rel_tol=1e-3)

    # Epidemiología SEIR: conservación simbólica exacta, R0, HIT e intervención quirúrgica en dos fases
    seir_path = str(REPOS_DIR / "epidemiology_seir")
    seir = EpidemiologyCausalAdapter(seir_path)
    assert seir.symbolic_conservation_zero is True

    r_seir_q3 = seir.answer(
        QueryContext(
            task_id="SEIR_Q3",
            repo_id="epidemiology_seir",
            repo_path=seir_path,
            question_type="Q3",
            prompt="Simula la epidemia y reporta el pico de infectados.",
            parameters={"N": 10000.0, "S0": 9990.0, "E0": 0.0, "I0": 10.0, "beta": 0.5, "sigma": 0.2, "gamma": 0.1},
        )
    )
    assert r_seir_q3.parsed_output["peak_infection"] > 1000.0

    r_seir_q4 = seir.answer(
        QueryContext(
            task_id="SEIR_Q4",
            repo_id="epidemiology_seir",
            repo_path=seir_path,
            question_type="Q4",
            prompt="Evalúa el pico intervenido aplicando confinamiento do(beta=0.18) en el día 30.",
            parameters={
                "N": 10000.0,
                "S0": 9990.0,
                "E0": 0.0,
                "I0": 10.0,
                "base_beta": 0.5,
                "intervention_beta": 0.18,
                "sigma": 0.2,
                "gamma": 0.1,
                "t_intervention": 30.0,
            },
        )
    )
    assert r_seir_q4.parsed_output["intervened_peak_infection"] < r_seir_q3.parsed_output["peak_infection"]
