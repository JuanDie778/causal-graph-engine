"""Nivel N8: Suite Externa de Benchmark OR-Bench (20 Problemas Clásicos de Investigación Operativa, Fase 10).

Evalúa la capacidad del motor para compilar, certificar fidelidad y resolver 20 problemas
clásicos de Investigación Operativa (10 públicos + 10 sellados) escritos en Python/pandas natural,
con 0 solapamiento con el patrón de la tostadora de café y 0 falsa confianza.
"""
from __future__ import annotations
import pytest

from benchmark.or_bench import (
    get_public_problems,
    get_sealed_problems,
    get_all_or_bench_problems,
    evaluate_single_or_problem,
    evaluate_or_bench_suite,
    format_or_bench_report_table,
)


PUBLIC_PROBS = get_public_problems()
SEALED_PROBS = get_sealed_problems()


@pytest.mark.parametrize("prob", PUBLIC_PROBS, ids=[p["id"] for p in PUBLIC_PROBS])
def test_n8_or_bench_public_problem(prob):
    """N8-A: Verifica cada uno de los 10 problemas públicos de OR-Bench frente a su oráculo independiente."""
    res = evaluate_single_or_problem(prob)
    assert res["false_confidence"] is False, (
        f"FALSA CONFIANZA detectada en {res['id']}: esperado={res['expected_status']}, obtenido={res['engine_status']}"
    )
    assert res["passed"] is True, (
        f"Fallo en problema público {res['id']} ({res['name']}): "
        f"status={res['engine_status']} (esperado={res['expected_status']}), "
        f"J_engine={res['engine_objective']}, J_oracle={res['oracle_objective']}, "
        f"gap={res['rel_optimality_gap']}, iis={res['iis_constraints']} vs {res['oracle_iis']}"
    )


@pytest.mark.parametrize("prob", SEALED_PROBS, ids=[p["id"] for p in SEALED_PROBS])
def test_n8_or_bench_sealed_problem(prob):
    """N8-B: Verifica cada uno de los 10 problemas sellados (holdout) de OR-Bench frente a su oráculo independiente."""
    res = evaluate_single_or_problem(prob)
    assert res["false_confidence"] is False, (
        f"FALSA CONFIANZA detectada en {res['id']}: esperado={res['expected_status']}, obtenido={res['engine_status']}"
    )
    assert res["passed"] is True, (
        f"Fallo en problema sellado {res['id']} ({res['name']}): "
        f"status={res['engine_status']} (esperado={res['expected_status']}), "
        f"J_engine={res['engine_objective']}, J_oracle={res['oracle_objective']}, "
        f"gap={res['rel_optimality_gap']}, iis={res['iis_constraints']} vs {res['oracle_iis']}"
    )


def test_n8_or_bench_aggregate_gates_and_report():
    """N8-C: Verifica las métricas agregadas de P9 (Público >= 90%, Sellado >= 85%, Fidelidad >= 99%, Violaciones = 0, Falsa Confianza = 0)."""
    pub_summary = evaluate_or_bench_suite("public")
    sealed_summary = evaluate_or_bench_suite("sealed")
    all_summary = evaluate_or_bench_suite("all")

    table_str = format_or_bench_report_table(all_summary)
    print("\n" + table_str)

    assert pub_summary["pass_rate"] >= 0.90, f"Pass rate público {pub_summary['pass_rate']:.1%} < 90%"
    assert sealed_summary["pass_rate"] >= 0.85, f"Pass rate sellado {sealed_summary['pass_rate']:.1%} < 85%"
    assert all_summary["false_confidence_count"] == 0, (
        f"Puerta de Falsa Confianza violada: {all_summary['false_confidence_count']} casos"
    )
    assert all_summary["max_code_constraint_violation"] == 0.0, (
        f"Violación de restricción dura en código original: {all_summary['max_code_constraint_violation']}"
    )
    assert all_summary["fidelity_verified_rate"] >= 0.99, (
        f"Tasa de fidelidad en problemas soportados {all_summary['fidelity_verified_rate']:.2%} < 99%"
    )
