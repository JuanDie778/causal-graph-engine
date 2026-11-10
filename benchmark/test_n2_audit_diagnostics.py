"""Nivel N2: Auditoría y Diagnóstico de Coherencia Causal (A-1 a A-7)."""
import json
import pytest
from cge.ingest import run_ingestion_pipeline
from cge.audit import SystemAuditor


@pytest.fixture(scope="module")
def audit_results():
    graph = run_ingestion_pipeline("golden_case", "schema.json")
    auditor = SystemAuditor("golden_case", graph)
    diags = auditor.audit_all()
    signs = auditor.check_sensitivity_signs()
    with open("oracle/expected_diagnostics.json", "r", encoding="utf-8") as f:
        expected = json.load(f)["diagnostics"]
    return {
        "auditor": auditor,
        "diagnostics": diags,
        "expected": expected,
        "signs": signs
    }


def test_a1_dead_node_detection_t1(audit_results):
    """A-1: Detecta T1 (nodo muerto) señalando brand_awareness."""
    diags = audit_results["diagnostics"]
    t1 = [d for d in diags if d.get("id") == "T1"]
    assert len(t1) == 1, "Debe detectar exactamente una alerta T1"
    assert t1[0]["node"] == "brand_awareness"
    assert t1[0]["code"] == "DEAD_NODE"


def test_a2_invariant_violation_t2(audit_results):
    """A-2: Detecta T2 (rompe c_ledger) señalando la función y las líneas exactas."""
    diags = audit_results["diagnostics"]
    t2 = [d for d in diags if d.get("id") == "T2"]
    assert len(t2) == 1, "Debe detectar exactamente una alerta T2"
    assert t2[0]["constraint_id"] == "c_ledger"
    assert t2[0]["function"] == "process_refund"
    assert t2[0]["line_range"] == [63, 73], f"Rango de líneas de process_refund incorrecto: {t2[0]['line_range']}"
    # Verificar físicamente el código fuente
    with open("golden_case/finance.py", "r", encoding="utf-8") as f:
        code_lines = f.readlines()
    fn_snippet = "".join(code_lines[62:73])
    assert "def process_refund" in fn_snippet
    assert "refund_cash_outflow" in fn_snippet
    assert "refund_tax_adjustment" in fn_snippet


def test_a3_doc_code_discrepancy_t3(audit_results):
    """A-3: Detecta T3 (README dice 2%, código hace 3%)."""
    diags = audit_results["diagnostics"]
    t3 = [d for d in diags if d.get("id") == "T3"]
    assert len(t3) == 1, "Debe detectar exactamente una alerta T3"
    assert abs(t3[0]["doc_value"] - 0.02) < 1e-4
    assert abs(t3[0]["code_value"] - 0.03) < 1e-4


def test_a4_local_optimum_quantification_t4(audit_results):
    """A-4: Detecta T4: explica que q local (500) es subóptimo y cuantifica pérdida en J (±2%)."""
    diags = audit_results["diagnostics"]
    exp = audit_results["expected"]
    t4 = [d for d in diags if d.get("id") == "T4"][0]
    t4_exp = [d for d in exp if d.get("id") == "T4"][0]

    found_loss = t4["profit_loss_at_best_p_m"]
    exp_loss = t4_exp["profit_loss_at_best_p_m"]
    rel_error = abs(found_loss - exp_loss) / exp_loss
    print(f"\nT4 Pérdida cuantificada: {found_loss}€ vs Esperada: {exp_loss}€ (Error: {rel_error*100:.2f}%)")
    assert rel_error <= 0.02, f"Error relativo en cuantificación de pérdida ({rel_error}) excede el 2%"


def test_a5_fragile_node_t6(audit_results):
    """A-5: Marca sales como frágil (T6) en el óptimo."""
    diags = audit_results["diagnostics"]
    t6 = [d for d in diags if d.get("id") == "T6"]
    assert len(t6) >= 1, "Debe detectar la fragilidad de sales (T6)"
    assert t6[0]["node"] == "sales"


def test_a6_false_positives_limit(audit_results):
    """A-6: Falsos positivos: diagnósticos que no están en expected_diagnostics.json <= 1."""
    found_ids = {d["id"] for d in audit_results["diagnostics"]}
    expected_ids = {d["id"] for d in audit_results["expected"]}
    false_positives = found_ids - expected_ids
    print(f"\nA-6 Falsos positivos encontrados: {false_positives} (Total: {len(false_positives)})")
    assert len(false_positives) <= 1


def test_a7_sensitivity_signs(audit_results):
    """A-7: Sensibilidades de la Capa 1 con el signo correcto (100%)."""
    signs = audit_results["signs"]
    assert "validations" in signs, "Faltan validaciones matemáticas de sensibilidades en Capa 1"
    assert signs["validations"]["price_positive"], "Sensibilidad del precio hacia beneficio debe ser positiva"
    assert signs["validations"]["brand_zero"], "Sensibilidad de brand_awareness hacia beneficio debe ser nula (0.0)"
    assert signs["validations"]["costs_negative"], "Costes y comisiones deben tener sensibilidad negativa hacia beneficio"
    assert signs["all_signs_correct"], "Hay signos de sensibilidad incorrectos en las palancas"
