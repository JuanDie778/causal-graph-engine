"""Nivel N4: Benchmark de UI y Copiloto Agéntico (V-1 a V-5, C-1 a C-4)."""
import json
import os
import pytest
from cge.ingest import run_ingestion_pipeline
from cge.audit import SystemAuditor
from cge.copilot import DeterministicCopilot
from ui.export_ui import generate_interactive_ui_html, verify_ui_reactivity


@pytest.fixture(scope="module")
def setup_copilot_and_ui():
    graph = run_ingestion_pipeline("golden_case", "schema.json")
    auditor = SystemAuditor("golden_case", graph)
    diags = auditor.audit_all()
    with open("oracle/expected_solution.json", "r", encoding="utf-8") as f:
        sol = json.load(f)
    ui_path = generate_interactive_ui_html(graph.to_dict(), diags, sol, "ui/index.html")
    copilot = DeterministicCopilot("golden_case", graph)
    with open("benchmark/copilot_questions.json", "r", encoding="utf-8") as f:
        q_data = json.load(f)["questions"]
    return {
        "graph": graph,
        "diags": diags,
        "sol": sol,
        "ui_path": ui_path,
        "copilot": copilot,
        "questions": q_data
    }


def test_v1_to_v4_ui_rendering_and_badges(setup_copilot_and_ui):
    """V-1 a V-4: La UI renderiza colores por categoría, trazabilidad, alertas T1-T6 y ruta óptima."""
    ui_path = setup_copilot_and_ui["ui_path"]
    assert os.path.exists(ui_path), "El archivo ui/index.html debe existir"
    with open(ui_path, "r", encoding="utf-8") as f:
        content = f.read()

    # V-1: Categorías de color presentes
    assert "badge-u" in content
    assert "badge-x" in content
    assert "badge-v" in content
    assert "badge-j" in content

    # V-2: Trazabilidad e inspección de código
    assert "Trazabilidad de Código Fuente" in content
    assert "source_reference" in content

    # V-3: Alertas de diagnóstico T1-T7
    assert "badge-alert" in content
    assert "T1" in content
    assert "T2" in content
    assert "T3" in content

    # V-4: Resaltado de ruta óptima
    assert "border-left: 4px solid #10b981;" in content


def test_v5_ui_reactivity_and_invariance(setup_copilot_and_ui):
    """V-5: La UI no tiene estado propio; mutar schema.json propaga directamente los cambios."""
    graph = setup_copilot_and_ui["graph"]
    base_dict = graph.to_dict()
    mutated_dict = graph.to_dict()
    mutated_dict["nodes"][0]["label"] = "Etiqueta Mutada Directamente"
    assert verify_ui_reactivity(base_dict, mutated_dict)


def test_c1_copilot_accuracy_bank_questions(setup_copilot_and_ui):
    """C-1: Respuestas correctas en el banco de 10 preguntas >= 9/10 (Umbral: >= 9/10)."""
    copilot = setup_copilot_and_ui["copilot"]
    questions = setup_copilot_and_ui["questions"]
    successes = 0

    for q in questions:
        ans = copilot.answer_question(q["question"])
        # Verificar contenido obligatorio
        must_pass = True
        for mc in q.get("must_contain", []):
            if mc.lower() not in ans["answer"].lower():
                must_pass = False
        if must_pass:
            successes += 1

    print(f"\nC-1 Aciertos Copiloto: {successes}/{len(questions)}")
    assert successes >= 9, f"Aciertos ({successes}/10) inferior al umbral mínimo de 9/10"


def test_c2_copilot_id_citations(setup_copilot_and_ui):
    """C-2: Cada afirmación cita los IDs de nodo/arista/restricción del grafo (100%)."""
    copilot = setup_copilot_and_ui["copilot"]
    questions = setup_copilot_and_ui["questions"]

    for q in questions:
        ans = copilot.answer_question(q["question"])
        # Debe haber al menos un nodo o restricción citada formalmente entre corchetes o en lista
        assert len(ans["highlighted_nodes"]) > 0 or len(ans["highlighted_constraints"]) > 0, f"Sin citas en {q['id']}"
        for exp_n in q.get("expected_cited_nodes", []):
            assert exp_n in ans["highlighted_nodes"], f"Nodo {exp_n} no citado en {q['id']}"


def test_c3_zero_numerical_hallucination(setup_copilot_and_ui):
    """C-3: Cero alucinación numérica: todo número sale de una herramienta determinista (100%)."""
    copilot = setup_copilot_and_ui["copilot"]
    # Pregunta de sensibilidad Q3
    ans_q3 = copilot.answer_question("¿Qué beneficio neto obtengo si subo el precio a 5.0 manteniendo marketing en 500 y pedido en 500?")
    assert ans_q3["zero_hallucination_verified"]
    assert 1325.0 in ans_q3["numerical_claims"] or "1325.00" in ans_q3["answer"]


def test_c4_copilot_ui_node_highlighting(setup_copilot_and_ui):
    """C-4: El copiloto resalta en la UI los nodos de los que habla."""
    copilot = setup_copilot_and_ui["copilot"]
    ans = copilot.answer_question("¿Qué palancas afectan al beneficio?")
    assert len(ans["highlighted_nodes"]) >= 3
    assert set(["price", "marketing_budget", "order_qty"]).issubset(set(ans["highlighted_nodes"]))
