"""Nivel N1: Ingesta Estática AST y Estructuración Canónica (I-1 a I-10)."""
import json
import os
import pytest
from cge.ingest import run_ingestion_pipeline
from cge.empirical import estimate_marketing_coefficient_from_csv


@pytest.fixture(scope="module")
def ingested_graph():
    return run_ingestion_pipeline("golden_case", "schema.json")


@pytest.fixture(scope="module")
def expected_graph():
    with open("oracle/expected_graph.json", "r", encoding="utf-8") as f:
        return json.load(f)


def test_i1_single_command_execution(ingested_graph):
    """I-1: El pipeline se ejecuta con un solo comando sobre golden_case/, sin etiquetado manual."""
    assert ingested_graph is not None
    assert os.path.exists("schema.json")


def test_i2_canonical_schema_validation(ingested_graph):
    """I-2: Genera un schema.json que valida contra el JSON Schema canónico (100%)."""
    assert ingested_graph.validate_canonical("schema/canonical_schema.json")


def test_i3_node_recall(ingested_graph, expected_graph):
    """I-3: Recall de nodos frente a expected_graph.json: 100% (12/12)."""
    ingested_node_ids = set(ingested_graph.nodes.keys())
    expected_node_ids = {n["id"] for n in expected_graph["nodes"]}
    print(f"\nI-3 Nodos encontrados: {len(ingested_node_ids)}/12")
    assert expected_node_ids.issubset(ingested_node_ids), f"Faltan nodos: {expected_node_ids - ingested_node_ids}"
    assert len(expected_node_ids) == 12


def test_i4_edge_precision(ingested_graph, expected_graph):
    """I-4: Precisión de aristas (sin aristas inventadas) >= 95%."""
    ingested_edges = {(e.source, e.target) for e in ingested_graph.edges}
    expected_edges = {(e["source"], e["target"]) for e in expected_graph["edges"]}
    common = ingested_edges.intersection(expected_edges)
    precision = len(common) / len(ingested_edges) if ingested_edges else 0.0
    print(f"\nI-4 Precisión de aristas: {precision:.4f} ({len(common)}/{len(ingested_edges)})")
    assert precision >= 0.95


def test_i5_edge_recall_to_target(ingested_graph, expected_graph):
    """I-5: Recall de aristas hacia J = 100%."""
    ingested_to_target = {(e.source, e.target) for e in ingested_graph.edges if e.target == "profit"}
    expected_to_target = {(e["source"], e["target"]) for e in expected_graph["edges"] if e["target"] == "profit"}
    print(f"\nI-5 Aristas hacia profit: {len(ingested_to_target)}/{len(expected_to_target)}")
    assert expected_to_target.issubset(ingested_to_target)


def test_i6_node_category_classification(ingested_graph, expected_graph):
    """I-6: Clasificación U / X / V / target correcta (100%)."""
    expected_cats = {n["id"]: n["category"] for n in expected_graph["nodes"]}
    for nid, exp_cat in expected_cats.items():
        assert nid in ingested_graph.nodes, f"Nodo {nid} no encontrado"
        actual_cat = ingested_graph.nodes[nid].category
        assert actual_cat == exp_cat, f"Categoría incorrecta para {nid}: {actual_cat} != {exp_cat}"


def test_i7_traceability(ingested_graph):
    """I-7: Trazabilidad: cada nodo y arista tiene file + line_range o columna CSV correcta (100%)."""
    for nid, node in ingested_graph.nodes.items():
        src = node.source_reference
        assert src.file is not None, f"Nodo {nid} sin archivo de origen"
        assert os.path.exists(src.file), f"Archivo no encontrado: {src.file}"
        assert src.line_range is not None and len(src.line_range) == 2, f"Nodo {nid} sin rango de líneas"
        assert src.line_range[0] <= src.line_range[1], f"Rango de líneas invertido en {nid}"
        with open(src.file, "r", encoding="utf-8") as f:
            file_lines = f.readlines()
        snippet = "".join(file_lines[src.line_range[0] - 1 : src.line_range[1]])
        assert src.entity in snippet or src.entity.lower() in snippet.lower(), (
            f"Entidad {src.entity} del nodo {nid} no aparece en {src.file} líneas {src.line_range}"
        )


def test_i8_edge_type_classification(ingested_graph):
    """I-8: Tipo de arista correcto (exact_code / discrete_rule / empirical_data) (100% en T5 y T7)."""
    # T5: order_qty -> unit_cost debe ser discrete_rule
    t5_edges = [e for e in ingested_graph.edges if e.source == "order_qty" and e.target == "unit_cost"]
    assert any(e.type == "discrete_rule" for e in t5_edges), "T5 debe ser de tipo discrete_rule"

    # T7: marketing_budget -> demand debe incluir empirical_data
    t7_edges = [e for e in ingested_graph.edges if e.source == "marketing_budget" and e.target == "demand" and e.type == "empirical_data"]
    assert len(t7_edges) >= 1, "T7 debe contener una arista empirical_data"


def test_i9_determinism():
    """I-9: Determinismo: 3 ejecuciones dan un schema.json idéntico (ignorando timestamps)."""
    g1 = run_ingestion_pipeline("golden_case", "temp1.json").to_dict()
    g2 = run_ingestion_pipeline("golden_case", "temp2.json").to_dict()
    g3 = run_ingestion_pipeline("golden_case", "temp3.json").to_dict()
    for temp_f in ["temp1.json", "temp2.json", "temp3.json"]:
        if os.path.exists(temp_f):
            os.remove(temp_f)
    assert g1 == g2 == g3, "Las 3 ejecuciones generaron grafos diferentes"


def test_i10_empirical_coefficient_estimation():
    """I-10: Coeficiente empírico de marketing estimado desde el CSV (Error <= 10% sobre el real)."""
    res = estimate_marketing_coefficient_from_csv("golden_case/data/sales_history.csv")
    print(f"\nI-10 Beta estimado: {res['estimated_coefficient']}, Real: {res['ground_truth']}, Error: {res['relative_error']*100:.2f}%")
    assert res["passes_threshold"], f"Error relativo ({res['relative_error']}) supera el 10%"
