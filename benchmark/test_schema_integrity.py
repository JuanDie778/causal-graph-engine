"""Pruebas unitarias de integridad lógica y validación de esquemas (CausalGraphValidator)."""
import pytest
from cge.validator import CausalGraphValidator


@pytest.fixture
def validator():
    return CausalGraphValidator()


def test_validator_valid_schema(validator):
    valid_graph = {
        "system_id": "test_sys",
        "version": "1.1.0",
        "nodes": [
            {
                "id": "u_ext",
                "label": "External Shock",
                "category": "exogenous",
                "data_type": "continuous",
                "source_reference": {"file": "mod.py", "entity": "u", "line_range": [1, 2]}
            },
            {
                "id": "x_ctrl",
                "label": "Control Lever",
                "category": "control",
                "data_type": "continuous",
                "source_reference": {"file": "mod.py", "entity": "x", "line_range": [3, 4]},
                "bounds": {"min": 0.0, "max": 100.0}
            },
            {
                "id": "v_state",
                "label": "State Var",
                "category": "state",
                "data_type": "continuous",
                "source_reference": {"file": "mod.py", "entity": "v", "line_range": [5, 6]}
            },
            {
                "id": "j_target",
                "label": "Target Metric",
                "category": "target",
                "data_type": "continuous",
                "source_reference": {"file": "mod.py", "entity": "j", "line_range": [7, 8]}
            }
        ],
        "edges": [
            {"source": "u_ext", "target": "v_state", "type": "exact_code"},
            {"source": "x_ctrl", "target": "v_state", "type": "exact_code"},
            {"source": "v_state", "target": "j_target", "type": "exact_code"}
        ],
        "constraints": [
            {"id": "c1", "type": "hard", "expression": "x_ctrl <= 80"}
        ]
    }
    report = validator.validate(valid_graph)
    assert report.is_valid, f"Errores inesperados: {[e.message for e in report.errors]}"
    assert len(report.errors) == 0


def test_validator_detects_cycles(validator):
    cyclic_graph = {
        "system_id": "cyclic_sys",
        "version": "1.1.0",
        "nodes": [
            {"id": "a", "label": "A", "category": "state", "data_type": "continuous", "source_reference": {"file": "a.py"}},
            {"id": "b", "label": "B", "category": "state", "data_type": "continuous", "source_reference": {"file": "b.py"}}
        ],
        "edges": [
            {"source": "a", "target": "b", "type": "exact_code"},
            {"source": "b", "target": "a", "type": "exact_code"}
        ],
        "constraints": []
    }
    report = validator.validate(cyclic_graph)
    assert not report.is_valid
    assert any(err.code == "CYCLIC_GRAPH_DETECTED" for err in report.errors)


def test_validator_detects_orphan_edges(validator):
    orphan_graph = {
        "system_id": "orphan_sys",
        "version": "1.1.0",
        "nodes": [
            {"id": "a", "label": "A", "category": "state", "data_type": "continuous", "source_reference": {"file": "a.py"}}
        ],
        "edges": [
            {"source": "a", "target": "non_existent_node", "type": "exact_code"}
        ],
        "constraints": []
    }
    report = validator.validate(orphan_graph)
    assert not report.is_valid
    assert any(err.code == "ORPHAN_EDGE_TARGET" for err in report.errors)


def test_validator_detects_inverted_bounds(validator):
    inverted_graph = {
        "system_id": "inv_sys",
        "version": "1.1.0",
        "nodes": [
            {
                "id": "x",
                "label": "X",
                "category": "control",
                "data_type": "continuous",
                "source_reference": {"file": "x.py"},
                "bounds": {"min": 50.0, "max": 10.0}
            }
        ],
        "edges": [],
        "constraints": []
    }
    report = validator.validate(inverted_graph)
    assert not report.is_valid
    assert any(err.code == "INVERTED_BOUNDS" for err in report.errors)


def test_validator_detects_exogenous_with_incoming_edges(validator):
    bad_exo_graph = {
        "system_id": "bad_exo",
        "version": "1.1.0",
        "nodes": [
            {"id": "v", "label": "V", "category": "state", "data_type": "continuous", "source_reference": {"file": "v.py"}},
            {"id": "u", "label": "U", "category": "exogenous", "data_type": "continuous", "source_reference": {"file": "u.py"}}
        ],
        "edges": [
            {"source": "v", "target": "u", "type": "exact_code"}
        ],
        "constraints": []
    }
    report = validator.validate(bad_exo_graph)
    assert not report.is_valid
    assert any(err.code == "EXOGENOUS_WITH_INCOMING_EDGES" for err in report.errors)


def test_validator_passes_on_current_workspace_schema(validator):
    import json
    with open("schema.json", "r", encoding="utf-8") as f:
        real_schema = json.load(f)
    report = validator.validate(real_schema)
    assert report.is_valid, f"Errores en schema.json del workspace: {[e.message for e in report.errors]}"
