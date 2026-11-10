"""Pruebas de Aceptación de API REST (Fase 6): TestClient sobre los 3 Mundos Heterogéneos y Blindaje de Seguridad."""
import json
import os
import shutil
import tempfile
import pytest
from fastapi.testclient import TestClient

from ui.server import app, engine_state
from generator import ProceduralWorldGenerator


@pytest.fixture(scope="module")
def api_client():
    return TestClient(app)


@pytest.fixture(scope="module")
def multi_domain_dirs():
    tmp_base = tempfile.mkdtemp(prefix="cge_f6_api_")
    pgen = ProceduralWorldGenerator(base_seed=7000)

    # 1. Logística multi-etapa
    logistics_dir = os.path.join(tmp_base, "fleet_logistics")
    pgen.generate_world(1, logistics_dir, domain="fleet_logistics", clean=False, include_csv=True)

    # 2. SaaS continuo
    saas_dir = os.path.join(tmp_base, "saas_cloud")
    pgen.generate_world(2, saas_dir, domain="saas_cloud", clean=False, include_csv=True)

    # 3. Caso Base (Golden Case existente)
    base_dir = "golden_case"

    yield {
        "base_case": base_dir,
        "fleet_logistics": logistics_dir,
        "saas_cloud": saas_dir
    }

    shutil.rmtree(tmp_base, ignore_errors=True)


def test_api_world1_base_case_lifecycle(api_client, multi_domain_dirs):
    """Verifica todos los endpoints REST sobre el Caso Base (Mundo 1)."""
    base_dir = multi_domain_dirs["base_case"]

    # 1. POST /api/ingest
    res_ingest = api_client.post("/api/ingest", json={"dir_path": base_dir})
    assert res_ingest.status_code == 200
    data_ingest = res_ingest.json()
    assert data_ingest["status"] == "ok"
    assert data_ingest["nodes_count"] >= 8
    assert data_ingest["edges_count"] >= 7

    # 2. GET /api/schema
    res_schema = api_client.get("/api/schema")
    assert res_schema.status_code == 200
    schema_data = res_schema.json()
    assert "version" in schema_data
    assert len(schema_data["nodes"]) == data_ingest["nodes_count"]

    # 3. GET /api/diagnostics
    res_diag = api_client.get("/api/diagnostics")
    assert res_diag.status_code == 200
    diags = res_diag.json()
    assert diags["count"] >= 1
    assert any(d["id"] == "T1" for d in diags["diagnostics"])

    # 4. GET /api/importance
    res_imp = api_client.get("/api/importance")
    assert res_imp.status_code == 200
    imp_data = res_imp.json()
    assert "adjoints" in imp_data
    assert "elasticities" in imp_data
    assert len(imp_data["ranked_controls"]) >= 2

    # 5. GET /api/paths
    res_paths = api_client.get("/api/paths")
    assert res_paths.status_code == 200
    paths_data = res_paths.json()
    assert len(paths_data["paths"]) >= 1

    # 6. POST /api/evaluate
    controls_nominal = {n["id"]: float(n["current_value"]) for n in schema_data["nodes"] if n["category"] == "control"}
    res_eval = api_client.post("/api/evaluate", json={"controls": controls_nominal})
    assert res_eval.status_code == 200
    eval_data = res_eval.json()
    assert eval_data["target_id"] is not None
    assert isinstance(eval_data["target_value"], (int, float))

    # 7. POST /api/optimize (Óptimo neutral factible)
    res_opt = api_client.post("/api/optimize", json={})
    assert res_opt.status_code == 200
    opt_data = res_opt.json()
    assert opt_data["status"] == "FEASIBLE"
    assert opt_data["objective_value"] > 0.0

    # 8. POST /api/optimize (Objetivo infactible con diagnóstico KKT)
    res_inf = api_client.post("/api/optimize", json={"target_value": 99999999.0})
    assert res_inf.status_code == 200
    inf_data = res_inf.json()
    assert inf_data["status"] == "INFEASIBLE"
    assert len(inf_data["blocking_constraints"]) >= 1

    # 9. POST /api/optimize (CVaR con aversión al riesgo)
    res_cvar = api_client.post("/api/optimize", json={"use_cvar": True, "cvar_beta": 0.95, "risk_aversion": 0.2})
    assert res_cvar.status_code == 200
    cvar_data = res_cvar.json()
    assert cvar_data["cvar_mode"] is True
    assert cvar_data["objective_value"] > 0.0

    # 10. POST /api/copilot
    res_cop = api_client.post("/api/copilot", json={"question": "¿Cuáles son las palancas de control?"})
    assert res_cop.status_code == 200
    cop_data = res_cop.json()
    assert len(cop_data["highlighted_nodes"]) >= 2
    assert "answer" in cop_data


def test_api_world2_logistics_lifecycle(api_client, multi_domain_dirs):
    """Verifica el funcionamiento del servidor sobre el mundo de logística multi-etapa (Mundo 2)."""
    logistics_dir = multi_domain_dirs["fleet_logistics"]

    # Ingestar logística
    res_ingest = api_client.post("/api/ingest", json={"dir_path": logistics_dir})
    assert res_ingest.status_code == 200
    data_ingest = res_ingest.json()
    assert data_ingest["nodes_count"] >= 8

    # Schema y auditoría
    res_schema = api_client.get("/api/schema")
    assert res_schema.status_code == 200

    res_diag = api_client.get("/api/diagnostics")
    assert res_diag.status_code == 200

    # Optimización de logística
    res_opt = api_client.post("/api/optimize", json={})
    assert res_opt.status_code == 200
    opt_data = res_opt.json()
    assert opt_data["status"] == "FEASIBLE"
    assert opt_data["objective_value"] > 0.0


def test_api_world3_saas_cloud_lifecycle(api_client, multi_domain_dirs):
    """Verifica el funcionamiento del servidor sobre el mundo de SaaS continuo (Mundo 3)."""
    saas_dir = multi_domain_dirs["saas_cloud"]

    # Ingestar SaaS
    res_ingest = api_client.post("/api/ingest", json={"dir_path": saas_dir})
    assert res_ingest.status_code == 200
    data_ingest = res_ingest.json()
    assert data_ingest["nodes_count"] >= 8

    # Optimización SaaS
    res_opt = api_client.post("/api/optimize", json={})
    assert res_opt.status_code == 200
    opt_data = res_opt.json()
    assert opt_data["status"] == "FEASIBLE"
    assert opt_data["objective_value"] > 0.0


def test_api_source_code_and_path_traversal_protection(api_client, multi_domain_dirs):
    """Verifica la inspección de código y el blindaje riguroso contra path traversal (GET /api/source)."""
    base_dir = multi_domain_dirs["base_case"]
    api_client.post("/api/ingest", json={"dir_path": base_dir})

    schema = api_client.get("/api/schema").json()
    first_node = schema["nodes"][0]["id"]

    # 1. Fragmento válido por node_id
    res_valid = api_client.get(f"/api/source?node_id={first_node}")
    assert res_valid.status_code == 200
    src_data = res_valid.json()
    assert "code" in src_data
    assert len(src_data["code"]) > 0
    assert "line_range" in src_data

    # 2. Intento de escape con secuencias '../' (Path Traversal clásico)
    res_escape1 = api_client.get("/api/source?file=../../etc/passwd")
    assert res_escape1.status_code == 400
    assert "path traversal" in res_escape1.json()["detail"].lower()

    # 3. Intento de escape hacia directorios superiores del sistema
    res_escape2 = api_client.get("/api/source?file=..\\..\\Windows\\System32\\calc.exe")
    assert res_escape2.status_code == 400

    # 4. Intento con byte nulo
    res_escape3 = api_client.get("/api/source?file=clean.py%00.txt")
    assert res_escape3.status_code in (400, 404)

    # 5. Intento de acceder a archivos del workspace pero fuera del proyecto activo (Aislamiento de Proyecto)
    res_escape4 = api_client.get("/api/source?file=eval_reports/latest_eval.json")
    assert res_escape4.status_code in (403, 404), f"Se permitió acceder a archivo fuera del proyecto: {res_escape4.status_code}"

    # 6. Intento con ruta absoluta fuera del proyecto
    res_escape5 = api_client.get("/api/source?file=C:/Windows/System32/drivers/etc/hosts")
    assert res_escape5.status_code in (400, 403, 404)


def test_api_schema_mutation_and_validation(api_client, multi_domain_dirs):
    """Verifica la validación canónica de mutaciones de esquema (POST /api/mutate)."""
    base_dir = multi_domain_dirs["base_case"]
    api_client.post("/api/ingest", json={"dir_path": base_dir})

    schema = api_client.get("/api/schema").json()
    mutated = dict(schema)
    mutated["nodes"] = list(schema["nodes"])
    mutated["nodes"][0] = dict(schema["nodes"][0])
    mutated["nodes"][0]["label"] = "Mutated Label for API Test"

    # Mutación válida
    res_mut = api_client.post("/api/mutate", json=mutated)
    assert res_mut.status_code == 200
    assert res_mut.json()["status"] == "ok"

    # Verificar que el esquema mutado se refleja en GET /api/schema
    updated = api_client.get("/api/schema").json()
    assert updated["nodes"][0]["label"] == "Mutated Label for API Test"

    # Mutación inválida (rompe esquema canónico eliminando campos obligatorios)
    invalid_schema = {"invalid_payload": True}
    res_bad = api_client.post("/api/mutate", json=invalid_schema)
    assert res_bad.status_code in (400, 500)


def test_api_concurrency_stress_eval_and_mutate(api_client, multi_domain_dirs):
    """Verifica que el estado en memoria sea seguro ante accesos concurrentes (POST /api/evaluate y /api/mutate)."""
    import concurrent.futures
    base_dir = multi_domain_dirs["base_case"]
    api_client.post("/api/ingest", json={"dir_path": base_dir})

    schema = api_client.get("/api/schema").json()
    controls_nominal = {n["id"]: float(n["current_value"]) for n in schema["nodes"] if n["category"] == "control"}

    def run_eval():
        res = api_client.post("/api/evaluate", json={"controls": controls_nominal})
        assert res.status_code == 200
        return res.json()["target_value"]

    def run_paths():
        res = api_client.get("/api/paths")
        assert res.status_code == 200
        return len(res.json()["paths"])

    def run_mutate():
        mut = dict(schema)
        mut["nodes"] = list(schema["nodes"])
        mut["nodes"][0] = dict(schema["nodes"][0])
        mut["nodes"][0]["label"] = "Concurrent Label"
        res = api_client.post("/api/mutate", json=mut)
        assert res.status_code == 200
        return res.json()["status"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        f_evals = [executor.submit(run_eval) for _ in range(10)]
        f_paths = [executor.submit(run_paths) for _ in range(5)]
        f_mut = executor.submit(run_mutate)

        results_eval = [f.result() for f in f_evals]
        results_paths = [f.result() for f in f_paths]
        res_mut = f_mut.result()

    assert len(results_eval) == 10
    assert len(results_paths) == 5
    assert res_mut == "ok"


def test_api_robustness_and_cvar_budget_metrics(api_client, multi_domain_dirs):
    """Verifica el cálculo de márgenes presupuestarios en CVaR y manejo de errores 404 en consultas de rutas inexistentes."""
    base_dir = multi_domain_dirs["base_case"]
    api_client.post("/api/ingest", json={"dir_path": base_dir})

    # Consulta de ruta para nodo inexistente
    res_bad_target = api_client.get("/api/importance?target=nodo_que_no_existe")
    assert res_bad_target.status_code == 404

    res_bad_source = api_client.get("/api/paths?source=nodo_inexistente")
    assert res_bad_source.status_code == 404

    # CVaR con métricas completas de presupuesto
    res_cvar = api_client.post("/api/optimize", json={"use_cvar": True, "n_scenarios": 30, "risk_aversion": 0.2})
    assert res_cvar.status_code == 200
    cvar_data = res_cvar.json()
    assert "budget_spent" in cvar_data
    assert "budget_margin" in cvar_data
    assert isinstance(cvar_data["budget_margin"], (int, float))

