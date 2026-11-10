"""Suite de Pruebas Nivel 7 (N7): Compilación Universal Multi-Fuente y Motor Causal Híbrido (Fases 7-9)."""
import os
import shutil
import tempfile
import pytest
from fastapi.testclient import TestClient

from cge.ingest import ASTIngestPipeline
from cge.layer2_exact import Layer2Solver
from cge.audit import SystemAuditor
from generator import MultiSourceWorldGenerator
from ui.server import app


def test_u1_tabular_dataframe_column_lineage():
    """U-1 (Fase 7A): Lineaje automático a nivel de columna para pipelines Pandas / NumPy / Polars."""
    tmp_dir = tempfile.mkdtemp(prefix="cge_u1_tabular_")
    try:
        code = '''"""Pipeline tabular con Pandas, NumPy y Polars."""
import numpy as np

UNIT_PRICE: float = 4.0
PROMO_SPEND: float = 400.0
LOT_CAP: int = 600

UNIT_PRICE_BOUNDS = (2.0, 8.0)
PROMO_SPEND_BOUNDS = (0.0, 1500.0)
LOT_CAP_BOUNDS = (100, 1200)
MAX_BUDGET: float = 2000.0
BASE_TRAFFIC_MEAN: float = 900.0
BASE_TRAFFIC_STD: float = 50.0

def sample_base_traffic(seed=None) -> float:
    return BASE_TRAFFIC_MEAN

def run_dataframe_pipeline(df):
    df["expected_orders"] = np.maximum(0.0, df["base_traffic"] * (1.0 - 0.12 * (df["unit_price"] - 3.0)) + 8.0 * np.sqrt(np.maximum(0.0, df["promo_spend"])))
    df["lot_unit_cost"] = np.where(df["lot_cap"] < 700, 1.25, 1.00)
    df = df.assign(
        dispatched_orders=np.minimum(df["expected_orders"], df["lot_cap"]),
        overstock_fee=0.20 * np.maximum(0.0, df["lot_cap"] - df["dispatched_orders"]),
    )
    df["gross_sales"] = df["unit_price"] * df["dispatched_orders"]
    df["net_margin"] = df["gross_sales"] - (df["lot_unit_cost"] * df["lot_cap"]) - df["promo_spend"] - df["overstock_fee"]
    return df
'''
        with open(os.path.join(tmp_dir, "df_pipeline.py"), "w", encoding="utf-8") as f:
            f.write(code)

        graph = ASTIngestPipeline(tmp_dir).run()
        assert graph.validate_canonical()

        for col in ("expected_orders", "lot_unit_cost", "dispatched_orders", "overstock_fee", "gross_sales", "net_margin"):
            assert col in graph.nodes, f"Columna tabular '{col}' no extraída"
            assert graph.nodes[col].source_reference.column_name == col

        assert graph.nodes["net_margin"].category == "target"
        assert len(graph.nodes["lot_unit_cost"].discontinuities) == 1
        assert graph.nodes["lot_unit_cost"].discontinuities[0]["threshold"] == 700.0
        assert "tabular_dataframe" in graph.metadata.get("source_modalities", [])

        solver = Layer2Solver(graph=graph, source_dir=tmp_dir)
        opt = solver.optimize()
        assert opt["status"] == "FEASIBLE"
        assert opt["objective_value"] > solver.evaluate_current_state()["objective_value"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_u2_sql_query_and_ddl_lineage():
    """U-2 (Fase 7B): Transpilación de consultas SQL (SELECT, CASE WHEN, LEAST, GREATEST) y DDL CHECK."""
    tmp_dir = tempfile.mkdtemp(prefix="cge_u2_sql_")
    try:
        py_cfg = '''"""Configuración base para motor SQL."""
TARIFF_VAL: float = 3.8
AD_OUTLAY: float = 300.0
CAPACITY_UNITS: int = 500

TARIFF_VAL_BOUNDS = (2.0, 6.5)
AD_OUTLAY_BOUNDS = (0.0, 1500.0)
CAPACITY_UNITS_BOUNDS = (100, 1400)
MAX_BUDGET: float = 2200.0
INFLOW_BASE_MEAN: float = 1000.0
INFLOW_BASE_STD: float = 70.0

def sample_inflow_base(seed=None) -> float:
    return INFLOW_BASE_MEAN
'''
        sql_query = '''-- Pipeline analítico SQL con CASE WHEN, LEAST, GREATEST y CHECK
CREATE TABLE fact_operations (
    capacity_units INT,
    CONSTRAINT c_sql_min_cap CHECK (capacity_units >= 250)
);

SELECT
    GREATEST(0.0, inflow_base * (1.0 - 0.14 * (tariff_val - 3.0)) + 7.5 * SQRT(GREATEST(0.0, ad_outlay))) AS raw_requests,
    CASE WHEN capacity_units < 750 THEN 1.30 ELSE 1.00 END AS unit_supply_cost,
    LEAST(raw_requests, capacity_units) AS settled_volume,
    0.18 * GREATEST(0.0, capacity_units - settled_volume) AS unused_cap_cost,
    tariff_val * settled_volume AS gross_turnover,
    0.03 * gross_turnover AS clearing_fee,
    gross_turnover - clearing_fee - (unit_supply_cost * capacity_units) - ad_outlay - unused_cap_cost AS total_operating_surplus
FROM fact_operations;
'''
        with open(os.path.join(tmp_dir, "config_params.py"), "w", encoding="utf-8") as f:
            f.write(py_cfg)
        with open(os.path.join(tmp_dir, "settlement.sql"), "w", encoding="utf-8") as f:
            f.write(sql_query)

        graph = ASTIngestPipeline(tmp_dir).run()
        assert graph.validate_canonical()

        for col in ("raw_requests", "unit_supply_cost", "settled_volume", "unused_cap_cost", "gross_turnover", "clearing_fee", "total_operating_surplus"):
            assert col in graph.nodes, f"Columna SQL '{col}' no compilada"
            assert graph.nodes[col].source_reference.file.endswith("settlement.sql")
            assert graph.nodes[col].source_reference.column_name == col

        assert "c_sql_min_cap" in graph.constraints
        assert len(graph.nodes["unit_supply_cost"].discontinuities) == 1
        assert graph.nodes["unit_supply_cost"].discontinuities[0]["threshold"] == 750.0

        solver = Layer2Solver(graph=graph, source_dir=tmp_dir)
        ev_viol = solver.evaluate(capacity_units=150)
        assert ev_viol["violations"].get("c_sql_min_cap") is True
        assert ev_viol["is_feasible"] is False

        opt = solver.optimize()
        assert opt["status"] == "FEASIBLE"
        assert opt["controls"]["capacity_units"] >= 250
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_u3_normative_manual_constraints_and_contradictions():
    """U-3 (Fase 8): Ingesta de manuales normativos (.md/.txt), detección de contradicciones T3 y cumplimiento en optimización."""
    tmp_dir = tempfile.mkdtemp(prefix="cge_u3_normative_")
    try:
        # El código tiene TARIFF_VAL = 5.40 con cota (2.0, 6.5), pero el manual normativo prohíbe superar 4.35
        py_code = '''"""Modelo con parámetro nominal que infringe el manual regulatorio."""
import math

TARIFF_VAL: float = 5.40
PROMO_BUDGET: float = 400.0
BATCH_SIZE: int = 750

TARIFF_VAL_BOUNDS = (2.0, 6.5)
PROMO_BUDGET_BOUNDS = (0.0, 1500.0)
BATCH_SIZE_BOUNDS = (200, 1400)
MAX_BUDGET: float = 2400.0
BASE_FLOW_MEAN: float = 1100.0
BASE_FLOW_STD: float = 60.0

def sample_base_flow(seed=None) -> float:
    return BASE_FLOW_MEAN

def compute_demand_vol(base_flow: float, tariff_val: float, promo_budget: float) -> float:
    return max(0.0, base_flow * (1.0 - 0.12 * (tariff_val - 3.0)) + 8.0 * math.sqrt(max(0.0, promo_budget)))

def compute_dispatched(demand_vol: float, batch_size: int) -> float:
    return min(demand_vol, float(batch_size))

def compute_service_fee(gross_rev: float) -> float:
    return 0.04 * gross_rev

def compute_gross_rev(tariff_val: float, dispatched: float) -> float:
    return tariff_val * dispatched

def compute_net_return(gross_rev: float, service_fee: float, batch_size: int, promo_budget: float) -> float:
    return gross_rev - service_fee - (1.0 * float(batch_size)) - promo_budget
'''
        manual_txt = '''# Manual de Cumplimiento y Regulación Operativa

## Sección 1.1: Límites Tarifarios Obligatorios
En ningún caso `tariff_val` superará 4.35 según resolución vigente.
[RULE c_regulator_tariff_cap]: tariff_val <= 4.35
`batch_size` must not be less than 300.

## Sección 2.1: Comisiones
- Comisión operativa documentada: 2.0%
'''
        with open(os.path.join(tmp_dir, "system_ops.py"), "w", encoding="utf-8") as f:
            f.write(py_code)
        with open(os.path.join(tmp_dir, "reglamento_operativo.md"), "w", encoding="utf-8") as f:
            f.write(manual_txt)

        graph = ASTIngestPipeline(tmp_dir).run()
        assert graph.validate_canonical()

        assert "c_regulator_tariff_cap" in graph.constraints
        assert "normative_manual" in graph.metadata.get("source_modalities", [])

        # Debe detectar tanto la discrepancia de tasa (2% vs 4%) como la violación normativa nominal (5.40 > 4.35)
        discs = graph.metadata.get("doc_discrepancies", [])
        subtypes = {d.get("subtype", "RATE_DISCREPANCY") for d in discs}
        assert "NORMATIVE_BOUND_VIOLATION" in subtypes

        solver = Layer2Solver(graph=graph, source_dir=tmp_dir)
        cur_ev = solver.evaluate_current_state()
        assert cur_ev["violations"].get("c_regulator_tariff_cap") is True
        assert cur_ev["is_feasible"] is False

        opt = solver.optimize()
        assert opt["status"] == "FEASIBLE"
        assert opt["controls"]["tariff_val"] <= 4.35 + 1e-4
        assert "c_regulator_tariff_cap" in opt["active_constraints"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_u4_hybrid_csv_discovery_and_blackbox_surrogate():
    """U-4 (Fase 9): Descubrimiento causal híbrido desde CSV para nodos sin código y síntesis de subrogado analítico para caja negra."""
    tmp_dir = tempfile.mkdtemp(prefix="cge_u4_hybrid_")
    try:
        # El código contiene una caja negra ML (con bucle iterativo que fuerza fallback python_callable)
        py_code = '''"""Sistema híbrido con modelo de caja negra iterable."""
import math

CTRL_ALPHA: float = 3.5
CTRL_BETA: float = 400.0
CTRL_ALPHA_BOUNDS = (2.0, 6.0)
CTRL_BETA_BOUNDS = (50.0, 1200.0)
MAX_BUDGET: float = 1800.0
EXO_SIGNAL_MEAN: float = 950.0
EXO_SIGNAL_STD: float = 40.0

def sample_exo_signal(seed=None) -> float:
    return EXO_SIGNAL_MEAN

def predict_opaque_response(exo_signal: float, ctrl_alpha: float, ctrl_beta: float) -> float:
    # Modelo imperativo con estado mutable while/dict que activa python_callable y el subrogado analítico (Fase 9B)
    state_bag = {"acc": 0.0, "step": 0}
    while state_bag["step"] < 2:
        state_bag["acc"] += 0.5 * (exo_signal * (1.0 - 0.10 * (ctrl_alpha - 3.0)) + 9.0 * math.sqrt(max(0.0, ctrl_beta)))
        state_bag["step"] += 1
    return max(0.0, state_bag["acc"])

def compute_hybrid_score(ctrl_alpha: float, opaque_response: float, ctrl_beta: float) -> float:
    return ctrl_alpha * opaque_response - ctrl_beta
'''
        with open(os.path.join(tmp_dir, "hybrid_model.py"), "w", encoding="utf-8") as f:
            f.write(py_code)

        graph = ASTIngestPipeline(tmp_dir).run()
        assert graph.validate_canonical()

        surrogates = graph.metadata.get("blackbox_surrogates", {})
        assert "opaque_response" in surrogates
        surr_info = surrogates["opaque_response"]
        assert surr_info["r2_fidelity"] > 0.99
        assert "ctrl_alpha" in surr_info["local_gradients"]
        assert "ctrl_beta" in surr_info["local_gradients"]

        # Verificar round-trip JSON (schema.json) y resolución en Layer2Solver mediante el subrogado analítico
        from cge.schema import CausalGraph
        reloaded_graph = CausalGraph.from_dict(graph.to_dict())
        solver_surr = Layer2Solver(graph=reloaded_graph)
        cur_state = solver_surr.evaluate_current_state()
        assert cur_state["opaque_response"] > 0
        opt_surr = solver_surr.optimize()
        assert opt_surr["status"] == "FEASIBLE"
        assert opt_surr["objective_value"] >= cur_state["objective_value"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.parametrize("world_id", [1, 2, 3])
def test_u5_end_to_end_multisource_fusion_vs_oracle(world_id: int):
    """U-5 (Fases 7-9): Fusión universal de 4 fuentes (Pandas + SQL + Manual Normativo + CSV) frente a Oráculo Independiente."""
    tmp_dir = tempfile.mkdtemp(prefix=f"cge_u5_ms_{world_id}_")
    try:
        gen = MultiSourceWorldGenerator(base_seed=7000)
        spec = gen.generate_multisource_world(world_id, tmp_dir, clean=False, discover_demand_from_csv=True)
        oracle_res = spec["oracle"]

        graph = ASTIngestPipeline(tmp_dir).run()
        assert graph.validate_canonical()

        modalities = set(graph.metadata.get("source_modalities", []))
        assert {"tabular_dataframe", "sql_query", "normative_manual", "empirical_csv_surrogate"}.issubset(modalities)

        # Verificar que raw_inflow_demand fue descubierto automáticamente desde data/telemetry_observations.csv
        assert "raw_inflow_demand" in graph.nodes
        assert graph.nodes["raw_inflow_demand"].source_reference.file.endswith("data/telemetry_observations.csv")

        # Verificar auditoría multi-fuente (T1, T3, T4, T5, T6, T7)
        solver = Layer2Solver(graph=graph, source_dir=tmp_dir)
        auditor = SystemAuditor(graph=graph, solver=solver, source_dir=tmp_dir)
        diags = auditor.audit_all()
        diag_ids = {d["id"] for d in diags}
        for req_t in ("T1", "T3", "T4", "T5", "T6", "T7"):
            assert req_t in diag_ids, f"Falta diagnóstico {req_t} en mundo multi-fuente {world_id}"

        # Verificar optimización restringida por el manual normativo y SQL frente al oráculo independiente
        opt = solver.optimize()
        assert opt["status"] == "FEASIBLE"
        assert opt["controls"]["ctrl_rate"] <= spec["normative_rate_cap"] + 1e-3
        assert opt["controls"]["ctrl_slots"] >= spec["sql_min_slots"]

        # Evaluar los controles prescritos por el solver en el oráculo verdadero independiente
        oracle_obj = oracle_res["optimal_objective"]
        from generator import MultiSourceWorldOracle
        true_eval_of_prescription = MultiSourceWorldOracle(spec).evaluate_world(
            opt["controls"]["ctrl_rate"],
            opt["controls"]["ctrl_boost"],
            opt["controls"]["ctrl_slots"],
        )
        rel_gap = abs(oracle_obj - true_eval_of_prescription["net_yield_score"]) / max(1.0, abs(oracle_obj))
        assert rel_gap < 0.01, f"Gap de optimalidad multi-fuente ({rel_gap:.4%}) excede el 1%"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_u6_multisource_clean_mode_zero_false_positives_and_api():
    """U-6: Modo limpio (0 falsos positivos T1-T4) e inspección multi-fuente vía REST API (/api/ingest, /api/source)."""
    tmp_dir = tempfile.mkdtemp(prefix="cge_u6_clean_api_")
    try:
        gen = MultiSourceWorldGenerator(base_seed=7500)
        spec = gen.generate_multisource_world(10, tmp_dir, clean=True, discover_demand_from_csv=True)

        client = TestClient(app)
        res_ing = client.post("/api/ingest", json={"target_dir": tmp_dir})
        assert res_ing.status_code == 200
        ing_data = res_ing.json()
        assert "sql_query" in ing_data.get("source_modalities", [])

        # En modo clean no debe haber falsas alarmas T1, T2, T3 ni T4
        res_diag = client.get("/api/diagnostics")
        assert res_diag.status_code == 200
        diag_ids = {d["id"] for d in res_diag.json()["diagnostics"]}
        assert diag_ids.isdisjoint({"T1", "T2", "T3", "T4"})

        # Inspeccionar código fuente de un nodo SQL, un nodo DataFrame y un nodo CSV
        for nid, expected_ext in [
            ("served_volume", ".sql"),
            ("idle_penalty", ".py"),
            ("raw_inflow_demand", ".csv"),
        ]:
            res_src = client.get(f"/api/source?node_id={nid}")
            assert res_src.status_code == 200
            src_payload = res_src.json()
            assert src_payload["file"].endswith(expected_ext)
            assert src_payload["column_name"] == nid
            assert nid in src_payload["code"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_u7_adversarial_edge_cases_sql_normative_csv_blackbox():
    """U-7: Casos de borde adversariales:
    - SQL CASE WHEN multi-rama + CHECK con paréntesis anidados, prefijos de tabla y conjunción AND.
    - Manual normativo con coeficientes líderes en [RULE], cotas repetidas en distintas secciones (conflicto T3)
      y diagnóstico KKT de infactibilidad por reglas mutuamente excluyentes.
    - Invarianza al orden de columnas en descubrimiento causal desde CSV.
    - Serialización JSON round-trip (CausalGraph.from_dict) de subrogado analítico de caja negra.
    """
    import math
    from cge.schema import CausalGraph, Node, SourceReference
    from cge.universal_compiler import (
        SQLLineageTranspiler,
        NormativeDocTranspiler,
        discover_missing_mechanisms_from_csv,
    )

    # 1. SQL CASE WHEN multi-rama y CHECK con paréntesis anidados + prefijo de tabla + AND
    sql_t = SQLLineageTranspiler()
    cols, cons = sql_t.parse_sql_text(
        """
        CREATE TABLE tbl (
            cap INT,
            tariff FLOAT,
            CONSTRAINT c_complex CHECK (((tbl.cap + 10) >= 250) AND (tbl.tariff <= 5.5))
        );
        SELECT
            CASE WHEN cap < 400 THEN 1.50 WHEN cap < 750 THEN 1.25 ELSE 1.00 END AS unit_cost
        FROM tbl;
        """,
        "complex.sql",
    )
    assert "unit_cost" in cols
    assert len(cols["unit_cost"]["discontinuities"]) == 2
    assert {c["expression"] for c in cons} == {"(cap + 10) >= 250", "tariff <= 5.5"}

    # 2. Manual normativo: coeficientes líderes en [RULE], conflicto entre secciones y dominio infactible
    tmp_dir = tempfile.mkdtemp(prefix="cge_u7_edge_")
    try:
        py_code = """import math
TARIFF_VAL: float = 4.60
PROMO_BUDGET: float = 400.0
BATCH_SIZE: int = 750
TARIFF_VAL_BOUNDS = (2.0, 6.5)
PROMO_BUDGET_BOUNDS = (0.0, 1500.0)
BATCH_SIZE_BOUNDS = (200, 1400)
MAX_BUDGET: float = 2400.0
BASE_FLOW_MEAN: float = 1100.0
BASE_FLOW_STD: float = 60.0
def sample_base_flow(seed=None) -> float: return BASE_FLOW_MEAN
def compute_demand_vol(base_flow: float, tariff_val: float, promo_budget: float) -> float:
    return max(0.0, base_flow * (1.0 - 0.12 * (tariff_val - 3.0)) + 8.0 * math.sqrt(max(0.0, promo_budget)))
def compute_dispatched(demand_vol: float, batch_size: int) -> float:
    return min(demand_vol, float(batch_size))
def compute_gross_rev(tariff_val: float, dispatched: float) -> float:
    return tariff_val * dispatched
def compute_net_return(gross_rev: float, batch_size: int, promo_budget: float) -> float:
    return gross_rev - (1.0 * float(batch_size)) - promo_budget
"""
        md_code = """# Manual con múltiples secciones y coeficientes líderes
## Sección 1
En ningún caso `tariff_val` superará 5.00.
[RULE c_weighted_load]: 0.5 * promo_budget + 1.2 * batch_size <= 1400
## Anexo B
El límite máximo normativo de `tariff_val` es 4.20.
"""
        with open(os.path.join(tmp_dir, "ops.py"), "w", encoding="utf-8") as f:
            f.write(py_code)
        with open(os.path.join(tmp_dir, "manual.md"), "w", encoding="utf-8") as f:
            f.write(md_code)

        g = ASTIngestPipeline(tmp_dir).run()
        assert g.constraints["c_weighted_load"].expression == "0.5 * promo_budget + 1.2 * batch_size <= 1400"
        discs = g.metadata.get("doc_discrepancies", [])
        assert any(d.get("subtype") == "NORMATIVE_RULE_CONFLICT" for d in discs)

        s = Layer2Solver(graph=g, source_dir=tmp_dir)
        opt = s.optimize()
        assert opt["status"] == "FEASIBLE"
        # Debe respetar la cota más estricta del Anexo B (4.20) y la restricción acoplada c_weighted_load
        assert opt["controls"]["tariff_val"] <= 4.20 + 1e-4
        assert 0.5 * opt["controls"]["promo_budget"] + 1.2 * opt["controls"]["batch_size"] <= 1400.0 + 1e-3

        # Ahora agregar una cota inferior contradictoria (tariff_val >= 4.80) y verificar diagnóstico de infactibilidad
        with open(os.path.join(tmp_dir, "manual.md"), "a", encoding="utf-8") as f:
            f.write("\n[RULE c_floor_conflict]: tariff_val >= 4.80\n")
        g_inf = ASTIngestPipeline(tmp_dir).run()
        s_inf = Layer2Solver(graph=g_inf, source_dir=tmp_dir)
        opt_inf = s_inf.optimize()
        assert opt_inf["status"] == "INFEASIBLE"
        assert "blocking_constraints" in opt_inf and len(opt_inf["blocking_constraints"]) >= 1
        assert "explanation" in opt_inf

        # 3. Invarianza al orden de columnas en descubrimiento desde CSV
        csv_path = os.path.join(tmp_dir, "reordered.csv")
        lines = ["ctrl_boost,ctrl_rate,exo_inflow,raw_inflow_demand"]
        for i in range(1, 65):
            u = 900.0 + (i * 11) % 300
            p = 2.5 + (i % 8) * 0.4
            m = 100.0 + (i % 15) * 80.0
            dem = u * (1.0 - 0.14 * (p - 3.0)) + 8.5 * math.sqrt(m)
            lines.append(f"{m},{p},{u},{dem:.4f}")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        sr = SourceReference(file="x.py", entity="x", line_range=[1, 1])
        known_nodes = {
            "exo_inflow": Node(id="exo_inflow", label="U", category="exogenous", data_type="continuous", source_reference=sr),
            "ctrl_boost": Node(id="ctrl_boost", label="M", category="control", data_type="continuous", source_reference=sr),
            "ctrl_rate": Node(id="ctrl_rate", label="P", category="control", data_type="continuous", source_reference=sr),
        }
        mechs, meta = discover_missing_mechanisms_from_csv(csv_path, "reordered.csv", {"raw_inflow_demand"}, known_nodes, {})
        assert meta[0]["selected_basis"] == "sqrt"
        assert meta[0]["source"] == "ctrl_boost"
        assert meta[0]["r_squared"] > 0.999
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

