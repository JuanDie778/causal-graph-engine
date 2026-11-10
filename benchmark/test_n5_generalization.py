"""Nivel N5: Benchmark de Generalización, Mutación y Prueba Anti-Memorización (G-1 a G-4)."""
import os
import shutil
import pytest
from generator import GoldenCaseGenerator, mutate_constant
from cge.ingest import run_ingestion_pipeline
from cge.audit import SystemAuditor
from cge.layer2_exact import Layer2Solver


@pytest.fixture(scope="module")
def variants_dir():
    base_dir = "benchmark_variants"
    os.makedirs(base_dir, exist_ok=True)
    yield base_dir
    # Cleanup after test suite completes
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir, ignore_errors=True)


def test_g1_and_g2_20_variants_generation_and_pipeline(variants_dir):
    """G-1 & G-2: generator.py crea 20 variantes y las etapas 1-3 pasan en >= 18/20."""
    gen = GoldenCaseGenerator(base_seed=42)
    success_count = 0
    total_variants = 20

    print(f"\n--- Ejecutando Benchmark G-1 y G-2 sobre {total_variants} variantes ---")
    for v_id in range(1, total_variants + 1):
        v_path = os.path.join(variants_dir, f"variant_{v_id}")
        meta = gen.generate_variant(variant_id=v_id, output_dir=v_path, clean=False)

        try:
            # Etapa 1: Ingesta
            schema_path = os.path.join(v_path, "schema.json")
            graph = run_ingestion_pipeline(v_path, schema_path)
            assert len(graph.nodes) >= 10, "Faltan nodos en variante"

            # Etapa 2: Auditoría (debe detectar al menos una patología plantada T2, T3, T4 o T5)
            auditor = SystemAuditor(v_path, graph)
            diags = auditor.audit_all()
            assert len(diags) >= 1, "Debe detectar patologías en variante con trampas"

            # Etapa 3: Resolución Capa 2 usando parámetros específicos de la variante
            solver = Layer2Solver.from_dir(v_path)
            sol = solver.optimize()
            assert sol["status"] == "FEASIBLE"
            assert sol["profit"] > 0
            # Verificar consistencia con los parámetros reales de la variante
            assert abs(solver.base_demand - meta["base_demand_mean"]) < 1e-4
            assert abs(solver.fee_rate - meta["code_fee_rate"]) < 1e-4
            assert solver.volume_threshold == meta["volume_threshold"]

            success_count += 1
        except Exception as e:
            print(f"Fallo en variante {v_id}: {e}")

    print(f"Variantes exitosas: {success_count}/{total_variants}")
    assert success_count >= 18, f"Solo pasaron {success_count}/20 variantes (umbral: >= 18/20)"


def test_g3_mutation_testing(variants_dir):
    """G-3: Prueba de mutación: alterar una constante (comisión 0.03 -> 0.05) altera la solución según el oráculo."""
    gen = GoldenCaseGenerator(base_seed=99)
    v_path = os.path.join(variants_dir, "mutation_variant")
    meta = gen.generate_variant(variant_id=99, output_dir=v_path, clean=True)

    # 1. Estado base
    graph_base = run_ingestion_pipeline(v_path, os.path.join(v_path, "schema_base.json"))
    solver_base = Layer2Solver.from_dir(v_path)
    sol_base = solver_base.optimize()

    # 2. Mutar constante de comisión en finance.py (de code_fee_rate a un valor mayor +0.02)
    fin_file = os.path.join(v_path, "finance.py")
    orig_fee_str = f"return {meta['code_fee_rate']} * revenue"
    new_fee_rate = round(meta['code_fee_rate'] + 0.02, 3)
    new_fee_str = f"return {new_fee_rate} * revenue"
    mutate_constant(fin_file, orig_fee_str, new_fee_str)

    # 3. Re-ingestar, auditar y resolver
    graph_mutated = run_ingestion_pipeline(v_path, os.path.join(v_path, "schema_mutated.json"))

    # Comprobar que el schema cambió y refleja la nueva comisión
    fee_edges_base = [e for e in graph_base.edges if e.source == "revenue" and e.target == "fees"]
    fee_edges_mutated = [e for e in graph_mutated.edges if e.source == "revenue" and e.target == "fees"]
    assert len(fee_edges_mutated) == 1
    assert abs(fee_edges_mutated[0].sensitivity_weight - new_fee_rate) < 1e-4, "El schema no actualizó la arista de comisión"

    # Resolver el problema mutado
    solver_mutated = Layer2Solver.from_dir(v_path)
    sol_mutated = solver_mutated.optimize()

    # El incremento en comisiones DEBE reducir el beneficio óptimo
    assert sol_mutated["profit"] < sol_base["profit"], f"El beneficio no disminuyó tras aumentar comisiones ({sol_mutated['profit']} vs {sol_base['profit']})"
    delta_profit = sol_base["profit"] - sol_mutated["profit"]
    print(f"\nG-3 Impacto de la mutación en beneficio: -{delta_profit:.2f}€ (Base: {sol_base['profit']}€ -> Mutado: {sol_mutated['profit']}€)")
    assert delta_profit > 5.0, "La magnitud del cambio en la solución es insignificante o errónea"

    # Verificar que el incremento en comisión generó la alerta T3 (discrepancia entre doc y código)
    auditor_mutated = SystemAuditor(v_path, graph_mutated)
    diags_mutated = auditor_mutated.audit_all()
    assert any(d["id"] == "T3" for d in diags_mutated), "La mutación no fue detectada por el auditor"


def test_g4_clean_variant_zero_false_positives(variants_dir):
    """G-4: Variante sana: un Golden Case sin trampas produce 0 diagnósticos de alerta."""
    gen = GoldenCaseGenerator(base_seed=777)
    clean_path = os.path.join(variants_dir, "clean_golden_case")
    meta = gen.generate_variant(variant_id=777, output_dir=clean_path, clean=True)

    graph_clean = run_ingestion_pipeline(clean_path, os.path.join(clean_path, "schema_clean.json"))
    auditor_clean = SystemAuditor(clean_path, graph_clean)
    diags_clean = auditor_clean.audit_all()

    # Filtramos alertas críticas y warnings de patología (T1, T2, T3, T4)
    critical_alerts = [d for d in diags_clean if d.get("id") in ["T1", "T2", "T3", "T4"]]
    print(f"\nG-4 Diagnósticos en variante sana: {len(critical_alerts)} (Esperado: 0)")
    assert len(critical_alerts) == 0, f"Falsos positivos detectados en variante sana: {critical_alerts}"
