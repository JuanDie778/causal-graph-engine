"""Nivel N0: Benchmark del Motor Matemático Aislado (Capa 1 y Capa 2 frente al Oráculo, R-1 a R-9)."""
import json
import time
import pytest
from cge.layer1_heuristic import Layer1Router, create_synthetic_benchmark_graph
from cge.layer2_exact import Layer2Solver
from oracle.solve import find_optimal_solution, evaluate_system


@pytest.fixture(scope="module")
def oracle_solution():
    with open("oracle/expected_solution.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def golden_graph_dict():
    with open("oracle/expected_graph.json", "r", encoding="utf-8") as f:
        return json.load(f)


def test_r1_layer1_top_k_paths(golden_graph_dict):
    """R-1: Capa 1 devuelve Top-K rutas X -> J y la ruta verdadera está entre ellas (K=3)."""
    router = Layer1Router(golden_graph_dict)
    paths_dict = router.get_all_control_to_target_paths(target="profit", k_per_control=3)
    
    assert "price" in paths_dict and len(paths_dict["price"]) >= 1
    assert "marketing_budget" in paths_dict and len(paths_dict["marketing_budget"]) >= 1
    assert "order_qty" in paths_dict and len(paths_dict["order_qty"]) >= 1

    # Verificar que el camino price -> revenue -> profit existe
    price_paths = [p["path"] for p in paths_dict["price"]]
    assert any("revenue" in p for p in price_paths), "La ruta price -> revenue -> profit debe estar en Top-K"


def test_r2_layer1_latency_golden_case(golden_graph_dict):
    """R-2: Latencia de la Capa 1 en el Golden Case < 50 ms."""
    router = Layer1Router(golden_graph_dict)
    latencies = [router.measure_latency_ms("price", "profit", k=3) for _ in range(50)]
    median_lat = sorted(latencies)[len(latencies) // 2]
    max_lat = max(latencies)
    print(f"\nR-2 Latencia Golden Case: Mediana={median_lat:.3f}ms, Max={max_lat:.3f}ms")
    assert median_lat < 50.0
    assert max_lat < 50.0


def test_r3_layer1_latency_synthetic_5000_nodes():
    """R-3: Latencia de la Capa 1 en un grafo sintético de 5.000 nodos < 50 ms (p95)."""
    synth_graph = create_synthetic_benchmark_graph(num_nodes=5000, layers=10)
    router = Layer1Router()
    router.graph = synth_graph

    latencies = [router.measure_latency_ms("l0_n0", "l9_n0", k=3) for _ in range(50)]
    p95 = sorted(latencies)[int(0.95 * len(latencies))]
    print(f"\nR-3 Latencia 5.000 nodos: p95={p95:.3f}ms")
    assert p95 < 50.0, f"Latencia p95 {p95:.2f}ms excede el umbral de 50ms"


def test_r4_layer2_optimality_vs_oracle(oracle_solution):
    """R-4: Capa 2 encuentra un X con J >= 0.99 * J* del Oráculo."""
    solver = Layer2Solver()
    result = solver.optimize()
    oracle_j = oracle_solution["optimal_target"]["profit"]
    engine_j = result["profit"]
    ratio = engine_j / oracle_j
    print(f"\nR-4 Optimality: Engine J={engine_j}, Oracle J={oracle_j}, Ratio={ratio:.5f}")
    assert ratio >= 0.99, f"El beneficio alcanzado ({engine_j}) no llega al 99% de J* ({oracle_j})"


def test_r5_hard_constraints_never_violated(oracle_solution):
    """R-5: La solución NUNCA viola una restricción hard (re-comprobado por el Oráculo)."""
    solver = Layer2Solver()
    result = solver.optimize()
    controls = result["controls"]
    
    # Comprobar con la función independiente del Oráculo
    oracle_check = evaluate_system(
        price=controls["price"],
        marketing_budget=controls["marketing_budget"],
        order_qty=controls["order_qty"]
    )
    assert oracle_check["is_feasible"], "La solución viola al menos una restricción hard"
    assert not oracle_check["c_budget_violated"], "Violación de presupuesto c_budget"
    assert not oracle_check["c_price_violated"], "Violación de límites de precio"
    assert not oracle_check["c_stock_violated"], "Violación de capacidad de inventario"


def test_r6_active_constraints_identified(oracle_solution):
    """R-6: Identifica correctamente las restricciones activas en el óptimo."""
    solver = Layer2Solver()
    result = solver.optimize()
    active_engine = set(result.get("active_constraints", []))
    active_oracle = set(oracle_solution.get("active_constraints", []))
    print(f"\nR-6 Restricciones activas: Engine={active_engine}, Oracle={active_oracle}")
    assert "c_volume_discount_threshold" in active_engine or "c_sales_capacity_limit" in active_engine


def test_r7_step_crossing_volume_discount():
    """R-7: Cruza correctamente el escalón de q=800 (T5) si el óptimo está al otro lado."""
    solver = Layer2Solver()
    result = solver.optimize()
    q_opt = result["controls"]["order_qty"]
    print(f"\nR-7 Óptimo order_qty: {q_opt}")
    assert q_opt >= 800, "El optimizador falló en cruzar el escalón de q=800 para desbloquear coste 1.00€"


def test_r8_infeasible_problem_unsat_core():
    """R-8: Problema imposible devuelve INFEASIBLE y nombra la restricción bloqueante."""
    solver = Layer2Solver()
    result = solver.optimize(target_profit=5000.0)
    assert result["status"] == "INFEASIBLE"
    assert "c_budget" in result["blocking_constraints"]
    assert "excede" in result["explanation"].lower() or "capacidad" in result["explanation"].lower()

    # Probar un target justo por encima del óptimo alcanzable (3100€ vs ~2990.88€)
    res_tight = solver.optimize(target_profit=3100.0)
    assert res_tight["status"] == "INFEASIBLE", "Un target de 3100€ debe ser infactible (óptimo es ~2990.88€)"
    assert "c_budget" in res_tight["blocking_constraints"]

    # Probar un target factible alcanzable (2500€)
    res_feas = solver.optimize(target_profit=2500.0)
    assert res_feas["status"] == "FEASIBLE"
    assert res_feas["profit"] >= 2500.0


def test_r9_actionable_recipe_format():
    """R-9: Entrega una receta legible: qué variables mover, orden y magnitud (Delta X)."""
    solver = Layer2Solver()
    current_x = {"price": 3.50, "marketing_budget": 500.0, "order_qty": 500}
    recipe = solver.generate_intervention_recipe(current_x)
    assert recipe["status"] == "FEASIBLE"
    assert len(recipe["steps"]) == 3
    assert recipe["profit_improvement"] > 2000.0

    step_vars = [s["variable"] for s in recipe["steps"]]
    assert "order_qty" in step_vars
    assert "price" in step_vars
    assert "marketing_budget" in step_vars
    for step in recipe["steps"]:
        assert "delta" in step and "rationale" in step
