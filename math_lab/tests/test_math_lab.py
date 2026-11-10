"""Suite de pruebas rigurosas de verificación matemática para `math_lab` (Exp 01 a Exp 05)."""
import sympy as sp
import pytest

from math_lab.experiments.exp01_symbolic_scm_and_routing import (
    ASTFunctionToSymPy,
    SymbolicSCM,
    TropicalMaxProductRouter,
    demonstrate_layer1_harmonic_sum_bug
)
from math_lab.experiments.exp02_universal_minlp_solver import (
    UniversalLayer2Solver,
    compare_legacy_vs_universal_on_variants
)
from math_lab.experiments.exp03_stochastic_cvar_optimizer import (
    StochasticCVaROptimizer,
    run_jensen_and_cvar_experiment
)
from math_lab.experiments.exp04_symbolic_auditor_and_econometrics import (
    FormalSymbolicAuditor,
    JointStructuralEconometricEstimator,
    demonstrate_t2_symbolic_vs_naive_counterexamples,
    detect_algebraic_path_cancellation_demo
)
from math_lab.experiments.exp05_advanced_edge_cases_mpcc_mdp_iv import (
    run_multi_kink_scalability_experiment,
    TemporalCyclicSCMAndMPCSolver,
    CausalEndogeneityIV2SLSEstimator
)


def test_exp01_ast_to_sympy_scm_zero_hardcoding():
    """Verifica que SymbolicSCM extrae los 12 nodos, expresiones simbólicas y estado nominal exacto desde AST."""
    scm = SymbolicSCM("golden_case")
    assert len(scm.node_meta) == 12
    assert "profit" in scm.mechanisms
    assert "demand" in scm.mechanisms
    assert "unit_cost" in scm.mechanisms

    # Verificar evaluación nominal exacta (p=3.5, m=500, q=500, d0=1000)
    state = scm.evaluate({"price": 3.50, "marketing_budget": 500.0, "order_qty": 500, "base_demand": 1000.0})
    assert abs(state["demand"] - 1103.8835) < 1e-2
    assert abs(state["sales"] - 500.0) < 1e-6
    assert abs(state["profit"] - 597.50) < 1e-4


def test_exp01_bounded_loops_augassign_and_step_discontinuity_separation():
    """Verifica que ASTFunctionToSymPy soporta bucles acotados, +=, desempaquetado y que Reverse-Mode AD
    separa discontinuidades de salto de orden 0 sin explosión artificial de Dirac (800001.5369)."""
    import ast
    src = """
def tiered_mechanism(x, y):
    acc, scale = 0.0, 1.0
    for w in [0.5, 1.5, 2.0]:
        acc += w * x
    if 10.0 <= x <= 50.0 and y > 0.0:
        return acc - y
    return acc
"""
    fn_node = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)][0]
    expr = ASTFunctionToSymPy().compile_function(fn_node)
    fn = sp.lambdify([sp.Symbol("x"), sp.Symbol("y")], expr, modules=["numpy", "math"])
    assert abs(float(fn(20.0, 5.0)) - (4.0 * 20.0 - 5.0)) < 1e-9
    assert abs(float(fn(5.0, 5.0)) - (4.0 * 5.0)) < 1e-9

    scm = SymbolicSCM("golden_case")
    p_opt = 244664.0 / 45000.0
    m_opt = ((150.0 * p_opt - 650.0) / 8.0) ** 2
    opt_inputs = {"price": p_opt, "marketing_budget": m_opt, "order_qty": 800, "base_demand": 1000.0}
    ad_res = scm.compute_reverse_mode_ad_jacobian(opt_inputs)

    # El adjunto suave respecto a order_qty en el punto medio de Clarke no debe explotar a ~800,001:
    assert abs(ad_res["adjoints_dJ_dv"]["order_qty"] - 1.5869) < 0.01
    step_info = ad_res["discrete_step_sensitivities"]["order_qty"]
    assert abs(step_info["delta_J_left_step"] - 164.0739) < 0.05
    assert abs(step_info["delta_J_right_step"] - (-1.20)) < 1e-4


def test_exp01_reverse_mode_ad_and_clarke_kink():
    """Verifica el Jacobiano en modo inverso O(|V|+|E|) y la detección del kink de Clarke en el óptimo."""
    scm = SymbolicSCM("golden_case")
    p_opt = 244664.0 / 45000.0
    m_opt = ((150.0 * p_opt - 650.0) / 8.0) ** 2
    opt_inputs = {"price": p_opt, "marketing_budget": m_opt, "order_qty": 800, "base_demand": 1000.0}
    ad_res = scm.compute_reverse_mode_ad_jacobian(opt_inputs)

    # En d* = q* = 800, sales = min(d, q) tiene subdiferencial de Clarke [0, 1] respecto a demand y order_qty
    clarke_d_sales = ad_res["clarke_reports"][("demand", "sales")]
    assert clarke_d_sales["clarke_subdifferential"] == [0.0, 1.0]
    assert clarke_d_sales["is_kink_active"] is True
    assert abs(clarke_d_sales["kink_asymmetry"] - 1.0) < 1e-4

    # brand_awareness debe tener adjunto exactamente 0.0
    assert ad_res["adjoints_dJ_dv"]["brand_awareness"] == 0.0


def test_exp01_tropical_router_fixes_harmonic_sum_bug():
    """Demuestra el contraejemplo donde Layer1Router falla por sumar 1/|w| y TropicalMaxProductRouter acierta."""
    res = demonstrate_layer1_harmonic_sum_bug()
    assert res["legacy_is_wrong"] is True
    assert res["legacy_layer1_selected_path"] == ["X", "B", "J"]
    assert res["tropical_is_exact"] is True
    assert res["tropical_router_selected_path"] == ["X", "A", "J"]
    assert res["tropical_router_selected_sensitivity"] == 1.0


def test_exp02_universal_minlp_matches_analytical_oracle():
    """Verifica que UniversalLayer2Solver alcanza el óptimo analítico exacto J* = 2990.8835€ sin rejillas fijas."""
    scm = SymbolicSCM("golden_case")
    solver = UniversalLayer2Solver(scm)
    sol = solver.optimize()

    assert sol["status"] == "FEASIBLE"
    assert abs(sol["profit"] - 2990.8835) < 1e-2
    assert sol["controls"]["order_qty"] == 800
    assert abs(sol["controls"]["price"] - 5.4370) < 1e-3
    assert abs(sol["controls"]["marketing_budget"] - 428.2140) < 1e-2


def test_exp02_interior_demand_limited_q1000_and_zero_hardcoding_mutation():
    """Verifica que para q=1000 la rama demand_limited supera la singularidad sqrt(m) en m=0 alcanzando 2754.78€,
    y que mutar simbólicamente holding_cost se refleja en evaluate_smooth_regime sin 0.20 hardcodeado."""
    scm = SymbolicSCM("golden_case")
    solver = UniversalLayer2Solver(scm)
    sol_1000 = solver.optimize(fixed_controls={"order_qty": 1000})
    assert sol_1000["status"] == "FEASIBLE"
    assert abs(sol_1000["profit"] - 2754.7755) < 0.05
    assert sol_1000["controls"]["marketing_budget"] > 400.0

    # Mutar simbólicamente el coeficiente de holding_cost de 0.20 a 1.50 y verificar cambio exacto:
    scm_mut = SymbolicSCM("golden_case")
    q_sym = sp.Symbol("order_qty", real=True)
    s_sym = sp.Symbol("sales", real=True)
    scm_mut.mechanisms["holding_cost"] = 1.50 * sp.Max(0.0, q_sym - s_sym)
    solver_mut = UniversalLayer2Solver(scm_mut)

    st_orig = solver.evaluate_smooth_regime(5.5, 100.0, 1000, "demand_limited")
    st_mut = solver_mut.evaluate_smooth_regime(5.5, 100.0, 1000, "demand_limited")
    unsold = 1000.0 - st_orig["sales"]
    assert unsold > 200.0
    assert abs(st_orig["holding_cost"] - 0.20 * unsold) < 1e-6
    assert abs(st_mut["holding_cost"] - 1.50 * unsold) < 1e-6


def test_exp02_infeasibility_and_kkt_complementary_slackness():
    """Verifica que UniversalLayer2Solver detecta infactibilidad en J_target = 3100€ y cumple holgura
    complementaria KKT (lambda_budget* == 0.0 cuando el presupuesto tiene holgura)."""
    scm = SymbolicSCM("golden_case")
    solver = UniversalLayer2Solver(scm)

    res_3100 = solver.optimize(target_profit=3100.0)
    assert res_3100["status"] == "INFEASIBLE"
    assert abs(res_3100["max_achievable_profit_J_star"] - 2990.8835) < 1e-2
    assert abs(res_3100["profit_shortfall"] - (3100.0 - 2990.8835)) < 1e-2
    assert res_3100["kkt_shadow_price_budget"] == 0.0
    assert res_3100["kkt_shadow_price_market_demand"] > 3.0


def test_exp02_sparse_l1_inverse_intervention():
    """Verifica que para un objetivo intermedio (J_target=1325€), la optimización inversa L1 mueve SOLO 1 variable."""
    scm = SymbolicSCM("golden_case")
    solver = UniversalLayer2Solver(scm)
    current_x = {"price": 3.50, "marketing_budget": 500.0, "order_qty": 500}

    inv_res = solver.solve_sparse_inverse_intervention(current_x, target_profit=1325.0)
    assert inv_res["status"] == "FEASIBLE"
    assert inv_res["l0_norm_active_controls"] == 1
    assert inv_res["achieved_profit"] >= 1325.0 - 1e-2
    assert inv_res["active_control_names"] == ["order_qty"]
    assert abs(inv_res["deltas"]["price"]) < 1e-4
    assert abs(inv_res["deltas"]["marketing_budget"]) < 1e-4


def test_exp02_universal_beats_legacy_on_generated_variants():
    """Demuestra que tanto Legacy default como Layer2Solver.from_graph(graph) alucinan en las variantes de
    generator.py mientras UniversalLayer2Solver acierta con error de alucinación 0.00€."""
    comp = compare_legacy_vs_universal_on_variants(num_variants=10, base_seed=42)
    assert comp["variants_where_legacy_hallucinates"] == 10
    assert comp["variants_where_from_graph_hallucinates"] >= 5
    assert comp["mean_legacy_profit_hallucination_eur"] > 100.0
    assert comp["mean_from_graph_hallucination_eur"] > 50.0
    for item in comp["details"]:
        assert item["universal_true_profit"] >= item["legacy_actual_profit_on_variant"] - 1e-3
        assert item["universal_true_profit"] >= item["from_dir_actual_profit"] - 1e-3


def test_exp03_jensen_gap_and_dynamic_kink_quadrature_exactness():
    """Verifica la fórmula cerrada de la Brecha de Jensen, que la cuadratura dividida dinámicamente en u_kink
    tiene error 0.0000€ tanto en X_det* como en X_stoch*, y que la política estocástica mejora E[J] y CVaR."""
    res = run_jensen_and_cvar_experiment()

    det = res["deterministic_policy"]
    assert abs(det["analytical"]["analytical_expected_profit"] - det["gauss_hermite"]["gh_expected_profit"]) < 1e-4
    assert abs(det["analytical"]["jensen_gap_eur"] - 138.54) < 0.1

    stoch_neutral = res["stochastic_risk_neutral_policy"]
    assert stoch_neutral["quadrature_vs_analytical_error_eur"] < 1e-4

    # La política estocástica supera el beneficio esperado de la política determinista en > +70€/día
    assert res["improvements_over_deterministic"]["expected_profit_gain_gh_eur"] > 70.0
    assert res["improvements_over_deterministic"]["expected_profit_gain_mc_eur"] > 70.0
    assert res["improvements_over_deterministic"]["stockout_prob_reduction_neutral"] > 0.30
    assert res["improvements_over_deterministic"]["cvar_5pct_improvement_risk_averse_eur"] > 0.0
    assert res["improvements_over_deterministic"]["cvar_5pct_improvement_global_q_eur"] > 90.0


def test_exp04_t2_symbolic_ledger_beats_naive_call_counter():
    """Verifica que SymbolicLedger valida asientos divididos (IVA), caza asientos escalados y detecta fugas en ramas if/else."""
    res = demonstrate_t2_symbolic_vs_naive_counterexamples()

    split_case = res["split_vat_valid_case"]
    assert split_case["is_balanced_symbolically"] is True
    assert split_case["naive_auditor_says_balanced"] is False

    scaled_case = res["scaled_debit_bug_case"]
    assert scaled_case["is_balanced_symbolically"] is False
    assert scaled_case["naive_auditor_says_balanced"] is True

    branch_case = res["conditional_branch_leak_case"]
    assert branch_case["is_balanced_symbolically"] is False
    assert branch_case["naive_auditor_says_balanced"] is True
    assert branch_case["branch_residuals"]["fee_mode > 0 -> main"] == "-0.05*amount"

    golden_case = res["golden_case_process_refund"]
    assert golden_case["is_balanced_symbolically"] is False
    assert golden_case["symbolic_residual"] == "-amount"


def test_exp04_t1_faithfulness_cancellation_and_full_audit():
    """Verifica la detección de cancelación algebraica de caminos (T1-B) y la auditoría formal completa T1..T7."""
    cancel_demo = detect_algebraic_path_cancellation_demo()
    assert cancel_demo["has_topological_path_nx"] is True
    assert cancel_demo["detected_as_dead_by_topology_only"] is False
    assert cancel_demo["detected_as_dead_by_symbolic_jacobian"] is True

    auditor = FormalSymbolicAuditor("golden_case")
    report = auditor.run_full_formal_audit()
    diag_ids = {d["id"] for d in report["diagnostics"]}
    assert {"T1", "T1-C", "T2", "T4", "T5", "T6", "T7"}.issubset(diag_ids)


def test_exp04_t7_joint_econometric_estimation_without_cheating():
    """Verifica que el estimador econométrico conjunto recupera alpha=0.15 y beta=8.0 y elige sqrt(m) por BIC."""
    est = JointStructuralEconometricEstimator.estimate_from_csv("golden_case/data/sales_history.csv")
    assert est["selected_functional_basis_by_bic"] == "sqrt_m"
    assert abs(est["estimated_alpha_price_elasticity"] - 0.15) < 0.005
    assert abs(est["estimated_beta_marketing_lift"] - 8.0) < 0.15
    assert est["is_structurally_identifiable"] is True
    assert est["r_squared"] > 0.99


def test_exp05_multi_kink_mpcc_scalability():
    """Verifica que la homotopía LSE + Epígrafe MPCC resuelve M=6 kinks (2^6=64 regímenes) en solo 3 llamadas NLP."""
    res = run_multi_kink_scalability_experiment(num_kinks=6)
    assert res["combinatorial_regimes_2_pow_M"] == 64
    assert res["homotopy_epigraph_mpcc_solver"]["num_nlp_subproblems"] == 3
    assert res["mpcc_error_vs_analytical"] < 0.01
    assert res["subproblem_reduction_factor"] > 20.0


def test_exp05_temporal_cyclic_dag_mpc_beats_myopic():
    """Verifica que el grafo cíclico de inventario se desenrolla en un DAG expandido en el tiempo y MPC supera a la política miope."""
    res = TemporalCyclicSCMAndMPCSolver().solve_myopic_vs_mpc_multi_period()
    assert res["static_graph_is_dag"] is False
    assert res["time_expanded_graph_is_dag"] is True
    assert res["mpc_gain_over_myopic_eur"] > 200.0
    assert res["mpc_time_expanded_policy"]["period_1"]["order_qty"] == 800
    assert res["mpc_time_expanded_policy"]["period_1"]["carry_over_inventory_I1"] > 300.0


def test_exp05_causal_endogeneity_hausman_and_2sls_iv():
    """Verifica que el Test de Hausman-Wu detecta la endogeneidad y que 2SLS/IV y Pearl Back-Door eliminan el sesgo OLS."""
    res = CausalEndogeneityIV2SLSEstimator.run_endogeneity_and_iv_experiment()
    assert res["naive_ols_under_confounding"]["beta_relative_error_pct"] > 20.0
    assert res["hausman_wu_endogeneity_test"]["endogeneity_detected"] is True
    assert res["hausman_wu_endogeneity_test"]["t_statistic"] > 10.0
    assert res["iv_2sls_estimator"]["beta_relative_error_pct"] < 2.0
    assert res["pearl_backdoor_adjustment_estimator"]["beta_relative_error_pct"] < 0.5

