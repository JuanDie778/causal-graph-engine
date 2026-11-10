"""Script maestro para ejecutar todos los experimentos matemáticos de `math_lab`
y exportar la evidencia cuantitativa completa en `math_lab/experimental_results.json`.
"""
import json
import os
import time

from math_lab.experiments.exp01_symbolic_scm_and_routing import (
    SymbolicSCM,
    demonstrate_layer1_harmonic_sum_bug
)
from math_lab.experiments.exp02_universal_minlp_solver import (
    UniversalLayer2Solver,
    compare_legacy_vs_universal_on_variants
)
from math_lab.experiments.exp03_stochastic_cvar_optimizer import (
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


def main():
    t0 = time.perf_counter()
    scm = SymbolicSCM("golden_case")
    solver = UniversalLayer2Solver(scm)

    # Exp 01
    opt_sol = solver.optimize()
    opt_inputs = {
        "price": opt_sol["controls"]["price"],
        "marketing_budget": opt_sol["controls"]["marketing_budget"],
        "order_qty": float(opt_sol["controls"]["order_qty"]),
        "base_demand": 1000.0
    }
    ad_opt = scm.compute_reverse_mode_ad_jacobian(opt_inputs)
    layer1_bug_demo = demonstrate_layer1_harmonic_sum_bug()

    # Exp 02
    opt_q1000 = solver.optimize(fixed_controls={"order_qty": 1000})
    infeas_3100 = solver.optimize(target_profit=3100.0)
    infeas_5000 = solver.optimize(target_profit=5000.0)
    sparse_1325 = solver.solve_sparse_inverse_intervention(
        {"price": 3.50, "marketing_budget": 500.0, "order_qty": 500},
        target_profit=1325.0
    )
    variants_comp = compare_legacy_vs_universal_on_variants(num_variants=10, base_seed=42)

    # Exp 03
    stoch_exp = run_jensen_and_cvar_experiment()

    # Exp 04
    auditor = FormalSymbolicAuditor("golden_case")
    audit_report = auditor.run_full_formal_audit()
    t2_counterexamples = demonstrate_t2_symbolic_vs_naive_counterexamples()
    t1_cancellation = detect_algebraic_path_cancellation_demo()
    t7_econometrics = JointStructuralEconometricEstimator.estimate_from_csv("golden_case/data/sales_history.csv")

    # Exp 05
    multi_kink_exp = run_multi_kink_scalability_experiment(num_kinks=6)
    temporal_mpc_exp = TemporalCyclicSCMAndMPCSolver().solve_myopic_vs_mpc_multi_period()
    endogeneity_iv_exp = CausalEndogeneityIV2SLSEstimator.run_endogeneity_and_iv_experiment()

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    results = {
        "execution_time_ms": round(elapsed_ms, 2),
        "exp01_scm_and_routing": {
            "extracted_mechanisms_sympy": {k: str(v) for k, v in scm.mechanisms.items()},
            "reverse_mode_ad_adjoints_at_optimum": {k: round(v, 4) for k, v in ad_opt["adjoints_dJ_dv"].items()},
            "discrete_step_sensitivities_at_optimum": ad_opt.get("discrete_step_sensitivities", {}),
            "layer1_harmonic_sum_counterexample": layer1_bug_demo
        },
        "exp02_universal_minlp_and_inverse": {
            "golden_case_optimum": opt_sol,
            "fixed_q_1000_demand_limited_optimum": opt_q1000,
            "infeasibility_at_3100_kkt": infeas_3100,
            "infeasibility_at_5000_kkt": infeas_5000,
            "sparse_inverse_at_1325": sparse_1325,
            "variants_generalization_benchmark": variants_comp
        },
        "exp03_stochastic_and_cvar": stoch_exp,
        "exp04_symbolic_audit_and_econometrics": {
            "formal_audit_diagnostics": audit_report["diagnostics"],
            "t2_symbolic_vs_naive_counterexamples": t2_counterexamples,
            "t1_faithfulness_cancellation_demo": t1_cancellation,
            "t7_joint_econometric_estimation": t7_econometrics
        },
        "exp05_advanced_edge_cases": {
            "multi_kink_mpcc_vs_2m": multi_kink_exp,
            "temporal_cyclic_dag_mpc": temporal_mpc_exp,
            "causal_endogeneity_iv_2sls": endogeneity_iv_exp
        }
    }

    out_path = os.path.join("math_lab", "experimental_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Experimentos completados en {elapsed_ms:.1f} ms. Resultados guardados en {out_path}")


if __name__ == "__main__":
    main()

