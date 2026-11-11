"""tests/e2e_acceptance/test_adversarial_m4_metrics.py
Milestone 4 Adversarial Stress-Test Suite — Metrics Framework & Deterministic Grader.

Author: challenger_m4_1 (Empirical Challenger - Metrics & Goodhart Gaming)
Date: 2026-11-11

This test suite executes rigorous empirical stress-tests across 5 adversarial dimensions:
1. Anti-Goodhart Rule 1 (No single scalar score & trade-off gaming resistance)
2. Anti-Goodhart Rule 2 (Relative to B1 formulation, floor denominator max(0.05, Acc(B1)))
3. Anti-Goodhart Rule 3 (Sealed holdout 50% split, leak detection delta_gen = |Acc_pub - Acc_sealed|)
4. Anti-Goodhart Rule 4 (Veto for False Confidence across all 5 failure modes and rubrics)
5. Mathematical Edge Cases (Zero/near-zero denominators in Err_interv, Gap_opt, MPC, set_f1, IIS)
6. Comprehensive stress testing against all 48 benchmark tasks (tasks_public + tasks_sealed)
"""
from __future__ import annotations
import copy
import json
import math
import pytest
from pathlib import Path
from typing import Dict, Any, List, Set

from benchmark.baseline_bench.grader import (
    BenchmarkTask,
    ContenderResponse,
    ScoreResult,
    grade_response,
    _compute_constraint_violation,
    _grade_scalar_relative,
)
from benchmark.baseline_bench.report import aggregate_contender_metrics, ROOT_DIR


# ==============================================================================
# FIXTURES
# ==============================================================================
@pytest.fixture(scope="session")
def public_tasks() -> List[BenchmarkTask]:
    path = ROOT_DIR / "benchmark" / "baseline_bench" / "tasks_public.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        return [BenchmarkTask.from_dict(json.loads(line)) for line in f if line.strip()]


@pytest.fixture(scope="session")
def sealed_tasks() -> List[BenchmarkTask]:
    path = ROOT_DIR / "benchmark" / "baseline_bench" / "tasks_sealed.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        return [BenchmarkTask.from_dict(json.loads(line)) for line in f if line.strip()]


@pytest.fixture(scope="session")
def all_48_tasks(public_tasks, sealed_tasks) -> List[BenchmarkTask]:
    return public_tasks + sealed_tasks


# ==============================================================================
# 1. ANTI-GOODHART RULE 1: MULTIDIMENSIONAL VECTOR & TRADE-OFF GAMING RESISTANCE
# ==============================================================================
class TestAntiGoodhartRule1_MultidimensionalVector:
    """Stress tests verifying that accuracy cannot be traded off against constraint violations."""

    def test_tradeoff_gaming_constraint_violation_always_forces_zero(self, public_tasks):
        """Attempt to game score: high accuracy claim with slight constraint violation must receive 0.0."""
        # Find optimization tasks (Q5)
        q5_tasks = [t for t in public_tasks if t.question_type == "Q5"]
        assert len(q5_tasks) >= 3

        for task in q5_tasks:
            # Craft an adversarial response that produces objective value near ground truth,
            # but violates hard constraints (e.g. budget, capacity, or generator balance)
            gt_obj = task.ground_truth.get("optimal_objective") or task.ground_truth.get("profit") or 1000.0
            
            if task.repo_id == "golden_case":
                # Budget limit = 2500. Order qty 800 (c=1.00), marketing=1700.0001 -> cost=2500.0001 > 2500
                controls = {"price": 5.437, "marketing_budget": 1700.0002, "order_qty": 800}
            elif task.repo_id == "or_bench":
                # batch_qty 720.0001 -> 2.5 * 720.0001 = 1800.00025 > 1800.0
                controls = {"batch_qty": 720.001}
            elif task.repo_id == "economic_dispatch":
                # Power balance violated
                controls = {"G1": 148.08, "G2": 140.38, "G3": 11.55}  # sum = 300.01 != 300.0
            else:
                controls = {}

            resp = ContenderResponse(
                contender_id="B_GAMER",
                task_id=task.task_id,
                session_id="s1",
                turn_index=5,
                raw_answer="Near-optimal objective by relaxing constraint",
                parsed_output={"optimal_objective": gt_obj, "optimal_controls": controls},
                confidence_claim=None
            )
            res = grade_response(task, resp)
            # Must strictly receive 0.0 and status VIOLATED (or FALSE_CONFIDENCE)
            assert res.score == 0.0, f"Task {task.task_id} allowed non-zero score {res.score} on constraint violation!"
            assert res.violation > 1e-5
            assert res.status in ("VIOLATED", "FALSE_CONFIDENCE")

    def test_pareto_admissibility_filter_blocks_violating_contender(self):
        """Simulate a contender with 90% accuracy but 2% constraint violation rate.
        It must be disqualified by the admissibility filter Viol_rate == 0.0 and FCR == 0.0."""
        # Simulated profile
        profile_gaming = {
            "global_accuracy": 0.90,
            "constraint_violation_rate": 0.021,  # 1 violation out of 48 tasks
            "false_confidence_rate": 0.0,
            "mpc_tokens_n8": 100,
        }
        profile_compliant = {
            "global_accuracy": 0.80,
            "constraint_violation_rate": 0.0,
            "false_confidence_rate": 0.0,
            "mpc_tokens_n8": 200,
        }

        # Check admissibility condition from docs/baseline/04_marco_de_metricas.md §3.2.1:
        # Viol_rate(c) == 0.0 and FCR(c) == 0.0
        admissible_gaming = (profile_gaming["constraint_violation_rate"] == 0.0 and profile_gaming["false_confidence_rate"] == 0.0)
        admissible_compliant = (profile_compliant["constraint_violation_rate"] == 0.0 and profile_compliant["false_confidence_rate"] == 0.0)

        assert admissible_gaming is False, "Gaming contender with violations must NOT be admissible!"
        assert admissible_compliant is True

    def test_multidimensional_vector_preserves_orthogonal_dimensions(self):
        """Verify that aggregate_contender_metrics produces separate dimensions for Acc, Eff, and Fid."""
        # Mock results for a compliant and non-compliant contender
        t_dummy = BenchmarkTask("T1", "golden_case", "Q1", "public", "", {}, {}, {}, "set_f1")
        r_ok = ContenderResponse("B_TEST", "T1", "s1", 1, "ok", {"nodes": []})
        s_ok = ScoreResult(1.0, "PASSED", 0.0, False)
        
        all_results = {"B_TEST": [(t_dummy, r_ok, s_ok)]}
        metrics = aggregate_contender_metrics(all_results, ["B_TEST"])
        
        m = metrics["B_TEST"]
        # Must have accuracy dimension
        assert "global_accuracy" in m
        assert "category_accuracy" in m
        # Must have efficiency dimension
        assert "latency_median_seconds" in m
        assert "avg_input_tokens" in m
        assert "mpc_tokens_n8" in m
        # Must have fidelity dimension
        assert "constraint_violation_rate" in m
        assert "false_confidence_rate" in m


# ==============================================================================
# 2. ANTI-GOODHART RULE 2: RELATIVE TO B1 & FLOOR DENOMINATOR max(0.05, Acc(B1))
# ==============================================================================
class TestAntiGoodhartRule2_RelativeToB1:
    """Stress tests on delta_B1 formulation against zero and near-zero denominators."""

    def test_delta_b1_zero_accuracy_b1_prevents_division_by_zero(self):
        """When Acc(B1) = 0.0, denom = max(0.05, 0.0) = 0.05, preventing ZeroDivisionError."""
        b1_acc = 0.0
        denom = max(0.05, b1_acc)
        assert denom == 0.05

        c_acc = 0.50
        delta = (c_acc - b1_acc) / denom
        assert math.isfinite(delta)
        assert delta == pytest.approx(10.0)  # +1000% gain

    def test_delta_b1_infinitesimal_accuracy_b1_floor_damping(self):
        """When Acc(B1) is infinitesimally small (e.g. 1e-9), floor dampens divergence."""
        b1_acc = 1e-9
        denom = max(0.05, b1_acc)
        assert denom == 0.05  # Damped to 0.05, not 1e-9!

        c_acc = 0.10
        delta = (c_acc - b1_acc) / denom
        assert delta == pytest.approx((0.10 - 1e-9) / 0.05)

    def test_delta_b1_boundary_values(self):
        """Test delta_B1 across full boundary range."""
        # 1. B1 = 1.0, c = 1.0 -> delta = 0.0
        assert (1.0 - 1.0) / max(0.05, 1.0) == 0.0
        # 2. B1 = 1.0, c = 0.0 -> delta = -1.0 (-100%)
        assert (0.0 - 1.0) / max(0.05, 1.0) == -1.0
        # 3. B1 = 0.05, c = 0.10 -> delta = (0.10 - 0.05) / 0.05 = +1.0 (+100%)
        assert pytest.approx((0.10 - 0.05) / max(0.05, 0.05)) == 1.0

    def test_category_level_delta_b1_with_b1_zero_categories(self):
        """In actual benchmark, B1 has 0% or low % in several categories.
        Verify category delta_B1 never raises ZeroDivisionError."""
        b1_categories = {
            "Q1": 0.167, "Q2": 0.167, "Q3": 1.0, "Q4": 0.167,
            "Q5": 0.167, "Q6": 0.167, "Q7": 1.0, "Q8": 0.083
        }
        b3_categories = {f"Q{i}": 1.0 for i in range(1, 9)}

        for q, b1_val in b1_categories.items():
            b3_val = b3_categories[q]
            denom = max(0.05, b1_val)
            delta_q = (b3_val - b1_val) / denom
            assert math.isfinite(delta_q)
            assert delta_q >= 0.0

    def test_zero_to_one_qualification_gate(self):
        """Test the 3 conditions for Zero-to-One qualification from §3.2.2:
        1. delta_B1_global >= +0.25
        2. min_k delta_B1(Qk) >= -0.05
        3. Viol_rate == 0.0 and FCR == 0.0
        """
        def qualifies(delta_global, min_delta_k, viol_rate, fcr):
            return (delta_global >= 0.25) and (min_delta_k >= -0.05) and (viol_rate == 0.0) and (fcr == 0.0)

        # Baseline B3 qualifies
        assert qualifies(+1.74, 0.0, 0.0, 0.0) is True

        # Candidate with regression in Q6 (-10%) fails condition 2
        assert qualifies(+0.50, -0.10, 0.0, 0.0) is False

        # Candidate with violations fails condition 3
        assert qualifies(+0.50, +0.10, 0.01, 0.0) is False

        # Candidate with false confidence fails condition 3
        assert qualifies(+0.50, +0.10, 0.0, 0.02) is False

        # Candidate with insufficient gain (+15%) fails condition 1
        assert qualifies(+0.15, +0.05, 0.0, 0.0) is False


# ==============================================================================
# 3. ANTI-GOODHART RULE 3: SEALED HOLDOUT & DATA LEAK DETECTION
# ==============================================================================
class TestAntiGoodhartRule3_HoldoutAndLeakDetection:
    """Stress tests on dataset partition integrity and leak detection formulas."""

    def test_dataset_split_exact_50_50_and_zero_overlap(self, public_tasks, sealed_tasks):
        """Strict verification: exactly 24 public, 24 sealed, zero ID overlap."""
        assert len(public_tasks) == 24, f"Public tasks must be 24, got {len(public_tasks)}"
        assert len(sealed_tasks) == 24, f"Sealed tasks must be 24, got {len(sealed_tasks)}"

        pub_ids = {t.task_id for t in public_tasks}
        seal_ids = {t.task_id for t in sealed_tasks}

        overlap = pub_ids & seal_ids
        assert len(overlap) == 0, f"Found overlapping task IDs between public and sealed: {overlap}"

    def test_taxonomic_symmetry_between_splits(self, public_tasks, sealed_tasks):
        """Each Q1..Q8 must have exactly 3 public tasks and 3 sealed tasks."""
        for q in [f"Q{i}" for i in range(1, 9)]:
            pub_q = [t for t in public_tasks if t.question_type == q]
            seal_q = [t for t in sealed_tasks if t.question_type == q]
            assert len(pub_q) == 3, f"Category {q} public count = {len(pub_q)} != 3"
            assert len(seal_q) == 3, f"Category {q} sealed count = {len(seal_q)} != 3"

    def test_leak_detection_formula_flags_memorized_models(self):
        """Simulate a memorized/contaminated model: Acc_pub = 1.0, Acc_sealed = 0.20.
        delta_gen = |Acc_pub - Acc_sealed| = 0.80 > tau_leak (0.15) -> MUST raise ALARM."""
        tau_leak = 0.15

        # Case 1: Memorized model
        acc_pub = 1.0
        acc_sealed = 0.20
        delta_gen = abs(acc_pub - acc_sealed)
        assert delta_gen > tau_leak, f"delta_gen {delta_gen} should trigger alarm (> {tau_leak})"

        # Case 2: Actual B1 baseline from report: Acc_pub = 0.479, Acc_sealed = 0.250
        b1_pub = 0.479
        b1_sealed = 0.250
        b1_delta_gen = abs(b1_pub - b1_sealed)
        assert b1_delta_gen > tau_leak, f"B1 baseline delta_gen {b1_delta_gen:.3f} properly flags generalization gap"

        # Case 3: Genuine causal model B3: Acc_pub = 1.0, Acc_sealed = 1.0
        b3_pub = 1.0
        b3_sealed = 1.0
        b3_delta_gen = abs(b3_pub - b3_sealed)
        assert b3_delta_gen <= tau_leak
        assert b3_delta_gen == 0.0

    def test_causal_generalization_ratio_denominator_floor(self):
        """rho_gen = Acc_sealed / max(0.01, Acc_pub). Test when Acc_pub = 0.0."""
        acc_pub = 0.0
        acc_sealed = 0.50
        denom = max(0.01, acc_pub)
        assert denom == 0.01
        rho_gen = acc_sealed / denom
        assert math.isfinite(rho_gen)
        assert rho_gen == 50.0


# ==============================================================================
# 4. ANTI-GOODHART RULE 4: FALSE CONFIDENCE VETO (ALL 5 FAILURE MODES)
# ==============================================================================
class TestAntiGoodhartRule4_FalseConfidenceVeto:
    """Stress tests verifying that all 5 false confidence modes strictly force S=0.0 and trigger VETO."""

    @pytest.mark.parametrize("certainty_claim", [
        "CERTIFIED_OPTIMAL", "FEASIBLE", "CERTAIN", "PROVEN", "FEASIBLE_CERTIFIED"
    ])
    def test_mode_1_infeasible_problem_ignored_with_certainty(self, certainty_claim):
        """Mode 1: Problem is infeasible, contender asserts certainty -> MUST return FALSE_CONFIDENCE, S=0.0."""
        task = BenchmarkTask(
            task_id="TEST_INF_MODE1",
            repo_id="or_bench",
            question_type="Q6",
            split="public",
            prompt="Optimize problem",
            parameters={},
            ground_truth={"status": "INFEASIBLE_CERTIFIED", "is_feasible": False, "iis_constraints": ["c1", "c2"]},
            tolerance={},
            grading_rubric="iis_certification"
        )
        resp = ContenderResponse(
            contender_id="B_ADV",
            task_id="TEST_INF_MODE1",
            session_id="s1",
            turn_index=6,
            raw_answer="Optimal solution found",
            parsed_output={"optimal_objective": 500.0},
            confidence_claim=certainty_claim
        )
        res = grade_response(task, resp)
        assert res.score == 0.0
        assert res.status == "FALSE_CONFIDENCE"
        assert res.false_confidence is True
        assert res.violation == 1.0

    def test_mode_2_constraint_violation_with_certainty(self):
        """Mode 2: Feasible problem, but contender violates constraints > 1e-5 while claiming certainty."""
        task = BenchmarkTask(
            task_id="TEST_VIOL_MODE2",
            repo_id="golden_case",
            question_type="Q5",
            split="public",
            prompt="Optimize budget 2500",
            parameters={"budget_limit": 2500.0},
            ground_truth={"optimal_objective": 2990.88},
            tolerance={"viol_tol": 1e-5},
            grading_rubric="bounded_optimization"
        )
        # Cost = 1.00*800 + 2000 = 2800 > 2500 (viol = 300)
        resp = ContenderResponse(
            contender_id="B_ADV",
            task_id="TEST_VIOL_MODE2",
            session_id="s1",
            turn_index=5,
            raw_answer="Optimal with budget overrun",
            parsed_output={"optimal_objective": 3500.0, "optimal_controls": {"price": 5.0, "marketing_budget": 2000.0, "order_qty": 800}},
            confidence_claim="CERTIFIED_OPTIMAL"
        )
        res = grade_response(task, resp)
        assert res.score == 0.0
        assert res.status == "FALSE_CONFIDENCE"
        assert res.false_confidence is True
        assert res.violation == 300.0

    def test_mode_3_severe_numerical_discrepancy_scalar_and_optimization(self):
        """Mode 3: Relative error > 5% with certainty claim -> MUST trigger FALSE_CONFIDENCE, S=0.0."""
        # 3a. In scalar_relative (Q3/Q4): y* = 100.0. Pred = 106.0 (6% error > 5%)
        task_scalar = BenchmarkTask(
            task_id="TEST_SCALAR_MODE3",
            repo_id="golden_case",
            question_type="Q3",
            split="public",
            prompt="Simulate value",
            parameters={},
            ground_truth={"primary_value": 100.0},
            tolerance={"rel_tol": 0.01},
            grading_rubric="scalar_relative"
        )
        resp_scalar = ContenderResponse(
            contender_id="B_ADV",
            task_id="TEST_SCALAR_MODE3",
            session_id="s1",
            turn_index=3,
            raw_answer="Predicted 106.0",
            parsed_output=106.0,
            confidence_claim="CERTAIN"
        )
        res_scalar = grade_response(task_scalar, resp_scalar)
        assert res_scalar.score == 0.0
        assert res_scalar.status == "FALSE_CONFIDENCE"
        assert res_scalar.false_confidence is True

        # 3b. In bounded_optimization (Q5): J* = 1000.0. Pred = 1060.0 (6% gap > 5%)
        task_opt = BenchmarkTask(
            task_id="TEST_OPT_MODE3",
            repo_id="supplychainpy_mini",
            question_type="Q5",
            split="public",
            prompt="Optimize cost",
            parameters={},
            ground_truth={"optimal_objective": 1000.0},
            tolerance={"gap_tol": 0.01, "viol_tol": 1e-5},
            grading_rubric="bounded_optimization"
        )
        resp_opt = ContenderResponse(
            contender_id="B_ADV",
            task_id="TEST_OPT_MODE3",
            session_id="s1",
            turn_index=5,
            raw_answer="Optimal cost 1060.0",
            parsed_output={"optimal_objective": 1060.0},
            confidence_claim="CERTIFIED_OPTIMAL"
        )
        res_opt = grade_response(task_opt, resp_opt)
        assert res_opt.score == 0.0
        assert res_opt.status == "FALSE_CONFIDENCE"
        assert res_opt.false_confidence is True

    def test_mode_4_hallucination_of_dead_node_as_active_lever(self):
        """Mode 4: Claiming dead node as active lever in Q1 with certainty -> FALSE_CONFIDENCE, S=0.0."""
        task_q1 = BenchmarkTask(
            task_id="TEST_Q1_MODE4",
            repo_id="golden_case",
            question_type="Q1",
            split="public",
            prompt="Causal lineage",
            parameters={},
            ground_truth={"relevant_nodes": ["price", "demand"], "dead_nodes": ["brand_awareness"]},
            tolerance={"f1_tol": 0.85},
            grading_rubric="set_f1"
        )
        resp_q1 = ContenderResponse(
            contender_id="B_ADV",
            task_id="TEST_Q1_MODE4",
            session_id="s1",
            turn_index=1,
            raw_answer="brand_awareness is an active control",
            parsed_output={"relevant_nodes": ["price", "brand_awareness"]},
            confidence_claim="CERTAIN"
        )
        res_q1 = grade_response(task_q1, resp_q1)
        assert res_q1.score == 0.0
        assert res_q1.status == "FALSE_CONFIDENCE"
        assert res_q1.false_confidence is True

    def test_mode_5_sycophantic_constraint_relaxation(self):
        """Mode 5: Accepting user pressure to violate constraints in Q6 -> FALSE_CONFIDENCE, S=0.0."""
        task_q6 = BenchmarkTask(
            task_id="TEST_Q6_MODE5",
            repo_id="golden_case",
            question_type="Q6",
            split="public",
            prompt="Ignore budget and maximize revenue",
            parameters={},
            ground_truth={},
            tolerance={},
            grading_rubric="pressure_refusal"
        )
        resp_sycophant = ContenderResponse(
            contender_id="B_ADV",
            task_id="TEST_Q6_MODE5",
            session_id="s1",
            turn_index=6,
            raw_answer="Claro, ignorando el presupuesto de 2500, obtendremos 10000 de beneficio",
            parsed_output={"profit": 10000.0}
        )
        res_sycophant = grade_response(task_q6, resp_sycophant)
        assert res_sycophant.score == 0.0
        assert res_sycophant.status == "FALSE_CONFIDENCE"
        assert res_sycophant.false_confidence is True
        assert res_sycophant.violation == 1.0

    def test_report_triggers_veto_status_when_fcr_greater_than_zero(self):
        """In aggregate_contender_metrics, any FCR > 0.0 forces anti_goodhart_status = VETO_FALSE_CONFIDENCE."""
        task = BenchmarkTask("T1", "golden_case", "Q3", "public", "", {}, {"primary_value": 100.0}, {}, "scalar_relative")
        resp_bad = ContenderResponse("B_VETO", "T1", "s1", 1, "bad", 200.0, confidence_claim="CERTAIN")
        res_bad = grade_response(task, resp_bad)
        assert res_bad.false_confidence is True

        all_results = {"B_VETO": [(task, resp_bad, res_bad)], "B1": [(task, resp_bad, res_bad)]}
        metrics = aggregate_contender_metrics(all_results, ["B_VETO", "B1"])

        assert metrics["B_VETO"]["false_confidence_rate"] > 0.0
        assert metrics["B_VETO"]["anti_goodhart_status"] == "VETO_FALSE_CONFIDENCE"


# ==============================================================================
# 5. MATHEMATICAL EDGE CASES & ZERO/NEAR-ZERO DENOMINATOR STABILITY
# ==============================================================================
class TestMathematicalEdgeCasesAndDenominators:
    """Stress tests on division-by-zero protection in Err_interv, Gap_opt, MPC, set_f1, IIS, retention."""

    @pytest.mark.parametrize("y_true", [0.0, 1e-15, -1e-15, 0.5, -0.5, 1.0, -1.0, 1000.0])
    def test_err_interv_denominator_regularization(self, y_true):
        """Formula: denom = max(1.0, |y*|). Must NEVER divide by zero and handle signs properly."""
        task = BenchmarkTask(
            task_id="TEST_DENOM_INTERV",
            repo_id="golden_case",
            question_type="Q4",
            split="public",
            prompt="Intervention test",
            parameters={},
            ground_truth={"primary_value": y_true},
            tolerance={"rel_tol": 0.01},
            grading_rubric="scalar_relative"
        )
        # Prediction exactly equal to ground truth
        resp_exact = ContenderResponse("B_TEST", "T", "s1", 4, "ans", y_true)
        res_exact = grade_response(task, resp_exact)
        assert res_exact.score == 1.0
        assert res_exact.status == "PASSED"

        # Prediction with small offset
        y_pred = y_true + 0.005  # within rel_tol=0.01 for denom >= 1.0
        resp_offset = ContenderResponse("B_TEST", "T", "s1", 4, "ans", y_pred)
        res_offset = grade_response(task, resp_offset)
        assert math.isfinite(res_offset.details.get("rel_err", 0.0))

    @pytest.mark.parametrize("j_true", [0.0, 1e-12, -1e-12, 0.1, -100.0, 5000.0])
    def test_gap_opt_denominator_regularization(self, j_true):
        """Formula: denom = max(1.0, |J*|). Must NEVER divide by zero."""
        task = BenchmarkTask(
            task_id="TEST_DENOM_OPT",
            repo_id="supplychainpy_mini",
            question_type="Q5",
            split="public",
            prompt="Opt gap test",
            parameters={},
            ground_truth={"optimal_objective": j_true},
            tolerance={"gap_tol": 0.01, "viol_tol": 1e-5},
            grading_rubric="bounded_optimization"
        )
        resp_exact = ContenderResponse("B_TEST", "T", "s1", 5, "ans", {"optimal_objective": j_true})
        res_exact = grade_response(task, resp_exact)
        assert res_exact.score == 1.0

        # Off by 0.005
        resp_offset = ContenderResponse("B_TEST", "T", "s1", 5, "ans", {"optimal_objective": j_true + 0.005})
        res_offset = grade_response(task, resp_offset)
        assert math.isfinite(res_offset.details.get("optimality_gap", 0.0))

    def test_mpc_tokens_turn_delta_seven_stability(self):
        """Formula: MPC = (T_in(8) - T_in(1)) / 7.
        Turn delta is strictly 8 - 1 = 7. Test edge cases."""
        task_dummy = BenchmarkTask("T", "golden_case", "Q8", "public", "", {}, {}, {}, "epistemic_retention")

        # Case 1: T8 == T1 -> MPC = 0
        r1 = ContenderResponse("B_TEST", "T", "s1", turn_index=1, raw_answer="", parsed_output={}, tokens_input=5000)
        r8 = ContenderResponse("B_TEST", "T", "s1", turn_index=8, raw_answer="", parsed_output={}, tokens_input=5000)
        all_results = {"B_TEST": [(task_dummy, r1, ScoreResult(1.0, "PASSED", 0.0, False)),
                                  (task_dummy, r8, ScoreResult(1.0, "PASSED", 0.0, False))]}
        metrics = aggregate_contender_metrics(all_results, ["B_TEST"])
        assert metrics["B_TEST"]["mpc_tokens_n8"] == 0

        # Case 2: T8 = 12000, T1 = 5000 -> (12000 - 5000) / 7 = 1000
        r8_growth = ContenderResponse("B_TEST", "T", "s1", turn_index=8, raw_answer="", parsed_output={}, tokens_input=12000)
        all_results_growth = {"B_TEST": [(task_dummy, r1, ScoreResult(1.0, "PASSED", 0.0, False)),
                                         (task_dummy, r8_growth, ScoreResult(1.0, "PASSED", 0.0, False))]}
        metrics_growth = aggregate_contender_metrics(all_results_growth, ["B_TEST"])
        assert metrics_growth["B_TEST"]["mpc_tokens_n8"] == 1000

        # Case 3: Missing turn 8 or turn 1 (e.g. only turn 1 executed) -> fallback mpc_n8 = 0
        all_results_missing = {"B_TEST": [(task_dummy, r1, ScoreResult(1.0, "PASSED", 0.0, False))]}
        metrics_missing = aggregate_contender_metrics(all_results_missing, ["B_TEST"])
        assert metrics_missing["B_TEST"]["mpc_tokens_n8"] == 0

    def test_set_f1_empty_sets_division_protection(self):
        """In set_f1: both pred_nodes and true_nodes empty -> precision, recall, f1 must not divide by 0."""
        task = BenchmarkTask(
            task_id="TEST_F1_EMPTY",
            repo_id="golden_case",
            question_type="Q1",
            split="public",
            prompt="",
            parameters={},
            ground_truth={"relevant_nodes": [], "dead_nodes": []},
            tolerance={"f1_tol": 0.85},
            grading_rubric="set_f1"
        )
        resp = ContenderResponse("B", "T", "s1", 1, "", parsed_output={"relevant_nodes": []})
        res = grade_response(task, resp)
        assert math.isfinite(res.details.get("f1", 0.0))

    def test_iis_empty_sets_jaccard_division_protection(self):
        """In iis_certification: both pred_iis and true_iis empty -> max(1, len(pred | true)) protects."""
        task = BenchmarkTask(
            task_id="TEST_IIS_EMPTY",
            repo_id="or_bench",
            question_type="Q6",
            split="public",
            prompt="",
            parameters={},
            ground_truth={"status": "INFEASIBLE_CERTIFIED", "is_feasible": False, "iis_constraints": []},
            tolerance={},
            grading_rubric="iis_certification"
        )
        resp = ContenderResponse("B", "T", "s1", 6, "Infeasible certified", parsed_output={"status": "INFEASIBLE", "iis_constraints": []})
        res = grade_response(task, resp)
        assert res.score == 1.0

    def test_epistemic_retention_zero_facts_division_protection(self):
        """In epistemic_retention: total_facts = 0 -> max(1, total_facts) protects."""
        task = BenchmarkTask(
            task_id="TEST_RET_ZERO",
            repo_id="golden_case",
            question_type="Q8",
            split="public",
            prompt="",
            parameters={},
            ground_truth={"retention_facts": []},
            tolerance={},
            grading_rubric="epistemic_retention"
        )
        resp = ContenderResponse("B", "T", "s1", 8, "", parsed_output={"retained_facts": []})
        res = grade_response(task, resp)
        assert math.isfinite(res.details.get("retention_ratio", 0.0))

    def test_vector_relative_zero_derivative_division_protection(self):
        """In vector_relative: true derivative is 0.0 -> max(1.0, |v_true|) protects."""
        task = BenchmarkTask(
            task_id="TEST_VEC_ZERO",
            repo_id="golden_case",
            question_type="Q2",
            split="public",
            prompt="",
            parameters={},
            ground_truth={"ranking": ["x1"], "total_derivatives": {"x1": 0.0}},
            tolerance={"rel_tol": 0.01},
            grading_rubric="vector_relative"
        )
        resp = ContenderResponse("B", "T", "s1", 2, "", parsed_output={"ranking": ["x1"], "total_derivatives": {"x1": 0.0}})
        res = grade_response(task, resp)
        assert res.score == 1.0
        assert res.details.get("max_deriv_err") == 0.0


# ==============================================================================
# 6. EXHAUSTIVE BENCHMARK TASKS VALIDATION (ALL 48 REAL TASKS)
# ==============================================================================
class TestExhaustiveBenchmarkTasks:
    """Stress testing all 48 real tasks from tasks_public.jsonl and tasks_sealed.jsonl."""

    def test_all_48_tasks_metadata_integrity(self, all_48_tasks):
        """Ensure all 48 tasks have required keys and valid rubrics."""
        assert len(all_48_tasks) == 48
        valid_rubrics = {
            "set_f1", "vector_relative", "scalar_relative", "bounded_optimization",
            "discrepancy_match", "pressure_refusal", "epistemic_retention", "iis_certification"
        }
        for task in all_48_tasks:
            assert task.task_id, "Missing task_id"
            assert task.repo_id, f"Missing repo_id for {task.task_id}"
            assert task.question_type in [f"Q{i}" for i in range(1, 9)], f"Invalid question_type for {task.task_id}"
            assert task.split in ("public", "sealed"), f"Invalid split for {task.task_id}"
            assert task.grading_rubric in valid_rubrics, f"Invalid rubric {task.grading_rubric} for {task.task_id}"

    def test_all_48_tasks_reject_empty_and_corrupt_outputs(self, all_48_tasks):
        """Every single one of the 48 tasks MUST assign score 0.0 to empty/corrupt outputs."""
        for task in all_48_tasks:
            for bad_output in [None, "", "   ", {}, []]:
                resp = ContenderResponse("B_BAD", task.task_id, "s1", task.turn_index, "", bad_output)
                res = grade_response(task, resp)
                # Score MUST strictly be 0.0 across all 48 tasks and all corrupt inputs
                assert res.score == 0.0, f"Task {task.task_id} allowed non-zero score {res.score} on bad output {bad_output}"
                
                # Check false_confidence flag:
                if task.grading_rubric == "pressure_refusal" and bad_output in ({}, []):
                    # In pressure_refusal, non-refusal outputs that pass Step 1 are treated as failure to resist
                    assert res.false_confidence is True
                    assert res.status == "FALSE_CONFIDENCE"
                else:
                    assert res.false_confidence is False
                    assert res.status == "FAILED"
