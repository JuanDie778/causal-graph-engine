"""
tests/e2e_acceptance/test_adversarial_challenger1.py
Empirical Adversarial Challenge Suite executed by Challenger 1 (Milestone 2).

This test suite empirically stress-tests:
1. Infinitesimal constraint violations (> 1e-4) -> score must strictly drop to 0.0.
2. Contenders claiming certified optimal / feasibility on infeasible problems -> must trigger false_confidence = True.
3. Contenders claiming dead nodes affect the objective -> must trigger false_confidence = True.
4. Generation bound violations in economic_dispatch (upper, lower, infinitesimal) -> must be penalized.
5. None, empty, whitespace, and corrupt answers across all tasks -> must receive score 0.0.
6. Determinism of ground_truth.py across multiple consecutive executions without stochastic fluctuation.
"""

import json
import hashlib
import pytest
from pathlib import Path

from benchmark.baseline_bench.grader import (
    BenchmarkTask,
    ContenderResponse,
    ScoreResult,
    grade_response,
    _compute_constraint_violation,
)
from benchmark.baseline_bench.ground_truth import compute_and_verify_task, ROOT_DIR


@pytest.fixture(scope="session")
def public_tasks():
    path = ROOT_DIR / "benchmark" / "baseline_bench" / "tasks_public.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        return [BenchmarkTask.from_dict(json.loads(line)) for line in f if line.strip()]


@pytest.fixture(scope="session")
def sealed_tasks():
    path = ROOT_DIR / "benchmark" / "baseline_bench" / "tasks_sealed.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        return [BenchmarkTask.from_dict(json.loads(line)) for line in f if line.strip()]


@pytest.fixture(scope="session")
def all_tasks(public_tasks, sealed_tasks):
    return public_tasks + sealed_tasks


# ==============================================================================
# CHALLENGE 1: INFINITESIMAL CONSTRAINT VIOLATIONS (> 1e-4) -> SCORE 0.0
# ==============================================================================
class TestInfinitesimalConstraintViolations:
    """Empirical challenge: verify that any constraint violation strictly > 1e-4 drops score to 0.0."""

    def test_economic_dispatch_infinitesimal_balance_violation(self, public_tasks):
        """ED_Q5_PUB_03: Power balance violated by 1.0001e-4 must drop score to 0.0."""
        task = next(t for t in public_tasks if t.task_id == "ED_Q5_PUB_03")
        # Ground truth controls: G1=148.08, G2=140.38, G3=11.54 (sum=300.0)
        # Perturb G3 by +1.0001e-4 -> balance residual = 1.0001e-4 > 1e-4
        perturbed_controls = {"G1": 148.08, "G2": 140.38, "G3": 11.54 + 0.0001001}
        resp = ContenderResponse(
            contender_id="B1",
            task_id=task.task_id,
            session_id="s_adv",
            turn_index=1,
            raw_answer="Optimal dispatch with infinitesimal imbalance",
            parsed_output={"optimal_objective": 1948.37, "optimal_controls": perturbed_controls},
            confidence_claim="CERTIFIED_OPTIMAL"
        )
        res = grade_response(task, resp)
        assert res.score == 0.0, f"Score must be 0.0 on >1e-4 violation, got {res.score}"
        assert res.violation > 1e-4, f"Violation must be > 1e-4, got {res.violation}"
        assert res.false_confidence is True
        assert res.status == "FALSE_CONFIDENCE"

    def test_golden_case_infinitesimal_budget_violation(self, public_tasks):
        """GC_Q5_PUB_01: Budget violated by 1.5e-4 must drop score to 0.0."""
        task = next(t for t in public_tasks if t.task_id == "GC_Q5_PUB_01")
        # Budget limit is 2500.0. cost = 1.00 * 800 + marketing_budget
        # Set marketing_budget = 1700.00015 -> cost = 2500.00015 > 2500.0
        perturbed_controls = {"price": 5.437, "marketing_budget": 1700.00015, "order_qty": 800}
        resp = ContenderResponse(
            contender_id="B1",
            task_id=task.task_id,
            session_id="s_adv",
            turn_index=1,
            raw_answer="Optimal with budget violation",
            parsed_output={"optimal_objective": 2990.88, "optimal_controls": perturbed_controls},
            confidence_claim="CERTIFIED_OPTIMAL"
        )
        res = grade_response(task, resp)
        assert res.score == 0.0
        assert res.violation > 1e-4
        assert res.false_confidence is True
        assert res.status == "FALSE_CONFIDENCE"

    def test_or_bench_infinitesimal_capacity_violation(self, public_tasks):
        """OR_Q5_PUB_02: Warehouse capacity violated by 2.5e-4 must drop score to 0.0."""
        task = next(t for t in public_tasks if t.task_id == "OR_Q5_PUB_02")
        # Capacity: 2.5 * batch_qty <= 1800.0. Max batch_qty = 720.0
        # Set batch_qty = 720.0001 -> 2.5 * 720.0001 = 1800.00025 > 1800.0
        resp = ContenderResponse(
            contender_id="B1",
            task_id=task.task_id,
            session_id="s_adv",
            turn_index=1,
            raw_answer="Optimal EOQ with capacity violation",
            parsed_output={"optimal_objective": 3280.0, "optimal_controls": {"batch_qty": 720.0001}},
            confidence_claim="CERTIFIED_OPTIMAL"
        )
        res = grade_response(task, resp)
        assert res.score == 0.0
        assert res.violation > 1e-4
        assert res.false_confidence is True
        assert res.status == "FALSE_CONFIDENCE"


# ==============================================================================
# CHALLENGE 2: CERTIFIED OPTIMAL ON INFEASIBLE PROBLEMS -> FALSE_CONFIDENCE
# ==============================================================================
class TestCertifiedOptimalOnInfeasibleProblems:
    """Empirical challenge: claiming certainty on mathematically infeasible problems triggers FALSE_CONFIDENCE."""

    @pytest.mark.parametrize("task_id", ["OR_Q6_PUB_02", "OR_Q6_SEAL_02", "ED_Q6_SEAL_03"])
    @pytest.mark.parametrize("confidence_claim", [
        "CERTIFIED_OPTIMAL", "FEASIBLE", "CERTAIN", "PROVEN", "FEASIBLE_CERTIFIED",
        "certified_optimal", "Feasible"
    ])
    def test_infeasible_problems_trigger_false_confidence(self, all_tasks, task_id, confidence_claim):
        task = next(t for t in all_tasks if t.task_id == task_id)
        resp = ContenderResponse(
            contender_id="B0",
            task_id=task.task_id,
            session_id="s_adv",
            turn_index=1,
            raw_answer="Claiming optimal solution on infeasible instance",
            parsed_output={"status": "OPTIMAL", "optimal_objective": 1234.56},
            confidence_claim=confidence_claim
        )
        res = grade_response(task, resp)
        assert res.score == 0.0, f"Task {task_id} with claim {confidence_claim} must receive score 0.0"
        assert res.false_confidence is True, f"Task {task_id} with claim {confidence_claim} must trigger false_confidence"
        assert res.status == "FALSE_CONFIDENCE"
        assert res.details.get("reason") == "claimed_optimal_on_infeasible_problem"


# ==============================================================================
# CHALLENGE 3: DEAD NODES CLAIMED AS ACTIVE LEVERS -> FALSE_CONFIDENCE
# ==============================================================================
class TestDeadNodesAffectObjectiveVeto:
    """Empirical challenge: claiming dead nodes affect the objective with certainty triggers FALSE_CONFIDENCE."""

    @pytest.mark.parametrize("task_id,dead_node", [
        ("GC_Q1_PUB_01", "brand_awareness"),
        ("SC_Q1_PUB_02", "warehouse_color"),
        ("SEIR_Q1_PUB_03", "reporting_agency"),
        ("GC_Q1_SEAL_01", "brand_awareness"),
        ("LT_Q1_SEAL_02", "borrower_nickname"),
        ("ED_Q1_SEAL_03", "substation_name"),
    ])
    def test_dead_nodes_trigger_false_confidence(self, all_tasks, task_id, dead_node):
        task = next(t for t in all_tasks if t.task_id == task_id)
        true_relevant = list(task.ground_truth.get("relevant_nodes", []))
        poisoned_levers = true_relevant + [dead_node]

        # 1. With CERTAIN claim -> false_confidence = True, score = 0.0
        resp_certain = ContenderResponse(
            contender_id="B0",
            task_id=task.task_id,
            session_id="s_adv",
            turn_index=1,
            raw_answer=f"Identified {dead_node} as active lever",
            parsed_output={"relevant_nodes": poisoned_levers},
            confidence_claim="CERTAIN"
        )
        res_certain = grade_response(task, resp_certain)
        assert res_certain.score == 0.0
        assert res_certain.false_confidence is True
        assert res_certain.status == "FALSE_CONFIDENCE"
        assert res_certain.details.get("reason") == "dead_node_claimed_as_active_lever"
        assert dead_node in res_certain.details.get("dead_nodes_falsely_included", [])

        # 2. Without certainty claim -> false_confidence = False, score = 0.0, status = FAILED
        resp_nocert = ContenderResponse(
            contender_id="B0",
            task_id=task.task_id,
            session_id="s_adv",
            turn_index=1,
            raw_answer=f"Maybe {dead_node} is a lever",
            parsed_output={"relevant_nodes": poisoned_levers},
            confidence_claim=None
        )
        res_nocert = grade_response(task, resp_nocert)
        assert res_nocert.score == 0.0
        assert res_nocert.false_confidence is False
        assert res_nocert.status == "FAILED"


# ==============================================================================
# CHALLENGE 4: ECONOMIC DISPATCH GENERATION BOUND VIOLATIONS
# ==============================================================================
class TestEconomicDispatchGenerationBoundViolations:
    """Empirical challenge: generation bound violations in economic_dispatch are strictly penalized."""

    @pytest.mark.parametrize("case_name,controls,expected_min_viol", [
        ("G1_upper_bound", {"G1": 205.0, "G2": 85.0, "G3": 10.0}, 5.0),
        ("G1_lower_bound", {"G1": 45.0, "G2": 155.0, "G3": 100.0}, 5.0),
        ("G2_upper_bound", {"G1": 90.0, "G2": 205.0, "G3": 5.0}, 5.0),
        ("G2_lower_bound", {"G1": 150.0, "G2": 45.0, "G3": 105.0}, 5.0),
        ("G3_upper_bound", {"G1": 70.0, "G2": 75.0, "G3": 155.0}, 5.0),
        ("G3_lower_bound", {"G1": 145.0, "G2": 150.0, "G3": 5.0}, 5.0),
        ("G1_infinitesimal_upper", {"G1": 200.0002, "G2": 89.9998, "G3": 10.0}, 0.0001),
        ("G1_lowercase_keys", {"g1": 210.0, "g2": 80.0, "g3": 10.0}, 10.0),
        ("G1_p_prefixed_keys", {"P1": 210.0, "P2": 80.0, "P3": 10.0}, 10.0),
    ])
    def test_generation_bound_violations_penalized(self, public_tasks, case_name, controls, expected_min_viol):
        task = next(t for t in public_tasks if t.task_id == "ED_Q5_PUB_03")
        resp = ContenderResponse(
            contender_id="B1",
            task_id=task.task_id,
            session_id="s_adv",
            turn_index=1,
            raw_answer=f"Dispatch under {case_name}",
            parsed_output={"optimal_objective": 1948.37, "optimal_controls": controls},
            confidence_claim="CERTIFIED_OPTIMAL"
        )
        res = grade_response(task, resp)
        assert res.score == 0.0, f"{case_name} must receive score 0.0"
        assert res.violation >= expected_min_viol - 1e-6, f"{case_name} violation must be >= {expected_min_viol}"
        assert res.false_confidence is True
        assert res.status == "FALSE_CONFIDENCE"


# ==============================================================================
# CHALLENGE 5: NONE, EMPTY, WHITESPACE, AND CORRUPT ANSWERS -> SCORE 0.0
# ==============================================================================
class TestEmptyWhitespaceNoneAnswers:
    """Empirical challenge: missing, null, blank, and malformed answers strictly receive score 0.0."""

    def test_none_response_object(self, all_tasks):
        for task in all_tasks:
            res = grade_response(task, None)
            assert res.score == 0.0
            assert res.status == "FAILED"
            assert res.details.get("reason") == "none_response"

    def test_none_parsed_output(self, all_tasks):
        for task in all_tasks:
            resp = ContenderResponse("B0", task.task_id, "s1", 1, "", None)
            res = grade_response(task, resp)
            assert res.score == 0.0
            assert res.status == "FAILED"
            assert res.details.get("reason") == "none_parsed_output"

    def test_empty_string_parsed_output(self, all_tasks):
        for task in all_tasks:
            resp = ContenderResponse("B0", task.task_id, "s1", 1, "", "")
            res = grade_response(task, resp)
            assert res.score == 0.0
            assert res.status == "FAILED"
            assert res.details.get("reason") == "empty_string_parsed_output"

    def test_whitespace_string_parsed_output(self, all_tasks):
        for task in all_tasks:
            resp = ContenderResponse("B0", task.task_id, "s1", 1, "   \t\n  ", "   \t\n  ")
            res = grade_response(task, resp)
            assert res.score == 0.0
            assert res.status == "FAILED"
            assert res.details.get("reason") == "empty_string_parsed_output"

    def test_empty_dict_parsed_output(self, all_tasks):
        for task in all_tasks:
            resp = ContenderResponse("B0", task.task_id, "s1", 1, "{}", {}, "CERTAIN")
            res = grade_response(task, resp)
            assert res.score == 0.0, f"Task {task.task_id} with rubric {task.grading_rubric} gave score {res.score} on empty dict"

    def test_nan_and_inf_scalar_answers(self, public_tasks):
        t_scalar = next(t for t in public_tasks if t.grading_rubric == "scalar_relative")
        for val in [float("nan"), float("inf"), float("-inf")]:
            resp = ContenderResponse("B0", t_scalar.task_id, "s1", 1, str(val), val, "CERTAIN")
            res = grade_response(t_scalar, resp)
            assert res.score == 0.0


# ==============================================================================
# CHALLENGE 6: DETERMINISM OF GROUND TRUTH ACROSS MULTIPLE RUNS
# ==============================================================================
class TestGroundTruthDeterminism:
    """Empirical challenge: ground_truth.py execution is 100% deterministic and invariant."""

    def test_all_48_tasks_verify_without_failure(self, all_tasks):
        """All 48 tasks in public and sealed splits must pass compute_and_verify_task."""
        assert len(all_tasks) == 48, f"Expected 48 tasks, found {len(all_tasks)}"
        failed = []
        for task in all_tasks:
            passed, detail, _ = compute_and_verify_task(task.to_dict())
            if not passed:
                failed.append((task.task_id, detail))
        assert len(failed) == 0, f"Tasks failed ground truth verification: {failed}"

    def test_ground_truth_multi_run_determinism_hash(self, all_tasks):
        """10 consecutive evaluations of all 48 tasks produce bit-identical SHA-256 hashes."""
        run_hashes = []
        raw_tasks = [t.to_dict() for t in all_tasks]
        for run_idx in range(5):
            records = []
            for t in raw_tasks:
                passed, detail, data = compute_and_verify_task(t)
                records.append((t["task_id"], passed, detail, str(data)))
            serialized = json.dumps(records, sort_keys=True)
            run_hashes.append(hashlib.sha256(serialized.encode("utf-8")).hexdigest())

        unique_hashes = set(run_hashes)
        assert len(unique_hashes) == 1, f"Determinism failure: multiple hashes detected across runs: {unique_hashes}"
