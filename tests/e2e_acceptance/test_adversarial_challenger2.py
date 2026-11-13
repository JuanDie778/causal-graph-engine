"""
tests/e2e_acceptance/test_adversarial_challenger2.py
Empirical Adversarial Challenge Suite executed by Acceptance Test Challenger 2.

This test file adversarially stress-tests and mutates:
1. Tier 3 Cross-Feature & Pairwise tests (test_tier3_integration.py)
   - Perturbed tolerance boundary sensitivity
   - False confidence veto activation on infeasible & constraint-violating tasks
   - Diagnostic of constructor signature & schema bugs in test_tier3_integration.py
2. Tier 4 Real-World Scenarios (test_tier4_scenarios.py)
   - Amnesia N=8 sequence breaks (skipped turns, mismatched session IDs, contradiction rate spikes)
   - Anti-Goodhart rules enforcement mutations (composite scores, missing B1 relative scores)
   - Diagnostic of session filtering bug and missing ERR/CR calculations
3. Integrity Suite (test_integrity.py)
   - Injected file detection in cge/
   - Injected placeholder detection (TODO, TBD, FIXME, [INSERT], Lorem ipsum)
   - Diagnostic of tolerance dict vs float schema bug
"""

import sys
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import pytest

from benchmark.baseline_bench.grader import grade_response, ScoreResult, BenchmarkTask
from benchmark.baseline_bench.harness import ContenderResponse


# ===========================================================================
# 1. TIER 3 ADVERSARIAL CHALLENGES
# ===========================================================================

class TestTier3AdversarialChallenges:
    """Stress-tests for Tier 3: Grader, Tolerances, and False Confidence Veto."""

    def test_challenge_tier3_tolerance_perturbation_sensitivity(self, tasks_public_path, parse_jsonl):
        """EMPIRICAL CHALLENGE 1.1: Verify sensitivity of grader to perturbed tolerance values.
        
        Tests:
        - 0.8 * tol: strictly inside tolerance -> score == 1.0
        - 1.2 * tol: outside tolerance -> score == 0.0
        - 10.0 * tol: gross deviation -> score == 0.0
        - Exact boundary floating-point edge case: IEEE 754 precision makes (1.0 + 1.0 * 0.01) yield rel_err=0.010000000000000038
        """
        tasks = parse_jsonl(tasks_public_path)
        # Find a numerical simulation task (e.g. GC_Q3_PUB_01)
        target_tasks = [t for t in tasks if t["question_type"] == "Q3" and "profit" in t["ground_truth"]]
        assert len(target_tasks) > 0, "Must have numerical Q3 task"
        raw_t = target_tasks[0]

        task_obj = BenchmarkTask.from_dict(raw_t)
        true_val = raw_t["ground_truth"]["profit"]
        rel_tol = float(raw_t["tolerance"].get("rel_tol", 0.01))

        def grade_val(val: float) -> ScoreResult:
            resp = ContenderResponse(
                contender_id="B1",
                task_id=task_obj.task_id,
                session_id=task_obj.session_id,
                turn_index=task_obj.turn_index,
                raw_answer=str(val),
                parsed_output=val,
                confidence_claim=None,
                tokens_input=100,
                tokens_output=20,
                latency_seconds=0.1,
                tool_calls_count=0,
                memory_peak_bytes=0,
                sandbox_errors=[]
            )
            return grade_response(task_obj, resp)

        # 1. Inside tolerance (0.8 * tol)
        res_inside = grade_val(true_val * (1.0 + 0.8 * rel_tol))
        assert abs(res_inside.score - 1.0) < 1e-6, f"Expected 1.0 at 0.8*tol, got {res_inside.score}"

        # 2. Outside tolerance (1.2 * tol)
        res_outside = grade_val(true_val * (1.0 + 1.2 * rel_tol))
        assert abs(res_outside.score - 0.0) < 1e-6, f"Expected 0.0 at 1.2*tol, got {res_outside.score}"

        # 3. Gross deviation (10.0 * tol)
        res_gross = grade_val(true_val * (1.0 + 10.0 * rel_tol))
        assert abs(res_gross.score - 0.0) < 1e-6, f"Expected 0.0 at 10.0*tol, got {res_gross.score}"

        # 4. Floating point boundary observation:
        # Notice that val = true_val * (1.0 + 1.0 * rel_tol) yields rel_err slightly > rel_tol in Python float
        # Grader requires rel_err <= rel_tol, so without float epsilon tolerance, exact mathematical border gets 0.0
        res_boundary = grade_val(true_val * (1.0 + 1.0 * rel_tol))
        assert res_boundary.details["rel_err"] > rel_tol, "Demonstrates IEEE 754 precision edge case at exact boundary"

    def test_challenge_tier3_false_confidence_veto_activation(self, tasks_public_path, parse_jsonl):
        """EMPIRICAL CHALLENGE 1.2: Verify false confidence veto on infeasible & constraint-violating tasks.
        
        Anti-Goodhart Rule 4 dictates that claiming certainty on:
        a) Infeasible tasks, or
        b) Tasks with constraint violation > 1e-5
        MUST trigger false_confidence=True and score=0.0.
        """
        tasks = parse_jsonl(tasks_public_path)

        # A) Infeasible task: OR_Q6_PUB_02 (status: INFEASIBLE_CERTIFIED)
        infeasible_tasks = [t for t in tasks if t.get("ground_truth", {}).get("status") == "INFEASIBLE_CERTIFIED"]
        assert len(infeasible_tasks) > 0, "Must have INFEASIBLE_CERTIFIED task"
        inf_task = BenchmarkTask.from_dict(infeasible_tasks[0])

        for cert_claim in ["CERTIFIED_OPTIMAL", "FEASIBLE", "CERTAIN", "PROVEN", "FEASIBLE_CERTIFIED"]:
            resp_false_cert = ContenderResponse(
                contender_id="B0",
                task_id=inf_task.task_id,
                session_id=inf_task.session_id,
                turn_index=inf_task.turn_index,
                raw_answer="Optimal solution found: profit=99999",
                parsed_output={"status": "FEASIBLE", "profit": 99999.0},
                confidence_claim=cert_claim,
                tokens_input=100,
                tokens_output=20,
                latency_seconds=0.1,
                tool_calls_count=0,
                memory_peak_bytes=0,
                sandbox_errors=[]
            )
            score_res = grade_response(inf_task, resp_false_cert)
            assert score_res.false_confidence is True, f"Veto failed for claim {cert_claim}"
            assert score_res.score == 0.0, f"Expected score 0.0, got {score_res.score}"
            assert score_res.status == "FALSE_CONFIDENCE"

        # B) Feasible task with constraint violation > 1e-5: GC_Q5_PUB_01
        q5_tasks = [t for t in tasks if t["question_type"] == "Q5" and t["repo_id"] == "golden_case"]
        assert len(q5_tasks) > 0, "Must have GC_Q5 task"
        q5_task = BenchmarkTask.from_dict(q5_tasks[0])

        # Violate budget constraint: max budget is 2500, order_qty = 2000, marketing_budget = 1000 (cost = 2000 + 1000 = 3000 > 2500)
        violating_controls = {"order_qty": 2000.0, "marketing_budget": 1000.0, "roast_profile": "medium"}

        # B.1: With certainty claim -> VETO
        resp_viol_cert = ContenderResponse(
            contender_id="B2",
            task_id=q5_task.task_id,
            session_id=q5_task.session_id,
            turn_index=q5_task.turn_index,
            raw_answer="Optimal controls found",
            parsed_output={"optimal_controls": violating_controls, "optimal_objective": 5000.0},
            confidence_claim="CERTIFIED_OPTIMAL",
            tokens_input=100,
            tokens_output=20,
            latency_seconds=0.1,
            tool_calls_count=0,
            memory_peak_bytes=0,
            sandbox_errors=[]
        )
        res_viol_cert = grade_response(q5_task, resp_viol_cert)
        assert res_viol_cert.score == 0.0, "Score must be 0.0 for constraint violation"
        assert res_viol_cert.false_confidence is True, "False confidence must be True when claiming certainty on violating solution"
        assert res_viol_cert.violation > 1e-5, f"Violation magnitude must be > 1e-5, got {res_viol_cert.violation}"
        assert res_viol_cert.status == "FALSE_CONFIDENCE"

        # B.2: Without certainty claim -> VIOLATED (not FALSE_CONFIDENCE)
        resp_viol_nocert = ContenderResponse(
            contender_id="B2",
            task_id=q5_task.task_id,
            session_id=q5_task.session_id,
            turn_index=q5_task.turn_index,
            raw_answer="Candidate controls found",
            parsed_output={"optimal_controls": violating_controls, "optimal_objective": 5000.0},
            confidence_claim=None,
            tokens_input=100,
            tokens_output=20,
            latency_seconds=0.1,
            tool_calls_count=0,
            memory_peak_bytes=0,
            sandbox_errors=[]
        )
        res_viol_nocert = grade_response(q5_task, resp_viol_nocert)
        assert res_viol_nocert.score == 0.0
        assert res_viol_nocert.false_confidence is False, "False confidence must be False when no certainty was claimed"
        assert res_viol_nocert.status == "VIOLATED"

    def test_challenge_tier3_reproduce_test_suite_bugs(self, run_cli, tasks_public_path):
        """EMPIRICAL CHALLENGE 1.3: Confirm and demonstrate bugs inside test_tier3_integration.py.
        
        Demonstrates that:
        1. Calling BenchmarkTask with `repository_id` raises TypeError.
        2. Filtering tasks for `status == 'INFEASIBLE'` yields 0 tasks because status is `'INFEASIBLE_CERTIFIED'`.
        """
        # Bug 1: BenchmarkTask signature
        test_script_sig = """
from benchmark.baseline_bench.harness import BenchmarkTask
try:
    BenchmarkTask(
        task_id="t1", repository_id="repo1", question_type="Q1",
        split="public", prompt="p", parameters={}, ground_truth={},
        tolerance={}, grading_rubric="r"
    )
    print("NO_ERROR")
except TypeError as e:
    print(f"TYPE_ERROR: {e}")
"""
        res_sig = run_cli(["-c", test_script_sig])
        assert "TYPE_ERROR: BenchmarkTask.__init__() got an unexpected keyword argument 'repository_id'" in res_sig.stdout

        # Bug 2: Task filtering
        test_script_filter = """
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    tasks = [json.loads(l) for l in f if l.strip()]
inf_exact = [t for t in tasks if t['ground_truth'].get('status') == 'INFEASIBLE' or 'infeasible' in t['task_id'].lower()]
inf_certified = [t for t in tasks if t['ground_truth'].get('status') == 'INFEASIBLE_CERTIFIED']
print(f"EXACT_COUNT: {len(inf_exact)}, CERTIFIED_COUNT: {len(inf_certified)}")
"""
        res_filter = run_cli(["-c", test_script_filter, str(tasks_public_path)])
        assert "EXACT_COUNT: 0, CERTIFIED_COUNT: 1" in res_filter.stdout


# ===========================================================================
# 2. TIER 4 ADVERSARIAL CHALLENGES
# ===========================================================================

class TestTier4AdversarialChallenges:
    """Stress-tests for Tier 4: Amnesia Progression & Anti-Goodhart Rules."""

    def test_challenge_tier4_amnesia_progression_mutation(self, tasks_public_path, parse_jsonl):
        """EMPIRICAL CHALLENGE 2.1: Stress-test amnesia session progression detection.
        
        Verifies:
        - When properly filtered by repository (golden_case), sequence covers Q1..Q8 across turns 1..8.
        - Injected skipped turn (e.g. Turn 2 missing) is detected.
        - Injected mismatched session ID (e.g. Turn 4 has wrong session_id) is detected.
        - Contradiction rate (CR) calculation: fact contradiction in Turn 8 is detected.
        """
        tasks = parse_jsonl(tasks_public_path)

        # Clean golden_case session extraction
        gc_tasks = [t for t in tasks if t["repo_id"] == "golden_case"]
        gc_tasks.sort(key=lambda t: t.get("turn_index", 0))

        assert len(gc_tasks) == 8, f"Expected 8 golden_case tasks, got {len(gc_tasks)}"
        expected_turns = [1, 2, 3, 4, 5, 6, 7, 8]
        actual_turns = [t["turn_index"] for t in gc_tasks]
        assert actual_turns == expected_turns, f"Expected turns {expected_turns}, got {actual_turns}"

        # MUTATION A: Skipped Turn (drop Turn 2)
        mutated_skipped = [t for t in gc_tasks if t["turn_index"] != 2]
        skipped_detected = False
        for idx, t in enumerate(mutated_skipped, start=1):
            if t["turn_index"] != idx:
                skipped_detected = True
                break
        assert skipped_detected is True, "Skipped turn mutation must be detected"

        # MUTATION B: Mismatched Session ID in Turn 4
        mutated_session = [dict(t) for t in gc_tasks]
        mutated_session[3]["session_id"] = "corrupted_session_id_x"
        base_session_id = mutated_session[0]["session_id"]
        mismatch_detected = any(t["session_id"] != base_session_id for t in mutated_session)
        assert mismatch_detected is True, "Mismatched session ID mutation must be detected"

        # MUTATION C: Contradiction Rate (CR) Spike
        facts = {"dead_nodes": ["brand_awareness"], "fee_rate": 0.03}
        contender_turn8_facts = {"dead_nodes": ["brand_awareness"], "fee_rate": 0.07}  # Contradiction!

        contradictions = 0
        total_checked = 0
        for k, v in facts.items():
            total_checked += 1
            if contender_turn8_facts.get(k) != v:
                contradictions += 1

        contradiction_rate = contradictions / total_checked
        assert contradiction_rate == 0.5, f"Expected 50% contradiction rate, got {contradiction_rate}"
        assert contradiction_rate > 0.0, "Contradiction spike must be detected by oracle"

    def test_challenge_tier4_anti_goodhart_mutation(self, report_script, run_cli):
        """EMPIRICAL CHALLENGE 2.2: Test susceptibility of test_anti_goodhart_rules_enforcement to evasion.
        
        Demonstrates that:
        1. A report script containing a composite single score will PASS the existing test
           if the word 'anti_goodhart' is also present in comments.
        2. A report script omitting the delta_B1 relative score column will PASS the existing test
           as long as the letters 'B1' exist anywhere in the code.
        """
        code_mutant = """
# Implementation of report.py with anti_goodhart notes
# Anti_Goodhart Rule 1 violation: we compute a composite single score!
def compute_single_score():
    return 88.0

contenders = ['B0', 'B1', 'B2', 'B3']
sealed = True
false_confidence = False
"""
        # Existing checks from test_tier4_scenarios.py line 135:
        rule1 = ("single_score" not in code_mutant.lower()) or ("anti_goodhart" in code_mutant.lower())
        rule2 = "B1" in code_mutant
        rule3 = "sealed" in code_mutant
        rule4 = "false_confidence" in code_mutant

        assert rule1 is True, "Rule 1 check was evaded by presence of 'anti_goodhart' in comments"
        assert rule2 is True, "Rule 2 check was evaded by substring 'B1' without checking table output"
        assert rule3 is True, "Rule 3 check was evaded by substring 'sealed'"
        assert rule4 is True, "Rule 4 check was evaded by substring 'false_confidence'"


# ===========================================================================
# 3. INTEGRITY ADVERSARIAL CHALLENGES
# ===========================================================================

class TestIntegrityAdversarialChallenges:
    """Stress-tests for Code & Deliverables Integrity."""

    def test_challenge_integrity_cge_directory_guardrail(self, repo_root, cge_dir):
        """EMPIRICAL CHALLENGE 3.1: Confirm that unauthorized changes in cge/ are strictly caught."""
        test_file = cge_dir / "challenger2_mutation_canary.tmp"
        try:
            test_file.write_text("mutation canary", encoding="utf-8")
            status_proc = subprocess.run(
                ["git", "status", "--porcelain", "cge/"],
                cwd=str(repo_root),
                capture_output=True,
                text=True
            )
            raw_output = status_proc.stdout.strip()
            lines = [l.strip() for l in raw_output.splitlines() if l.strip()]
            new_or_untracked = [l for l in lines if l.startswith("??")]
            assert any("challenger2_mutation_canary.tmp" in l for l in new_or_untracked), (
                "Canary file in cge/ must appear in git untracked files"
            )
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_challenge_integrity_placeholder_token_regex(self):
        """EMPIRICAL CHALLENGE 3.2: Verify placeholder detection sensitivity.
        
        Tests:
        - TODO, TBD, FIXME, [INSERT], [placeholder], Lorem ipsum are caught.
        - Spanish word 'todo' (lowercase) is NOT falsely flagged.
        """
        text_clean = "En este documento analizamos todo el flujo de trabajo."
        text_with_todo = "En este documento TODO: completar analisis."
        text_with_tbd = "El valor es TBD segun el diseno."

        placeholders_case_sensitive = [r"\bTODO\b", r"\bTBD\b", r"\bFIXME\b", r"\[INSERT\]", r"\[placeholder\]"]

        matches_clean = []
        for p in placeholders_case_sensitive:
            matches_clean.extend(re.findall(p, text_clean))
        assert len(matches_clean) == 0, "Spanish 'todo' must not trigger false positive"

        matches_todo = []
        for p in placeholders_case_sensitive:
            matches_todo.extend(re.findall(p, text_with_todo))
        assert len(matches_todo) == 1, "Uppercase TODO must be caught"

        matches_tbd = []
        for p in placeholders_case_sensitive:
            matches_tbd.extend(re.findall(p, text_with_tbd))
        assert len(matches_tbd) == 1, "Uppercase TBD must be caught"

    def test_challenge_integrity_tolerance_schema_bug(self, tasks_public_path, parse_jsonl):
        """EMPIRICAL CHALLENGE 3.3: Demonstrate the tolerance dict vs float schema bug in test_integrity.py.
        
        In test_integrity.py lines 218 & 282:
        - Line 218: `isinstance(task['tolerance'], (float, int))` fails because task['tolerance'] is a dict.
        - Line 282: `0.0 <= tol <= 0.1` raises TypeError: '<=' not supported between float and dict.
        """
        tasks = parse_jsonl(tasks_public_path)
        first_task = tasks[0]
        tol = first_task.get("tolerance")

        assert isinstance(tol, dict), f"Task tolerance in JSONL is a dict: {tol}"
        assert not isinstance(tol, (float, int)), "Tolerance is not a float or int"

        # Verify that evaluating 0.0 <= tol raises TypeError
        with pytest.raises(TypeError, match="'<=' not supported between instances of 'float' and 'dict'"):
            _ = 0.0 <= tol <= 0.1
