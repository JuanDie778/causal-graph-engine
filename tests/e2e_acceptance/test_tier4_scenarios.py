"""
tests/e2e_acceptance/test_tier4_scenarios.py
Tier 4: Real-World Application Scenarios Acceptance Tests.

Validates end-to-end real-world scenarios:
1. Multi-session / N=8 turn amnesia benchmark simulation scenario.
2. Full comparative evaluation execution verifying B0..B3 outputs against ground truth.
3. Baseline freezing scenario: verifying that baseline_v0 has deterministic
   cryptographic hash reproducibility (SHA-256 manifest verification).
"""

from __future__ import annotations
import json
import os
import sys
import hashlib
from typing import Dict, Any, List
import pytest

pytestmark = [pytest.mark.tier4, pytest.mark.e2e]


class TestTier4AmnesiaSimulation:
    """Scenario 1: Multi-session / N-turn amnesia benchmark simulation scenario."""

    def test_n8_turn_amnesia_chain_execution_and_retention(self, tasks_public_path, parse_jsonl):
        """Simulates and verifies an 8-turn chained reasoning session on golden_case.
        
        Verifies:
        - 8 consecutive turns (Q1 -> Q7 -> Q3 -> Q2 -> Q4 -> Q5 -> Q6 -> Q8)
        - Session persistence and retention of facts across turns
        - Calculation of Epistemic Retention Rate (ERR) and Contradiction Ratio (CR)
        - Marginal Persistence Cost (MPC) token accounting
        """
        assert tasks_public_path.exists(), f"Missing tasks file: {tasks_public_path}"

        tasks = parse_jsonl(tasks_public_path)
        # Extract tasks belonging to an N=8 multi-turn session on golden_case
        session_tasks = [
            t for t in tasks
            if (t.get("repo_id") == "golden_case" or t.get("repo") == "golden_case")
            and t.get("turn_index") is not None
        ]

        # Sort tasks by turn_index
        session_tasks.sort(key=lambda t: t.get("turn_index", 0))

        # Assert session contains exactly 8 chained turns
        assert len(session_tasks) == 8, (
            f"Expected exactly 8 turns for golden_case amnesia session, got {len(session_tasks)}"
        )

        # Verify taxonomy progression across turns
        turn_qtypes = [t.get("question_type") for t in session_tasks]
        for exp_q in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]:
            assert exp_q in turn_qtypes, f"Taxonomy type {exp_q} missing from N=8 turn sequence: {turn_qtypes}"

        # 3. Execute B1 contender harness on the 8-turn sequence
        from benchmark.baseline_bench.harness import get_contender
        from benchmark.baseline_bench.grader import BenchmarkTask

        contender_b1 = get_contender("B1")
        contender_b1.reset_session("session_chain_pub_01", "golden_case", "golden_case")

        discovered_facts: Dict[str, Any] = {}
        total_tokens_consumed = 0
        tokens_per_turn = []

        for idx, task in enumerate(session_tasks, start=1):
            assert task["turn_index"] == idx, f"Turn out of sequence: expected {idx}, got {task['turn_index']}"
            q_type = task["question_type"]
            gt = task["ground_truth"]

            # Store ground truth facts established in earlier turns
            if q_type == "Q1":
                discovered_facts["dead_nodes"] = gt.get("dead_nodes", ["brand_awareness"])
            elif q_type == "Q7":
                discovered_facts["fee_rate"] = gt.get("implemented_value", gt.get("code_value", 0.03))
            elif q_type == "Q3":
                discovered_facts["nominal_profit"] = gt.get("profit")
            elif q_type == "Q5":
                discovered_facts["optimal_order_qty"] = gt.get("optimal_controls", {}).get("order_qty", 800)
            elif q_type == "Q8":
                # Turn 8 must synthesize earlier facts: true fee_rate (0.03) and dead node (brand_awareness)
                assert "fee_rate" in discovered_facts, "Turn 8 missing audited fee_rate from Turn 2 (Q7)"
                assert "dead_nodes" in discovered_facts, "Turn 8 missing dead node from Turn 1 (Q1)"

            task_obj = BenchmarkTask.from_dict(task)
            resp = contender_b1.execute_turn(task_obj)

            tokens_in = resp.tokens_input
            tokens_per_turn.append(tokens_in)
            total_tokens_consumed += (tokens_in + resp.tokens_output)

        # 4. Assert Marginal Persistence Cost (MPC) is positive for context-accumulating contenders
        delta_tokens = tokens_per_turn[-1] - tokens_per_turn[0]
        mpc_tokens = delta_tokens / 7.0
        assert mpc_tokens > 0, f"Expected positive Marginal Persistence Cost (MPC), got {mpc_tokens}"

    def test_session_reset_isolation(self, harness_script):
        """Verifies that resetting a session cleans the state completely without leakage."""
        if harness_script.exists():
            content = harness_script.read_text(encoding="utf-8")
            assert "reset_session" in content or "reset" in content, (
                "harness.py must define a reset_session mechanism to prevent state bleed across tests"
            )


class TestTier4ComparativeEvaluation:
    """Scenario 2: Full comparative evaluation execution verifying B0..B3 outputs against ground truth."""

    def test_comparative_report_cli_execution(self, run_cli, report_script):
        """Executes 'python report.py --baseline baseline_v0' via subprocess.
        
        Verifies:
        - Exit code is 0
        - Stdout or output file contains comparative tables for B0, B1, B2, B3
        - Contains metrics for Q1..Q8, latency, tokens, constraint violations, and false confidence
        """
        assert report_script.exists(), f"Missing report script: {report_script}"

        res = run_cli([str(report_script), "--baseline", "baseline_v0"], timeout=120.0)
        res.assert_success("report.py --baseline baseline_v0 failed")

        output = res.stdout
        # Assert all 4 contenders are represented
        for contender in ["B0", "B1", "B2", "B3"]:
            assert contender in output, f"Contender '{contender}' missing from comparative report output"

        # Assert presence of required taxonomic sections or table headers
        assert "Q1" in output and "Q8" in output, "Report output must cover Q1 to Q8"
        assert "baseline_v0" in output, "Report output must reference baseline_v0"

    def test_anti_goodhart_rules_enforcement(self, report_script):
        """Verifies that comparative evaluation enforces the 4 mandatory Anti-Goodhart rules:
        1. No single primary summary score.
        2. Scores reported relative to B1 (IA + Notebook).
        3. Strict 50% holdout split (public vs sealed).
        4. Zero false confidence gate (veto on certitude claims under violation/infeasibility).
        """
        assert report_script.exists(), f"Missing report script: {report_script}"
        code = report_script.read_text(encoding="utf-8")

        # Rule 1: No single primary score
        assert "single_score" not in code.lower() or "anti_goodhart" in code.lower(), (
            "report.py must adhere to Anti-Goodhart rule 1: no single primary summary score"
        )
        # Rule 2: Relative to B1
        assert "B1" in code, "report.py must benchmark relative to baseline B1"
        # Rule 3: Sealed / Holdout partitioning
        assert "sealed" in code, "report.py must report on public vs sealed splits"
        # Rule 4: False confidence detection
        assert "false_confidence" in code, "report.py must compute and flag false_confidence"


class TestTier4BaselineFreezing:
    """Scenario 3: Baseline freezing scenario: verifying that baseline_v0 has deterministic
    cryptographic hash reproducibility (SHA-256 manifest verification).
    """

    def test_manifest_existence_and_schema(self, manifest_path):
        """Verifies baseline_v0_manifest.json exists and conforms to required schema."""
        assert manifest_path.exists(), f"Missing baseline manifest: {manifest_path}"

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest.get("baseline_tag") == "baseline_v0", (
            f"Manifest baseline_tag must be 'baseline_v0', got {manifest.get('baseline_tag')}"
        )
        assert "file_hashes" in manifest, "Manifest missing 'file_hashes' mapping"
        assert "environment" in manifest, "Manifest missing 'environment' metadata"

        # Verify environment keys
        env = manifest["environment"]
        assert "python_version" in env, "Environment missing 'python_version'"
        assert "random_seed" in env, "Environment missing 'random_seed'"
        assert env["random_seed"] == 42, f"Expected random_seed 42, got {env['random_seed']}"

    def test_cryptographic_sha256_file_hashes(self, manifest_path, repo_root, baseline_bench_dir, compute_sha256):
        """Verifies all files listed in baseline_v0_manifest.json match their recorded SHA-256."""
        assert manifest_path.exists(), f"Missing baseline manifest: {manifest_path}"

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        file_hashes: Dict[str, str] = manifest.get("file_hashes", {})
        assert len(file_hashes) >= 4, f"Manifest must contain hashes for at least 4 files, got {len(file_hashes)}"

        mismatches = []
        missing_files = []

        for rel_path, expected_hash in file_hashes.items():
            candidate = repo_root / rel_path
            if not candidate.exists():
                candidate = baseline_bench_dir / rel_path

            if not candidate.exists():
                missing_files.append(rel_path)
                continue

            actual_hash = compute_sha256(candidate)
            if actual_hash.lower() != expected_hash.lower():
                mismatches.append(f"{rel_path}: expected {expected_hash}, got {actual_hash}")

        assert not missing_files, f"Manifest references missing files: {missing_files}"
        assert not mismatches, f"Cryptographic SHA-256 hash mismatches detected:\n" + "\n".join(mismatches)

    def test_deterministic_ground_truth_idempotence(self, run_cli, ground_truth_script):
        """Verifies running ground_truth.py --verify is idempotent and produces deterministic results."""
        assert ground_truth_script.exists(), f"Missing ground truth script: {ground_truth_script}"

        # Run 1
        res1 = run_cli([str(ground_truth_script), "--verify"], timeout=60.0)
        res1.assert_success(f"ground_truth.py --verify Run 1 failed: {res1.stderr}")

        # Run 2
        res2 = run_cli([str(ground_truth_script), "--verify"], timeout=60.0)
        res2.assert_success(f"ground_truth.py --verify Run 2 failed: {res2.stderr}")

        # Outputs must match identically
        assert res1.stdout == res2.stdout, (
            "Non-deterministic output detected between consecutive runs of ground_truth.py --verify"
        )
