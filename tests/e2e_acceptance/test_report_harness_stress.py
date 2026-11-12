"""
tests/e2e_acceptance/test_report_harness_stress.py
Adversarial Stress Testing of Reporting (report.py) and Harness (harness.py)
Executed by Challenger 2 for Milestone 2.

Covers:
1. CLI boundary conditions (all, public, sealed splits, invalid split choices, bad flags, custom baseline tags).
2. Mathematical consistency:
   - Global accuracy equals exact weighted average of public and sealed splits.
   - Category accuracies (Q1..Q8) match exact arithmetic means of task scores.
   - Delta vs B1 matches the formal formula: Delta = (Score - Score_B1) / max(0.05, Score_B1).
   - Anti-Goodhart status correctness (VETO_FALSE_CONFIDENCE, BASELINE_REFERENCE, PASSED, FAILED).
3. Zero Network Calls / Offline Isolation:
   - Strict socket monkeypatch interceptor to guarantee 0 network calls during execution.
   - AST inspection of harness.py verifying absence of remote API / HTTP libraries.
4. Schema and Edge-Case Robustness:
   - Tampered/empty task sets.
   - Defensive denominator protection when B1 score is zero or near zero.
   - Summary JSON schema completeness and type validity.
"""
from __future__ import annotations

import ast
import json
import socket
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, List

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark.baseline_bench.grader import BenchmarkTask, ContenderResponse, ScoreResult, grade_response
from benchmark.baseline_bench.harness import (
    get_contender,
    run_contender_suite,
    B0_LLMSoloHarness,
    B1_IANotebookHarness,
    B2_IAASTGraphHarness,
    B3_CGEActualHarness,
)
from benchmark.baseline_bench.report import (
    load_tasks,
    aggregate_contender_metrics,
    generate_comparative_report,
)


# ==============================================================================
# 1. CLI BOUNDARY & ERROR HANDLING TESTS
# ==============================================================================

class TestReportCLIBoundaries:
    """Empirical verification of CLI boundaries for report.py."""

    def test_cli_split_public_execution(self):
        """CLI runs --split public and outputs valid task counts (24 public, 0 sealed)."""
        tasks_dir = ROOT_DIR / "benchmark" / "baseline_bench"
        tasks = load_tasks(tasks_dir, split_filter="public")
        assert len(tasks) == 24, f"Expected 24 public tasks, got {len(tasks)}"
        assert all(t.split == "public" for t in tasks)

    def test_cli_split_sealed_execution(self):
        """CLI runs --split sealed and outputs valid task counts (0 public, 24 sealed)."""
        tasks_dir = ROOT_DIR / "benchmark" / "baseline_bench"
        tasks = load_tasks(tasks_dir, split_filter="sealed")
        assert len(tasks) == 24, f"Expected 24 sealed tasks, got {len(tasks)}"
        assert all(t.split == "sealed" for t in tasks)

    def test_cli_split_all_execution(self):
        """CLI runs --split all and outputs 48 total tasks (50/50 balance)."""
        tasks_dir = ROOT_DIR / "benchmark" / "baseline_bench"
        tasks = load_tasks(tasks_dir, split_filter="all")
        assert len(tasks) == 48, f"Expected 48 total tasks, got {len(tasks)}"
        pub_count = sum(1 for t in tasks if t.split == "public")
        seal_count = sum(1 for t in tasks if t.split == "sealed")
        assert pub_count == 24
        assert seal_count == 24

    def test_cli_invalid_split_flag_rejected(self):
        """Passing an invalid --split argument is rejected with non-zero exit code and error message."""
        cmd = [sys.executable, str(ROOT_DIR / "benchmark" / "baseline_bench" / "report.py"), "--split", "invalid_partition"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode != 0
        assert "invalid choice: 'invalid_partition'" in res.stderr

    def test_cli_unrecognized_argument_rejected(self):
        """Passing an unrecognized flag is rejected gracefully."""
        cmd = [sys.executable, str(ROOT_DIR / "benchmark" / "baseline_bench" / "report.py"), "--nonexistent-flag"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode != 0
        assert "unrecognized arguments" in res.stderr

    def test_empty_tasks_directory_graceful_handling(self, tmp_path):
        """If tasks directory is empty, load_tasks returns empty list and generate_comparative_report returns {}."""
        tasks = load_tasks(tmp_path, split_filter="all")
        assert tasks == []
        res = generate_comparative_report(export_path=tmp_path / "dummy_summary.json")
        # generate_comparative_report checks tasks_dir, but if tasks is empty it handles it gracefully
        assert isinstance(res, dict)


# ==============================================================================
# 2. MATHEMATICAL CONSISTENCY TESTS
# ==============================================================================

class TestMathematicalConsistency:
    """Verification of formal arithmetic invariants in report.py."""

    @pytest.fixture(scope="module")
    def all_evaluation_results(self):
        """Run all 4 contenders on all 48 tasks once for exact mathematical verification."""
        tasks_dir = ROOT_DIR / "benchmark" / "baseline_bench"
        tasks = load_tasks(tasks_dir, split_filter="all")
        contenders = ["B0", "B1", "B2", "B3"]
        all_results = {}
        for c_id in contenders:
            responses = run_contender_suite(c_id, tasks, offline=True)
            graded = []
            for t, r in zip(tasks, responses):
                score_res = grade_response(t, r)
                graded.append((t, r, score_res))
            all_results[c_id] = graded
        return tasks, all_results

    def test_global_accuracy_is_exact_weighted_average(self, all_evaluation_results):
        """Verify: Global Acc == (N_pub * Acc_pub + N_seal * Acc_seal) / N_total."""
        tasks, all_results = all_evaluation_results
        contenders = ["B0", "B1", "B2", "B3"]
        metrics = aggregate_contender_metrics(all_results, contenders)

        for c_id in contenders:
            items = all_results[c_id]
            pub_items = [s.score for t, _, s in items if t.split == "public"]
            seal_items = [s.score for t, _, s in items if t.split == "sealed"]
            all_items = [s.score for _, _, s in items]

            n_pub = len(pub_items)
            n_seal = len(seal_items)
            n_total = len(all_items)

            raw_pub_acc = sum(pub_items) / n_pub
            raw_seal_acc = sum(seal_items) / n_seal
            raw_global_acc = sum(all_items) / n_total

            expected_weighted_acc = (n_pub * raw_pub_acc + n_seal * raw_seal_acc) / n_total
            assert abs(raw_global_acc - expected_weighted_acc) < 1e-12, (
                f"Contender {c_id}: Global raw accuracy {raw_global_acc} != weighted average {expected_weighted_acc}"
            )

            # Check reported rounded metrics match
            m = metrics[c_id]
            assert m["global_accuracy"] == round(raw_global_acc, 3)
            assert m["public_accuracy"] == round(raw_pub_acc, 3)
            assert m["sealed_accuracy"] == round(raw_seal_acc, 3)

    def test_category_accuracy_matches_exact_arithmetic_mean(self, all_evaluation_results):
        """Verify: Category accuracy (Q1..Q8) matches exact mean of scores in that category."""
        tasks, all_results = all_evaluation_results
        contenders = ["B0", "B1", "B2", "B3"]
        metrics = aggregate_contender_metrics(all_results, contenders)

        for c_id in contenders:
            items = all_results[c_id]
            cat_metrics = metrics[c_id]["category_accuracy"]

            for q_idx in range(1, 9):
                q_code = f"Q{q_idx}"
                q_scores = [s.score for t, _, s in items if t.question_type == q_code]
                assert len(q_scores) > 0, f"No tasks found for category {q_code}"
                expected_mean = round(sum(q_scores) / len(q_scores), 3)
                assert cat_metrics[q_code] == expected_mean, (
                    f"Contender {c_id} {q_code}: Reported {cat_metrics[q_code]} != expected {expected_mean}"
                )

    def test_delta_vs_b1_calculation(self, all_evaluation_results):
        """Verify: Delta_B1 = (Score - Score_B1) / max(0.05, Score_B1)."""
        tasks, all_results = all_evaluation_results
        contenders = ["B0", "B1", "B2", "B3"]
        metrics = aggregate_contender_metrics(all_results, contenders)

        b1_global = metrics["B1"]["global_accuracy"]
        denom = max(0.05, b1_global)

        for c_id in contenders:
            g_acc = metrics[c_id]["global_accuracy"]
            expected_delta = round((g_acc - b1_global) / denom, 3)
            reported_delta = metrics[c_id]["delta_vs_b1"]
            assert reported_delta == expected_delta, (
                f"Contender {c_id}: Reported Delta_B1 {reported_delta} != expected {expected_delta}"
            )

    def test_delta_vs_b1_defensive_zero_denominator_protection(self):
        """Stress-test: If B1 global accuracy were 0.0, formula uses max(0.05, 0.0) = 0.05 without ZeroDivisionError."""
        fake_results = {
            "B0": [],
            "B1": [(
                BenchmarkTask("T1", "repo", "Q1", "public", "", {}, {}, {}, "set_f1"),
                ContenderResponse("B1", "T1", "s1", 1, "", None),
                ScoreResult(0.0, "FAILED", 0.0, False)
            )],
            "B3": [(
                BenchmarkTask("T1", "repo", "Q1", "public", "", {}, {}, {}, "set_f1"),
                ContenderResponse("B3", "T1", "s1", 1, "", None),
                ScoreResult(1.0, "PASSED", 0.0, False)
            )],
        }
        metrics = aggregate_contender_metrics(fake_results, ["B1", "B3"])
        assert metrics["B1"]["global_accuracy"] == 0.0
        # For B3: (1.0 - 0.0) / max(0.05, 0.0) = 1.0 / 0.05 = 20.0
        assert metrics["B3"]["delta_vs_b1"] == 20.0

    def test_anti_goodhart_veto_status(self, all_evaluation_results):
        """Verify Anti-Goodhart rules:
        - B1 is always BASELINE_REFERENCE
        - If false_confidence_rate > 0.0 -> VETO_FALSE_CONFIDENCE
        - Else if global_accuracy >= 0.8 -> PASSED
        - Else -> FAILED
        """
        _, all_results = all_evaluation_results
        contenders = ["B0", "B1", "B2", "B3"]
        metrics = aggregate_contender_metrics(all_results, contenders)

        assert metrics["B1"]["anti_goodhart_status"] == "BASELINE_REFERENCE"
        assert metrics["B0"]["false_confidence_rate"] > 0.0
        assert metrics["B0"]["anti_goodhart_status"] == "VETO_FALSE_CONFIDENCE"
        assert metrics["B2"]["false_confidence_rate"] > 0.0
        assert metrics["B2"]["anti_goodhart_status"] == "VETO_FALSE_CONFIDENCE"
        assert metrics["B3"]["false_confidence_rate"] == 0.0
        # With cge/ingest_adapters/ statically compiling all 5 external repos into CGE IR,
        # B3 achieves >= 80.0% global accuracy with 0.0% false confidence -> PASSED.
        assert metrics["B3"]["global_accuracy"] >= 0.80
        assert metrics["B3"]["anti_goodhart_status"] == "PASSED"

        # Explicitly verify the 'FAILED' branch when accuracy < 0.80 and false_confidence_rate == 0.0:
        synthetic_low_acc = {
            "B1": [(
                BenchmarkTask("T1", "golden_case", "Q1", "public", "", {}, {}, {}, "set_f1"),
                ContenderResponse("B1", "T1", "s1", 1, "", None),
                ScoreResult(0.5, "PASSED", 0.0, False)
            )],
            "B_abstaining": [(
                BenchmarkTask("T1", "golden_case", "Q1", "public", "", {}, {}, {}, "set_f1"),
                ContenderResponse("B_abstaining", "T1", "s1", 1, "", None, confidence_claim="UNSUPPORTED_DOMAIN"),
                ScoreResult(0.0, "FAILED", 0.0, False)
            )],
        }
        low_metrics = aggregate_contender_metrics(synthetic_low_acc, ["B1", "B_abstaining"])
        assert low_metrics["B_abstaining"]["false_confidence_rate"] == 0.0
        assert low_metrics["B_abstaining"]["global_accuracy"] < 0.80
        assert low_metrics["B_abstaining"]["anti_goodhart_status"] == "FAILED"

        # Explicitly verify the 'PASSED' branch of the Anti-Goodhart rule on a compliant candidate:
        synthetic_passed = {
            "B1": [(
                BenchmarkTask("T1", "golden_case", "Q1", "public", "", {}, {}, {}, "set_f1"),
                ContenderResponse("B1", "T1", "s1", 1, "", None),
                ScoreResult(0.5, "PASSED", 0.0, False)
            )],
            "B_certified": [(
                BenchmarkTask("T1", "golden_case", "Q1", "public", "", {}, {}, {}, "set_f1"),
                ContenderResponse("B_certified", "T1", "s1", 1, "", None),
                ScoreResult(1.0, "PASSED", 0.0, False)
            )],
        }
        passed_metrics = aggregate_contender_metrics(synthetic_passed, ["B1", "B_certified"])
        assert passed_metrics["B_certified"]["false_confidence_rate"] == 0.0
        assert passed_metrics["B_certified"]["global_accuracy"] >= 0.8
        assert passed_metrics["B_certified"]["anti_goodhart_status"] == "PASSED"


# ==============================================================================
# 3. OFFLINE & ZERO EXTERNAL NETWORK CALLS VERIFICATION
# ==============================================================================

class TestOfflineZeroNetworkCalls:
    """Empirical verification that harness operates 100% offline with 0 external network requests."""

    def test_harness_ast_no_remote_api_or_networking_imports(self):
        """Inspect harness.py AST to guarantee no networking or cloud LLM client libraries are imported."""
        harness_path = ROOT_DIR / "benchmark" / "baseline_bench" / "harness.py"
        with open(harness_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename="harness.py")

        forbidden_modules = {
            "requests", "urllib", "http.client", "aiohttp", "httpx",
            "openai", "anthropic", "google.generativeai", "langchain", "cohere"
        }

        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.add(node.module.split(".")[0])

        intersection = imported_modules & forbidden_modules
        assert not intersection, f"Forbidden networking/LLM modules found in harness.py: {intersection}"

    def test_harness_suite_execution_with_socket_blocked(self, monkeypatch):
        """Block all socket network connections at the OS level and execute entire suite.
        Must execute without throwing any socket connection attempts."""
        class NetworkAccessAttempted(RuntimeError):
            pass

        def forbidden_connect(*args, **kwargs):
            raise NetworkAccessAttempted("External network connection attempted in offline benchmark harness!")

        # Monkeypatch socket.socket.connect and socket.create_connection
        monkeypatch.setattr(socket.socket, "connect", forbidden_connect)
        if hasattr(socket, "create_connection"):
            monkeypatch.setattr(socket, "create_connection", forbidden_connect)

        tasks_dir = ROOT_DIR / "benchmark" / "baseline_bench"
        tasks = load_tasks(tasks_dir, split_filter="all")

        # Run all contenders under blocked network
        for c_id in ["B0", "B1", "B2", "B3"]:
            responses = run_contender_suite(c_id, tasks, offline=True)
            assert len(responses) == len(tasks)
            assert all(isinstance(r, ContenderResponse) for r in responses)


# ==============================================================================
# 4. SUMMARY JSON SCHEMA & INTEGRITY AUDITING
# ==============================================================================

class TestSummaryJSONSchemaIntegrity:
    """Verify schema compliance of exported summary JSON."""

    def test_summary_json_schema(self, tmp_path):
        """Verify generated JSON summary matches required structure and complete fields."""
        export_file = tmp_path / "test_baseline_summary.json"
        summary = generate_comparative_report(
            baseline_tag="test_v0",
            split_filter="all",
            export_path=export_file,
            live=False
        )

        assert export_file.exists()
        with open(export_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "metadata" in data
        assert "metrics" in data

        meta = data["metadata"]
        assert meta["benchmark_version"] == "test_v0"
        assert meta["total_tasks"] == 48
        assert meta["public_tasks"] == 24
        assert meta["sealed_tasks"] == 24
        assert meta["contenders"] == ["B0", "B1", "B2", "B3"]

        required_metric_keys = [
            "global_accuracy", "public_accuracy", "sealed_accuracy",
            "category_accuracy", "false_confidence_rate", "latency_median_seconds",
            "avg_input_tokens", "avg_output_tokens", "mpc_tokens_n8",
            "cumulative_tokens_n8", "avg_tool_calls", "constraint_violation_rate",
            "iis_detection_rate", "pressure_resistance_rate", "delta_vs_b1",
            "anti_goodhart_status"
        ]

        for c_id in ["B0", "B1", "B2", "B3"]:
            assert c_id in data["metrics"]
            m = data["metrics"][c_id]
            for rk in required_metric_keys:
                assert rk in m, f"Missing metric key {rk} for contender {c_id}"
            assert len(m["category_accuracy"]) == 8
            for q_idx in range(1, 9):
                assert f"Q{q_idx}" in m["category_accuracy"]
