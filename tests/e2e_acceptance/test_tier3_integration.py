"""
tests/e2e_acceptance/test_tier3_integration.py
Tier 3: Cross-Feature Combinations & Pairwise Interactions Acceptance Tests.
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import pytest

pytestmark = [pytest.mark.tier3, pytest.mark.e2e]


# ---------------------------------------------------------------------------
# Test 3.1: Ground Truth Feeds Grader with Perfect Score
# ---------------------------------------------------------------------------
def test_pairwise_ground_truth_feeds_grader_perfect_score(
    run_cli, repo_root, tasks_public_path, parse_jsonl
):
    """
    PAIRWISE INTERACTION: ground_truth.py verification -> test_grader.py evaluation.
    Verifies that for tasks verified by ground_truth.py, a contender response exactly
    matching the verified ground truth yields a 100% score (1.0), status PASS/CERTIFIED,
    0.0 constraint violation, and no false confidence flag across all Q1-Q8 categories.
    """
    # 1. Verify that ground_truth.py runs and validates the public dataset
    gt_res = run_cli(["benchmark/baseline_bench/ground_truth.py", "--verify"], timeout=60.0)
    gt_res.assert_success("ground_truth.py --verify must pass before evaluating grader coupling")

    # 2. Load public tasks
    tasks = parse_jsonl(tasks_public_path)
    assert len(tasks) > 0, "Public task dataset cannot be empty"

    # 3. Create an in-memory test script to evaluate ground truth responses through grader
    eval_script = """
import sys, json
from benchmark.baseline_bench.test_grader import grade_response
from benchmark.baseline_bench.harness import BenchmarkTask, ContenderResponse

tasks_file = sys.argv[1]
with open(tasks_file, 'r', encoding='utf-8') as f:
    tasks = [json.loads(line) for line in f if line.strip()]

failures = []
for t in tasks:
    task_obj = BenchmarkTask.from_dict(t)
    gt = t['ground_truth']
    q = t['question_type']
    
    # Synthesize perfect contender output from ground truth
    if q == 'Q8':
        parsed = {f['fact_id']: True for f in gt.get('retention_facts', [])}
        if t.get('repo_id') == 'economic_dispatch':
            parsed.update({'P1': 148.08, 'P2': 140.38, 'P3': 11.54})
    elif q == 'Q6' and gt.get('status') == 'INFEASIBLE_CERTIFIED':
        parsed = {'status': 'INFEASIBLE_CERTIFIED', 'iis_constraints': gt.get('iis_constraints', [])}
    elif q == 'Q6' and 'primary_value' in gt:
        parsed = gt.get('primary_value')
    elif 'optimal_controls' in gt:
        parsed = {'optimal_objective': gt.get('optimal_objective'), 'optimal_controls': gt.get('optimal_controls')}
    elif 'optimal_objective' in gt:
        parsed = gt.get('optimal_objective')
    elif 'primary_value' in gt:
        parsed = gt.get('primary_value')
    elif 'profit' in gt:
        parsed = gt.get('profit')
    else:
        parsed = gt

    claim = 'INFEASIBLE_CERTIFIED' if gt.get('status') == 'INFEASIBLE_CERTIFIED' else 'CERTAIN'
    perfect_response = ContenderResponse(
        contender_id="B3",
        task_id=t['task_id'],
        session_id=t.get('session_id', 's0'),
        turn_index=t.get('turn_index', 1),
        raw_answer=json.dumps(t['ground_truth']),
        parsed_output=parsed,
        confidence_claim=claim,
        tokens_input=100,
        tokens_output=50,
        latency_seconds=0.1,
        tool_calls_count=1,
        memory_peak_bytes=1024,
        sandbox_errors=[]
    )
    score_result = grade_response(task_obj, perfect_response)
    if score_result.score < 0.999 or score_result.false_confidence:
        failures.append((t['task_id'], t['question_type'], score_result))

if failures:
    print(f"FAILED on {len(failures)} tasks: {failures[:5]}")
    sys.exit(1)
print("SUCCESS: All ground truths scored 100%")
sys.exit(0)
"""
    res = run_cli(["-c", eval_script, str(tasks_public_path)], timeout=30.0)
    res.assert_success("Every verified ground truth must achieve 1.0 score in grader")
    assert "SUCCESS: All ground truths scored 100%" in res.stdout


# ---------------------------------------------------------------------------
# Test 3.2: Perturbed Ground Truth Triggers Grader Boundary Penalties
# ---------------------------------------------------------------------------
def test_pairwise_grader_perturbed_tolerance_violation(run_cli, tasks_public_path):
    """
    PAIRWISE INTERACTION: Perturbation outside tolerance -> Grader rejection.
    Verifies that perturbing numerical outputs by +1.2 * rel_tol causes the grader
    to assign score 0.0, while perturbation within 0.8 * rel_tol retains score 1.0.
    """
    perturb_script = """
import sys, json
from benchmark.baseline_bench.test_grader import grade_response
from benchmark.baseline_bench.harness import BenchmarkTask, ContenderResponse

tasks_file = sys.argv[1]
with open(tasks_file, 'r', encoding='utf-8') as f:
    tasks = [json.loads(line) for line in f if line.strip()]

# Find a scalar numerical task (Q3 or Q5)
num_tasks = [
    t for t in tasks
    if t['question_type'] in ('Q3', 'Q5') and (
        isinstance(t['ground_truth'].get('profit'), (int, float)) or
        isinstance(t['ground_truth'].get('optimal_objective'), (int, float))
    )
]
assert len(num_tasks) > 0, "Must have at least one numerical Q3/Q5 task"
task = num_tasks[0]

rel_tol = task['tolerance'].get('rel_tol', 0.01) if isinstance(task['tolerance'], dict) else task.get('tolerance', 0.01)
true_val = task['ground_truth'].get('profit') or task['ground_truth'].get('optimal_objective')

task_obj = BenchmarkTask.from_dict(task)

# 1. Just INSIDE tolerance (e.g. 0.8 * rel_tol)
inside_val = true_val * (1.0 + 0.8 * rel_tol)
resp_inside = ContenderResponse(
    contender_id="B1", task_id=task['task_id'], session_id="s1", turn_index=1,
    raw_answer=str(inside_val), parsed_output=inside_val, confidence_claim=None,
    tokens_input=50, tokens_output=10, latency_seconds=0.1, tool_calls_count=0,
    memory_peak_bytes=0, sandbox_errors=[]
)
score_inside = grade_response(task_obj, resp_inside)
assert abs(score_inside.score - 1.0) < 1e-6, f"Expected 1.0 inside tolerance, got {score_inside.score}"

# 2. Just OUTSIDE tolerance (e.g. 1.2 * rel_tol)
outside_val = true_val * (1.0 + 1.2 * rel_tol)
resp_outside = ContenderResponse(
    contender_id="B1", task_id=task['task_id'], session_id="s1", turn_index=1,
    raw_answer=str(outside_val), parsed_output=outside_val, confidence_claim=None,
    tokens_input=50, tokens_output=10, latency_seconds=0.1, tool_calls_count=0,
    memory_peak_bytes=0, sandbox_errors=[]
)
score_outside = grade_response(task_obj, resp_outside)
assert abs(score_outside.score - 0.0) < 1e-6, f"Expected 0.0 outside tolerance, got {score_outside.score}"

print("SUCCESS: Tolerance boundary correctly enforced")
"""
    res = run_cli(["-c", perturb_script, str(tasks_public_path)], timeout=15.0)
    res.assert_success("Grader must cleanly distinguish inside vs outside tolerance")


# ---------------------------------------------------------------------------
# Test 3.3: Infeasible Tasks with False Confidence Claims Trigger Security Veto
# ---------------------------------------------------------------------------
def test_pairwise_grader_false_confidence_veto(run_cli, tasks_public_path):
    """
    PAIRWISE INTERACTION: Infeasible problem + CERTIFIED_OPTIMAL claim -> False Confidence Veto.
    Verifies that claiming certainty on an INFEASIBLE or constraint-violating task
    triggers `false_confidence=True` and a score of 0.0 (Anti-Goodhart Rule 4).
    """
    veto_script = """
import sys, json
from benchmark.baseline_bench.test_grader import grade_response
from benchmark.baseline_bench.harness import BenchmarkTask, ContenderResponse

tasks_file = sys.argv[1]
with open(tasks_file, 'r', encoding='utf-8') as f:
    tasks = [json.loads(line) for line in f if line.strip()]

# Find an infeasible task (Q6 IIS or infeasible OR-bench)
infeasible_tasks = [
    t for t in tasks 
    if t['ground_truth'].get('status', '').startswith('INFEASIBLE') 
    or 'infeasible' in str(t.get('parameters', {})).lower() 
    or 'infeasible' in t['task_id'].lower()
]
assert len(infeasible_tasks) > 0, "Must have at least one infeasible task"
task = infeasible_tasks[0]

task_obj = BenchmarkTask.from_dict(task)

# Contender falsely claims CERTIFIED_OPTIMAL with arbitrary control values
resp_false = ContenderResponse(
    contender_id="B0", task_id=task['task_id'], session_id="s_veto", turn_index=1,
    raw_answer="Optimal solution found: profit=99999", parsed_output=99999.0,
    confidence_claim="CERTIFIED_OPTIMAL", tokens_input=100, tokens_output=20,
    latency_seconds=0.2, tool_calls_count=0, memory_peak_bytes=0, sandbox_errors=[]
)

score_result = grade_response(task_obj, resp_false)
assert score_result.score == 0.0, f"Expected 0.0 score, got {score_result.score}"
assert score_result.false_confidence is True, "False confidence veto must be triggered"

print("SUCCESS: False confidence veto correctly triggered")
"""
    res = run_cli(["-c", veto_script, str(tasks_public_path)], timeout=15.0)
    res.assert_success("False confidence veto must be triggered on infeasible tasks")


# ---------------------------------------------------------------------------
# Test 3.4: Grader Results Feed Report Generator Pipeline
# ---------------------------------------------------------------------------
def test_pairwise_grader_to_report_pipeline(run_cli, repo_root):
    """
    PAIRWISE INTERACTION: Contender evaluation -> report.py Markdown table generation.
    Verifies that executing `report.py --baseline baseline_v0` consumes baseline results
    and outputs the canonical Markdown tables.
    """
    res = run_cli(["benchmark/baseline_bench/report.py", "--baseline", "baseline_v0"], timeout=60.0)
    res.assert_success("report.py --baseline baseline_v0 must execute successfully")

    # Verify table headers exist in stdout
    stdout = res.stdout
    assert "B0" in stdout and "B1" in stdout and "B2" in stdout and "B3" in stdout, "All 4 contenders must be represented"
    assert "Q1" in stdout and "Q8" in stdout, "Taxonomy categories must be represented"
    assert "baseline_v0" in stdout or "BASELINE" in stdout, "Baseline tag must be referenced"


# ---------------------------------------------------------------------------
# Test 3.5: Public vs Sealed Dataset Strict Parity & Isolation
# ---------------------------------------------------------------------------
def test_pairwise_public_vs_sealed_parity(tasks_public_path, tasks_sealed_path, parse_jsonl):
    """
    PAIRWISE INTERACTION: tasks_public.jsonl vs tasks_sealed.jsonl.
    Verifies the 50/50 balance, ensures zero task_id collisions (disjoint sets),
    and validates that both splits exercise all 8 categories Q1..Q8.
    """
    public_tasks = parse_jsonl(tasks_public_path)
    sealed_tasks = parse_jsonl(tasks_sealed_path)

    # 1. 50/50 split count parity
    assert len(public_tasks) == len(sealed_tasks), (
        f"Public task count ({len(public_tasks)}) must equal sealed task count ({len(sealed_tasks)})"
    )

    # 2. Zero task_id overlap (strictly disjoint)
    public_ids = {t["task_id"] for t in public_tasks}
    sealed_ids = {t["task_id"] for t in sealed_tasks}
    overlap = public_ids.intersection(sealed_ids)
    assert len(overlap) == 0, f"Public and sealed datasets must have zero overlap. Found: {overlap}"

    # 3. Question type coverage in both splits (Q1 through Q8)
    public_qtypes = {t["question_type"] for t in public_tasks}
    sealed_qtypes = {t["question_type"] for t in sealed_tasks}
    expected_qtypes = {f"Q{i}" for i in range(1, 9)}
    assert expected_qtypes.issubset(public_qtypes), f"Public tasks missing: {expected_qtypes - public_qtypes}"
    assert expected_qtypes.issubset(sealed_qtypes), f"Sealed tasks missing: {expected_qtypes - sealed_qtypes}"


# ---------------------------------------------------------------------------
# Test 3.6: Multi-Repository Ground Truth Oracle Isolation
# ---------------------------------------------------------------------------
def test_pairwise_multi_repo_oracle_isolation(run_cli, tasks_public_path):
    """
    PAIRWISE INTERACTION: Cross-repository execution isolation.
    Verifies that running ground truth computation on multiple distinct repos
    does not bleed state or cause namespace/import collisions across modules.
    """
    isolation_script = """
import sys, json
from benchmark.baseline_bench.ground_truth import compute_and_verify_task

tasks_file = sys.argv[1]
with open(tasks_file, 'r', encoding='utf-8') as f:
    tasks = [json.loads(line) for line in f if line.strip()]

# Select tasks from different repositories
repos = {t.get('repo_id') or t.get('repo') for t in tasks}
assert len(repos) >= 2, f"Expected tasks from at least 2 repos, got {repos}"

repo_list = sorted(list(repos))[:3]
sample_tasks = []
for r in repo_list:
    t = next(task for task in tasks if (task.get('repo_id') or task.get('repo')) == r)
    sample_tasks.append(t)

# Run in forward sequence
forward_results = []
for t in sample_tasks:
    ok, err, res = compute_and_verify_task(t)
    assert ok, f"Task {t['task_id']} failed in forward sequence: {err}"
    forward_results.append(res)

# Run in reverse sequence
reverse_results = []
for t in reversed(sample_tasks):
    ok, err, res = compute_and_verify_task(t)
    assert ok, f"Task {t['task_id']} failed in reverse sequence: {err}"
    reverse_results.append(res)

reverse_results.reverse()
assert forward_results == reverse_results, "Execution order must not alter oracle computation outputs"
print("SUCCESS: Multi-repository oracle execution is order-invariant")
"""
    res = run_cli(["-c", isolation_script, str(tasks_public_path)], timeout=30.0)
    res.assert_success("Multi-repository oracle execution must be order-invariant")


# ---------------------------------------------------------------------------
# Test 3.7: Cryptographic Manifest Cross-Verification
# ---------------------------------------------------------------------------
def test_pairwise_manifest_cross_check_with_artifacts(
    repo_root, baseline_bench_dir, manifest_path, compute_sha256
):
    """
    PAIRWISE INTERACTION: baseline_v0_manifest.json vs actual files on disk.
    Verifies that SHA-256 hashes recorded in the baseline manifest match the actual
    hashes of ground_truth.py, test_grader.py, report.py, and both task datasets.
    """
    assert manifest_path.exists(), f"Manifest file missing: {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    file_hashes = manifest.get("file_hashes", {})
    assert len(file_hashes) > 0, "Manifest must contain file_hashes dictionary"

    for rel_path, expected_hash in file_hashes.items():
        target_file = repo_root / rel_path
        if not target_file.exists():
            target_file = baseline_bench_dir / rel_path
        assert target_file.exists(), f"Manifest references non-existent file: {rel_path}"
        actual_hash = compute_sha256(target_file)
        assert actual_hash.lower() == expected_hash.lower(), (
            f"Cryptographic hash mismatch for {rel_path}:\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}"
        )
