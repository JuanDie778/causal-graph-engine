# TEST_INFRA — E2E Acceptance Testing Infrastructure

## 1. Executive Overview & Purpose

This document establishes the architecture, execution protocols, and governance standards for the independent End-to-End (E2E) Acceptance Test Suite for the **ActorDB Baseline Study, Zero-to-One Gap Map, and Metrics Framework**.

The primary mandate of the acceptance testing track is to provide an objective, uncompromised verification harness that certifies all requirements specified in:
- `ORIGINAL_REQUEST.md` (Requirements R1, R2, R3, R4 and core acceptance criteria)
- `SCOPE.md` (E2E Testing Track specifications)
- `PROJECT.md` (System interface contracts, mathematical rubrics, and Anti-Goodhart rules)

---

## 2. Core Testing Principles

### 2.1 Strict Opaque-Box, Requirement-Driven Testing
- **Subprocess Isolation**: All CLI tools (`ground_truth.py`, `report.py`) and external harnesses are tested via isolated `subprocess.run` executions utilizing `sys.executable` and an explicit UTF-8 environment (`PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8`).
- **Observable Contracts Only**: Verification evaluates public observable interfaces: exit codes (0 vs non-zero), standard output/stderr formatting, canonical JSONL dataset lines, JSON cryptographic manifests, and Markdown summary tables.
- **Zero In-Process Coupling**: Tests do not rely on monkey-patches, internal private attributes, or undocumented class hierarchies.

### 2.2 Four-Tier Acceptance Hierarchy
1. **Tier 1 — Feature Coverage**: Positive validation of all functional capabilities, CLI arguments, return codes, output formatting, schema compliance, document existence, and basic integrity across 7 core areas.
2. **Tier 2 — Boundary & Corner Cases**: Systematic fault-injection and boundary analysis covering missing flags, empty responses, tolerance margins ($\pm \epsilon$), constraint violation penalties ($\text{Viol} > 10^{-5}$), poison pills (dead nodes), and malformed payloads.
3. **Tier 3 — Cross-Feature & Pairwise Interactions**: End-to-end dataflow verification between adjacent subsystems:
   - Ground truth verification feeding the grader with 100% scores.
   - Tolerance perturbation triggering grader boundary rejections.
   - Infeasible tasks claiming certitude activating the false confidence veto.
   - Graded contender logs feeding the report generator pipeline.
   - Public vs sealed split parity, disjointness, and coverage.
   - Multi-repository oracle execution isolation without namespace leakage.
   - Cryptographic manifest cross-checking against on-disk artifacts.
4. **Tier 4 — Real-World Scenarios**: High-fidelity operational simulations:
   - Multi-turn N=8 epistemic amnesia chain testing fact retention across turns and Marginal Persistence Cost ($MPC_{\text{tokens}}$).
   - Full comparative evaluation reproducing the B0..B3 benchmarking tables under Anti-Goodhart rules.
   - Cryptographic baseline freezing (`baseline_v0_manifest.json`) and execution idempotence.

### 2.3 Strict Integrity Guardrails
- **Read-Only Invariant on `cge/`**: The core Causal Graph Engine directory (`cge/`) is strictly read-only. No files may be created, modified, or deleted within `cge/`.
- **Substantive Deliverables Verification**: Verification that all 4 study documents exist, meet minimum substantive byte/word counts, and contain mandatory technical sections without placeholders.
- **50/50 Partition Balance**: Strict equality between `tasks_public.jsonl` and `tasks_sealed.jsonl` with zero task ID leakage.

### 2.4 Anti-Goodhart Enforcement
- **No Single Summary Score**: Rejection of unidimensional composite metrics that hide critical trade-offs.
- **Relative Benchmark vs B1**: Enforcing $\Delta_{\text{contender}} = \frac{\text{Score} - \text{Score}_{B1}}{\max(0.05, \text{Score}_{B1})}$.
- **Zero False Confidence Veto**: Automatic disqualification (`Score = 0.0`, `false_confidence = True`) when a contender claims certainty on an infeasible or constraint-violating task.

---

## 3. Directory Layout

The acceptance test suite is co-located in `tests/e2e_acceptance/`:

```text
Sistema de solcuiones automatizado/
├── TEST_INFRA.md                     # Architectural authority for acceptance testing (this file)
├── TEST_READY.md                     # Delivery declaration & verification run instructions
├── pytest.ini                         # Project-wide pytest configuration (pythonpath = .)
├── tests/
│   └── e2e_acceptance/
│       ├── __init__.py               # Package marker
│       ├── conftest.py               # Universal fixtures (run_cli, path fixtures, JSONL parsers, sandbox)
│       ├── test_tier1_features.py    # 35 Tier 1 Feature Coverage tests across 7 areas
│       ├── test_tier2_boundaries.py  # 24 Tier 2 Boundary & Corner Case tests
│       ├── test_tier3_integration.py # 7 Tier 3 Cross-Feature & Pairwise tests
│       ├── test_tier4_scenarios.py   # Tier 4 Real-World Application Scenarios
│       └── test_integrity.py         # Deliverables & Codebase Integrity tests
```

---

## 4. Test Execution Protocols & Runner Commands

### 4.1 Master Test Commands

All acceptance tests are executed from the workspace root:

```powershell
# 1. Full Acceptance Test Suite Execution
python -m pytest tests/e2e_acceptance/ -v --tb=short

# 2. Dry-Run / Test Discovery Check (100% collect rate)
python -m pytest tests/e2e_acceptance/ --collect-only

# 3. Individual Tier Executions
python -m pytest tests/e2e_acceptance/test_tier1_features.py -v
python -m pytest tests/e2e_acceptance/test_tier2_boundaries.py -v
python -m pytest tests/e2e_acceptance/test_tier3_integration.py -v
python -m pytest tests/e2e_acceptance/test_tier4_scenarios.py -v
python -m pytest tests/e2e_acceptance/test_integrity.py -v

# 4. Marker-Based Smoke Execution
python -m pytest tests/e2e_acceptance/ -m "tier1 or integrity" -v

# 5. Non-Regression Baseline Verification
python -m pytest golden_case/tests/ benchmark/ -v
```

### 4.2 Subprocess Runtime Environment
All subprocess invocations configured by `conftest.py::run_cli` automatically inject:
```python
env = {
    "PYTHONPATH": repo_root,
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
    "PATH": os.environ["PATH"]
}
```
This guarantees deterministic UTF-8 handling on Windows, preventing `UnicodeEncodeError` when emitting mathematical notation ($\le$, $\Delta$, $\lambda$, $\epsilon$).

---

## 5. Test Suite Specifications by Tier

### 5.1 Tier 1: Feature Coverage (`test_tier1_features.py`)
Covers the 7 primary functional areas with at least 5 tests per area (35 total tests):
- **Area 1: Ground Truth CLI (`ground_truth.py --verify`)**:
  - `test_ground_truth_cli_verify_exit_code_zero`: Verifies exit code 0 and affirmative output.
  - `test_ground_truth_no_network_or_llm_calls`: Verifies purely local execution without external network calls.
  - `test_ground_truth_cli_help_flag`: Verifies `--help` returns exit code 0 and usage documentation.
  - `test_ground_truth_summary_output_format`: Verifies counts for public, sealed, and total tasks.
  - `test_ground_truth_deterministic_consecutive_runs`: Verifies identical consecutive outputs.
- **Area 2: Grader Engine (`test_grader.py`)**:
  - `test_grader_pytest_suite_passes_100_percent`: Pytest suite passes with 0 failures, 0 errors.
  - `test_grader_synthetic_perfect_score_is_100`: Perfect synthetic response scores 1.0.
  - `test_grader_empty_and_null_response_score_is_zero`: Null/empty response scores 0.0.
  - `test_grader_erroneous_response_score_is_zero`: Erroneous response scores 0.0.
  - `test_grader_false_confidence_veto_activation`: Certitude claim on infeasible task triggers veto.
- **Area 3: Report CLI (`report.py --baseline baseline_v0`)**:
  - `test_report_cli_baseline_v0_exit_code_zero`: Clean exit code 0.
  - `test_report_contains_all_four_contenders`: Displays rows for B0, B1, B2, B3.
  - `test_report_contains_all_eight_question_types`: Displays columns for Q1 through Q8.
  - `test_report_contains_efficiency_metrics`: Reports latency, input/output tokens, tool calls.
  - `test_report_contains_persistence_and_anti_goodhart_metrics`: Reports ERR, $\Delta_{B1}$, FCR.
- **Area 4: Task Datasets (`tasks_public.jsonl` & `tasks_sealed.jsonl`)**:
  - `test_datasets_files_exist_and_non_empty`: Both files exist and > 0 bytes.
  - `test_datasets_strict_50_50_balance`: $N_{\text{pub}} == N_{\text{sea}} \ge 8$.
  - `test_datasets_full_q1_to_q8_coverage_public`: Public set covers Q1..Q8.
  - `test_datasets_full_q1_to_q8_coverage_sealed`: Sealed set covers Q1..Q8.
  - `test_datasets_canonical_schema_validation`: Conformance to `BenchmarkTask` schema.
- **Area 5: Baseline Study Documents (`docs/baseline/`)**:
  - `test_docs_all_four_documents_exist`: 4 documents present.
  - `test_docs_minimum_content_size`: Each file > 5,000 bytes.
  - `test_docs_01_covers_all_nine_families`: Covers F1 through F9 with technical sheets.
  - `test_docs_03_covers_matrix_and_triangular_disconnection`: Covers 9x8 matrix and gap classification.
  - `test_docs_04_covers_anti_goodhart_rules_and_metrics`: Covers formulas and 4 rules.
- **Area 6: `cge/` Directory Integrity**:
  - `test_cge_directory_exists_and_intact`: Core files present.
  - `test_cge_pre_post_checksum_invariance`: Pre/post benchmark checksum snapshot matches 100%.
  - `test_cge_zero_added_or_deleted_files`: File set before == file set after.
  - `test_cge_read_only_filesystem_execution`: Execution succeeds without modifying `cge/`.
  - `test_cge_git_status_purity`: No new untracked or modified files in `cge/`.
- **Area 7: Cryptographic Freezing Manifest (`baseline_v0_manifest.json`)**:
  - `test_manifest_file_exists`: File exists on disk.
  - `test_manifest_valid_json_schema`: Valid JSON with `baseline_tag == 'baseline_v0'`, `seed == 42`.
  - `test_manifest_required_file_hashes_present`: Hashes present for key scripts and datasets.
  - `test_manifest_sha256_cryptographic_verification`: Recomputed SHA-256 matches manifest.
  - `test_manifest_environment_metadata_completeness`: Captures python version and platform.

### 5.2 Tier 2: Boundary & Corner Cases (`test_tier2_boundaries.py`)
24 boundary tests exercising error handling and edge behaviors:
- Missing, unknown, or corrupted CLI flags and files.
- Exact numerical tolerance borders ($y^* \times (1 \pm \epsilon)$).
- Denominator handling when ground truth $y^* = 0.0$ ($\max(1.0, |y^*|)$).
- Constraint violation threshold at $10^{-5}$ ($\text{Viol} \le 10^{-5}$ vs $\text{Viol} > 10^{-5}$).
- Dead node detection (`brand_awareness` poison pill in Q1).
- Non-existent baseline tag and zero-division guard in relative scoring ($\max(0.05, \text{Score}_{B1})$).
- Disjoint task ID validation and tolerance sanity checks ($0.0 \le \text{tol} \le 0.1$).
- Absence of placeholder tokens (`TODO`, `TBD`, `FIXME`).
- Manifest single-byte tamper detection.

### 5.3 Tier 3: Cross-Feature Interactions (`test_tier3_integration.py`)
7 pairwise integration tests verifying multi-module dataflow:
- `test_pairwise_ground_truth_feeds_grader_perfect_score`: Verified ground truths achieve 1.0 in grader.
- `test_pairwise_grader_perturbed_tolerance_violation`: Perturbation boundary transitions from 1.0 to 0.0.
- `test_pairwise_grader_false_confidence_veto`: Infeasible task claiming certitude triggers veto.
- `test_pairwise_grader_to_report_pipeline`: Grader outputs populate the 3 Markdown report tables.
- `test_pairwise_public_vs_sealed_parity`: 50/50 split balance and zero task ID collision.
- `test_pairwise_multi_repo_oracle_isolation`: Multi-repo execution order-invariance and isolation.
- `test_pairwise_manifest_cross_check_with_artifacts`: Byte-for-byte SHA-256 match of manifest.

### 5.4 Tier 4: Real-World Scenarios (`test_tier4_scenarios.py`)
High-level simulation scenarios:
- `TestTier4AmnesiaSimulation`: 8-turn sequence (Q1->Q7->Q3->Q2->Q4->Q5->Q6->Q8), fact retention across turns, ERR/CR metrics, and positive MPC token tracking.
- `TestTier4ComparativeEvaluation`: Subprocess run of `report.py --baseline baseline_v0`, B0..B3 evaluation, and strict Anti-Goodhart rules enforcement.
- `TestTier4BaselineFreezing`: Manifest schema compliance, cryptographic SHA-256 file matching, and `ground_truth.py --verify` idempotence.

### 5.5 Integrity Verification (`test_integrity.py`)
Deliverables and codebase integrity:
- `TestCgeDirectoryIntegrity`: Git status and diff verification ensuring `cge/` remains untouched.
- `TestBaselineDocumentsIntegrity`: Existence, size, word counts, required headings, and technical tokens for all 4 study documents.
- `TestTasksDatasetsIntegrity`: Syntactic and semantic validation for `tasks_public.jsonl` and `tasks_sealed.jsonl` (schema, 50/50 balance, Q1..Q8 coverage, non-empty GT, valid tolerances).

---

## 6. Milestone Acceptance Gate Criteria

| Milestone | Deliverables Under Test | Acceptance Gate Verification |
|-----------|-------------------------|------------------------------|
| **M1** | `docs/baseline/01_estado_del_arte.md` | `test_docs_01_covers_all_nine_families`, `test_document_existence_and_size[01...]` |
| **M2** | `benchmark/baseline_bench/` (harness, ground_truth, grader, report, datasets) | `test_tier1_features.py` (Areas 1, 2, 3, 4), `test_tier2_boundaries.py`, `test_tier3_integration.py` |
| **M3** | `docs/baseline/03_mapa_de_vacios.md` | `test_docs_03_covers_matrix_and_triangular_disconnection`, `test_document_existence_and_size[03...]` |
| **M4** | `docs/baseline/04_marco_de_metricas.md`, `baseline_v0_manifest.json` | `test_docs_04_covers_anti_goodhart_rules_and_metrics`, `test_manifest_*`, `test_tier4_scenarios.py` |
| **M5** | Full Repository & Final Delivery | 100% of `python -m pytest tests/e2e_acceptance/ -v` passes cleanly; zero changes in `cge/`. |
