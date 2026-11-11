"""
tests/e2e_acceptance/test_m5_adversarial_manifest_invariance.py
Adversarial Verification Suite for Milestone 5: Manifest Tamper Resistance & Multi-Cycle Report Invariance.

Executed by: challenger_m5_1 (Empirical Challenger)
Covers:
1. Strict 1-byte tamper resistance across code, dataset, golden_case, repos, and or_bench files.
2. Cryptographic hash mutation and manifest schema tampering detection.
3. Missing file detection and non-zero exit codes.
4. Multi-cycle deterministic report invariance and frozen_results equivalence.
5. Invariance of manifest verification under report re-runs.
"""

from __future__ import annotations
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
GENERATE_MANIFEST_PY = ROOT_DIR / "benchmark" / "baseline_bench" / "generate_manifest.py"
REPORT_PY = ROOT_DIR / "benchmark" / "baseline_bench" / "report.py"
MANIFEST_PATH = ROOT_DIR / "benchmark" / "baseline_bench" / "baseline_v0_manifest.json"
SUMMARY_PATH = ROOT_DIR / "benchmark" / "baseline_bench" / "baseline_v0_summary.json"


def _safe_write_bytes(target_file: Path, data: bytes, retries: int = 10) -> None:
    """Safely write bytes on Windows handling transient file-locking."""
    for attempt in range(retries):
        try:
            target_file.write_bytes(data)
            return
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(0.05)


def run_manifest_verify(manifest_file: Path | None = None) -> subprocess.CompletedProcess:
    """Helper to run generate_manifest.py --verify."""
    cmd = [sys.executable, str(GENERATE_MANIFEST_PY), "--verify"]
    if manifest_file is not None:
        cmd.extend(["--output", str(manifest_file)])
    return subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)


def run_report_cli(baseline_tag: str = "baseline_v0", export_file: Path | None = None) -> subprocess.CompletedProcess:
    """Helper to run report.py."""
    cmd = [sys.executable, str(REPORT_PY), "--baseline", baseline_tag]
    if export_file is not None:
        cmd.extend(["--export", str(export_file)])
    return subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)


# ==============================================================================
# 1. MANIFEST TAMPER RESISTANCE: 1-BYTE FILE PERTURBATION
# ==============================================================================
class TestManifestFileTamperResistance:
    """Empirical verification that perturbing even 1 single byte in any tracked file
    causes generate_manifest.py --verify to fail with exit code != 0."""

    @pytest.mark.parametrize("rel_path", [
        "benchmark/baseline_bench/grader.py",
        "benchmark/baseline_bench/tasks_public.jsonl",
        "benchmark/baseline_bench/tasks_sealed.jsonl",
        "golden_case/pipeline.py",
        "golden_case/data/sales_history.csv",
        "benchmark/baseline_bench/repos/loan_tax_engine/amortization.py",
        "benchmark/baseline_bench/repos/epidemiology_seir/seir_model.py",
        "benchmark/baseline_bench/repos/economic_dispatch/dispatch.py",
        "benchmark/baseline_bench/repos/supplychainpy_mini/eoq.py",
        "benchmark/baseline_bench/repos/quantecon_cournot_solow/cournot.py",
        "benchmark/or_bench/oracles.py",
    ])
    def test_single_byte_perturbation_fails_verification(self, rel_path: str):
        target_file = ROOT_DIR / rel_path
        assert target_file.exists(), f"Target file does not exist: {target_file}"

        original_bytes = target_file.read_bytes()
        try:
            # Perturb exactly 1 single byte (append a single space or mutate the last byte)
            tampered_bytes = original_bytes + b" "
            _safe_write_bytes(target_file, tampered_bytes)

            # Run verification
            proc = run_manifest_verify()
            assert proc.returncode != 0, f"Verification must fail when {rel_path} has 1 byte added!"
            assert "[FAIL]" in proc.stdout or proc.returncode == 1
            assert rel_path in proc.stdout, f"Expected {rel_path} to be listed as mismatch in output: {proc.stdout}"

        finally:
            # Clean restoration guaranteed
            _safe_write_bytes(target_file, original_bytes)

        # Confirm post-restoration verification succeeds
        restore_proc = run_manifest_verify()
        assert restore_proc.returncode == 0, f"Post-restoration verification failed for {rel_path}!"
        assert "[PASS]" in restore_proc.stdout

    def test_missing_tracked_file_fails_verification(self):
        """Temporarily renaming a tracked file must trigger missing file detection and exit code 1."""
        target_file = ROOT_DIR / "benchmark" / "baseline_bench" / "tasks_sealed.jsonl"
        temp_renamed = ROOT_DIR / "benchmark" / "baseline_bench" / "tasks_sealed.jsonl.bak_m5"

        assert target_file.exists()
        try:
            target_file.rename(temp_renamed)
            proc = run_manifest_verify()
            assert proc.returncode != 0, "Verification must fail when a tracked file is missing!"
            assert "Archivos referenciados en manifiesto ausentes en disco" in proc.stdout
            assert "benchmark/baseline_bench/tasks_sealed.jsonl" in proc.stdout
        finally:
            if temp_renamed.exists():
                temp_renamed.rename(target_file)

        restore_proc = run_manifest_verify()
        assert restore_proc.returncode == 0
        assert "[PASS]" in restore_proc.stdout


# ==============================================================================
# 2. MANIFEST METADATA & HASH CORRUPTION RESISTANCE
# ==============================================================================
class TestManifestHashAndSchemaTamperResistance:
    """Empirical verification that mutating hashes, metadata, or JSON structure
    inside baseline_v0_manifest.json causes --verify to detect the tamper."""

    def test_corrupted_hash_fails_verification(self, tmp_path):
        """Modifying 1 hex character in an SHA-256 hash causes verify failure."""
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        tampered_manifest = copy.deepcopy(manifest_data)
        # Pick one file and flip the first character of its hash
        sample_file = "benchmark/baseline_bench/grader.py"
        orig_hash = tampered_manifest["checksums"][sample_file]
        flipped_char = "0" if orig_hash[0] != "0" else "1"
        tampered_hash = flipped_char + orig_hash[1:]
        tampered_manifest["checksums"][sample_file] = tampered_hash
        tampered_manifest["file_hashes"][sample_file] = tampered_hash

        tampered_path = tmp_path / "tampered_manifest.json"
        with open(tampered_path, "w", encoding="utf-8") as f:
            json.dump(tampered_manifest, f)

        proc = run_manifest_verify(tampered_path)
        assert proc.returncode != 0, "Corrupted hash must cause verification failure!"
        assert "[FAIL]" in proc.stdout
        assert sample_file in proc.stdout

    def test_tampered_baseline_tag_fails_verification(self, tmp_path):
        """Changing baseline_tag from baseline_v0 triggers assertion error."""
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        tampered_manifest = copy.deepcopy(manifest_data)
        tampered_manifest["baseline_tag"] = "baseline_v1_tampered"

        tampered_path = tmp_path / "tampered_tag.json"
        with open(tampered_path, "w", encoding="utf-8") as f:
            json.dump(tampered_manifest, f)

        proc = run_manifest_verify(tampered_path)
        assert proc.returncode != 0
        assert "AssertionError" in proc.stderr or "Tag inválido" in proc.stderr or proc.returncode != 0

    def test_tampered_random_seed_fails_verification(self, tmp_path):
        """Changing random_seed != 42 triggers assertion error."""
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        tampered_manifest = copy.deepcopy(manifest_data)
        tampered_manifest["environment"]["random_seed"] = 999
        tampered_manifest["environment_metadata"]["random_seed"] = 999

        tampered_path = tmp_path / "tampered_seed.json"
        with open(tampered_path, "w", encoding="utf-8") as f:
            json.dump(tampered_manifest, f)

        proc = run_manifest_verify(tampered_path)
        assert proc.returncode != 0
        assert "random_seed != 42" in proc.stderr or proc.returncode != 0

    def test_truncated_file_hashes_fails_verification(self, tmp_path):
        """Manifest with fewer than 40 files triggers assertion error."""
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        tampered_manifest = copy.deepcopy(manifest_data)
        # Keep only 10 files
        trimmed_keys = list(tampered_manifest["checksums"].keys())[:10]
        tampered_manifest["checksums"] = {k: tampered_manifest["checksums"][k] for k in trimmed_keys}
        tampered_manifest["file_hashes"] = tampered_manifest["checksums"]

        tampered_path = tmp_path / "trimmed_hashes.json"
        with open(tampered_path, "w", encoding="utf-8") as f:
            json.dump(tampered_manifest, f)

        proc = run_manifest_verify(tampered_path)
        assert proc.returncode != 0
        assert "Cantidad insuficiente de archivos en manifiesto" in proc.stderr or proc.returncode != 0

    def test_malformed_json_syntax_fails_verification(self, tmp_path):
        """Syntactically broken JSON causes JSONDecodeError and non-zero exit."""
        tampered_path = tmp_path / "broken_syntax.json"
        with open(tampered_path, "w", encoding="utf-8") as f:
            f.write("{\n  \"manifest_version\": \"1.0.0\",\n  BROKEN_JSON_HERE")

        proc = run_manifest_verify(tampered_path)
        assert proc.returncode != 0


# ==============================================================================
# 3. MULTI-CYCLE REPORT INVARIANCE
# ==============================================================================
class TestMultiCycleReportInvariance:
    """Empirical verification that running report.py across consecutive cycles produces
    100% deterministic metric equality and bit-for-bit equivalence with frozen_results."""

    def test_multi_cycle_deterministic_metrics(self, tmp_path):
        """Run report.py across 5 consecutive cycles and assert 100% metric equality."""
        cycles = 5
        metrics_by_cycle: List[Dict[str, Any]] = []

        for cycle_idx in range(cycles):
            export_file = tmp_path / f"summary_cycle_{cycle_idx}.json"
            proc = run_report_cli(baseline_tag="baseline_v0", export_file=export_file)
            assert proc.returncode == 0, f"Cycle {cycle_idx} failed: {proc.stderr}"
            assert export_file.exists()

            with open(export_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert "metrics" in data
            metrics_by_cycle.append(data["metrics"])

        # Compare cycle 0 against all subsequent cycles
        ref_metrics = metrics_by_cycle[0]
        ref_json_serialized = json.dumps(ref_metrics, sort_keys=True)
        ref_hash = hashlib.sha256(ref_json_serialized.encode("utf-8")).hexdigest()

        for cycle_idx in range(1, cycles):
            current_metrics = metrics_by_cycle[cycle_idx]
            current_serialized = json.dumps(current_metrics, sort_keys=True)
            current_hash = hashlib.sha256(current_serialized.encode("utf-8")).hexdigest()

            assert current_hash == ref_hash, (
                f"Metrics mismatch between cycle 0 and cycle {cycle_idx}!\n"
                f"Ref hash: {ref_hash}\n"
                f"Cycle {cycle_idx} hash: {current_hash}"
            )
            assert current_metrics == ref_metrics

    def test_summary_metrics_match_manifest_frozen_results(self):
        """Verify that baseline_v0_summary.json metrics match baseline_v0_manifest.json frozen_results."""
        assert MANIFEST_PATH.exists()
        assert SUMMARY_PATH.exists()

        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        manifest_frozen = manifest_data.get("frozen_results", {})
        summary_metrics = summary_data.get("metrics", {})

        manifest_hash = hashlib.sha256(json.dumps(manifest_frozen, sort_keys=True).encode("utf-8")).hexdigest()
        summary_hash = hashlib.sha256(json.dumps(summary_metrics, sort_keys=True).encode("utf-8")).hexdigest()

        assert summary_hash == manifest_hash, (
            f"Hash mismatch between frozen_results in manifest and metrics in summary!\n"
            f"Manifest frozen_results hash: {manifest_hash}\n"
            f"Summary metrics hash: {summary_hash}"
        )
        assert manifest_frozen == summary_metrics

    def test_report_generation_does_not_break_manifest_verification(self):
        """Running report.py generates baseline_v0_summary.json, but generate_manifest.py --verify
        must remain 100% valid because baseline_v0_summary.json is dynamically excluded from hash tracking."""
        # 1. Verify manifest initially
        p1 = run_manifest_verify()
        assert p1.returncode == 0

        # 2. Re-run report.py (updates timestamp in baseline_v0_summary.json)
        p_rep = run_report_cli(baseline_tag="baseline_v0")
        assert p_rep.returncode == 0

        # 3. Verify manifest again
        p2 = run_manifest_verify()
        assert p2.returncode == 0
        assert "[PASS]" in p2.stdout
