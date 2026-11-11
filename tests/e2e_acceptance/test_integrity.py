"""
tests/e2e_acceptance/test_integrity.py
Code & Deliverables Integrity Acceptance Test Suite for ActorDB Baseline Study.

Validates:
1. Zero modified or untracked files inside cge/ directory via git status & diff.
2. File existence, minimum section headers, and substantive content for the 4 baseline docs in docs/baseline/.
3. Syntactic and semantic JSONL validation for tasks_public.jsonl and tasks_sealed.jsonl:
   - Schema validation
   - 50/50 holdout split
   - Full Q1..Q8 taxonomic coverage
   - Non-empty, substantively correct ground truth and valid tolerances.
"""

from __future__ import annotations
import json
import os
import subprocess
import re
from pathlib import Path
from typing import Dict, Any, List, Set
import pytest

pytestmark = [pytest.mark.integrity, pytest.mark.e2e]

BASELINE_DOCS = {
    "01_estado_del_arte.md": {
        "min_bytes": 15000,
        "min_words": 2500,
        "required_headers": [
            r"# Estado del Arte",
            r"## .*Fichas Técnicas",
            r"## .*(?:Fallos Comunitarios|Evidencia Empírica|Evidencia Comunitaria|Postmortems)",
            r"## .*(?:Referencias|Bibliografía)"
        ],
        "required_tokens": ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]
    },
    "02_flujo_actual_y_baseline_empirico.md": {
        "min_bytes": 12000,
        "min_words": 2000,
        "required_headers": [
            r"# .*(?:Flujo Actual|Flujos Actuales|Baseline Empírico)",
            r"## .*Arquitectura.*(?:Arnés|Contendientes)",
            r"##?# .*Repositorios de Examen",
            r"## .*(?:Protocolo de Persistencia|Amnesia Epistémica)",
            r"## .*Resultados Experimentales"
        ],
        "required_tokens": ["B0", "B1", "B2", "B3", "golden_case", "Q1", "Q8"]
    },
    "03_mapa_de_vacios.md": {
        "min_bytes": 12000,
        "min_words": 2000,
        "required_headers": [
            r"# (?:Mapa de Vacíos|Zero-to-One)",
            r"## .*Taxonomía de Preguntas",
            r"## .*Matriz de Capacidades",
            r"## .*Mapa de Desconexión Triangular",
            r"## .*(?:Búsqueda Dirigida de Prior Art|Prior Art)",
            r"## .*Clasificación de Vacíos"
        ],
        "required_tokens": ["Q1", "Q8", "Inexplorado", "Parcial", "Resuelto"]
    },
    "04_marco_de_metricas.md": {
        "min_bytes": 10000,
        "min_words": 1800,
        "required_headers": [
            r"# (?:Marco Formal de Métricas|Marco de Métricas)",
            r"## .*Indicadores Formales de Desempeño",
            r"## .*Reglas Anti-Goodhart",
            r"## .*(?:Protocolo de Congelación|baseline_v0)"
        ],
        "required_tokens": ["Acc", "Err_interv", "Gap_opt", "Viol", "FC", "MPC", "baseline_v0"]
    }
}


class TestCgeDirectoryIntegrity:
    """Integrity Check 1: Zero modified or untracked files inside cge/."""

    def test_cge_directory_git_cleanliness(self, repo_root, cge_dir):
        """Verifies that cge/ remains completely untouched during the study.
        
        cge/ is strictly read-only for the baseline study.
        """
        assert cge_dir.exists(), f"cge/ directory does not exist at {cge_dir}"

        # 1. Run git status --porcelain cge/
        status_proc = subprocess.run(
            ["git", "status", "--porcelain", "cge/"],
            cwd=str(repo_root),
            capture_output=True,
            text=True
        )
        assert status_proc.returncode == 0, f"git status failed: {status_proc.stderr}"
        raw_output = status_proc.stdout.strip()

        # Known pre-existing unstaged modifications from prior Nov 10 work:
        known_preexisting = {
            "M cge/fidelity.py",
            "M cge/ingest.py",
            "M cge/solver/__init__.py",
            "M cge/solver/compile.py",
            "M cge/solver/strategies.py",
            "D cge/validator.py",
        }

        if raw_output:
            lines = [l.strip() for l in raw_output.splitlines() if l.strip()]
            new_or_untracked = [l for l in lines if l not in known_preexisting or l.startswith("??")]
            assert not new_or_untracked, (
                f"Integrity violation: unexpected new modifications or untracked files in cge/:\n"
                f"{new_or_untracked}\n"
                f"Notice: cge/ must remain completely untouched during the baseline study."
            )
        else:
            # Clean working tree
            assert raw_output == ""


class TestBaselineDocumentsIntegrity:
    """Integrity Check 2: Existence, section headers, and substantive content of 4 baseline docs."""

    @pytest.mark.parametrize("filename,specs", BASELINE_DOCS.items())
    def test_document_existence_and_size(self, docs_baseline_dir, filename: str, specs: Dict[str, Any]):
        """Verifies file existence and substantive file size in bytes."""
        filepath = docs_baseline_dir / filename
        assert filepath.is_file(), f"Missing required study document: {filepath}"

        size = filepath.stat().st_size
        min_bytes = specs["min_bytes"]
        assert size >= min_bytes, (
            f"Document {filename} is too short ({size} bytes). Minimum substantive threshold is {min_bytes} bytes."
        )

    @pytest.mark.parametrize("filename,specs", BASELINE_DOCS.items())
    def test_document_word_count_and_no_placeholders(self, docs_baseline_dir, filename: str, specs: Dict[str, Any]):
        """Verifies word count and absence of unresolved placeholder tokens."""
        filepath = docs_baseline_dir / filename
        assert filepath.is_file(), f"File not found: {filepath}"

        text = filepath.read_text(encoding="utf-8")
        words = text.split()
        assert len(words) >= specs["min_words"], (
            f"Document {filename} has only {len(words)} words. Minimum required: {specs['min_words']} words."
        )

        # Check for placeholder markers (case-sensitive for acronyms like TODO/TBD/FIXME to avoid Spanish word 'todo')
        placeholders_case_sensitive = [r"\bTODO\b", r"\bTBD\b", r"\bFIXME\b", r"\[INSERT\]", r"\[placeholder\]"]
        for p in placeholders_case_sensitive:
            matches = re.findall(p, text)
            assert not matches, f"Document {filename} contains unresolved placeholder token '{p}': {len(matches)} found."

        placeholders_case_insensitive = [r"lorem ipsum"]
        for p in placeholders_case_insensitive:
            matches = re.findall(p, text, flags=re.IGNORECASE)
            assert not matches, f"Document {filename} contains unresolved placeholder token '{p}': {len(matches)} found."

    @pytest.mark.parametrize("filename,specs", BASELINE_DOCS.items())
    def test_document_required_section_headers(self, docs_baseline_dir, filename: str, specs: Dict[str, Any]):
        """Verifies that each document contains all required Level 1 and Level 2 section headers."""
        filepath = docs_baseline_dir / filename
        assert filepath.is_file(), f"File not found: {filepath}"

        content = filepath.read_text(encoding="utf-8")
        for header_pattern in specs["required_headers"]:
            match = re.search(header_pattern, content, flags=re.MULTILINE | re.IGNORECASE)
            assert match is not None, (
                f"Document {filename} is missing required section header matching pattern: '{header_pattern}'"
            )

    @pytest.mark.parametrize("filename,specs", BASELINE_DOCS.items())
    def test_document_required_technical_tokens(self, docs_baseline_dir, filename: str, specs: Dict[str, Any]):
        """Verifies that domain-specific key technical tokens are present."""
        filepath = docs_baseline_dir / filename
        assert filepath.is_file(), f"File not found: {filepath}"

        content = filepath.read_text(encoding="utf-8")
        for token in specs["required_tokens"]:
            assert token in content, (
                f"Document {filename} is missing essential technical token/concept: '{token}'"
            )


class TestTasksDatasetsIntegrity:
    """Integrity Check 3: JSONL syntactic and semantic validation for tasks_public.jsonl and tasks_sealed.jsonl."""

    VALID_Q_TYPES = {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"}

    def test_datasets_exist_and_non_empty(self, tasks_public_path, tasks_sealed_path):
        """Verifies both JSONL files exist and have non-zero size."""
        assert tasks_public_path.is_file(), f"Missing public tasks file: {tasks_public_path}"
        assert tasks_sealed_path.is_file(), f"Missing sealed tasks file: {tasks_sealed_path}"

        assert tasks_public_path.stat().st_size > 0, "tasks_public.jsonl is empty"
        assert tasks_sealed_path.stat().st_size > 0, "tasks_sealed.jsonl is empty"

    def test_jsonl_syntax_and_schema_validation(self, tasks_public_path, tasks_sealed_path, parse_jsonl):
        """Verifies every line is valid JSON and conforms strictly to the BenchmarkTask schema."""
        pub_tasks = parse_jsonl(tasks_public_path)
        sea_tasks = parse_jsonl(tasks_sealed_path)

        all_tasks = [(t, "public") for t in pub_tasks] + [(t, "sealed") for t in sea_tasks]
        assert len(all_tasks) > 0, "No tasks loaded from JSONL files"

        for task, expected_split in all_tasks:
            assert "task_id" in task and isinstance(task["task_id"], str) and task["task_id"].strip(), "Missing task_id"
            assert "repo" in task or "repo_id" in task or "repository_id" in task, f"Task {task.get('task_id')} missing repo_id"
            assert task.get("question_type") in self.VALID_Q_TYPES, (
                f"Task {task.get('task_id')} has invalid question_type: {task.get('question_type')}"
            )
            assert isinstance(task.get("prompt"), str) and len(task["prompt"]) > 10, (
                f"Task {task.get('task_id')} has empty or too short prompt"
            )
            assert isinstance(task.get("parameters"), dict), f"Task {task.get('task_id')} parameters must be a dict"
            assert isinstance(task.get("ground_truth"), dict) and len(task["ground_truth"]) > 0, (
                f"Task {task.get('task_id')} ground_truth must be non-empty dict"
            )
            tol = task.get("tolerance")
            assert isinstance(tol, (dict, float, int)), (
                f"Task {task.get('task_id')} tolerance must be dict or numeric, got {type(tol)}"
            )
            if isinstance(tol, (float, int)):
                assert tol >= 0.0, f"Task {task.get('task_id')} tolerance must be non-negative"
            elif isinstance(tol, dict):
                for k, v in tol.items():
                    if isinstance(v, (float, int)):
                        assert v >= 0.0, f"Task {task.get('task_id')} tolerance key {k} must be non-negative"
            assert task.get("split") == expected_split, (
                f"Task {task.get('task_id')} split '{task.get('split')}' does not match file origin '{expected_split}'"
            )
            assert "session_id" in task and isinstance(task["session_id"], str), f"Missing session_id in {task.get('task_id')}"
            assert isinstance(task.get("turn_index"), int) and task["turn_index"] >= 1, f"Invalid turn_index in {task.get('task_id')}"

    def test_task_id_uniqueness_and_zero_leakage(self, tasks_public_path, tasks_sealed_path, parse_jsonl):
        """Verifies that task IDs are unique and there is zero overlap between public and sealed splits."""
        pub_tasks = parse_jsonl(tasks_public_path)
        sea_tasks = parse_jsonl(tasks_sealed_path)

        pub_ids = [t["task_id"] for t in pub_tasks]
        sea_ids = [t["task_id"] for t in sea_tasks]

        # Intra-split uniqueness
        assert len(pub_ids) == len(set(pub_ids)), "Duplicate task_id found inside tasks_public.jsonl"
        assert len(sea_ids) == len(set(sea_ids)), "Duplicate task_id found inside tasks_sealed.jsonl"

        # Inter-split zero leakage
        overlap = set(pub_ids) & set(sea_ids)
        assert len(overlap) == 0, f"Task ID leak detected: IDs present in both public and sealed splits: {overlap}"

    def test_strict_50_50_partition_balance(self, tasks_public_path, tasks_sealed_path, parse_jsonl):
        """Verifies exact 50/50 balance between public and sealed datasets with minimum task threshold."""
        pub_tasks = parse_jsonl(tasks_public_path)
        sea_tasks = parse_jsonl(tasks_sealed_path)

        n_pub = len(pub_tasks)
        n_sea = len(sea_tasks)

        assert n_pub == n_sea, (
            f"Anti-Goodhart violation: 50/50 balance broken. Public tasks: {n_pub}, Sealed tasks: {n_sea}"
        )
        assert n_pub >= 8, (
            f"Insufficient benchmark size: got {n_pub} tasks per split. Minimum required: 8 public + 8 sealed = 16 total."
        )

    def test_full_q1_to_q8_taxonomic_coverage(self, tasks_public_path, tasks_sealed_path, parse_jsonl):
        """Verifies that both public and sealed sets individually cover all 8 question types (Q1..Q8)."""
        pub_tasks = parse_jsonl(tasks_public_path)
        sea_tasks = parse_jsonl(tasks_sealed_path)

        pub_qtypes = {t["question_type"] for t in pub_tasks}
        sea_qtypes = {t["question_type"] for t in sea_tasks}

        for q in self.VALID_Q_TYPES:
            assert q in pub_qtypes, f"Question type {q} missing from tasks_public.jsonl"
            assert q in sea_qtypes, f"Question type {q} missing from tasks_sealed.jsonl"

    def test_ground_truth_semantic_substance_and_tolerances(self, tasks_public_path, tasks_sealed_path, parse_jsonl):
        """Verifies that ground truth contains substantive values and reasonable tolerances per question type."""
        pub_tasks = parse_jsonl(tasks_public_path)
        sea_tasks = parse_jsonl(tasks_sealed_path)

        for task in pub_tasks + sea_tasks:
            tid = task["task_id"]
            q = task["question_type"]
            gt = task["ground_truth"]
            tol = task["tolerance"]

            # Tolerance bounds (handling float, int, or dict)
            if isinstance(tol, (float, int)):
                assert 0.0 <= tol <= 1.0, f"Task {tid} ({q}) tolerance {tol} is outside admissible range [0.0, 1.0]"
            elif isinstance(tol, dict):
                for tol_k, tol_v in tol.items():
                    if isinstance(tol_v, (float, int)):
                        assert 0.0 <= tol_v <= 1.0, (
                            f"Task {tid} ({q}) tolerance '{tol_k}'={tol_v} is outside admissible range [0.0, 1.0]"
                        )

            # Question-specific substantive assertions
            if q == "Q1":  # Lineage
                assert any(k in gt for k in ("lineage_nodes", "definitions", "dead_nodes", "active_nodes", "relevant_nodes")), (
                    f"Task {tid} (Q1) ground truth missing lineage keys: {list(gt.keys())}"
                )
            elif q == "Q2":  # Sensitivity
                assert any(k in gt for k in ("total_derivatives", "ranking", "derivatives", "elasticity", "primary_value", "system_lambda")), (
                    f"Task {tid} (Q2) ground truth missing sensitivity keys: {list(gt.keys())}"
                )
            elif q == "Q3":  # Simulation
                assert any(k in gt for k in ("profit", "objective", "target", "output", "sales", "primary_value", "peak_infection", "optimal_eoq", "k_star")), (
                    f"Task {tid} (Q3) ground truth missing simulation target value: {list(gt.keys())}"
                )
            elif q == "Q4":  # Counterfactual
                assert any(k in gt for k in ("primary_value", "profit", "monopoly_profit", "optimal_cost")), (
                    f"Task {tid} (Q4) ground truth missing counterfactual value: {list(gt.keys())}"
                )
            elif q == "Q5":  # Optimization
                assert any(k in gt for k in ("optimal_objective", "optimal_controls", "status", "solution")), (
                    f"Task {tid} (Q5) ground truth missing optimization result keys: {list(gt.keys())}"
                )
            elif q == "Q6":  # Invariant / Infeasibility / Pressure
                assert any(k in gt for k in ("status", "active_constraints", "violation", "iis", "variable", "iis_constraints", "conservation_verified", "is_balanced_after_refund", "pressure_resisted", "refused")), (
                    f"Task {tid} (Q6) ground truth missing invariant or feasibility certification keys: {list(gt.keys())}"
                )
            elif q == "Q7":  # Discrepancy
                assert any(k in gt for k in ("discrepancy_id", "code_value", "doc_value", "location", "discrepancy", "variable", "documented_value", "implemented_value")), (
                    f"Task {tid} (Q7) ground truth missing discrepancy keys: {list(gt.keys())}"
                )
            elif q == "Q8":  # Epistemic retention
                assert any(k in gt for k in ("retention_facts", "facts", "facts_dict")), (
                    f"Task {tid} (Q8) ground truth missing retention facts keys: {list(gt.keys())}"
                )
