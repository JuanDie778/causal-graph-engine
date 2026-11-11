"""
tests/e2e_acceptance/conftest.py
Global pytest fixtures and opaque-box execution utilities for ActorDB E2E Acceptance Testing.
"""

import sys
import os
import json
import hashlib
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator
import pytest

# ---------------------------------------------------------------------------
# Pytest Marker Registration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Registers acceptance test markers to prevent PytestUnknownMarkWarning."""
    config.addinivalue_line("markers", "e2e: mark test as part of the E2E acceptance suite")
    config.addinivalue_line("markers", "tier1: mark test as Tier 1 Feature Coverage")
    config.addinivalue_line("markers", "tier2: mark test as Tier 2 Boundary & Corner Cases")
    config.addinivalue_line("markers", "tier3: mark test as Tier 3 Cross-Feature & Pairwise")
    config.addinivalue_line("markers", "tier4: mark test as Tier 4 Real-World Scenarios")
    config.addinivalue_line("markers", "integrity: mark test as Deliverables & Code Integrity")


# ---------------------------------------------------------------------------
# Path & Directory Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Returns the absolute Path to the workspace root."""
    return Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="session")
def python_exe() -> str:
    """Returns the path to the current Python executable."""
    return sys.executable


@pytest.fixture(scope="session")
def baseline_bench_dir(repo_root: Path) -> Path:
    """Returns the directory path for benchmark/baseline_bench."""
    return repo_root / "benchmark" / "baseline_bench"


@pytest.fixture(scope="session")
def ground_truth_script(baseline_bench_dir: Path) -> Path:
    """Returns the path to ground_truth.py."""
    return baseline_bench_dir / "ground_truth.py"


@pytest.fixture(scope="session")
def test_grader_script(baseline_bench_dir: Path) -> Path:
    """Returns the path to test_grader.py."""
    return baseline_bench_dir / "test_grader.py"


@pytest.fixture(scope="session")
def report_script(baseline_bench_dir: Path) -> Path:
    """Returns the path to report.py."""
    return baseline_bench_dir / "report.py"


@pytest.fixture(scope="session")
def harness_script(baseline_bench_dir: Path) -> Path:
    """Returns the path to harness.py."""
    return baseline_bench_dir / "harness.py"


@pytest.fixture(scope="session")
def tasks_public_path(baseline_bench_dir: Path) -> Path:
    """Returns the path to tasks_public.jsonl."""
    return baseline_bench_dir / "tasks_public.jsonl"


@pytest.fixture(scope="session")
def tasks_sealed_path(baseline_bench_dir: Path) -> Path:
    """Returns the path to tasks_sealed.jsonl."""
    return baseline_bench_dir / "tasks_sealed.jsonl"


@pytest.fixture(scope="session")
def manifest_path(baseline_bench_dir: Path) -> Path:
    """Returns the path to baseline_v0_manifest.json."""
    return baseline_bench_dir / "baseline_v0_manifest.json"


@pytest.fixture(scope="session")
def docs_baseline_dir(repo_root: Path) -> Path:
    """Returns the directory path for docs/baseline."""
    return repo_root / "docs" / "baseline"


@pytest.fixture(scope="session")
def cge_dir(repo_root: Path) -> Path:
    """Returns the directory path for cge."""
    return repo_root / "cge"


# ---------------------------------------------------------------------------
# Opaque-Box Subprocess Runner Fixture
# ---------------------------------------------------------------------------

class ProcessResult:
    """Encapsulates the execution result of an opaque-box subprocess call."""

    def __init__(self, returncode: int, stdout: str, stderr: str, duration: float, cmd: List[str]):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.duration = duration
        self.cmd = cmd

    @property
    def success(self) -> bool:
        return self.returncode == 0

    def assert_success(self, msg: str = "") -> None:
        cmd_str = " ".join(self.cmd)
        assert self.returncode == 0, (
            f"Process failed with exit code {self.returncode}.\n"
            f"Command: {cmd_str}\n"
            f"Context: {msg}\n"
            f"--- STDOUT ---\n{self.stdout}\n"
            f"--- STDERR ---\n{self.stderr}"
        )


@pytest.fixture(scope="session")
def run_cli(repo_root: Path, python_exe: str):
    """
    Opaque-box CLI runner executing commands in an isolated subprocess.
    Ensures UTF-8 encoding on Windows, sets PYTHONPATH, and enforces timeouts.
    """
    def _runner(
        args: List[str],
        cwd: Optional[Path] = None,
        timeout: float = 60.0,
        extra_env: Optional[Dict[str, str]] = None,
        check: bool = False
    ) -> ProcessResult:
        work_dir = cwd or repo_root
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        if extra_env:
            env.update(extra_env)

        # Build command: if first arg is a python script or module, format with sys.executable
        if args and args[0].endswith(".py"):
            cmd = [python_exe] + args
        elif args and args[0] == "python":
            cmd = [python_exe] + args[1:]
        elif args and args[0] == "pytest":
            cmd = [python_exe, "-m", "pytest"] + args[1:]
        elif args and args[0] in ("-c", "-m"):
            cmd = [python_exe] + args
        else:
            cmd = args

        start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=str(work_dir),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout
        )
        duration = time.perf_counter() - start

        res = ProcessResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration=duration,
            cmd=cmd
        )
        if check:
            res.assert_success(f"Command {' '.join(cmd)} failed unexpectedly")
        return res

    return _runner


# ---------------------------------------------------------------------------
# Data Parsing & Cryptographic Utilities
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def parse_jsonl():
    """Utility to parse JSONL files into lists of dictionaries."""
    def _parser(filepath: Path) -> List[Dict[str, Any]]:
        assert filepath.exists(), f"JSONL file does not exist: {filepath}"
        tasks = []
        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                clean = line.strip()
                if clean:
                    try:
                        tasks.append(json.loads(clean))
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Line {idx} in {filepath} is not valid JSON: {e}")
        return tasks
    return _parser


@pytest.fixture(scope="session")
def compute_sha256():
    """Utility to compute the SHA-256 hash of a file."""
    def _hasher(filepath: Path) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    return _hasher


# ---------------------------------------------------------------------------
# Isolated Sandbox Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_sandbox(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Creates an isolated scratch directory so tests can simulate destructive
    or boundary operations without contaminating the workspace.
    """
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    yield sandbox
    shutil.rmtree(sandbox, ignore_errors=True)


# ---------------------------------------------------------------------------
# CGE Snapshot Fixture (Pre/Post Execution Invariance)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cge_baseline_snapshot(cge_dir: Path, compute_sha256) -> Dict[str, str]:
    """
    Takes a snapshot of SHA-256 hashes of all python files in cge/ at test startup
    to verify post-execution invariance.
    """
    snapshot: Dict[str, str] = {}
    if cge_dir.exists():
        for py_file in cge_dir.rglob("*.py"):
            if "__pycache__" not in str(py_file):
                rel_path = str(py_file.relative_to(cge_dir))
                snapshot[rel_path] = compute_sha256(py_file)
    return snapshot
