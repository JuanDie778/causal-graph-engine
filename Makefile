.PHONY: install install-dev test test-fast test-unit test-e2e test-full or_bench fidelity lint verify-baseline mcp diff ui clean

PY ?= python
LINT_PATHS = cge/diff cge/ingest_adapters cge/mcp tests/unit tests/test_ingest_adapters.py examples run_ui.py
FAST_PATHS = tests/unit tests/test_ingest_adapters.py golden_case/tests benchmark

install:
	$(PY) -m pip install -e .

install-dev:
	$(PY) -m pip install -e ".[dev]"

test: test-fast

test-fast:
	$(PY) -m pytest $(FAST_PATHS) -m "not slow" -q

test-unit:
	$(PY) -m pytest $(FAST_PATHS) -q

test-e2e:
	$(PY) -m pytest tests/e2e_acceptance -q

test-full:
	$(PY) -m pytest -q

or_bench:
	$(PY) -m pytest benchmark/test_n8_or_bench.py -v

fidelity:
	$(PY) -m pytest benchmark/test_f10_fidelity.py -v

lint:
	$(PY) -m ruff check $(LINT_PATHS)

verify-baseline:
	$(PY) benchmark/baseline_bench/generate_manifest.py --verify
	$(PY) benchmark/baseline_bench/ground_truth.py --verify

mcp:
	$(PY) -m cge.mcp --transport stdio

diff:
	$(PY) -m cge.diff --git-repo . --base-ref HEAD~1 --head-ref WORKTREE --format markdown

ui:
	$(PY) run_ui.py golden_case

clean:
	$(PY) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('.pytest_cache', '.ruff_cache', 'build', 'dist')]"
