.PHONY: test or_bench fidelity ui clean

test:
	python -m pytest golden_case/tests/ benchmark/ -v

or_bench:
	python -m pytest benchmark/test_n8_or_bench.py -v

fidelity:
	python -m pytest benchmark/test_f10_fidelity.py -v

ui:
	python run_ui.py golden_case

clean:
	python -c "import shutil; shutil.rmtree('.pytest_cache', ignore_errors=True)"
