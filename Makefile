.PHONY: benchmark eval f1 oracle n0 n1 n2 n3 n4 n5 clean

benchmark:
	python run_benchmark.py --level all

eval:
	python run_eval.py

f1:
	python -m pytest benchmark/test_f1_anti_memorization.py -v

oracle:
	python oracle/solve.py

n0:
	python run_benchmark.py --level N0

n1:
	python run_benchmark.py --level N1

n2:
	python run_benchmark.py --level N2

n3:
	python run_benchmark.py --level N3

n4:
	python run_benchmark.py --level N4

n5:
	python run_benchmark.py --level N5

clean:
	python -c "import shutil, os; shutil.rmtree('benchmark_variants', ignore_errors=True)"

