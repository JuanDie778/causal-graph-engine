"""Runner pytest directo para `python -m pytest benchmark/or_bench -v`."""
from benchmark.test_n8_or_bench import (
    test_n8_or_bench_public_problem,
    test_n8_or_bench_sealed_problem,
    test_n8_or_bench_aggregate_gates_and_report,
)

__all__ = [
    "test_n8_or_bench_public_problem",
    "test_n8_or_bench_sealed_problem",
    "test_n8_or_bench_aggregate_gates_and_report",
]
