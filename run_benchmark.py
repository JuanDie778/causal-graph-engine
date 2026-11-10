"""Harness principal y CLI de ejecución del Benchmark del DoD (Niveles N0 a N5)."""
import argparse
import sys
import subprocess
import time
from typing import Dict, Any, List


LEVEL_MAP = {
    "N0": {
        "name": "N0 - Motor Matemático Aislado (Capas 1 y 2)",
        "file": "benchmark/test_n0_math_engine.py",
        "criteria": ["R-1", "R-2", "R-3", "R-4", "R-5", "R-6", "R-7", "R-8", "R-9"]
    },
    "N1": {
        "name": "N1 - Ingesta AST y Estructuración Canónica",
        "file": "benchmark/test_n1_ingestion.py",
        "criteria": ["I-1", "I-2", "I-3", "I-4", "I-5", "I-6", "I-7", "I-8", "I-9", "I-10"]
    },
    "N2": {
        "name": "N2 - Trampas T1-T6 y Auditoría Matemática",
        "file": "benchmark/test_n2_audit_diagnostics.py",
        "criteria": ["A-1", "A-2", "A-3", "A-4", "A-5", "A-6", "A-7"]
    },
    "N3": {
        "name": "N3 - Incertidumbre U, CSV y Monte Carlo",
        "file": "benchmark/test_n3_uncertainty.py",
        "criteria": ["R-10", "T7"]
    },
    "N4": {
        "name": "N4 - Visualización e Inmutabilidad + Copiloto Agéntico",
        "file": "benchmark/test_n4_copilot_ui.py",
        "criteria": ["V-1", "V-2", "V-3", "V-4", "V-5", "C-1", "C-2", "C-3", "C-4"]
    },
    "N5": {
        "name": "N5 - Generalización, 20 Variantes y Mutación (MVP HECHO)",
        "file": "benchmark/test_n5_generalization.py",
        "criteria": ["G-1", "G-2", "G-3", "G-4"]
    },
    "N6": {
        "name": "N6 - Google Maps Multidimensional (UI Interactiva + Copiloto F6)",
        "file": "benchmark/test_f6_api.py",
        "criteria": ["UI-API", "UI-E2E", "UI-REACTIVITY", "PATH-TRAVERSAL-DEFENSE"]
    },
    "N7": {
        "name": "N7 - Universalización Multi-Fuente (Pandas/SQL + Manuales + CSV/ML)",
        "file": "benchmark/test_n7_universal_multisource.py",
        "criteria": ["U-1", "U-2", "U-3", "U-4", "U-5", "U-6", "U-7"]
    },
    "N8": {
        "name": "N8 - Núcleo Verificable y OR-Bench (20 Problemas Clásicos OR + Certificados)",
        "file": "benchmark/test_n8_or_bench.py",
        "criteria": ["F-1", "F-2", "OR-PUB", "OR-SEALED", "ZERO-FALSE-CONFIDENCE"]
    }
}


def run_level(level_id: str) -> Dict[str, Any]:
    level_info = LEVEL_MAP[level_id]
    test_file = level_info["file"]
    print(f"\n=======================================================")
    print(f"  EJECUTANDO {level_info['name']}")
    print(f"=======================================================")
    
    t0 = time.time()
    cmd = [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - t0

    is_pass = (res.returncode == 0)
    return {
        "level": level_id,
        "name": level_info["name"],
        "criteria": level_info["criteria"],
        "passed": is_pass,
        "duration_sec": round(duration, 2),
        "stdout": res.stdout,
        "stderr": res.stderr
    }


def print_summary_table(results: List[Dict[str, Any]]):
    print("\n" + "=" * 80)
    print("                  INFORME DE CUMPLIMIENTO DEL DoD — CGE MVP")
    print("=" * 80)
    print(f"{'Nivel':<8} | {'Estado':<8} | {'Duración':<10} | {'Criterios Verificados'}")
    print("-" * 80)

    total_pass = 0
    total_criteria_count = 0
    for r in results:
        status_str = "PASS" if r["passed"] else "FAIL"
        if r["passed"]:
            total_pass += 1
        total_criteria_count += len(r["criteria"])
        crit_str = ", ".join(r["criteria"])
        print(f"{r['level']:<8} | {status_str:<8} | {r['duration_sec']:<6} s   | {crit_str}")

    print("-" * 80)
    success_rate = (total_pass / len(results)) * 100.0 if results else 0.0
    print(f"Total Niveles: {total_pass}/{len(results)} ({success_rate:.1f}%) | Criterios Cubiertos: {total_criteria_count}")
    print("=" * 80)

    if total_pass == len(results):
        print("\n>>> DEFINICIÓN DE HECHO (DoD) CUMPLIDA AL 100%: EL MVP DE CGE ESTÁ LISTO PARA ITERACIÓN. <<<")
    else:
        print("\n>>> ATENCIÓN: ALGUNOS CRITERIOS NO PASARON. REVISE EL LOG SUPERIOR. <<<")


def main():
    parser = argparse.ArgumentParser(description="Harness del Benchmark del DoD de CGE")
    parser.add_argument(
        "--level",
        type=str,
        default="all",
        choices=["all"] + list(LEVEL_MAP.keys()),
        help=f"Nivel del benchmark a ejecutar (por defecto: all; opciones: {', '.join(['all'] + list(LEVEL_MAP.keys()))})"
    )
    args = parser.parse_args()

    levels_to_run = list(LEVEL_MAP.keys()) if args.level == "all" else [args.level]
    results = []

    for lvl in levels_to_run:
        res = run_level(lvl)
        results.append(res)
        if not res["passed"]:
            print(res["stdout"])
            print(res["stderr"])

    print_summary_table(results)
    
    # Salir con código 1 si algún test falló
    if any(not r["passed"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
