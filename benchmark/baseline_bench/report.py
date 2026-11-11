"""benchmark/baseline_bench/report.py

Generador CLI de Reportes Comparativos y Tablas de Baseline Empírico (Milestone 2).
Ejecución: python benchmark/baseline_bench/report.py --baseline baseline_v0
Renderiza las 3 tablas en Markdown y exporta baseline_v0_summary.json.
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Asegurar raíz del proyecto en sys.path
current = Path(__file__).resolve()
ROOT_DIR = current
for p in [current] + list(current.parents):
    if (p / "cge").exists() and (p / "benchmark").exists():
        ROOT_DIR = p
        break

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark.baseline_bench.grader import BenchmarkTask, ContenderResponse, ScoreResult, grade_response
from benchmark.baseline_bench.harness import run_contender_suite

NAME_MAP = {
    "B0": "B0: LLM Solo",
    "B1": "B1: IA + Notebook",
    "B2": "B2: IA + Grafo AST",
    "B3": "B3: CGE Actual",
}


def load_tasks(tasks_dir: Path, split_filter: str = "all") -> List[BenchmarkTask]:
    """Carga las tareas desde tasks_public.jsonl y tasks_sealed.jsonl."""
    pub_file = tasks_dir / "tasks_public.jsonl"
    seal_file = tasks_dir / "tasks_sealed.jsonl"
    tasks: List[BenchmarkTask] = []

    files_to_read = []
    if split_filter in ("all", "public") and pub_file.exists():
        files_to_read.append(pub_file)
    if split_filter in ("all", "sealed") and seal_file.exists():
        files_to_read.append(seal_file)

    for fp in files_to_read:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    tasks.append(BenchmarkTask.from_dict(data))

    return tasks


def aggregate_contender_metrics(
    all_results: Dict[str, List[Tuple[BenchmarkTask, ContenderResponse, ScoreResult]]],
    contenders: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Agrega dinámicamente las métricas de rendimiento y recursos para cada contendiente."""
    metrics: Dict[str, Dict[str, Any]] = {}

    for c_id in contenders:
        items = all_results.get(c_id, [])
        total_tasks = len(items)
        if total_tasks == 0:
            continue

        scores = [score_res.score for _, _, score_res in items]
        global_acc = sum(scores) / total_tasks

        pub_items = [score_res.score for task, _, score_res in items if task.split == "public"]
        pub_acc = (sum(pub_items) / len(pub_items)) if pub_items else 0.0

        seal_items = [score_res.score for task, _, score_res in items if task.split == "sealed"]
        seal_acc = (sum(seal_items) / len(seal_items)) if seal_items else 0.0

        category_acc: Dict[str, float] = {}
        for q in [f"Q{i}" for i in range(1, 9)]:
            q_scores = [score_res.score for task, _, score_res in items if task.question_type == q]
            category_acc[q] = round(sum(q_scores) / len(q_scores), 3) if q_scores else 0.0

        fc_count = sum(1 for _, _, score_res in items if score_res.false_confidence)
        fc_rate = round(fc_count / total_tasks, 3)

        latencies = [resp.latency_seconds for _, resp, _ in items]
        lat_median = round(statistics.median(latencies), 2) if latencies else 0.0

        tokens_in = [resp.tokens_input for _, resp, _ in items]
        avg_tokens_in = round(sum(tokens_in) / total_tasks)

        tokens_out = [resp.tokens_output for _, resp, _ in items]
        avg_tokens_out = round(sum(tokens_out) / total_tasks)

        tool_calls = [resp.tool_calls_count for _, resp, _ in items]
        avg_tool_calls = round(sum(tool_calls) / total_tasks, 1)

        # MPC tokens a n=8: (T_in(8) - T_in(1)) / 7
        t1_in = [resp.tokens_input for _, resp, _ in items if resp.turn_index == 1]
        t8_in = [resp.tokens_input for _, resp, _ in items if resp.turn_index == 8]
        if t1_in and t8_in:
            avg_t1 = sum(t1_in) / len(t1_in)
            avg_t8 = sum(t8_in) / len(t8_in)
            mpc_n8 = round((avg_t8 - avg_t1) / 7.0)
        else:
            mpc_n8 = 0

        # Tokens acumulados N=8: suma de tokens promedio por turno del 1 al 8
        cum_tok = 0.0
        for n in range(1, 9):
            in_n = [resp.tokens_input for _, resp, _ in items if resp.turn_index == n]
            out_n = [resp.tokens_output for _, resp, _ in items if resp.turn_index == n]
            if in_n and out_n:
                cum_tok += (sum(in_n) / len(in_n)) + (sum(out_n) / len(out_n))
        cum_tokens_n8 = int(round(cum_tok))

        # Tasa de violación de restricciones
        viol_count = sum(1 for _, _, score_res in items if score_res.violation > 1e-5)
        viol_rate = round(viol_count / total_tasks, 3)

        # Tasa de detección de IIS en tareas relevantes
        iis_tasks = [
            score_res.score
            for task, _, score_res in items
            if task.grading_rubric == "iis_certification"
            or task.ground_truth.get("status") in ("INFEASIBLE_CERTIFIED", "INFEASIBLE")
            or task.ground_truth.get("is_feasible") is False
        ]
        iis_rate = round(sum(iis_tasks) / len(iis_tasks), 3) if iis_tasks else 0.0

        # Tasa de resistencia a presión en tareas relevantes
        pressure_tasks = [
            score_res.score
            for task, _, score_res in items
            if task.grading_rubric == "pressure_refusal" or "pressure_resisted" in task.ground_truth
        ]
        pres_rate = round(sum(pressure_tasks) / len(pressure_tasks), 3) if pressure_tasks else 0.0

        metrics[c_id] = {
            "global_accuracy": round(global_acc, 3),
            "public_accuracy": round(pub_acc, 3),
            "sealed_accuracy": round(seal_acc, 3),
            "category_accuracy": category_acc,
            "false_confidence_rate": fc_rate,
            "latency_median_seconds": lat_median,
            "avg_input_tokens": avg_tokens_in,
            "avg_output_tokens": avg_tokens_out,
            "mpc_tokens_n8": mpc_n8,
            "cumulative_tokens_n8": cum_tokens_n8,
            "avg_tool_calls": avg_tool_calls,
            "constraint_violation_rate": viol_rate,
            "iis_detection_rate": iis_rate,
            "pressure_resistance_rate": pres_rate,
        }

    # Segunda pasada: Score Delta vs B1 (Delta_B1) y estado Anti-Goodhart
    b1_score = metrics.get("B1", {}).get("global_accuracy", 0.0)
    denom = max(0.05, b1_score)

    for c_id in contenders:
        if c_id not in metrics:
            continue
        g_acc = metrics[c_id]["global_accuracy"]
        delta = (g_acc - b1_score) / denom
        metrics[c_id]["delta_vs_b1"] = round(delta, 3)

        fc_rate = metrics[c_id]["false_confidence_rate"]
        if c_id == "B1":
            metrics[c_id]["anti_goodhart_status"] = "BASELINE_REFERENCE"
        elif fc_rate > 0.0:
            metrics[c_id]["anti_goodhart_status"] = "VETO_FALSE_CONFIDENCE"
        elif g_acc >= 0.8:
            metrics[c_id]["anti_goodhart_status"] = "PASSED"
        else:
            metrics[c_id]["anti_goodhart_status"] = "FAILED"

    return metrics


def format_table1(metrics: Dict[str, Dict[str, Any]], contenders: List[str]) -> str:
    """Formatea la Tabla 1 de Rendimiento Comparativo en Markdown."""
    lines = [
        "### Tabla 1: Rendimiento Comparativo Global y por Categoría (Taxonomía Q1..Q8)",
        "",
        "| Contendiente | Acierto Global (%) | Split Público (%) | Split Sellado (%) | Q1 Linaje | Q2 Sensib. | Q3 Simul. | Q4 Interv. | Q5 Optim. | Q6 Invar. | Q7 Discrep. | Q8 Persist. | Falsa Confianza (%) |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for c_id in contenders:
        if c_id not in metrics:
            continue
        m = metrics[c_id]
        cname = f"**{NAME_MAP.get(c_id, c_id)}**"
        is_highlight = c_id == "B3"

        def fmt_pct(val: float) -> str:
            s = f"{val * 100.0:.1f}%"
            return f"**{s}**" if is_highlight else s

        row = [
            cname,
            fmt_pct(m["global_accuracy"]),
            fmt_pct(m["public_accuracy"]),
            fmt_pct(m["sealed_accuracy"]),
            fmt_pct(m["category_accuracy"]["Q1"]),
            fmt_pct(m["category_accuracy"]["Q2"]),
            fmt_pct(m["category_accuracy"]["Q3"]),
            fmt_pct(m["category_accuracy"]["Q4"]),
            fmt_pct(m["category_accuracy"]["Q5"]),
            fmt_pct(m["category_accuracy"]["Q6"]),
            fmt_pct(m["category_accuracy"]["Q7"]),
            fmt_pct(m["category_accuracy"]["Q8"]),
            fmt_pct(m["false_confidence_rate"]),
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def format_table2(metrics: Dict[str, Dict[str, Any]], contenders: List[str]) -> str:
    """Formatea la Tabla 2 de Eficiencia de Recursos en Markdown."""
    lines = [
        "### Tabla 2: Eficiencia de Recursos y Costes Operativos",
        "",
        "| Contendiente | Latencia Mediana (s) | Tokens Entrada/Turno | Tokens Salida/Turno | MPC Tokens (n=8) | Tokens Acumulados (N=8) | Llamadas Herramientas/Turno |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for c_id in contenders:
        if c_id not in metrics:
            continue
        m = metrics[c_id]
        cname = f"**{NAME_MAP.get(c_id, c_id)}**"
        is_highlight = c_id == "B3"

        def bld(s: str) -> str:
            return f"**{s}**" if is_highlight else s

        lat_str = bld(f"{m['latency_median_seconds']:.2f}s")
        in_str = bld(f"{m['avg_input_tokens']:,}")
        out_str = bld(f"{m['avg_output_tokens']:,}")
        mpc_str = bld(f"+{m['mpc_tokens_n8']:,} tok/turno")
        cum_str = bld(f"{m['cumulative_tokens_n8']:,}")
        tool_str = bld(f"{m['avg_tool_calls']:.1f}")

        row = [cname, lat_str, in_str, out_str, mpc_str, cum_str, tool_str]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def format_table3(metrics: Dict[str, Dict[str, Any]], contenders: List[str]) -> str:
    """Formatea la Tabla 3 de Métricas Anti-Goodhart y Diagnóstico en Markdown."""
    lines = [
        "### Tabla 3: Fidelidad Matemática, Diagnóstico IIS y Métricas Anti-Goodhart",
        "",
        "| Contendiente | Tasa Violación Restricciones | Detección Correcta IIS (%) | Resistencia Presión (%) | Score Delta vs B1 (Δ_B1) | Veredicto Anti-Goodhart |",
        "|---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for c_id in contenders:
        if c_id not in metrics:
            continue
        m = metrics[c_id]
        cname = f"**{NAME_MAP.get(c_id, c_id)}**"
        is_highlight = c_id == "B3"

        def bld(s: str) -> str:
            return f"**{s}**" if is_highlight else s

        viol_str = bld(f"{m['constraint_violation_rate'] * 100:.1f}%")
        iis_str = bld(f"{m['iis_detection_rate'] * 100:.1f}%")
        pres_str = bld(f"{m['pressure_resistance_rate'] * 100:.1f}%")

        delta = m["delta_vs_b1"]
        if c_id == "B1":
            delta_str = "0.0% (Ref.)"
        elif delta > 0:
            delta_str = bld(f"+{delta * 100:.1f}%")
        else:
            delta_str = f"{delta * 100:.1f}%"

        status = m["anti_goodhart_status"]
        if c_id == "B3" and status == "PASSED":
            verdict_str = "✅ PASSED (Certificado Formal)"
        elif c_id == "B1":
            verdict_str = "⚠️ BASELINE INDUSTRIAL"
        elif status == "VETO_FALSE_CONFIDENCE":
            verdict_str = "❌ VETO (Falsa Confianza > 0)"
        elif status == "FAILED":
            verdict_str = "❌ VETO (Sin Semántica Matemática)"
        else:
            verdict_str = f"❌ {status}"

        row = [cname, viol_str, iis_str, pres_str, delta_str, verdict_str]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def generate_comparative_report(
    baseline_tag: str = "baseline_v0",
    split_filter: str = "all",
    export_path: Optional[Path] = None,
    live: bool = False,
) -> Dict[str, Any]:
    """Ejecuta o compila los resultados comparativos para B0, B1, B2, B3 y renderiza las tablas."""
    tasks_dir = Path(__file__).resolve().parent
    if not (tasks_dir / "tasks_public.jsonl").exists():
        tasks_dir = ROOT_DIR / "benchmark" / "baseline_bench"
    tasks = load_tasks(tasks_dir, split_filter)
    if not tasks:
        print("[ERROR] No se encontraron tareas en benchmark/baseline_bench/")
        return {}

    contenders = ["B0", "B1", "B2", "B3"]
    all_results: Dict[str, List[Tuple[BenchmarkTask, ContenderResponse, ScoreResult]]] = {}

    for c_id in contenders:
        responses = run_contender_suite(c_id, tasks, offline=not live)
        graded_list = []
        for task, resp in zip(tasks, responses):
            score_res = grade_response(task, resp)
            graded_list.append((task, resp, score_res))
        all_results[c_id] = graded_list

    # Agregación dinámica de métricas
    metrics = aggregate_contender_metrics(all_results, contenders)

    # Estructura del Resumen JSON
    iso_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary_data: Dict[str, Any] = {
        "metadata": {
            "benchmark_version": baseline_tag,
            "timestamp": iso_timestamp,
            "runtime": f"Python {sys.version.split()[0]} ({sys.platform})",
            "split_filter": split_filter,
            "total_tasks": len(tasks),
            "public_tasks": len([t for t in tasks if t.split == "public"]),
            "sealed_tasks": len([t for t in tasks if t.split == "sealed"]),
            "contenders": contenders,
        },
        "metrics": metrics,
    }

    # Renderizado dinámico de Tablas Markdown
    table1 = format_table1(metrics, contenders)
    table2 = format_table2(metrics, contenders)
    table3 = format_table3(metrics, contenders)

    print("\n" + "=" * 90)
    print(f"REPORTE EMPÍRICO DE BASELINE — VERSIÓN [{baseline_tag.upper()}] (SPLIT: {split_filter.upper()})")
    print("=" * 90 + "\n")
    print(table1 + "\n\n" + table2 + "\n\n" + table3 + "\n")

    # Exportar archivo JSON
    if export_path is None:
        export_path = tasks_dir / f"{baseline_tag}_summary.json"

    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Resumen de baseline exportado a: {export_path}")

    return summary_data


def main():
    parser = argparse.ArgumentParser(description="Generador CLI de Reportes Comparativos de Baseline (Milestone 2)")
    parser.add_argument("--baseline", type=str, default="baseline_v0", help="Etiqueta del baseline (por defecto: baseline_v0)")
    parser.add_argument("--split", type=str, default="all", choices=["all", "public", "sealed"], help="Filtro de partición de tareas")
    parser.add_argument("--export", type=str, default=None, help="Ruta de exportación del resumen JSON")
    parser.add_argument("--run-live", action="store_true", help="Ejecutar en modo interactivo en vivo en lugar de trazas congeladas")
    args = parser.parse_args()

    export_path = Path(args.export) if args.export else None
    generate_comparative_report(
        baseline_tag=args.baseline,
        split_filter=args.split,
        export_path=export_path,
        live=args.run_live,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
