"""Ejecutor CLI para iniciar el servidor UI y abrir la visualización en el navegador."""
from __future__ import annotations
import argparse
import os
import sys
import webbrowser
import uvicorn

from ui.server import app, engine_state


def main():
    parser = argparse.ArgumentParser(description="Lanza el servidor interactivo de Causal Graph Engine (Fase 6).")
    parser.add_argument(
        "target_dir",
        nargs="?",
        default=None,
        help="Directorio del sistema causal a analizar (opcional)."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Dirección host (por defecto: 127.0.0.1)."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Puerto HTTP (por defecto: 8765)."
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="No abrir automáticamente el navegador."
    )

    args = parser.parse_args()

    # Determinar el directorio a ingestar
    if args.target_dir and not os.path.isdir(args.target_dir):
        sys.stderr.write(f"[CGE UI] Error: El directorio objetivo '{args.target_dir}' no existe.\n")
        sys.exit(1)

    chosen_dir = args.target_dir or os.environ.get("CGE_TARGET_DIR")
    if not chosen_dir:
        excluded_names = {
            "cge", "benchmark", "oracle", "ui", "math_lab", "docs",
            "schema", "eval_reports", "benchmark_variants", ".pytest_cache", "__pycache__"
        }
        for entry in sorted(os.listdir(".")):
            if os.path.isdir(entry) and not entry.startswith(".") and entry not in excluded_names:
                sub_files = os.listdir(entry)
                if any(sf.endswith(".py") for sf in sub_files):
                    chosen_dir = entry
                    break

    if chosen_dir and os.path.isdir(chosen_dir):
        print(f"[CGE UI] Ingestando directorio objetivo: {chosen_dir}")
        try:
            engine_state.load_directory(chosen_dir)
            print(f"[CGE UI] Grafo cargado exitosamente: {len(engine_state.graph.nodes)} nodos, "
                  f"{len(engine_state.graph.edges)} aristas.")
        except Exception as err:
            print(f"[CGE UI] Advertencia: Error durante la ingesta previa: {err}")

    server_url = f"http://{args.host}:{args.port}"
    print("=" * 72)
    print("   CAUSAL GRAPH ENGINE — SERVIDOR DE VISUALIZACIÓN INTERACTIVA (FASE 6)")
    print("=" * 72)
    print(f"   URL del visualizador: {server_url}")
    print(f"   Documentación OpenAPI: {server_url}/docs")
    print("=" * 72)

    if not args.no_browser:
        import threading

        def _open_browser_when_ready():
            try:
                webbrowser.open(server_url)
            except Exception:
                pass

        threading.Timer(0.8, _open_browser_when_ready).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
