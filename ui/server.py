"""Servidor Web REST FastAPI para la Interfaz Interactiva CGE y Copiloto Agéntico (Fase 6)."""
from __future__ import annotations
import json
import os
import re
import threading
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cge.schema import CausalGraph
from cge.ingest import ASTIngestPipeline
from cge.layer1_heuristic import Layer1Router
from cge.layer2_exact import Layer2Solver
from cge.audit import SystemAuditor
from cge.monte_carlo import optimize_stochastic_saa_cvar
from cge.copilot import DeterministicCopilot


class IngestRequest(BaseModel):
    path: Optional[str] = None
    dir_path: Optional[str] = None
    target_dir: Optional[str] = None
    model_config = {"extra": "allow"}


class EvaluateRequest(BaseModel):
    controls: Optional[Dict[str, Any]] = None
    model_config = {"extra": "allow"}


class OptimizeRequest(BaseModel):
    target_value: Optional[float] = None
    use_cvar: bool = False
    cvar_beta: Optional[float] = 0.95
    risk_aversion: Optional[float] = 0.2
    n_scenarios: int = 100
    model_config = {"extra": "allow"}


class CopilotRequest(BaseModel):
    question: str


class MutateRequest(BaseModel):
    schema_data: Optional[Dict[str, Any]] = Field(default=None, alias="schema")
    model_config = {"extra": "allow"}


class EngineState:
    """Estado en memoria desacoplado y protegido contra condiciones de carrera del motor CGE."""

    def __init__(self):
        self._lock = threading.RLock()
        self.current_dir: Optional[str] = None
        self.graph: Optional[CausalGraph] = None
        self.router: Optional[Layer1Router] = None
        self.solver: Optional[Layer2Solver] = None
        self.auditor: Optional[SystemAuditor] = None
        self.copilot: Optional[DeterministicCopilot] = None
        self.diagnostics: List[Dict[str, Any]] = []

    def is_loaded(self) -> bool:
        with self._lock:
            return self.graph is not None and self.solver is not None

    def load_directory(self, target_dir: str):
        if not os.path.isdir(target_dir):
            raise FileNotFoundError(f"Target directory '{target_dir}' does not exist.")
        clean_dir = os.path.abspath(target_dir)
        pipeline = ASTIngestPipeline(clean_dir)
        graph = pipeline.run()

        solver = Layer2Solver(graph=graph, source_dir=clean_dir)
        router = Layer1Router(graph=graph, source_dir=clean_dir)
        auditor = SystemAuditor(graph=graph, solver=solver, source_dir=clean_dir)
        copilot = DeterministicCopilot(graph=graph, solver=solver, source_dir=clean_dir)
        diagnostics = auditor.audit_all()

        with self._lock:
            self.current_dir = clean_dir
            self.graph = graph
            self.solver = solver
            self.router = router
            self.auditor = auditor
            self.copilot = copilot
            self.diagnostics = diagnostics

    def ensure_initialized(self):
        with self._lock:
            if self.is_loaded():
                return
            env_dir = os.environ.get("CGE_TARGET_DIR")
            if env_dir and os.path.isdir(env_dir):
                self.load_directory(env_dir)
                return

            excluded_names = {
                "cge", "benchmark", "oracle", "ui", "math_lab", "docs",
                "schema", "eval_reports", "benchmark_variants", ".pytest_cache", "__pycache__"
            }
            for entry in sorted(os.listdir(".")):
                if os.path.isdir(entry) and not entry.startswith(".") and entry not in excluded_names:
                    sub_files = os.listdir(entry)
                    if any(sf.endswith(".py") for sf in sub_files):
                        self.load_directory(entry)
                        return

            if os.path.exists("schema.json"):
                with open("schema.json", "r", encoding="utf-8") as f:
                    raw_schema = json.load(f)
                graph = CausalGraph.from_dict(raw_schema)
                solver = Layer2Solver.from_graph(graph)
                router = Layer1Router(graph=graph)
                auditor = SystemAuditor(graph=graph, solver=solver)
                copilot = DeterministicCopilot(graph=graph, solver=solver)
                diagnostics = auditor.audit_all()

                self.current_dir = None
                self.graph = graph
                self.solver = solver
                self.router = router
                self.auditor = auditor
                self.copilot = copilot
                self.diagnostics = diagnostics
                return

            raise RuntimeError("No target directory or graph available. Please call /api/ingest first.")


engine_state = EngineState()

app = FastAPI(
    title="Causal Graph Engine REST API",
    description="Backend desacoplado para visualización interactiva y copiloto causal.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/ingest")
def api_ingest(payload: IngestRequest):
    """Analiza un directorio arbitrario y actualiza el estado en memoria."""
    target_path = payload.dir_path or payload.path or payload.target_dir
    if not target_path:
        raise HTTPException(status_code=400, detail="Missing required field 'dir_path', 'path' or 'target_dir'.")
    if not os.path.isdir(target_path):
        raise HTTPException(status_code=404, detail=f"Directory '{target_path}' not found.")

    try:
        engine_state.load_directory(target_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(exc)}")

    with engine_state._lock:
        meta = engine_state.graph.metadata or {}
        return {
            "status": "ok",
            "system_id": engine_state.graph.system_id,
            "version": engine_state.graph.version,
            "nodes_count": len(engine_state.graph.nodes),
            "edges_count": len(engine_state.graph.edges),
            "constraints_count": len(engine_state.graph.constraints),
            "diagnostics_count": len(engine_state.diagnostics),
            "source_modalities": meta.get("source_modalities", {})
        }


@app.get("/api/schema")
def api_schema():
    """Devuelve el schema.json actual y su versión."""
    engine_state.ensure_initialized()
    with engine_state._lock:
        data = engine_state.graph.to_dict()
    return data


@app.get("/api/source")
def api_source(
    node_id: Optional[str] = Query(default=None),
    file: Optional[str] = Query(default=None),
    start_line: Optional[int] = Query(default=None),
    end_line: Optional[int] = Query(default=None),
    context_lines: int = Query(default=5)
):
    """Devuelve el fragmento de código exacto de un nodo con protección estricta contra path traversal y escapes del proyecto."""
    engine_state.ensure_initialized()

    with engine_state._lock:
        allowed_dir = os.path.abspath(engine_state.current_dir) if engine_state.current_dir else os.path.abspath(".")
        active_nodes = engine_state.graph.nodes if engine_state.graph else {}

    target_file = file
    s_line = start_line
    e_line = end_line
    entity_name = None
    col_name = None

    if node_id:
        if node_id not in active_nodes:
            raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found in active graph.")
        node_obj = active_nodes[node_id]
        if node_obj.source_reference:
            target_file = target_file or node_obj.source_reference.file
            entity_name = node_obj.source_reference.entity
            col_name = node_obj.source_reference.column_name
            if node_obj.source_reference.line_range:
                s_line = s_line or node_obj.source_reference.line_range[0]
                e_line = e_line or node_obj.source_reference.line_range[1]

    if not target_file:
        raise HTTPException(status_code=400, detail="Source file not specified and node has no reference.")

    # Blindaje estricto contra Path Traversal ('..' o byte nulo)
    if ".." in target_file or "\x00" in target_file:
        raise HTTPException(status_code=400, detail="Path traversal detected: invalid path sequence.")

    clean_file = target_file.replace("\\", "/").strip()

    # Si es ruta absoluta o con letra de unidad Windows, verificar pertenencia al proyecto
    is_absolute_drive = bool(re.match(r"^[a-zA-Z]:", clean_file)) or clean_file.startswith("/") or os.path.isabs(clean_file)
    candidate_paths: List[str] = []

    if is_absolute_drive:
        abs_target = os.path.abspath(clean_file)
        try:
            if os.path.commonpath([allowed_dir, abs_target]) != allowed_dir:
                raise HTTPException(status_code=403, detail="Access denied: file resides outside project directory.")
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied: file resides outside project directory.")
        candidate_paths.append(abs_target)
    else:
        # Rutas relativas: deben resolverse exclusivamente dentro de allowed_dir
        rel_clean = clean_file.lstrip("/")
        candidate_paths.append(os.path.abspath(os.path.join(allowed_dir, rel_clean)))

        top_name = os.path.basename(allowed_dir)
        if rel_clean.startswith(top_name + "/"):
            sub_part = rel_clean[len(top_name) + 1:]
            candidate_paths.append(os.path.abspath(os.path.join(allowed_dir, sub_part)))

    chosen_path = None
    for cp in candidate_paths:
        try:
            if os.path.commonpath([allowed_dir, cp]) != allowed_dir:
                raise HTTPException(status_code=403, detail="Access denied: file resides outside project directory.")
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied: file resides outside project directory.")

        if os.path.isfile(cp):
            chosen_path = cp
            break

    if not chosen_path:
        raise HTTPException(status_code=404, detail=f"Source file '{clean_file}' not found.")

    with open(chosen_path, "r", encoding="utf-8", errors="replace") as f:
        file_lines = f.readlines()

    total_lines = len(file_lines)
    s_idx = max(1, int(s_line)) if s_line else 1
    e_idx = min(total_lines, int(e_line)) if e_line else total_lines
    if s_idx > e_idx:
        s_idx, e_idx = e_idx, s_idx

    ctx_start = max(1, s_idx - context_lines)
    ctx_end = min(total_lines, e_idx + context_lines)

    code_snippet = "".join(file_lines[ctx_start - 1:ctx_end])

    return {
        "node_id": node_id,
        "file": target_file,
        "entity": entity_name,
        "column_name": col_name,
        "line_range": [s_idx, e_idx],
        "context_range": [ctx_start, ctx_end],
        "code": code_snippet,
        "total_lines": total_lines
    }


@app.get("/api/diagnostics")
def api_diagnostics():
    """Devuelve los diagnósticos T1-T7 del auditor."""
    engine_state.ensure_initialized()
    with engine_state._lock:
        diags = list(engine_state.diagnostics)
    return {
        "diagnostics": diags,
        "count": len(diags)
    }


@app.get("/api/importance")
def api_importance(target: Optional[str] = Query(default=None)):
    """Devuelve las derivadas totales y elasticidades adimensionales de la Capa 1."""
    engine_state.ensure_initialized()
    with engine_state._lock:
        router = engine_state.router
        graph = engine_state.graph
        solver = engine_state.solver

    if not router or not graph or not solver:
        raise HTTPException(status_code=500, detail="Engine router not available.")

    if target and target not in graph.nodes:
        raise HTTPException(status_code=404, detail=f"Target node '{target}' not found in active graph.")

    target_id = target or solver.target_id or router._get_default_target()
    adjoints = router.compute_reverse_mode_ad_adjoints(target_id)
    ranked_controls = router.rank_control_variables(target_id)

    target_scale = 1.0
    if target_id in graph.nodes:
        target_scale = float(graph.nodes[target_id].characteristic_scale or 1.0)

    elasticities: Dict[str, float] = {}
    for nid, adj_val in adjoints.items():
        node_scale = 1.0
        if nid in graph.nodes:
            node_scale = float(graph.nodes[nid].characteristic_scale or 1.0)
        elast = float(adj_val) * (node_scale / max(target_scale, 1e-6))
        elasticities[nid] = round(elast, 4)

    return {
        "target": target_id,
        "adjoints": adjoints,
        "elasticities": elasticities,
        "ranked_controls": ranked_controls
    }


@app.get("/api/paths")
def api_paths(
    source: Optional[str] = Query(default=None),
    target: Optional[str] = Query(default=None),
    k: int = Query(default=5)
):
    """Devuelve los Top-K caminos causales hasta el target."""
    engine_state.ensure_initialized()
    with engine_state._lock:
        router = engine_state.router
        graph = engine_state.graph
        solver = engine_state.solver

    if not router or not graph or not solver:
        raise HTTPException(status_code=500, detail="Engine router not available.")

    if target and target not in graph.nodes:
        raise HTTPException(status_code=404, detail=f"Target node '{target}' not found in active graph.")
    if source and source not in graph.nodes:
        raise HTTPException(status_code=404, detail=f"Source node '{source}' not found in active graph.")

    target_id = target or solver.target_id or router._get_default_target()

    if source:
        paths = router.find_top_k_paths(source, target_id, k=k)
        return {
            "target": target_id,
            "source": source,
            "paths": paths
        }

    paths_by_control = router.get_all_control_to_target_paths(target=target_id, k_per_control=k)
    flat_paths = []
    for c_id, plist in paths_by_control.items():
        flat_paths.extend(plist)

    return {
        "target": target_id,
        "paths_by_control": paths_by_control,
        "paths": flat_paths
    }


@app.post("/api/evaluate")
def api_evaluate(payload: Dict[str, Any]):
    """Evaluación qué-pasa-si en tiempo real dado un vector de controles."""
    engine_state.ensure_initialized()
    with engine_state._lock:
        solver = engine_state.solver

    if not solver:
        raise HTTPException(status_code=500, detail="Engine solver not available.")

    ctrl_input = payload.get("controls") if "controls" in payload and isinstance(payload["controls"], dict) else payload

    numeric_controls: Dict[str, float] = {}
    for k, v in ctrl_input.items():
        if k != "controls":
            try:
                numeric_controls[k] = float(v)
            except (ValueError, TypeError):
                pass

    try:
        eval_result = solver.evaluate(**numeric_controls)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(exc)}")

    target_id = solver.target_id
    target_val = float(eval_result.get(target_id, 0.0))

    serialized_states = {}
    for k, v in eval_result.items():
        if isinstance(v, (int, float)):
            serialized_states[k] = float(v)
        elif isinstance(v, dict):
            serialized_states[k] = v
        else:
            try:
                serialized_states[k] = float(v)
            except (ValueError, TypeError):
                serialized_states[k] = str(v)

    return {
        "target_id": target_id,
        "target_value": target_val,
        "state_values": serialized_states
    }


@app.post("/api/optimize")
def api_optimize(payload: Optional[Dict[str, Any]] = None):
    """Resolución de recetas óptimas, soporte de riesgo CVaR o diagnóstico KKT de infactibilidad."""
    engine_state.ensure_initialized()
    with engine_state._lock:
        solver = engine_state.solver

    if not solver:
        raise HTTPException(status_code=500, detail="Engine solver not available.")

    data = payload or {}
    use_cvar = bool(data.get("use_cvar", False))
    target_value = data.get("target_value")

    if target_value is not None:
        try:
            target_value = float(target_value)
        except (ValueError, TypeError):
            target_value = None

    if use_cvar:
        n_scenarios = int(data.get("n_scenarios", 100))
        beta = float(data.get("cvar_beta", 0.95))
        risk_aversion = float(data.get("risk_aversion", 0.2))
        res = optimize_stochastic_saa_cvar(
            n_scenarios=n_scenarios,
            beta=beta,
            risk_aversion=risk_aversion,
            seed=42,
            solver=solver
        )
        optimal_controls = res.get("controls", {})
        eval_res = solver.evaluate(**optimal_controls) if optimal_controls else {}
        b_spent = float(eval_res.get("budget_used", 0.0))
        b_margin = float(eval_res.get("budget_margin", 0.0))

        mc_metrics = res.get("monte_carlo_metrics", {})
        cvar_loss_val = float(res.get("cvar_95", res.get("cvar_loss", mc_metrics.get("cvar_5pct", 0.0))))
        var_candidate = next((float(v) for k, v in mc_metrics.items() if k.startswith("p5") and isinstance(v, (int, float))), 0.0)
        var_loss_val = float(res.get("var_loss", mc_metrics.get("var_95", var_candidate)))

        return {
            "status": res.get("status", "FEASIBLE"),
            "controls": optimal_controls,
            "objective_value": float(res.get("expected_objective", res.get("objective_value", 0.0))),
            "expected_objective": float(res.get("expected_objective", 0.0)),
            "cvar_loss": cvar_loss_val,
            "var_loss": var_loss_val,
            "budget_spent": b_spent,
            "budget_margin": b_margin,
            "cvar_mode": True
        }

    opt_res = solver.optimize(target_value=target_value)
    if opt_res.get("status") == "UNSUPPORTED":
        return {
            "status": "UNSUPPORTED",
            "certificate_status": "UNSUPPORTED",
            "certificate": opt_res.get("certificate"),
            "reason": opt_res.get("reason"),
            "unverified_nodes": opt_res.get("unverified_nodes", []),
            "controls": {},
            "objective_value": None,
            "cvar_mode": False,
        }

    kkt_safe = {str(k): float(v) for k, v in opt_res.get("kkt_multipliers", {}).items()}
    max_j = opt_res.get("max_achievable_J_star")
    obj_val = opt_res.get("objective_value")

    return {
        "status": opt_res.get("status", "FEASIBLE"),
        "certificate_status": opt_res.get("certificate_status"),
        "certificate": opt_res.get("certificate"),
        "controls": opt_res.get("controls", {}),
        "objective_value": float(obj_val) if obj_val is not None else None,
        "budget_spent": float(opt_res.get("budget_spent", 0.0)),
        "budget_margin": float(opt_res.get("budget_margin", 0.0)),
        "active_regime": opt_res.get("active_regime"),
        "kkt_multipliers": kkt_safe,
        "blocking_constraints": opt_res.get("blocking_constraints", []),
        "iis_constraints": opt_res.get("iis_constraints", []),
        "max_achievable_J_star": float(max_j) if max_j is not None else None,
        "cvar_mode": False
    }


@app.post("/api/copilot")
def api_copilot(payload: Dict[str, Any]):
    """Consultas semánticas devolviendo respuesta y entidades/aristas a resaltar."""
    engine_state.ensure_initialized()
    with engine_state._lock:
        copilot = engine_state.copilot

    if not copilot:
        raise HTTPException(status_code=500, detail="Engine copilot not available.")

    question = payload.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    ans = copilot.answer_question(question)
    return ans


@app.post("/api/mutate")
def api_mutate(payload: Dict[str, Any]):
    """Mutaciones estructuradas validadas atómicamente contra canonical_schema.json."""
    engine_state.ensure_initialized()
    schema_candidate = payload.get("schema") if "schema" in payload and isinstance(payload["schema"], dict) else payload

    # Localizar archivo canonical_schema.json de manera determinista
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "schema", "canonical_schema.json"),
        os.path.join("schema", "canonical_schema.json")
    ]
    canonical_path = next((p for p in candidates if os.path.exists(p)), None)
    if canonical_path:
        import jsonschema
        with open(canonical_path, "r", encoding="utf-8") as f:
            canonical_def = json.load(f)
        try:
            jsonschema.validate(instance=schema_candidate, schema=canonical_def)
        except jsonschema.ValidationError as err:
            raise HTTPException(status_code=400, detail=f"Canonical schema validation error: {err.message}")

    with engine_state._lock:
        curr_dir = engine_state.current_dir

    try:
        new_graph = CausalGraph.from_dict(schema_candidate)
        new_solver = Layer2Solver.from_graph(new_graph)
        new_router = Layer1Router(graph=new_graph, source_dir=curr_dir)
        new_auditor = SystemAuditor(graph=new_graph, solver=new_solver, source_dir=curr_dir)
        new_copilot = DeterministicCopilot(graph=new_graph, solver=new_solver, source_dir=curr_dir)
        new_diagnostics = new_auditor.audit_all()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"State re-hydration failed: {str(exc)}")

    # Asignación atómica protegida bajo lock
    with engine_state._lock:
        engine_state.graph = new_graph
        engine_state.solver = new_solver
        engine_state.router = new_router
        engine_state.auditor = new_auditor
        engine_state.copilot = new_copilot
        engine_state.diagnostics = new_diagnostics

    return {
        "status": "ok",
        "message": "Schema successfully mutated and state updated",
        "version": new_graph.version,
        "nodes_count": len(new_graph.nodes)
    }


# Montar archivos estáticos y ruta principal
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
def serve_index():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.isfile(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h1>CGE UI Server</h1><p>Frontend static files not found in ui/static/index.html.</p>")
