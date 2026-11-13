"""Servidor MCP (Model Context Protocol) estándar de ActorDB & Causal Graph Engine (CGE).

Expone las 4 herramientas causales deterministas sin alucinaciones para agentes externos:
- `cge_trace`: Linaje causal, derivadas totales exactas, ranking de palancas y nodos muertos (T1).
- `cge_do`: Intervención contrafactual exacta de Pearl `do(X=x, V=v)` con cirugía de grafo.
- `cge_solve`: Optimización matemática prescriptiva certificada (LP/MILP/QP/NLP/MINLP/IIS).
- `cge_prove`: Verificación formal de invariantes, búsqueda de contraejemplos exactos y certificación IIS.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import sys
import threading
from typing import Any, Callable, Dict, Iterator, List, Optional

from mcp.server.fastmcp import FastMCP

from .oracle import ActorDBOracle


_STDOUT_REDIRECT_LOCK = threading.Lock()
_STDOUT_REDIRECT_DEPTH = 0
_SAVED_REAL_STDOUT: Any = None
_ORACLE_EXEC_LOCK = threading.RLock()
_ORACLE_EXECUTOR_INIT_LOCK = threading.Lock()
_ORACLE_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None
_WORKER_STACK_BYTES = 8 * 1024 * 1024


@contextlib.contextmanager
def _safe_redirect_stdout_to_stderr() -> Iterator[None]:
    """Redirige `sys.stdout -> sys.stderr` de forma thread-safe mediante conteo de referencias."""
    global _STDOUT_REDIRECT_DEPTH, _SAVED_REAL_STDOUT
    with _STDOUT_REDIRECT_LOCK:
        if _STDOUT_REDIRECT_DEPTH == 0:
            _SAVED_REAL_STDOUT = sys.stdout
            sys.stdout = sys.stderr
        _STDOUT_REDIRECT_DEPTH += 1
    try:
        yield
    finally:
        with _STDOUT_REDIRECT_LOCK:
            _STDOUT_REDIRECT_DEPTH -= 1
            if _STDOUT_REDIRECT_DEPTH == 0 and _SAVED_REAL_STDOUT is not None:
                sys.stdout = _SAVED_REAL_STDOUT
                _SAVED_REAL_STDOUT = None


def _get_oracle_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Inicializa un worker dedicado con pila de 8 MB (evita stack overflow en Windows en SymPy/HiGHS)
    y serializa el acceso a las cachés globales no thread-safe de SymPy sin bloquear el event loop ASGI.
    """
    global _ORACLE_EXECUTOR
    if _ORACLE_EXECUTOR is not None:
        return _ORACLE_EXECUTOR
    with _ORACLE_EXECUTOR_INIT_LOCK:
        if _ORACLE_EXECUTOR is None:
            prev_stack = 0
            stack_updated = False
            try:
                prev_stack = threading.stack_size(_WORKER_STACK_BYTES)
                stack_updated = True
            except (ValueError, RuntimeError):
                stack_updated = False
            try:
                executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="cge-oracle-worker",
                )
                # Forzar materialización inmediata del hilo bajo el tamaño de pila de 8 MB
                executor.submit(lambda: None).result()
                _ORACLE_EXECUTOR = executor
            finally:
                if stack_updated:
                    with contextlib.suppress(ValueError, RuntimeError):
                        threading.stack_size(prev_stack)
        return _ORACLE_EXECUTOR


async def _run_oracle_tool(call: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    """Ejecuta una primitiva causal fuera del event loop ASGI con protección thread-safe de `stdout` y SymPy."""
    def _sync_wrapper() -> Dict[str, Any]:
        with _ORACLE_EXEC_LOCK, _safe_redirect_stdout_to_stderr():
            return call()

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_get_oracle_executor(), _sync_wrapper)


SERVER_INSTRUCTIONS = (
    "ActorDB Causal Mathematical Oracle (Zero-Hallucination Engine). "
    "Use `cge_trace` to inspect causal lineage, total derivatives, and dead nodes (T1); "
    "`cge_do` to execute Pearlian counterfactual interventions do(X=x) with graph surgery; "
    "`cge_solve` to compute certified optimal policies under hard constraints (or Minimal IIS if infeasible); "
    "and `cge_prove` to formally verify invariants (`PROVEN`, `SAMPLED_VERIFIED`, or `FALSIFIED` with exact counterexample). "
    "`source_path` accepts adapter repo IDs (e.g. `supplychainpy_mini`, `economic_dispatch`, `loan_tax_engine`, "
    "`quantecon_cournot_solow`, `epidemiology_seir`), explicit `:submodel` selectors (e.g. `quantecon_cournot_solow:cournot`), "
    "repository directories, `.py` modules, JSON IR files, or inline Python source."
)


def create_mcp_server(
    name: str = "actordb-cge",
    default_source: Optional[Any] = None,
    allow_auto_discovery: bool = True,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> FastMCP:
    """Crea y configura una instancia estándar de Servidor MCP con las 4 herramientas causales."""
    oracle = ActorDBOracle(
        default_source=default_source,
        allow_auto_discovery=allow_auto_discovery,
    )
    server = FastMCP(name=name, instructions=SERVER_INSTRUCTIONS, host=host, port=port)
    # Adjuntar referencia al oráculo para introspección y gestión de caché programática
    server.oracle = oracle  # type: ignore[attr-defined]

    @server.tool(
        name="cge_trace",
        description=(
            "Trace causal lineage toward a target actor, compute exact symbolic and numeric total derivatives "
            "(dTarget/dSource) and dimensionless elasticities, rank control levers, and detect dead actors "
            "(T1-A structural orphans, T1-B algebraic path cancellations, T1-C disconnected controls)."
        ),
    )
    async def cge_trace(
        target_id: Optional[str] = None,
        source_id: Optional[str] = None,
        top_k_paths: int = 5,
        source_path: Optional[str] = None,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return await _run_oracle_tool(
            lambda: oracle.trace(
                target_id=target_id,
                source_id=source_id,
                top_k_paths=top_k_paths,
                source_path=source_path,
                graph_data=graph_data,
            )
        )

    @server.tool(
        name="cge_do",
        description=(
            "Execute a deterministic Pearlian counterfactual intervention do(Actor=value) with exact graph surgery "
            "(severing incoming causal edges into intervened actors), propagate state forward, compute causal effect "
            "vs baseline, and check post-intervention hard constraints."
        ),
    )
    async def cge_do(
        interventions: Dict[str, float | int | bool],
        target_id: Optional[str] = None,
        context: Optional[Dict[str, float]] = None,
        source_path: Optional[str] = None,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return await _run_oracle_tool(
            lambda: oracle.do(
                interventions=interventions,
                target_id=target_id,
                context=context,
                source_path=source_path,
                graph_data=graph_data,
            )
        )

    @server.tool(
        name="cge_solve",
        description=(
            "Compute a certified optimal control policy (LP, MILP, QP, NLP, MINLP) subject to all hard constraints "
            "and Epistemic Honesty Gate. Returns CERTIFIED_OPTIMAL/BEST_FOUND recipes, INFEASIBLE_CERTIFIED with "
            "Minimal Irreducible Inconsistent Subsystem (IIS) and KKT shadow multipliers, or UNSUPPORTED."
        ),
    )
    async def cge_solve(
        target_id: Optional[str] = None,
        sense: Optional[str] = None,
        target_value: Optional[float] = None,
        fixed_controls: Optional[Dict[str, float]] = None,
        exo_overrides: Optional[Dict[str, float]] = None,
        ignore_constraints: bool = False,
        source_path: Optional[str] = None,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return await _run_oracle_tool(
            lambda: oracle.solve(
                target_id=target_id,
                sense=sense,
                target_value=target_value,
                fixed_controls=fixed_controls,
                exo_overrides=exo_overrides,
                ignore_constraints=ignore_constraints,
                source_path=source_path,
                graph_data=graph_data,
            )
        )

    @server.tool(
        name="cge_prove",
        description=(
            "Formally verify a quantitative, conservation, or logical invariant (e.g. 'sales <= order_qty', "
            "'order_qty >= 800 => unit_cost <= 1.0', or a declared constraint_id) across the admissible domain. "
            "Emits strict epistemic seals: PROVEN, SAMPLED_VERIFIED, FALSIFIED (with concrete counterexample), "
            "INFEASIBLE_DOMAIN (with Minimal IIS), or UNSUPPORTED."
        ),
    )
    async def cge_prove(
        invariant_expr: Optional[str] = None,
        constraint_id: Optional[str] = None,
        assume_system_constraints: bool = True,
        domain_bounds: Optional[Dict[str, List[float]]] = None,
        n_samples: int = 128,
        source_path: Optional[str] = None,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return await _run_oracle_tool(
            lambda: oracle.prove(
                invariant_expr=invariant_expr,
                constraint_id=constraint_id,
                assume_system_constraints=assume_system_constraints,
                domain_bounds=domain_bounds,
                n_samples=n_samples,
                source_path=source_path,
                graph_data=graph_data,
            )
        )

    return server


# Instancia global estándar lista para `mcp run cge/mcp/server.py` o importación directa
mcp: FastMCP = create_mcp_server()
default_oracle: ActorDBOracle = mcp.oracle  # type: ignore[attr-defined]


def cge_trace(
    target_id: Optional[str] = None,
    source_id: Optional[str] = None,
    top_k_paths: int = 5,
    source_path: Optional[str] = None,
    graph_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return default_oracle.trace(
        target_id=target_id,
        source_id=source_id,
        top_k_paths=top_k_paths,
        source_path=source_path,
        graph_data=graph_data,
    )


def cge_do(
    interventions: Dict[str, Any],
    target_id: Optional[str] = None,
    context: Optional[Dict[str, float]] = None,
    source_path: Optional[str] = None,
    graph_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return default_oracle.do(
        interventions=interventions,
        target_id=target_id,
        context=context,
        source_path=source_path,
        graph_data=graph_data,
    )


def cge_solve(
    target_id: Optional[str] = None,
    sense: Optional[str] = None,
    target_value: Optional[float] = None,
    fixed_controls: Optional[Dict[str, float]] = None,
    exo_overrides: Optional[Dict[str, float]] = None,
    ignore_constraints: bool = False,
    source_path: Optional[str] = None,
    graph_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return default_oracle.solve(
        target_id=target_id,
        sense=sense,
        target_value=target_value,
        fixed_controls=fixed_controls,
        exo_overrides=exo_overrides,
        ignore_constraints=ignore_constraints,
        source_path=source_path,
        graph_data=graph_data,
    )


def cge_prove(
    invariant_expr: Optional[str] = None,
    constraint_id: Optional[str] = None,
    assume_system_constraints: bool = True,
    domain_bounds: Optional[Dict[str, List[float]]] = None,
    n_samples: int = 128,
    source_path: Optional[str] = None,
    graph_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return default_oracle.prove(
        invariant_expr=invariant_expr,
        constraint_id=constraint_id,
        assume_system_constraints=assume_system_constraints,
        domain_bounds=domain_bounds,
        n_samples=n_samples,
        source_path=source_path,
        graph_data=graph_data,
    )
