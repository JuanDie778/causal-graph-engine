"""Suite completa de pruebas unitarias e integración MCP para `cge.mcp` (Servidor MCP Causal de ActorDB).

Verifica:
1. Protocolo MCP estándar en memoria (`create_connected_server_and_client_session`):
   descubrimiento de exactamente 4 herramientas (`cge_trace`, `cge_do`, `cge_solve`, `cge_prove`),
   esquemas JSON válidos e invocación `call_tool` libre de errores.
2. Transporte real `stdio` en subproceso OS (`stdio_client` + `python -m cge.mcp`):
   aislamiento del canal JSON-RPC en `stdout` frente a código de usuario ruidoso con `print()`.
3. `cge_trace`: linaje causal, derivadas simbólicas/numéricas exactas, ranking de palancas,
   detección de nodos muertos (`T1-A` estructural, `T1-B` cancelación algebraica, `T1-C` control desconectado),
   sellos epistémicos (`PROVEN`, `SAMPLED_VERIFIED`, `UNSUPPORTED`), ciclos mal planteados y errores.
4. `cge_do`: cirugía de grafo exacta de Pearl `do(V=v)` (mutilación de aristas entrantes incluso en
   estados intermedios), blindaje epistémico bajo mutilación, efecto causal nulo en actores muertos,
   auditoría de restricciones post-intervención, identidad `do({})` y validación de entradas.
5. `cge_solve`: optimización certificada (`MINLP`, `LP`, `QP` con `sense='min'`), grafos de 0 controles,
   metas inalcanzables con multiplicadores KKT, certificación Minimal IIS en dominios infactibles,
   resistencia anti-sicofancia (`ignore_constraints=True`) y Compuerta de Honestidad Epistémica.
6. `cge_prove`: demostración algebraica simbólica, cotas duales globales MILP (incluyendo implicaciones
   lógicas `P => Q` y funciones a trozos `Min`/`Piecewise`), aritmética de intervalos B&B en no lineales,
   búsqueda exacta de contraejemplos (`FALSIFIED`), inmunidad a verdad vacua (`INFEASIBLE_DOMAIN` con IIS),
   degradación honesta `SAMPLED_VERIFIED` ante cajas negras y auditoría completa del sistema.
7. Memoria epistémica indexada O(1) (`cache_hit`) e invalidación automática ante cambios en disco.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import threading
import time
from typing import Any, Dict

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.memory import create_connected_server_and_client_session
import pytest
from sse_starlette.sse import AppStatus
import sympy as sp
import uvicorn

from cge.mcp import (
    ActorDBOracle,
    cge_do,
    cge_prove,
    cge_solve,
    cge_trace,
    create_mcp_server,
)
from cge.mcp.prover import _eval_expr_interval, _iv_cosh, _iv_pow, _iv_sinh
from cge.mcp.server import _safe_redirect_stdout_to_stderr
from cge.schema import CausalGraph, Constraint, Edge, Node, SourceReference


REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_CASE_DIR = str(REPO_ROOT / "golden_case")


def _node(
    id: str,
    category: str,
    current_value: Any = 0.0,
    bounds: Dict[str, Any] | None = None,
    symbolic_expr: str | None = None,
    data_type: str = "continuous",
    fidelity: Dict[str, Any] | None = None,
) -> Node:
    return Node(
        id=id,
        label=id,
        category=category,
        data_type=data_type,
        source_reference=SourceReference(file="test_synthetic.py", entity=id, line_range=[1, 2]),
        current_value=current_value,
        bounds=bounds,
        symbolic_expr=symbolic_expr,
        fidelity=fidelity,
    )


def _edge(source: str, target: str, type: str = "exact_code") -> Edge:
    return Edge(source=source, target=target, type=type)


def _constr(id: str, expression: str, type: str = "hard") -> Constraint:
    return Constraint(id=id, type=type, expression=expression)


def _extract_mcp_json(call_result: Any) -> Dict[str, Any]:
    """Extrae y parsea el payload JSON devuelto por `ClientSession.call_tool`."""
    assert not call_result.isError, f"MCP tool returned error: {call_result}"
    if getattr(call_result, "structuredContent", None) and isinstance(call_result.structuredContent, dict):
        # FastMCP puede envolver en `result` o devolver directamente el diccionario
        sc = call_result.structuredContent
        if "result" in sc and isinstance(sc["result"], dict) and "status" in sc["result"]:
            return sc["result"]
        if "status" in sc:
            return sc
    assert call_result.content, "MCP tool returned empty content"
    text = call_result.content[0].text
    return json.loads(text)


# ==============================================================================
# 1. Protocolo MCP Estándar (Memoria + Subproceso `stdio` real)
# ==============================================================================


async def _shutdown_uvicorn_server(
    uv_srv: uvicorn.Server,
    serve_task: asyncio.Task[Any],
    sock: socket.socket,
) -> None:
    """Cierra `uvicorn.Server` con timeout estricto y resetea el singleton global `AppStatus.should_exit`."""
    uv_srv.should_exit = True
    AppStatus.should_exit = True
    try:
        await asyncio.wait_for(serve_task, timeout=5.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        uv_srv.force_exit = True
        serve_task.cancel()
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(serve_task, timeout=2.0)
    finally:
        sock.close()
        AppStatus.should_exit = False


class TestMCPProtocolAndTransport:
    @pytest.fixture(autouse=True)
    def _reset_sse_app_status(self):
        AppStatus.should_exit = False
        yield
        AppStatus.should_exit = False

    def test_in_memory_mcp_tool_discovery_and_invocation(self):
        server = create_mcp_server(name="test-actordb", default_source=GOLDEN_CASE_DIR)

        async def _scenario() -> None:
            async with create_connected_server_and_client_session(server) as session:
                tools_resp = await session.list_tools()
                tool_map = {t.name: t for t in tools_resp.tools}
                assert set(tool_map.keys()) == {"cge_trace", "cge_do", "cge_solve", "cge_prove"}

                for _t_name, tool in tool_map.items():
                    assert tool.description and len(tool.description) > 20
                    assert isinstance(tool.inputSchema, dict)
                    assert tool.inputSchema.get("type") == "object"

                # Invocar las 4 herramientas a través de sesiones MCP reales
                r_trace = _extract_mcp_json(await session.call_tool("cge_trace", {"target_id": "profit"}))
                assert r_trace["status"] == "OK"
                assert r_trace["zero_hallucination_verified"] is True
                assert "brand_awareness" in r_trace["dead_node_ids"]

                r_do = _extract_mcp_json(
                    await session.call_tool("cge_do", {"interventions": {"order_qty": 850}, "target_id": "profit"})
                )
                assert r_do["status"] == "OK"
                assert r_do["intervened_state"]["unit_cost"] == pytest.approx(1.0)

                r_solve = _extract_mcp_json(await session.call_tool("cge_solve", {"target_id": "profit"}))
                assert r_solve["status"] == "OPTIMAL"
                assert r_solve["certificate_status"] == "CERTIFIED_OPTIMAL"
                assert r_solve["controls"]["order_qty"] >= 800

                r_prove = _extract_mcp_json(
                    await session.call_tool("cge_prove", {"invariant_expr": "sales <= order_qty"})
                )
                assert r_prove["status"] == "PROVEN"
                assert r_prove["seal"] == "PROVEN"

        anyio.run(_scenario)

    def test_real_stdio_subprocess_transport_and_stdout_protection(self, tmp_path: Path):
        """Verifica que `python -m cge.mcp --transport stdio` funcione en un proceso real
        y que `print()` en el código analizado jamás corrompa el stream JSON-RPC en `stdout`.
        """
        noisy_module = tmp_path / "noisy_pricing.py"
        noisy_module.write_text(
            'print("NOISY IMPORT BANNER ON STDOUT")\n'
            "def run_noisy(price: float = 10.0, cost: float = 4.0) -> float:\n"
            '    print("NOISY EXECUTION PRINT")\n'
            "    assert price <= 25.0\n"
            "    margin = price - cost\n"
            "    revenue = margin * 100.0\n"
            "    return revenue\n",
            encoding="utf-8",
        )

        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "cge.mcp", "--source", str(noisy_module), "--transport", "stdio", "--no-auto-discovery"],
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

        async def _stdio_scenario() -> None:
            with anyio.fail_after(30.0):
                async with stdio_client(params) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        tools = await session.list_tools()
                        assert {t.name for t in tools.tools} == {"cge_trace", "cge_do", "cge_solve", "cge_prove"}

                        res_trace = _extract_mcp_json(await session.call_tool("cge_trace", {}))
                        assert res_trace["status"] == "OK"
                        assert res_trace["target_id"] == "run_noisy"

                        res_prove = _extract_mcp_json(
                            await session.call_tool("cge_prove", {"invariant_expr": "run_noisy <= 2500.0 - 100.0 * cost"})
                        )
                        assert res_prove["status"] == "PROVEN"
                        assert res_prove["seal"] == "PROVEN"

                        # Consultar también `golden_case` (que usa `logging.info` + CSV + multi-módulo) sobre el mismo canal stdio
                        res_gc = _extract_mcp_json(
                            await session.call_tool("cge_solve", {"target_id": "profit", "source_path": GOLDEN_CASE_DIR})
                        )
                        assert res_gc["status"] == "OPTIMAL"
                        assert res_gc["controls"]["order_qty"] >= 800

        anyio.run(_stdio_scenario)

    def test_real_streamable_http_loopback_transport(self):
        """Verifica el transporte de red MCP `streamable-http` sobre un socket TCP loopback real
        (`127.0.0.1:<puerto_efímero>`) con negociación de sesión HTTP, descubrimiento e invocación de las 4 herramientas.
        """
        async def _http_scenario() -> None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 0))
            sock.listen(16)
            port = int(sock.getsockname()[1])

            mcp_srv = create_mcp_server(
                name="http-actordb-test",
                default_source=GOLDEN_CASE_DIR,
                allow_auto_discovery=False,
                host="127.0.0.1",
                port=port,
            )
            assert mcp_srv.settings.host == "127.0.0.1"
            assert mcp_srv.settings.port == port

            uv_cfg = uvicorn.Config(
                mcp_srv.streamable_http_app(),
                host="127.0.0.1",
                port=port,
                log_level="error",
                timeout_graceful_shutdown=1,
            )
            uv_srv = uvicorn.Server(uv_cfg)
            serve_task = asyncio.create_task(uv_srv.serve(sockets=[sock]))
            try:
                deadline = time.monotonic() + 5.0
                while not uv_srv.started:
                    assert time.monotonic() < deadline, "Timed out waiting for uvicorn streamable-http server startup"
                    await asyncio.sleep(0.02)

                url = f"http://127.0.0.1:{port}/mcp"
                async with streamable_http_client(url) as (read_stream, write_stream, get_session_id):
                    async with ClientSession(read_stream, write_stream) as session:
                        init_res = await session.initialize()
                        assert init_res.serverInfo.name == "http-actordb-test"
                        assert get_session_id() is not None

                        tools_resp = await session.list_tools()
                        assert {t.name for t in tools_resp.tools} == {
                            "cge_trace",
                            "cge_do",
                            "cge_solve",
                            "cge_prove",
                        }

                        r_trace = _extract_mcp_json(
                            await session.call_tool("cge_trace", {"target_id": "profit", "source_id": "price"})
                        )
                        assert r_trace["status"] == "OK"
                        assert r_trace["source_trace"]["total_derivative"] == pytest.approx(485.0, rel=1e-4)

                        r_do = _extract_mcp_json(
                            await session.call_tool(
                                "cge_do",
                                {"interventions": {"order_qty": 850}, "target_id": "profit"},
                            )
                        )
                        assert r_do["status"] == "OK"
                        assert r_do["intervened_state"]["unit_cost"] == pytest.approx(1.0)

                        r_solve = _extract_mcp_json(
                            await session.call_tool("cge_solve", {"target_id": "profit"})
                        )
                        assert r_solve["status"] == "OPTIMAL"
                        assert r_solve["certificate_status"] == "CERTIFIED_OPTIMAL"
                        assert r_solve["controls"]["order_qty"] >= 800

                        # Prueba formal sobre grafo trigonométrico enviado como payload JSON sobre HTTP
                        g_trig = CausalGraph(system_id="http_trig")
                        g_trig.add_node(_node(id="x", category="control", current_value=0.5, bounds={"min": -2.0, "max": 2.0}))
                        g_trig.add_node(_node(id="y", category="target", current_value=0.0, symbolic_expr="sin(x) + tanh(x)"))
                        g_trig.add_edge(_edge(source="x", target="y"))
                        r_prove = _extract_mcp_json(
                            await session.call_tool(
                                "cge_prove",
                                {"invariant_expr": "y <= 2.0 and y >= -2.0", "graph_data": g_trig.to_dict()},
                            )
                        )
                        assert r_prove["status"] == "PROVEN"
                        assert r_prove["seal"] == "PROVEN"
            finally:
                await _shutdown_uvicorn_server(uv_srv, serve_task, sock)

        asyncio.run(asyncio.wait_for(_http_scenario(), timeout=30.0))

    def test_concurrent_streamable_http_and_stdout_thread_safety(self):
        """Verifica sesiones concurrentes (`asyncio.gather`) sobre `streamable-http` sin bloqueo del event loop
        y seguridad multi-hilo de la redirección de `stdout -> stderr`.
        """
        orig_stdout = sys.stdout
        barrier = threading.Barrier(8)

        def _redirect_worker() -> None:
            barrier.wait()
            for _ in range(40):
                with _safe_redirect_stdout_to_stderr():
                    assert sys.stdout is sys.stderr

        workers = [threading.Thread(target=_redirect_worker) for _ in range(8)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        assert sys.stdout is orig_stdout

        async def _concurrent_http_scenario() -> None:
            sock_http = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock_http.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock_http.bind(("127.0.0.1", 0))
            sock_http.listen(16)
            port_http = int(sock_http.getsockname()[1])

            srv_http = create_mcp_server(
                name="concurrent-http-actordb",
                default_source=GOLDEN_CASE_DIR,
                allow_auto_discovery=False,
                host="127.0.0.1",
                port=port_http,
            )
            uv_http = uvicorn.Server(
                uvicorn.Config(
                    srv_http.streamable_http_app(),
                    host="127.0.0.1",
                    port=port_http,
                    log_level="error",
                    timeout_graceful_shutdown=1,
                )
            )
            task_http = asyncio.create_task(uv_http.serve(sockets=[sock_http]))
            try:
                deadline = time.monotonic() + 5.0
                while not uv_http.started:
                    assert time.monotonic() < deadline
                    await asyncio.sleep(0.02)

                url_http = f"http://127.0.0.1:{port_http}/mcp"

                async def _call_one(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
                    async with streamable_http_client(url_http) as (r_s, w_s, _):
                        async with ClientSession(r_s, w_s) as sess:
                            await sess.initialize()
                            return _extract_mcp_json(await sess.call_tool(tool_name, args))

                r_tr, r_do, r_sol, r_pr = await asyncio.gather(
                    _call_one("cge_trace", {"target_id": "profit", "source_id": "price"}),
                    _call_one("cge_do", {"interventions": {"order_qty": 850}, "target_id": "profit"}),
                    _call_one("cge_solve", {"target_id": "profit"}),
                    _call_one("cge_prove", {"invariant_expr": "order_qty >= 800 => unit_cost == 1.0"}),
                )
                assert r_tr["status"] == "OK"
                assert r_do["status"] == "OK" and r_do["intervened_state"]["unit_cost"] == pytest.approx(1.0)
                assert r_sol["status"] == "OPTIMAL"
                assert r_pr["status"] == "PROVEN"
            finally:
                await _shutdown_uvicorn_server(uv_http, task_http, sock_http)

        asyncio.run(asyncio.wait_for(_concurrent_http_scenario(), timeout=30.0))

    def test_real_sse_loopback_transport(self):
        """Verifica el transporte de red MCP `sse` (`Server-Sent Events`) sobre un socket TCP loopback real."""
        async def _sse_scenario() -> None:
            sock_sse = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock_sse.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock_sse.bind(("127.0.0.1", 0))
            sock_sse.listen(16)
            port_sse = int(sock_sse.getsockname()[1])

            srv_sse = create_mcp_server(
                name="sse-actordb-test",
                default_source=GOLDEN_CASE_DIR,
                allow_auto_discovery=False,
                host="127.0.0.1",
                port=port_sse,
            )
            uv_sse = uvicorn.Server(
                uvicorn.Config(
                    srv_sse.sse_app(),
                    host="127.0.0.1",
                    port=port_sse,
                    log_level="error",
                    timeout_graceful_shutdown=1,
                )
            )
            task_sse = asyncio.create_task(uv_sse.serve(sockets=[sock_sse]))
            try:
                deadline = time.monotonic() + 5.0
                while not uv_sse.started:
                    assert time.monotonic() < deadline
                    await asyncio.sleep(0.02)

                async with sse_client(f"http://127.0.0.1:{port_sse}/sse") as (r_sse, w_sse):
                    async with ClientSession(r_sse, w_sse) as sess_sse:
                        init_sse = await sess_sse.initialize()
                        assert init_sse.serverInfo.name == "sse-actordb-test"
                        tools_sse = await sess_sse.list_tools()
                        assert {t.name for t in tools_sse.tools} == {
                            "cge_trace",
                            "cge_do",
                            "cge_solve",
                            "cge_prove",
                        }
                        res_sse = _extract_mcp_json(
                            await sess_sse.call_tool(
                                "cge_prove",
                                {"invariant_expr": "order_qty >= 800 => unit_cost == 1.0"},
                            )
                        )
                        assert res_sse["status"] == "PROVEN"
            finally:
                await _shutdown_uvicorn_server(uv_sse, task_sse, sock_sse)

        asyncio.run(asyncio.wait_for(_sse_scenario(), timeout=30.0))


# ==============================================================================
# 2. Pruebas de `cge_trace` (Linaje, Derivadas y Nodos Muertos T1-A/B/C)
# ==============================================================================


class TestCgeTrace:
    def test_golden_case_lineage_derivatives_and_t1a_orphan(self):
        oracle = ActorDBOracle(default_source=GOLDEN_CASE_DIR, allow_auto_discovery=False)
        res = oracle.trace(target_id="profit", source_id="price")

        assert res["status"] == "OK"
        assert res["seal"] == "PROVEN"
        assert res["target_id"] == "profit"
        assert "brand_awareness" in res["dead_node_ids"]
        dead_by_id = {d["node"]: d for d in res["dead_nodes"]}
        assert dead_by_id["brand_awareness"]["subtype"] == "T1-A_STRUCTURAL_ORPHAN"

        # En el estado nominal de golden_case: price=3.5, marketing=500, order_qty=500, base_demand=1000
        # demand = 1103.8854 > order_qty (500) => sales = 500, revenue = 500*price, fees = 0.03*revenue
        # d(profit)/d(price) = (1 - 0.03) * 500 = 485.0
        st = res["source_trace"]
        assert st["has_directed_path"] is True
        assert st["is_causally_dead_to_target"] is False
        assert st["total_derivative"] == pytest.approx(485.0, rel=1e-4)

    def test_detects_t1b_algebraic_cancellation_and_t1c_disconnected_control(self):
        g = CausalGraph(system_id="cancellation_sys")
        g.add_node(_node(id="u_live", category="control", current_value=2.0, bounds={"min": 0.0, "max": 10.0}))
        g.add_node(_node(id="u_dead", category="control", current_value=5.0, bounds={"min": 0.0, "max": 10.0}))
        g.add_node(_node(id="s_cancel", category="state", current_value=6.0, symbolic_expr="3.0 * u_live"))
        g.add_node(_node(id="s_plus", category="state", current_value=8.0, symbolic_expr="u_live + s_cancel"))
        # s_cancel se suma en s_plus y se resta exactamente en profit => derivada total respecto a s_cancel == 0
        g.add_node(_node(id="profit", category="target", current_value=2.0, symbolic_expr="s_plus - s_cancel"))
        g.add_edge(_edge(source="u_live", target="s_cancel"))
        g.add_edge(_edge(source="u_live", target="s_plus"))
        g.add_edge(_edge(source="s_cancel", target="s_plus"))
        g.add_edge(_edge(source="s_plus", target="profit"))
        g.add_edge(_edge(source="s_cancel", target="profit"))

        res = cge_trace(target_id="profit", source_id="s_cancel", graph_data=g.to_dict())
        assert res["status"] == "OK"
        dead_map = {d["node"]: d["subtype"] for d in res["dead_nodes"]}
        assert dead_map["s_cancel"] == "T1-B_ALGEBRAIC_PATH_CANCELLATION"
        assert dead_map["u_dead"] == "T1-C_DISCONNECTED_CONTROL"
        assert res["source_trace"]["has_directed_path"] is True
        assert res["source_trace"]["total_derivative"] == 0.0
        assert res["source_trace"]["is_causally_dead_to_target"] is True

    def test_state_guarding_hard_constraint_is_not_flagged_as_dead(self):
        g = CausalGraph(system_id="guard_sys")
        g.add_node(_node(id="x", category="control", current_value=4.0, bounds={"min": 0.0, "max": 10.0}))
        g.add_node(_node(id="safety_temp", category="state", current_value=80.0, symbolic_expr="20.0 * x"))
        g.add_node(_node(id="output", category="target", current_value=12.0, symbolic_expr="3.0 * x"))
        g.add_edge(_edge(source="x", target="safety_temp"))
        g.add_edge(_edge(source="x", target="output"))
        g.add_constraint(_constr(id="c_temp", expression="safety_temp <= 150.0", type="hard"))

        res = cge_trace(target_id="output", graph_data=g.to_dict())
        assert res["status"] == "OK"
        assert "safety_temp" not in res["dead_node_ids"]

    def test_empirical_seal_and_cycle_and_unknown_actor_errors(self):
        # 1. Arista puramente empírica degrada honestamente el sello a SAMPLED_VERIFIED
        g_emp = CausalGraph(system_id="emp_sys")
        g_emp.add_node(_node(id="x", category="control", current_value=1.0))
        g_emp.add_node(_node(id="y", category="target", current_value=2.0, symbolic_expr="2.0 * x"))
        g_emp.add_edge(_edge(source="x", target="y", type="empirical_data"))
        r_emp = cge_trace(target_id="y", graph_data=g_emp.to_dict())
        assert r_emp["seal"] == "SAMPLED_VERIFIED"

        # 2. Ciclo estático mal planteado -> ILL_POSED_CYCLE
        g_cyc = CausalGraph(system_id="cyc_sys")
        g_cyc.add_node(_node(id="a", category="state", symbolic_expr="b + 1"))
        g_cyc.add_node(_node(id="b", category="target", symbolic_expr="a + 1"))
        g_cyc.add_edge(_edge(source="a", target="b"))
        g_cyc.add_edge(_edge(source="b", target="a"))
        r_cyc = cge_trace(target_id="b", graph_data=g_cyc.to_dict())
        assert r_cyc["status"] == "ILL_POSED_CYCLE"
        assert len(r_cyc["cycles"]) >= 1

        # 3. Actor inexistente -> ERROR UNKNOWN_ACTOR
        r_unk = cge_trace(target_id="ghost_metric", source_path=GOLDEN_CASE_DIR)
        assert r_unk["status"] == "ERROR"
        assert r_unk["error_code"] == "UNKNOWN_ACTOR"

    def test_blackbox_callable_finite_difference_seal_and_kary_minmax(self):
        g_bb = CausalGraph(system_id="bb_trace")
        g_bb.add_node(_node(id="x", category="control", current_value=2.0, bounds={"min": 0.0, "max": 10.0}))
        g_bb.add_node(_node(id="mid", category="state", current_value=6.0, symbolic_expr=None))
        g_bb.add_node(_node(id="obj", category="target", current_value=12.0, symbolic_expr="2.0 * mid"))
        g_bb.add_edge(_edge(source="x", target="mid"))
        g_bb.add_edge(_edge(source="mid", target="obj"))
        g_bb._blackbox_callables = {"mid": (["x"], lambda x: 3.0 * x)}

        oracle = ActorDBOracle(allow_auto_discovery=False)
        res = oracle.trace(target_id="obj", source_id="x", graph_data=g_bb)
        assert res["status"] == "OK"
        assert res["seal"] == "SAMPLED_VERIFIED"
        assert "x" not in res["dead_node_ids"]
        assert "mid" not in res["dead_node_ids"]
        assert res["source_trace"]["total_derivative"] == pytest.approx(6.0, rel=1e-4)
        assert res["control_ranking"][0]["is_dead"] is False

        # k-ary Min(a, b, c) con 3 argumentos calcula subgradiente exacto sin error
        g_k = CausalGraph(system_id="kary_min_trace")
        g_k.add_node(_node(id="a", category="control", current_value=2.0, bounds={"min": 0.0, "max": 10.0}))
        g_k.add_node(_node(id="b", category="control", current_value=5.0, bounds={"min": 0.0, "max": 10.0}))
        g_k.add_node(_node(id="c", category="control", current_value=8.0, bounds={"min": 0.0, "max": 10.0}))
        g_k.add_node(_node(id="bottleneck", category="target", current_value=2.0, symbolic_expr="Min(a, b, c)"))
        for src_id in ("a", "b", "c"):
            g_k.add_edge(_edge(source=src_id, target="bottleneck"))
        rk = oracle.trace(target_id="bottleneck", source_id="a", graph_data=g_k)
        assert rk["status"] == "OK"
        assert rk["source_trace"]["total_derivative"] == pytest.approx(1.0)


# ==============================================================================
# 3. Pruebas de `cge_do` (Cirugía de Grafo Exacta de Pearl)
# ==============================================================================


class TestCgeDo:
    def test_intermediate_state_pearlian_graph_surgery(self):
        """`do(demand=400.0)` debe cortar las aristas entrantes hacia `demand`,
        fijar `demand=400.0` (ignorando la ecuación estructural de `demand`), propagar aguas abajo
        (`sales=min(400, 500)=400`) y dejar intactos los nodos aguas arriba (`price`, `marketing_budget`).
        """
        oracle = ActorDBOracle(default_source=GOLDEN_CASE_DIR, allow_auto_discovery=False)
        res = oracle.do(interventions={"demand": 400.0}, target_id="profit")

        assert res["status"] == "OK"
        assert res["seal"] == "PROVEN"
        mutilated_pairs = {(e["source"], e["target"]) for e in res["mutilated_edges"]}
        assert ("base_demand", "demand") in mutilated_pairs
        assert ("price", "demand") in mutilated_pairs
        assert ("marketing_budget", "demand") in mutilated_pairs

        # Aguas arriba inalterado
        assert res["intervened_state"]["price"] == pytest.approx(3.5)
        assert res["intervened_state"]["marketing_budget"] == pytest.approx(500.0)
        # Estado mutilado fijo y propagación aguas abajo exacta
        assert res["intervened_state"]["demand"] == pytest.approx(400.0)
        assert res["intervened_state"]["sales"] == pytest.approx(400.0)
        assert res["intervened_state"]["holding_cost"] == pytest.approx(20.0)
        # revenue = 3.5 * 400 = 1400; fees = 42; goods = 500 * 1.2 = 600; mkt = 500; holding = 20 => profit = 238.0
        assert res["intervened_target_value"] == pytest.approx(238.0, abs=1e-4)

    def test_epistemic_shield_when_mutilation_severs_unverified_upstream(self):
        """Si un nodo aguas arriba `noisy` no está verificado, `do({})` debe rechazar con `UNSUPPORTED`,
        pero `do(clean_mid=10.0)` corta la arista `noisy -> clean_mid` y aísla causalmente el objetivo,
        permitiendo una evaluación exacta `PROVEN`.
        """
        g = CausalGraph(system_id="mutilation_shield")
        g.add_node(
            _node(
                id="noisy_rng",
                category="state",
                current_value=1.0,
                fidelity={"verified": False, "reason": "Uses random.gauss"},
            )
        )
        g.add_node(_node(id="clean_mid", category="state", current_value=2.0, symbolic_expr="2.0 * noisy_rng"))
        g.add_node(_node(id="target_y", category="target", current_value=6.0, symbolic_expr="3.0 * clean_mid"))
        g.add_edge(_edge(source="noisy_rng", target="clean_mid"))
        g.add_edge(_edge(source="clean_mid", target="target_y"))

        r_blocked = cge_do(interventions={}, target_id="target_y", graph_data=g.to_dict())
        assert r_blocked["status"] == "UNSUPPORTED"
        assert "noisy_rng" in r_blocked["unverified_nodes"]

        r_shielded = cge_do(interventions={"clean_mid": 10.0}, target_id="target_y", graph_data=g.to_dict())
        assert r_shielded["status"] == "OK"
        assert r_shielded["seal"] == "PROVEN"
        assert r_shielded["intervened_target_value"] == pytest.approx(30.0)

    def test_dead_actor_intervention_and_post_intervention_constraint_violation(self):
        oracle = ActorDBOracle(default_source=GOLDEN_CASE_DIR, allow_auto_discovery=False)

        # Intervenir sobre actor muerto produce efecto causal idénticamente 0.0
        r_dead = oracle.do(interventions={"brand_awareness": 99999.0}, target_id="profit")
        assert r_dead["status"] == "OK"
        assert r_dead["causal_effect"] == 0.0
        assert "brand_awareness" in r_dead["intervened_on_disconnected_actors"]

        # Intervención que viola el presupuesto duro c_budget <= 2500 (1200 * 1.0 + 1500 = 2700 > 2500)
        r_infeas = oracle.do(
            interventions={"order_qty": 1200, "marketing_budget": 1500.0},
            target_id="profit",
        )
        assert r_infeas["status"] == "OK"
        assert r_infeas["is_feasible"] is False
        assert "c_budget" in r_infeas["violated_constraints"]

        # Identidad do({}) => efecto causal 0.0
        r_id = oracle.do(interventions={}, target_id="profit")
        assert r_id["status"] == "OK"
        assert r_id["causal_effect"] == 0.0

    def test_do_rejects_nan_inf_unknown_actor_and_bad_types(self):
        oracle = ActorDBOracle(default_source=GOLDEN_CASE_DIR, allow_auto_discovery=False)

        r_nan = oracle.do(interventions={"price": float("nan")})
        assert r_nan["status"] == "ERROR"
        assert r_nan["error_code"] == "NON_FINITE_VALUE"

        r_inf = oracle.do(interventions={"price": float("inf")})
        assert r_inf["status"] == "ERROR"
        assert r_inf["error_code"] == "NON_FINITE_VALUE"

        r_str = oracle.do(interventions={"price": "expensive"})  # type: ignore[dict-item]
        assert r_str["status"] == "ERROR"
        assert r_str["error_code"] == "INVALID_VALUE_TYPE"

        r_unk = oracle.do(interventions={"nonexistent_lever": 1.0})
        assert r_unk["status"] == "ERROR"
        assert r_unk["error_code"] == "UNKNOWN_ACTOR"

    def test_enforces_logical_implication_disjunction_and_singularity_guard(self):
        oracle = ActorDBOracle(allow_auto_discovery=False)
        g = CausalGraph(system_id="kyc_policy")
        g.add_node(_node(id="payout_sent", category="control", data_type="boolean", current_value=0, bounds={"min": 0, "max": 1}))
        g.add_node(_node(id="is_kyc_verified", category="control", data_type="boolean", current_value=0, bounds={"min": 0, "max": 1}))
        g.add_node(_node(id="risk_flag", category="control", data_type="boolean", current_value=0, bounds={"min": 0, "max": 1}))
        g.add_node(_node(id="score", category="target", current_value=0.0, symbolic_expr="10.0 * payout_sent"))
        g.add_edge(_edge(source="payout_sent", target="score"))
        g.add_constraint(_constr(id="c_kyc", expression="payout_sent >= 1 => is_kyc_verified >= 1"))
        g.add_constraint(_constr(id="c_risk", expression="not (payout_sent >= 1 and risk_flag >= 1)"))

        r_bad_kyc = oracle.do(
            interventions={"payout_sent": True, "is_kyc_verified": False, "risk_flag": False},
            target_id="score",
            graph_data=g,
        )
        assert r_bad_kyc["status"] == "OK"
        assert r_bad_kyc["is_feasible"] is False
        assert "c_kyc" in r_bad_kyc["violated_constraints"]

        r_bad_risk = oracle.do(
            interventions={"payout_sent": True, "is_kyc_verified": True, "risk_flag": True},
            target_id="score",
            graph_data=g,
        )
        assert r_bad_risk["is_feasible"] is False
        assert "c_risk" in r_bad_risk["violated_constraints"]

        r_ok = oracle.do(
            interventions={"payout_sent": True, "is_kyc_verified": True, "risk_flag": False},
            target_id="score",
            graph_data=g,
        )
        assert r_ok["is_feasible"] is True
        assert r_ok["violated_constraints"] == []

        # División por cero directa en do(x=0.0) retorna error estructurado sin excepción no capturada
        g_div = CausalGraph(system_id="div_do")
        g_div.add_node(_node(id="x", category="control", current_value=2.0, bounds={"min": -5.0, "max": 5.0}))
        g_div.add_node(_node(id="inv_x", category="target", current_value=0.5, symbolic_expr="1.0 / x"))
        g_div.add_edge(_edge(source="x", target="inv_x"))
        r_pole = oracle.do(interventions={"x": 0.0}, target_id="inv_x", graph_data=g_div)
        assert r_pole["status"] == "ERROR"
        assert r_pole["error_code"] == "DOMAIN_SINGULARITY"


# ==============================================================================
# 4. Pruebas de `cge_solve` (Optimización, IIS, KKT y Resistencia Anti-Sicofancia)
# ==============================================================================


class TestCgeSolve:
    def test_golden_case_certified_optimal_and_pressure_resistance(self):
        oracle = ActorDBOracle(default_source=GOLDEN_CASE_DIR, allow_auto_discovery=False)

        sol = oracle.solve(target_id="profit", sense="max")
        assert sol["status"] == "OPTIMAL"
        assert sol["certificate_status"] == "CERTIFIED_OPTIMAL"
        assert sol["seal"] == "PROVEN"
        assert sol["controls"]["order_qty"] >= 800
        assert sol["objective_value"] > 2300.0
        assert sol["pressure_resisted"] is False

        # Presión de bypass de restricciones (GAP-03): jamás viola c_budget <= 2500
        resisted = oracle.solve(target_id="profit", sense="max", ignore_constraints=True)
        assert resisted["status"] == "REJECTED_CONSTRAINT_BYPASS"
        assert resisted["pressure_resisted"] is True
        assert resisted["objective_value"] == pytest.approx(sol["objective_value"], abs=1e-3)
        total_outlay = (
            resisted["evaluated_nodes"]["unit_cost"] * resisted["controls"]["order_qty"]
            + resisted["controls"]["marketing_budget"]
        )
        assert total_outlay <= 2500.0 + 1e-4

    def test_minimization_sense_and_zero_control_graphs(self):
        # 1. QP convexo con sense="min": min (u - 3)^2 + 5 en u in [0, 10] => u*=3, J*=5
        g_qp = CausalGraph(system_id="qp_min")
        g_qp.add_node(_node(id="u", category="control", current_value=8.0, bounds={"min": 0.0, "max": 10.0}))
        g_qp.add_node(_node(id="loss", category="target", current_value=30.0, symbolic_expr="(u - 3)**2 + 5"))
        g_qp.add_edge(_edge(source="u", target="loss"))

        r_min = cge_solve(target_id="loss", sense="min", graph_data=g_qp.to_dict())
        assert r_min["status"] == "OPTIMAL"
        assert r_min["controls"]["u"] == pytest.approx(3.0, abs=1e-3)
        assert r_min["objective_value"] == pytest.approx(5.0, abs=1e-3)
        assert r_min["objective_improvement"] == pytest.approx(25.0, abs=1e-3)

        # 2. Grafo sin controles (0 grados de libertad) factible e infactible
        g_zero = CausalGraph(system_id="zero_ctrl")
        g_zero.add_node(_node(id="exo", category="exogenous", current_value=10.0))
        g_zero.add_node(_node(id="metric", category="target", current_value=20.0, symbolic_expr="2.0 * exo"))
        g_zero.add_edge(_edge(source="exo", target="metric"))
        g_zero.add_constraint(_constr(id="c_cap", expression="metric <= 25.0", type="hard"))

        r_z_ok = cge_solve(target_id="metric", graph_data=g_zero.to_dict())
        assert r_z_ok["status"] == "OPTIMAL"
        assert r_z_ok["objective_value"] == pytest.approx(20.0)

        r_z_inf = cge_solve(target_id="metric", exo_overrides={"exo": 15.0}, graph_data=g_zero.to_dict())
        assert r_z_inf["status"] == "INFEASIBLE"
        assert "c_cap" in r_z_inf["iis_constraints"]

    def test_infeasible_goal_kkt_and_minimal_iis_and_epistemic_gate(self):
        oracle = ActorDBOracle(default_source=GOLDEN_CASE_DIR, allow_auto_discovery=False)

        # 1. Meta cuantitativa imposible ($50,000) -> INFEASIBLE con brecha exacta y multiplicadores sombra KKT
        r_goal = oracle.solve(target_id="profit", target_value=50000.0)
        assert r_goal["status"] == "INFEASIBLE"
        max_ach = r_goal.get("max_achievable_profit") or r_goal.get("max_achievable_objective")
        assert max_ach is not None and max_ach < 50000.0
        assert len(r_goal["iis_constraints"]) >= 1

        # 2. Dominio contradictorio con restricciones señuelo -> Minimal IIS exacto {"c1_pair", "c2_shared"}
        src_iis = """
def sys_iis(x: float = 5.0, y: float = 5.0) -> float:
    assert 0.0 <= x <= 20.0
    assert 0.0 <= y <= 20.0
    assert x + y <= 10.0, "c1_pair"
    assert x >= 6.0, "c2_shared"
    assert y >= 6.0, "c3_decoy_sym"
    assert x <= 19.0, "c4_decoy"
    z = x + y
    return z
"""
        r_src_iis = cge_solve(source_path=src_iis)
        assert r_src_iis["status"] == "INFEASIBLE"
        assert len(r_src_iis["iis_constraints"]) == 3

        g_iis = CausalGraph(system_id="iis_test")
        g_iis.add_node(_node(id="x", category="control", current_value=5.0, bounds={"min": 0.0, "max": 20.0}))
        g_iis.add_node(_node(id="y", category="control", current_value=5.0, bounds={"min": 0.0, "max": 20.0}))
        g_iis.add_node(_node(id="z", category="target", current_value=10.0, symbolic_expr="x + y"))
        g_iis.add_edge(_edge(source="x", target="z"))
        g_iis.add_edge(_edge(source="y", target="z"))
        g_iis.add_constraint(_constr(id="c1_pair", expression="x + y <= 10.0", type="hard"))
        g_iis.add_constraint(_constr(id="c2_up_x", expression="x >= 6.0", type="hard"))
        g_iis.add_constraint(_constr(id="c3_up_y", expression="y >= 6.0", type="hard"))
        g_iis.add_constraint(_constr(id="c4_decoy", expression="x <= 19.0", type="hard"))

        r_iis = cge_solve(target_id="z", graph_data=g_iis.to_dict())
        assert r_iis["status"] == "INFEASIBLE"
        assert set(r_iis["iis_constraints"]) == {"c1_pair", "c2_up_x", "c3_up_y"}
        assert "c4_decoy" not in r_iis["iis_constraints"]

        # 3. Compuerta de Honestidad Epistémica ante RNG -> UNSUPPORTED
        rng_code = (
            "import random\n"
            "def rng_model(u: float = 2.0) -> float:\n"
            "    noise = random.random()\n"
            "    out = u + noise\n"
            "    return out\n"
        )
        r_rng = oracle.solve(source_path=rng_code)
        assert r_rng["status"] == "UNSUPPORTED"
        assert r_rng["seal"] == "UNSUPPORTED"

        # 4. Parámetros inválidos
        assert oracle.solve(sense="sideways")["error_code"] == "INVALID_SENSE"
        assert oracle.solve(fixed_controls={"demand": 100.0})["error_code"] == "INVALID_FIXED_CONTROL"
        assert oracle.solve(fixed_controls={"price": float("nan")})["error_code"] == "NON_FINITE_VALUE"

    def test_solve_enforces_propositional_not_or_and_implication_constraints(self):
        # 1. Restricción disyuntiva negada `not (x > 5 and y > 5)` <=> `x <= 5 or y <= 5`:
        # max x + y en [0, 10]^2 debe dar exactamente J* = 15.0 (jamás 20.0 ignorando `not (...)`!)
        g_prop = CausalGraph(system_id="solve_prop")
        g_prop.add_node(_node(id="x", category="control", current_value=1.0, bounds={"min": 0.0, "max": 10.0}))
        g_prop.add_node(_node(id="y", category="control", current_value=1.0, bounds={"min": 0.0, "max": 10.0}))
        g_prop.add_node(_node(id="obj", category="target", current_value=2.0, symbolic_expr="x + y"))
        g_prop.add_edge(_edge(source="x", target="obj"))
        g_prop.add_edge(_edge(source="y", target="obj"))
        g_prop.add_constraint(_constr(id="c_demorgan", expression="not (x > 5.0 and y > 5.0)"))

        r_s = cge_solve(target_id="obj", sense="max", graph_data=g_prop.to_dict())
        assert r_s["status"] == "OPTIMAL"
        assert r_s["objective_value"] == pytest.approx(15.0, abs=1e-3)
        assert min(r_s["controls"]["x"], r_s["controls"]["y"]) <= 5.0 + 1e-4

        # 2. Restricción implicacional binaria `payout >= 1 => kyc >= 1`
        g_imp = CausalGraph(system_id="solve_imp")
        g_imp.add_node(_node(id="payout", category="control", data_type="boolean", current_value=0, bounds={"min": 0, "max": 1}))
        g_imp.add_node(_node(id="kyc", category="control", data_type="boolean", current_value=0, bounds={"min": 0, "max": 1}))
        g_imp.add_node(_node(id="net", category="target", current_value=0.0, symbolic_expr="10.0 * payout - 2.0 * kyc"))
        g_imp.add_edge(_edge(source="payout", target="net"))
        g_imp.add_edge(_edge(source="kyc", target="net"))
        g_imp.add_constraint(_constr(id="c_kyc", expression="payout >= 1 => kyc >= 1"))

        r_imp = cge_solve(target_id="net", sense="max", graph_data=g_imp.to_dict())
        assert r_imp["status"] == "OPTIMAL"
        assert r_imp["certificate_status"] == "CERTIFIED_OPTIMAL"
        assert r_imp["controls"]["payout"] == 1
        assert r_imp["controls"]["kyc"] == 1
        assert r_imp["objective_value"] == pytest.approx(8.0, abs=1e-3)

        # 3. Restricción mixta continua-booleana `amount >= 100.0 => kyc >= 1` y bicondicional en IIS
        g_mix = CausalGraph(system_id="solve_mix_imp")
        g_mix.add_node(_node(id="amount", category="control", current_value=0.0, bounds={"min": 0.0, "max": 500.0}))
        g_mix.add_node(_node(id="kyc", category="control", data_type="boolean", current_value=0, bounds={"min": 0, "max": 1}))
        g_mix.add_node(_node(id="profit", category="target", current_value=0.0, symbolic_expr="amount - 20.0 * kyc"))
        g_mix.add_edge(_edge(source="amount", target="profit"))
        g_mix.add_edge(_edge(source="kyc", target="profit"))
        g_mix.add_constraint(_constr(id="c_aml", expression="amount >= 100.0 => kyc >= 1"))

        r_mix = cge_solve(target_id="profit", sense="max", graph_data=g_mix.to_dict())
        assert r_mix["status"] == "OPTIMAL"
        assert r_mix["certificate_status"] == "CERTIFIED_OPTIMAL"
        assert r_mix["controls"]["amount"] == pytest.approx(500.0, abs=1e-3)
        assert r_mix["controls"]["kyc"] == 1
        assert r_mix["objective_value"] == pytest.approx(480.0, abs=1e-3)

        # Bicondicional multicláusula preserva ID exacto `c_bicond` en IIS ante controles contradictorios
        g_bicond = CausalGraph(system_id="solve_bicond_iis")
        g_bicond.add_node(_node(id="payout", category="control", data_type="boolean", current_value=0, bounds={"min": 0, "max": 1}))
        g_bicond.add_node(_node(id="kyc", category="control", data_type="boolean", current_value=0, bounds={"min": 0, "max": 1}))
        g_bicond.add_node(_node(id="net", category="target", current_value=0.0, symbolic_expr="payout + kyc"))
        g_bicond.add_edge(_edge(source="payout", target="net"))
        g_bicond.add_edge(_edge(source="kyc", target="net"))
        g_bicond.add_constraint(_constr(id="c_bicond", expression="payout >= 1 <=> kyc >= 1"))

        r_bi_inf = cge_solve(
            target_id="net",
            sense="max",
            fixed_controls={"payout": 1, "kyc": 0},
            graph_data=g_bicond.to_dict(),
        )
        assert r_bi_inf["status"] == "INFEASIBLE"
        assert "c_bicond" in r_bi_inf["iis_constraints"]
        assert all("__cnf_" not in cid for cid in r_bi_inf["iis_constraints"])


# ==============================================================================
# 5. Pruebas de `cge_prove` (Verificación Formal, Contraejemplos e IIS)
# ==============================================================================


class TestCgeProve:
    def test_symbolic_milp_and_implication_proofs_on_golden_case(self):
        oracle = ActorDBOracle(default_source=GOLDEN_CASE_DIR, allow_auto_discovery=False)

        # 1. Identidad algebraica exacta: fees == 0.03 * revenue
        p_id = oracle.prove(invariant_expr="fees == 0.03 * revenue")
        assert p_id["status"] == "PROVEN"
        assert p_id["seal"] == "PROVEN"
        assert p_id["proof_method"] == "symbolic_algebraic_identity"

        # 2. Cota global MILP sobre Min(demand, order_qty): sales <= order_qty
        p_sales = oracle.prove(invariant_expr="sales <= order_qty")
        assert p_sales["status"] == "PROVEN"
        assert p_sales["seal"] == "PROVEN"

        # 3. Implicación acoplada con c_budget (order_qty >= 800 => unit_cost == 1.0 y order_qty + mkt <= 2500 => mkt <= 1700)
        p_mkt = oracle.prove(
            invariant_expr="order_qty >= 800 => marketing_budget <= 1700.0",
            assume_system_constraints=True,
        )
        assert p_mkt["status"] == "PROVEN"
        assert p_mkt["seal"] == "PROVEN"

        # 4. Implicación lógica condicional sobre escalón entero Piecewise: order_qty >= 800 => unit_cost <= 1.0
        p_imp = oracle.prove(invariant_expr="order_qty >= 800 => unit_cost <= 1.0")
        assert p_imp["status"] == "PROVEN"
        assert p_imp["seal"] == "PROVEN"

    def test_nonlinear_interval_arithmetic_proof(self):
        # 1. En golden_case con base_demand acotada en 3-sigma [700, 1300]:
        # demand = Max(0, base_demand*(1.45 - 0.15*price) + 8*sqrt(marketing_budget))
        # Mínimo global en base_demand=700, price=6.0, mkt=0 => demand = 700 * 0.55 = 385.0 >= 380.0
        oracle = ActorDBOracle(default_source=GOLDEN_CASE_DIR, allow_auto_discovery=False)
        p_dem = oracle.prove(
            invariant_expr="demand >= 380.0",
            domain_bounds={"base_demand": [700.0, 1300.0]},
        )
        assert p_dem["status"] == "PROVEN"
        assert p_dem["seal"] == "PROVEN"
        assert p_dem["proof_method"] == "interval_arithmetic_enclosure"

        # Sin acotar base_demand (su min en código es 0.0), demand >= 40.0 se falsifica honestamente en base_demand=0, mkt=0
        p_dem_raw = oracle.prove(invariant_expr="demand >= 40.0")
        assert p_dem_raw["status"] == "FALSIFIED"

        # 2. Polinomio cuadrático convexo no monótono: y = x**2 - 2*x + 2 tiene mínimo global 1.0 en x=1 dentro de [-2, 4]
        g_quad = CausalGraph(system_id="quad_bnb")
        g_quad.add_node(_node(id="x", category="control", current_value=0.0, bounds={"min": -2.0, "max": 4.0}))
        g_quad.add_node(_node(id="y", category="target", current_value=2.0, symbolic_expr="x**2 - 2*x + 2"))
        g_quad.add_edge(_edge(source="x", target="y"))

        p_quad = cge_prove(invariant_expr="y >= 0.9", graph_data=g_quad.to_dict())
        assert p_quad["status"] == "PROVEN"
        assert p_quad["seal"] == "PROVEN"
        assert p_quad["proof_method"] == "interval_arithmetic_enclosure"

    def test_trigonometric_and_hyperbolic_outward_interval_bounds_and_proofs(self):
        # 1. Auditoría directa de inclusión garantizada outward-rounded en `_eval_expr_interval`
        x_sym = sp.Symbol("x", real=True)
        fn_specs = [
            (sp.sin(x_sym), math.sin),
            (sp.cos(x_sym), math.cos),
            (sp.tanh(x_sym), math.tanh),
            (sp.sinh(x_sym), math.sinh),
            (sp.cosh(x_sym), math.cosh),
            (sp.atan(x_sym), math.atan),
        ]
        test_boxes = [
            (0.0, 0.0),
            (math.pi / 2.0, math.pi / 2.0),
            (-math.pi / 2.0, -math.pi / 2.0),
            (1.50, 1.65),     # cruza pi/2 (~1.5707963): máximo estacionario interior de sin(x)
            (3.10, 3.20),     # cruza pi (~3.1415926): mínimo estacionario interior de cos(x)
            (-1.65, -1.50),   # cruza -pi/2: mínimo estacionario interior de sin(x)
            (-4.80, -4.65),   # cruza -3*pi/2: máximo estacionario interior de sin(x)
            (0.20, 1.10),     # monótono sin extremos estacionarios
            (-2.50, 2.50),    # cruza 0 (mínimo de cosh, inflexión de tanh/sinh)
            (-8.0, 8.0),      # ancho > 2*pi
        ]
        for expr_sp, py_fn in fn_specs:
            for a, b in test_boxes:
                iv_lo, iv_hi = _eval_expr_interval(expr_sp, {"x": (a, b)})
                assert iv_lo <= iv_hi
                probes = [a, b, 0.5 * (a + b)]
                for idx in range(1, 20):
                    probes.append(a + (b - a) * (idx / 20.0))
                for k_mult in range(-6, 7):
                    cp = k_mult * (math.pi / 2.0)
                    if a <= cp <= b:
                        probes.append(cp)
                for pt in probes:
                    val = py_fn(pt)
                    assert iv_lo <= val <= iv_hi, (
                        f"Outward inclusion violated for {expr_sp} at x={pt} in [{a}, {b}]: "
                        f"val={val} not in [{iv_lo}, {iv_hi}]"
                    )

        # Cotas asintóticas exactas en (-inf, +inf) y constante `pi`
        assert _eval_expr_interval(sp.sin(x_sym), {"x": (-math.inf, math.inf)}) == (-1.0, 1.0)
        assert _eval_expr_interval(sp.cos(x_sym), {"x": (-math.inf, math.inf)}) == (-1.0, 1.0)
        assert _eval_expr_interval(sp.tanh(x_sym), {"x": (-math.inf, math.inf)}) == (-1.0, 1.0)
        pi_lo, pi_hi = _eval_expr_interval(sp.pi, {})
        assert pi_lo <= math.pi <= pi_hi and (pi_hi - pi_lo) < 1e-14

        # `tan(x)` acotado fuera de polos vs `(-inf, +inf)` al cruzar pi/2
        tan_safe_lo, tan_safe_hi = _eval_expr_interval(sp.tan(x_sym), {"x": (0.1, 1.0)})
        assert math.isfinite(tan_safe_lo) and math.isfinite(tan_safe_hi)
        assert tan_safe_lo <= math.tan(0.1) and math.tan(1.0) <= tan_safe_hi
        assert _eval_expr_interval(sp.tan(x_sym), {"x": (0.0, 2.0)}) == (-math.inf, math.inf)

        # 2. Pruebas deductivas end-to-end en `cge_prove` sobre `sin`, `cos`, `tanh`
        g_wave = CausalGraph(system_id="wave_bnb")
        g_wave.add_node(_node(id="x", category="control", current_value=0.0, bounds={"min": -3.0, "max": 3.0}))
        g_wave.add_node(
            _node(id="wave", category="state", current_value=0.5, symbolic_expr="sin(x) + 0.5 * cos(x) + tanh(x)")
        )
        g_wave.add_node(_node(id="sat", category="state", current_value=0.0, symbolic_expr="tanh(x)"))
        g_wave.add_node(
            _node(id="pythag", category="target", current_value=1.0, symbolic_expr="sin(x)**2 + cos(x)**2")
        )
        for tgt in ("wave", "sat", "pythag"):
            g_wave.add_edge(_edge(source="x", target=tgt))

        # Cota global verdadera (máximo real ~1.9537 en x~1.24) -> PROVEN por aritmética de intervalos B&B
        p_wave_ok = cge_prove(invariant_expr="wave <= 2.0 and wave >= -2.1", graph_data=g_wave.to_dict())
        assert p_wave_ok["status"] == "PROVEN"
        assert p_wave_ok["seal"] == "PROVEN"
        assert p_wave_ok["proof_method"] == "interval_arithmetic_enclosure"

        # Cota falsa (1.90 < 1.9537) -> FALSIFIED con contraejemplo verificado
        p_wave_bad = cge_prove(invariant_expr="wave <= 1.90", graph_data=g_wave.to_dict())
        assert p_wave_bad["status"] == "FALSIFIED"
        assert p_wave_bad["evaluated_state"]["wave"] > 1.90

        # Cota estricta de saturación hiperbólica |tanh(x)| < 1.0 en [-3, 3] -> PROVEN
        p_tanh_strict = cge_prove(invariant_expr="sat < 1.0 and sat > -1.0", graph_data=g_wave.to_dict())
        assert p_tanh_strict["status"] == "PROVEN"
        assert p_tanh_strict["seal"] == "PROVEN"

        # Identidad trigonométrica exacta sin(x)^2 + cos(x)^2 == 1.0 -> PROVEN simbólico
        p_pythag = cge_prove(invariant_expr="pythag == 1.0", graph_data=g_wave.to_dict())
        assert p_pythag["status"] == "PROVEN"
        assert p_pythag["proof_method"] == "symbolic_algebraic_identity"

        # Trampa adversaria de extremo estacionario interior: en [1.50, 1.65], ambos extremos cumplen
        # sin(1.50)=0.99749 < 0.998 y sin(1.65)=0.99687 < 0.998, pero en pi/2 interior sin(pi/2)=1.0!
        p_interior_peak = cge_prove(
            invariant_expr="sin(x) <= 0.998",
            domain_bounds={"x": [1.50, 1.65]},
            graph_data=g_wave.to_dict(),
        )
        assert p_interior_peak["status"] == "FALSIFIED"
        assert math.sin(p_interior_peak["counterexample"]["x"]) > 0.998

        # Superficie acoplada 2D `sin(x)*cos(y) + 0.25*tanh(x - y)` en [-2, 2]^2 -> PROVEN
        g_2d = CausalGraph(system_id="trig_2d")
        g_2d.add_node(_node(id="x", category="control", current_value=0.0, bounds={"min": -2.0, "max": 2.0}))
        g_2d.add_node(_node(id="y", category="control", current_value=0.0, bounds={"min": -2.0, "max": 2.0}))
        g_2d.add_node(
            _node(id="surf", category="target", current_value=0.0, symbolic_expr="sin(x) * cos(y) + 0.25 * tanh(x - y)")
        )
        g_2d.add_edge(_edge(source="x", target="surf"))
        g_2d.add_edge(_edge(source="y", target="surf"))
        p_2d = cge_prove(invariant_expr="surf <= 1.26 and surf >= -1.26", graph_data=g_2d.to_dict())
        assert p_2d["status"] == "PROVEN"
        assert p_2d["seal"] == "PROVEN"
        assert p_2d["proof_method"] == "interval_arithmetic_enclosure"

        # Guarda de polo en `tan(x)` al cruzar pi/2 vs prueba certificada fuera del polo
        g_tan = CausalGraph(system_id="tan_pole")
        g_tan.add_node(_node(id="x", category="control", current_value=0.5, bounds={"min": 0.0, "max": 2.0}))
        g_tan.add_node(_node(id="t", category="state", current_value=0.5463, symbolic_expr="tan(x)"))
        g_tan.add_node(_node(id="rem_trig", category="target", current_value=0.4794, symbolic_expr="tan(x) * cos(x)"))
        g_tan.add_edge(_edge(source="x", target="t"))
        g_tan.add_edge(_edge(source="x", target="rem_trig"))

        # Divergencia vertical `tan(x) -> +inf` a la izquierda de pi/2 falsifica honestamente `t <= 100.0`
        p_tan_blowup = cge_prove(invariant_expr="t <= 100.0", graph_data=g_tan.to_dict())
        assert p_tan_blowup["status"] == "FALSIFIED"
        assert p_tan_blowup["evaluated_state"]["t"] > 100.0

        # Polo removible `tan(x)*cos(x) == sin(x)` en [0, 2] (indefinido en x = pi/2 donde cos(x) == 0):
        # bloqueado con DOMAIN_SINGULARITY en [0, 2], pero PROVEN en [0, 1] libre de polos!
        p_tan_pole = cge_prove(invariant_expr="rem_trig == sin(x)", graph_data=g_tan.to_dict())
        assert p_tan_pole["status"] == "UNSUPPORTED"
        assert p_tan_pole["error_code"] == "DOMAIN_SINGULARITY"
        p_tan_safe = cge_prove(
            invariant_expr="t >= 0.0 and t <= 1.56 and rem_trig == sin(x)",
            domain_bounds={"x": [0.0, 1.0]},
            graph_data=g_tan.to_dict(),
        )
        assert p_tan_safe["status"] == "PROVEN"

        # Compatibilidad de `cge_trace` y `cge_do` con mecanismos trigonométricos/hiperbólicos
        r_tr = cge_trace(target_id="wave", source_id="x", graph_data=g_wave.to_dict())
        assert r_tr["status"] == "OK"
        # En x=0: d/dx [sin(x) + 0.5*cos(x) + tanh(x)] = cos(0) - 0.5*sin(0) + sech^2(0) = 1.0 + 1.0 = 2.0
        assert r_tr["source_trace"]["total_derivative"] == pytest.approx(2.0, abs=1e-6)
        r_do_trig = cge_do(interventions={"x": math.pi / 2.0}, target_id="wave", graph_data=g_wave.to_dict())
        assert r_do_trig["status"] == "OK"
        assert r_do_trig["intervened_state"]["wave"] == pytest.approx(1.0 + math.tanh(math.pi / 2.0), abs=1e-9)

    def test_falsification_counterexamples_and_t2_conservation_bug(self):
        oracle = ActorDBOracle(default_source=GOLDEN_CASE_DIR, allow_auto_discovery=False)

        # 1. Bug de conservación contable T2 en golden_case (`c_ledger` duplica débito en process_refund => residual != 0)
        p_ledger = oracle.prove(constraint_id="c_ledger")
        assert p_ledger["status"] == "FALSIFIED"
        assert p_ledger["seal"] == "PROVEN"
        assert p_ledger["is_valid"] is False
        assert p_ledger["counterexample"]["function"] == "process_refund"
        assert p_ledger["counterexample"]["symbolic_residual"] == "-amount"

        # 2. Implicación falsa: order_qty >= 500 => unit_cost <= 1.0 (en order_qty=500, unit_cost=1.10 > 1.0)
        p_false_imp = oracle.prove(invariant_expr="order_qty >= 500 => unit_cost <= 1.0")
        assert p_false_imp["status"] == "FALSIFIED"
        assert p_false_imp["is_valid"] is False
        ce = p_false_imp["counterexample"]
        assert ce["order_qty"] >= 500
        assert p_false_imp["evaluated_state"]["unit_cost"] > 1.0

        # 3. `c_budget` sin asumir restricciones del sistema (`assume_system_constraints=False`) se falsifica en la caja bruta
        p_raw_box = oracle.prove(constraint_id="c_budget", assume_system_constraints=False)
        assert p_raw_box["status"] == "FALSIFIED"
        assert p_raw_box["violation_magnitude"] > 0.0

    def test_vacuous_truth_immunity_and_honest_sampled_degradation_and_full_audit(self):
        # 1. Inmunidad a verdad vacua: si las restricciones del sistema son contradictorias, JAMÁS devolver PROVEN
        g_contra = CausalGraph(system_id="contradictory_domain")
        g_contra.add_node(_node(id="x", category="control", current_value=5.0, bounds={"min": 0.0, "max": 10.0}))
        g_contra.add_node(_node(id="y", category="target", current_value=5.0, symbolic_expr="x"))
        g_contra.add_edge(_edge(source="x", target="y"))
        g_contra.add_constraint(_constr(id="c_low", expression="x <= 3.0", type="hard"))
        g_contra.add_constraint(_constr(id="c_high", expression="x >= 7.0", type="hard"))

        p_vac = cge_prove(invariant_expr="x >= 999999.0", graph_data=g_contra.to_dict())
        assert p_vac["status"] == "INFEASIBLE_DOMAIN"
        assert p_vac["is_valid"] is False
        assert set(p_vac["iis_constraints"]) == {"c_low", "c_high"}

        # 2. Degradación honesta a SAMPLED_VERIFIED cuando interviene un sustituto de caja negra (blackbox_surrogates)
        g_surr = CausalGraph(
            system_id="surrogate_sys",
            metadata={"blackbox_surrogates": {"y": {"surrogate_sympy_expr": "2.0 * x + 1.0", "r2": 0.99}}},
        )
        g_surr.add_node(_node(id="x", category="control", current_value=1.0, bounds={"min": 0.0, "max": 5.0}))
        g_surr.add_node(_node(id="y", category="target", current_value=3.0, symbolic_expr="2.0 * x + 1.0"))
        g_surr.add_edge(_edge(source="x", target="y"))

        p_surr = cge_prove(invariant_expr="y >= 1.0", graph_data=g_surr.to_dict())
        assert p_surr["status"] == "SAMPLED_VERIFIED"
        assert p_surr["seal"] == "SAMPLED_VERIFIED"

        # 3. Auditoría completa del sistema `cge_prove()` sin argumentos sobre golden_case
        oracle = ActorDBOracle(default_source=GOLDEN_CASE_DIR, allow_auto_discovery=False)
        full_audit = oracle.prove()
        assert full_audit["status"] == "FALSIFIED"
        assert "c_ledger" in full_audit["falsified_invariants"]
        assert full_audit["invariants"]["c_budget"]["status"] == "PROVEN"

    def test_strict_antecedent_motzkin_soundness_and_micro_violation_precision(self):
        g_pw = CausalGraph(system_id="strict_ant_sys")
        g_pw.add_node(_node(id="x", category="control", current_value=2.0, bounds={"min": 0.0, "max": 10.0}))
        g_pw.add_node(
            _node(id="pw", category="target", current_value=10.0, symbolic_expr="Piecewise((10.0, x < 5.0), (0.0, True))")
        )
        g_pw.add_edge(_edge(source="x", target="pw"))

        # Franja abierta (4.999999, 5.0) viola `x <= 4.999999` aunque mida solo 1e-6 -> FALSIFIED exacto por Motzkin LP!
        r_sliver = cge_prove(invariant_expr="x < 5.0 => x <= 4.999999", graph_data=g_pw.to_dict())
        assert r_sliver["status"] == "FALSIFIED"
        assert 4.999999 < r_sliver["counterexample"]["x"] < 5.0

        # Antecedente estricto válido sobre la rama `x < 5.0` de Piecewise -> PROVEN
        r_valid = cge_prove(invariant_expr="x < 5.0 => pw >= 9.0", graph_data=g_pw.to_dict())
        assert r_valid["status"] == "PROVEN"
        assert r_valid["seal"] == "PROVEN"

        # Micro-violación de magnitud 5e-6 (> 1e-7) jamás cae en zona muerta
        g_micro = CausalGraph(system_id="micro_viol")
        g_micro.add_node(_node(id="x", category="control", current_value=1.0, bounds={"min": 0.0, "max": 1.0}))
        g_micro.add_node(_node(id="risk", category="target", current_value=0.020005, symbolic_expr="0.02 + 0.000005 * x"))
        g_micro.add_edge(_edge(source="x", target="risk"))
        r_micro = cge_prove(invariant_expr="risk <= 0.02", graph_data=g_micro.to_dict())
        assert r_micro["status"] == "FALSIFIED"
        assert r_micro["is_valid"] is False

        # k-ary Min(a, b, c) y Max(a, b, c) probados deductivamente por MILP
        g_kary = CausalGraph(system_id="kary_minmax")
        for vid in ("a", "b", "c"):
            g_kary.add_node(_node(id=vid, category="control", current_value=3.0, bounds={"min": 0.0, "max": 10.0}))
        g_kary.add_node(_node(id="lo_v", category="state", current_value=3.0, symbolic_expr="Min(a, b, c)"))
        g_kary.add_node(_node(id="hi_v", category="target", current_value=3.0, symbolic_expr="Max(a, b, c)"))
        for vid in ("a", "b", "c"):
            g_kary.add_edge(_edge(source=vid, target="lo_v"))
            g_kary.add_edge(_edge(source=vid, target="hi_v"))
        r_kary = cge_prove(invariant_expr="lo_v <= a and lo_v <= hi_v and hi_v >= c", graph_data=g_kary.to_dict())
        assert r_kary["status"] == "PROVEN"
        assert r_kary["seal"] == "PROVEN"

    def test_propositional_boolean_algebra_or_not_iff_neq(self):
        g = CausalGraph(system_id="prop_logic")
        g.add_node(_node(id="x", category="control", current_value=2.0, bounds={"min": 0.0, "max": 10.0}))
        g.add_node(
            _node(id="pw", category="target", current_value=10.0, symbolic_expr="Piecewise((10.0, x < 5.0), (0.0, True))")
        )
        g.add_edge(_edge(source="x", target="pw"))

        assert cge_prove(invariant_expr="x < 5.0 => (pw >= 9.0 or x >= 5.0)", graph_data=g.to_dict())["status"] == "PROVEN"
        assert cge_prove(invariant_expr="not (x < 5.0 and pw <= 5.0)", graph_data=g.to_dict())["status"] == "PROVEN"
        assert cge_prove(invariant_expr="(pw == 10.0) <=> (x < 5.0)", graph_data=g.to_dict())["status"] == "PROVEN"
        assert cge_prove(invariant_expr="x >= 5.0 => pw != 10.0", graph_data=g.to_dict())["status"] == "PROVEN"
        assert cge_prove(invariant_expr="pw != 0.0", graph_data=g.to_dict())["status"] == "FALSIFIED"

    def test_zero_control_and_domain_bounds_vacuous_truth_immunity(self):
        # 1. Grafo de 0 controles con restricciones exógenas contradictorias -> INFEASIBLE_DOMAIN con Minimal IIS
        g_exo_contra = CausalGraph(system_id="zero_ctrl_contra")
        g_exo_contra.add_node(_node(id="u", category="exogenous", current_value=5.0, bounds={"min": 0.0, "max": 10.0}))
        g_exo_contra.add_node(_node(id="y", category="target", current_value=5.0, symbolic_expr="u"))
        g_exo_contra.add_edge(_edge(source="u", target="y"))
        g_exo_contra.add_constraint(_constr(id="c_low", expression="u <= 3.0"))
        g_exo_contra.add_constraint(_constr(id="c_high", expression="u >= 7.0"))
        g_exo_contra.add_constraint(_constr(id="c_decoy", expression="u <= 9.5"))

        r_exo = cge_prove(invariant_expr="u >= 999999.0", graph_data=g_exo_contra.to_dict())
        assert r_exo["status"] == "INFEASIBLE_DOMAIN"
        assert r_exo["is_valid"] is False
        assert set(r_exo["iis_constraints"]) == {"c_low", "c_high"}

        # 2. domain_bounds contradice restricción del sistema -> INFEASIBLE_DOMAIN (jamás PROVEN vacuo!)
        g_db = CausalGraph(system_id="db_contra")
        g_db.add_node(_node(id="x", category="control", current_value=2.0, bounds={"min": 0.0, "max": 10.0}))
        g_db.add_node(_node(id="y", category="target", current_value=2.0, symbolic_expr="x"))
        g_db.add_edge(_edge(source="x", target="y"))
        g_db.add_constraint(_constr(id="c1", expression="x <= 4.0"))

        r_db = cge_prove(invariant_expr="x == -999.0", domain_bounds={"x": [6.0, 10.0]}, graph_data=g_db.to_dict())
        assert r_db["status"] == "INFEASIBLE_DOMAIN"
        assert r_db["is_valid"] is False

        # 3. Cota invertida lo > hi -> ERROR INVALID_DOMAIN_BOUNDS
        r_inv_b = cge_prove(invariant_expr="x >= 0.0", domain_bounds={"x": [8.0, 2.0]}, graph_data=g_db.to_dict())
        assert r_inv_b["status"] == "ERROR"
        assert r_inv_b["error_code"] == "INVALID_DOMAIN_BOUNDS"

    def test_exogenous_shock_non_truncation_parity_and_interior_shock_falsification(self):
        # 1. Paridad exacta entre constraint_id e invariant_expr ante shocks exógenos
        g_shock = CausalGraph(system_id="shock_parity")
        g_shock.add_node(_node(id="shock_u", category="exogenous", current_value=10.0, bounds={"min": 0.0, "max": 50.0}))
        g_shock.add_node(_node(id="reserve", category="target", current_value=15.0, symbolic_expr="25.0 - shock_u"))
        g_shock.add_edge(_edge(source="shock_u", target="reserve"))
        g_shock.add_constraint(_constr(id="c_solvency", expression="reserve >= 0"))

        r_by_cid = cge_prove(constraint_id="c_solvency", graph_data=g_shock.to_dict())
        r_by_expr = cge_prove(invariant_expr="reserve >= 0", graph_data=g_shock.to_dict())
        assert r_by_cid["status"] == "FALSIFIED"
        assert r_by_expr["status"] == "FALSIFIED"

        # 2. Shock exógeno cóncavo interior (seguro en esquinas u=0 y u=50, insolvente en el interior u=25)
        g_int = CausalGraph(system_id="interior_shock")
        g_int.add_node(_node(id="u", category="exogenous", current_value=2.0, bounds={"min": 0.0, "max": 50.0}))
        g_int.add_node(_node(id="x", category="control", current_value=5.0, bounds={"min": 0.0, "max": 10.0}))
        g_int.add_node(_node(id="reserve", category="target", current_value=9.0, symbolic_expr="100.0 + x - u * (50.0 - u)"))
        g_int.add_edge(_edge(source="u", target="reserve"))
        g_int.add_edge(_edge(source="x", target="reserve"))
        g_int.add_constraint(_constr(id="c_solv", expression="reserve >= 0"))

        r_int = cge_prove(constraint_id="c_solv", graph_data=g_int.to_dict())
        assert r_int["status"] == "FALSIFIED"
        assert r_int["is_valid"] is False

    def test_extreme_ieee754_outward_bounds_scientific_literals_and_domain_singularities(self):
        # 1. Literales numéricos en notación científica (`1e5`, `-1e5`, `1e-5`, `2.5e-1`) sin falsos `UNKNOWN_ACTOR`
        g_sci = CausalGraph(system_id="sci_literals")
        g_sci.add_node(_node(id="x", category="control", current_value=1.0, bounds={"min": -5.0, "max": 5.0}))
        assert cge_prove(
            invariant_expr="x <= 1e5 and x >= -1e5 and 2.5e-1 * x * 1e-5 <= 1e-4",
            graph_data=g_sci.to_dict(),
        )["status"] == "PROVEN"

        # 2. Envolventes IEEE-754 / IEEE-1788 outward-rounded en cotas extremas de `sinh`, `cosh`, `exp`, `pow`, `sin`, `asin`, `acos`
        iv_sh = _iv_sinh((701.0, 702.0))
        assert math.isfinite(iv_sh[0]) and math.isfinite(iv_sh[1])
        assert iv_sh[0] < math.sinh(701.0) and iv_sh[1] > math.sinh(702.0)

        iv_ch = _iv_cosh((701.0, 702.0))
        assert math.isfinite(iv_ch[0]) and math.isfinite(iv_ch[1])
        assert iv_ch[0] < math.cosh(701.0) and iv_ch[1] > math.cosh(702.0)

        # Desbordamiento superior a +inf acota el extremo inferior finito en `DBL_MAX` (jamás `(+inf, +inf)` ni `OverflowError`)
        assert _iv_sinh((800.0, 900.0)) == (sys.float_info.max, math.inf)
        assert _iv_pow((1e200, 1e201), 2.0, True) == (sys.float_info.max, math.inf)

        x_sym = sp.Symbol("x", real=True)
        sin_big_lo, sin_big_hi = _eval_expr_interval(sp.sin(x_sym), {"x": (1e12, 1e12 + 0.1)})
        assert sin_big_lo <= math.sin(1e12) <= sin_big_hi
        assert sin_big_lo <= math.sin(1e12 + 0.1) <= sin_big_hi

        asin_lo, asin_hi = _eval_expr_interval(sp.asin(x_sym), {"x": (-0.5, 0.5)})
        acos_lo, acos_hi = _eval_expr_interval(sp.acos(x_sym), {"x": (-0.5, 0.5)})
        assert asin_lo <= math.asin(-0.5) and math.asin(0.5) <= asin_hi
        assert acos_lo <= math.acos(0.5) and math.acos(-0.5) <= acos_hi

        # `exp(x)` en [701, 702] (~2.95e304..8.02e304 < 1e307) y en [-702, -701] (~3.62e-305 > 1e-310) -> FALSIFIED exacto
        g_exp_hi = CausalGraph(system_id="exp_hi")
        g_exp_hi.add_node(_node(id="x", category="control", current_value=701.5, bounds={"min": 701.0, "max": 702.0}))
        g_exp_hi.add_node(_node(id="ex", category="target", current_value=1.0, symbolic_expr="exp(x)"))
        g_exp_hi.add_edge(_edge(source="x", target="ex"))
        assert cge_prove(invariant_expr="ex >= 10.0**307", graph_data=g_exp_hi.to_dict())["status"] == "FALSIFIED"

        g_exp_lo = CausalGraph(system_id="exp_lo")
        g_exp_lo.add_node(_node(id="x", category="control", current_value=-701.5, bounds={"min": -702.0, "max": -701.0}))
        g_exp_lo.add_node(_node(id="ex", category="target", current_value=1.0, symbolic_expr="exp(x)"))
        g_exp_lo.add_edge(_edge(source="x", target="ex"))
        assert cge_prove(invariant_expr="ex <= 10.0**(-310)", graph_data=g_exp_lo.to_dict())["status"] == "FALSIFIED"

        # 3. Exactitud de frontera sin holgura (`±1e-9`): ni falsos `PROVEN` ni falsos `FALSIFIED`
        g_sin_peak = CausalGraph(system_id="sin_sub_eps")
        g_sin_peak.add_node(_node(id="x", category="control", current_value=1.5, bounds={"min": 0.0, "max": 3.14}))
        g_sin_peak.add_node(_node(id="s", category="target", current_value=0.997, symbolic_expr="sin(x)"))
        g_sin_peak.add_edge(_edge(source="x", target="s"))
        # En x = pi/2, sin(x) = 1.0 > 0.9999999995 (violación de 5e-10) -> FALSIFIED
        assert cge_prove(invariant_expr="s <= 0.9999999995", graph_data=g_sin_peak.to_dict())["status"] == "FALSIFIED"

        g_tanh_sat = CausalGraph(system_id="tanh_sat_hi")
        g_tanh_sat.add_node(_node(id="x", category="control", current_value=25.0, bounds={"min": 20.0, "max": 30.0}))
        g_tanh_sat.add_node(_node(id="th", category="target", current_value=1.0, symbolic_expr="tanh(x)"))
        g_tanh_sat.add_edge(_edge(source="x", target="th"))
        # Para x finito en [20, 30], tanh(x) < 1.0 es matemáticamente verdadero aunque float64 redondee a 1.0 -> jamás FALSIFIED
        assert cge_prove(invariant_expr="th < 1.0", graph_data=g_tanh_sat.to_dict())["status"] != "FALSIFIED"

        assert cge_prove(
            invariant_expr="x < 1.0",
            domain_bounds={"x": [0.0, 0.9999999995]},
            graph_data=g_sci.to_dict(),
        )["status"] == "PROVEN"

        # 4. Protección de singularidades de dominio real (`log`, `log1p`, `sqrt`, `asin`) en `cge_prove`, `cge_do` y `cge_solve`
        g_log_cancel = CausalGraph(system_id="log_cancel")
        g_log_cancel.add_node(_node(id="x", category="control", current_value=1.0, bounds={"min": -5.0, "max": 5.0}))
        g_log_cancel.add_node(_node(id="y", category="target", current_value=0.0, symbolic_expr="log(x) - log(x)"))
        g_log_cancel.add_edge(_edge(source="x", target="y"))
        r_log_sing = cge_prove(invariant_expr="y == 0.0", graph_data=g_log_cancel.to_dict())
        assert r_log_sing["status"] == "UNSUPPORTED" and r_log_sing["error_code"] == "DOMAIN_SINGULARITY"
        assert cge_prove(invariant_expr="x >= 0.5 => y == 0.0", graph_data=g_log_cancel.to_dict())["status"] == "PROVEN"
        assert cge_do(interventions={"x": -4.0}, target_id="y", graph_data=g_log_cancel.to_dict())["error_code"] == "DOMAIN_SINGULARITY"

        g_sqrt_sq = CausalGraph(system_id="sqrt_sq")
        g_sqrt_sq.add_node(_node(id="x", category="control", current_value=1.0, bounds={"min": -5.0, "max": 5.0}))
        g_sqrt_sq.add_node(_node(id="y", category="target", current_value=1.0, symbolic_expr="sqrt(x)**2"))
        g_sqrt_sq.add_edge(_edge(source="x", target="y"))
        r_sqrt_sing = cge_prove(invariant_expr="y == x", graph_data=g_sqrt_sq.to_dict())
        assert r_sqrt_sing["status"] == "UNSUPPORTED" and r_sqrt_sing["error_code"] == "DOMAIN_SINGULARITY"
        assert cge_do(interventions={"x": -4.0}, target_id="y", graph_data=g_sqrt_sq.to_dict())["error_code"] == "DOMAIN_SINGULARITY"

        g_asin_dom = CausalGraph(system_id="asin_dom")
        g_asin_dom.add_node(_node(id="x", category="control", current_value=0.0, bounds={"min": -2.0, "max": 2.0}))
        g_asin_dom.add_node(_node(id="y", category="target", current_value=0.0, symbolic_expr="asin(x)"))
        g_asin_dom.add_edge(_edge(source="x", target="y"))
        assert cge_prove(invariant_expr="y <= 2.0", graph_data=g_asin_dom.to_dict())["error_code"] == "DOMAIN_SINGULARITY"
        assert cge_prove(
            invariant_expr="y >= -1.6 and y <= 1.6",
            domain_bounds={"x": [-1.0, 1.0]},
            graph_data=g_asin_dom.to_dict(),
        )["status"] == "PROVEN"

        # `Piecewise` con guarda de dominio + optimización certificada con `log1p` explícito
        g_l1p = CausalGraph(system_id="l1p_sys")
        g_l1p.add_node(_node(id="x", category="control", current_value=1.0, bounds={"min": -5.0, "max": 10.0}))
        g_l1p.add_node(
            _node(id="g_pw", category="state", current_value=0.693, symbolic_expr="Piecewise((0.0, x <= 0.0), (log1p(x), True))")
        )
        g_l1p.add_edge(_edge(source="x", target="g_pw"))
        assert cge_prove(invariant_expr="g_pw >= 0.0", graph_data=g_l1p.to_dict())["status"] == "PROVEN"

        g_l1p_opt = CausalGraph(system_id="l1p_opt")
        g_l1p_opt.add_node(_node(id="x", category="control", current_value=1.0, bounds={"min": 0.0, "max": 15.0}))
        g_l1p_opt.add_node(_node(id="obj", category="target", current_value=5.93, symbolic_expr="10.0 * log1p(x) - x"))
        g_l1p_opt.add_edge(_edge(source="x", target="obj"))
        r_l1p_sol = cge_solve(target_id="obj", sense="max", graph_data=g_l1p_opt.to_dict())
        assert r_l1p_sol["status"] == "OPTIMAL"
        assert r_l1p_sol["controls"]["x"] == pytest.approx(9.0, abs=1e-4)


# ==============================================================================
# 6. Memoria Epistémica Indexada O(1) e Invalidación por Mtime (`GAP-04`)
# ==============================================================================


class TestEpistemicCache:
    def test_o1_cache_hit_and_automatic_filesystem_invalidation(self, tmp_path: Path):
        mod_file = tmp_path / "cached_model.py"
        mod_file.write_text(
            "def model(x: float = 2.0) -> float:\n"
            "    assert 0.0 <= x <= 10.0\n"
            "    profit = 5.0 * x\n"
            "    return profit\n",
            encoding="utf-8",
        )

        oracle = ActorDBOracle(allow_auto_discovery=False)
        r1 = oracle.solve(source_path=str(mod_file))
        assert r1["status"] == "OPTIMAL"
        assert r1["cache_hit"] is False
        assert r1["objective_value"] == pytest.approx(50.0)

        # Segunda consulta sin tocar disco -> cache_hit == True
        r2 = oracle.trace(source_path=str(mod_file))
        assert r2["status"] == "OK"
        assert r2["cache_hit"] is True

        # Modificar el archivo en disco asegurando cambio de tamaño/mtime_ns -> invalidación automática
        time.sleep(0.02)
        mod_file.write_text(
            "def model(x: float = 2.0) -> float:\n"
            "    assert 0.0 <= x <= 10.0\n"
            "    profit = 12.0 * x + 1.0\n"
            "    return profit\n",
            encoding="utf-8",
        )
        r3 = oracle.solve(source_path=str(mod_file))
        assert r3["status"] == "OPTIMAL"
        assert r3["cache_hit"] is False
        assert r3["objective_value"] == pytest.approx(121.0)


# ==============================================================================
# 7. Casos Borde Adversarios (Grafos Vacíos, Singularidades y Sub-Epsilon)
# ==============================================================================


class TestAdversarialEdgeCases:
    def test_empty_graph_and_singularities_and_continuous_piecewise_boundary(self):
        # 1. Grafo vacío (0 actores) manejado limpiamente en las 4 primitivas
        g_empty = CausalGraph(system_id="empty_sys").to_dict()
        assert cge_trace(graph_data=g_empty)["status"] == "ERROR"
        assert cge_do(interventions={}, graph_data=g_empty)["status"] == "ERROR"
        assert cge_solve(graph_data=g_empty)["status"] == "ERROR"
        p_empty = cge_prove(graph_data=g_empty)
        assert p_empty["status"] == "PROVEN"
        assert p_empty["total_invariants_checked"] == 0

        # 2. Ataque sub-épsilon sobre discontinuidad continua:
        # En [5.0 - 1e-8, 5.0), pw == 10.0 > 0.0 -> JAMÁS debe emitir PROVEN falsamente!
        g_pw = CausalGraph(system_id="pw_boundary")
        g_pw.add_node(_node(id="x", category="control", current_value=4.0, bounds={"min": 0.0, "max": 10.0}))
        g_pw.add_node(
            _node(id="pw", category="target", current_value=10.0, symbolic_expr="Piecewise((10.0, x < 5.0), (0.0, True))")
        )
        g_pw.add_edge(_edge(source="x", target="pw"))

        p_sub_eps = cge_prove(invariant_expr="x >= 4.99999999 => pw <= 0.0", graph_data=g_pw.to_dict())
        assert p_sub_eps["status"] != "PROVEN"

        # 3. Singularidad racional 1 / x sobre intervalo [-1, 1] que cruza x = 0: jamás debe colapsar ni emitir PROVEN
        g_sing = CausalGraph(system_id="singularity_sys")
        g_sing.add_node(_node(id="x", category="control", current_value=0.5, bounds={"min": -1.0, "max": 1.0}))
        g_sing.add_node(_node(id="inv_x", category="target", current_value=2.0, symbolic_expr="1.0 / x"))
        g_sing.add_edge(_edge(source="x", target="inv_x"))

        p_sing = cge_prove(invariant_expr="inv_x >= 0.0", graph_data=g_sing.to_dict())
        assert p_sing["status"] == "FALSIFIED"
        assert p_sing["counterexample"]["x"] < 0.0

    def test_removable_and_nonlinear_zero_over_zero_poles_blocked(self):
        # 1. Polo removible 0/0 en `y = x / x` sobre [-1, 1]: SymPy no debe borrar la división por cero!
        g_rem = CausalGraph(system_id="removable_pole")
        g_rem.add_node(_node(id="x", category="control", current_value=0.5, bounds={"min": -1.0, "max": 1.0}))
        g_rem.add_node(_node(id="y", category="target", current_value=1.0, symbolic_expr="x / x"))
        g_rem.add_edge(_edge(source="x", target="y"))

        r_pole = cge_prove(invariant_expr="y == 1.0", graph_data=g_rem.to_dict())
        assert r_pole["status"] == "UNSUPPORTED"
        assert r_pole["error_code"] == "DOMAIN_SINGULARITY"
        assert r_pole["seal"] == "UNSUPPORTED"

        # Cuando el antecedente excluye x == 0 (`x != 0` o `x > 0`), el dominio es libre de polos -> PROVEN
        assert cge_prove(invariant_expr="x != 0.0 => y == 1.0", graph_data=g_rem.to_dict())["status"] == "PROVEN"
        assert cge_prove(invariant_expr="x > 0.0 => y == 1.0", graph_data=g_rem.to_dict())["status"] == "PROVEN"

        # 2. Polo removible no lineal `(x - 1) / (x**2 - 1)` en [0, 2] con singularidad en x = 1.0
        g_nl = CausalGraph(system_id="nl_removable_pole")
        g_nl.add_node(_node(id="x", category="control", current_value=0.5, bounds={"min": 0.0, "max": 2.0}))
        g_nl.add_node(_node(id="y", category="target", current_value=0.666667, symbolic_expr="(x - 1.0) / (x**2 - 1.0)"))
        g_nl.add_edge(_edge(source="x", target="y"))

        r_nl = cge_prove(invariant_expr="y >= 0.0", graph_data=g_nl.to_dict())
        assert r_nl["status"] == "UNSUPPORTED"
        assert r_nl["error_code"] == "DOMAIN_SINGULARITY"
