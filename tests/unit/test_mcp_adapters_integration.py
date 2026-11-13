"""Suite de integración profunda MCP (`cge/mcp/`) <-> Adaptadores AST (`cge/ingest_adapters/`) y Causal Diff."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Any

import anyio
import pytest
import uvicorn
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.memory import create_connected_server_and_client_session

from cge.diff import diff_sources
from cge.ingest_adapters.engine import RepositoryCausalEngine
from cge.mcp import ActorDBOracle, create_mcp_server

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
REPOS_ROOT = WORKSPACE_ROOT / "benchmark" / "baseline_bench" / "repos"


def _extract_tool_payload(call_result: Any) -> dict[str, Any]:
    assert call_result.isError is False, f"MCP tool returned protocol error: {call_result}"
    if getattr(call_result, "structuredContent", None) and isinstance(call_result.structuredContent, dict):
        sc = call_result.structuredContent
        if "result" in sc and isinstance(sc["result"], dict) and "status" in sc["result"]:
            return sc["result"]
        if "status" in sc:
            return sc
    assert call_result.content, "MCP CallToolResult contained empty content"
    return json.loads(call_result.content[0].text)


def test_all_adapter_submodels_across_four_mcp_tools():
    """Verifica los 5 repositorios y sus 14 submodelos en `cge_trace`, `cge_do`, `cge_solve` y `cge_prove`."""

    async def _scenario():
        server = create_mcp_server(allow_auto_discovery=False)
        eng = RepositoryCausalEngine()

        async with create_connected_server_and_client_session(server) as session:
            tools_res = await session.list_tools()
            tool_names = {t.name for t in tools_res.tools}
            assert tool_names == {"cge_trace", "cge_do", "cge_solve", "cge_prove"}

            for repo_id in eng.ADAPTER_REGISTRY:
                adapter = eng.get_adapter(repo_id)
                assert adapter is not None
                submodels = adapter.list_submodels()
                assert len(submodels) >= 1

                for sub in submodels:
                    src = f"{repo_id}:{sub}"

                    tr = _extract_tool_payload(
                        await session.call_tool("cge_trace", {"source_path": src})
                    )
                    assert tr["status"] == "OK", f"trace failed on {src}: {tr}"
                    assert tr["dead_node_ids"] == [], f"dead nodes in {src}: {tr['dead_nodes']}"
                    assert tr["zero_hallucination_verified"] is True

                    tgt = tr["target_id"]
                    do_res = _extract_tool_payload(
                        await session.call_tool(
                            "cge_do",
                            {"interventions": {}, "target_id": tgt, "source_path": src},
                        )
                    )
                    assert do_res["status"] == "OK", f"do failed on {src}: {do_res}"
                    assert do_res["is_feasible"] is True

                    sol = _extract_tool_payload(
                        await session.call_tool("cge_solve", {"source_path": src})
                    )
                    assert sol["status"] == "OPTIMAL", f"solve failed on {src}: {sol}"

                    prv = _extract_tool_payload(
                        await session.call_tool(
                            "cge_prove",
                            {"source_path": src, "n_samples": 24},
                        )
                    )
                    assert prv["status"] in ("PROVEN", "SAMPLED_VERIFIED"), f"prove failed on {src}: {prv}"
                    assert prv["jointly_feasible"] is True

    anyio.run(_scenario)


def test_supplychain_and_dispatch_exact_optima_and_pressure_refusal():
    """Verifica optimalidad exacta (EOQ, descuento por volumen, despacho KKT) y rechazo de bypass."""
    oracle = ActorDBOracle(allow_auto_discovery=False)

    # 1. EOQ con techo de almacén activo y resistencia a ignore_constraints=True
    r_eoq = oracle.solve(
        source_path="supplychainpy_mini",
        exo_overrides={"warehouse_capacity": 800.0},
        ignore_constraints=True,
    )
    assert r_eoq["status"] == "REJECTED_CONSTRAINT_BYPASS"
    assert r_eoq["seal"] == "PROVEN"
    assert r_eoq["pressure_resisted"] is True
    assert r_eoq["controls"]["order_quantity"] == pytest.approx(800.0, rel=1e-4)

    r_eoq_id = oracle.prove(
        invariant_expr="total_cost == purchase_cost + ordering_cost + holding_cost",
        source_path="supplychainpy_mini",
    )
    assert r_eoq_id["status"] == "PROVEN"
    assert r_eoq_id["seal"] == "PROVEN"

    # 2. Descuento por volumen: óptimo interior del tramo 2 (D=12000) y salto exacto al umbral Q=1000 (D=6000)
    r_disc = oracle.solve(source_path="adapter://supplychainpy_mini:volume_discount")
    assert r_disc["status"] == "OPTIMAL"
    assert r_disc["seal"] == "PROVEN"
    assert r_disc["controls"]["order_quantity"] == pytest.approx((2.0 * 12000.0 * 100.0 / 1.7) ** 0.5, rel=1e-4)
    assert r_disc["evaluated_nodes"]["unit_cost"] == pytest.approx(8.5, rel=1e-6)

    r_disc_jump = oracle.solve(
        source_path="adapter://supplychainpy_mini:volume_discount",
        exo_overrides={"annual_demand": 6000.0},
    )
    assert r_disc_jump["status"] == "OPTIMAL"
    assert r_disc_jump["seal"] == "PROVEN"
    assert r_disc_jump["controls"]["order_quantity"] == pytest.approx(1000.0, rel=1e-4)
    assert r_disc_jump["objective_value"] == pytest.approx(52450.0, rel=1e-5)

    # 3. Safety stock: singularidad fuera de (0, 1) capturada limpiamente
    r_ss_bad = oracle.do(
        interventions={"service_level": 1.05},
        source_path=str(REPOS_ROOT / "supplychainpy_mini" / "safety_stock.py"),
    )
    assert r_ss_bad["status"] == "ERROR"
    assert r_ss_bad["error_code"] == "DOMAIN_SINGULARITY"

    # 4. Economic dispatch sobre directorio físico: KKT exacto e IIS ante sobrecarga
    disp_dir = str(REPOS_ROOT / "economic_dispatch")
    r_qp = oracle.solve(source_path=disp_dir)
    assert r_qp["status"] == "OPTIMAL"
    assert r_qp["seal"] == "PROVEN"
    p_sum = sum(r_qp["controls"][k] for k in ("P_G1", "P_G2", "P_G3"))
    assert p_sum == pytest.approx(300.0, abs=1e-4)
    assert r_qp["objective_value"] == pytest.approx(1948.365385, rel=1e-5)

    r_infeas = oracle.solve(source_path=disp_dir, exo_overrides={"load_demand": 600.0})
    assert r_infeas["status"] == "INFEASIBLE"
    assert r_infeas["seal"] == "PROVEN"
    assert "power_balance" in r_infeas["iis_constraints"]


def test_implicit_submodel_routing_and_cache_isolation():
    """Verifica enrutamiento automático entre submodelos sin colisiones de caché en `ActorDBOracle`."""
    oracle = ActorDBOracle(allow_auto_discovery=False)

    # Loan & Tax Engine: 3 submodelos enrutados implícitamente sobre el mismo repo_id
    r_tax = oracle.do(
        interventions={"taxable_income": 50000.0},
        target_id="net_income",
        source_path="loan_tax_engine",
    )
    assert r_tax["status"] == "OK"
    assert r_tax["cache_hit"] is False
    assert (
        r_tax["intervened_state"]["tax_total"] + r_tax["intervened_state"]["net_income"]
        == pytest.approx(50000.0, rel=1e-7)
    )

    # Segunda consulta al mismo submodelo debe golpear caché O(1)
    r_tax_cached = oracle.trace(target_id="net_income", source_path="loan_tax_engine")
    assert r_tax_cached["status"] == "OK"
    assert r_tax_cached["cache_hit"] is True

    # Consulta a `amortization` sobre el mismo `source_path="loan_tax_engine"` (caso borde r == 0)
    r_loan_zero = oracle.do(
        interventions={"annual_rate": 0.0},
        target_id="payment_per_period",
        source_path="loan_tax_engine",
    )
    assert r_loan_zero["status"] == "OK"
    assert r_loan_zero["cache_hit"] is False
    assert r_loan_zero["intervened_target_value"] == pytest.approx(100000.0 / 120.0, rel=1e-6)

    # Verificación de constraint_id perteneciente a un submodelo no por defecto + identidad contable exacta
    r_cap_prv = oracle.prove(constraint_id="c_payment_cap", source_path="loan_tax_engine", n_samples=24)
    assert r_cap_prv["system_id"] == "loan_tax_engine_amortization"
    assert r_cap_prv["status"] in ("PROVEN", "SAMPLED_VERIFIED")

    r_loan_id = oracle.prove(
        invariant_expr="total_paid == total_principal + total_interest",
        source_path="loan_tax_engine",
    )
    assert r_loan_id["system_id"] == "loan_tax_engine_amortization"
    assert r_loan_id["status"] == "PROVEN"
    assert r_loan_id["seal"] == "PROVEN"

    # QuantEcon: Solow Golden Rule (s* = alpha = 0.33) vs Cournot active-set exit
    solow_py = str(REPOS_ROOT / "quantecon_cournot_solow" / "solow.py")
    cournot_py = str(REPOS_ROOT / "quantecon_cournot_solow" / "cournot.py")

    r_gold = oracle.solve(source_path=solow_py)
    assert r_gold["status"] == "OPTIMAL"
    assert r_gold["seal"] == "PROVEN"
    assert r_gold["controls"]["s"] == pytest.approx(0.33, abs=1e-3)

    r_exit = oracle.do(
        interventions={"c2": 150.0},
        target_id="q1",
        source_path=cournot_py,
    )
    assert r_exit["status"] == "OK"
    assert r_exit["seal"] == "PROVEN"
    assert r_exit["intervened_state"]["q2"] == pytest.approx(0.0, abs=1e-8)
    assert r_exit["intervened_target_value"] == pytest.approx(45.0, rel=1e-6)


def test_live_ast_mutation_cache_invalidation_and_causal_diff(tmp_path: Path):
    """Copia un repositorio a un directorio temporal con otro nombre, muta el AST en disco y verifica
    que tanto `ActorDBOracle` (invalidación de caché) como `diff_sources` detectan la rotura de conservación."""
    src_seir = REPOS_ROOT / "epidemiology_seir"
    cloned_dir = tmp_path / "seir_feature_branch"
    shutil.copytree(src_seir, cloned_dir)

    oracle = ActorDBOracle(allow_auto_discovery=False)

    # 1. El clon intacto prueba conservación exacta de masa dS/dt + dE/dt + dI/dt + dR/dt == 0
    r_clean_1 = oracle.prove(
        constraint_id="c_ode_mass_conservation",
        source_path=str(cloned_dir),
    )
    assert r_clean_1["status"] == "PROVEN"
    assert r_clean_1["seal"] == "PROVEN"
    assert r_clean_1["cache_hit"] is False

    r_clean_2 = oracle.prove(
        constraint_id="c_ode_mass_conservation",
        source_path=str(cloned_dir),
    )
    assert r_clean_2["status"] == "PROVEN"
    assert r_clean_2["cache_hit"] is True

    # 2. Inyectar fuga no conservativa en `seir_model.py` en disco
    seir_file = cloned_dir / "seir_model.py"
    orig_code = seir_file.read_text(encoding="utf-8")
    mutated_code = orig_code.replace(
        "dI_dt = sigma * E - gamma * I",
        "dI_dt = sigma * E - gamma * I - 0.05 * I",
    )
    assert mutated_code != orig_code
    seir_file.write_text(mutated_code, encoding="utf-8")
    # Garantizar cambio de mtime_ns en sistemas de archivos con resolución gruesa
    st = os.stat(seir_file)
    os.utime(seir_file, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))

    # 3. La MISMA instancia `oracle` invalida su caché y refuta la conservación con contraejemplo exacto
    r_broken = oracle.prove(
        constraint_id="c_ode_mass_conservation",
        source_path=str(cloned_dir),
    )
    assert r_broken["cache_hit"] is False
    assert r_broken["status"] == "FALSIFIED"
    assert r_broken["counterexample"] is not None

    # 4. `diff_sources` (`cge/diff/`) diagnostica formalmente la regresión de invariante (`FAIL`)
    diff_rep = diff_sources(str(src_seir), str(cloned_dir))
    assert diff_rep.verdict == "FAIL"
    assert diff_rep.has_regressions is True
    reg_codes = [item.get("code") for item in diff_rep.invariant_regressions]
    assert any("CONSERVATION" in str(c) or "INVARIANT" in str(c) for c in reg_codes)


def test_adapter_error_paths_and_stdio_subprocess_roundtrip():
    """Verifica errores estructurados ante submodelos/rutas inválidas y transporte MCP `stdio` real."""
    oracle = ActorDBOracle(allow_auto_discovery=False)

    r_bad_sub = oracle.trace(source_path="loan_tax_engine:unknown_submodel")
    assert r_bad_sub["status"] == "ERROR"
    assert r_bad_sub["error_code"] == "INVALID_SUBMODEL"
    assert "tax_brackets" in r_bad_sub["available_submodels"]

    r_bad_uri = oracle.trace(source_path="adapter://nonexistent_domain_repo")
    assert r_bad_uri["status"] == "ERROR"
    assert r_bad_uri["error_code"] == "UNSUPPORTED_ADAPTER_REPO"

    # Ruta inexistente con letra de unidad Windows no debe confundirse con selector `:submodel`
    r_win_missing = oracle.trace(source_path=r"C:\nonexistent_cge_folder_999\module.py")
    assert r_win_missing["status"] == "ERROR"
    assert r_win_missing["error_code"] == "SOURCE_NOT_FOUND"

    async def _stdio_check():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "cge.mcp", "--transport", "stdio", "--source", "economic_dispatch"],
            cwd=str(WORKSPACE_ROOT),
        )
        async with (
            stdio_client(params) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            sol = _extract_tool_payload(await session.call_tool("cge_solve", {}))
            assert sol["status"] == "OPTIMAL"
            assert sol["seal"] == "PROVEN"
            assert sol["system_id"] == "economic_dispatch_qp_ir"

    anyio.run(_stdio_check)


def test_sandboxed_ast_no_toplevel_exec_and_relative_imports_safety(tmp_path: Path):
    """Garantiza cero ejecución arbitraria de top-level (`exec()`) y soporte de imports relativos."""
    oracle = ActorDBOracle(allow_auto_discovery=False)

    cases = [
        ("epidemiology_seir", "seir_model.py", "seir_trajectory", "from .metrics import calculate_r0\n"),
        ("supplychainpy_mini", "inventory_sim.py", "inventory_sim", "from .eoq import calculate_eoq\n"),
    ]
    for repo_name, filename, submodel, rel_import_stmt in cases:
        cloned = tmp_path / f"safe_{repo_name}"
        shutil.copytree(REPOS_ROOT / repo_name, cloned)
        marker = tmp_path / f"pwned_{repo_name}.txt"

        py_file = next(cloned.rglob(filename))
        orig_src = py_file.read_text(encoding="utf-8")
        injected = f"import pathlib\npathlib.Path({str(marker)!r}).write_text('EXECUTED')\n" + rel_import_stmt
        src_lines = orig_src.splitlines(keepends=True)
        insert_at = 1 if (src_lines and "from __future__" in src_lines[0]) else 0
        src_lines.insert(insert_at, injected)
        py_file.write_text("".join(src_lines), encoding="utf-8")

        res = oracle.do(interventions={}, source_path=f"{cloned}:{submodel}")
        assert not marker.exists(), f"Untrusted module top-level executed in {repo_name}!"
        assert res["status"] == "OK", f"Relative import broke {repo_name}:{submodel}: {res}"
        assert res["is_feasible"] is True


def test_blackbox_interior_falsification_and_piecewise_zero_rate_proof():
    """Verifica falsificación interior en cajas negras (`safety_stock`) y prueba `PROVEN` con `r == 0` en `amortization`."""
    oracle = ActorDBOracle(allow_auto_discovery=False)

    r_false_smooth = oracle.prove(
        invariant_expr="safety_stock <= 1e5 * ((service_level - 0.8731)**2 + (daily_demand_std - 13.71)**2)",
        source_path="supplychainpy_mini:safety_stock",
        n_samples=16,
    )
    assert r_false_smooth["status"] == "FALSIFIED"
    assert r_false_smooth["counterexample"] is not None

    r_false_disj = oracle.prove(
        invariant_expr="(service_level - 0.8731)**2 + (daily_demand_std - 13.71)**2 >= 1e-5 or safety_stock <= 0",
        source_path="supplychainpy_mini:safety_stock",
        n_samples=16,
    )
    assert r_false_disj["status"] == "FALSIFIED"
    assert r_false_disj["counterexample"] is not None

    # Dominio por defecto de amortización incluye annual_rate=0.0 con guarda Piecewise exacta
    r_pos_pay = oracle.prove(
        constraint_id="c_positive_payment",
        source_path="loan_tax_engine:amortization",
    )
    assert r_pos_pay["status"] == "PROVEN"
    assert r_pos_pay["seal"] == "PROVEN"
    assert r_pos_pay["domain_bounds"]["annual_rate"][0] == pytest.approx(0.0, abs=1e-15)

    r_zero_rate = oracle.prove(
        invariant_expr="total_paid >= principal",
        source_path="loan_tax_engine:amortization",
        domain_bounds={"annual_rate": [0.0, 0.0]},
    )
    assert r_zero_rate["status"] == "PROVEN"
    assert r_zero_rate["seal"] == "PROVEN"


def test_causal_diff_blackbox_and_nondefault_submodel_mutations(tmp_path: Path):
    """Verifica que `diff_sources` audita mutaciones en cajas negras y en submodelos no por defecto a nivel directorio."""
    # 1. Mutación dentro del bucle de simulación `inventory_sim.py`
    orig_sc = REPOS_ROOT / "supplychainpy_mini"
    mut_sc = tmp_path / "supplychainpy_mini_mut"
    shutil.copytree(orig_sc, mut_sc)
    sim_py = next(mut_sc.rglob("inventory_sim.py"))
    sim_txt = sim_py.read_text(encoding="utf-8")
    assert "orders_placed += 1" in sim_txt
    sim_py.write_text(sim_txt.replace("orders_placed += 1", "orders_placed += 100"), encoding="utf-8")

    rep_sim_dir = diff_sources(orig_sc, mut_sc)
    assert rep_sim_dir.is_semantically_equivalent is False
    assert "orders_placed" in rep_sim_dir.structural_diff["nodes_semantically_modified"]
    assert rep_sim_dir.verdict in ("WARN", "FAIL")

    # 2. Mutación en `cournot.py` (submodelos no por defecto `cournot` / `cournot_cartel`)
    orig_qe = REPOS_ROOT / "quantecon_cournot_solow"
    mut_qe = tmp_path / "quantecon_mut"
    shutil.copytree(orig_qe, mut_qe)
    cournot_py = next(mut_qe.rglob("cournot.py"))
    c_txt = cournot_py.read_text(encoding="utf-8")
    assert "market_price = max(0.0, float(a - b * total_q))" in c_txt
    cournot_py.write_text(
        c_txt.replace(
            "market_price = max(0.0, float(a - b * total_q))",
            "market_price = max(0.0, float(a - 2.0 * b * total_q))",
        ),
        encoding="utf-8",
    )
    rep_qe_dir = diff_sources(orig_qe, mut_qe)
    assert rep_qe_dir.is_semantically_equivalent is False
    assert rep_qe_dir.verdict == "FAIL"

    # 3. Mutación en `amortization.py` (submodelo no por defecto de `loan_tax_engine`)
    orig_lt = REPOS_ROOT / "loan_tax_engine"
    mut_lt = tmp_path / "loan_tax_mut"
    shutil.copytree(orig_lt, mut_lt)
    amort_py = next(mut_lt.rglob("amortization.py"))
    a_txt = amort_py.read_text(encoding="utf-8")
    assert "r = annual_rate / float(freq_per_year)" in a_txt
    amort_py.write_text(
        a_txt.replace(
            "r = annual_rate / float(freq_per_year)",
            "r = 2.0 * annual_rate / float(freq_per_year)",
        ),
        encoding="utf-8",
    )
    rep_lt_dir = diff_sources(orig_lt, mut_lt)
    assert rep_lt_dir.is_semantically_equivalent is False
    assert rep_lt_dir.verdict == "FAIL"


def test_implicit_cross_submodel_solve_and_streamable_http_adapter_transport():
    """Verifica enrutamiento implícito hacia `cournot_cartel` en `solve` y transporte real `streamable-http`."""
    oracle = ActorDBOracle(allow_auto_discovery=False)
    sol_cartel = oracle.solve(source_path="quantecon_cournot_solow", fixed_controls={"q1": 20.0})
    assert sol_cartel["status"] == "OPTIMAL"
    assert sol_cartel["seal"] == "PROVEN"
    assert sol_cartel["system_id"] == "quantecon_cournot_cartel"
    assert sol_cartel["controls"]["q2"] == pytest.approx(25.0, abs=1e-4)
    assert sol_cartel["objective_value"] == pytest.approx(2025.0, rel=1e-5)

    async def _http_adapter_check() -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(16)
        port = int(sock.getsockname()[1])

        mcp_srv = create_mcp_server(
            name="http-adapter-oracle",
            default_source="supplychainpy_mini",
            allow_auto_discovery=False,
            host="127.0.0.1",
            port=port,
        )
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
                assert time.monotonic() < deadline, "Timed out starting streamable-http server"
                await asyncio.sleep(0.02)

            url = f"http://127.0.0.1:{port}/mcp"
            async with (
                streamable_http_client(url) as (read_stream, write_stream, _),
                ClientSession(read_stream, write_stream) as session,
            ):
                await session.initialize()
                r_default = _extract_tool_payload(await session.call_tool("cge_solve", {}))
                assert r_default["status"] == "OPTIMAL"
                assert r_default["seal"] == "PROVEN"
                assert r_default["controls"]["order_quantity"] == pytest.approx(1095.445115, rel=1e-4)

                r_cross = _extract_tool_payload(
                    await session.call_tool(
                        "cge_prove",
                        {"constraint_id": "c_positive_payment", "source_path": "loan_tax_engine"},
                    )
                )
                assert r_cross["status"] == "PROVEN"
                assert r_cross["system_id"] == "loan_tax_engine_amortization"
        finally:
            uv_srv.should_exit = True
            await serve_task
            sock.close()

    anyio.run(_http_adapter_check)

