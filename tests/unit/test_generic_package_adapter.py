"""Suite de pruebas para la Ingesta Universal en 2 Capas (`GenericPackageAdapter` + `actordb.yaml`) y P5."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
import pytest

from cge.diff import CausalDiffEngine, DiffVerdict, diff_git_revisions, diff_sources
from cge.ingest_adapters import (
    GENERIC_PACKAGE_REPO_ID,
    ContractASTMismatchError,
    GenericPackageAdapter,
    RepositoryCausalEngine,
)
from cge.mcp.oracle import ActorDBOracle


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent


def _git(cwd: Path, *args: str) -> None:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr}")


def _build_unseen_saas_package(
    pkg_dir: Path,
    *,
    infra_per_seat: float = 6.0,
    include_dead_local: bool = True,
    churn_multiplier: float = 1.0,
) -> Path:
    """Crea un paquete Python multi-módulo realista y no registrado en `ADAPTER_REGISTRY`."""
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text('"""Unseen B2B SaaS unit economics package."""\n', encoding="utf-8")

    (pkg_dir / "tiers.py").write_text(
        f'''"""Constantes, dataclass de defaults y helper cross-módulo."""
from dataclasses import dataclass

BASE_PLATFORM_FEE: float = 100.0
INFRA_PER_SEAT: float = {infra_per_seat}


@dataclass
class PlanDefaults:
    seats: int = 20
    seat_price: float = 50.0
    discount_rate: float = 0.10
    vat_rate: float = 0.21


def compute_infra_cost(seats: int) -> float:
    if seats <= 0:
        raise ValueError("seats must be strictly positive")
    return BASE_PLATFORM_FEE + INFRA_PER_SEAT * seats
''',
        encoding="utf-8",
    )

    dead_stmt = "    vanity_badge_score = seats * 0.5\n" if include_dead_local else ""
    (pkg_dir / "evaluator.py").write_text(
        f'''"""Punto de entrada simbólico L0/L1 con imports relativos cross-módulo."""
from .tiers import compute_infra_cost


def evaluate_plan(
    seats: int = 20,
    seat_price: float = 50.0,
    discount_rate: float = 0.10,
    vat_rate: float = 0.21,
    legacy_referral_code: float = 0.0,
) -> dict:
    if seat_price <= 0.0:
        raise ValueError("seat_price must be positive")
    if discount_rate < 0.0 or discount_rate > 0.50:
        raise ValueError("discount_rate out of admissible range")
{dead_stmt}    gross_mrr = seats * seat_price * (1.0 - discount_rate)
    infra_cost = compute_infra_cost(seats)
    gross_profit = gross_mrr - infra_cost
    net_mrr = gross_profit - (0.0 * vat_rate)
    return {{
        "gross_mrr": gross_mrr,
        "infra_cost": infra_cost,
        "gross_profit": gross_profit,
        "net_mrr": net_mrr,
    }}
''',
        encoding="utf-8",
    )

    (pkg_dir / "cohort_sim.py").write_text(
        f'''"""Punto de entrada imperativo L2 con bucle y helper cross-módulo."""
from .tiers import compute_infra_cost


def simulate_cohort_ltv(
    initial_seats: int = 30,
    seat_price: float = 50.0,
    monthly_churn: float = 0.04,
    horizon_months: int = 6,
) -> dict:
    if initial_seats <= 0 or seat_price <= 0.0:
        raise ValueError("initial_seats and seat_price must be positive")
    if monthly_churn < 0.0 or monthly_churn >= 0.50:
        raise ValueError("monthly_churn out of bounds")
    active = float(initial_seats)
    cumulative_net_cashflow = 0.0
    for _m in range(int(horizon_months)):
        rev = active * seat_price
        cost = compute_infra_cost(active)
        cumulative_net_cashflow += (rev - cost)
        active = active * (1.0 - {churn_multiplier} * monthly_churn)
    return {{
        "ending_active_seats": active,
        "cumulative_net_cashflow": cumulative_net_cashflow,
    }}
''',
        encoding="utf-8",
    )

    # Añadir archivo de test interno para comprobar que el extractor lo ignora limpiamente
    tests_sub = pkg_dir / "tests"
    tests_sub.mkdir(exist_ok=True)
    (tests_sub / "test_internal.py").write_text(
        "def test_placeholder(x: float = 1.0) -> float:\n    return x * 999.0\n",
        encoding="utf-8",
    )
    return pkg_dir


def _write_sidecar(
    pkg_dir: Path,
    *,
    max_price: float = 90.0,
    min_profit_floor: float = 300.0,
    sense: str = "max",
) -> Path:
    sidecar = pkg_dir / "actordb.yaml"
    sidecar.write_text(
        f'''version: "1.0"
system_id: "saas_unit_economics"
default_submodel: "evaluate_plan"

submodels:
  evaluate_plan:
    entrypoint: "evaluator.py:evaluate_plan"
    target: "net_mrr"
    sense: "{sense}"
    controls:
      seat_price:
        bounds: [15.0, {max_price}]
      discount_rate:
        bounds: [0.0, 0.30]
    exogenous:
      seats:
        bounds: [5, 200]
      vat_rate:
        bounds: [0.0, 0.25]
    constraints:
      c_profit_floor:
        type: "hard"
        expression: "gross_profit >= {min_profit_floor}"
        description: "Minimum monthly gross profit floor"

  simulate_cohort_ltv:
    entrypoint: "cohort_sim.py:simulate_cohort_ltv"
    target: "cumulative_net_cashflow"
    sense: "max"
    controls:
      seat_price:
        bounds: [20.0, {max_price}]
    exogenous:
      initial_seats:
        bounds: [10, 200]
      monthly_churn:
        bounds: [0.01, 0.20]
      horizon_months:
        bounds: [3, 12]
''',
        encoding="utf-8",
    )
    return sidecar


class TestLayer1ZeroConfigAndP5:
    def test_zero_config_discovery_trace_do_prove_and_p5_gate(self, tmp_path: Path):
        pkg_dir = _build_unseen_saas_package(tmp_path / "saas_zero_cfg")
        repo_eng = RepositoryCausalEngine()
        spec = repo_eng.resolve_adapter_source(str(pkg_dir))
        assert spec is not None
        assert spec["repo_id"] == GENERIC_PACKAGE_REPO_ID
        assert spec["is_generic_package"] is True

        adapter = GenericPackageAdapter(str(pkg_dir))
        subs = adapter.list_submodels()
        # Descubre únicamente puntos de entrada raíz (ignora helper `compute_infra_cost` y `tests/test_internal.py`)
        assert set(subs) == {"evaluate_plan", "simulate_cohort_ltv"}

        oracle = ActorDBOracle(allow_auto_discovery=False)

        # 1. TRACE sobre submodelo simbólico L0/L1 (`PROVEN`)
        tr = oracle.trace(
            source_path=str(pkg_dir),
            target_id="net_mrr",
            source_id="seat_price",
        )
        assert tr["status"] == "OK"
        assert tr["seal"] == "PROVEN"
        # gross_mrr = 20 * 50 * 0.9 = 900; infra_cost = 100 + 6*20 = 220 => net_mrr = 680
        assert tr["nominal_target_value"] == pytest.approx(680.0)
        assert tr["source_trace"]["total_derivative"] == pytest.approx(18.0)  # 20 * (1 - 0.10)
        assert "vanity_badge_score" in tr["dead_node_ids"]

        # Verificar que `legacy_referral_code` y `vat_rate` (cancelado por 0.0 * vat_rate) son causalmente nulos hacia net_mrr
        tr_dead_param = oracle.trace(
            source_path=str(pkg_dir),
            target_id="net_mrr",
            source_id="legacy_referral_code",
        )
        assert tr_dead_param["source_trace"]["is_causally_dead_to_target"] is True
        assert tr_dead_param["source_trace"]["total_derivative"] == pytest.approx(0.0)

        # 2. DO con cirugía sobre estado intermedio SSA (`infra_cost = 0`)
        do_res = oracle.do(
            source_path=str(pkg_dir),
            interventions={"infra_cost": 0.0},
            target_id="net_mrr",
        )
        assert do_res["status"] == "OK"
        assert do_res["seal"] == "PROVEN"
        assert do_res["intervened_target_value"] == pytest.approx(900.0)
        assert any(e["target"] == "infra_cost" for e in do_res["mutilated_edges"])

        # 3. PROVE sobre submodelo simbólico (`PROVEN`) y submodelo con bucle L2 (`SAMPLED_VERIFIED`)
        pr_sym = oracle.prove(
            source_path=str(pkg_dir),
            invariant_expr="net_mrr <= gross_mrr",
            domain_bounds={"seats": [5.0, 100.0], "seat_price": [10.0, 100.0], "discount_rate": [0.0, 0.4]},
        )
        assert pr_sym["status"] == "PROVEN"
        assert pr_sym["seal"] == "PROVEN"

        pr_l2 = oracle.prove(
            source_path=str(pkg_dir),
            invariant_expr="ending_active_seats <= initial_seats",
            domain_bounds={
                "initial_seats": [10.0, 60.0],
                "seat_price": [20.0, 80.0],
                "monthly_churn": [0.01, 0.20],
                "horizon_months": [3.0, 6.0],
            },
        )
        assert pr_l2["status"] == "SAMPLED_VERIFIED"
        assert pr_l2["seal"] == "SAMPLED_VERIFIED"

        # 4. Compuerta epistémica P5 en `cge_solve`: rechazo honesto si no se declaran `modifiable_controls`
        sol_ambig = oracle.solve(source_path=str(pkg_dir), target_id="net_mrr", sense="max")
        assert sol_ambig["status"] == "UNSUPPORTED"
        assert sol_ambig["error_code"] == "AMBIGUOUS_LEAF_ROLES_P5"
        assert "seat_price" in sol_ambig["candidate_leaves"]
        assert "vat_rate" in sol_ambig["candidate_leaves"]

        # Rechazar intento de promover un nodo calculado (`gross_mrr`) como control hoja
        sol_bad_ctrl = oracle.solve(
            source_path=str(pkg_dir),
            target_id="net_mrr",
            modifiable_controls=["gross_mrr"],
        )
        assert sol_bad_ctrl["status"] == "ERROR"
        assert sol_bad_ctrl["error_code"] == "INVALID_MODIFIABLE_CONTROL"

        # 5. `cge_solve` Zero-Config pasando `modifiable_controls` y `control_bounds` explícitos
        sol_ok = oracle.solve(
            source_path=str(pkg_dir),
            target_id="net_mrr",
            sense="max",
            modifiable_controls=["seat_price", "discount_rate"],
            control_bounds={"seat_price": [20.0, 85.0], "discount_rate": [0.0, 0.25]},
        )
        assert sol_ok["status"] == "OPTIMAL"
        assert sol_ok["seal"] == "PROVEN"
        assert sol_ok["controls"]["seat_price"] == pytest.approx(85.0, rel=1e-4)
        assert sol_ok["controls"]["discount_rate"] == pytest.approx(0.0, abs=1e-5)
        # net_mrr* = 20 * 85 * 1.0 - (100 + 6*20) = 1700 - 220 = 1480
        assert sol_ok["objective_value"] == pytest.approx(1480.0, rel=1e-4)

        # 6. `cge_solve` Zero-Config sobre el submodelo L2 (`simulate_cohort_ltv`) con `_blackbox_callables`
        sol_l2 = oracle.solve(
            source_path=str(pkg_dir),
            target_id="cumulative_net_cashflow",
            sense="max",
            modifiable_controls=["seat_price"],
            control_bounds={"seat_price": [25.0, 80.0]},
        )
        assert sol_l2["status"] == "OPTIMAL"
        assert sol_l2["seal"] == "SAMPLED_VERIFIED"
        assert sol_l2["controls"]["seat_price"] == pytest.approx(80.0, rel=1e-3)

        # 7. Conflicto duro entre `control_bounds` solicitados y guarda AST (`if seat_price <= 0: raise`)
        sol_conflict = oracle.solve(
            source_path=str(pkg_dir),
            target_id="net_mrr",
            modifiable_controls=["seat_price"],
            control_bounds={"seat_price": [-50.0, -10.0]},
        )
        assert sol_conflict["status"] == "INFEASIBLE"
        assert "ast_precondition_bounds:seat_price" in sol_conflict["iis_constraints"]


class TestLayer2SidecarContractAndMismatch:
    def test_actordb_yaml_coevolution_iis_and_contract_ast_mismatch(self, tmp_path: Path):
        pkg_dir = _build_unseen_saas_package(tmp_path / "saas_with_yaml")
        _write_sidecar(pkg_dir, max_price=90.0, min_profit_floor=300.0)

        oracle = ActorDBOracle(allow_auto_discovery=False)

        # 1. Con `actordb.yaml`, `cge_solve` funciona directamente sin `modifiable_controls`
        sol = oracle.solve(source_path=str(pkg_dir))
        assert sol["status"] == "OPTIMAL"
        assert sol["seal"] == "PROVEN"
        assert sol["controls"]["seat_price"] == pytest.approx(90.0, rel=1e-4)
        assert sol["controls"]["discount_rate"] == pytest.approx(0.0, abs=1e-5)
        assert sol["objective_value"] == pytest.approx(20 * 90.0 - 220.0, rel=1e-4)

        # 2. Piso de beneficio imposible en `actordb.yaml` -> `INFEASIBLE` con `c_profit_floor` en IIS
        _write_sidecar(pkg_dir, max_price=90.0, min_profit_floor=5000.0)
        sol_infeas = oracle.solve(source_path=str(pkg_dir))
        assert sol_infeas["status"] == "INFEASIBLE"
        assert sol_infeas["cache_hit"] is False
        assert "c_profit_floor" in sol_infeas["iis_constraints"]

        # 3. Divergencia AST <-> YAML: control renombrado inexistente -> `CONTRACT_AST_MISMATCH`
        (pkg_dir / "actordb.yaml").write_text(
            '''version: "1.0"
submodels:
  evaluate_plan:
    entrypoint: "evaluate_plan"
    target: "net_mrr"
    controls:
      non_existent_price_lever:
        bounds: [10.0, 90.0]
''',
            encoding="utf-8",
        )
        err_ctrl = oracle.trace(source_path=str(pkg_dir))
        assert err_ctrl["status"] == "ERROR"
        assert err_ctrl["error_code"] == "CONTRACT_AST_MISMATCH"
        assert "non_existent_price_lever" in err_ctrl["missing_identifiers"]

        # 4. Divergencia AST <-> YAML: estado calculado declarado erróneamente como `control`
        (pkg_dir / "actordb.yaml").write_text(
            '''version: "1.0"
submodels:
  evaluate_plan:
    entrypoint: "evaluate_plan"
    target: "net_mrr"
    controls:
      gross_mrr:
        bounds: [100.0, 2000.0]
''',
            encoding="utf-8",
        )
        with pytest.raises(ContractASTMismatchError):
            GenericPackageAdapter(str(pkg_dir))

        # 5. Divergencia AST <-> YAML: invariante referencia actor inexistente `ghost_metric`
        (pkg_dir / "actordb.yaml").write_text(
            '''version: "1.0"
submodels:
  evaluate_plan:
    entrypoint: "evaluate_plan"
    target: "net_mrr"
    controls: [seat_price]
    constraints:
      c_bad: "ghost_metric >= 10.0"
''',
            encoding="utf-8",
        )
        err_inv = oracle.prove(source_path=str(pkg_dir), constraint_id="c_bad")
        assert err_inv["status"] == "ERROR"
        assert err_inv["error_code"] == "CONTRACT_AST_MISMATCH"
        assert "ghost_metric" in err_inv["missing_identifiers"]


class TestGenericPackageCausalDiffAndGit:
    def test_multi_submodel_diff_across_python_and_sidecar_mutations(self, tmp_path: Path):
        v1 = _build_unseen_saas_package(
            tmp_path / "pkg_v1",
            infra_per_seat=6.0,
            include_dead_local=False,
            churn_multiplier=1.0,
        )
        _write_sidecar(v1, max_price=90.0, min_profit_floor=400.0)

        # Copia idéntica -> PASS y semánticamente equivalente
        v1_copy = tmp_path / "pkg_v1_copy"
        shutil.copytree(v1, v1_copy)
        rep_eq = diff_sources(v1, v1_copy)
        assert rep_eq.verdict == DiffVerdict.PASS.value
        assert rep_eq.is_semantically_equivalent is True

        # v2 introduce simultáneamente:
        # - En Python (`evaluator.py`): nodo muerto `vanity_badge_score` y mayor coste en `tiers.py` (`infra_per_seat=12.0`)
        # - En Python (`cohort_sim.py`): empeora churn (`churn_multiplier=2.5`) en el segundo submodelo L2
        # - En `actordb.yaml`: relaja cota superior de `seat_price` (90 -> 95) y debilita `c_profit_floor` (400 -> 200)
        v2 = _build_unseen_saas_package(
            tmp_path / "pkg_v2",
            infra_per_seat=12.0,
            include_dead_local=True,
            churn_multiplier=2.5,
        )
        _write_sidecar(v2, max_price=95.0, min_profit_floor=200.0)

        rep_reg = diff_sources(v1, v2)
        assert rep_reg.verdict == DiffVerdict.FAIL.value
        assert rep_reg.has_regressions is True
        assert set(rep_reg.summary["changed_submodels"]) == {"evaluate_plan", "simulate_cohort_ltv"}
        inv_codes = {r["code"] for r in rep_reg.invariant_regressions}
        assert "HARD_CONSTRAINT_WEAKENED" in inv_codes
        assert "CONTROL_BOUNDS_WEAKENED" in inv_codes
        assert any(d["node"] == "vanity_badge_score" for d in rep_reg.dead_nodes["added_dead_nodes"])

    def test_git_revisions_diff_with_actordb_yaml(self, tmp_path: Path):
        repo = tmp_path / "saas_git_repo"
        repo.mkdir()
        _git(repo, "init")
        _git(repo, "config", "user.email", "ci@actordb.test")
        _git(repo, "config", "user.name", "ActorDB CI")

        pkg = _build_unseen_saas_package(repo / "billing_pkg", infra_per_seat=6.0, include_dead_local=False)
        _write_sidecar(pkg, max_price=90.0, min_profit_floor=400.0)
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "feat: baseline billing package and actordb.yaml")

        # Commit 2: PR debilita restricción dura en actordb.yaml e inyecta coste mayor en helper tiers.py
        _build_unseen_saas_package(pkg, infra_per_seat=18.0, include_dead_local=False)
        _write_sidecar(pkg, max_price=90.0, min_profit_floor=150.0)
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "chore: weaken profit floor and raise infra cost")

        rep = diff_git_revisions(repo, base_ref="HEAD~1", head_ref="HEAD", subpath="billing_pkg")
        assert rep.verdict == DiffVerdict.FAIL.value
        assert any(r["code"] == "HARD_CONSTRAINT_WEAKENED" for r in rep.invariant_regressions)
        assert rep.policy_optimality["is_regression"] is True

        # También funciona contra WORKTREE cuando se revierte la regresión en disco antes de commitear
        _build_unseen_saas_package(pkg, infra_per_seat=6.0, include_dead_local=False)
        _write_sidecar(pkg, max_price=90.0, min_profit_floor=400.0)
        rep_wt = CausalDiffEngine().diff_git(pkg, base_ref="HEAD~1", head_ref="WORKTREE")
        assert rep_wt.verdict == DiffVerdict.PASS.value
        assert rep_wt.is_semantically_equivalent is True


class TestBackwardCompatibilityGuards:
    def test_golden_case_and_specialized_adapters_remain_untouched(self):
        repo_eng = RepositoryCausalEngine()
        golden_dir = WORKSPACE_ROOT / "golden_case"
        # `golden_case` (híbrido CSV + `*_BOUNDS`) sigue delegándose intacto a `ASTIngestPipeline`
        assert repo_eng.resolve_adapter_source(str(golden_dir)) is None

        oracle = ActorDBOracle(allow_auto_discovery=False)
        tr_golden = oracle.trace(source_path=str(golden_dir))
        assert tr_golden["status"] == "OK"
        assert "brand_awareness" in tr_golden["dead_node_ids"]

        # `supplychainpy_mini` conserva su adaptador especializado de `ADAPTER_REGISTRY`
        sc_spec = repo_eng.resolve_adapter_source("supplychainpy_mini")
        assert sc_spec is not None
        assert sc_spec["repo_id"] == "supplychainpy_mini"
        assert sc_spec["is_generic_package"] is False


def _extract_mcp_payload(call_result: Any) -> dict[str, Any]:
    assert call_result.isError is False, f"MCP tool returned protocol error: {call_result}"
    if getattr(call_result, "structuredContent", None) and isinstance(call_result.structuredContent, dict):
        sc = call_result.structuredContent
        if "result" in sc and isinstance(sc["result"], dict) and "status" in sc["result"]:
            return sc["result"]
        if "status" in sc:
            return sc
    assert call_result.content, "MCP CallToolResult contained empty content"
    return json.loads(call_result.content[0].text)


class TestMCPStdioWireIntegrationOnGenericPackage:
    def test_stdio_wire_protocol_on_examples_saas_package(self):
        saas_pkg = WORKSPACE_ROOT / "examples" / "saas_unit_economics"
        assert saas_pkg.is_dir()

        async def _wire_scenario() -> None:
            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "cge.mcp", "--transport", "stdio", "--no-auto-discovery"],
                cwd=str(WORKSPACE_ROOT),
                env={**os.environ, "PYTHONUTF8": "1"},
            )
            async with (
                stdio_client(params) as (read_stream, write_stream),
                ClientSession(read_stream, write_stream) as session,
            ):
                await session.initialize()

                # 1. Validar esquema JSON expuesto por FastMCP en `cge_solve`
                tools_res = await session.list_tools()
                solve_tool = next(t for t in tools_res.tools if t.name == "cge_solve")
                props = solve_tool.inputSchema.get("properties", {})
                assert "modifiable_controls" in props
                assert "control_bounds" in props

                # 2. `cge_trace` Zero-Config sobre `examples/saas_unit_economics`
                tr = _extract_mcp_payload(
                    await session.call_tool(
                        "cge_trace",
                        {"source_path": str(saas_pkg), "target_id": "contribution_margin"},
                    )
                )
                assert tr["status"] == "OK"
                dead_ids = set(tr["dead_node_ids"])
                assert {"nps_vanity_index", "legacy_crm_score", "partner_rebate_rate"}.issubset(dead_ids)

                # 3. `cge_do` sobre trampa T1-B (`partner_rebate_rate`) y parámetro muerto (`legacy_crm_score`)
                do_cancel = _extract_mcp_payload(
                    await session.call_tool(
                        "cge_do",
                        {
                            "source_path": str(saas_pkg),
                            "target_id": "contribution_margin",
                            "interventions": {"partner_rebate_rate": 0.25, "legacy_crm_score": 10.0},
                        },
                    )
                )
                assert do_cancel["status"] == "OK"
                assert do_cancel["causal_effect"] == pytest.approx(0.0, abs=1e-9)
                assert {"partner_rebate_rate", "legacy_crm_score"}.issubset(
                    set(do_cancel["intervened_on_disconnected_actors"])
                )

                # 4. Compuerta epistémica P5 en `cge_solve` cuando no hay `actordb.yaml` ni `modifiable_controls`
                sol_p5 = _extract_mcp_payload(
                    await session.call_tool(
                        "cge_solve",
                        {"source_path": str(saas_pkg), "target_id": "contribution_margin", "sense": "max"},
                    )
                )
                assert sol_p5["status"] == "UNSUPPORTED"
                assert sol_p5["error_code"] == "AMBIGUOUS_LEAF_ROLES_P5"
                assert "seat_price" in sol_p5["candidate_leaves"]

                # 5. `cge_solve` declarando `modifiable_controls` explícitos (y validando `assert contribution_margin >= 400.0`)
                sol_max = _extract_mcp_payload(
                    await session.call_tool(
                        "cge_solve",
                        {
                            "source_path": str(saas_pkg),
                            "target_id": "contribution_margin",
                            "sense": "max",
                            "modifiable_controls": ["seat_price", "annual_discount"],
                        },
                    )
                )
                assert sol_max["status"] == "OPTIMAL"
                assert sol_max["seal"] == "PROVEN"
                assert sol_max["controls"]["seat_price"] == pytest.approx(150.0, rel=1e-4)
                assert sol_max["controls"]["annual_discount"] == pytest.approx(0.0, abs=1e-5)
                assert sol_max["objective_value"] == pytest.approx(5100.0, rel=1e-4)

                sol_min = _extract_mcp_payload(
                    await session.call_tool(
                        "cge_solve",
                        {
                            "source_path": str(saas_pkg),
                            "target_id": "contribution_margin",
                            "sense": "min",
                            "modifiable_controls": ["seat_price", "annual_discount"],
                        },
                    )
                )
                assert sol_min["status"] == "OPTIMAL"
                assert sol_min["objective_value"] == pytest.approx(400.0, abs=1e-3)

                # 6. Submodelo imperativo L2 (`simulate_cohort_retention`) vía selector `:submodel`
                do_l2 = _extract_mcp_payload(
                    await session.call_tool(
                        "cge_do",
                        {
                            "source_path": f"{saas_pkg}:simulate_cohort_retention",
                            "target_id": "cumulative_margin",
                            "interventions": {"monthly_churn": 0.01},
                        },
                    )
                )
                assert do_l2["status"] == "OK"
                assert do_l2["seal"] == "SAMPLED_VERIFIED"
                assert do_l2["causal_effect"] > 0.0

                # 7. `cge_prove` simbólico sobre identidad algebraica de cancelación de canal
                prv = _extract_mcp_payload(
                    await session.call_tool(
                        "cge_prove",
                        {
                            "source_path": str(saas_pkg),
                            "invariant_expr": "net_channel_effect == 0.0",
                        },
                    )
                )
                assert prv["status"] == "PROVEN"
                assert prv["seal"] == "PROVEN"

        anyio.run(_wire_scenario)

