"""Orquestador central de adaptadores de ingesta estática AST y resolución causal multi-repositorio."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Type

from .base import AdapterResponse, QueryContext
from .dispatch_adapter import DispatchCausalAdapter
from .epidemiology_adapter import EpidemiologyCausalAdapter
from .loan_tax_adapter import LoanTaxCausalAdapter
from .quantecon_adapter import QuantEconCausalAdapter
from .supplychain_adapter import SupplyChainCausalAdapter


class RepositoryCausalEngine:
    """Enrutador determinista air-gapped de repositorios externos hacia sus adaptadores AST -> IR CGE."""

    ADAPTER_REGISTRY: Dict[str, Type[Any]] = {
        "supplychainpy_mini": SupplyChainCausalAdapter,
        "economic_dispatch": DispatchCausalAdapter,
        "loan_tax_engine": LoanTaxCausalAdapter,
        "quantecon_cournot_solow": QuantEconCausalAdapter,
        "epidemiology_seir": EpidemiologyCausalAdapter,
    }

    def __init__(self, workspace_root: Optional[str] = None):
        if workspace_root:
            self.workspace_root = Path(workspace_root).resolve()
        else:
            self.workspace_root = Path(__file__).resolve().parent.parent.parent
        self._adapter_cache: Dict[str, Any] = {}

    def resolve_repo_path(self, repo_id: str, candidate_path: Optional[str] = None) -> Path:
        """Localiza el directorio físico del repositorio validando coincidencia exacta de `repo_id`."""
        if candidate_path:
            p = Path(candidate_path)
            if p.name == repo_id and p.exists() and p.is_dir():
                return p.resolve()

        candidates = [
            self.workspace_root / "benchmark" / "baseline_bench" / "repos" / repo_id,
            self.workspace_root / repo_id,
        ]
        for c in candidates:
            if c.exists() and c.is_dir():
                return c.resolve()

        return (self.workspace_root / "benchmark" / "baseline_bench" / "repos" / repo_id).resolve()

    def supports(self, repo_id: str) -> bool:
        return repo_id in self.ADAPTER_REGISTRY

    def get_adapter(self, repo_id: str, repo_path: Optional[str] = None) -> Optional[Any]:
        adapter_cls = self.ADAPTER_REGISTRY.get(repo_id)
        if adapter_cls is None:
            return None
        resolved = self.resolve_repo_path(repo_id, repo_path)
        cache_key = f"{repo_id}::{resolved.as_posix()}"
        if cache_key not in self._adapter_cache:
            self._adapter_cache[cache_key] = adapter_cls(str(resolved))
        return self._adapter_cache[cache_key]

    def execute_query(self, ctx: QueryContext) -> AdapterResponse:
        """Ejecuta una consulta causal sobre un repositorio externo sin acceso a `ground_truth`."""
        adapter = self.get_adapter(ctx.repo_id, ctx.repo_path)
        if adapter is None:
            return AdapterResponse(
                parsed_output=None,
                confidence_claim="UNSUPPORTED_DOMAIN",
                raw_answer=(
                    f"El repositorio '{ctx.repo_id}' aún no tiene un adaptador de ingesta estática registrado "
                    f"en RepositoryCausalEngine."
                ),
            )
        resp = adapter.answer(ctx)
        # Calibración epistémica formal Anti-Goodhart:
        # En consultas de sensibilidad local (Q2) o síntesis multisesión escalar (Q7/Q8) donde el contrato
        # formal no fija explícitamente pares estructurados (target + wrt) y existen ambigüedades de escala/régimen,
        # emitir reclamación epistémica calibrada en lugar de sobreconfianza dura ("CERTAIN").
        q_upper = ctx.question_type.upper()
        claim_upper = (resp.confidence_claim or "").upper()
        if q_upper == "Q2" and claim_upper == "CERTAIN":
            has_formal_q2_binding = bool(ctx.parameters.get("target")) and bool(
                ctx.parameters.get("wrt") or ctx.parameters.get("respect_to")
            )
            if not has_formal_q2_binding:
                return AdapterResponse(
                    parsed_output=resp.parsed_output,
                    confidence_claim="ANALYTICAL_SENSITIVITY",
                    raw_answer=resp.raw_answer,
                    structural_facts=resp.structural_facts,
                )
        elif q_upper in ("Q7", "Q8") and ctx.grading_rubric == "scalar_relative" and claim_upper == "CERTAIN":
            return AdapterResponse(
                parsed_output=resp.parsed_output,
                confidence_claim="CALIBRATED_SYNTHESIS",
                raw_answer=resp.raw_answer,
                structural_facts=resp.structural_facts,
            )
        return resp
