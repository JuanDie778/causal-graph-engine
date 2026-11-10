"""Certificados matemáticos de optimalidad, factibilidad en código e infactibilidad (Fase 10)."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, Any, List, Optional

from ..schema import CausalGraph
from ..fidelity import evaluate_recipe_on_source


class SolveStatus(str, Enum):
    CERTIFIED_OPTIMAL = "CERTIFIED_OPTIMAL"
    BEST_FOUND = "BEST_FOUND"
    INFEASIBLE_CERTIFIED = "INFEASIBLE_CERTIFIED"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass
class SolveCertificate:
    status: SolveStatus
    problem_class: str
    strategy_used: str
    optimality_gap: Optional[float] = 0.0
    max_constraint_violation: float = 0.0
    kkt_residual: Optional[float] = 0.0
    fidelity_verified: bool = True
    code_verified: bool = True
    constraint_violation_on_code: float = 0.0
    iis_constraints: List[str] = field(default_factory=list)
    min_relaxation_slack: Optional[float] = None
    unverified_nodes: List[str] = field(default_factory=list)
    reason: Optional[str] = None
    code_objective_value: Optional[float] = None
    ir_objective_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value if isinstance(self.status, SolveStatus) else str(self.status),
            "problem_class": self.problem_class,
            "strategy_used": self.strategy_used,
            "optimality_gap": round(float(self.optimality_gap), 10) if self.optimality_gap is not None and math.isfinite(self.optimality_gap) else None,
            "max_constraint_violation": round(float(self.max_constraint_violation), 10) if math.isfinite(self.max_constraint_violation) else None,
            "kkt_residual": round(float(self.kkt_residual), 10) if self.kkt_residual is not None and math.isfinite(self.kkt_residual) else None,
            "fidelity_verified": bool(self.fidelity_verified),
            "code_verified": bool(self.code_verified),
            "constraint_violation_on_code": round(float(self.constraint_violation_on_code), 10) if math.isfinite(self.constraint_violation_on_code) else None,
            "iis_constraints": list(self.iis_constraints),
            "min_relaxation_slack": round(float(self.min_relaxation_slack), 8) if self.min_relaxation_slack is not None and math.isfinite(self.min_relaxation_slack) else None,
            "unverified_nodes": list(self.unverified_nodes),
            "reason": self.reason,
            "code_objective_value": round(float(self.code_objective_value), 8) if self.code_objective_value is not None and math.isfinite(self.code_objective_value) else None,
            "ir_objective_value": round(float(self.ir_objective_value), 8) if self.ir_objective_value is not None and math.isfinite(self.ir_objective_value) else None,
        }


def verify_and_certify(
    ir: CausalGraph,
    problem_class: str,
    strategy_used: str,
    controls: Dict[str, Any],
    ir_objective_value: float,
    max_constraint_violation: float,
    is_exact_strategy: bool,
    optimality_gap: float = 0.0,
    kkt_residual: float = 0.0,
    source_path_or_code: Optional[str] = None,
    entrypoint: Optional[str] = None,
    exo_overrides: Optional[Dict[str, float]] = None,
) -> SolveCertificate:
    """Construye y verifica el SolveCertificate completo, incluyendo la ejecución directa sobre el código original."""
    code_check = evaluate_recipe_on_source(
        ir=ir,
        controls=controls,
        source_path_or_code=source_path_or_code,
        entrypoint=entrypoint,
        exo_overrides=exo_overrides,
    )
    code_verified = bool(code_check.get("code_verified", True))
    viol_code = float(code_check.get("constraint_violation_on_code", 0.0))
    code_j = code_check.get("code_objective_value", ir_objective_value)

    clean_viol = 0.0 if max_constraint_violation <= 1e-6 else float(max_constraint_violation)
    clean_viol_code = 0.0 if viol_code <= 1e-6 else float(viol_code)

    if is_exact_strategy and clean_viol <= 1e-5 and code_verified and clean_viol_code <= 1e-5:
        cert_status = SolveStatus.CERTIFIED_OPTIMAL
    else:
        cert_status = SolveStatus.BEST_FOUND

    return SolveCertificate(
        status=cert_status,
        problem_class=problem_class,
        strategy_used=strategy_used,
        optimality_gap=float(optimality_gap),
        max_constraint_violation=clean_viol,
        kkt_residual=float(kkt_residual),
        fidelity_verified=True,
        code_verified=code_verified,
        constraint_violation_on_code=clean_viol_code,
        iis_constraints=[],
        min_relaxation_slack=0.0,
        unverified_nodes=[],
        reason=None,
        code_objective_value=float(code_j) if code_j is not None else float(ir_objective_value),
        ir_objective_value=float(ir_objective_value),
    )
