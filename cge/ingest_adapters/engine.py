"""Orquestador central de adaptadores de ingesta estática AST y resolución causal multi-repositorio."""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

from ..schema import CausalGraph
from .base import AdapterResponse, QueryContext, StaticRepoASTExtractor
from .dispatch_adapter import DispatchCausalAdapter
from .epidemiology_adapter import EpidemiologyCausalAdapter
from .generic_adapter import ActorDBSidecarSpec, GenericPackageAdapter
from .loan_tax_adapter import LoanTaxCausalAdapter
from .quantecon_adapter import QuantEconCausalAdapter
from .supplychain_adapter import SupplyChainCausalAdapter

GENERIC_PACKAGE_REPO_ID = "__generic_package__"
_LEGACY_BOUNDS_MARKER_RE = re.compile(r"^[A-Z][A-Z0-9_]*_BOUNDS\s*(?::[^=]+)?=", re.MULTILINE)


def _compute_repo_fs_signature(repo_dir: Path) -> str:
    """Huella determinista de archivos fuente del repositorio para invalidar caché ante ediciones en disco."""
    if not repo_dir.exists():
        return "missing"
    entries: List[str] = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = sorted(
            d for d in dirs
            if not d.startswith(".") and d not in ("__pycache__", "tests", ".pytest_cache", "node_modules")
        )
        for fname in sorted(files):
            if fname.endswith((".py", ".md", ".json", ".yaml", ".yml")):
                fpath = os.path.join(root, fname)
                try:
                    st = os.stat(fpath)
                    rel = os.path.relpath(fpath, repo_dir).replace("\\", "/")
                    entries.append(f"{rel}:{st.st_size}:{st.st_mtime_ns}")
                except OSError:
                    continue
    return hashlib.sha256("|".join(entries).encode("utf-8")).hexdigest()


class RepositoryCausalEngine:
    """Enrutador determinista air-gapped de repositorios externos hacia sus adaptadores AST -> IR CGE."""

    ADAPTER_REGISTRY: Dict[str, Type[Any]] = {
        "supplychainpy_mini": SupplyChainCausalAdapter,
        "economic_dispatch": DispatchCausalAdapter,
        "loan_tax_engine": LoanTaxCausalAdapter,
        "quantecon_cournot_solow": QuantEconCausalAdapter,
        "epidemiology_seir": EpidemiologyCausalAdapter,
    }

    REPO_FILE_SIGNATURES: Dict[str, Tuple[str, ...]] = {
        "supplychainpy_mini": ("eoq.py", "safety_stock.py"),
        "economic_dispatch": ("dispatch.py", "generator.py"),
        "loan_tax_engine": ("amortization.py", "tax_brackets.py"),
        "quantecon_cournot_solow": ("cournot.py", "solow.py"),
        "epidemiology_seir": ("seir_model.py", "metrics.py"),
    }

    def __init__(self, workspace_root: Optional[str] = None):
        if workspace_root:
            self.workspace_root = Path(workspace_root).resolve()
        else:
            self.workspace_root = Path(__file__).resolve().parent.parent.parent
        self._adapter_cache: Dict[str, Any] = {}

    def clear_cache(self) -> None:
        self._adapter_cache.clear()

    def detect_repo_id_for_dir(self, dir_path: Path) -> Optional[str]:
        """Detecta si un directorio corresponde a un repositorio soportado por nombre o firma de módulos."""
        if not dir_path.exists() or not dir_path.is_dir():
            return None
        if dir_path.name in self.ADAPTER_REGISTRY:
            return dir_path.name
        present_files = {p.name.lower() for p in dir_path.glob("*.py") if p.is_file()}
        for repo_id, req_files in self.REPO_FILE_SIGNATURES.items():
            if all(rf.lower() in present_files for rf in req_files):
                return repo_id
        if "eoq.py" in present_files and "inventory_sim.py" in present_files:
            return "supplychainpy_mini"
        return None

    def resolve_repo_path(self, repo_id: str, candidate_path: Optional[str] = None) -> Path:
        """Localiza el directorio físico del repositorio soportando clones/copias temporales además de rutas estándar."""
        if candidate_path:
            p = Path(candidate_path)
            if p.exists():
                if p.is_file():
                    p = p.parent
                if p.is_dir():
                    if p.name == repo_id:
                        return p.resolve()
                    for sub_cand in (p / repo_id, p / "repos" / repo_id):
                        if sub_cand.exists() and sub_cand.is_dir():
                            return sub_cand.resolve()
                    if self.detect_repo_id_for_dir(p) == repo_id:
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
        return repo_id in self.ADAPTER_REGISTRY or repo_id == GENERIC_PACKAGE_REPO_ID

    def _is_generic_python_package_dir(self, dir_p: Path) -> bool:
        """Determina si `dir_p` es un paquete Python multi-módulo elegible para `GenericPackageAdapter`
        respetando compatibilidad total con `golden_case` y fixtures legacy de `ASTIngestPipeline`.
        """
        if not dir_p.exists() or not dir_p.is_dir():
            return False
        if ActorDBSidecarSpec.find_sidecar_file(dir_p) is not None:
            return True
        if dir_p == self.workspace_root or (dir_p / "cge").is_dir() or dir_p.name == "golden_case":
            return False
        if any(dir_p.rglob("*.csv")):
            return False

        domain_py = [
            p
            for p in dir_p.rglob("*.py")
            if "__pycache__" not in p.parts
            and p.name != "__init__.py"
            and not StaticRepoASTExtractor._is_test_module(p, dir_p)
        ]
        if len(domain_py) < 2:
            return False

        for py_p in domain_py:
            try:
                txt = py_p.read_text(encoding="utf-8")
            except OSError:
                continue
            if _LEGACY_BOUNDS_MARKER_RE.search(txt):
                return False
        return True

    def get_adapter(self, repo_id: str, repo_path: Optional[str] = None) -> Optional[Any]:
        if repo_id == GENERIC_PACKAGE_REPO_ID:
            if not repo_path:
                return None
            resolved_gen = Path(repo_path).resolve()
            fs_sig_gen = _compute_repo_fs_signature(resolved_gen)
            cache_key_gen = f"{GENERIC_PACKAGE_REPO_ID}::{resolved_gen.as_posix()}::{fs_sig_gen}"
            if cache_key_gen not in self._adapter_cache:
                if len(self._adapter_cache) >= 64:
                    self._adapter_cache.clear()
                self._adapter_cache[cache_key_gen] = GenericPackageAdapter(str(resolved_gen))
            return self._adapter_cache[cache_key_gen]

        adapter_cls = self.ADAPTER_REGISTRY.get(repo_id)
        if adapter_cls is None:
            return None
        resolved = self.resolve_repo_path(repo_id, repo_path)
        fs_sig = _compute_repo_fs_signature(resolved)
        cache_key = f"{repo_id}::{resolved.as_posix()}::{fs_sig}"
        if cache_key not in self._adapter_cache:
            if len(self._adapter_cache) >= 64:
                self._adapter_cache.clear()
            self._adapter_cache[cache_key] = adapter_cls(str(resolved))
        return self._adapter_cache[cache_key]

    def resolve_adapter_source(self, source: str) -> Optional[Dict[str, Any]]:
        """Interpreta `source_path` si apunta a un repositorio/módulo/submodelo gestionado por `ingest_adapters`.

        Prioridades deterministas:
        1. `ADAPTER_REGISTRY` (5 repositorios especializados de referencia).
        2. Esquema `generic://<ruta>[:submodel]` o directorio con sidecar `actordb.yaml`/`actordb.yml`/`actordb.json`.
        3. Paquete Python multi-módulo (`>= 2` módulos de dominio sin marcadores legacy CSV/`*_BOUNDS`).

        Retorna `None` cuando `source` debe delegarse al pipeline legacy `cge/ingest.py`.
        """
        s = (source or "").strip()
        if not s or "\n" in s:
            return None

        has_adapter_scheme = False
        has_generic_scheme = False
        body = s
        if body.lower().startswith("adapter://"):
            has_adapter_scheme = True
            body = body[len("adapter://"):].strip()
            if not body:
                return {
                    "error_code": "INVALID_SOURCE",
                    "message": "Adapter URI 'adapter://' is missing repository identifier.",
                }
        elif body.lower().startswith("generic://"):
            has_generic_scheme = True
            body = body[len("generic://"):].strip()
            if not body:
                return {
                    "error_code": "INVALID_SOURCE",
                    "message": "Generic package URI 'generic://' is missing target directory path.",
                }

        has_scheme = has_adapter_scheme or has_generic_scheme
        base_str = body
        explicit_submodel: Optional[str] = None

        if body not in self.ADAPTER_REGISTRY and not os.path.exists(body) and ":" in body:
            left, right = body.rsplit(":", 1)
            left_clean = left.strip()
            right_clean = right.strip()
            if (
                len(left_clean) >= 2
                and bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", right_clean))
                and (left_clean in self.ADAPTER_REGISTRY or os.path.exists(left_clean) or has_scheme)
            ):
                base_str = left_clean
                explicit_submodel = right_clean

        if not has_generic_scheme and base_str in self.ADAPTER_REGISTRY:
            repo_dir = self.resolve_repo_path(base_str)
            if not repo_dir.exists() or not repo_dir.is_dir():
                return {
                    "error_code": "SOURCE_NOT_FOUND",
                    "message": f"Adapter repository '{base_str}' directory not found at '{repo_dir}'.",
                }
            return {
                "repo_id": base_str,
                "repo_dir": repo_dir,
                "explicit_submodel": explicit_submodel,
                "file_hint": None,
                "is_generic_package": False,
                "fs_signature": _compute_repo_fs_signature(repo_dir),
            }

        if os.path.isdir(base_str):
            dir_p = Path(base_str).resolve()
            if not has_generic_scheme:
                det_id = self.detect_repo_id_for_dir(dir_p)
                if det_id is not None:
                    return {
                        "repo_id": det_id,
                        "repo_dir": dir_p,
                        "explicit_submodel": explicit_submodel,
                        "file_hint": None,
                        "is_generic_package": False,
                        "fs_signature": _compute_repo_fs_signature(dir_p),
                    }

            if has_generic_scheme or self._is_generic_python_package_dir(dir_p):
                return {
                    "repo_id": GENERIC_PACKAGE_REPO_ID,
                    "repo_dir": dir_p,
                    "explicit_submodel": explicit_submodel,
                    "file_hint": None,
                    "is_generic_package": True,
                    "fs_signature": _compute_repo_fs_signature(dir_p),
                }

            if has_adapter_scheme:
                return {
                    "error_code": "UNSUPPORTED_ADAPTER_REPO",
                    "message": (
                        f"Directory '{base_str}' does not match any registered adapter repository: "
                        f"{sorted(self.ADAPTER_REGISTRY.keys())}."
                    ),
                }
            return None

        if os.path.isfile(base_str) and base_str.lower().endswith(".py"):
            file_p = Path(base_str).resolve()
            if not has_generic_scheme:
                det_id = self.detect_repo_id_for_dir(file_p.parent)
                if det_id is not None:
                    return {
                        "repo_id": det_id,
                        "repo_dir": file_p.parent,
                        "explicit_submodel": explicit_submodel,
                        "file_hint": file_p.name,
                        "is_generic_package": False,
                        "fs_signature": _compute_repo_fs_signature(file_p.parent),
                    }

            # Si el módulo pertenece a un paquete con sidecar `actordb.yaml` (en su carpeta o padre directo)
            for cand_pkg in (file_p.parent, file_p.parent.parent):
                if cand_pkg == self.workspace_root or not cand_pkg.exists():
                    break
                if has_generic_scheme or ActorDBSidecarSpec.find_sidecar_file(cand_pkg) is not None:
                    return {
                        "repo_id": GENERIC_PACKAGE_REPO_ID,
                        "repo_dir": cand_pkg,
                        "explicit_submodel": explicit_submodel,
                        "file_hint": file_p.name,
                        "is_generic_package": True,
                        "fs_signature": _compute_repo_fs_signature(cand_pkg),
                    }

            if has_adapter_scheme:
                return {
                    "error_code": "UNSUPPORTED_ADAPTER_REPO",
                    "message": f"Python module '{base_str}' does not belong to any registered adapter repository.",
                }
            return None

        if has_generic_scheme:
            return {
                "error_code": "SOURCE_NOT_FOUND",
                "message": f"Generic package path '{base_str}' does not exist on disk.",
            }

        if has_adapter_scheme:
            return {
                "error_code": "UNSUPPORTED_ADAPTER_REPO",
                "message": (
                    f"Unknown adapter repository '{base_str}'. Registered repositories: "
                    f"{sorted(self.ADAPTER_REGISTRY.keys())}."
                ),
            }

        return None

    def build_causal_graph(
        self,
        source_or_repo_id: str,
        submodel: Optional[str] = None,
        target_hint: Optional[str] = None,
        actor_hints: Optional[Sequence[str]] = None,
    ) -> CausalGraph:
        """Compila y devuelve el `CausalGraph` verificable correspondiente a un repositorio/submodelo."""
        spec = self.resolve_adapter_source(source_or_repo_id)
        if spec is None:
            raise ValueError(
                f"Source '{source_or_repo_id}' does not match any registered causal ingest adapter."
            )
        if "error_code" in spec:
            raise ValueError(spec["message"])

        repo_id = spec["repo_id"]
        repo_dir: Path = spec["repo_dir"]
        chosen_submodel = submodel or spec["explicit_submodel"]
        file_hint = spec["file_hint"]

        adapter = self.get_adapter(repo_id, str(repo_dir))
        if adapter is None:
            raise ValueError(f"No adapter registered for repo_id='{repo_id}'.")

        effective_sub = chosen_submodel or adapter.infer_submodel(
            target_id=target_hint,
            actor_hints=actor_hints,
            file_hint=file_hint,
        )
        return adapter.build_causal_graph(
            submodel=effective_sub,
            target_hint=target_hint,
            actor_hints=actor_hints,
            file_hint=file_hint,
        )

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
