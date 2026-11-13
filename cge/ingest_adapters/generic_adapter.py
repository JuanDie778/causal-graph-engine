"""Adaptador universal en 2 capas (`GenericPackageAdapter` + sidecar declarativo `actordb.yaml`).

Capa 1 (Zero-Config):
  Compila estáticamente cualquier paquete Python multi-módulo no visto (resolviendo imports
  relativos/locales de helpers y constantes, dataclasses, guardas `raise ValueError` / `assert`,
  ecuaciones intermedias SSA y retornos escalares o diccionarios de métricas) en un `CausalGraph`
  verificable (`PROVEN` para el fragmento simbólico `L0/L1` y `SAMPLED_VERIFIED` mediante
  `_SafeSimulationASTInterpreter` para bucles acotados `L2`, sin usar `exec`/`eval`).
  Por honestidad epistémica (Problemática P5), sin sidecar declarativo todas las hojas de entrada
  se registran como `exogenous` y se exponen en `metadata["unclassified_leaves"]`.

Capa 2 (Co-evolución Declarativa — `actordb.yaml` / `actordb.yml` / `actordb.json`):
  Vincula roles (`controls`, `exogenous`), cotas duras (`bounds`), objetivo (`target`, `sense`)
  e invariantes (`constraints`) validándolos determinísticamente contra el AST vivo. Si cualquier
  identificador del contrato diverge del código Python real, aborta en cerrado con
  `ContractASTMismatchError` (`error_code = "CONTRACT_AST_MISMATCH"`).
"""
from __future__ import annotations

import ast
import json
import keyword
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import sympy as sp
import yaml

from ..ingest import ASTFunctionToSymPy
from ..schema import CausalGraph, Constraint
from .base import (
    ExtractedFunctionSpec,
    StaticRepoASTExtractor,
    build_verified_causal_graph,
    make_verified_node,
)

SIDECAR_FILENAMES: Tuple[str, ...] = ("actordb.yaml", "actordb.yml", "actordb.json")

_RESERVED_FORMULA_TOKENS: Set[str] = {
    "Abs",
    "And",
    "Eq",
    "False",
    "GreaterThan",
    "LessThan",
    "Max",
    "Min",
    "Ne",
    "Not",
    "Or",
    "Piecewise",
    "StrictGreaterThan",
    "StrictLessThan",
    "True",
    "abs",
    "and",
    "cos",
    "exp",
    "fabs",
    "false",
    "log",
    "log1p",
    "max",
    "min",
    "not",
    "or",
    "pi",
    "pow",
    "sin",
    "sqrt",
    "sum",
    "tanh",
    "true",
}


def _extract_formula_identifiers(expr_str: str) -> List[str]:
    if not expr_str:
        return []
    seen: List[str] = []
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", expr_str):
        tok = m.group(1)
        if tok in _RESERVED_FORMULA_TOKENS or keyword.iskeyword(tok):
            continue
        if tok not in seen:
            seen.append(tok)
    return seen


class ContractASTMismatchError(ValueError):
    """Error determinista emitido cuando `actordb.yaml` contradice o diverge del AST Python real."""

    error_code: str = "CONTRACT_AST_MISMATCH"

    def __init__(
        self,
        message: str,
        *,
        submodel: Optional[str] = None,
        missing_identifiers: Optional[List[str]] = None,
        available_actors: Optional[List[str]] = None,
    ):
        super().__init__(message)
        self.submodel = submodel
        self.missing_identifiers = list(missing_identifiers or [])
        self.available_actors = list(available_actors or [])


@dataclass
class ActorRoleSpec:
    """Especificación declarativa de una hoja (`control` o `exogenous`) en `actordb.yaml`."""

    name: str
    role: str  # "control" | "exogenous"
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None
    default: Optional[float] = None
    data_type: Optional[str] = None


@dataclass
class SubmodelSidecarSpec:
    """Contrato declarativo para un submodelo específico dentro de `actordb.yaml`."""

    name: str
    entrypoint_fn: str
    entrypoint_file: Optional[str] = None
    target: Optional[str] = None
    sense: str = "max"
    controls: Dict[str, ActorRoleSpec] = field(default_factory=dict)
    exogenous: Dict[str, ActorRoleSpec] = field(default_factory=dict)
    extra_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = field(default_factory=dict)
    constraints: List[Constraint] = field(default_factory=list)


@dataclass
class ActorDBSidecarSpec:
    """Contrato declarativo completo cargado desde `actordb.yaml` / `actordb.yml` / `actordb.json`."""

    path: Path
    version: str = "1.0"
    system_id: Optional[str] = None
    default_submodel: Optional[str] = None
    submodels: Dict[str, SubmodelSidecarSpec] = field(default_factory=dict)

    @staticmethod
    def find_sidecar_file(repo_dir: Path) -> Optional[Path]:
        for fname in SIDECAR_FILENAMES:
            cand = repo_dir / fname
            if cand.exists() and cand.is_file():
                return cand
        return None

    @staticmethod
    def _parse_bounds_pair(raw_b: Any, ctx_label: str) -> Tuple[Optional[float], Optional[float]]:
        if raw_b is None:
            return (None, None)
        lb: Optional[float] = None
        ub: Optional[float] = None
        if isinstance(raw_b, (list, tuple)) and len(raw_b) == 2:
            lb = float(raw_b[0]) if raw_b[0] is not None else None
            ub = float(raw_b[1]) if raw_b[1] is not None else None
        elif isinstance(raw_b, dict):
            lo_raw = raw_b.get("min", raw_b.get("lb"))
            hi_raw = raw_b.get("max", raw_b.get("ub"))
            lb = float(lo_raw) if lo_raw is not None else None
            ub = float(hi_raw) if hi_raw is not None else None
        else:
            raise ContractASTMismatchError(
                f"Invalid bounds specification for '{ctx_label}' in actordb sidecar: expected [min, max] or {{min, max}}."
            )
        if lb is not None and ub is not None and lb > ub:
            raise ContractASTMismatchError(
                f"Inverted bounds [{lb}, {ub}] for '{ctx_label}' in actordb sidecar (min > max)."
            )
        return (lb, ub)

    @classmethod
    def _parse_role_group(cls, raw_group: Any, role: str, sub_name: str) -> Dict[str, ActorRoleSpec]:
        out: Dict[str, ActorRoleSpec] = {}
        if raw_group is None:
            return out
        if isinstance(raw_group, list):
            for item in raw_group:
                if isinstance(item, str) and item.strip():
                    aname = item.strip()
                    out[aname] = ActorRoleSpec(name=aname, role=role)
                elif isinstance(item, dict) and "name" in item:
                    aname = str(item["name"]).strip()
                    b_pair = cls._parse_bounds_pair(item.get("bounds"), f"{sub_name}.{aname}") if "bounds" in item else None
                    def_v = float(item["default"]) if item.get("default") is not None else None
                    dt = str(item["data_type"]).strip() if item.get("data_type") else None
                    out[aname] = ActorRoleSpec(name=aname, role=role, bounds=b_pair, default=def_v, data_type=dt)
                else:
                    raise ContractASTMismatchError(
                        f"Invalid entry {item!r} in '{role}' of submodel '{sub_name}'."
                    )
            return out
        if isinstance(raw_group, dict):
            for raw_k, raw_v in raw_group.items():
                aname = str(raw_k).strip()
                if raw_v is None:
                    out[aname] = ActorRoleSpec(name=aname, role=role)
                elif isinstance(raw_v, (list, tuple)):
                    b_pair = cls._parse_bounds_pair(raw_v, f"{sub_name}.{aname}")
                    out[aname] = ActorRoleSpec(name=aname, role=role, bounds=b_pair)
                elif isinstance(raw_v, dict):
                    b_src = raw_v.get("bounds", raw_v if ("min" in raw_v or "max" in raw_v) else None)
                    b_pair = cls._parse_bounds_pair(b_src, f"{sub_name}.{aname}") if b_src is not None else None
                    def_v = float(raw_v["default"]) if raw_v.get("default") is not None else None
                    dt = str(raw_v["data_type"]).strip() if raw_v.get("data_type") else None
                    out[aname] = ActorRoleSpec(name=aname, role=role, bounds=b_pair, default=def_v, data_type=dt)
                else:
                    raise ContractASTMismatchError(
                        f"Invalid role config for '{aname}' in '{role}' of submodel '{sub_name}'."
                    )
            return out
        raise ContractASTMismatchError(
            f"Section '{role}' in submodel '{sub_name}' must be a list or mapping."
        )

    @classmethod
    def _parse_constraints(cls, raw_constrs: Any, sub_name: str) -> List[Constraint]:
        if raw_constrs is None:
            return []
        result: List[Constraint] = []
        if isinstance(raw_constrs, dict):
            for cid, cbody in raw_constrs.items():
                cid_s = str(cid).strip()
                if isinstance(cbody, str):
                    result.append(
                        Constraint(
                            id=cid_s,
                            type="hard",
                            expression=cbody.strip(),
                            description=f"Declarative constraint '{cid_s}' from actordb sidecar.",
                        )
                    )
                elif isinstance(cbody, dict):
                    expr = str(cbody.get("expression", "")).strip()
                    if not expr:
                        raise ContractASTMismatchError(
                            f"Constraint '{cid_s}' in submodel '{sub_name}' is missing 'expression'."
                        )
                    ctype = str(cbody.get("type", "hard")).strip().lower()
                    if ctype not in ("hard", "soft"):
                        raise ContractASTMismatchError(
                            f"Constraint '{cid_s}' in submodel '{sub_name}' has invalid type '{ctype}'."
                        )
                    desc = str(cbody.get("description") or f"Declarative constraint '{cid_s}' from actordb sidecar.")
                    result.append(Constraint(id=cid_s, type=ctype, expression=expr, description=desc))
                else:
                    raise ContractASTMismatchError(
                        f"Invalid constraint specification for '{cid_s}' in submodel '{sub_name}'."
                    )
            return result
        if isinstance(raw_constrs, list):
            for idx, item in enumerate(raw_constrs):
                if isinstance(item, str):
                    result.append(
                        Constraint(
                            id=f"c_sidecar_{idx + 1}",
                            type="hard",
                            expression=item.strip(),
                            description="Declarative constraint from actordb sidecar.",
                        )
                    )
                elif isinstance(item, dict):
                    cid_s = str(item.get("id") or f"c_sidecar_{idx + 1}").strip()
                    expr = str(item.get("expression", "")).strip()
                    if not expr:
                        raise ContractASTMismatchError(
                            f"Constraint '{cid_s}' in submodel '{sub_name}' is missing 'expression'."
                        )
                    ctype = str(item.get("type", "hard")).strip().lower()
                    desc = str(item.get("description") or f"Declarative constraint '{cid_s}' from actordb sidecar.")
                    result.append(Constraint(id=cid_s, type=ctype, expression=expr, description=desc))
                else:
                    raise ContractASTMismatchError(
                        f"Invalid constraint item {item!r} in submodel '{sub_name}'."
                    )
            return result
        raise ContractASTMismatchError(
            f"Constraints in submodel '{sub_name}' must be a mapping or list."
        )

    @classmethod
    def _parse_submodel_block(cls, sub_name: str, raw_sub: Dict[str, Any]) -> SubmodelSidecarSpec:
        if not isinstance(raw_sub, dict):
            raise ContractASTMismatchError(
                f"Submodel '{sub_name}' in actordb sidecar must be a mapping.",
                submodel=sub_name,
            )
        ep_raw = str(raw_sub.get("entrypoint") or sub_name).strip()
        ep_file: Optional[str] = None
        ep_fn = ep_raw
        if ":" in ep_raw:
            left_f, right_fn = ep_raw.rsplit(":", 1)
            ep_file = left_f.strip() or None
            ep_fn = right_fn.strip()

        raw_sense = str(raw_sub.get("sense", "max")).strip().lower()
        if raw_sense not in ("max", "min", "maximize", "minimize"):
            raise ContractASTMismatchError(
                f"Invalid optimization sense '{raw_sense}' in submodel '{sub_name}' (expected 'max' or 'min').",
                submodel=sub_name,
            )
        norm_sense = "min" if raw_sense.startswith("min") else "max"
        tgt = str(raw_sub["target"]).strip() if raw_sub.get("target") else None

        ctrls = cls._parse_role_group(raw_sub.get("controls"), "control", sub_name)
        exos = cls._parse_role_group(raw_sub.get("exogenous"), "exogenous", sub_name)
        overlap = sorted(set(ctrls.keys()) & set(exos.keys()))
        if overlap:
            raise ContractASTMismatchError(
                f"Actor(s) {overlap} declared as both 'controls' and 'exogenous' in submodel '{sub_name}'.",
                submodel=sub_name,
                missing_identifiers=overlap,
            )

        extra_b: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        raw_bounds_map = raw_sub.get("bounds")
        if raw_bounds_map is not None:
            if not isinstance(raw_bounds_map, dict):
                raise ContractASTMismatchError(
                    f"'bounds' in submodel '{sub_name}' must be a mapping of actor_id -> [min, max].",
                    submodel=sub_name,
                )
            for b_k, b_v in raw_bounds_map.items():
                extra_b[str(b_k).strip()] = cls._parse_bounds_pair(b_v, f"{sub_name}.{b_k}")

        raw_c = raw_sub.get("constraints")
        raw_inv = raw_sub.get("invariants")
        constrs: List[Constraint] = []
        if raw_c is not None:
            constrs.extend(cls._parse_constraints(raw_c, sub_name))
        if raw_inv is not None:
            constrs.extend(cls._parse_constraints(raw_inv, sub_name))

        return SubmodelSidecarSpec(
            name=sub_name,
            entrypoint_fn=ep_fn,
            entrypoint_file=ep_file,
            target=tgt,
            sense=norm_sense,
            controls=ctrls,
            exogenous=exos,
            extra_bounds=extra_b,
            constraints=constrs,
        )

    @classmethod
    def load(cls, sidecar_path: Path) -> "ActorDBSidecarSpec":
        raw_text = sidecar_path.read_text(encoding="utf-8")
        try:
            if sidecar_path.suffix.lower() == ".json":
                data = json.loads(raw_text)
            else:
                data = yaml.safe_load(raw_text)
        except Exception as exc:
            raise ContractASTMismatchError(
                f"Failed to parse sidecar file '{sidecar_path.name}': {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ContractASTMismatchError(
                f"Sidecar file '{sidecar_path.name}' must contain a top-level YAML/JSON mapping."
            )

        version = str(data.get("version", "1.0"))
        sys_id = str(data["system_id"]).strip() if data.get("system_id") else None
        def_sub = str(data["default_submodel"]).strip() if data.get("default_submodel") else None

        submodels_map: Dict[str, SubmodelSidecarSpec] = {}
        if "submodels" in data and data["submodels"] is not None:
            if not isinstance(data["submodels"], dict) or not data["submodels"]:
                raise ContractASTMismatchError(
                    f"'submodels' in '{sidecar_path.name}' must be a non-empty mapping."
                )
            for s_name, s_body in data["submodels"].items():
                s_clean = str(s_name).strip()
                submodels_map[s_clean] = cls._parse_submodel_block(s_clean, s_body)
        elif any(k in data for k in ("entrypoint", "controls", "exogenous", "constraints", "invariants", "target", "bounds")):
            ep_default = str(data.get("entrypoint") or "default").strip()
            s_key = ep_default.rsplit(":", 1)[-1] if ":" in ep_default else ep_default
            submodels_map[s_key] = cls._parse_submodel_block(s_key, data)

        if def_sub and submodels_map and def_sub not in submodels_map:
            raise ContractASTMismatchError(
                f"default_submodel '{def_sub}' is not defined in submodels {sorted(submodels_map.keys())}."
            )

        return cls(
            path=sidecar_path,
            version=version,
            system_id=sys_id,
            default_submodel=def_sub,
            submodels=submodels_map,
        )


class GenericPackageAdapter:
    """Adaptador estático universal para paquetes Python multi-módulo con o sin `actordb.yaml`."""

    IS_GENERIC_PACKAGE = True

    def __init__(self, repo_path: str):
        self.repo_path = str(Path(repo_path).resolve())
        self.repo_dir = Path(self.repo_path)
        self.extractor = StaticRepoASTExtractor(self.repo_path, exclude_tests=True).extract()

        sidecar_file = ActorDBSidecarSpec.find_sidecar_file(self.repo_dir)
        self.sidecar: Optional[ActorDBSidecarSpec] = (
            ActorDBSidecarSpec.load(sidecar_file) if sidecar_file is not None else None
        )
        self.REPO_ID = (
            self.sidecar.system_id
            if (self.sidecar and self.sidecar.system_id)
            else f"pkg_{self.repo_dir.name}"
        )

        # Recopilar valores por defecto declarados en @dataclass / clases de configuración del paquete
        self._dataclass_defaults: Dict[str, float] = {}
        for cls_spec in self.extractor.classes.values():
            for f_name, f_val in cls_spec.fields.items():
                if isinstance(f_val, bool):
                    self._dataclass_defaults.setdefault(f_name, 1.0 if f_val else 0.0)
                elif isinstance(f_val, (int, float)) and math.isfinite(float(f_val)):
                    self._dataclass_defaults.setdefault(f_name, float(f_val))

        self._submodel_to_fn: Dict[str, ExtractedFunctionSpec] = {}
        self._ordered_submodels: List[str] = []
        self._discover_and_validate_submodels()

    @staticmethod
    def _has_value_return(fn_node: ast.FunctionDef) -> bool:
        return any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(fn_node))

    def _pkg_rel_file(self, rel_file: str) -> str:
        """Normaliza `rel_file` respecto a `self.repo_dir` evitando duplicar el nombre del paquete."""
        prefix = f"{self.repo_dir.name}/"
        norm = rel_file.replace("\\", "/")
        return norm[len(prefix):] if norm.startswith(prefix) else norm

    @staticmethod
    def _extract_return_dict_key_lines(stmts: Sequence[ast.stmt]) -> Dict[str, int]:
        """Mapea claves literales de `return {...}` a su línea exacta en el AST."""
        key_lines: Dict[str, int] = {}
        for s in stmts:
            for node in ast.walk(s):
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
                    for k_node, v_node in zip(node.value.keys, node.value.values, strict=False):
                        if isinstance(k_node, ast.Constant) and isinstance(k_node.value, str):
                            k_str = str(k_node.value).strip()
                            if k_str:
                                line_v = getattr(
                                    k_node,
                                    "lineno",
                                    getattr(v_node, "lineno", getattr(node, "lineno", 1)),
                                )
                                key_lines[k_str] = int(line_v)
        return key_lines

    def _discover_and_validate_submodels(self) -> None:
        public_fns: Dict[str, ExtractedFunctionSpec] = {
            name: spec
            for name, spec in self.extractor.functions.items()
            if not name.startswith("_") and self._has_value_return(spec.ast_node)
        }
        # Respaldo: métodos públicos de clases si no hubiera funciones top-level
        if not public_fns:
            for cls_spec in self.extractor.classes.values():
                for m_name, m_spec in cls_spec.methods.items():
                    if not m_name.startswith("_") and self._has_value_return(m_spec.ast_node):
                        public_fns[m_name] = m_spec

        if self.sidecar and self.sidecar.submodels:
            # Si el sidecar usó placeholder `"default"` sin `entrypoint` explícito y hay 1+ funciones públicas
            if (
                list(self.sidecar.submodels.keys()) == ["default"]
                and "default" not in public_fns
                and public_fns
            ):
                root_cands = self._select_root_entrypoints(public_fns)
                chosen_fn_name = root_cands[0]
                default_spec = self.sidecar.submodels.pop("default")
                default_spec.name = chosen_fn_name
                default_spec.entrypoint_fn = chosen_fn_name
                self.sidecar.submodels[chosen_fn_name] = default_spec
                self.sidecar.default_submodel = chosen_fn_name

            for sub_name, sub_cfg in self.sidecar.submodels.items():
                ep_fn = sub_cfg.entrypoint_fn
                if ep_fn not in public_fns:
                    raise ContractASTMismatchError(
                        f"Entrypoint function '{ep_fn}' declared in actordb sidecar for submodel '{sub_name}' "
                        f"does not exist in package AST. Available public functions: {sorted(public_fns.keys())}",
                        submodel=sub_name,
                        missing_identifiers=[ep_fn],
                        available_actors=sorted(public_fns.keys()),
                    )
                fn_spec = public_fns[ep_fn]
                if sub_cfg.entrypoint_file:
                    norm_ep_file = sub_cfg.entrypoint_file.replace("\\", "/").lstrip("./")
                    if not fn_spec.rel_file.endswith(norm_ep_file):
                        raise ContractASTMismatchError(
                            f"Entrypoint '{ep_fn}' was found in '{fn_spec.rel_file}', which does not match "
                            f"declared file '{sub_cfg.entrypoint_file}' in submodel '{sub_name}'.",
                            submodel=sub_name,
                            missing_identifiers=[sub_cfg.entrypoint_file],
                        )
                self._submodel_to_fn[sub_name] = fn_spec
                self._ordered_submodels.append(sub_name)

            if self.sidecar.default_submodel and self.sidecar.default_submodel in self._ordered_submodels:
                d_sub = self.sidecar.default_submodel
                self._ordered_submodels = [d_sub] + [s for s in self._ordered_submodels if s != d_sub]

            # Validar proactivamente todos los contratos de submodelos al inicializar el adaptador
            for sub_name in self._ordered_submodels:
                self._compile_submodel_graph(sub_name)
            return

        if not public_fns:
            raise ValueError(
                f"Python package at '{self.repo_dir}' contains zero public functions returning numeric/metric values."
            )

        roots = self._select_root_entrypoints(public_fns)
        for fn_name in roots:
            self._submodel_to_fn[fn_name] = public_fns[fn_name]
            self._ordered_submodels.append(fn_name)

    def _select_root_entrypoints(self, public_fns: Dict[str, ExtractedFunctionSpec]) -> List[str]:
        """Selecciona funciones públicas raíz (no invocadas únicamente como helpers internos)."""
        called_internally: Set[str] = set()
        for caller, callees in self.extractor.call_graph.items():
            if caller in public_fns:
                for ce in callees:
                    if ce != caller and ce in public_fns:
                        called_internally.add(ce)
        roots = [
            fn_name
            for fn_name, spec in public_fns.items()
            if fn_name not in called_internally and len(spec.params) >= 1
        ]
        if not roots:
            roots = [fn_name for fn_name, spec in public_fns.items() if len(spec.params) >= 1]
        if not roots:
            roots = list(public_fns.keys())
        return roots

    def list_submodels(self) -> List[str]:
        return list(self._ordered_submodels)

    def infer_submodel(
        self,
        target_id: Optional[str] = None,
        actor_hints: Optional[Sequence[str]] = None,
        file_hint: Optional[str] = None,
    ) -> str:
        if not self._ordered_submodels:
            raise ValueError(f"No submodels discovered in package '{self.repo_dir}'.")
        if len(self._ordered_submodels) == 1:
            return self._ordered_submodels[0]

        if file_hint:
            fh_low = file_hint.lower().replace("\\", "/")
            for sub_name in self._ordered_submodels:
                fn_spec = self._submodel_to_fn[sub_name]
                if fn_spec.rel_file.lower().endswith(fh_low):
                    return sub_name

        hints_set: Set[str] = set()
        if target_id:
            hints_set.add(str(target_id))
        for h in actor_hints or []:
            if h:
                hints_set.add(str(h))

        if target_id and target_id in self._ordered_submodels:
            return target_id

        if hints_set:
            best_sub = self._ordered_submodels[0]
            best_score = -1
            for sub_name in self._ordered_submodels:
                g_cand = self._compile_submodel_graph(sub_name)
                cand_actors = set(g_cand.nodes.keys()) | set(g_cand.constraints.keys())
                score = 0
                if target_id and target_id in g_cand.nodes:
                    score += 100
                score += sum(10 for h in hints_set if h in cand_actors)
                if score > best_score:
                    best_score = score
                    best_sub = sub_name
            if best_score > 0:
                return best_sub

        return self._ordered_submodels[0]

    @staticmethod
    def _infer_arg_data_type(fn_node: ast.FunctionDef, param_name: str, default_val: Any) -> str:
        for arg in fn_node.args.args:
            if arg.arg == param_name and arg.annotation is not None:
                ann = arg.annotation
                ann_name = ann.id if isinstance(ann, ast.Name) else (ann.attr if isinstance(ann, ast.Attribute) else "")
                if ann_name == "bool":
                    return "boolean"
                if ann_name == "int":
                    return "discrete"
                if ann_name == "float":
                    return "continuous"
        if isinstance(default_val, bool):
            return "boolean"
        if isinstance(default_val, int):
            return "discrete"
        return "continuous"

    def _collect_transitive_helper_bounds(
        self,
        fn_spec: ExtractedFunctionSpec,
    ) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """Propaga guardas de precondición desde helpers llamados directamente con parámetros de entrada."""
        merged: Dict[str, List[Optional[float]]] = {
            k: [v[0], v[1]] for k, v in fn_spec.precondition_bounds.items()
        }
        param_set = set(fn_spec.params)
        for node in ast.walk(fn_spec.ast_node):
            if not isinstance(node, ast.Call):
                continue
            h_name: Optional[str] = None
            if isinstance(node.func, ast.Name):
                h_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                h_name = node.func.attr
            if not h_name or h_name not in self.extractor.functions:
                continue
            h_spec = self.extractor.functions[h_name]
            for idx, call_arg in enumerate(node.args):
                if idx < len(h_spec.params) and isinstance(call_arg, ast.Name) and call_arg.id in param_set:
                    h_param = h_spec.params[idx]
                    if h_param in h_spec.precondition_bounds:
                        hlb, hub = h_spec.precondition_bounds[h_param]
                        cur = merged.setdefault(call_arg.id, [None, None])
                        if hlb is not None:
                            cur[0] = hlb if cur[0] is None else max(cur[0], hlb)
                        if hub is not None:
                            cur[1] = hub if cur[1] is None else min(cur[1], hub)
        return {k: (v[0], v[1]) for k, v in merged.items()}

    @staticmethod
    def _merge_bounds(
        ast_bounds: Tuple[Optional[float], Optional[float]],
        sidecar_bounds: Optional[Tuple[Optional[float], Optional[float]]],
        data_type: str,
    ) -> Dict[str, Optional[float]]:
        lb, ub = ast_bounds
        if sidecar_bounds is not None:
            slb, sub = sidecar_bounds
            if slb is not None:
                lb = slb if lb is None else max(lb, slb)
            if sub is not None:
                ub = sub if ub is None else min(ub, sub)
        if data_type == "boolean":
            lb = 0.0 if lb is None else max(0.0, lb)
            ub = 1.0 if ub is None else min(1.0, ub)
        elif data_type == "discrete":
            if lb is not None:
                lb = float(math.ceil(lb - 1e-12))
            if ub is not None:
                ub = float(math.floor(ub + 1e-12))
        return {"min": lb, "max": ub}

    @staticmethod
    def _choose_nominal_value(
        explicit_default: Optional[float],
        bounds: Dict[str, Optional[float]],
        data_type: str,
    ) -> float:
        lb, ub = bounds.get("min"), bounds.get("max")
        if explicit_default is not None and math.isfinite(float(explicit_default)):
            val = float(explicit_default)
            if lb is not None and val < lb:
                val = lb
            if ub is not None and val > ub:
                val = ub
            return float(int(round(val))) if data_type in ("discrete", "boolean") else val

        if lb is not None and ub is not None:
            mid = 0.5 * (lb + ub)
            if abs(mid) < 1e-9 and ub > 0.0:
                mid = min(ub, 1.0)
            return float(int(round(mid))) if data_type in ("discrete", "boolean") else mid
        if lb is not None:
            v_lb = max(lb + (1.0 if lb >= 0.0 else 0.5 * abs(lb)), 1.0 if lb >= 0.0 else lb + 1.0)
            return float(int(round(v_lb))) if data_type in ("discrete", "boolean") else v_lb
        if ub is not None:
            v_ub = min(ub, 1.0)
            return float(int(round(v_ub))) if data_type in ("discrete", "boolean") else v_ub
        return 1.0

    @staticmethod
    def _collect_loaded_names(stmts: Sequence[ast.stmt]) -> Set[str]:
        loaded: Set[str] = set()
        for s in stmts:
            for n in ast.walk(s):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                    loaded.add(n.id)
        return loaded

    def _compile_submodel_graph(
        self,
        submodel_name: str,
        target_hint: Optional[str] = None,
    ) -> CausalGraph:
        if submodel_name not in self._submodel_to_fn:
            if (
                submodel_name in self.extractor.functions
                and not submodel_name.startswith("_")
                and self._has_value_return(self.extractor.functions[submodel_name].ast_node)
            ):
                fn_spec = self.extractor.functions[submodel_name]
            else:
                raise ValueError(
                    f"Unknown submodel '{submodel_name}' for package '{self.REPO_ID}'. "
                    f"Available submodels: {self._ordered_submodels}"
                )
        else:
            fn_spec = self._submodel_to_fn[submodel_name]

        sidecar_sub = self.sidecar.submodels.get(submodel_name) if self.sidecar else None
        has_explicit_controls = bool(sidecar_sub and len(sidecar_sub.controls) > 0)
        ast_bounds_map = self._collect_transitive_helper_bounds(fn_spec)
        param_set = set(fn_spec.params)
        pkg_rel = self._pkg_rel_file(fn_spec.rel_file)
        arg_line_map = {
            a.arg: int(getattr(a, "lineno", fn_spec.lineno))
            for a in (
                list(getattr(fn_spec.ast_node.args, "posonlyargs", []))
                + list(fn_spec.ast_node.args.args)
                + list(fn_spec.ast_node.args.kwonlyargs)
            )
        }

        # 1. Construir nodos hoja (parámetros de entrada)
        nodes_list = []
        nominal_env: Dict[str, float] = {}
        unclassified_leaves: List[str] = []

        for p_name in fn_spec.params:
            raw_py_def = fn_spec.defaults.get(p_name)
            dt = self._infer_arg_data_type(fn_spec.ast_node, p_name, raw_py_def)

            role_spec: Optional[ActorRoleSpec] = None
            if sidecar_sub:
                role_spec = sidecar_sub.controls.get(p_name) or sidecar_sub.exogenous.get(p_name)
            if role_spec and role_spec.data_type:
                dt = role_spec.data_type

            s_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None
            if role_spec and role_spec.bounds is not None:
                s_bounds = role_spec.bounds
            elif sidecar_sub and p_name in sidecar_sub.extra_bounds:
                s_bounds = sidecar_sub.extra_bounds[p_name]

            merged_b = self._merge_bounds(ast_bounds_map.get(p_name, (None, None)), s_bounds, dt)

            expl_def: Optional[float] = None
            if role_spec and role_spec.default is not None:
                expl_def = role_spec.default
            elif isinstance(raw_py_def, bool):
                expl_def = 1.0 if raw_py_def else 0.0
            elif isinstance(raw_py_def, (int, float)) and math.isfinite(float(raw_py_def)):
                expl_def = float(raw_py_def)
            elif p_name in self._dataclass_defaults:
                expl_def = self._dataclass_defaults[p_name]

            nom_val = self._choose_nominal_value(expl_def, merged_b, dt)
            nominal_env[p_name] = nom_val

            if has_explicit_controls:
                cat = "control" if (sidecar_sub and p_name in sidecar_sub.controls) else "exogenous"
            else:
                cat = "exogenous"
                unclassified_leaves.append(p_name)

            nodes_list.append(
                make_verified_node(
                    node_id=p_name,
                    category=cat,
                    rel_file=pkg_rel,
                    entity=p_name,
                    lineno=arg_line_map.get(p_name, fn_spec.lineno),
                    current_value=nom_val,
                    bounds=merged_b,
                    data_type=dt,
                )
            )

        # 2. Compilar cuerpo a ecuaciones SSA (`L0/L1` simbólico) o simulación segura (`L2` blackbox)
        edges_list: List[Tuple[str, str]] = []
        blackbox_callables: Dict[str, Tuple[List[str], Callable[..., Any]]] = {}
        clean_body = StaticRepoASTExtractor._strip_raise_guards(fn_spec.ast_node.body)

        safe_consts = {
            k: v
            for k, v in self.extractor.module_constants.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))
        }

        symbolic_succeeded = False
        if fn_spec.sympy_expr is not None:
            try:
                nodes_sym, edges_sym = self._build_symbolic_ssa_nodes(
                    fn_spec=fn_spec,
                    clean_body=clean_body,
                    safe_consts=safe_consts,
                    nominal_env=nominal_env,
                    sidecar_sub=sidecar_sub,
                    target_hint=target_hint,
                )
                nodes_list.extend(nodes_sym)
                edges_list.extend(edges_sym)
                symbolic_succeeded = True
            except Exception:
                symbolic_succeeded = False

        if not symbolic_succeeded:
            nodes_bb, edges_bb, bb_map = self._build_safe_simulation_nodes(
                fn_spec=fn_spec,
                clean_body=clean_body,
                safe_consts=safe_consts,
                nominal_env=nominal_env,
                sidecar_sub=sidecar_sub,
                target_hint=target_hint,
            )
            nodes_list.extend(nodes_bb)
            edges_list.extend(edges_bb)
            blackbox_callables.update(bb_map)

        nodes_by_id = {n.id: n for n in nodes_list}

        # 3. Validar determinísticamente el sidecar `actordb.yaml` contra los actores reales del AST
        if sidecar_sub is not None:
            self._validate_sidecar_against_ast(
                sidecar_sub=sidecar_sub,
                nodes_by_id=nodes_by_id,
                param_set=param_set,
            )
            # Aplicar cotas extra sobre nodos si las hubiera
            for b_actor, b_pair in sidecar_sub.extra_bounds.items():
                if b_actor in nodes_by_id and b_actor not in param_set:
                    n_obj = nodes_by_id[b_actor]
                    n_obj.bounds = self._merge_bounds((None, None), b_pair, n_obj.data_type)

        # 4. Ensamblar restricciones duras desde guardas AST + `assert` + `actordb.yaml`
        constraints_list: List[Constraint] = []
        seen_cids: Set[str] = set()

        for raw_gc in list(fn_spec.guard_constraints) + list(fn_spec.assert_constraints):
            expr_gc = str(raw_gc.get("expression", "")).strip()
            if not expr_gc:
                continue
            ref_ids = _extract_formula_identifiers(expr_gc)
            if ref_ids and all(rid in nodes_by_id for rid in ref_ids):
                cid_gc = str(raw_gc.get("id") or f"c_ast_{len(constraints_list) + 1}")
                if cid_gc in seen_cids:
                    cid_gc = f"{cid_gc}_{len(constraints_list) + 1}"
                seen_cids.add(cid_gc)
                constraints_list.append(
                    Constraint(
                        id=cid_gc,
                        type="hard",
                        expression=expr_gc,
                        description=str(raw_gc.get("description") or "AST precondition constraint"),
                    )
                )

        if sidecar_sub is not None:
            for sc_c in sidecar_sub.constraints:
                constraints_list.append(sc_c)
                seen_cids.add(sc_c.id)

        resolved_sense = sidecar_sub.sense if sidecar_sub else "max"
        sys_id = (
            f"{self.REPO_ID}_{submodel_name}"
            if len(self._ordered_submodels) > 1
            else self.REPO_ID
        )
        metadata: Dict[str, Any] = {
            "sense": resolved_sense,
            "repo_id": self.REPO_ID,
            "submodel": submodel_name,
            "available_submodels": list(self._ordered_submodels) or [submodel_name],
            "generic_package_adapter": True,
            "sidecar_loaded": self.sidecar is not None,
            "sidecar_path": self.sidecar.path.name if self.sidecar else None,
            "roles_inferred": False,
            "unclassified_leaves": unclassified_leaves,
            "compilation_tier": "L0_L1_SYMBOLIC" if symbolic_succeeded else "L2_SAFE_AST_SIMULATION",
        }

        graph = build_verified_causal_graph(
            system_id=sys_id,
            nodes=nodes_list,
            edges=edges_list,
            constraints=constraints_list,
            metadata=metadata,
        )
        if blackbox_callables:
            graph._blackbox_callables = blackbox_callables
        return graph

    def _build_symbolic_ssa_nodes(
        self,
        *,
        fn_spec: ExtractedFunctionSpec,
        clean_body: List[ast.stmt],
        safe_consts: Dict[str, Any],
        nominal_env: Dict[str, float],
        sidecar_sub: Optional[SubmodelSidecarSpec],
        target_hint: Optional[str],
    ) -> Tuple[List[Any], List[Tuple[str, str]]]:
        """Extrae asignaciones intermedias SSA y salidas simbólicas (`PROVEN`)."""
        # Contar cuántas veces se asigna cada identificador en el cuerpo superior (para respetar SSA puro)
        assign_counts: Dict[str, int] = {}
        for stmt in clean_body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                v = stmt.targets[0].id
                assign_counts[v] = assign_counts.get(v, 0) + 1
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                v = stmt.target.id
                assign_counts[v] = assign_counts.get(v, 0) + 1
            elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                v = stmt.target.id
                assign_counts[v] = assign_counts.get(v, 0) + 2

        # Detectar si existen `if` con asignaciones en ramas antes del return
        branched_vars: Set[str] = set()
        for stmt in clean_body:
            if isinstance(stmt, (ast.If, ast.For, ast.While)):
                for sub_n in ast.walk(stmt):
                    if isinstance(sub_n, ast.Name) and isinstance(sub_n.ctx, ast.Store):
                        branched_vars.add(sub_n.id)

        compiler = ASTFunctionToSymPy({}, safe_consts, self.extractor._helper_ast_map)
        for p in fn_spec.params:
            compiler.env[p] = sp.Symbol(p, real=True)

        ssa_exprs: Dict[str, Tuple[sp.Expr, int]] = {}
        remaining_stmts: List[ast.stmt] = []

        for idx_s, stmt in enumerate(clean_body):
            if (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            ):
                continue

            is_single_top_assign = False
            var_name = ""
            val_ast: Optional[ast.AST] = None
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                var_name = stmt.targets[0].id
                val_ast = stmt.value
                is_single_top_assign = True
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                var_name = stmt.target.id
                val_ast = stmt.value
                is_single_top_assign = True

            if (
                is_single_top_assign
                and val_ast is not None
                and not var_name.startswith("_")
                and var_name not in fn_spec.params
                and assign_counts.get(var_name, 0) == 1
                and var_name not in branched_vars
            ):
                val_expr = compiler.eval_expr(val_ast)
                if isinstance(val_expr, sp.Expr):
                    ssa_exprs[var_name] = (val_expr, getattr(stmt, "lineno", fn_spec.lineno))
                    compiler.env[var_name] = sp.Symbol(var_name, real=True)
                    continue
                compiler.env[var_name] = val_expr
                continue

            remaining_stmts = clean_body[idx_s:]
            break

        ret_val = compiler.compile_statements(remaining_stmts, dict(compiler.env)) if remaining_stmts else None
        ret_lineno = getattr(remaining_stmts[-1], "lineno", fn_spec.end_lineno) if remaining_stmts else fn_spec.end_lineno
        ret_key_lines = self._extract_return_dict_key_lines(clean_body)

        # Determinar mapa ordenado de ecuaciones calculadas (estados + salidas)
        computed_exprs: Dict[str, Tuple[sp.Expr, int]] = dict(ssa_exprs)

        if isinstance(ret_val, dict):
            dict_keys_added: List[str] = []
            for k_raw, v_raw in ret_val.items():
                k_name = str(k_raw).strip()
                if not k_name or k_name in fn_spec.params:
                    continue
                if isinstance(v_raw, (int, float)) and not isinstance(v_raw, bool):
                    v_expr = sp.Float(float(v_raw))
                elif isinstance(v_raw, sp.Expr):
                    v_expr = v_raw
                else:
                    continue
                # Si la clave del diccionario simplemente re-exporta el propio símbolo SSA `k_name`, conservar su expresión SSA
                if k_name in computed_exprs and v_expr == sp.Symbol(k_name, real=True):
                    dict_keys_added.append(k_name)
                    continue
                computed_exprs[k_name] = (v_expr, ret_key_lines.get(k_name, ret_lineno))
                dict_keys_added.append(k_name)
            if not dict_keys_added:
                raise ValueError(f"Returned dictionary in '{fn_spec.name}' contained zero numeric/symbolic metrics.")
            default_target = dict_keys_added[-1]
        elif isinstance(ret_val, (sp.Expr, int, float)) and not isinstance(ret_val, bool):
            r_expr = ret_val if isinstance(ret_val, sp.Expr) else sp.Float(float(ret_val))
            # Si `return var_name` devuelve directamente un nodo SSA existente, ése es el target natural
            if isinstance(r_expr, sp.Symbol) and r_expr.name in computed_exprs:
                default_target = r_expr.name
            else:
                out_name = (
                    sidecar_sub.target
                    if (sidecar_sub and sidecar_sub.target and sidecar_sub.target not in computed_exprs and sidecar_sub.target not in fn_spec.params)
                    else fn_spec.name
                )
                computed_exprs[out_name] = (r_expr, ret_lineno)
                default_target = out_name
        else:
            raise ValueError(f"Unsupported symbolic return value in '{fn_spec.name}'.")

        chosen_target = default_target
        if sidecar_sub and sidecar_sub.target:
            if sidecar_sub.target in computed_exprs:
                chosen_target = sidecar_sub.target
        elif target_hint and target_hint in computed_exprs:
            chosen_target = target_hint

        # Evaluar valores nominales en orden topológico y construir nodos + aristas
        all_known_ids = set(fn_spec.params) | set(computed_exprs.keys())
        dep_dag = nx.DiGraph()
        for cid in computed_exprs:
            dep_dag.add_node(cid)
        edges_out: List[Tuple[str, str]] = []
        for cid, (c_expr, _) in computed_exprs.items():
            for sym in sorted(c_expr.free_symbols, key=lambda s: s.name):
                sname = sym.name
                if sname in all_known_ids and sname != cid:
                    edges_out.append((sname, cid))
                    if sname in computed_exprs:
                        dep_dag.add_edge(sname, cid)

        eval_state = dict(nominal_env)
        topo_computed = list(nx.topological_sort(dep_dag)) if nx.is_directed_acyclic_graph(dep_dag) else list(computed_exprs.keys())
        for cid in topo_computed:
            c_expr, _ = computed_exprs[cid]
            subs_d = {sp.Symbol(k, real=True): v for k, v in eval_state.items()}
            for s in c_expr.free_symbols:
                if s.name in eval_state:
                    subs_d[s] = eval_state[s.name]
            try:
                eval_state[cid] = float(c_expr.evalf(subs=subs_d))
            except Exception:
                eval_state[cid] = 0.0

        pkg_rel = self._pkg_rel_file(fn_spec.rel_file)
        nodes_out = []
        for cid, (c_expr, c_line) in computed_exprs.items():
            cat_c = "target" if cid == chosen_target else "state"
            nodes_out.append(
                make_verified_node(
                    node_id=cid,
                    category=cat_c,
                    rel_file=pkg_rel,
                    entity=fn_spec.name,
                    lineno=c_line,
                    current_value=eval_state.get(cid, 0.0),
                    symbolic_expr=str(c_expr),
                )
            )
        return nodes_out, edges_out

    def _build_safe_simulation_nodes(
        self,
        *,
        fn_spec: ExtractedFunctionSpec,
        clean_body: List[ast.stmt],
        safe_consts: Dict[str, Any],
        nominal_env: Dict[str, float],
        sidecar_sub: Optional[SubmodelSidecarSpec],
        target_hint: Optional[str],
    ) -> Tuple[List[Any], List[Tuple[str, str]], Dict[str, Tuple[List[str], Callable[..., Any]]]]:
        """Construye nodos respaldados por `_SafeSimulationASTInterpreter` (`SAMPLED_VERIFIED`) para funciones `L2`."""
        sim_fn = self.extractor.build_safe_simulation_callable(fn_spec.name)
        if sim_fn is None:
            raise ValueError(f"Could not construct safe AST simulation callable for '{fn_spec.name}'.")

        ordered_params = list(fn_spec.params)
        nom_args = [nominal_env[p] for p in ordered_params]
        nom_ret = sim_fn(*nom_args)

        loaded_in_body = self._collect_loaded_names(clean_body)
        active_params = [p for p in ordered_params if p in loaded_in_body]

        pkg_rel = self._pkg_rel_file(fn_spec.rel_file)
        ret_key_lines = self._extract_return_dict_key_lines(clean_body)

        # Detectar variables locales muertas top-level (asignadas y jamás leídas ni retornadas)
        dead_local_nodes = []
        edges_out: List[Tuple[str, str]] = []
        for stmt in clean_body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                v_local = stmt.targets[0].id
                if not v_local.startswith("_") and v_local not in ordered_params and v_local not in loaded_in_body:
                    # Compilar simbólicamente si es posible para trazabilidad del huérfano T1-A
                    sym_local_str: Optional[str] = None
                    try:
                        c_loc = ASTFunctionToSymPy({}, safe_consts, self.extractor._helper_ast_map)
                        for p in ordered_params:
                            c_loc.env[p] = sp.Symbol(p, real=True)
                        e_loc = c_loc.eval_expr(stmt.value)
                        if isinstance(e_loc, sp.Expr):
                            sym_local_str = str(e_loc)
                            for s in e_loc.free_symbols:
                                if s.name in ordered_params:
                                    edges_out.append((s.name, v_local))
                    except Exception:
                        sym_local_str = None
                    dead_local_nodes.append(
                        make_verified_node(
                            node_id=v_local,
                            category="state",
                            rel_file=pkg_rel,
                            entity=fn_spec.name,
                            lineno=getattr(stmt, "lineno", fn_spec.lineno),
                            current_value=0.0,
                            symbolic_expr=sym_local_str,
                        )
                    )

        nodes_out = list(dead_local_nodes)
        bb_map: Dict[str, Tuple[List[str], Callable[..., Any]]] = {}

        if isinstance(nom_ret, dict):
            metric_items: List[Tuple[str, float]] = []
            for k_raw, v_raw in nom_ret.items():
                k_str = str(k_raw).strip()
                if k_str and k_str not in ordered_params and isinstance(v_raw, (int, float)) and not isinstance(v_raw, bool):
                    metric_items.append((k_str, float(v_raw)))
            if not metric_items:
                raise ValueError(f"Safe AST simulation of '{fn_spec.name}' returned dict with zero numeric values.")

            default_tgt = metric_items[-1][0]
            chosen_tgt = default_tgt
            metric_names = {m[0] for m in metric_items}
            if sidecar_sub and sidecar_sub.target and sidecar_sub.target in metric_names:
                chosen_tgt = sidecar_sub.target
            elif target_hint and target_hint in metric_names:
                chosen_tgt = target_hint

            for m_name, m_nom in metric_items:
                cat_m = "target" if m_name == chosen_tgt else "state"
                nodes_out.append(
                    make_verified_node(
                        node_id=m_name,
                        category=cat_m,
                        rel_file=pkg_rel,
                        entity=fn_spec.name,
                        lineno=ret_key_lines.get(m_name, fn_spec.end_lineno),
                        current_value=m_nom,
                        symbolic_expr=None,
                    )
                )
                for ap in active_params:
                    edges_out.append((ap, m_name))

                def _make_dict_proj(key_sel: str) -> Callable[..., float]:
                    def _proj(*xs: Any) -> float:
                        res_d = sim_fn(*xs)
                        return float(res_d[key_sel])

                    return _proj

                bb_map[m_name] = (list(ordered_params), _make_dict_proj(m_name))

        elif isinstance(nom_ret, (int, float, bool)):
            val_f = (1.0 if nom_ret else 0.0) if isinstance(nom_ret, bool) else float(nom_ret)
            out_id = (
                sidecar_sub.target
                if (sidecar_sub and sidecar_sub.target and sidecar_sub.target not in ordered_params)
                else fn_spec.name
            )
            nodes_out.append(
                make_verified_node(
                    node_id=out_id,
                    category="target",
                    rel_file=pkg_rel,
                    entity=fn_spec.name,
                    lineno=fn_spec.end_lineno,
                    current_value=val_f,
                    symbolic_expr=None,
                )
            )
            for ap in active_params:
                edges_out.append((ap, out_id))

            def _scalar_wrap(*xs: Any) -> float:
                rv = sim_fn(*xs)
                if isinstance(rv, bool):
                    return 1.0 if rv else 0.0
                return float(rv)

            bb_map[out_id] = (list(ordered_params), _scalar_wrap)
        else:
            raise ValueError(
                f"Safe AST simulation of '{fn_spec.name}' returned unsupported type '{type(nom_ret).__name__}'."
            )

        return nodes_out, edges_out, bb_map

    @staticmethod
    def _validate_sidecar_against_ast(
        *,
        sidecar_sub: SubmodelSidecarSpec,
        nodes_by_id: Dict[str, Any],
        param_set: Set[str],
    ) -> None:
        avail_all = sorted(nodes_by_id.keys())
        avail_params = sorted(param_set)

        # 1. Validar target
        if sidecar_sub.target and sidecar_sub.target not in nodes_by_id:
            raise ContractASTMismatchError(
                f"Target '{sidecar_sub.target}' declared in actordb sidecar for submodel '{sidecar_sub.name}' "
                f"does not exist in extracted AST actors {avail_all}.",
                submodel=sidecar_sub.name,
                missing_identifiers=[sidecar_sub.target],
                available_actors=avail_all,
            )

        # 2. Validar controls y exogenous (deben existir y ser hojas de entrada del AST)
        missing_roles: List[str] = []
        non_leaf_roles: List[str] = []
        for c_id in sidecar_sub.controls:
            if c_id not in nodes_by_id:
                missing_roles.append(c_id)
            elif c_id not in param_set:
                non_leaf_roles.append(c_id)
        for e_id in sidecar_sub.exogenous:
            if e_id not in nodes_by_id:
                missing_roles.append(e_id)
            elif e_id not in param_set:
                non_leaf_roles.append(e_id)

        if missing_roles:
            raise ContractASTMismatchError(
                f"Sidecar submodel '{sidecar_sub.name}' declares actor(s) {missing_roles} in controls/exogenous "
                f"that do not exist in Python AST parameters {avail_params}.",
                submodel=sidecar_sub.name,
                missing_identifiers=missing_roles,
                available_actors=avail_all,
            )
        if non_leaf_roles:
            raise ContractASTMismatchError(
                f"Sidecar submodel '{sidecar_sub.name}' declares computed state/target actor(s) {non_leaf_roles} "
                f"as leaf controls/exogenous. Only function input parameters {avail_params} can be leaf roles.",
                submodel=sidecar_sub.name,
                missing_identifiers=non_leaf_roles,
                available_actors=avail_params,
            )

        # 3. Validar bounds sueltos
        missing_bounds = [bk for bk in sidecar_sub.extra_bounds if bk not in nodes_by_id]
        if missing_bounds:
            raise ContractASTMismatchError(
                f"Sidecar submodel '{sidecar_sub.name}' declares bounds for unknown actor(s) {missing_bounds}. "
                f"Available AST actors: {avail_all}",
                submodel=sidecar_sub.name,
                missing_identifiers=missing_bounds,
                available_actors=avail_all,
            )

        # 4. Validar identificadores dentro de cada expresión de constraint/invariant
        for c_obj in sidecar_sub.constraints:
            c_vars = _extract_formula_identifiers(c_obj.expression)
            missing_c = [v for v in c_vars if v not in nodes_by_id]
            if missing_c:
                raise ContractASTMismatchError(
                    f"Constraint '{c_obj.id}' ('{c_obj.expression}') in submodel '{sidecar_sub.name}' "
                    f"references unknown AST actor(s) {missing_c}. Available AST actors: {avail_all}",
                    submodel=sidecar_sub.name,
                    missing_identifiers=missing_c,
                    available_actors=avail_all,
                )

    def build_causal_graph(
        self,
        submodel: Optional[str] = None,
        target_hint: Optional[str] = None,
        actor_hints: Optional[Sequence[str]] = None,
        file_hint: Optional[str] = None,
    ) -> CausalGraph:
        chosen = submodel or self.infer_submodel(
            target_id=target_hint,
            actor_hints=actor_hints,
            file_hint=file_hint,
        )
        return self._compile_submodel_graph(chosen, target_hint=target_hint)
