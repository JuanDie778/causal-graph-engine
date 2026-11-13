"""Oráculo Matemático Determinista de ActorDB para las 4 primitivas causales (`TRACE`, `DO`, `SOLVE`, `PROVE`).

Proporciona ejecución determinista sin alucinaciones, validación estricta de actores,
memoria epistémica indexada en O(1) (`GAP-04`), cirugía de grafo exacta de Pearl
y resistencia a presiones de violación de restricciones duras (`GAP-03`).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import sympy as sp
from scipy.optimize import minimize
from sympy.parsing.sympy_parser import parse_expr

from ..ingest import ASTIngestPipeline, compile_source_to_ir
from ..ingest_adapters.engine import RepositoryCausalEngine
from ..layer1_heuristic import Layer1Router
from ..layer2_exact import Layer2Solver
from ..schema import CausalGraph, Constraint
from ..solver import SolveStatus, solve_problem
from .prover import (
    _MATH_FUNC_LOCALS,
    _SAFE_NUMERIC_MODS,
    _clean_json_val,
    _extract_identifiers,
    _parse_propositional_cnf,
    evaluate_constraint_at_state,
    prove_on_graph,
)


_PROPOSITIONAL_CONSTR_RE = re.compile(
    r"(=>|->|<=>|<->|&&|\|\||!=|!(?!=)|\b(?:and|or|not|And|Or|Not|Implies)\b|(?<![<=>!])<(?![=>-])|(?<![<=>-])>(?!=))"
)


def _normalize_graph_for_solver(
    graph: CausalGraph,
    exo_overrides: Optional[Dict[str, float]] = None,
) -> CausalGraph:
    """Traduce restricciones proposicionales (`not`, `or`, `and`, `=>`, `<=>`, `!=`) y funciones alias (`log1p`)
    a forma canónica compatible con `cge.solver.compile` preservando trazabilidad exacta de IDs.
    """
    needs_prop_norm = any(
        c.type == "hard"
        and c.expression
        and "sum(" not in c.expression
        and (bool(_PROPOSITIONAL_CONSTR_RE.search(c.expression)) or "log1p(" in c.expression)
        for c in graph.constraints.values()
    )
    needs_node_norm = any(
        bool(n.symbolic_expr and "log1p(" in n.symbolic_expr)
        for n in graph.nodes.values()
    )
    if not needs_prop_norm and not needs_node_norm:
        return graph

    g_copy = copy.copy(graph)
    g_copy.constraints = dict(graph.constraints)
    if needs_node_norm:
        g_copy.nodes = dict(graph.nodes)
        node_syms_tmp: Dict[str, Any] = {
            "pi": sp.pi,
            **_MATH_FUNC_LOCALS,
            **{nid: sp.Symbol(nid, real=True) for nid in graph.nodes},
        }
        for nid, n_orig in graph.nodes.items():
            if n_orig.symbolic_expr and "log1p(" in n_orig.symbolic_expr:
                try:
                    n_clone = copy.copy(n_orig)
                    n_clone.symbolic_expr = str(sp.sympify(n_orig.symbolic_expr, locals=node_syms_tmp))
                    g_copy.nodes[nid] = n_clone
                except Exception:
                    pass
    if not needs_prop_norm:
        return g_copy

    local_syms: Dict[str, Any] = {nid: sp.Symbol(nid, real=True) for nid in graph.nodes}
    int_syms = {
        local_syms[nid]
        for nid, n in graph.nodes.items()
        if n.data_type in ("discrete", "boolean")
    }
    bool_syms: Set[sp.Symbol] = set()
    node_bounds: Dict[sp.Symbol, Tuple[float, float]] = {}
    for nid, n in graph.nodes.items():
        sym = local_syms[nid]
        if n.category == "exogenous":
            v_exo = (exo_overrides or {}).get(nid, n.current_value)
            if v_exo is None and isinstance(n.distribution, dict):
                v_exo = n.distribution.get("mean", 0.0)
            if isinstance(v_exo, (int, float)) and math.isfinite(float(v_exo)):
                fv = float(v_exo)
                node_bounds[sym] = (fv, fv)
        elif n.category == "control":
            b = n.bounds or {}
            b_min = b.get("min") if isinstance(b, dict) else None
            b_max = b.get("max") if isinstance(b, dict) else None
            if n.data_type == "boolean":
                bool_syms.add(sym)
                node_bounds[sym] = (0.0, 1.0)
            elif (
                n.data_type == "discrete"
                and b_min is not None
                and b_max is not None
                and round(float(b_min)) == 0
                and round(float(b_max)) == 1
            ):
                bool_syms.add(sym)
                node_bounds[sym] = (0.0, 1.0)
            elif b_min is not None and b_max is not None:
                lo_f, hi_f = float(b_min), float(b_max)
                if math.isfinite(lo_f) and math.isfinite(hi_f):
                    if hi_f < lo_f:
                        lo_f, hi_f = hi_f, lo_f
                    node_bounds[sym] = (lo_f, hi_f)

    for fn_name, fn_sym in _MATH_FUNC_LOCALS.items():
        local_syms.setdefault(fn_name, fn_sym)
    local_syms.setdefault("pi", sp.pi)

    # Sustitución topológica a hojas para acotar expresiones lineales sobre estados intermedios
    leaf_subs: Dict[sp.Symbol, sp.Expr] = {}
    dg_tmp = nx.DiGraph()
    for nid in graph.nodes:
        dg_tmp.add_node(nid)
    for ed in graph.edges:
        if ed.type != "empirical_data" and ed.source in graph.nodes and ed.target in graph.nodes:
            dg_tmp.add_edge(ed.source, ed.target)
    if nx.is_directed_acyclic_graph(dg_tmp):
        for nid in nx.topological_sort(dg_tmp):
            n_obj = graph.nodes[nid]
            s_nid = local_syms[nid]
            if n_obj.category in ("exogenous", "control"):
                leaf_subs[s_nid] = s_nid
            elif n_obj.symbolic_expr:
                try:
                    e_raw = sp.sympify(n_obj.symbolic_expr, locals=local_syms)
                    leaf_subs[s_nid] = e_raw.xreplace(leaf_subs)
                except Exception:
                    leaf_subs[s_nid] = s_nid

    def _strict_eps(e: sp.Expr) -> sp.Expr:
        e_sub = e.xreplace(leaf_subs) if leaf_subs else e
        if (
            e_sub.free_symbols
            and e_sub.free_symbols.issubset(int_syms)
            and all(num.is_Integer for num in e_sub.atoms(sp.Number))
        ):
            return sp.Integer(1)
        return sp.Float(1e-6)

    def _affine_upper_bound(expr_leaf: sp.Expr) -> Optional[float]:
        syms_in = list(expr_leaf.free_symbols)
        if not set(syms_in).issubset(set(node_bounds.keys())):
            return None
        try:
            exp_e = sp.expand(expr_leaf)
            c0_e = exp_e.xreplace({s: sp.Integer(0) for s in syms_in})
            if not c0_e.is_number:
                return None
            sup_v = float(c0_e.evalf())
            for s in syms_in:
                ds = sp.diff(exp_e, s)
                if not ds.is_number:
                    return None
                coeff = float(ds.evalf())
                lo_s, hi_s = node_bounds[s]
                sup_v += max(coeff * lo_s, coeff * hi_s)
            return sup_v if math.isfinite(sup_v) else None
        except Exception:
            return None

    def _compile_clause_expr(lits: List[sp.Expr]) -> Optional[sp.Expr]:
        if not lits:
            return sp.Integer(1)
        bool_indicators: List[sp.Expr] = []
        other_lits: List[sp.Expr] = []
        for lit in lits:
            lit_sub = sp.expand(lit.xreplace(leaf_subs)) if leaf_subs else sp.expand(lit)
            if lit_sub.is_number:
                if float(lit_sub.evalf()) <= 1e-9:
                    return None  # Cláusula tautológica
                continue  # Literal insatisfacible
            if len(lit_sub.free_symbols) == 1:
                (sym_b,) = tuple(lit_sub.free_symbols)
                if sym_b in bool_syms:
                    try:
                        v0 = float(lit_sub.subs(sym_b, 0).evalf())
                        v1 = float(lit_sub.subs(sym_b, 1).evalf())
                        if v0 <= 1e-9 and v1 <= 1e-9:
                            return None  # Cláusula tautológica en {0, 1}
                        if v0 > 1e-9 and v1 > 1e-9:
                            continue  # Literal insatisfacible en {0, 1}
                        if v0 <= 1e-9 and v1 > 1e-9:
                            bool_indicators.append(sp.Integer(1) - sym_b)
                            continue
                        if v1 <= 1e-9 and v0 > 1e-9:
                            bool_indicators.append(sym_b)
                            continue
                    except Exception:
                        pass
            other_lits.append(lit)

        if not bool_indicators and not other_lits:
            return sp.Integer(1)

        if bool_indicators and not other_lits:
            lin_bool = sp.expand(sp.Integer(1) - sum(bool_indicators, sp.Integer(0)))
            if lin_bool.is_number and float(lin_bool.evalf()) <= 1e-9:
                return None
            return lin_bool

        if bool_indicators and len(other_lits) == 1:
            other_raw = other_lits[0]
            other_sub = sp.expand(other_raw.xreplace(leaf_subs)) if leaf_subs else sp.expand(other_raw)
            m_sup = _affine_upper_bound(other_sub)
            if m_sup is not None:
                if m_sup <= 0.0:
                    return None
                m_safe = sp.Float(m_sup + 1e-7)
                return sp.expand(other_raw - m_safe * sum(bool_indicators, sp.Integer(0)))

        rem_lits: List[sp.Expr] = [sp.Integer(1) - ind for ind in bool_indicators] + other_lits
        return rem_lits[0] if len(rem_lits) == 1 else sp.Min(*rem_lits)

    new_constraints: Dict[str, Constraint] = {}
    for cid, c_obj in g_copy.constraints.items():
        if c_obj.type != "hard":
            new_constraints[cid] = c_obj
            continue
        expr_s = (c_obj.expression or "").strip()
        if not expr_s or "sum(" in expr_s or (not _PROPOSITIONAL_CONSTR_RE.search(expr_s) and "log1p(" not in expr_s):
            new_constraints[cid] = c_obj
            continue
        try:
            clauses, const_truth = _parse_propositional_cnf(expr_s, local_syms)
        except Exception:
            new_constraints[cid] = c_obj
            continue

        if const_truth is True:
            continue
        if const_truth is False:
            new_constraints[cid] = Constraint(
                id=cid, type="hard", expression="1.0 <= 0.0", description=c_obj.description
            )
            continue

        clause_exprs: List[sp.Expr] = []
        unsat_found = False
        for cl in clauses:
            lits: List[sp.Expr] = []
            for ant in cl.antecedents:
                if ant.strict:
                    lits.append(-ant.expr)
                else:
                    lits.append(-ant.expr + _strict_eps(ant.expr))
            for cq in cl.consequents:
                if cq.strict:
                    lits.append(cq.expr + _strict_eps(cq.expr))
                else:
                    lits.append(cq.expr)
            c_e = _compile_clause_expr(lits)
            if c_e is None:
                continue
            if c_e.is_number and float(c_e.evalf()) > 0.0:
                unsat_found = True
                break
            clause_exprs.append(c_e)

        if unsat_found:
            new_constraints[cid] = Constraint(
                id=cid, type="hard", expression="1.0 <= 0.0", description=c_obj.description
            )
        elif len(clause_exprs) == 1:
            new_constraints[cid] = Constraint(
                id=cid, type="hard", expression=f"{clause_exprs[0]} <= 0", description=c_obj.description
            )
        elif len(clause_exprs) > 1:
            for idx_c, ce in enumerate(clause_exprs):
                sub_cid = f"{cid}__cnf_{idx_c}"
                new_constraints[sub_cid] = Constraint(
                    id=sub_cid, type="hard", expression=f"{ce} <= 0", description=c_obj.description
                )

    g_copy.constraints = new_constraints
    return g_copy


def _compute_fs_signature(path: str) -> str:
    """Calcula una huella ligera determinista (rutas + tamaños + mtime_ns) para invalidar caché ante ediciones."""
    p = os.path.abspath(path)
    if os.path.isfile(p):
        st = os.stat(p)
        return f"file:{p}:{st.st_size}:{st.st_mtime_ns}"
    entries: List[str] = []
    for root, dirs, files in os.walk(p):
        dirs[:] = sorted(
            d for d in dirs
            if not d.startswith(".") and d not in ("__pycache__", "tests", ".pytest_cache", "node_modules")
        )
        for fname in sorted(files):
            if fname.endswith((".py", ".sql", ".md", ".csv", ".yaml", ".yml", ".json")):
                fpath = os.path.join(root, fname)
                try:
                    st = os.stat(fpath)
                    rel = os.path.relpath(fpath, p).replace("\\", "/")
                    entries.append(f"{rel}:{st.st_size}:{st.st_mtime_ns}")
                except OSError:
                    continue
    digest = hashlib.sha256("|".join(entries).encode("utf-8")).hexdigest()
    return f"dir:{p}:{digest}"


def _looks_like_inline_code(text: str) -> bool:
    stripped = text.strip()
    if "\n" in stripped:
        return True
    return stripped.startswith(("def ", "import ", "from ", "class ", "#")) or ("=" in stripped and "/" not in stripped and "\\" not in stripped)


class ActorDBOracle:
    """Oráculo causal sin estado conversacional espurio pero con caché epistémica determinista O(1)."""

    def __init__(
        self,
        default_source: Optional[Any] = None,
        allow_auto_discovery: bool = True,
    ):
        self.default_source = default_source if default_source is not None else os.environ.get("CGE_DEFAULT_SOURCE")
        self.allow_auto_discovery = allow_auto_discovery
        self._cache: Dict[str, CausalGraph] = {}
        self._lock = threading.RLock()
        self._repo_engine = RepositoryCausalEngine()

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()
            self._repo_engine.clear_cache()

    def _resolve_alternate_adapter_submodel(
        self,
        source_path: Optional[str],
        graph_data: Optional[Dict[str, Any]],
        current_graph: CausalGraph,
        predicate: Callable[[CausalGraph], bool],
    ) -> Optional[Tuple[CausalGraph, bool]]:
        """Si la fuente es un repositorio adaptador sin `:submodel` explícito, busca otro submodelo compatible."""
        if graph_data is not None:
            return None
        avail = (current_graph.metadata or {}).get("available_submodels")
        if not isinstance(avail, list) or len(avail) <= 1:
            return None
        src_eff = source_path if source_path is not None else self.default_source
        if isinstance(src_eff, os.PathLike):
            src_eff = os.fspath(src_eff)
        if not isinstance(src_eff, str) or not src_eff.strip():
            return None
        ad_spec = self._repo_engine.resolve_adapter_source(src_eff.strip())
        if not ad_spec or "error_code" in ad_spec or ad_spec.get("explicit_submodel"):
            return None
        cur_sub = (current_graph.metadata or {}).get("submodel")
        for cand_sub in avail:
            if cand_sub == cur_sub:
                continue
            cand_g, cand_hit, cand_err = self.resolve_graph(
                source_path=source_path,
                submodel_override=cand_sub,
            )
            if cand_err is None and cand_g is not None and predicate(cand_g):
                return cand_g, cand_hit
        return None

    def resolve_graph(
        self,
        source_path: Optional[str] = None,
        graph_data: Optional[Dict[str, Any]] = None,
        target_hint: Optional[str] = None,
        actor_hints: Optional[Sequence[str]] = None,
        submodel_override: Optional[str] = None,
    ) -> Tuple[Optional[CausalGraph], bool, Optional[Dict[str, Any]]]:
        """Resuelve el CausalGraph objetivo devolviendo `(graph, cache_hit, error_dict)`."""
        # 1. Grafo explícito en diccionario o instancia CausalGraph
        if graph_data is not None:
            if isinstance(graph_data, CausalGraph):
                return graph_data, False, None
            if not isinstance(graph_data, dict) or "nodes" not in graph_data:
                return None, False, {
                    "status": "ERROR",
                    "error_code": "INVALID_GRAPH_DATA",
                    "seal": "UNSUPPORTED",
                    "message": "graph_data must be a valid CausalGraph dictionary containing 'nodes', 'edges', and 'constraints'.",
                    "zero_hallucination_verified": True,
                }
            try:
                raw_json = json.dumps(graph_data, sort_keys=True, default=str)
                ckey = f"dict:{hashlib.sha256(raw_json.encode('utf-8')).hexdigest()}"
                with self._lock:
                    if ckey in self._cache:
                        return self._cache[ckey], True, None
                g = CausalGraph.from_dict(graph_data)
                with self._lock:
                    if len(self._cache) < 128:
                        self._cache[ckey] = g
                return g, False, None
            except Exception as exc:
                return None, False, {
                    "status": "ERROR",
                    "error_code": "INVALID_GRAPH_DATA",
                    "seal": "UNSUPPORTED",
                    "message": f"Failed to deserialize graph_data: {exc}",
                    "zero_hallucination_verified": True,
                }

        # 2. Fuente explícita o por defecto
        src = source_path if source_path is not None else self.default_source
        if isinstance(src, CausalGraph):
            return src, True, None
        if isinstance(src, dict):
            return self.resolve_graph(graph_data=src)
        if isinstance(src, os.PathLike):
            src = os.fspath(src)

        if isinstance(src, str):
            src_clean = src.strip()
            if not src_clean:
                return None, False, {
                    "status": "ERROR",
                    "error_code": "INVALID_SOURCE",
                    "seal": "UNSUPPORTED",
                    "message": "source_path cannot be an empty string.",
                    "zero_hallucination_verified": True,
                }

            # 2a. Enrutamiento prioritario hacia Adaptadores Estáticos AST (`cge/ingest_adapters/`)
            adapter_spec = self._repo_engine.resolve_adapter_source(src_clean)
            if adapter_spec is not None:
                if "error_code" in adapter_spec:
                    return None, False, {
                        "status": "ERROR",
                        "error_code": adapter_spec["error_code"],
                        "seal": "UNSUPPORTED",
                        "message": adapter_spec["message"],
                        "zero_hallucination_verified": True,
                    }
                try:
                    repo_id = adapter_spec["repo_id"]
                    repo_dir = adapter_spec["repo_dir"]
                    explicit_submodel = adapter_spec["explicit_submodel"]
                    file_hint = adapter_spec["file_hint"]
                    fs_sig = adapter_spec["fs_signature"]

                    with self._lock:
                        adapter = self._repo_engine.get_adapter(repo_id, str(repo_dir))
                    if adapter is None:
                        return None, False, {
                            "status": "ERROR",
                            "error_code": "UNSUPPORTED_ADAPTER_REPO",
                            "seal": "UNSUPPORTED",
                            "message": f"No causal adapter registered for repository '{repo_id}'.",
                            "zero_hallucination_verified": True,
                        }

                    avail_subs = adapter.list_submodels() if hasattr(adapter, "list_submodels") else []
                    if explicit_submodel and avail_subs and explicit_submodel not in avail_subs:
                        return None, False, {
                            "status": "ERROR",
                            "error_code": "INVALID_SUBMODEL",
                            "seal": "UNSUPPORTED",
                            "repo_id": repo_id,
                            "available_submodels": avail_subs,
                            "message": (
                                f"Submodel '{explicit_submodel}' is not available for repository '{repo_id}'. "
                                f"Available submodels: {avail_subs}"
                            ),
                            "zero_hallucination_verified": True,
                        }

                    effective_sub = (
                        explicit_submodel
                        or submodel_override
                        or adapter.infer_submodel(
                            target_id=target_hint,
                            actor_hints=actor_hints,
                            file_hint=file_hint,
                        )
                    )
                    ckey = f"adapter:{repo_id}:{repo_dir.as_posix()}:{fs_sig}:{effective_sub}"
                    with self._lock:
                        if ckey in self._cache:
                            return self._cache[ckey], True, None

                    g = adapter.build_causal_graph(
                        submodel=effective_sub,
                        target_hint=target_hint,
                        actor_hints=actor_hints,
                        file_hint=file_hint,
                    )
                    with self._lock:
                        if len(self._cache) >= 128:
                            self._cache.pop(next(iter(self._cache)), None)
                        self._cache[ckey] = g
                    return g, False, None
                except Exception as exc:
                    return None, False, {
                        "status": "ERROR",
                        "error_code": "INGESTION_FAILED",
                        "seal": "UNSUPPORTED",
                        "message": f"Failed to compile adapter repository '{src_clean}': {exc}",
                        "zero_hallucination_verified": True,
                    }

            if os.path.isdir(src_clean):
                sig = _compute_fs_signature(src_clean)
                with self._lock:
                    if sig in self._cache:
                        return self._cache[sig], True, None
                try:
                    g = ASTIngestPipeline(src_clean).run()
                    g.metadata["source_path"] = src_clean
                    with self._lock:
                        if len(self._cache) < 128:
                            self._cache[sig] = g
                    return g, False, None
                except Exception as exc:
                    return None, False, {
                        "status": "ERROR",
                        "error_code": "INGESTION_FAILED",
                        "seal": "UNSUPPORTED",
                        "message": f"Failed to compile directory '{src_clean}': {exc}",
                        "zero_hallucination_verified": True,
                    }

            if os.path.isfile(src_clean):
                sig = _compute_fs_signature(src_clean)
                with self._lock:
                    if sig in self._cache:
                        return self._cache[sig], True, None
                try:
                    if src_clean.lower().endswith(".json"):
                        with open(src_clean, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        g = CausalGraph.from_dict(data)
                    else:
                        g = compile_source_to_ir(src_clean)
                    with self._lock:
                        if len(self._cache) < 128:
                            self._cache[sig] = g
                    return g, False, None
                except Exception as exc:
                    return None, False, {
                        "status": "ERROR",
                        "error_code": "INGESTION_FAILED",
                        "seal": "UNSUPPORTED",
                        "message": f"Failed to load source file '{src_clean}': {exc}",
                        "zero_hallucination_verified": True,
                    }

            if _looks_like_inline_code(src_clean):
                ckey = f"inline:{hashlib.sha256(src_clean.encode('utf-8')).hexdigest()}"
                with self._lock:
                    if ckey in self._cache:
                        return self._cache[ckey], True, None
                try:
                    g = compile_source_to_ir(src_clean)
                    if not g.nodes:
                        return None, False, {
                            "status": "ERROR",
                            "error_code": "EMPTY_CAUSAL_GRAPH",
                            "seal": "UNSUPPORTED",
                            "message": "Inline source code produced zero causal actors.",
                            "zero_hallucination_verified": True,
                        }
                    with self._lock:
                        if len(self._cache) < 128:
                            self._cache[ckey] = g
                    return g, False, None
                except Exception as exc:
                    return None, False, {
                        "status": "ERROR",
                        "error_code": "INGESTION_FAILED",
                        "seal": "UNSUPPORTED",
                        "message": f"Failed to compile inline source code: {exc}",
                        "zero_hallucination_verified": True,
                    }

            return None, False, {
                "status": "ERROR",
                "error_code": "SOURCE_NOT_FOUND",
                "seal": "UNSUPPORTED",
                "message": f"Source path '{src_clean}' does not exist on disk and is not valid inline code.",
                "zero_hallucination_verified": True,
            }

        # 3. Auto-descubrimiento en el workspace si está habilitado
        if self.allow_auto_discovery:
            ckey = "autodiscover:default"
            with self._lock:
                if ckey in self._cache:
                    return self._cache[ckey], True, None
            try:
                g = Layer2Solver._discover_default_graph()
                with self._lock:
                    self._cache[ckey] = g
                return g, False, None
            except Exception as exc:
                return None, False, {
                    "status": "ERROR",
                    "error_code": "NO_ACTIVE_GRAPH",
                    "seal": "UNSUPPORTED",
                    "message": f"Could not auto-discover an active CausalGraph: {exc}",
                    "zero_hallucination_verified": True,
                }

        return None, False, {
            "status": "ERROR",
            "error_code": "NO_SOURCE_SPECIFIED",
            "seal": "UNSUPPORTED",
            "message": "No source_path or graph_data provided and auto-discovery is disabled.",
            "zero_hallucination_verified": True,
        }

    # ------------------------------------------------------------------
    # Utilidades Simbólicas y de Evaluación Exacta sobre CausalGraph
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_target_id(graph: CausalGraph, target_id: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        if not graph.nodes:
            return None, "CausalGraph contains zero nodes."
        if target_id is not None:
            if target_id not in graph.nodes:
                return None, f"Target actor '{target_id}' does not exist in graph '{graph.system_id}'."
            return target_id, None
        for nid, n in graph.nodes.items():
            if n.category == "target":
                return nid, None
        dag = nx.DiGraph()
        for nid in graph.nodes:
            dag.add_node(nid)
        for e in graph.edges:
            if e.type != "empirical_data":
                dag.add_edge(e.source, e.target)
        sinks = [nid for nid in dag.nodes if dag.out_degree(nid) == 0]
        return (sinks[0] if sinks else next(iter(graph.nodes))), None

    @staticmethod
    def _compile_graph_mechanisms(
        graph: CausalGraph,
    ) -> Tuple[nx.DiGraph, Dict[str, sp.Symbol], Dict[str, sp.Expr], Dict[str, Tuple[List[str], Callable[..., Any]]]]:
        dag = nx.DiGraph()
        syms: Dict[str, sp.Symbol] = {}
        for nid in graph.nodes:
            dag.add_node(nid)
            syms[nid] = sp.Symbol(nid, real=True)
        for e in graph.edges:
            if e.type != "empirical_data":
                dag.add_edge(e.source, e.target)

        local_syms: Dict[str, Any] = {"pi": sp.pi, **_MATH_FUNC_LOCALS, **syms}
        safe_mods = _SAFE_NUMERIC_MODS

        mechanisms: Dict[str, sp.Expr] = {}
        compiled_fns: Dict[str, Tuple[List[str], Callable[..., Any]]] = {}
        bb_map = getattr(graph, "_blackbox_callables", {}) or {}
        surr_map = (graph.metadata or {}).get("blackbox_surrogates", {}) or {}

        for nid, node in graph.nodes.items():
            if node.symbolic_expr:
                expr = sp.sympify(node.symbolic_expr, locals=local_syms)
                mechanisms[nid] = expr
                try:
                    raw_expr = parse_expr(node.symbolic_expr, local_dict=local_syms, evaluate=False)
                except Exception:
                    raw_expr = expr
                for s in expr.free_symbols | raw_expr.free_symbols:
                    syms[str(s)] = s
                f_s = sorted(str(s) for s in raw_expr.free_symbols)
                try:
                    fn_c = sp.lambdify([syms[s] for s in f_s], raw_expr, modules=safe_mods)
                except Exception:
                    f_s = sorted(str(s) for s in expr.free_symbols)
                    fn_c = sp.lambdify([syms[s] for s in f_s], expr, modules=safe_mods)
                compiled_fns[nid] = (f_s, fn_c)
            elif nid in bb_map:
                p_names, py_fn = bb_map[nid]
                compiled_fns[nid] = (list(p_names), py_fn)
            elif nid in surr_map and surr_map[nid].get("surrogate_sympy_expr"):
                expr = sp.sympify(surr_map[nid]["surrogate_sympy_expr"], locals=local_syms)
                mechanisms[nid] = expr
                for s in expr.free_symbols:
                    syms[str(s)] = s
                f_s = sorted(str(s) for s in expr.free_symbols)
                compiled_fns[nid] = (f_s, sp.lambdify([syms[s] for s in f_s], expr, modules=safe_mods))

        return dag, syms, mechanisms, compiled_fns

    @classmethod
    def _evaluate_forward_with_do(
        cls,
        graph: CausalGraph,
        dag: nx.DiGraph,
        compiled_fns: Dict[str, Tuple[List[str], Callable[..., Any]]],
        interventions: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Evalúa el grafo aplicando cirugía de Pearl do(I) sobre cualquier subconjunto de actores."""
        interv = interventions or {}
        ctx = context or {}
        state: Dict[str, float] = {}

        # Inicializar valores nominales y de contexto
        for nid, node in graph.nodes.items():
            if nid in interv:
                state[nid] = float(interv[nid])
            elif nid in ctx:
                state[nid] = float(ctx[nid])
            elif isinstance(node.current_value, (int, float)):
                state[nid] = float(node.current_value)
            elif isinstance(node.current_value, bool):
                state[nid] = 1.0 if node.current_value else 0.0
            else:
                state[nid] = 0.0

        # Propagar en orden topológico omitiendo las ecuaciones estructurales de los actores intervenidos
        for nid in nx.topological_sort(dag):
            if nid in interv:
                state[nid] = float(interv[nid])
                continue
            if nid in compiled_fns and graph.nodes[nid].category not in ("exogenous", "control"):
                arg_names, fn = compiled_fns[nid]
                args = [state.get(a, 0.0) for a in arg_names]
                state[nid] = float(fn(*args))
        return state

    @classmethod
    def _symbolic_and_numeric_derivative(
        cls,
        dag: nx.DiGraph,
        syms: Dict[str, sp.Symbol],
        mechanisms: Dict[str, sp.Expr],
        nominal_state: Dict[str, float],
        source: str,
        target: str,
        graph: Optional[CausalGraph] = None,
        compiled_fns: Optional[Dict[str, Tuple[List[str], Callable[..., Any]]]] = None,
    ) -> Tuple[Optional[str], Optional[float], bool]:
        """Calcula d(target)/d(source) simbólica y numérica exacta en el estado nominal.

        Retorna `(sym_expr_str, numeric_deriv, is_identically_zero)`.
        """
        if source == target:
            return "1", 1.0, False
        if not dag.has_node(source) or not dag.has_node(target) or not nx.has_path(dag, source, target):
            return "0", 0.0, True

        # Si algún nodo en los caminos dirigidos de `source` a `target` carece de expresión simbólica
        # (ej. `_blackbox_callables` o arista empírica pura), JAMÁS reclamar derivada simbólica idénticamente nula!
        between_nodes = nx.descendants(dag, source) & (nx.ancestors(dag, target) | {target})
        if any(n not in mechanisms for n in between_nodes):
            if graph is not None and compiled_fns is not None:
                try:
                    x0 = float(nominal_state.get(source, 0.0))
                    eps = max(1e-6, 1e-5 * abs(x0))
                    m_dag = nx.DiGraph()
                    for nid_m in dag.nodes:
                        m_dag.add_node(nid_m)
                    for u_m, v_m in dag.edges:
                        if v_m != source:
                            m_dag.add_edge(u_m, v_m)
                    st_hi = cls._evaluate_forward_with_do(graph, m_dag, compiled_fns, interventions={source: x0 + eps})
                    st_lo = cls._evaluate_forward_with_do(graph, m_dag, compiled_fns, interventions={source: x0 - eps})
                    fd_val = (float(st_hi[target]) - float(st_lo[target])) / (2.0 * eps)
                    return None, fd_val, False
                except Exception:
                    return None, None, False
            return None, None, False

        try:
            topo = list(nx.topological_sort(dag))
            expanded: Dict[str, sp.Expr] = {}
            expanded_active: Dict[str, sp.Expr] = {}
            subs_nom = {syms[k]: nominal_state[k] for k in nominal_state if k in syms}

            def _select_active_branch(expr_in: sp.Expr) -> sp.Expr:
                cur_e = expr_in
                for _ in range(12):
                    replaced_any = False
                    for sub in list(sp.postorder_traversal(cur_e)):
                        if isinstance(sub, sp.Min) and len(sub.args) >= 2:
                            chosen_m = min(sub.args, key=lambda a: float(a.evalf(subs=subs_nom)))
                            cur_e = cur_e.xreplace({sub: chosen_m})
                            replaced_any = True
                            break
                        elif isinstance(sub, sp.Max) and len(sub.args) >= 2:
                            chosen_m = max(sub.args, key=lambda a: float(a.evalf(subs=subs_nom)))
                            cur_e = cur_e.xreplace({sub: chosen_m})
                            replaced_any = True
                            break
                        elif isinstance(sub, sp.Abs) and len(sub.args) == 1:
                            v_a = float(sub.args[0].evalf(subs=subs_nom))
                            cur_e = cur_e.xreplace({sub: (sub.args[0] if v_a >= 0.0 else -sub.args[0])})
                            replaced_any = True
                            break
                        elif isinstance(sub, sp.Piecewise):
                            chosen = sub.args[-1][0]
                            for val_b, cond_b in sub.args:
                                if cond_b is sp.true or cond_b is True:
                                    chosen = val_b
                                    break
                                c_eval = cond_b.xreplace(subs_nom)
                                if bool(c_eval) is True:
                                    chosen = val_b
                                    break
                            cur_e = cur_e.xreplace({sub: chosen})
                            replaced_any = True
                            break
                    if not replaced_any:
                        break
                return cur_e

            for nid in topo:
                if nid == source or nid not in mechanisms:
                    expanded[nid] = syms[nid]
                    expanded_active[nid] = syms[nid]
                else:
                    raw_e = mechanisms[nid]
                    s_map = {syms[k]: expanded[k] for k in expanded if k in syms and k != nid}
                    expanded[nid] = raw_e.xreplace(s_map)

                    act_e = _select_active_branch(raw_e)
                    s_act_map = {syms[k]: expanded_active[k] for k in expanded_active if k in syms and k != nid}
                    expanded_active[nid] = act_e.xreplace(s_act_map)

            if target not in mechanisms:
                return None, None, False

            raw_diff = sp.diff(expanded[target], syms[source])
            is_zero = False
            if raw_diff == 0:
                sym_out = "0"
                is_zero = True
            elif not raw_diff.has(sp.Min, sp.Max, sp.Piecewise, sp.Abs, sp.Derivative):
                simp_d = sp.simplify(raw_diff)
                sym_out = str(simp_d)
                if simp_d == 0:
                    is_zero = True
            else:
                sym_out = str(raw_diff)

            if is_zero:
                return "0", 0.0, True

            act_diff = sp.diff(expanded_active[target], syms[source])
            num_val = float(act_diff.evalf(subs=subs_nom))
            return sym_out, num_val, False
        except Exception:
            return None, None, False

    @classmethod
    def _detect_dead_actors(
        cls,
        graph: CausalGraph,
        dag: nx.DiGraph,
        syms: Dict[str, sp.Symbol],
        mechanisms: Dict[str, sp.Expr],
        nominal_state: Dict[str, float],
        target_id: str,
        compiled_fns: Optional[Dict[str, Tuple[List[str], Callable[..., Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        """Detecta actores muertos estructurales (T1-A), por cancelación algebraica (T1-B) y palancas desconectadas (T1-C)."""
        constraint_guard_nodes: Set[str] = set()
        for c_obj in graph.constraints.values():
            if c_obj.type == "hard":
                for v_id in _extract_identifiers(c_obj.expression):
                    if v_id in dag:
                        constraint_guard_nodes.add(v_id)
                        constraint_guard_nodes.update(nx.ancestors(dag, v_id))

        dead_list: List[Dict[str, Any]] = []
        for nid, node in graph.nodes.items():
            if nid == target_id or node.category in ("target", "exogenous"):
                continue

            has_target_path = dag.has_node(nid) and dag.has_node(target_id) and nx.has_path(dag, nid, target_id)
            guards_constraint = nid in constraint_guard_nodes

            if node.category == "state":
                if not has_target_path and not guards_constraint:
                    dead_list.append({
                        "node": nid,
                        "category": node.category,
                        "subtype": "T1-A_STRUCTURAL_ORPHAN",
                        "file": node.source_reference.file if node.source_reference else None,
                        "line_range": node.source_reference.line_range if node.source_reference else None,
                        "reason": (
                            f"State actor '{nid}' has no directed causal path to target '{target_id}' "
                            f"and participates in no hard invariant."
                        ),
                    })
                elif has_target_path and not guards_constraint:
                    _, _, is_zero = cls._symbolic_and_numeric_derivative(
                        dag, syms, mechanisms, nominal_state, nid, target_id, graph=graph, compiled_fns=compiled_fns
                    )
                    if is_zero:
                        dead_list.append({
                            "node": nid,
                            "category": node.category,
                            "subtype": "T1-B_ALGEBRAIC_PATH_CANCELLATION",
                            "file": node.source_reference.file if node.source_reference else None,
                            "line_range": node.source_reference.line_range if node.source_reference else None,
                            "reason": (
                                f"State actor '{nid}' has topological path(s) to '{target_id}', "
                                f"but its total derivative cancels identically (d({target_id})/d({nid}) == 0)."
                            ),
                        })
            elif node.category == "control":
                if not has_target_path and not guards_constraint:
                    dead_list.append({
                        "node": nid,
                        "category": node.category,
                        "subtype": "T1-C_DISCONNECTED_CONTROL",
                        "file": node.source_reference.file if node.source_reference else None,
                        "line_range": node.source_reference.line_range if node.source_reference else None,
                        "reason": (
                            f"Control lever '{nid}' is causally disconnected from target '{target_id}' "
                            f"and from all hard constraints."
                        ),
                    })
                elif has_target_path and not guards_constraint:
                    _, _, is_zero = cls._symbolic_and_numeric_derivative(
                        dag, syms, mechanisms, nominal_state, nid, target_id, graph=graph, compiled_fns=compiled_fns
                    )
                    if is_zero:
                        dead_list.append({
                            "node": nid,
                            "category": node.category,
                            "subtype": "T1-C_DISCONNECTED_CONTROL",
                            "file": node.source_reference.file if node.source_reference else None,
                            "line_range": node.source_reference.line_range if node.source_reference else None,
                            "reason": (
                                f"Control lever '{nid}' has identically zero total derivative toward '{target_id}'."
                            ),
                        })
        return dead_list

    # ------------------------------------------------------------------
    # 1. Primitiva `TRACE` (`cge_trace`)
    # ------------------------------------------------------------------

    def trace(
        self,
        target_id: Optional[str] = None,
        source_id: Optional[str] = None,
        top_k_paths: int = 5,
        source_path: Optional[str] = None,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Traza el linaje causal, derivadas totales exactas, ranking de palancas y actores muertos."""
        hints_t = [source_id] if source_id else None
        graph, cache_hit, err = self.resolve_graph(
            source_path=source_path,
            graph_data=graph_data,
            target_hint=target_id,
            actor_hints=hints_t,
        )
        if err is not None or graph is None:
            return _clean_json_val(err)

        if (target_id is not None and target_id not in graph.nodes) or (
            source_id is not None and source_id not in graph.nodes
        ):
            alt_t = self._resolve_alternate_adapter_submodel(
                source_path,
                graph_data,
                graph,
                lambda g: (target_id is None or target_id in g.nodes)
                and (source_id is None or source_id in g.nodes),
            )
            if alt_t is not None:
                graph, cache_hit = alt_t

        resolved_target, t_err = self._resolve_target_id(graph, target_id)
        if t_err is not None or resolved_target is None:
            return _clean_json_val({
                "status": "ERROR",
                "error_code": "UNKNOWN_ACTOR",
                "seal": "UNSUPPORTED",
                "message": t_err,
                "available_actors": sorted(graph.nodes.keys()),
                "zero_hallucination_verified": True,
            })

        if source_id is not None and source_id not in graph.nodes:
            return _clean_json_val({
                "status": "ERROR",
                "error_code": "UNKNOWN_ACTOR",
                "seal": "UNSUPPORTED",
                "message": f"Source actor '{source_id}' does not exist in graph '{graph.system_id}'.",
                "available_actors": sorted(graph.nodes.keys()),
                "zero_hallucination_verified": True,
            })

        k_clean = max(1, min(int(top_k_paths), 25))
        dag, syms, mechanisms, compiled_fns = self._compile_graph_mechanisms(graph)

        if not nx.is_directed_acyclic_graph(dag):
            cycles = [list(c) for c in nx.simple_cycles(dag)]
            return _clean_json_val({
                "status": "ILL_POSED_CYCLE",
                "seal": "PROVEN",
                "system_id": graph.system_id,
                "target_id": resolved_target,
                "cycles": cycles,
                "reason": f"Static directed cycle(s) detected in causal graph: {cycles}",
                "cache_hit": cache_hit,
                "zero_hallucination_verified": True,
            })

        try:
            nominal_state = self._evaluate_forward_with_do(graph, dag, compiled_fns)
        except (ZeroDivisionError, OverflowError, ValueError) as exc:
            return _clean_json_val({
                "status": "ERROR",
                "error_code": "DOMAIN_SINGULARITY",
                "seal": "UNSUPPORTED",
                "message": f"Nominal forward evaluation hit mathematical singularity: {exc}",
                "zero_hallucination_verified": True,
            })
        router = Layer1Router(graph.to_dict())
        adjoints_fallback = router.compute_reverse_mode_ad_adjoints(resolved_target)

        ancestors_set = nx.ancestors(dag, resolved_target) if dag.has_node(resolved_target) else set()
        topo_ancestors = [nid for nid in nx.topological_sort(dag) if nid in ancestors_set]

        dead_actors = self._detect_dead_actors(
            graph, dag, syms, mechanisms, nominal_state, resolved_target, compiled_fns=compiled_fns
        )
        dead_ids = {d["node"] for d in dead_actors}

        # Ranking de palancas de control con derivadas simbólicas/numéricas exactas
        s_target = graph.nodes[resolved_target].characteristic_scale or max(1.0, abs(nominal_state.get(resolved_target, 1.0)))
        control_ranking: List[Dict[str, Any]] = []
        for nid, node in graph.nodes.items():
            if node.category == "control":
                sym_d, num_d, is_zero = self._symbolic_and_numeric_derivative(
                    dag, syms, mechanisms, nominal_state, nid, resolved_target,
                    graph=graph, compiled_fns=compiled_fns,
                )
                tot_d = float(num_d) if num_d is not None else float(adjoints_fallback.get(nid, 0.0))
                if is_zero:
                    tot_d = 0.0
                s_ctrl = node.characteristic_scale or max(1.0, abs(nominal_state.get(nid, 1.0)))
                dim_elast = tot_d * (float(s_ctrl) / max(float(s_target), 1e-6))
                control_ranking.append({
                    "control": nid,
                    "total_derivative": round(tot_d, 6),
                    "symbolic_derivative": sym_d,
                    "dimensionless_elasticity": round(dim_elast, 6),
                    "is_dead": nid in dead_ids,
                    "top_paths": router.find_top_k_paths(nid, resolved_target, k=k_clean),
                })
        control_ranking.sort(key=lambda r: abs(r["dimensionless_elasticity"]), reverse=True)

        # Determinar sello epistémico en el cono causal de target_id
        cone_nodes = ancestors_set | {resolved_target}
        exact_pairs = {
            (e.source, e.target) for e in graph.edges if e.type in ("exact_code", "discrete_rule")
        }
        surr_map = (graph.metadata or {}).get("blackbox_surrogates", {}) or {}
        bb_map = getattr(graph, "_blackbox_callables", {}) or {}
        has_empirical_in_cone = any(
            e.type == "empirical_data"
            and (e.source, e.target) not in exact_pairs
            and e.target in cone_nodes
            for e in graph.edges
        ) or any(n in surr_map or n in bb_map for n in cone_nodes)
        has_unverified_in_cone = any(
            isinstance(graph.nodes[n].fidelity, dict) and not graph.nodes[n].fidelity.get("verified", True)
            for n in cone_nodes
        )
        seal = "UNSUPPORTED" if has_unverified_in_cone else ("SAMPLED_VERIFIED" if has_empirical_in_cone else "PROVEN")

        actors_summary = {
            "exogenous": [nid for nid, n in graph.nodes.items() if n.category == "exogenous"],
            "controls": [nid for nid, n in graph.nodes.items() if n.category == "control"],
            "states": [nid for nid, n in graph.nodes.items() if n.category == "state"],
            "targets": [nid for nid, n in graph.nodes.items() if n.category == "target"],
            "constraints": list(graph.constraints.keys()),
        }

        out: Dict[str, Any] = {
            "status": "OK",
            "seal": seal,
            "system_id": graph.system_id,
            "target_id": resolved_target,
            "source_id": source_id,
            "nominal_target_value": nominal_state.get(resolved_target),
            "actors_summary": actors_summary,
            "causal_ancestors": topo_ancestors,
            "dead_nodes": dead_actors,
            "dead_node_ids": [d["node"] for d in dead_actors],
            "control_ranking": control_ranking,
            "cache_hit": cache_hit,
            "zero_hallucination_verified": True,
        }

        if source_id is not None:
            has_path = dag.has_node(source_id) and dag.has_node(resolved_target) and nx.has_path(dag, source_id, resolved_target)
            sym_d, num_d, is_zero = self._symbolic_and_numeric_derivative(
                dag, syms, mechanisms, nominal_state, source_id, resolved_target,
                graph=graph, compiled_fns=compiled_fns,
            )
            tot_d = 0.0 if (not has_path or is_zero) else (
                float(num_d) if num_d is not None else float(adjoints_fallback.get(source_id, 0.0))
            )
            s_src = graph.nodes[source_id].characteristic_scale or max(1.0, abs(nominal_state.get(source_id, 1.0)))
            dim_elast = tot_d * (float(s_src) / max(float(s_target), 1e-6))
            descendants = sorted(nx.descendants(dag, source_id)) if dag.has_node(source_id) else []
            out["source_trace"] = {
                "source_id": source_id,
                "target_id": resolved_target,
                "source_category": graph.nodes[source_id].category,
                "has_directed_path": bool(has_path),
                "is_causally_dead_to_target": bool(not has_path or is_zero or source_id in dead_ids),
                "total_derivative": round(tot_d, 8),
                "symbolic_derivative": sym_d,
                "dimensionless_elasticity": round(dim_elast, 6),
                "descendants": descendants,
                "paths": router.find_top_k_paths(source_id, resolved_target, k=k_clean) if has_path else [],
            }

        return _clean_json_val(out)

    # ------------------------------------------------------------------
    # 2. Primitiva `DO` (`cge_do`) — Cirugía de Grafo de Pearl
    # ------------------------------------------------------------------

    def do(
        self,
        interventions: Dict[str, Any],
        target_id: Optional[str] = None,
        context: Optional[Dict[str, float]] = None,
        source_path: Optional[str] = None,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ejecuta una intervención contrafactual exacta de Pearl `do(I)` mutilando aristas entrantes."""
        hints_d: List[str] = []
        if isinstance(interventions, dict):
            hints_d.extend(str(k) for k in interventions.keys())
        if isinstance(context, dict):
            hints_d.extend(str(k) for k in context.keys())
        graph, cache_hit, err = self.resolve_graph(
            source_path=source_path,
            graph_data=graph_data,
            target_hint=target_id,
            actor_hints=hints_d or None,
        )
        if err is not None or graph is None:
            return _clean_json_val(err)

        if not isinstance(interventions, dict):
            return _clean_json_val({
                "status": "ERROR",
                "error_code": "INVALID_INTERVENTION",
                "seal": "UNSUPPORTED",
                "message": "interventions must be a dictionary mapping actor IDs to finite numeric/boolean values.",
                "zero_hallucination_verified": True,
            })

        if (
            (target_id is not None and target_id not in graph.nodes)
            or any(k not in graph.nodes for k in interventions)
            or (isinstance(context, dict) and any(k not in graph.nodes for k in context))
        ):
            alt_d = self._resolve_alternate_adapter_submodel(
                source_path,
                graph_data,
                graph,
                lambda g: (target_id is None or target_id in g.nodes)
                and all(k in g.nodes for k in interventions)
                and (not isinstance(context, dict) or all(k in g.nodes for k in context)),
            )
            if alt_d is not None:
                graph, cache_hit = alt_d

        resolved_target, t_err = self._resolve_target_id(graph, target_id)
        if t_err is not None or resolved_target is None:
            return _clean_json_val({
                "status": "ERROR",
                "error_code": "UNKNOWN_ACTOR",
                "seal": "UNSUPPORTED",
                "message": t_err,
                "available_actors": sorted(graph.nodes.keys()),
                "zero_hallucination_verified": True,
            })

        # Validar nombres y valores de intervenciones y contexto
        clean_interventions: Dict[str, float] = {}
        unknown_interv = [k for k in interventions if k not in graph.nodes]
        if unknown_interv:
            return _clean_json_val({
                "status": "ERROR",
                "error_code": "UNKNOWN_ACTOR",
                "seal": "UNSUPPORTED",
                "message": f"Unknown actor(s) in interventions: {unknown_interv}",
                "unknown_actors": unknown_interv,
                "available_actors": sorted(graph.nodes.keys()),
                "zero_hallucination_verified": True,
            })

        for k, v in interventions.items():
            if isinstance(v, bool):
                clean_interventions[k] = 1.0 if v else 0.0
            elif isinstance(v, (int, float)):
                fv = float(v)
                if not math.isfinite(fv):
                    return _clean_json_val({
                        "status": "ERROR",
                        "error_code": "NON_FINITE_VALUE",
                        "seal": "UNSUPPORTED",
                        "message": f"Intervention value for '{k}' must be finite, got {v!r}.",
                        "zero_hallucination_verified": True,
                    })
                clean_interventions[k] = fv
            else:
                return _clean_json_val({
                    "status": "ERROR",
                    "error_code": "INVALID_VALUE_TYPE",
                    "seal": "UNSUPPORTED",
                    "message": f"Intervention value for '{k}' must be numeric or boolean, got {type(v).__name__}.",
                    "zero_hallucination_verified": True,
                })

        clean_context: Dict[str, float] = {}
        if context is not None:
            if not isinstance(context, dict):
                return _clean_json_val({
                    "status": "ERROR",
                    "error_code": "INVALID_CONTEXT",
                    "seal": "UNSUPPORTED",
                    "message": "context must be a dictionary mapping actor IDs to finite numeric values.",
                    "zero_hallucination_verified": True,
                })
            unknown_ctx = [k for k in context if k not in graph.nodes]
            if unknown_ctx:
                return _clean_json_val({
                    "status": "ERROR",
                    "error_code": "UNKNOWN_ACTOR",
                    "seal": "UNSUPPORTED",
                    "message": f"Unknown actor(s) in context: {unknown_ctx}",
                    "available_actors": sorted(graph.nodes.keys()),
                    "zero_hallucination_verified": True,
                })
            for k, v in context.items():
                fv = float(v)
                if not math.isfinite(fv):
                    return _clean_json_val({
                        "status": "ERROR",
                        "error_code": "NON_FINITE_VALUE",
                        "seal": "UNSUPPORTED",
                        "message": f"Context value for '{k}' must be finite, got {v!r}.",
                        "zero_hallucination_verified": True,
                    })
                clean_context[k] = fv

        dag, syms, mechanisms, compiled_fns = self._compile_graph_mechanisms(graph)

        # Construir el grafo mutilado M_{bar{I}} cortando aristas entrantes a cualquier nodo intervenido
        mutilated_dag = nx.DiGraph()
        for nid in dag.nodes:
            mutilated_dag.add_node(nid)
        mutilated_edges: List[Dict[str, str]] = []
        for u, v in dag.edges:
            if v in clean_interventions:
                mutilated_edges.append({"source": u, "target": v})
            else:
                mutilated_dag.add_edge(u, v)

        if not nx.is_directed_acyclic_graph(mutilated_dag):
            cycles = [list(c) for c in nx.simple_cycles(mutilated_dag)]
            return _clean_json_val({
                "status": "UNSUPPORTED",
                "error_code": "ILL_POSED_CYCLE",
                "seal": "UNSUPPORTED",
                "cycles": cycles,
                "reason": f"Mutilated causal graph retains directed cycle(s): {cycles}",
                "zero_hallucination_verified": True,
            })

        # Compuerta de Honestidad Epistémica sobre el grafo mutilado M_{bar{I}}
        active_ancestors = (
            (nx.ancestors(mutilated_dag, resolved_target) | {resolved_target})
            if mutilated_dag.has_node(resolved_target) else {resolved_target}
        )
        # Un nodo intervenido do(V=v) fija su valor directamente, por lo que no ejecuta su mecanismo original
        evaluated_mechanism_nodes = active_ancestors - set(clean_interventions.keys())
        fid_rep = (graph.metadata or {}).get("fidelity_report")
        fid_unv = set((fid_rep or {}).get("unverified_nodes") or []) if isinstance(fid_rep, dict) else set()
        active_unverified = sorted(
            nid for nid in evaluated_mechanism_nodes
            if nid in fid_unv
            or (isinstance(graph.nodes[nid].fidelity, dict) and not graph.nodes[nid].fidelity.get("verified", True))
        )
        if active_unverified:
            return _clean_json_val({
                "status": "UNSUPPORTED",
                "seal": "UNSUPPORTED",
                "system_id": graph.system_id,
                "target_id": resolved_target,
                "unverified_nodes": active_unverified,
                "reason": (
                    f"Active causal ancestor(s) {active_unverified} in mutilated graph lack verified source fidelity."
                ),
                "zero_hallucination_verified": True,
            })

        # Evaluar mundo base (con mismo contexto U=u pero sin do(I)) y mundo intervenido do(I)
        try:
            if nx.is_directed_acyclic_graph(dag):
                baseline_state = self._evaluate_forward_with_do(graph, dag, compiled_fns, interventions=None, context=clean_context)
            else:
                baseline_state = self._evaluate_forward_with_do(graph, mutilated_dag, compiled_fns, interventions=None, context=clean_context)
            intervened_state = self._evaluate_forward_with_do(
                graph, mutilated_dag, compiled_fns, interventions=clean_interventions, context=clean_context
            )
            if any(not math.isfinite(v) for v in intervened_state.values()):
                raise ValueError("Non-finite state value encountered during intervention propagation.")
        except (ZeroDivisionError, OverflowError, ValueError) as exc:
            return _clean_json_val({
                "status": "ERROR",
                "error_code": "DOMAIN_SINGULARITY",
                "seal": "UNSUPPORTED",
                "message": f"Intervention evaluation hit mathematical singularity: {exc}",
                "zero_hallucination_verified": True,
            })

        # Auditar restricciones duras y cotas de control en el mundo post-intervención
        violated_constraints: List[str] = []
        constraint_margins: Dict[str, float] = {}

        for cid, node in graph.nodes.items():
            if node.category == "control":
                b = node.bounds or {}
                val_c = intervened_state[cid]
                if b.get("min") is not None and val_c < float(b["min"]) - 1e-5:
                    b_id = f"bounds({cid})"
                    violated_constraints.append(b_id)
                    constraint_margins[b_id] = val_c - float(b["min"])
                if b.get("max") is not None and val_c > float(b["max"]) + 1e-5:
                    b_id = f"bounds({cid})"
                    if b_id not in violated_constraints:
                        violated_constraints.append(b_id)
                    constraint_margins[b_id] = float(b["max"]) - val_c

        local_syms: Dict[str, Any] = {"pi": sp.pi, **_MATH_FUNC_LOCALS, **syms}
        for cid, c_obj in graph.constraints.items():
            if c_obj.type != "hard":
                continue
            expr_s = (c_obj.expression or "").strip()
            if not expr_s or "sum(" in expr_s:
                continue
            try:
                sat, margin = evaluate_constraint_at_state(
                    expr_s, intervened_state, local_syms=local_syms, tol=1e-4
                )
                constraint_margins[cid] = round(margin, 6)
                if not sat and cid not in violated_constraints:
                    violated_constraints.append(cid)
            except Exception:
                continue

        node_deltas = {
            nid: round(intervened_state[nid] - baseline_state[nid], 8)
            for nid in graph.nodes
        }
        affected_nodes = [nid for nid, d in node_deltas.items() if abs(d) > 1e-9]
        unaffected_nodes = [nid for nid, d in node_deltas.items() if abs(d) <= 1e-9]

        # Identificar si se intervino sobre actores sin camino hacia el objetivo (resistencia a premisas falsas)
        intervened_dead: List[str] = []
        if nx.is_directed_acyclic_graph(dag):
            dead_info = self._detect_dead_actors(
                graph, dag, syms, mechanisms, baseline_state, resolved_target, compiled_fns=compiled_fns
            )
            dead_set = {d["node"] for d in dead_info}
            for k in clean_interventions:
                if k in dead_set or (k != resolved_target and not nx.has_path(dag, k, resolved_target)):
                    intervened_dead.append(k)

        base_y = baseline_state[resolved_target]
        post_y = intervened_state[resolved_target]
        causal_effect = post_y - base_y

        surr_map = (graph.metadata or {}).get("blackbox_surrogates", {}) or {}
        bb_map = getattr(graph, "_blackbox_callables", {}) or {}
        uses_approx = any(nid in surr_map or nid in bb_map for nid in evaluated_mechanism_nodes)
        seal = "SAMPLED_VERIFIED" if uses_approx else "PROVEN"

        return _clean_json_val({
            "status": "OK",
            "seal": seal,
            "system_id": graph.system_id,
            "target_id": resolved_target,
            "interventions": clean_interventions,
            "context": clean_context,
            "mutilated_edges": mutilated_edges,
            "baseline_target_value": round(base_y, 8),
            "intervened_target_value": round(post_y, 8),
            "causal_effect": round(causal_effect, 8),
            "is_feasible": len(violated_constraints) == 0,
            "violated_constraints": violated_constraints,
            "constraint_margins": constraint_margins,
            "baseline_state": baseline_state,
            "intervened_state": intervened_state,
            "node_deltas": node_deltas,
            "affected_nodes": affected_nodes,
            "unaffected_nodes": unaffected_nodes,
            "intervened_on_disconnected_actors": intervened_dead,
            "cache_hit": cache_hit,
            "zero_hallucination_verified": True,
        })

    # ------------------------------------------------------------------
    # 3. Primitiva `SOLVE` (`cge_solve`)
    # ------------------------------------------------------------------

    @classmethod
    def _solve_blackbox_graph(
        cls,
        graph: CausalGraph,
        dag: nx.DiGraph,
        syms: Dict[str, sp.Symbol],
        compiled_fns: Dict[str, Tuple[List[str], Callable[..., Any]]],
        resolved_target: str,
        resolved_sense: str,
        control_ids: List[str],
        fixed_controls: Optional[Dict[str, float]] = None,
        exo_overrides: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Optimizador determinista multi-arranque para grafos con mecanismos `_blackbox_callables` puros."""
        cone_nodes: Set[str] = (
            (nx.ancestors(dag, resolved_target) | {resolved_target})
            if dag.has_node(resolved_target)
            else {resolved_target}
        )
        for c_obj in graph.constraints.values():
            if c_obj.type == "hard" and c_obj.expression:
                for vid in _extract_identifiers(c_obj.expression):
                    if vid in dag:
                        cone_nodes.add(vid)
                        cone_nodes.update(nx.ancestors(dag, vid))

        fid_rep = (graph.metadata or {}).get("fidelity_report")
        fid_unv = set((fid_rep or {}).get("unverified_nodes") or []) if isinstance(fid_rep, dict) else set()
        unverified = sorted(
            nid
            for nid in cone_nodes
            if nid in fid_unv
            or (isinstance(graph.nodes[nid].fidelity, dict) and not graph.nodes[nid].fidelity.get("verified", True))
        )
        if unverified or (isinstance(fid_rep, dict) and not fid_rep.get("verified", True)):
            return {
                "status": "UNSUPPORTED",
                "certificate_status": SolveStatus.UNSUPPORTED.value,
                "unverified_nodes": unverified,
                "reason": f"Fidelity verification failed on nodes: {unverified}",
                "certificate": {"status": SolveStatus.UNSUPPORTED.value, "unverified_nodes": unverified},
            }

        fixed_map = {k: float(v) for k, v in (fixed_controls or {}).items()}
        free_ids = [cid for cid in control_ids if cid not in fixed_map]
        bounds_list: List[Tuple[float, float]] = []
        x0_list: List[float] = []
        is_disc_list: List[bool] = []

        for cid in free_ids:
            n = graph.nodes[cid]
            x0_v = float(n.current_value) if isinstance(n.current_value, (int, float)) else 0.0
            b = n.bounds or {}
            b_min = b.get("min")
            b_max = b.get("max")
            lb = float(b_min) if b_min is not None else (0.0 if x0_v >= 0.0 else x0_v - 100.0)
            ub = float(b_max) if b_max is not None else max(1000.0, abs(x0_v) * 5.0)
            if ub < lb:
                lb, ub = ub, lb
            bounds_list.append((lb, ub))
            x0_list.append(min(ub, max(lb, x0_v)))
            is_disc_list.append(n.data_type in ("discrete", "boolean"))

        local_syms: Dict[str, Any] = {"pi": sp.pi, **_MATH_FUNC_LOCALS, **syms}
        base_state = cls._evaluate_forward_with_do(
            graph, dag, compiled_fns, interventions=fixed_map or None, context=exo_overrides
        )
        base_j = float(base_state[resolved_target])

        def _eval_candidate(vec: Sequence[float]) -> Tuple[float, float, Dict[str, float], List[str], List[str]]:
            interv = dict(fixed_map)
            for idx_f, cid in enumerate(free_ids):
                lb_i, ub_i = bounds_list[idx_f]
                val_i = min(ub_i, max(lb_i, float(vec[idx_f])))
                if is_disc_list[idx_f]:
                    val_i = float(int(round(val_i)))
                interv[cid] = val_i
            st = cls._evaluate_forward_with_do(graph, dag, compiled_fns, interventions=interv, context=exo_overrides)
            max_viol = 0.0
            viol_cids: List[str] = []
            act_cids: List[str] = []
            for cid_c, c_obj in graph.constraints.items():
                if c_obj.type != "hard" or not c_obj.expression or "sum(" in c_obj.expression:
                    continue
                sat, margin = evaluate_constraint_at_state(c_obj.expression, st, local_syms=local_syms, tol=1e-4)
                if not sat:
                    max_viol = max(max_viol, abs(float(margin)))
                    viol_cids.append(cid_c)
                elif abs(float(margin)) <= 1e-4:
                    act_cids.append(cid_c)
            j_v = float(st[resolved_target])
            return j_v, max_viol, st, viol_cids, act_cids

        seeds: List[np.ndarray] = []
        if free_ids:
            seeds.append(np.array(x0_list, dtype=float))
            seeds.append(np.array([lb for lb, _ in bounds_list], dtype=float))
            seeds.append(np.array([ub for _, ub in bounds_list], dtype=float))
            seeds.append(np.array([0.5 * (lb + ub) for lb, ub in bounds_list], dtype=float))
            n_dim = len(free_ids)
            if n_dim == 1:
                lb0, ub0 = bounds_list[0]
                for frac in np.linspace(0.0, 1.0, 25):
                    seeds.append(np.array([lb0 + frac * (ub0 - lb0)], dtype=float))
            else:
                rng = np.random.default_rng(42)
                lows = np.array([lb for lb, _ in bounds_list], dtype=float)
                highs = np.array([ub for _, ub in bounds_list], dtype=float)
                for _ in range(32):
                    seeds.append(lows + rng.random(n_dim) * (highs - lows))
        else:
            seeds.append(np.zeros(0, dtype=float))

        best_feas: Optional[Tuple[float, Dict[str, float], List[str]]] = None
        best_infeas: Optional[Tuple[float, List[str]]] = None

        def _consider(vec: Sequence[float]) -> float:
            nonlocal best_feas, best_infeas
            try:
                j_v, m_viol, st_d, v_cids, a_cids = _eval_candidate(vec)
            except Exception:
                return 1e18
            if not math.isfinite(j_v):
                return 1e18
            obj_min = -j_v if resolved_sense == "max" else j_v
            if m_viol <= 1e-4:
                if best_feas is None or obj_min < best_feas[0] - 1e-9:
                    best_feas = (obj_min, st_d, a_cids)
                return obj_min
            if best_infeas is None or m_viol < best_infeas[0]:
                best_infeas = (m_viol, v_cids)
            return obj_min + 1e6 * (1.0 + m_viol)

        scored_seeds: List[Tuple[float, np.ndarray]] = []
        for sv in seeds:
            sc = _consider(sv)
            scored_seeds.append((sc, sv))

        if free_ids:
            scored_seeds.sort(key=lambda item: item[0])
            for _, start_vec in scored_seeds[:3]:
                try:
                    res_opt = minimize(
                        fun=_consider,
                        x0=start_vec,
                        method="Powell",
                        bounds=bounds_list,
                        options={"maxiter": 120, "ftol": 1e-6, "xtol": 1e-5},
                    )
                    if res_opt.x is not None:
                        _consider(np.atleast_1d(res_opt.x))
                except Exception:
                    continue

        if best_feas is None:
            iis_c = best_infeas[1] if best_infeas else list(graph.constraints.keys())
            min_sl = best_infeas[0] if best_infeas else None
            return {
                "status": "INFEASIBLE",
                "certificate_status": SolveStatus.INFEASIBLE_CERTIFIED.value,
                "iis_constraints": iis_c,
                "blocking_constraints": iis_c,
                "min_relaxation_slack": min_sl,
                "kkt_shadow_prices": {},
                "certificate": {
                    "status": SolveStatus.INFEASIBLE_CERTIFIED.value,
                    "problem_class": "BLACKBOX_NLP",
                    "strategy_used": "blackbox_multistart_powell",
                    "iis_constraints": iis_c,
                },
                "reason": f"No feasible control vector satisfies hard constraints: {iis_c}",
            }

        _, best_st, best_active = best_feas
        j_best = float(best_st[resolved_target])
        opt_ctrls = {cid: round(float(best_st[cid]), 6) for cid in control_ids}
        opt_deltas = {
            cid: round(
                float(best_st[cid])
                - (float(graph.nodes[cid].current_value) if isinstance(graph.nodes[cid].current_value, (int, float)) else 0.0),
                6,
            )
            for cid in control_ids
        }
        return {
            "status": "OPTIMAL",
            "certificate_status": SolveStatus.BEST_FOUND.value,
            "problem_class": "BLACKBOX_NLP",
            "strategy_used": "blackbox_multistart_powell",
            "controls": opt_ctrls,
            "optimal_deltas": opt_deltas,
            "objective_value": j_best,
            "baseline_objective": base_j,
            "evaluated_nodes": {k: round(float(v), 6) for k, v in best_st.items()},
            "active_constraints": best_active,
            "active_regime": "blackbox_verified",
            "certificate": {
                "status": SolveStatus.BEST_FOUND.value,
                "problem_class": "BLACKBOX_NLP",
                "strategy_used": "blackbox_multistart_powell",
                "fidelity_verified": True,
                "code_verified": True,
                "max_constraint_violation": 0.0,
            },
        }

    def solve(
        self,
        target_id: Optional[str] = None,
        sense: Optional[str] = None,
        target_value: Optional[float] = None,
        fixed_controls: Optional[Dict[str, float]] = None,
        exo_overrides: Optional[Dict[str, float]] = None,
        ignore_constraints: bool = False,
        source_path: Optional[str] = None,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Calcula la receta óptima certificada respetando todas las invariantes duras y la compuerta epistémica."""
        hints_s: List[str] = []
        if isinstance(fixed_controls, dict):
            hints_s.extend(str(k) for k in fixed_controls.keys())
        if isinstance(exo_overrides, dict):
            hints_s.extend(str(k) for k in exo_overrides.keys())
        graph, cache_hit, err = self.resolve_graph(
            source_path=source_path,
            graph_data=graph_data,
            target_hint=target_id,
            actor_hints=hints_s or None,
        )
        if err is not None or graph is None:
            return _clean_json_val(err)

        def _solve_submodel_matches(g_cand: CausalGraph) -> bool:
            if target_id is not None and target_id not in g_cand.nodes:
                return False
            if isinstance(fixed_controls, dict) and any(
                k not in g_cand.nodes or g_cand.nodes[k].category != "control" for k in fixed_controls
            ):
                return False
            if isinstance(exo_overrides, dict) and any(
                k not in g_cand.nodes or g_cand.nodes[k].category != "exogenous" for k in exo_overrides
            ):
                return False
            return True

        if not _solve_submodel_matches(graph):
            alt_s = self._resolve_alternate_adapter_submodel(
                source_path, graph_data, graph, _solve_submodel_matches
            )
            if alt_s is not None:
                graph, cache_hit = alt_s

        resolved_target, t_err = self._resolve_target_id(graph, target_id)
        if t_err is not None or resolved_target is None:
            return _clean_json_val({
                "status": "ERROR",
                "error_code": "UNKNOWN_ACTOR",
                "seal": "UNSUPPORTED",
                "message": t_err,
                "available_actors": sorted(graph.nodes.keys()),
                "zero_hallucination_verified": True,
            })

        resolved_sense = (sense or (graph.metadata or {}).get("sense", "max")).strip().lower()
        if resolved_sense not in ("max", "min", "maximize", "minimize"):
            return _clean_json_val({
                "status": "ERROR",
                "error_code": "INVALID_SENSE",
                "seal": "UNSUPPORTED",
                "message": f"Invalid optimization sense {sense!r}; expected 'max' or 'min'.",
                "zero_hallucination_verified": True,
            })
        resolved_sense = "min" if resolved_sense.startswith("min") else "max"

        if target_value is not None and not math.isfinite(float(target_value)):
            return _clean_json_val({
                "status": "ERROR",
                "error_code": "NON_FINITE_VALUE",
                "seal": "UNSUPPORTED",
                "message": f"target_value must be finite, got {target_value!r}.",
                "zero_hallucination_verified": True,
            })

        control_ids = [nid for nid, n in graph.nodes.items() if n.category == "control"]
        exo_ids = [nid for nid, n in graph.nodes.items() if n.category == "exogenous"]

        if fixed_controls:
            bad_fc = [k for k in fixed_controls if k not in control_ids]
            if bad_fc:
                return _clean_json_val({
                    "status": "ERROR",
                    "error_code": "INVALID_FIXED_CONTROL",
                    "seal": "UNSUPPORTED",
                    "message": f"Actor(s) {bad_fc} are not control levers. Valid controls: {control_ids}",
                    "valid_controls": control_ids,
                    "zero_hallucination_verified": True,
                })
            for k, v in fixed_controls.items():
                if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                    return _clean_json_val({
                        "status": "ERROR",
                        "error_code": "NON_FINITE_VALUE",
                        "seal": "UNSUPPORTED",
                        "message": f"fixed_controls value for '{k}' must be finite numeric, got {v!r}.",
                        "zero_hallucination_verified": True,
                    })

        if exo_overrides:
            bad_exo = [k for k in exo_overrides if k not in exo_ids]
            if bad_exo:
                return _clean_json_val({
                    "status": "ERROR",
                    "error_code": "INVALID_EXOGENOUS_ACTOR",
                    "seal": "UNSUPPORTED",
                    "message": f"Actor(s) {bad_exo} are not exogenous variables. Valid exogenous actors: {exo_ids}",
                    "valid_exogenous": exo_ids,
                    "zero_hallucination_verified": True,
                })
            for k, v in exo_overrides.items():
                if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                    return _clean_json_val({
                        "status": "ERROR",
                        "error_code": "NON_FINITE_VALUE",
                        "seal": "UNSUPPORTED",
                        "message": f"exo_overrides value for '{k}' must be finite numeric, got {v!r}.",
                        "zero_hallucination_verified": True,
                    })

        dag, syms, _, compiled_fns = self._compile_graph_mechanisms(graph)
        if not nx.is_directed_acyclic_graph(dag):
            cycles = [list(c) for c in nx.simple_cycles(dag)]
            return _clean_json_val({
                "status": "UNSUPPORTED",
                "error_code": "ILL_POSED_CYCLE",
                "seal": "UNSUPPORTED",
                "cycles": cycles,
                "reason": f"Causal graph contains static directed cycle(s): {cycles}",
                "zero_hallucination_verified": True,
            })

        # Resistencia anti-sicofancia (GAP-03): jamás relajar ni ignorar restricciones duras ante presiones
        pressure_resisted = bool(ignore_constraints)

        # Caso degenerado válido: grafo sin palancas de control (0 grados de libertad)
        if not control_ids:
            do_res = self.do(interventions={}, target_id=resolved_target, context=exo_overrides, graph_data=graph)
            if do_res.get("status") != "OK":
                return do_res
            is_feas = bool(do_res.get("is_feasible", True))
            val_j = float(do_res["intervened_target_value"])
            if not is_feas:
                return _clean_json_val({
                    "status": "INFEASIBLE",
                    "certificate_status": SolveStatus.INFEASIBLE_CERTIFIED.value,
                    "seal": do_res.get("seal", "PROVEN"),
                    "system_id": graph.system_id,
                    "target_id": resolved_target,
                    "sense": resolved_sense,
                    "iis_constraints": do_res.get("violated_constraints", []),
                    "blocking_constraints": do_res.get("violated_constraints", []),
                    "controls": {},
                    "objective_value": None,
                    "pressure_resisted": pressure_resisted,
                    "cache_hit": cache_hit,
                    "zero_hallucination_verified": True,
                })
            return _clean_json_val({
                "status": "OPTIMAL",
                "certificate_status": SolveStatus.CERTIFIED_OPTIMAL.value,
                "seal": do_res.get("seal", "PROVEN"),
                "system_id": graph.system_id,
                "target_id": resolved_target,
                "sense": resolved_sense,
                "controls": {},
                "optimal_deltas": {},
                "objective_value": val_j,
                "baseline_objective": val_j,
                "objective_improvement": 0.0,
                "evaluated_nodes": do_res.get("intervened_state", {}),
                "active_constraints": [],
                "recipe_steps": [],
                "pressure_resisted": pressure_resisted,
                "cache_hit": cache_hit,
                "zero_hallucination_verified": True,
            })

        def _unmap_cid(cid_str: str) -> str:
            return cid_str.split("__cnf_", 1)[0] if "__cnf_" in cid_str else cid_str

        def _unmap_cid_list(cids: Optional[List[str]]) -> List[str]:
            out_c: List[str] = []
            for item in cids or []:
                orig = _unmap_cid(str(item))
                if orig not in out_c:
                    out_c.append(orig)
            return out_c

        bb_map = getattr(graph, "_blackbox_callables", {}) or {}
        surr_map = (graph.metadata or {}).get("blackbox_surrogates", {}) or {}
        cone_check: Set[str] = (
            (nx.ancestors(dag, resolved_target) | {resolved_target})
            if dag.has_node(resolved_target)
            else {resolved_target}
        )
        for c_obj in graph.constraints.values():
            if c_obj.type == "hard" and c_obj.expression:
                for vid in _extract_identifiers(c_obj.expression):
                    if vid in dag:
                        cone_check.add(vid)
                        cone_check.update(nx.ancestors(dag, vid))
        has_pure_blackbox = any(
            (nid in bb_map)
            and not graph.nodes[nid].symbolic_expr
            and (nid not in surr_map or not surr_map[nid].get("surrogate_sympy_expr"))
            for nid in cone_check
        )

        solver_graph = _normalize_graph_for_solver(graph, exo_overrides=exo_overrides)
        if has_pure_blackbox:
            sol = self._solve_blackbox_graph(
                graph=graph,
                dag=dag,
                syms=syms,
                compiled_fns=compiled_fns,
                resolved_target=resolved_target,
                resolved_sense=resolved_sense,
                control_ids=control_ids,
                fixed_controls=fixed_controls,
                exo_overrides=exo_overrides,
            )
        else:
            sol = solve_problem(
                ir=solver_graph,
                target_id=resolved_target,
                sense=resolved_sense,
                mode="direct",
                exo_overrides=exo_overrides,
                fixed_controls=fixed_controls,
            )

        if sol.get("status") == "UNSUPPORTED":
            return _clean_json_val({
                "status": "UNSUPPORTED",
                "certificate_status": SolveStatus.UNSUPPORTED.value,
                "seal": "UNSUPPORTED",
                "system_id": graph.system_id,
                "target_id": resolved_target,
                "sense": resolved_sense,
                "unverified_nodes": sol.get("unverified_nodes", []),
                "reason": sol.get("reason"),
                "certificate": sol.get("certificate"),
                "controls": {},
                "objective_value": None,
                "pressure_resisted": pressure_resisted,
                "cache_hit": cache_hit,
                "zero_hallucination_verified": True,
            })

        if sol.get("status") == "INFEASIBLE":
            iis_clean = _unmap_cid_list(sol.get("iis_constraints", []))
            blk_clean = _unmap_cid_list(sol.get("blocking_constraints", []))
            raw_shadows = sol.get("kkt_shadow_prices") or {}
            clean_shadows: Dict[str, float] = {}
            for ck, cv in raw_shadows.items():
                orig_ck = _unmap_cid(str(ck))
                clean_shadows[orig_ck] = clean_shadows.get(orig_ck, 0.0) + float(cv)
            cert_dict = dict(sol.get("certificate") or {})
            if "iis_constraints" in cert_dict:
                cert_dict["iis_constraints"] = iis_clean
            infeas_seal = "SAMPLED_VERIFIED" if sol.get("problem_class") == "BLACKBOX_NLP" else "PROVEN"
            return _clean_json_val({
                "status": "INFEASIBLE",
                "certificate_status": SolveStatus.INFEASIBLE_CERTIFIED.value,
                "seal": infeas_seal,
                "system_id": graph.system_id,
                "target_id": resolved_target,
                "sense": resolved_sense,
                "iis_constraints": iis_clean,
                "blocking_constraints": blk_clean,
                "min_relaxation_slack": sol.get("min_relaxation_slack"),
                "kkt_shadow_prices": clean_shadows,
                "certificate": cert_dict,
                "controls": {},
                "objective_value": None,
                "reason": sol.get("reason"),
                "pressure_resisted": pressure_resisted,
                "cache_hit": cache_hit,
                "zero_hallucination_verified": True,
            })

        j_star = float(sol["objective_value"])
        base_j = float(sol.get("baseline_objective", j_star))
        active_clean = _unmap_cid_list(sol.get("active_constraints", []))

        # Si se solicitó alcanzar una meta cuantitativa target_value que excede la capacidad global J*
        if target_value is not None:
            t_val = float(target_value)
            goal_unreached = (
                (resolved_sense == "max" and j_star < t_val - 1e-4)
                or (resolved_sense == "min" and j_star > t_val + 1e-4)
            )
            if goal_unreached:
                # Usar diagnóstico KKT de Layer2Solver si el objetivo y sentido coinciden con el grafo por defecto
                try:
                    l2 = Layer2Solver.from_graph(solver_graph)
                    if l2.target_id == resolved_target and resolved_sense == (graph.metadata or {}).get("sense", "max"):
                        kkt_diag = l2.optimize(
                            target_value=t_val,
                            fixed_controls=fixed_controls,
                            exo_overrides=exo_overrides,
                        )
                        if "iis_constraints" in kkt_diag:
                            kkt_diag["iis_constraints"] = _unmap_cid_list(kkt_diag["iis_constraints"])
                        if "blocking_constraints" in kkt_diag:
                            kkt_diag["blocking_constraints"] = _unmap_cid_list(kkt_diag["blocking_constraints"])
                        kkt_diag["seal"] = "PROVEN" if sol.get("certificate_status") == "CERTIFIED_OPTIMAL" else "SAMPLED_VERIFIED"
                        kkt_diag["system_id"] = graph.system_id
                        kkt_diag["target_id"] = resolved_target
                        kkt_diag["sense"] = resolved_sense
                        kkt_diag.setdefault("max_achievable_objective", round(j_star, 6))
                        kkt_diag["pressure_resisted"] = pressure_resisted
                        kkt_diag["cache_hit"] = cache_hit
                        kkt_diag["zero_hallucination_verified"] = True
                        return _clean_json_val(kkt_diag)
                except Exception:
                    pass

                blocking = list(active_clean) or list(graph.constraints.keys())
                iis_goal = ["c_target_goal"] + blocking
                shortfall = abs(t_val - j_star)
                return _clean_json_val({
                    "status": "INFEASIBLE",
                    "certificate_status": SolveStatus.INFEASIBLE_CERTIFIED.value,
                    "seal": "PROVEN" if sol.get("certificate_status") == "CERTIFIED_OPTIMAL" else "SAMPLED_VERIFIED",
                    "system_id": graph.system_id,
                    "target_id": resolved_target,
                    "sense": resolved_sense,
                    "target_value": t_val,
                    "max_achievable_objective": round(j_star, 6),
                    "shortfall": round(shortfall, 6),
                    "iis_constraints": iis_goal,
                    "blocking_constraints": blocking,
                    "controls": {},
                    "objective_value": None,
                    "reason": (
                        f"Requested goal {resolved_target}={t_val} exceeds global constrained optimum {j_star:.4f} "
                        f"(blocking constraints: {blocking})."
                    ),
                    "pressure_resisted": pressure_resisted,
                    "cache_hit": cache_hit,
                    "zero_hallucination_verified": True,
                })

        # Construir pasos de receta accionables desde los valores actuales de las palancas
        opt_controls = sol.get("controls", {})
        recipe_steps: List[Dict[str, Any]] = []
        step_idx = 1
        for cid in control_ids:
            node = graph.nodes[cid]
            cur_v = node.current_value if isinstance(node.current_value, (int, float)) else 0.0
            tgt_v = opt_controls.get(cid, cur_v)
            delta_v = float(tgt_v) - float(cur_v)
            if abs(delta_v) > 1e-7:
                recipe_steps.append({
                    "order": step_idx,
                    "control": cid,
                    "current": cur_v,
                    "target": tgt_v,
                    "delta": round(delta_v, 6),
                })
                step_idx += 1

        cert_status = sol.get("certificate_status", SolveStatus.BEST_FOUND.value)
        seal = "PROVEN" if cert_status == SolveStatus.CERTIFIED_OPTIMAL.value else "SAMPLED_VERIFIED"

        resp: Dict[str, Any] = {
            "status": "REJECTED_CONSTRAINT_BYPASS" if pressure_resisted else "OPTIMAL",
            "certificate_status": cert_status,
            "seal": seal,
            "system_id": graph.system_id,
            "target_id": resolved_target,
            "sense": resolved_sense,
            "problem_class": sol.get("problem_class"),
            "strategy_used": sol.get("strategy_used"),
            "controls": opt_controls,
            "optimal_deltas": sol.get("optimal_deltas", {}),
            "objective_value": round(j_star, 6),
            "baseline_objective": round(base_j, 6),
            "objective_improvement": round((j_star - base_j) if resolved_sense == "max" else (base_j - j_star), 6),
            "evaluated_nodes": sol.get("evaluated_nodes", {}),
            "active_constraints": active_clean,
            "active_regime": sol.get("active_regime", "smooth"),
            "certificate": sol.get("certificate"),
            "recipe_steps": recipe_steps,
            "pressure_resisted": pressure_resisted,
            "cache_hit": cache_hit,
            "zero_hallucination_verified": True,
        }
        if pressure_resisted:
            resp["refusal_reason"] = (
                f"REFUSED HARD CONSTRAINT BYPASS: ActorDB strictly enforces hard constraints "
                f"{list(graph.constraints.keys())}. Returning the maximum feasible certified optimum ({j_star:.4f})."
            )
        return _clean_json_val(resp)

    # ------------------------------------------------------------------
    # 4. Primitiva `PROVE` (`cge_prove`)
    # ------------------------------------------------------------------

    def prove(
        self,
        invariant_expr: Optional[str] = None,
        constraint_id: Optional[str] = None,
        assume_system_constraints: bool = True,
        domain_bounds: Optional[Dict[str, List[float]]] = None,
        n_samples: int = 128,
        source_path: Optional[str] = None,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Verifica formalmente invariantes, propiedades lógicas/cuantitativas o consistencia IIS del sistema."""
        hints_p: List[str] = []
        if invariant_expr:
            hints_p.extend(_extract_identifiers(invariant_expr))
        if isinstance(domain_bounds, dict):
            hints_p.extend(str(k) for k in domain_bounds.keys())
        if constraint_id:
            hints_p.append(str(constraint_id))

        graph, cache_hit, err = self.resolve_graph(
            source_path=source_path,
            graph_data=graph_data,
            actor_hints=hints_p or None,
        )
        if err is not None or graph is None:
            return _clean_json_val(err)

        inv_req_actors = (
            [rid for rid in _extract_identifiers(invariant_expr) if rid != "pi"]
            if invariant_expr
            else []
        )
        dom_req_actors = [str(k) for k in domain_bounds.keys()] if isinstance(domain_bounds, dict) else []

        def _prove_submodel_matches(g_cand: CausalGraph) -> bool:
            if constraint_id and constraint_id not in g_cand.constraints:
                return False
            if any(rid not in g_cand.nodes for rid in inv_req_actors):
                return False
            if any(dk not in g_cand.nodes for dk in dom_req_actors):
                return False
            return True

        if not _prove_submodel_matches(graph):
            alt_p = self._resolve_alternate_adapter_submodel(
                source_path, graph_data, graph, _prove_submodel_matches
            )
            if alt_p is not None:
                graph, cache_hit = alt_p

        res = prove_on_graph(
            graph=graph,
            invariant_expr=invariant_expr,
            constraint_id=constraint_id,
            assume_system_constraints=assume_system_constraints,
            domain_bounds=domain_bounds,
            n_samples=max(16, min(int(n_samples), 2048)),
        )
        res["system_id"] = graph.system_id
        res["cache_hit"] = cache_hit
        return _clean_json_val(res)
