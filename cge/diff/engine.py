"""Motor de Causal Diff (`cge/diff/engine.py`) para CI/CD y Gobernanza Causal en ActorDB.

Compara formalmente dos estados de un sistema (`base` M_0 vs `head` M_1) y diagnostica:
1. Regresiones de invariantes (roturas de conservación T2, relajación o eliminación de restricciones
   duras, violaciones nominales/universales, colapso de factibilidad con IIS, discrepancias doc-código T3,
   regresiones epistémicas de fidelidad y ciclos algebraicos mal planteados Lambda=0).
2. Alteraciones de optimalidad de políticas (degradación o mejora de J* respetando max/min, desplazamiento
   de palancas óptimas X*, evaluación cruzada de arrepentimiento/factibilidad de X*_base sobre M_1 y trampas T4).
3. Adición o desconexión de actores muertos (huérfanos estructurales T1-A, cancelaciones algebraicas T1-B
   y palancas de control desconectadas T1-C).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import sympy as sp

from ..audit import SystemAuditor
from ..ingest import ASTIngestPipeline, compile_source_to_ir
from ..ingest_adapters.engine import RepositoryCausalEngine
from ..mcp.oracle import ActorDBOracle, _looks_like_inline_code
from ..mcp.prover import (
    _MATH_FUNC_LOCALS,
    _clean_json_val,
    _compute_leaf_bounds,
    _extract_identifiers,
    prove_on_graph,
)
from ..schema import CausalGraph, Edge, Node, SourceReference


class CausalDiffError(ValueError):
    """Excepción formal emitida ante entradas o referencias Git inválidas en Causal Diff."""


class DiffVerdict(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class CausalDiffReport:
    """Informe formal determinista de diferencias causales Delta M = M_head - M_base."""

    base_system_id: str
    head_system_id: str
    verdict: str
    has_regressions: bool
    is_semantically_equivalent: bool
    summary: Dict[str, Any]
    invariant_regressions: List[Dict[str, Any]] = field(default_factory=list)
    invariants_fixed: List[Dict[str, Any]] = field(default_factory=list)
    policy_optimality: Dict[str, Any] = field(default_factory=dict)
    dead_nodes: Dict[str, Any] = field(default_factory=dict)
    structural_diff: Dict[str, Any] = field(default_factory=dict)
    epistemic_diff: Dict[str, Any] = field(default_factory=dict)
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    git_metadata: Optional[Dict[str, Any]] = None

    @property
    def policy_diff(self) -> Dict[str, Any]:
        return self.policy_optimality

    @property
    def dead_nodes_diff(self) -> Dict[str, Any]:
        return self.dead_nodes

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "base_system_id": self.base_system_id,
            "head_system_id": self.head_system_id,
            "verdict": self.verdict,
            "has_regressions": bool(self.has_regressions),
            "is_semantically_equivalent": bool(self.is_semantically_equivalent),
            "summary": self.summary,
            "invariant_regressions": self.invariant_regressions,
            "invariants_fixed": self.invariants_fixed,
            "policy_optimality": self.policy_optimality,
            "dead_nodes": self.dead_nodes,
            "structural_diff": self.structural_diff,
            "epistemic_diff": self.epistemic_diff,
            "diagnostics": self.diagnostics,
        }
        if self.git_metadata is not None:
            d["git_metadata"] = self.git_metadata
        return _clean_json_val(d)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        lines = [
            f"# Causal Diff Report (`{self.base_system_id}` -> `{self.head_system_id}`)",
            f"- **Verdict:** `{self.verdict}`",
            f"- **Has Regressions:** `{self.has_regressions}`",
            f"- **Semantically Equivalent:** `{self.is_semantically_equivalent}`",
        ]
        if self.git_metadata:
            lines.append(
                f"- **Git Refs:** `{self.git_metadata.get('base_ref')}` ({str(self.git_metadata.get('base_commit', ''))[:8]}) "
                f"-> `{self.git_metadata.get('head_ref')}` ({str(self.git_metadata.get('head_commit', ''))[:8]})"
            )
        lines.append("")
        lines.append("## 1. Invariant Regressions")
        if not self.invariant_regressions:
            lines.append("- None detected.")
        else:
            for item in self.invariant_regressions:
                lines.append(f"- **[{item.get('severity', 'critical').upper()}] `{item.get('code')}`**: {item.get('reason')}")

        lines.append("")
        lines.append("## 2. Policy Optimality")
        po = self.policy_optimality
        lines.append(
            f"- **Target (`{po.get('sense', 'max')}` `{po.get('target_id')}`):** "
            f"`{po.get('base_status')}` ({po.get('base_optimal_value')}) -> "
            f"`{po.get('head_status')}` ({po.get('head_optimal_value')}) "
            f"[direction: `{po.get('direction')}`, policy_changed: `{po.get('policy_changed')}`]"
        )

        lines.append("")
        lines.append("## 3. Dead Nodes")
        added_dn = self.dead_nodes.get("added_dead_nodes", [])
        if not added_dn:
            lines.append("- Zero dead actors added.")
        else:
            for dn in added_dn:
                lines.append(f"- **`{dn['node']}`** (`{dn['subtype']}`, `{dn['origin']}`): {dn['reason']}")
        return "\n".join(lines)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()


def load_causal_graph(
    source: Union[CausalGraph, Dict[str, Any], str, Path],
    entrypoint: Optional[str] = None,
    system_id: Optional[str] = None,
) -> CausalGraph:
    """Normaliza cualquier representación válida (CausalGraph, dict, ruta o código Python) a `CausalGraph`."""
    if isinstance(source, CausalGraph):
        return source
    if isinstance(source, dict):
        if "nodes" not in source:
            raise CausalDiffError("Dictionary input is missing required 'nodes' key.")
        try:
            g = CausalGraph.from_dict(source)
            if system_id:
                g.system_id = system_id
            return g
        except Exception as exc:
            raise CausalDiffError(f"Invalid CausalGraph dictionary: {exc}") from exc

    if isinstance(source, (str, Path)):
        s = str(source).strip()
        if not s:
            raise CausalDiffError("Source string cannot be empty.")

        repo_eng = RepositoryCausalEngine()
        adapter_spec = repo_eng.resolve_adapter_source(s)
        if adapter_spec is not None:
            if "error_code" in adapter_spec:
                raise CausalDiffError(adapter_spec["message"])
            try:
                adapter = repo_eng.get_adapter(adapter_spec["repo_id"], str(adapter_spec["repo_dir"]))
                if adapter is None:
                    raise CausalDiffError(f"No causal adapter registered for '{adapter_spec['repo_id']}'.")
                sub_sel = adapter_spec["explicit_submodel"]
                f_hint = adapter_spec["file_hint"]
                if sub_sel is None and entrypoint:
                    avail = adapter.list_submodels() if hasattr(adapter, "list_submodels") else []
                    if entrypoint in avail:
                        sub_sel = entrypoint
                    else:
                        f_hint = entrypoint
                g = adapter.build_causal_graph(submodel=sub_sel, file_hint=f_hint)
                if system_id:
                    g.system_id = system_id
                return g
            except CausalDiffError:
                raise
            except Exception as exc:
                raise CausalDiffError(f"Failed to compile adapter source '{s}': {exc}") from exc

        if os.path.isdir(s):
            g = ASTIngestPipeline(s, entrypoint=entrypoint).run()
            if system_id:
                g.system_id = system_id
            g.metadata["source_path"] = s
            return g
        if os.path.isfile(s):
            if s.lower().endswith(".json"):
                with open(s, "r", encoding="utf-8") as f:
                    data = json.load(f)
                g = CausalGraph.from_dict(data)
                if system_id:
                    g.system_id = system_id
                return g
            g = compile_source_to_ir(s, entrypoint=entrypoint, system_id=system_id)
            return g
        if _looks_like_inline_code(s):
            g = compile_source_to_ir(s, entrypoint=entrypoint, system_id=system_id)
            if not g.nodes:
                raise CausalDiffError("Inline source code produced an empty CausalGraph (0 nodes).")
            return g
        raise CausalDiffError(f"Source path '{s}' does not exist on disk and is not valid inline Python code.")

    raise CausalDiffError(f"Unsupported graph source type: {type(source).__name__}")


def _are_symbolic_exprs_equivalent(expr_a: Optional[str], expr_b: Optional[str]) -> bool:
    """Determina si dos expresiones simbólicas son algebraicamente idénticas aunque difieran sintácticamente."""
    if expr_a is None and expr_b is None:
        return True
    if expr_a is None or expr_b is None:
        return False
    sa = expr_a.strip()
    sb = expr_b.strip()
    if sa == sb:
        return True

    names = set(_extract_identifiers(sa)) | set(_extract_identifiers(sb))
    local_syms: Dict[str, Any] = {
        "pi": sp.pi,
        **_MATH_FUNC_LOCALS,
        **{n: (sp.pi if n == "pi" else sp.Symbol(n, real=True)) for n in names},
    }
    try:
        ea = sp.sympify(sa, locals=local_syms)
        eb = sp.sympify(sb, locals=local_syms)
        if ea == eb:
            return True
        diff = ea - eb
        if not diff.has(sp.Piecewise, sp.Min, sp.Max):
            simp = sp.simplify(diff)
            if simp == 0:
                return True
            if simp.is_Number and abs(float(simp.evalf())) <= 1e-12:
                return True
        else:
            folded = sp.simplify(sp.piecewise_fold(diff))
            if folded == 0:
                return True
        # Comprobar mediante evaluación determinista en puntos de prueba si comparten exactamente los mismos símbolos
        free_syms = sorted(ea.free_symbols | eb.free_symbols, key=str)
        if not free_syms:
            return abs(float(ea.evalf()) - float(eb.evalf())) <= 1e-12
        if bool(ea.equals(eb)):
            return True
    except Exception:
        return False
    return False


def _analyze_static_cycles(graph: CausalGraph) -> List[Dict[str, Any]]:
    """Detecta y clasifica ciclos en el grafo estático por su retardo causal Lambda(C) y bien-planteamiento."""
    dag = nx.DiGraph()
    edge_lags: Dict[Tuple[str, str], int] = {}
    for nid in graph.nodes:
        dag.add_node(nid)
    for e in graph.edges:
        if e.type != "empirical_data":
            dag.add_edge(e.source, e.target)
            lag_val = int((e.exact_mechanism or {}).get("lag", 0) or 0)
            edge_lags[(e.source, e.target)] = lag_val

    if nx.is_directed_acyclic_graph(dag):
        return []

    out_cycles: List[Dict[str, Any]] = []
    local_syms: Dict[str, Any] = {
        "pi": sp.pi,
        **_MATH_FUNC_LOCALS,
        **{nid: sp.Symbol(nid, real=True) for nid in graph.nodes},
    }

    for cyc in nx.simple_cycles(dag):
        cyc_list = list(cyc)
        total_lag = 0
        for idx in range(len(cyc_list)):
            u = cyc_list[idx]
            v = cyc_list[(idx + 1) % len(cyc_list)]
            total_lag += edge_lags.get((u, v), 0)

        if total_lag >= 1:
            out_cycles.append({
                "nodes": cyc_list,
                "total_lag": total_lag,
                "classification": "DYNAMIC_MEMORY_CYCLE",
                "is_ill_posed": False,
            })
            continue

        # Lazo algebraico simultáneo Lambda(C) == 0: auditar matriz Jacobiana I - J_H
        classification = "ILL_POSED_CYCLE"
        is_ill_posed = True
        try:
            z_syms = [local_syms[n] for n in cyc_list]
            residuals = []
            all_have_expr = True
            for n in cyc_list:
                s_expr = graph.nodes[n].symbolic_expr
                if not s_expr:
                    all_have_expr = False
                    break
                h_expr = sp.sympify(s_expr, locals=local_syms)
                residuals.append(local_syms[n] - h_expr)
            if all_have_expr and residuals:
                jac = sp.Matrix(residuals).jacobian(z_syms)
                det_j = sp.simplify(jac.det())
                if det_j == 0:
                    classification = "ILL_POSED_CYCLE"
                    is_ill_posed = True
                elif det_j.is_Number and abs(float(det_j.evalf())) > 1e-9:
                    # Incluso con equilibrio algebraico único, un grafo sin solver implícito requiere flag explícito
                    classification = "ALGEBRAIC_EQUILIBRIUM_CYCLE"
                    is_ill_posed = True
                else:
                    classification = "LOCAL_EQUILIBRIUM_NONUNIQUE_RISK"
                    is_ill_posed = True
        except Exception:
            classification = "ILL_POSED_CYCLE"
            is_ill_posed = True

        out_cycles.append({
            "nodes": cyc_list,
            "total_lag": 0,
            "classification": classification,
            "is_ill_posed": is_ill_posed,
        })
    return out_cycles


def _rename_expr_identifiers(expr_str: str, mapping: Dict[str, str]) -> str:
    if not mapping:
        return expr_str
    return re.sub(
        r"(?<![0-9.])\b([A-Za-z_][A-Za-z0-9_]*)\b",
        lambda m: mapping.get(m.group(1), m.group(1)),
        expr_str,
    )


def _compare_constraint_strength(
    base_graph: CausalGraph,
    head_graph: CausalGraph,
    base_expr_str: str,
    head_expr_str: str,
) -> Dict[str, Any]:
    """Compara formalmente dos restricciones duras (lineales, no lineales o proposicionales).

    Construye un grafo causal conjunto de comparación y verifica mediante `prove_on_graph`
    (SymPy + Motzkin MILP + Interval B&B + búsqueda adversarial QMC/L-BFGS-B) ambas implicaciones
    `(head) => (base)` y `(base) => (head)` sobre la unión exacta del dominio admisible y, si ambas
    son no vinculantes en la caja base pero difieren algebraicamente, sondea una caja extendida
    siempre que no cruce singularidades de dominio fuera de las cotas declaradas.
    """
    if base_expr_str.strip() == head_expr_str.strip():
        return {"relation": "equivalent", "is_weakened": False, "witness": None}

    if "sum(" in base_expr_str or "sum(" in head_expr_str:
        return {"relation": "modified", "is_weakened": True, "witness": None}

    def _build_dag(g: CausalGraph) -> nx.DiGraph:
        d = nx.DiGraph()
        for nid in g.nodes:
            d.add_node(nid)
        for e in g.edges:
            if e.type != "empirical_data":
                d.add_edge(e.source, e.target)
        return d

    b_dag = _build_dag(base_graph)
    h_dag = _build_dag(head_graph)
    if not nx.is_directed_acyclic_graph(b_dag) or not nx.is_directed_acyclic_graph(h_dag):
        if _are_symbolic_exprs_equivalent(base_expr_str, head_expr_str):
            return {"relation": "equivalent", "is_weakened": False, "witness": None}
        return {"relation": "modified", "is_weakened": True, "witness": None}

    b_ids = set(_extract_identifiers(base_expr_str))
    h_ids = set(_extract_identifiers(head_expr_str))

    b_cone: Set[str] = set()
    for nid in b_ids:
        if nid in b_dag:
            b_cone.add(nid)
            b_cone |= nx.ancestors(b_dag, nid)

    h_cone: Set[str] = set()
    for nid in h_ids:
        if nid in h_dag:
            h_cone.add(nid)
            h_cone |= nx.ancestors(h_dag, nid)

    sr = SourceReference(file="<constraint_diff>", entity="cmp", line_range=[1, 1])
    cmp_graph = CausalGraph(system_id="__constraint_strength_cmp__")

    b_map: Dict[str, str] = {}
    h_map: Dict[str, str] = {}

    for nid in b_cone:
        nb = base_graph.nodes[nid]
        if nb.category in ("state", "target") and nb.symbolic_expr:
            b_map[nid] = f"__base_{nid}"
    for nid in h_cone:
        nh = head_graph.nodes[nid]
        if nh.category in ("state", "target") and nh.symbolic_expr:
            h_map[nid] = f"__head_{nid}"

    # Detectar si alguna expresión involucra log/sqrt para no cruzar a reales negativos al ampliar cotas
    all_expr_texts = [base_expr_str, head_expr_str]
    for nid in b_cone:
        if base_graph.nodes[nid].symbolic_expr:
            all_expr_texts.append(base_graph.nodes[nid].symbolic_expr)
    for nid in h_cone:
        if head_graph.nodes[nid].symbolic_expr:
            all_expr_texts.append(head_graph.nodes[nid].symbolic_expr)
    has_pos_domain_fn = any(("sqrt(" in t or "log(" in t or "log1p(" in t) for t in all_expr_texts)

    # Hojas exógenas que cambiaron de valor/cota entre base y head se separan para fidelidad exacta
    shared_exo_candidates = {
        nid for nid in (b_cone & h_cone)
        if base_graph.nodes[nid].category == "exogenous" and head_graph.nodes[nid].category == "exogenous"
    }
    for nid in shared_exo_candidates:
        nb = base_graph.nodes[nid]
        nh = head_graph.nodes[nid]
        if (nb.bounds or {}) != (nh.bounds or {}) or nb.current_value != nh.current_value:
            b_map[nid] = f"__base_{nid}"
            h_map[nid] = f"__head_{nid}"

    def _widen_bounds(lo: float, hi: float, is_bool: bool, is_int: bool) -> Tuple[float, float]:
        if is_bool:
            return 0.0, 1.0
        span = max(20.0, hi - lo)
        w_lo = lo - span
        w_hi = hi + span
        if has_pos_domain_fn and lo >= 0.0:
            w_lo = max(0.5 * lo, 1e-6 if lo > 0.0 else 0.0)
        if is_int:
            w_lo = float(math.floor(w_lo))
            w_hi = float(math.ceil(w_hi))
        return w_lo, w_hi

    widened_bounds_override: Dict[str, List[float]] = {}

    # Añadir hojas compartidas (controles, exógenos sin cambio y símbolos libres) con unión exacta de cotas
    all_leaf_candidates = (
        {nid for nid in b_cone if nid not in b_map}
        | {nid for nid in h_cone if nid not in h_map}
        | {nid for nid in (b_ids | h_ids) if nid not in b_cone and nid not in h_cone}
    )
    for v in sorted(all_leaf_candidates):
        nb = base_graph.nodes.get(v)
        nh = head_graph.nodes.get(v)
        if nb is None and nh is None:
            cmp_graph.add_node(Node(
                id=v, label=v, category="control", data_type="continuous",
                source_reference=sr, current_value=0.0, bounds={"min": -1000.0, "max": 1000.0},
            ))
            widened_bounds_override[v] = [-1000.0, 1000.0]
            continue

        # Si es exógeno constante idéntico (lo == hi), preservar constante exacta sin abrir +-100
        if (
            (nb is None or nb.category == "exogenous")
            and (nh is None or nh.category == "exogenous")
        ):
            ref_g = head_graph if nh is not None else base_graph
            ref_n = nh if nh is not None else nb
            b_tmp, _ = _compute_leaf_bounds(ref_g, [v])
            lo_e, hi_e = b_tmp[v]
            if abs(hi_e - lo_e) <= 1e-12:
                cmp_graph.add_node(Node(
                    id=v, label=v, category="exogenous", data_type=ref_n.data_type,
                    source_reference=sr, current_value=lo_e, bounds={"min": lo_e, "max": hi_e},
                ))
                continue

        los: List[float] = []
        his: List[float] = []
        is_int = False
        is_bool = False
        dtype = "continuous"
        if nb is not None:
            bb, ib = _compute_leaf_bounds(base_graph, [v])
            los.append(bb[v][0])
            his.append(bb[v][1])
            is_int = is_int or (v in ib)
            is_bool = is_bool or (nb.data_type == "boolean")
            dtype = nb.data_type
        if nh is not None:
            bh, ih = _compute_leaf_bounds(head_graph, [v])
            los.append(bh[v][0])
            his.append(bh[v][1])
            is_int = is_int or (v in ih)
            is_bool = is_bool or (nh.data_type == "boolean")
            dtype = nh.data_type

        exact_lo, exact_hi = min(los), max(his)
        w_lo, w_hi = _widen_bounds(exact_lo, exact_hi, is_bool=is_bool, is_int=is_int)
        widened_bounds_override[v] = [w_lo, w_hi]
        mid = 0.5 * (exact_lo + exact_hi)
        if is_int:
            mid = float(round(mid))
        cmp_graph.add_node(Node(
            id=v, label=v, category="control", data_type=dtype,
            source_reference=sr, current_value=mid, bounds={"min": exact_lo, "max": exact_hi},
        ))

    # Añadir hojas exógenas separadas y nodos de estado prefijados para base y head
    for prefix_map, g_src, dag_src, cone_src in (
        (b_map, base_graph, b_dag, b_cone),
        (h_map, head_graph, h_dag, h_cone),
    ):
        for nid in nx.topological_sort(dag_src):
            if nid not in prefix_map:
                continue
            mapped_id = prefix_map[nid]
            n_obj = g_src.nodes[nid]
            if n_obj.category == "exogenous":
                b_tmp, _ = _compute_leaf_bounds(g_src, [nid])
                lo_e, hi_e = b_tmp[nid]
                cmp_graph.add_node(Node(
                    id=mapped_id, label=mapped_id, category="exogenous", data_type=n_obj.data_type,
                    source_reference=sr,
                    current_value=float(n_obj.current_value) if isinstance(n_obj.current_value, (int, float)) else lo_e,
                    bounds={"min": lo_e, "max": hi_e},
                ))
            else:
                renamed_expr = _rename_expr_identifiers(n_obj.symbolic_expr or "0.0", prefix_map)
                cmp_graph.add_node(Node(
                    id=mapped_id, label=mapped_id, category="state", data_type=n_obj.data_type,
                    source_reference=sr, symbolic_expr=renamed_expr,
                ))
                for pred in dag_src.predecessors(nid):
                    if pred in cone_src:
                        pred_mapped = prefix_map.get(pred, pred)
                        if pred_mapped in cmp_graph.nodes:
                            cmp_graph.add_edge(Edge(source=pred_mapped, target=mapped_id, type="exact_code"))

    b_formula = _rename_expr_identifiers(base_expr_str, b_map)
    h_formula = _rename_expr_identifiers(head_expr_str, h_map)

    def _classify_pair(r_rel: Dict[str, Any], r_tig: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        st_r = r_rel.get("status")
        st_t = r_tig.get("status")
        if st_r in ("ERROR", "UNSUPPORTED") or st_t in ("ERROR", "UNSUPPORTED"):
            return None
        adm_new = st_r == "FALSIFIED"
        forb_old = st_t == "FALSIFIED"
        raw_w = r_rel.get("counterexample")
        clean_w = (
            {k: v for k, v in raw_w.items() if not k.startswith("__base_") and not k.startswith("__head_")}
            if isinstance(raw_w, dict)
            else None
        )
        if st_r == "PROVEN" and st_t == "PROVEN":
            return {"relation": "equivalent", "is_weakened": False, "witness": None}
        if adm_new and not forb_old:
            return {"relation": "weakened", "is_weakened": True, "witness": clean_w}
        if forb_old and not adm_new:
            return {"relation": "tightened", "is_weakened": False, "witness": None}
        if adm_new and forb_old:
            return {"relation": "shifted", "is_weakened": True, "witness": clean_w}
        return {"relation": "modified", "is_weakened": False, "witness": None}

    res_relax = prove_on_graph(
        cmp_graph,
        invariant_expr=f"({h_formula}) => ({b_formula})",
        assume_system_constraints=False,
        n_samples=128,
    )
    res_tight = prove_on_graph(
        cmp_graph,
        invariant_expr=f"({b_formula}) => ({h_formula})",
        assume_system_constraints=False,
        n_samples=128,
    )

    exact_cls = _classify_pair(res_relax, res_tight)
    if exact_cls is None:
        return {"relation": "modified", "is_weakened": True, "witness": None}

    if exact_cls["relation"] == "equivalent" and widened_bounds_override:
        if not (not b_map and not h_map and _are_symbolic_exprs_equivalent(base_expr_str, head_expr_str)):
            res_w_relax = prove_on_graph(
                cmp_graph,
                invariant_expr=f"({h_formula}) => ({b_formula})",
                assume_system_constraints=False,
                domain_bounds=widened_bounds_override,
                n_samples=128,
            )
            res_w_tight = prove_on_graph(
                cmp_graph,
                invariant_expr=f"({b_formula}) => ({h_formula})",
                assume_system_constraints=False,
                domain_bounds=widened_bounds_override,
                n_samples=128,
            )
            w_cls = _classify_pair(res_w_relax, res_w_tight)
            if w_cls is not None and w_cls["relation"] != "equivalent":
                return w_cls

    return exact_cls


class CausalDiffEngine:
    """Motor determinista de comparación diferencial de grafos causales (ActorDB Causal Diff)."""

    def __init__(
        self,
        optimality_tol: float = 1e-3,
        lever_shift_tol: float = 1e-4,
        fail_on_dead_nodes: bool = True,
    ):
        self.optimality_tol = float(optimality_tol)
        self.lever_shift_tol = float(lever_shift_tol)
        self.fail_on_dead_nodes = bool(fail_on_dead_nodes)
        self._oracle = ActorDBOracle(allow_auto_discovery=False)

    def _resolve_multi_submodel_adapter_pair(
        self,
        base: Union[CausalGraph, Dict[str, Any], str, Path],
        head: Union[CausalGraph, Dict[str, Any], str, Path],
        entrypoint: Optional[str],
    ) -> Optional[List[Tuple[str, CausalGraph, CausalGraph]]]:
        """Si `base` y `head` son el mismo repositorio adaptador multi-submodelo sin submodelo explícito, carga todos sus submodelos."""
        if entrypoint is not None or not isinstance(base, (str, Path)) or not isinstance(head, (str, Path)):
            return None
        sb_str = str(base).strip()
        sh_str = str(head).strip()
        if not sb_str or not sh_str:
            return None
        repo_eng = RepositoryCausalEngine()
        spec_b = repo_eng.resolve_adapter_source(sb_str)
        spec_h = repo_eng.resolve_adapter_source(sh_str)
        if (
            not spec_b
            or not spec_h
            or "error_code" in spec_b
            or "error_code" in spec_h
            or spec_b.get("repo_id") != spec_h.get("repo_id")
            or spec_b.get("explicit_submodel") is not None
            or spec_h.get("explicit_submodel") is not None
            or spec_b.get("file_hint") is not None
            or spec_h.get("file_hint") is not None
        ):
            return None
        adapter_b = repo_eng.get_adapter(spec_b["repo_id"], str(spec_b["repo_dir"]))
        adapter_h = repo_eng.get_adapter(spec_h["repo_id"], str(spec_h["repo_dir"]))
        avail_subs_b = adapter_b.list_submodels() if (adapter_b and hasattr(adapter_b, "list_submodels")) else []
        avail_subs_h = adapter_h.list_submodels() if (adapter_h and hasattr(adapter_h, "list_submodels")) else []
        shared_subs = [s for s in avail_subs_b if s in avail_subs_h]
        if len(shared_subs) <= 1:
            return None
        pairs: List[Tuple[str, CausalGraph, CausalGraph]] = []
        for sub_name in shared_subs:
            gb_sub = load_causal_graph(base, entrypoint=sub_name)
            gh_sub = load_causal_graph(head, entrypoint=sub_name)
            pairs.append((sub_name, gb_sub, gh_sub))
        return pairs

    @staticmethod
    def _has_any_structural_change(sdiff: Dict[str, Any]) -> bool:
        return bool(
            sdiff.get("nodes_added")
            or sdiff.get("nodes_removed")
            or sdiff.get("nodes_semantically_modified")
            or sdiff.get("edges_added")
            or sdiff.get("edges_removed")
            or sdiff.get("constraints_added")
            or sdiff.get("constraints_removed")
            or sdiff.get("constraints_semantically_modified")
        )

    def _merge_submodel_reports(
        self,
        sub_reports: List[Tuple[str, CausalDiffReport]],
        git_metadata: Optional[Dict[str, Any]],
    ) -> CausalDiffReport:
        rank_map = {DiffVerdict.FAIL.value: 2, DiffVerdict.WARN.value: 1, DiffVerdict.PASS.value: 0}
        ordered = sorted(
            sub_reports,
            key=lambda item: (
                rank_map.get(item[1].verdict, 0),
                len(item[1].invariant_regressions),
                int(bool(item[1].policy_optimality.get("is_regression"))),
            ),
            reverse=True,
        )
        _, primary = ordered[0]

        merged_inv_regs: List[Dict[str, Any]] = []
        merged_inv_fixed: List[Dict[str, Any]] = []
        merged_diags: List[Dict[str, Any]] = []
        merged_added_dead: List[Dict[str, Any]] = []
        merged_removed_dead: List[str] = []

        sdiff_merged = dict(primary.structural_diff)
        for list_k in (
            "nodes_added",
            "nodes_removed",
            "nodes_semantically_modified",
            "constraints_added",
            "constraints_removed",
            "constraints_semantically_modified",
        ):
            seen_items: List[str] = []
            for _, rep in sub_reports:
                for val_it in rep.structural_diff.get(list_k, []):
                    if val_it not in seen_items:
                        seen_items.append(val_it)
            sdiff_merged[list_k] = seen_items
        sdiff_merged["changed_submodels"] = [s for s, _ in sub_reports]

        for sub_n, rep in sub_reports:
            for ir_item in rep.invariant_regressions:
                tagged = dict(ir_item)
                tagged.setdefault("submodel", sub_n)
                merged_inv_regs.append(tagged)
            for if_item in rep.invariants_fixed:
                tagged_f = dict(if_item)
                tagged_f.setdefault("submodel", sub_n)
                merged_inv_fixed.append(tagged_f)
            for dg_item in rep.diagnostics:
                tagged_d = dict(dg_item)
                tagged_d.setdefault("submodel", sub_n)
                merged_diags.append(tagged_d)
            for dn_item in rep.dead_nodes.get("added_dead_nodes", []):
                tagged_dn = dict(dn_item)
                tagged_dn.setdefault("submodel", sub_n)
                merged_added_dead.append(tagged_dn)
            for r_id in rep.dead_nodes.get("removed_dead_node_ids", []):
                if r_id not in merged_removed_dead:
                    merged_removed_dead.append(r_id)

        has_regs = any(rep.has_regressions for _, rep in sub_reports)
        is_sem_eq = all(rep.is_semantically_equivalent for _, rep in sub_reports)
        verdict = ordered[0][1].verdict

        crit_cnt = sum(1 for d in merged_diags if d.get("severity") == "critical")
        warn_cnt = sum(1 for d in merged_diags if d.get("severity") == "warning")
        info_cnt = sum(1 for d in merged_diags if d.get("severity") == "info")

        dead_merged = dict(primary.dead_nodes)
        dead_merged["added_dead_nodes"] = merged_added_dead
        dead_merged["removed_dead_node_ids"] = merged_removed_dead

        summary = {
            "verdict": verdict,
            "has_regressions": has_regs,
            "is_semantically_equivalent": is_sem_eq,
            "changed_submodels": [s for s, _ in sub_reports],
            "critical_count": crit_cnt,
            "warning_count": warn_cnt,
            "info_count": info_cnt,
            "invariant_regressions_count": len(merged_inv_regs),
            "dead_nodes_added_count": len(merged_added_dead),
            "dead_nodes_removed_count": len(merged_removed_dead),
            "policy_is_regression": any(bool(r.policy_optimality.get("is_regression")) for _, r in sub_reports),
            "policy_changed": any(bool(r.policy_optimality.get("policy_changed")) for _, r in sub_reports),
            "old_policy_became_infeasible": any(
                bool(r.policy_optimality.get("old_policy_became_infeasible")) for _, r in sub_reports
            ),
        }
        return CausalDiffReport(
            base_system_id=primary.base_system_id,
            head_system_id=primary.head_system_id,
            verdict=verdict,
            has_regressions=has_regs,
            is_semantically_equivalent=is_sem_eq,
            summary=summary,
            invariant_regressions=merged_inv_regs,
            invariants_fixed=merged_inv_fixed,
            policy_optimality=primary.policy_optimality,
            dead_nodes=dead_merged,
            structural_diff=sdiff_merged,
            epistemic_diff=primary.epistemic_diff,
            diagnostics=merged_diags,
            git_metadata=git_metadata,
        )

    def diff(
        self,
        base: Union[CausalGraph, Dict[str, Any], str, Path],
        head: Union[CausalGraph, Dict[str, Any], str, Path],
        entrypoint: Optional[str] = None,
        git_metadata: Optional[Dict[str, Any]] = None,
    ) -> CausalDiffReport:
        """Compara `base` (antes) contra `head` (después) y retorna un `CausalDiffReport` formal."""
        submodel_pairs = self._resolve_multi_submodel_adapter_pair(base, head, entrypoint)
        if submodel_pairs is not None:
            changed_pairs = [
                (sub_n, gb_s, gh_s)
                for sub_n, gb_s, gh_s in submodel_pairs
                if self._has_any_structural_change(self._compute_structural_diff(gb_s, gh_s))
                or str((gb_s.metadata or {}).get("sense", "max")).lower()
                != str((gh_s.metadata or {}).get("sense", "max")).lower()
            ]
            if len(changed_pairs) == 1:
                _, gb_one, gh_one = changed_pairs[0]
                return self.diff(gb_one, gh_one, git_metadata=git_metadata)
            if len(changed_pairs) > 1:
                sub_reps = [
                    (sub_n, self.diff(gb_s, gh_s, git_metadata=git_metadata))
                    for sub_n, gb_s, gh_s in changed_pairs
                ]
                return self._merge_submodel_reports(sub_reps, git_metadata=git_metadata)

        base_graph = load_causal_graph(base, entrypoint=entrypoint)
        head_graph = load_causal_graph(head, entrypoint=entrypoint)

        diagnostics: List[Dict[str, Any]] = []
        invariant_regressions: List[Dict[str, Any]] = []
        invariants_fixed: List[Dict[str, Any]] = []

        # ------------------------------------------------------------------
        # 1. Diff Estructural y Equivalencia Simbólica de Mecanismos
        # ------------------------------------------------------------------
        structural_diff = self._compute_structural_diff(base_graph, head_graph)
        for mod_item in structural_diff.get("nodes_modified", []):
            bb_chg = (mod_item.get("changes") or {}).get("blackbox_mechanism")
            if isinstance(bb_chg, dict) and not bb_chg.get("equivalent", True):
                diagnostics.append({
                    "code": "BLACKBOX_MECHANISM_MODIFIED",
                    "category": "structural_diff",
                    "severity": "warning",
                    "node": mod_item["node"],
                    "witness": bb_chg.get("witness"),
                    "reason": (
                        f"Blackbox mechanism for actor '{mod_item['node']}' diverges numerically "
                        f"between base and head ({bb_chg.get('witness')})."
                    ),
                })

        base_cycles = _analyze_static_cycles(base_graph)
        head_cycles = _analyze_static_cycles(head_graph)
        base_cycle_sets = {frozenset(c["nodes"]) for c in base_cycles if c["is_ill_posed"]}
        new_ill_cycles = [c for c in head_cycles if c["is_ill_posed"] and frozenset(c["nodes"]) not in base_cycle_sets]
        structural_diff["cycles_in_base"] = base_cycles
        structural_diff["cycles_in_head"] = head_cycles
        structural_diff["new_ill_posed_cycles"] = new_ill_cycles

        for cyc in new_ill_cycles:
            reg = {
                "code": "ILL_POSED_CYCLE_INTRODUCED",
                "category": "invariant_regression",
                "severity": "critical",
                "nodes": cyc["nodes"],
                "classification": cyc["classification"],
                "reason": (
                    f"Simultaneous causal cycle {cyc['nodes']} (total lag Lambda=0, {cyc['classification']}) "
                    f"introduced in head graph."
                ),
            }
            invariant_regressions.append(reg)
            diagnostics.append(reg)

        base_is_dag = len([c for c in base_cycles if c["total_lag"] == 0]) == 0
        head_is_dag = len([c for c in head_cycles if c["total_lag"] == 0]) == 0

        # ------------------------------------------------------------------
        # 2. Diff Epistémico / Fidelidad
        # ------------------------------------------------------------------
        epistemic_diff = self._compute_epistemic_diff(base_graph, head_graph)
        for unv_nid in epistemic_diff["newly_unverified_nodes"]:
            reg = {
                "code": "EPISTEMIC_FIDELITY_REGRESSION",
                "category": "invariant_regression",
                "severity": "critical",
                "node": unv_nid,
                "reason": (
                    f"Actor '{unv_nid}' degraded from verified fidelity in base to unverified in head "
                    f"({epistemic_diff['head_unverified_reasons'].get(unv_nid, 'impure or divergent mechanism')})."
                ),
            }
            invariant_regressions.append(reg)
            diagnostics.append(reg)

        # ------------------------------------------------------------------
        # 3. Auditoría de Nodos Muertos (T1-A, T1-B, T1-C)
        # ------------------------------------------------------------------
        dead_nodes_diff = self._compute_dead_nodes_diff(base_graph, head_graph, base_is_dag, head_is_dag)
        for dn in dead_nodes_diff["added_dead_nodes"]:
            sev = "critical" if self.fail_on_dead_nodes else "warning"
            diag = {
                "code": "DEAD_NODE_ADDED",
                "category": "dead_node",
                "severity": sev,
                "node": dn["node"],
                "subtype": dn["subtype"],
                "origin": dn["origin"],
                "reason": dn["reason"],
            }
            diagnostics.append(diag)

        # ------------------------------------------------------------------
        # 4. Auditoría de Regresiones de Invariantes (Conservación T2, C, T3, IIS)
        # ------------------------------------------------------------------
        inv_regs, inv_fixed = self._compute_invariant_regressions(
            base_graph, head_graph, base_is_dag, head_is_dag
        )
        invariant_regressions.extend(inv_regs)
        invariants_fixed.extend(inv_fixed)
        diagnostics.extend(inv_regs)

        # ------------------------------------------------------------------
        # 5. Alteración de Optimalidad de Políticas (J*, X*, Regret Cruzado, T4)
        # ------------------------------------------------------------------
        policy_optimality = self._compute_policy_optimality_diff(
            base_graph, head_graph, base_is_dag, head_is_dag
        )
        if policy_optimality.get("status_regressed"):
            # Si el espacio factible colapsó a INFEASIBLE en head cuando base era factible, registrarlo también como regresión de invariantes/factibilidad
            if policy_optimality.get("head_status") == "INFEASIBLE" and policy_optimality.get("base_status") == "OPTIMAL":
                already = any(r.get("code") == "FEASIBLE_SPACE_COLLAPSED" for r in invariant_regressions)
                if not already:
                    f_reg = {
                        "code": "FEASIBLE_SPACE_COLLAPSED",
                        "category": "invariant_regression",
                        "severity": "critical",
                        "iis_constraints": policy_optimality.get("head_iis_constraints", []),
                        "reason": (
                            f"Admissible feasible space collapsed from OPTIMAL in base to INFEASIBLE in head "
                            f"(Minimal IIS: {policy_optimality.get('head_iis_constraints', [])})."
                        ),
                    }
                    invariant_regressions.append(f_reg)
                    diagnostics.append(f_reg)

        if policy_optimality.get("is_regression"):
            p_diag = {
                "code": "POLICY_OPTIMALITY_REGRESSION",
                "category": "policy_optimality",
                "severity": "critical",
                "target_id": policy_optimality.get("target_id"),
                "base_optimal_value": policy_optimality.get("base_optimal_value"),
                "head_optimal_value": policy_optimality.get("head_optimal_value"),
                "objective_delta": policy_optimality.get("objective_delta"),
                "reason": policy_optimality.get("summary_reason", "Optimal policy degraded in head."),
            }
            diagnostics.append(p_diag)

        if policy_optimality.get("old_policy_became_infeasible"):
            viol_list = (policy_optimality.get("base_policy_in_head") or {}).get("violated_constraints", [])
            diagnostics.append({
                "code": "PREVIOUS_POLICY_RENDERED_INFEASIBLE",
                "category": "policy_optimality",
                "severity": "critical",
                "target_id": policy_optimality.get("target_id"),
                "control_deltas": policy_optimality.get("control_deltas"),
                "objective_delta": policy_optimality.get("objective_delta"),
                "violated_constraints": viol_list,
                "reason": (
                    f"Previously optimal policy {policy_optimality.get('base_optimal_controls')} became INFEASIBLE in head "
                    f"(violates {viol_list})."
                ),
            })
        elif not policy_optimality.get("is_regression") and policy_optimality.get("policy_changed"):
            diagnostics.append({
                "code": "POLICY_OPTIMALITY_SHIFTED",
                "category": "policy_optimality",
                "severity": "warning",
                "target_id": policy_optimality.get("target_id"),
                "control_deltas": policy_optimality.get("control_deltas"),
                "objective_delta": policy_optimality.get("objective_delta"),
                "reason": policy_optimality.get("summary_reason", "Optimal policy shifted between base and head."),
            })

        for t4_reg in policy_optimality.get("local_optima_regressions", []):
            diagnostics.append(t4_reg)

        # ------------------------------------------------------------------
        # 6. Síntesis del Veredicto y Equivalencia Semántica
        # ------------------------------------------------------------------
        has_dead_reg = bool(dead_nodes_diff["added_dead_nodes"]) and self.fail_on_dead_nodes
        has_policy_reg = bool(policy_optimality.get("is_regression")) or bool(policy_optimality.get("old_policy_became_infeasible"))
        has_inv_reg = len(invariant_regressions) > 0

        has_regressions = bool(has_inv_reg or has_policy_reg or has_dead_reg)

        is_semantically_equivalent = bool(
            not structural_diff["nodes_added"]
            and not structural_diff["nodes_removed"]
            and not structural_diff["nodes_semantically_modified"]
            and not structural_diff["constraints_added"]
            and not structural_diff["constraints_removed"]
            and not structural_diff["constraints_semantically_modified"]
            and not has_regressions
            and not policy_optimality.get("policy_changed")
            and not dead_nodes_diff["added_dead_nodes"]
            and not dead_nodes_diff["removed_dead_node_ids"]
        )

        critical_count = sum(1 for d in diagnostics if d.get("severity") == "critical")
        warning_count = sum(1 for d in diagnostics if d.get("severity") == "warning")
        info_count = sum(1 for d in diagnostics if d.get("severity") == "info")

        if has_regressions or critical_count > 0:
            verdict = DiffVerdict.FAIL.value
        elif warning_count > 0 or bool(dead_nodes_diff["added_dead_nodes"]) or bool(policy_optimality.get("policy_changed")):
            verdict = DiffVerdict.WARN.value
        else:
            verdict = DiffVerdict.PASS.value

        summary = {
            "verdict": verdict,
            "has_regressions": has_regressions,
            "is_semantically_equivalent": is_semantically_equivalent,
            "critical_count": critical_count,
            "warning_count": warning_count,
            "info_count": info_count,
            "invariant_regressions_count": len(invariant_regressions),
            "dead_nodes_added_count": len(dead_nodes_diff["added_dead_nodes"]),
            "dead_nodes_removed_count": len(dead_nodes_diff["removed_dead_node_ids"]),
            "policy_is_regression": bool(policy_optimality.get("is_regression")),
            "policy_changed": bool(policy_optimality.get("policy_changed")),
            "old_policy_became_infeasible": bool(policy_optimality.get("old_policy_became_infeasible")),
        }

        return CausalDiffReport(
            base_system_id=base_graph.system_id,
            head_system_id=head_graph.system_id,
            verdict=verdict,
            has_regressions=has_regressions,
            is_semantically_equivalent=is_semantically_equivalent,
            summary=summary,
            invariant_regressions=invariant_regressions,
            invariants_fixed=invariants_fixed,
            policy_optimality=policy_optimality,
            dead_nodes=dead_nodes_diff,
            structural_diff=structural_diff,
            epistemic_diff=epistemic_diff,
            diagnostics=diagnostics,
            git_metadata=git_metadata,
        )

    def diff_git(
        self,
        repo_path: Union[str, Path],
        base_ref: str = "HEAD~1",
        head_ref: str = "HEAD",
        subpath: Optional[str] = None,
        entrypoint: Optional[str] = None,
    ) -> CausalDiffReport:
        """Ejecuta Causal Diff entre dos revisiones Git (`base_ref` vs `head_ref` o `'WORKTREE'`)."""
        from .git_integration import diff_git_revisions
        return diff_git_revisions(
            repo_path=repo_path,
            base_ref=base_ref,
            head_ref=head_ref,
            subpath=subpath,
            entrypoint=entrypoint,
            engine=self,
        )

    # ------------------------------------------------------------------
    # Sub-analizadores Internos
    # ------------------------------------------------------------------

    def _are_blackbox_nodes_equivalent(
        self,
        base_graph: CausalGraph,
        head_graph: CausalGraph,
        nid: str,
        probe_states: Optional[List[Dict[str, float]]] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Compara determinísticamente mecanismos `_blackbox_callables` de `nid` sobre puntos de prueba del dominio."""
        bb_b = getattr(base_graph, "_blackbox_callables", {}) or {}
        bb_h = getattr(head_graph, "_blackbox_callables", {}) or {}
        if nid not in bb_b and nid not in bb_h:
            return True, None
        if (nid in bb_b) != (nid in bb_h):
            return False, {"reason": "callable_presence_changed"}
        args_b, fn_b = bb_b[nid]
        args_h, fn_h = bb_h[nid]
        if list(args_b) != list(args_h):
            return False, {"reason": "callable_signature_changed", "base_args": list(args_b), "head_args": list(args_h)}

        if probe_states is None:
            shared_leaves = sorted(
                lid
                for lid in (set(base_graph.nodes) & set(head_graph.nodes))
                if base_graph.nodes[lid].category in ("exogenous", "control")
                and head_graph.nodes[lid].category in ("exogenous", "control")
            )
            b_map, int_doms = _compute_leaf_bounds(base_graph, shared_leaves)
            leaf_probes: List[Dict[str, float]] = []
            nom_p = {
                lid: float(
                    base_graph.nodes[lid].current_value
                    if isinstance(base_graph.nodes[lid].current_value, (int, float))
                    else 0.5 * (b_map[lid][0] + b_map[lid][1])
                )
                for lid in shared_leaves
            }
            leaf_probes.append(nom_p)
            if shared_leaves:
                leaf_probes.append({lid: b_map[lid][0] for lid in shared_leaves})
                leaf_probes.append({lid: b_map[lid][1] for lid in shared_leaves})
                for frac in (0.23, 0.5, 0.77):
                    pt_f: Dict[str, float] = {}
                    for idx_l, lid in enumerate(shared_leaves):
                        lo_l, hi_l = b_map[lid]
                        u_l = ((frac + 0.17 * idx_l) % 0.94) + 0.03
                        v_l = lo_l + u_l * (hi_l - lo_l)
                        pt_f[lid] = float(int(round(v_l))) if lid in int_doms else v_l
                    leaf_probes.append(pt_f)

            dag_b, _, _, fns_b = self._oracle._compile_graph_mechanisms(base_graph)
            probe_states = []
            if nx.is_directed_acyclic_graph(dag_b):
                for lp in leaf_probes:
                    try:
                        st_p = self._oracle._evaluate_forward_with_do(base_graph, dag_b, fns_b, interventions=lp)
                        probe_states.append(st_p)
                    except Exception:
                        continue

        for st_probe in probe_states or []:
            if not all(a in st_probe for a in args_b):
                continue
            arg_tuple = [float(st_probe[a]) for a in args_b]
            err_b = False
            err_h = False
            vb = math.nan
            vh = math.nan
            try:
                vb = float(fn_b(*arg_tuple))
            except Exception:
                err_b = True
            try:
                vh = float(fn_h(*arg_tuple))
            except Exception:
                err_h = True
            if err_b != err_h:
                return False, {
                    "args": dict(zip(args_b, arg_tuple, strict=True)),
                    "base_error": err_b,
                    "head_error": err_h,
                }
            if not err_b:
                if math.isfinite(vb) != math.isfinite(vh):
                    return False, {
                        "args": dict(zip(args_b, arg_tuple, strict=True)),
                        "base_value": vb,
                        "head_value": vh,
                    }
                if math.isfinite(vb) and abs(vb - vh) > 1e-6 * max(1.0, abs(vb), abs(vh)):
                    return False, {
                        "args": {k_a: round(v_a, 6) for k_a, v_a in zip(args_b, arg_tuple, strict=True)},
                        "base_value": round(vb, 6),
                        "head_value": round(vh, 6),
                    }
        return True, None

    def _compute_structural_diff(
        self, base_graph: CausalGraph, head_graph: CausalGraph
    ) -> Dict[str, Any]:
        b_nodes = set(base_graph.nodes.keys())
        h_nodes = set(head_graph.nodes.keys())

        nodes_added = sorted(h_nodes - b_nodes)
        nodes_removed = sorted(b_nodes - h_nodes)
        nodes_modified: List[Dict[str, Any]] = []
        nodes_semantically_modified: List[str] = []

        bb_present = bool(
            getattr(base_graph, "_blackbox_callables", None) or getattr(head_graph, "_blackbox_callables", None)
        )
        cached_bb_probes: Optional[List[Dict[str, float]]] = None

        for nid in sorted(b_nodes & h_nodes):
            nb = base_graph.nodes[nid]
            nh = head_graph.nodes[nid]
            changes: Dict[str, Any] = {}
            sem_changed = False

            if nb.category != nh.category:
                changes["category"] = {"base": nb.category, "head": nh.category}
                sem_changed = True
            if nb.data_type != nh.data_type:
                changes["data_type"] = {"base": nb.data_type, "head": nh.data_type}
                sem_changed = True
            if (nb.bounds or {}) != (nh.bounds or {}):
                changes["bounds"] = {"base": nb.bounds, "head": nh.bounds}
                sem_changed = True
            if nb.current_value != nh.current_value and nb.category in ("exogenous", "control"):
                changes["current_value"] = {"base": nb.current_value, "head": nh.current_value}
                sem_changed = True
            if (nb.symbolic_expr or "") != (nh.symbolic_expr or ""):
                alg_eq = _are_symbolic_exprs_equivalent(nb.symbolic_expr, nh.symbolic_expr)
                changes["symbolic_expr"] = {
                    "base": nb.symbolic_expr,
                    "head": nh.symbolic_expr,
                    "algebraically_equivalent": alg_eq,
                }
                if not alg_eq:
                    sem_changed = True
            elif bb_present and nb.symbolic_expr is None and nh.symbolic_expr is None:
                bb_b_map = getattr(base_graph, "_blackbox_callables", {}) or {}
                bb_h_map = getattr(head_graph, "_blackbox_callables", {}) or {}
                if nid in bb_b_map or nid in bb_h_map:
                    if cached_bb_probes is None:
                        shared_leaves = sorted(
                            lid
                            for lid in (b_nodes & h_nodes)
                            if base_graph.nodes[lid].category in ("exogenous", "control")
                            and head_graph.nodes[lid].category in ("exogenous", "control")
                        )
                        b_map, int_doms = _compute_leaf_bounds(base_graph, shared_leaves)
                        leaf_probes: List[Dict[str, float]] = [
                            {
                                lid: float(
                                    base_graph.nodes[lid].current_value
                                    if isinstance(base_graph.nodes[lid].current_value, (int, float))
                                    else 0.5 * (b_map[lid][0] + b_map[lid][1])
                                )
                                for lid in shared_leaves
                            }
                        ]
                        if shared_leaves:
                            leaf_probes.append({lid: b_map[lid][0] for lid in shared_leaves})
                            leaf_probes.append({lid: b_map[lid][1] for lid in shared_leaves})
                            for frac in (0.23, 0.5, 0.77):
                                pt_f = {}
                                for idx_l, lid in enumerate(shared_leaves):
                                    lo_l, hi_l = b_map[lid]
                                    u_l = ((frac + 0.17 * idx_l) % 0.94) + 0.03
                                    v_l = lo_l + u_l * (hi_l - lo_l)
                                    pt_f[lid] = float(int(round(v_l))) if lid in int_doms else v_l
                                leaf_probes.append(pt_f)
                        dag_b, _, _, fns_b = self._oracle._compile_graph_mechanisms(base_graph)
                        cached_bb_probes = []
                        if nx.is_directed_acyclic_graph(dag_b):
                            for lp in leaf_probes:
                                try:
                                    cached_bb_probes.append(
                                        self._oracle._evaluate_forward_with_do(base_graph, dag_b, fns_b, interventions=lp)
                                    )
                                except Exception:
                                    continue
                    bb_eq, bb_wit = self._are_blackbox_nodes_equivalent(
                        base_graph, head_graph, nid, probe_states=cached_bb_probes
                    )
                    if not bb_eq:
                        changes["blackbox_mechanism"] = {
                            "equivalent": False,
                            "witness": bb_wit,
                        }
                        sem_changed = True

            if changes:
                nodes_modified.append({
                    "node": nid,
                    "semantically_changed": sem_changed,
                    "changes": changes,
                })
                if sem_changed:
                    nodes_semantically_modified.append(nid)

        b_edges = {(e.source, e.target, e.type) for e in base_graph.edges}
        h_edges = {(e.source, e.target, e.type) for e in head_graph.edges}
        edges_added = [{"source": u, "target": v, "type": t} for u, v, t in sorted(h_edges - b_edges)]
        edges_removed = [{"source": u, "target": v, "type": t} for u, v, t in sorted(b_edges - h_edges)]

        b_cids = set(base_graph.constraints.keys())
        h_cids = set(head_graph.constraints.keys())
        constraints_added = sorted(h_cids - b_cids)
        constraints_removed = sorted(b_cids - h_cids)
        constraints_modified: List[Dict[str, Any]] = []
        constraints_semantically_modified: List[str] = []

        for cid in sorted(b_cids & h_cids):
            cb = base_graph.constraints[cid]
            ch = head_graph.constraints[cid]
            if cb.expression.strip() != ch.expression.strip() or cb.type != ch.type:
                cmp_info = _compare_constraint_strength(base_graph, head_graph, cb.expression, ch.expression)
                is_eq = (cb.type == ch.type) and (cmp_info["relation"] == "equivalent")
                constraints_modified.append({
                    "constraint_id": cid,
                    "base_type": cb.type,
                    "head_type": ch.type,
                    "base_expression": cb.expression,
                    "head_expression": ch.expression,
                    "relation": cmp_info["relation"],
                    "is_weakened": bool(cmp_info["is_weakened"] or (cb.type == "hard" and ch.type == "soft")),
                    "witness": cmp_info.get("witness"),
                })
                if not is_eq:
                    constraints_semantically_modified.append(cid)

        return {
            "nodes_added": nodes_added,
            "nodes_removed": nodes_removed,
            "nodes_modified": nodes_modified,
            "nodes_semantically_modified": nodes_semantically_modified,
            "edges_added": edges_added,
            "edges_removed": edges_removed,
            "constraints_added": constraints_added,
            "constraints_removed": constraints_removed,
            "constraints_modified": constraints_modified,
            "constraints_semantically_modified": constraints_semantically_modified,
        }

    def _compute_epistemic_diff(
        self, base_graph: CausalGraph, head_graph: CausalGraph
    ) -> Dict[str, Any]:
        def _extract_unverified(g: CausalGraph) -> Tuple[Set[str], Dict[str, str]]:
            unv: Set[str] = set()
            reasons: Dict[str, str] = {}
            fid_rep = (g.metadata or {}).get("fidelity_report")
            if isinstance(fid_rep, dict):
                for nid in fid_rep.get("unverified_nodes") or []:
                    unv.add(nid)
                for nid, r in (fid_rep.get("unverified_reasons") or {}).items():
                    reasons[nid] = str(r)
            for nid, node in g.nodes.items():
                if isinstance(node.fidelity, dict) and not node.fidelity.get("verified", True):
                    unv.add(nid)
                    if node.fidelity.get("reason"):
                        reasons.setdefault(nid, str(node.fidelity["reason"]))
            return unv, reasons

        b_unv, _ = _extract_unverified(base_graph)
        h_unv, h_reasons = _extract_unverified(head_graph)
        newly_unverified = sorted(h_unv - b_unv)
        newly_verified = sorted(b_unv - h_unv)
        return {
            "base_unverified_nodes": sorted(b_unv),
            "head_unverified_nodes": sorted(h_unv),
            "newly_unverified_nodes": newly_unverified,
            "newly_verified_nodes": newly_verified,
            "head_unverified_reasons": h_reasons,
        }

    def _compute_dead_nodes_diff(
        self,
        base_graph: CausalGraph,
        head_graph: CausalGraph,
        base_is_dag: bool,
        head_is_dag: bool,
    ) -> Dict[str, Any]:
        base_dead_map: Dict[str, Dict[str, Any]] = {}
        head_dead_map: Dict[str, Dict[str, Any]] = {}

        if base_is_dag and base_graph.nodes:
            tr_b = self._oracle.trace(graph_data=base_graph)
            if tr_b.get("status") == "OK":
                for d in tr_b.get("dead_nodes", []):
                    base_dead_map[d["node"]] = d

        if head_is_dag and head_graph.nodes:
            tr_h = self._oracle.trace(graph_data=head_graph)
            if tr_h.get("status") == "OK":
                for d in tr_h.get("dead_nodes", []):
                    head_dead_map[d["node"]] = d

        added_dead_nodes: List[Dict[str, Any]] = []
        for nid in sorted(set(head_dead_map.keys()) - set(base_dead_map.keys())):
            info = dict(head_dead_map[nid])
            info["origin"] = "newly_added_actor" if nid not in base_graph.nodes else "disconnected_existing_actor"
            added_dead_nodes.append(info)

        removed_dead_ids = sorted(set(base_dead_map.keys()) - set(head_dead_map.keys()))

        return {
            "base_dead_nodes": sorted(base_dead_map.keys()),
            "head_dead_nodes": sorted(head_dead_map.keys()),
            "added_dead_nodes": added_dead_nodes,
            "added_dead_node_ids": [d["node"] for d in added_dead_nodes],
            "removed_dead_node_ids": removed_dead_ids,
            "has_new_dead_nodes": len(added_dead_nodes) > 0,
        }

    def _compute_invariant_regressions(
        self,
        base_graph: CausalGraph,
        head_graph: CausalGraph,
        base_is_dag: bool,
        head_is_dag: bool,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        regressions: List[Dict[str, Any]] = []
        fixed: List[Dict[str, Any]] = []

        # 1. Invariantes de Conservación (T2: conservation_checks)
        b_cons_viol = {
            (chk.get("constraint_id", "c_ledger"), chk.get("function", "")): chk
            for chk in (base_graph.metadata or {}).get("conservation_checks", [])
            if not chk.get("is_balanced", True)
        }
        h_cons_viol = {
            (chk.get("constraint_id", "c_ledger"), chk.get("function", "")): chk
            for chk in (head_graph.metadata or {}).get("conservation_checks", [])
            if not chk.get("is_balanced", True)
        }
        for key, chk in h_cons_viol.items():
            if key not in b_cons_viol:
                regressions.append({
                    "code": "CONSERVATION_INVARIANT_BROKEN",
                    "category": "invariant_regression",
                    "severity": "critical",
                    "constraint_id": key[0],
                    "function": key[1],
                    "file": chk.get("file"),
                    "symbolic_residual": chk.get("symbolic_residual"),
                    "reason": (
                        f"Conservation invariant '{key[0]}' broken in function '{key[1]}' "
                        f"with symbolic residual R = {chk.get('symbolic_residual')} != 0."
                    ),
                })
        for key, _chk in b_cons_viol.items():
            if key not in h_cons_viol:
                fixed.append({
                    "code": "CONSERVATION_INVARIANT_RESTORED",
                    "constraint_id": key[0],
                    "function": key[1],
                    "reason": f"Previously unbalanced conservation check '{key[0]}' in '{key[1]}' is now balanced.",
                })

        # 2. Eliminación o debilitamiento/relajación de restricciones duras declaradas en C
        for cid, cb in base_graph.constraints.items():
            if cb.type != "hard":
                continue
            if cid not in head_graph.constraints:
                regressions.append({
                    "code": "HARD_CONSTRAINT_REMOVED",
                    "category": "invariant_regression",
                    "severity": "critical",
                    "constraint_id": cid,
                    "base_expression": cb.expression,
                    "reason": f"Hard invariant/constraint '{cid}' ({cb.expression}) was removed in head.",
                })
            else:
                ch = head_graph.constraints[cid]
                if ch.type != "hard":
                    regressions.append({
                        "code": "HARD_CONSTRAINT_DOWNGRADED",
                        "category": "invariant_regression",
                        "severity": "critical",
                        "constraint_id": cid,
                        "base_expression": cb.expression,
                        "reason": f"Hard constraint '{cid}' was downgraded from 'hard' to '{ch.type}' in head.",
                    })
                elif cb.expression.strip() != ch.expression.strip():
                    cmp_c = _compare_constraint_strength(base_graph, head_graph, cb.expression, ch.expression)
                    if cmp_c["is_weakened"]:
                        regressions.append({
                            "code": "HARD_CONSTRAINT_WEAKENED",
                            "category": "invariant_regression",
                            "severity": "critical",
                            "constraint_id": cid,
                            "base_expression": cb.expression,
                            "head_expression": ch.expression,
                            "relation": cmp_c["relation"],
                            "witness": cmp_c.get("witness"),
                            "reason": (
                                f"Hard constraint '{cid}' was {cmp_c['relation']} from '{cb.expression}' "
                                f"to '{ch.expression}', admitting previously forbidden operating states."
                            ),
                        })

        # 2b. Relajación o eliminación de cotas duras en palancas de control (X_BOUNDS)
        for nid in sorted(set(base_graph.nodes.keys()) & set(head_graph.nodes.keys())):
            nb = base_graph.nodes[nid]
            nh = head_graph.nodes[nid]
            if nb.category != "control" or nh.category != "control":
                continue
            b_bounds = nb.bounds or {}
            h_bounds = nh.bounds or {}
            b_min, b_max = b_bounds.get("min"), b_bounds.get("max")
            h_min, h_max = h_bounds.get("min"), h_bounds.get("max")
            min_relaxed = (b_min is not None) and (h_min is None or float(h_min) < float(b_min) - 1e-9)
            max_relaxed = (b_max is not None) and (h_max is None or float(h_max) > float(b_max) + 1e-9)
            if min_relaxed or max_relaxed:
                regressions.append({
                    "code": "CONTROL_BOUNDS_WEAKENED",
                    "category": "invariant_regression",
                    "severity": "critical",
                    "node": nid,
                    "base_bounds": b_bounds,
                    "head_bounds": h_bounds,
                    "reason": (
                        f"Hard control box bounds on '{nid}' were relaxed from {b_bounds} in base "
                        f"to {h_bounds} in head, expanding the admissible action space."
                    ),
                })

        # 3. Violaciones de restricciones duras en el estado nominal actual (`do({})`)
        base_nom_viol: Set[str] = set()
        if base_is_dag and base_graph.nodes:
            b_nom = self._oracle.do(interventions={}, graph_data=base_graph)
            if b_nom.get("status") == "OK":
                base_nom_viol = set(b_nom.get("violated_constraints", []))

        if head_is_dag and head_graph.nodes:
            h_nom = self._oracle.do(interventions={}, graph_data=head_graph)
            if h_nom.get("status") == "OK":
                head_nom_viol = set(h_nom.get("violated_constraints", []))
                margins_h = h_nom.get("constraint_margins", {})
                for v_cid in sorted(head_nom_viol - base_nom_viol):
                    regressions.append({
                        "code": "NOMINAL_INVARIANT_VIOLATION",
                        "category": "invariant_regression",
                        "severity": "critical",
                        "constraint_id": v_cid,
                        "margin": margins_h.get(v_cid),
                        "reason": (
                            f"Nominal operating state in head violates hard constraint '{v_cid}' "
                            f"(safety margin = {margins_h.get(v_cid)})."
                        ),
                    })
                for f_cid in sorted(base_nom_viol - head_nom_viol):
                    fixed.append({
                        "code": "NOMINAL_INVARIANT_RESTORED",
                        "constraint_id": f_cid,
                        "reason": f"Hard constraint '{f_cid}' violated in base nominal state is now satisfied in head.",
                    })

        # 4. Invariantes universales sobre estados/shocks exógenos falsificadas en head pero válidas en base
        if head_is_dag and head_graph.nodes:
            control_set_h = {nid for nid, n in head_graph.nodes.items() if n.category == "control"}
            for cid, ch in head_graph.constraints.items():
                if ch.type != "hard" or "sum(" in (ch.expression or ""):
                    continue
                vars_in_c = set(_extract_identifiers(ch.expression))
                # Auditar leyes sobre estados / variables exógenas (ej. conservación, saldos >= 0, SLAs)
                is_state_or_exo_law = bool(vars_in_c) and not vars_in_c.issubset(control_set_h)
                if not is_state_or_exo_law:
                    continue
                # Verificar sobre el dominio admisible Omega_C de head
                p_head = prove_on_graph(head_graph, constraint_id=cid, assume_system_constraints=True, n_samples=64)
                if p_head.get("status") == "FALSIFIED":
                    was_falsified_in_base = False
                    if base_is_dag and cid in base_graph.constraints:
                        p_base = prove_on_graph(base_graph, constraint_id=cid, assume_system_constraints=True, n_samples=64)
                        was_falsified_in_base = (p_base.get("status") == "FALSIFIED")
                    if not was_falsified_in_base:
                        already_nom = any(
                            r.get("code") == "NOMINAL_INVARIANT_VIOLATION" and r.get("constraint_id") == cid
                            for r in regressions
                        )
                        if not already_nom:
                            regressions.append({
                                "code": "STATE_INVARIANT_FALSIFIED",
                                "category": "invariant_regression",
                                "severity": "critical",
                                "constraint_id": cid,
                                "expression": ch.expression,
                                "counterexample": p_head.get("counterexample"),
                                "violation_magnitude": p_head.get("violation_magnitude"),
                                "reason": (
                                    f"Hard state invariant '{cid}' ({ch.expression}) is falsified in head "
                                    f"at witness {p_head.get('counterexample')}."
                                ),
                            })

        # 5. Discrepancias Normativas Documentación vs Código (T3)
        b_t3 = {
            (d.get("node"), d.get("doc_file")): d
            for d in (base_graph.metadata or {}).get("doc_discrepancies", [])
        }
        h_t3 = {
            (d.get("node"), d.get("doc_file")): d
            for d in (head_graph.metadata or {}).get("doc_discrepancies", [])
        }
        for key, d_item in h_t3.items():
            prev = b_t3.get(key)
            if prev is None or (prev.get("code_value") != d_item.get("code_value")) or (prev.get("doc_value") != d_item.get("doc_value")):
                regressions.append({
                    "code": "DOC_CODE_DISCREPANCY_INTRODUCED",
                    "category": "invariant_regression",
                    "severity": "critical",
                    "node": d_item.get("node"),
                    "doc_file": d_item.get("doc_file"),
                    "doc_value": d_item.get("doc_value"),
                    "code_value": d_item.get("code_value"),
                    "reason": (d_item.get("details") or {}).get(
                        "reason",
                        f"Normative documentation vs code discrepancy introduced on actor '{d_item.get('node')}'.",
                    ),
                })

        return regressions, fixed

    def _compute_policy_optimality_diff(
        self,
        base_graph: CausalGraph,
        head_graph: CausalGraph,
        base_is_dag: bool,
        head_is_dag: bool,
    ) -> Dict[str, Any]:
        base_sense = str((base_graph.metadata or {}).get("sense", "max")).lower()
        head_sense = str((head_graph.metadata or {}).get("sense", "max")).lower()
        base_sense = "min" if base_sense.startswith("min") else "max"
        head_sense = "min" if head_sense.startswith("min") else "max"

        b_target, _ = self._oracle._resolve_target_id(base_graph, None)
        h_target, _ = self._oracle._resolve_target_id(head_graph, None)

        sol_b = (
            self._oracle.solve(target_id=b_target, sense=base_sense, graph_data=base_graph)
            if (base_is_dag and b_target is not None)
            else {"status": "UNSUPPORTED", "controls": {}, "objective_value": None}
        )
        sol_h = (
            self._oracle.solve(target_id=h_target, sense=head_sense, graph_data=head_graph)
            if (head_is_dag and h_target is not None)
            else {"status": "UNSUPPORTED", "controls": {}, "objective_value": None}
        )

        b_status = sol_b.get("status", "UNSUPPORTED")
        h_status = sol_h.get("status", "UNSUPPORTED")
        b_cert = sol_b.get("certificate_status")
        h_cert = sol_h.get("certificate_status")
        b_val = sol_b.get("objective_value")
        h_val = sol_h.get("objective_value")
        b_ctrls = sol_b.get("controls") or {}
        h_ctrls = sol_h.get("controls") or {}

        status_regressed = (b_status == "OPTIMAL" and h_status != "OPTIMAL")
        cert_degraded = (
            b_status == "OPTIMAL"
            and h_status == "OPTIMAL"
            and b_cert == "CERTIFIED_OPTIMAL"
            and h_cert != "CERTIFIED_OPTIMAL"
        )
        sense_changed = (base_sense != head_sense)
        target_changed = (b_target != h_target)

        obj_delta: Optional[float] = None
        rel_obj_delta: Optional[float] = None
        direction = "incomparable"
        is_regression = bool(status_regressed or cert_degraded)

        if b_status == "OPTIMAL" and h_status == "OPTIMAL" and b_val is not None and h_val is not None:
            if not sense_changed and not target_changed:
                obj_delta = float(h_val) - float(b_val)
                rel_obj_delta = obj_delta / max(1.0, abs(float(b_val)))
                if abs(obj_delta) <= self.optimality_tol:
                    direction = "unchanged"
                elif head_sense == "max":
                    if obj_delta < -self.optimality_tol:
                        direction = "degraded"
                        is_regression = True
                    else:
                        direction = "improved"
                else:  # head_sense == "min"
                    if obj_delta > self.optimality_tol:
                        direction = "degraded"
                        is_regression = True
                    else:
                        direction = "improved"
            else:
                direction = "sense_or_target_changed"
        elif status_regressed:
            direction = "status_regressed"
        elif b_status != "OPTIMAL" and h_status == "OPTIMAL":
            direction = "status_improved"

        # Desplazamiento de las palancas óptimas X*
        shared_ctrls = sorted(set(b_ctrls.keys()) & set(h_ctrls.keys()))
        ctrl_deltas: Dict[str, float] = {
            cid: round(float(h_ctrls[cid]) - float(b_ctrls[cid]), 6)
            for cid in shared_ctrls
        }
        policy_shifted = bool(
            any(abs(d) > self.lever_shift_tol for d in ctrl_deltas.values())
            or (set(b_ctrls.keys()) != set(h_ctrls.keys()) and b_status == "OPTIMAL" and h_status == "OPTIMAL")
        )

        # Evaluación cruzada: ¿Qué ocurre si aplicamos la política óptima anterior X*_base en el nuevo grafo head?
        base_policy_in_head: Optional[Dict[str, Any]] = None
        old_policy_became_infeasible = False
        if b_status == "OPTIMAL" and b_ctrls and head_is_dag and h_target is not None:
            shared_interventions = {cid: b_ctrls[cid] for cid in shared_ctrls if cid in head_graph.nodes}
            if shared_interventions:
                cross_do = self._oracle.do(
                    interventions=shared_interventions,
                    target_id=h_target,
                    graph_data=head_graph,
                )
                if cross_do.get("status") == "OK":
                    cross_feas = bool(cross_do.get("is_feasible", True))
                    cross_j = float(cross_do["intervened_target_value"])
                    regret = None
                    if h_val is not None:
                        regret = (float(h_val) - cross_j) if head_sense == "max" else (cross_j - float(h_val))
                        regret = round(max(0.0, regret), 6)
                    old_policy_became_infeasible = not cross_feas
                    base_policy_in_head = {
                        "evaluated_controls": shared_interventions,
                        "is_feasible": cross_feas,
                        "violated_constraints": cross_do.get("violated_constraints", []),
                        "objective_value": round(cross_j, 6),
                        "regret_vs_head_optimum": regret,
                    }

        # Auditar optimizadores locales miopes (T4)
        local_optima_regressions: List[Dict[str, Any]] = []
        if head_is_dag and head_graph.nodes and (head_graph.metadata or {}).get("local_optimizers"):
            try:
                h_t4 = [d for d in SystemAuditor(graph=head_graph).audit_all() if d.get("id") == "T4"]
                b_t4_map = {}
                if base_is_dag and base_graph.nodes and (base_graph.metadata or {}).get("local_optimizers"):
                    b_t4_map = {
                        d.get("function"): float(d.get("bilevel_loss", 0.0))
                        for d in SystemAuditor(graph=base_graph).audit_all()
                        if d.get("id") == "T4"
                    }
                for d4 in h_t4:
                    fn_name = d4.get("function")
                    loss_h = float(d4.get("bilevel_loss", 0.0))
                    loss_b = b_t4_map.get(fn_name)
                    if loss_b is None or loss_h > loss_b + self.optimality_tol:
                        is_regression = True
                        local_optima_regressions.append({
                            "code": "LOCAL_OPTIMA_TRAP_REGRESSION",
                            "category": "policy_optimality",
                            "severity": "critical",
                            "function": fn_name,
                            "bilevel_loss": loss_h,
                            "previous_bilevel_loss": loss_b,
                            "reason": (d4.get("details") or {}).get(
                                "reason", f"Myopic local optimizer '{fn_name}' causes bilevel loss {loss_h:.2f}."
                            ),
                        })
            except Exception:
                pass

        b_active = set(sol_b.get("active_constraints") or [])
        h_active = set(sol_h.get("active_constraints") or [])
        active_added = sorted(h_active - b_active)
        active_removed = sorted(b_active - h_active)

        policy_changed = bool(
            is_regression
            or policy_shifted
            or direction in ("improved", "degraded", "status_improved", "sense_or_target_changed")
            or sense_changed
            or target_changed
            or old_policy_became_infeasible
        )

        if status_regressed:
            summary_reason = f"Solver status regressed from {b_status} to {h_status}."
        elif direction == "degraded" and obj_delta is not None:
            summary_reason = (
                f"Optimal {head_sense} objective '{h_target}' degraded from {b_val:.4f} to {h_val:.4f} "
                f"(delta = {obj_delta:+.4f})."
            )
        elif old_policy_became_infeasible:
            summary_reason = (
                f"Previously optimal policy {b_ctrls} became INFEASIBLE in head "
                f"(violates {base_policy_in_head['violated_constraints'] if base_policy_in_head else []})."
            )
        elif policy_shifted:
            summary_reason = f"Optimal control policy shifted ({ctrl_deltas}) with objective direction '{direction}'."
        else:
            summary_reason = f"Policy optimality status '{direction}'."

        return {
            "target_id": h_target,
            "base_target_id": b_target,
            "sense": head_sense,
            "base_sense": base_sense,
            "sense_changed": sense_changed,
            "target_changed": target_changed,
            "base_status": b_status,
            "head_status": h_status,
            "base_certificate_status": b_cert,
            "head_certificate_status": h_cert,
            "status_regressed": status_regressed,
            "certificate_degraded": cert_degraded,
            "base_optimal_value": b_val,
            "head_optimal_value": h_val,
            "objective_delta": round(obj_delta, 6) if obj_delta is not None else None,
            "relative_objective_delta": round(rel_obj_delta, 6) if rel_obj_delta is not None else None,
            "direction": direction,
            "is_regression": is_regression,
            "policy_shifted": policy_shifted,
            "policy_changed": policy_changed,
            "old_policy_became_infeasible": old_policy_became_infeasible,
            "base_optimal_controls": b_ctrls,
            "head_optimal_controls": h_ctrls,
            "control_deltas": ctrl_deltas,
            "base_policy_in_head": base_policy_in_head,
            "active_constraints_added": active_added,
            "active_constraints_removed": active_removed,
            "head_iis_constraints": sol_h.get("iis_constraints", []),
            "local_optima_regressions": local_optima_regressions,
            "summary_reason": summary_reason,
        }


def diff_graphs(
    base_graph: Union[CausalGraph, Dict[str, Any]],
    head_graph: Union[CausalGraph, Dict[str, Any]],
    optimality_tol: float = 1e-3,
    fail_on_dead_nodes: bool = True,
) -> CausalDiffReport:
    """Función directa de alto nivel para comparar dos grafos causales."""
    engine = CausalDiffEngine(optimality_tol=optimality_tol, fail_on_dead_nodes=fail_on_dead_nodes)
    return engine.diff(base=base_graph, head=head_graph)


def diff_sources(
    base_source: Union[CausalGraph, Dict[str, Any], str, Path],
    head_source: Union[CausalGraph, Dict[str, Any], str, Path],
    entrypoint: Optional[str] = None,
    optimality_tol: float = 1e-3,
    fail_on_dead_nodes: bool = True,
) -> CausalDiffReport:
    """Compara cualesquiera dos artefactos causales (directorios, archivos `.py`/`.json`, código inline o grafos)."""
    engine = CausalDiffEngine(optimality_tol=optimality_tol, fail_on_dead_nodes=fail_on_dead_nodes)
    return engine.diff(base=base_source, head=head_source, entrypoint=entrypoint)
