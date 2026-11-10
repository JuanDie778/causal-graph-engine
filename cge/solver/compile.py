"""Compilador simbólico de CausalGraph a Problema Matemático Canónico (Fase 10)."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Dict, Any, List, Optional, Tuple, Callable
import networkx as nx
import numpy as np
import sympy as sp

from ..schema import CausalGraph


@dataclass
class Var:
    id: str
    symbol: sp.Symbol
    domain: str  # "continuous" | "integer" | "binary"
    lb: float
    ub: float
    x0: float
    scale: float


@dataclass
class CompiledConstraint:
    id: str
    kind: str  # "ineq" (g(x) <= 0) | "eq" (h(x) == 0)
    expr: sp.Expr
    raw_expression: str
    description: str = ""
    is_linear: bool = False
    is_smooth: bool = True


@dataclass
class NonsmoothAtom:
    atom_id: int
    kind: str  # "Min" | "Max" | "Abs" | "Piecewise"
    sympy_atom: sp.Expr
    placeholder: sp.Symbol
    branches: List[Tuple[sp.Expr, List[sp.Expr], str]]  # (branch_expr, [g_cond <= 0], label)


@dataclass
class Problem:
    ir: CausalGraph
    target_id: str
    sense: str  # "max" | "min"
    mode: str  # "direct" | "target_reaching"
    problem_class: str  # "LP" | "MILP" | "QP" | "NLP_SMOOTH" | "NLP_NONSMOOTH" | "MINLP"
    vars: List[Var]
    exo_values: Dict[str, float]
    node_exprs_raw: Dict[str, sp.Expr]
    node_exprs_substituted: Dict[str, sp.Expr]
    j_expr: sp.Expr
    min_obj_expr: sp.Expr
    constraints: List[CompiledConstraint]
    nonsmooth_atoms: List[NonsmoothAtom]
    is_piecewise_linear: bool = False
    target_value: Optional[float] = None
    baseline_j: float = 0.0
    _fast_j_fn: Optional[Callable[..., float]] = field(default=None, repr=False)
    _fast_node_fns: Dict[str, Callable[..., float]] = field(default_factory=dict, repr=False)
    _fast_constr_fns: List[Tuple[str, str, Callable[..., float]]] = field(default_factory=list, repr=False)

    @property
    def ctrl_ids(self) -> List[str]:
        return [v.id for v in self.vars]

    @property
    def ctrl_symbols(self) -> List[sp.Symbol]:
        return [v.symbol for v in self.vars]

    def evaluate_j(self, x_vec: np.ndarray) -> float:
        if self._fast_j_fn is not None:
            try:
                val = float(self._fast_j_fn(*[float(v) for v in x_vec]))
                if math.isfinite(val):
                    return val
            except Exception:
                pass
        subs = {v.symbol: float(x_vec[i]) for i, v in enumerate(self.vars)}
        return float(self.j_expr.evalf(subs=subs))

    def evaluate_all_nodes(self, x_vec: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = dict(self.exo_values)
        args = [float(v) for v in x_vec]
        for i, v in enumerate(self.vars):
            out[v.id] = float(int(round(args[i]))) if v.domain in ("integer", "binary") else args[i]
        for nid, fn in self._fast_node_fns.items():
            try:
                out[nid] = float(fn(*args))
            except Exception:
                subs = {v.symbol: args[i] for i, v in enumerate(self.vars)}
                out[nid] = float(self.node_exprs_substituted[nid].evalf(subs=subs))
        return out

    def max_constraint_violation(self, x_vec: np.ndarray) -> Tuple[float, Dict[str, float]]:
        args = [float(v) for v in x_vec]
        viol_map: Dict[str, float] = {}
        max_v = 0.0
        for i, v in enumerate(self.vars):
            val = args[i]
            if val < v.lb - 1e-9:
                dv = v.lb - val
                viol_map[f"bound_lb_{v.id}"] = dv
                max_v = max(max_v, dv)
            if val > v.ub + 1e-9:
                dv = val - v.ub
                viol_map[f"bound_ub_{v.id}"] = dv
                max_v = max(max_v, dv)
            if v.domain in ("integer", "binary"):
                iv = abs(val - round(val))
                if iv > 1e-6:
                    viol_map[f"integrality_{v.id}"] = iv
                    max_v = max(max_v, iv)

        for cid, kind, fn in self._fast_constr_fns:
            try:
                raw_v = float(fn(*args))
            except Exception:
                raw_v = 0.0
            cv = max(0.0, raw_v) if kind == "ineq" else abs(raw_v)
            viol_map[cid] = cv
            if cv > max_v:
                max_v = cv
        return max_v, viol_map


def _has_nonsmooth_subexpr(expr: sp.Expr) -> bool:
    for sub in sp.preorder_traversal(expr):
        if isinstance(sub, (sp.Min, sp.Max, sp.Abs, sp.Piecewise, sp.sign, sp.Heaviside, sp.floor, sp.ceiling)):
            return True
    return False


def _poly_degree(expr: sp.Expr, syms: List[sp.Symbol]) -> Optional[int]:
    if _has_nonsmooth_subexpr(expr):
        return None
    if not syms:
        return 0 if expr.is_number else None
    try:
        expanded = sp.expand(expr)
        if expanded.is_number:
            return 0
        poly = sp.Poly(expanded, *syms)
        return int(poly.total_degree())
    except Exception:
        return None


def _cond_to_ineqs_le_zero(cond: Any, discrete_syms: Optional[set] = None) -> List[sp.Expr]:
    """Convierte una condición relacional SymPy en una lista de desigualdades g(x) <= 0."""
    disc = discrete_syms or set()
    if cond == sp.true or cond is True:
        return []
    if cond == sp.false or cond is False:
        return [sp.Float(1.0)]
    if isinstance(cond, sp.LessThan):
        return [cond.lhs - cond.rhs]
    if isinstance(cond, sp.StrictLessThan):
        diff = cond.lhs - cond.rhs
        if diff.free_symbols and diff.free_symbols.issubset(disc):
            return [diff + sp.Integer(1)]
        return [diff + sp.Float(1e-6)]
    if isinstance(cond, sp.GreaterThan):
        return [cond.rhs - cond.lhs]
    if isinstance(cond, sp.StrictGreaterThan):
        diff = cond.rhs - cond.lhs
        if diff.free_symbols and diff.free_symbols.issubset(disc):
            return [diff + sp.Integer(1)]
        return [diff + sp.Float(1e-6)]
    if isinstance(cond, sp.Eq):
        return [cond.lhs - cond.rhs, cond.rhs - cond.lhs]
    if isinstance(cond, sp.And):
        out: List[sp.Expr] = []
        for a in cond.args:
            out.extend(_cond_to_ineqs_le_zero(a, disc))
        return out
    return []


def _negate_cond_to_ineqs_le_zero(cond: Any, discrete_syms: Optional[set] = None) -> List[sp.Expr]:
    """Convierte la negación de una condición relacional simple en desigualdades g(x) <= 0."""
    disc = discrete_syms or set()
    if cond == sp.true or cond is True:
        return [sp.Float(1.0)]
    if cond == sp.false or cond is False:
        return []
    if isinstance(cond, sp.LessThan):
        diff = cond.rhs - cond.lhs
        if diff.free_symbols and diff.free_symbols.issubset(disc):
            return [diff + sp.Integer(1)]
        return [diff + sp.Float(1e-6)]
    if isinstance(cond, sp.StrictLessThan):
        return [cond.rhs - cond.lhs]
    if isinstance(cond, sp.GreaterThan):
        diff = cond.lhs - cond.rhs
        if diff.free_symbols and diff.free_symbols.issubset(disc):
            return [diff + sp.Integer(1)]
        return [diff + sp.Float(1e-6)]
    if isinstance(cond, sp.StrictGreaterThan):
        return [cond.lhs - cond.rhs]
    if isinstance(cond, sp.Not):
        return _cond_to_ineqs_le_zero(cond.args[0], disc)
    return []


def _normalize_nested_minmax(expr: sp.Expr) -> sp.Expr:
    """Normaliza Min/Max con >2 argumentos en árboles binarios para facilitar lifting y enumeración."""
    if not isinstance(expr, sp.Basic) or not expr.args:
        return expr
    new_args = [_normalize_nested_minmax(a) for a in expr.args]
    if isinstance(expr, sp.Min) and len(new_args) > 2:
        acc = new_args[0]
        for a in new_args[1:]:
            acc = sp.Min(acc, a)
        return acc
    if isinstance(expr, sp.Max) and len(new_args) > 2:
        acc = new_args[0]
        for a in new_args[1:]:
            acc = sp.Max(acc, a)
        return acc
    try:
        return expr.func(*new_args)
    except Exception:
        return expr


def _extract_nonsmooth_atoms(exprs: List[sp.Expr], ctrl_syms: List[sp.Symbol], discrete_syms: Optional[set] = None) -> List[NonsmoothAtom]:
    """Extrae todos los átomos no suaves (Min, Max, Abs, Piecewise) dependientes de X en orden bottom-up."""
    ctrl_set = set(ctrl_syms)
    disc_set = discrete_syms or set()
    seen_atoms: List[sp.Expr] = []

    def _visit(e: sp.Basic) -> None:
        if not isinstance(e, sp.Basic):
            return
        for a in e.args:
            _visit(a)
        if isinstance(e, (sp.Min, sp.Max, sp.Abs, sp.Piecewise)):
            if e.free_symbols & ctrl_set:
                if not any(e == prev for prev in seen_atoms):
                    seen_atoms.append(e)

    for ex in exprs:
        _visit(ex)

    atoms: List[NonsmoothAtom] = []
    for idx, atom_expr in enumerate(seen_atoms):
        ph = sp.Symbol(f"__ns_atom_{idx}__", real=True)
        branches: List[Tuple[sp.Expr, List[sp.Expr], str]] = []
        if isinstance(atom_expr, sp.Min):
            args = list(atom_expr.args)
            for k, ak in enumerate(args):
                conds = [ak - aj for j, aj in enumerate(args) if j != k]
                branches.append((ak, conds, f"min_arg{k}"))
        elif isinstance(atom_expr, sp.Max):
            args = list(atom_expr.args)
            for k, ak in enumerate(args):
                conds = [aj - ak for j, aj in enumerate(args) if j != k]
                branches.append((ak, conds, f"max_arg{k}"))
        elif isinstance(atom_expr, sp.Abs):
            a0 = atom_expr.args[0]
            branches.append((a0, [-a0], "abs_pos"))
            branches.append((-a0, [a0], "abs_neg"))
        elif isinstance(atom_expr, sp.Piecewise):
            prior_neg_ineqs: List[sp.Expr] = []
            for p_idx, (val_e, cond_e) in enumerate(atom_expr.args):
                cur_ineqs = list(prior_neg_ineqs) + _cond_to_ineqs_le_zero(cond_e, disc_set)
                branches.append((val_e, cur_ineqs, f"pw_branch{p_idx}"))
                prior_neg_ineqs.extend(_negate_cond_to_ineqs_le_zero(cond_e, disc_set))

        kind_name = (
            "Min" if isinstance(atom_expr, sp.Min)
            else ("Max" if isinstance(atom_expr, sp.Max) else ("Abs" if isinstance(atom_expr, sp.Abs) else "Piecewise"))
        )
        atoms.append(NonsmoothAtom(
            atom_id=idx,
            kind=kind_name,
            sympy_atom=atom_expr,
            placeholder=ph,
            branches=branches,
        ))
    return atoms


def compile_problem(
    ir: CausalGraph,
    target_id: Optional[str] = None,
    sense: Optional[str] = None,
    mode: str = "direct",
    target_delta: Optional[float] = None,
    target_value: Optional[float] = None,
    exo_overrides: Optional[Dict[str, float]] = None,
    ceiling_override: Optional[float] = None,
    fixed_controls: Optional[Dict[str, float]] = None,
    ctrl_overrides: Optional[Dict[str, float]] = None,
    bounds_overrides: Optional[Dict[str, Any]] = None,
) -> Problem:
    """Compila cualquier CausalGraph en un Problem matemático canónico sin inspeccionar nombres de dominio."""
    dag = nx.DiGraph()
    for nid in ir.nodes:
        dag.add_node(nid)
    for e in ir.edges:
        if e.type != "empirical_data":
            dag.add_edge(e.source, e.target)

    topo_order = list(nx.topological_sort(dag))

    if not target_id:
        t_candidates = [nid for nid, n in ir.nodes.items() if n.category == "target"]
        target_id = t_candidates[0] if t_candidates else (ir.metadata or {}).get("target_id", topo_order[-1] if topo_order else "")

    resolved_sense = (sense or (ir.metadata or {}).get("sense", "max") or "max").strip().lower()
    if resolved_sense not in ("max", "min", "maximize", "minimize"):
        resolved_sense = "max"
    resolved_sense = "min" if resolved_sense.startswith("min") else "max"

    if target_delta is not None or target_value is not None:
        if mode == "direct" and isinstance(target_delta, (int, float)):
            mode = "target_reaching"

    symbols: Dict[str, sp.Symbol] = {nid: sp.Symbol(nid, real=True) for nid in ir.nodes}
    local_syms: Dict[str, Any] = dict(symbols)
    local_syms.update({
        "Piecewise": sp.Piecewise, "Min": sp.Min, "Max": sp.Max,
        "sqrt": sp.sqrt, "log": sp.log, "exp": sp.exp, "Abs": sp.Abs,
        "sin": sp.sin, "cos": sp.cos, "tanh": sp.tanh,
    })

    # 1. Extraer variables exógenas Z0 y variables de control X
    exo_values: Dict[str, float] = {}
    vars_list: List[Var] = []

    for nid in topo_order:
        if nid not in ir.nodes:
            continue
        node = ir.nodes[nid]
        if node.category == "exogenous":
            val = (exo_overrides or {}).get(nid, node.current_value)
            if val is None and node.distribution:
                val = node.distribution.get("mean", 0.0)
            exo_values[nid] = float(val) if val is not None else 0.0
        elif node.category == "control":
            x0_raw = (ctrl_overrides or {}).get(nid, node.current_value)
            x0 = float(x0_raw) if x0_raw is not None else 0.0
            b = node.bounds or {}
            if bounds_overrides and nid in bounds_overrides:
                b_ov = bounds_overrides[nid]
                if isinstance(b_ov, (list, tuple)) and len(b_ov) == 2:
                    b = {"min": b_ov[0], "max": b_ov[1]}
                elif isinstance(b_ov, dict):
                    b = b_ov
            b_min = b.get("min") if isinstance(b, dict) else (b[0] if isinstance(b, (list, tuple)) and len(b) >= 1 else None)
            b_max = b.get("max") if isinstance(b, dict) else (b[1] if isinstance(b, (list, tuple)) and len(b) >= 2 else None)
            lb = float(b_min) if b_min is not None else (0.0 if x0 >= 0 else x0 - 100.0)
            ub = float(b_max) if b_max is not None else (max(1000.0, abs(x0) * 5.0) if x0 != 0 else 1000.0)
            if ub < lb:
                lb, ub = ub, lb

            if node.data_type == "boolean" or (node.data_type == "discrete" and round(lb) == 0 and round(ub) == 1):
                dom = "binary"
                lb, ub = 0.0, 1.0
            elif node.data_type == "discrete":
                dom = "integer"
                lb, ub = float(math.ceil(lb - 1e-9)), float(math.floor(ub + 1e-9))
            else:
                dom = "continuous"

            if fixed_controls and nid in fixed_controls and fixed_controls[nid] is not None:
                f_val = float(fixed_controls[nid])
                if dom in ("integer", "binary"):
                    f_val = float(int(round(f_val)))
                lb = f_val
                ub = f_val
                x0 = f_val

            scale = float(node.characteristic_scale or (ub - lb if ub > lb else max(abs(x0), 1.0)))
            vars_list.append(Var(
                id=nid,
                symbol=symbols[nid],
                domain=dom,
                lb=lb,
                ub=ub,
                x0=x0,
                scale=max(scale, 1e-6),
            ))

    # 2. Compilar y sustituir simbólicamente todas las ecuaciones estructurales en orden topológico
    surr_map = (ir.metadata or {}).get("blackbox_surrogates", {}) or {}
    node_exprs_raw: Dict[str, sp.Expr] = {}
    node_exprs_substituted: Dict[str, sp.Expr] = {}

    subs_env: Dict[sp.Symbol, sp.Expr] = {
        symbols[k]: sp.Float(v) for k, v in exo_values.items()
    }
    for v in vars_list:
        subs_env[v.symbol] = v.symbol

    nonneg_replacements: Dict[sp.Expr, sp.Expr] = {}
    for v in vars_list:
        if v.lb >= 0.0:
            nonneg_replacements[sp.Max(sp.Integer(0), v.symbol)] = v.symbol
            nonneg_replacements[sp.Max(sp.Float(0.0), v.symbol)] = v.symbol
            nonneg_replacements[sp.Abs(v.symbol)] = v.symbol

    for nid in topo_order:
        if nid not in ir.nodes:
            continue
        node = ir.nodes[nid]
        if node.category in ("exogenous", "control"):
            continue
        expr_str = node.symbolic_expr
        if not expr_str and nid in surr_map:
            expr_str = surr_map[nid].get("surrogate_sympy_expr")
        if expr_str:
            raw_e = sp.sympify(expr_str, locals=local_syms)
        else:
            raw_e = sp.Float(node.current_value or 0.0)

        node_exprs_raw[nid] = raw_e
        sub_e = raw_e.xreplace(subs_env)
        if nonneg_replacements:
            sub_e = sub_e.xreplace(nonneg_replacements)
        node_exprs_substituted[nid] = sub_e
        subs_env[symbols[nid]] = sub_e

    j_expr = node_exprs_substituted.get(target_id, sp.Float(0.0))
    baseline_subs = {v.symbol: v.x0 for v in vars_list}
    try:
        baseline_j = float(j_expr.evalf(subs=baseline_subs))
    except Exception:
        baseline_j = 0.0

    # 3. Compilar restricciones declaradas en ir.constraints
    ctrl_syms = [v.symbol for v in vars_list]
    ctrl_sym_names = {v.id for v in vars_list}
    compiled_constraints: List[CompiledConstraint] = []

    ceil_consts = (ir.metadata or {}).get("ceiling_constants", {}) or {}
    default_ceil = next(iter(ceil_consts.values()), None) if ceil_consts else None

    for cid, c_obj in ir.constraints.items():
        if c_obj.type != "hard":
            continue
        expr_str = (c_obj.expression or "").strip()
        if not expr_str or "sum(" in expr_str:
            continue

        # Ignorar restricciones de capacidad tautológicas derivadas de Min(a, b) donde el nodo ya es Min(a, b)
        try:
            if ">=" in expr_str and "<=" not in expr_str:
                lhs_s, rhs_s = expr_str.split(">=", 1)
                lhs_raw = sp.sympify(lhs_s.strip(), locals=local_syms)
                rhs_raw = sp.sympify(rhs_s.strip(), locals=local_syms)
                diff_raw = lhs_raw - rhs_raw
                # Si diff_raw es de la forma cap - Min(dem, cap) >= 0, es idénticamente >= 0 para todo X
                diff_sub = diff_raw.xreplace(subs_env)
                if isinstance(diff_raw, sp.Add) and any(
                    str(s) in node_exprs_raw and isinstance(node_exprs_raw[str(s)], sp.Min)
                    for s in diff_raw.free_symbols
                ):
                    # Verificar si diff_sub >= 0 es una identidad algebraica de Min
                    for s in diff_raw.free_symbols:
                        s_id = str(s)
                        if s_id in node_exprs_raw and isinstance(node_exprs_raw[s_id], sp.Min):
                            min_args = [a.xreplace(subs_env) for a in node_exprs_raw[s_id].args]
                            # Si lhs_raw == min_arg - s_id y rhs_raw == 0:
                            if any(sp.simplify((lhs_raw - rhs_raw).subs(s, ma)) == 0 for ma in node_exprs_raw[s_id].args):
                                diff_sub = sp.Float(0.0)
                                break
                    if diff_sub == sp.Float(0.0):
                        continue
        except Exception:
            pass

        # Soportar encadenamiento a <= x <= b
        if expr_str.count("<=") == 2 and ">=" not in expr_str:
            parts = [p.strip() for p in expr_str.split("<=")]
            if len(parts) == 3:
                # Si es una cota simple sobre una variable de control, ajustar directamente lb/ub de la variable
                if parts[1] in ctrl_sym_names:
                    try:
                        lo_v = float(sp.sympify(parts[0], locals=local_syms).xreplace(subs_env).evalf())
                        hi_v = float(sp.sympify(parts[2], locals=local_syms).xreplace(subs_env).evalf())
                        for v in vars_list:
                            if v.id == parts[1]:
                                v.lb = max(v.lb, lo_v)
                                v.ub = min(v.ub, hi_v)
                        continue
                    except Exception:
                        pass
                sub_pairs = [(f"{cid}_lo", f"{parts[0]} <= {parts[1]}"), (f"{cid}_hi", f"{parts[1]} <= {parts[2]}")]
            else:
                sub_pairs = [(cid, expr_str)]
        else:
            sub_pairs = [(cid, expr_str)]

        for sub_id, sub_s in sub_pairs:
            try:
                if "<=" in sub_s and ">=" not in sub_s:
                    lhs_s, rhs_s = sub_s.split("<=", 1)
                    lhs_e = sp.sympify(lhs_s.strip(), locals=local_syms).xreplace(subs_env)
                    rhs_e = sp.sympify(rhs_s.strip(), locals=local_syms).xreplace(subs_env)
                    if ceiling_override is not None and default_ceil is not None and rhs_e.is_number and abs(float(rhs_e) - float(default_ceil)) < 1e-6:
                        rhs_e = sp.Float(ceiling_override)
                    g_expr = lhs_e - rhs_e
                    kind = "ineq"
                elif ">=" in sub_s and "<=" not in sub_s:
                    lhs_s, rhs_s = sub_s.split(">=", 1)
                    lhs_e = sp.sympify(lhs_s.strip(), locals=local_syms).xreplace(subs_env)
                    rhs_e = sp.sympify(rhs_s.strip(), locals=local_syms).xreplace(subs_env)
                    g_expr = rhs_e - lhs_e
                    kind = "ineq"
                elif "==" in sub_s:
                    lhs_s, rhs_s = sub_s.split("==", 1)
                    lhs_e = sp.sympify(lhs_s.strip(), locals=local_syms).xreplace(subs_env)
                    rhs_e = sp.sympify(rhs_s.strip(), locals=local_syms).xreplace(subs_env)
                    g_expr = lhs_e - rhs_e
                    kind = "eq"
                else:
                    continue

                # Ignorar si no depende de ninguna variable del grafo y ya se cumple
                unknown_syms = {str(s) for s in g_expr.free_symbols} - ctrl_sym_names
                if unknown_syms:
                    continue

                deg = _poly_degree(g_expr, ctrl_syms)
                is_lin = (deg is not None and deg <= 1)
                is_sm = not _has_nonsmooth_subexpr(g_expr)

                compiled_constraints.append(CompiledConstraint(
                    id=sub_id,
                    kind=kind,
                    expr=g_expr,
                    raw_expression=sub_s,
                    description=c_obj.description or sub_s,
                    is_linear=is_lin,
                    is_smooth=is_sm,
                ))
            except Exception:
                continue

    # 4. Configurar objetivo según el modo (Direct Optimization vs Target-Reaching Minimum Intervention)
    resolved_target_val: Optional[float] = None
    if mode == "target_reaching":
        if target_value is not None:
            resolved_target_val = float(target_value)
        elif target_delta is not None:
            resolved_target_val = baseline_j + float(target_delta)
        else:
            resolved_target_val = baseline_j

        # Restricción de alcance de meta: J(X) >= target_val (si max) o J(X) <= target_val (si min)
        if resolved_sense == "max":
            t_g = sp.Float(resolved_target_val) - j_expr
            raw_t = f"{target_id} >= {resolved_target_val}"
        else:
            t_g = j_expr - sp.Float(resolved_target_val)
            raw_t = f"{target_id} <= {resolved_target_val}"

        t_deg = _poly_degree(t_g, ctrl_syms)
        compiled_constraints.append(CompiledConstraint(
            id="c_target_goal",
            kind="ineq",
            expr=t_g,
            raw_expression=raw_t,
            description=f"Meta requerida sobre {target_id}: {raw_t}",
            is_linear=(t_deg is not None and t_deg <= 1),
            is_smooth=not _has_nonsmooth_subexpr(t_g),
        ))
        min_obj_expr = sum(((v.symbol - sp.Float(v.x0)) / sp.Float(v.scale)) ** 2 for v in vars_list) if vars_list else sp.Float(0.0)
    else:
        min_obj_expr = -j_expr if resolved_sense == "max" else j_expr

    # 5. Extraer átomos no suaves (Min, Max, Abs, Piecewise) y clasificar estructura del problema
    norm_obj = _normalize_nested_minmax(min_obj_expr)
    norm_constrs = [_normalize_nested_minmax(c.expr) for c in compiled_constraints]
    disc_sym_set = {v.symbol for v in vars_list if v.domain in ("integer", "binary")}
    nonsmooth_atoms = _extract_nonsmooth_atoms([norm_obj] + norm_constrs, ctrl_syms, disc_sym_set)

    has_discrete = any(v.domain in ("integer", "binary") for v in vars_list)
    has_nonsmooth = len(nonsmooth_atoms) > 0

    obj_deg = _poly_degree(min_obj_expr, ctrl_syms)
    all_constrs_linear = all(c.is_linear for c in compiled_constraints)

    # Comprobar si el problema es lineal a trozos (todos los átomos tienen ramas y condiciones lineales)
    is_pw_linear = False
    if has_nonsmooth and all(_poly_degree(c.expr, ctrl_syms) is not None and _poly_degree(c.expr, ctrl_syms) <= 1 for c in compiled_constraints if c.is_smooth):
        atoms_linear = True
        all_syms_ext = list(ctrl_syms) + [a.placeholder for a in nonsmooth_atoms]
        for idx_a, atom in enumerate(nonsmooth_atoms):
            for br_e, br_conds, _ in atom.branches:
                br_sub = br_e
                for r_prev in range(idx_a - 1, -1, -1):
                    br_sub = br_sub.xreplace({nonsmooth_atoms[r_prev].sympy_atom: nonsmooth_atoms[r_prev].placeholder})
                d_br = _poly_degree(br_sub, all_syms_ext)
                if d_br is None or d_br > 1:
                    atoms_linear = False
                    break
                for gc in br_conds:
                    gc_sub = gc
                    for r_prev in range(idx_a - 1, -1, -1):
                        gc_sub = gc_sub.xreplace({nonsmooth_atoms[r_prev].sympy_atom: nonsmooth_atoms[r_prev].placeholder})
                    d_gc = _poly_degree(gc_sub, all_syms_ext)
                    if d_gc is None or d_gc > 1:
                        atoms_linear = False
                        break
            if not atoms_linear:
                break
        if atoms_linear:
            temp_obj = norm_obj
            temp_cs = list(norm_constrs)
            for atom in reversed(nonsmooth_atoms):
                temp_obj = temp_obj.xreplace({atom.sympy_atom: atom.placeholder})
                temp_cs = [tc.xreplace({atom.sympy_atom: atom.placeholder}) for tc in temp_cs]
            d_top = _poly_degree(temp_obj, all_syms_ext)
            if d_top is not None and d_top <= 1 and all(
                (_poly_degree(tc, all_syms_ext) is not None and _poly_degree(tc, all_syms_ext) <= 1)
                for tc in temp_cs
            ):
                is_pw_linear = True

    if not has_nonsmooth and obj_deg is not None and obj_deg <= 1 and all_constrs_linear:
        problem_class = "MILP" if has_discrete else "LP"
        is_pw_linear = True
    elif not has_nonsmooth and obj_deg is not None and obj_deg == 2 and all_constrs_linear and not has_discrete:
        problem_class = "QP"
    elif has_discrete:
        problem_class = "MINLP"
    elif has_nonsmooth:
        problem_class = "NLP_NONSMOOTH"
    else:
        problem_class = "NLP_SMOOTH"

    # 6. Compilar evaluadores numéricos rápidos (lambdify)
    safe_modules = [
        {
            "sqrt": lambda v: math.sqrt(max(0.0, float(v))),
            "log": lambda v: math.log(max(1e-12, float(v))),
            "log1p": lambda v: math.log1p(max(0.0, float(v))),
        },
        "math",
    ]
    fast_j_fn = None
    try:
        fast_j_fn = sp.lambdify(ctrl_syms, j_expr, modules=safe_modules)
    except Exception:
        pass

    fast_node_fns: Dict[str, Callable[..., float]] = {}
    for nid, n_expr in node_exprs_substituted.items():
        try:
            fast_node_fns[nid] = sp.lambdify(ctrl_syms, n_expr, modules=safe_modules)
        except Exception:
            pass

    fast_constr_fns: List[Tuple[str, str, Callable[..., float]]] = []
    for c in compiled_constraints:
        try:
            fn_c = sp.lambdify(ctrl_syms, c.expr, modules=safe_modules)
            fast_constr_fns.append((c.id, c.kind, fn_c))
        except Exception:
            pass

    return Problem(
        ir=ir,
        target_id=target_id,
        sense=resolved_sense,
        mode=mode,
        problem_class=problem_class,
        vars=vars_list,
        exo_values=exo_values,
        node_exprs_raw=node_exprs_raw,
        node_exprs_substituted=node_exprs_substituted,
        j_expr=j_expr,
        min_obj_expr=min_obj_expr,
        constraints=compiled_constraints,
        nonsmooth_atoms=nonsmooth_atoms,
        is_piecewise_linear=is_pw_linear,
        target_value=resolved_target_val,
        baseline_j=baseline_j,
        _fast_j_fn=fast_j_fn,
        _fast_node_fns=fast_node_fns,
        _fast_constr_fns=fast_constr_fns,
    )
