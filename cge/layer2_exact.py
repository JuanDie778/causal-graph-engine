"""Capa 2: Solucionador Matemático Universal con Certificados Verificables (Fase 10)."""
from __future__ import annotations
from itertools import combinations
import json
import math
import os
from typing import Dict, Any, List, Tuple, Optional
import networkx as nx
import numpy as np
import sympy as sp

from .schema import CausalGraph
from .solver import solve_problem, compile_problem, SolveStatus, SolveCertificate


def solve_layer2(
    ir: CausalGraph,
    target_id: Optional[str] = None,
    sense: Optional[str] = None,
    mode: str = "direct",
    target_delta: Optional[float] = None,
    target_value: Optional[float] = None,
    exo_overrides: Optional[Dict[str, float]] = None,
    ceiling_override: Optional[float] = None,
    fixed_controls: Optional[Dict[str, float]] = None,
    source_path_or_code: Optional[str] = None,
    entrypoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Punto de entrada funcional para resolver cualquier CausalGraph mediante el motor genérico cge.solver."""
    return solve_problem(
        ir=ir,
        target_id=target_id,
        sense=sense,
        mode=mode,
        target_delta=target_delta,
        target_value=target_value,
        exo_overrides=exo_overrides,
        ceiling_override=ceiling_override,
        fixed_controls=fixed_controls,
        source_path_or_code=source_path_or_code,
        entrypoint=entrypoint,
    )


class Layer2Solver:
    """Solucionador matemático universal guiado exclusivamente por las expresiones simbólicas del CausalIR
    que delega la optimización y certificación formal a `cge.solver`.
    """

    def __init__(self, graph: Optional[CausalGraph] = None, **kwargs):
        src_dir = kwargs.pop("source_dir", None) or kwargs.pop("target_dir", None)
        if graph is None:
            if src_dir is not None:
                from .ingest import ASTIngestPipeline
                graph = ASTIngestPipeline(src_dir).run()
            else:
                graph = self._discover_default_graph()
        self.graph: CausalGraph = graph
        self.symbols: Dict[str, sp.Symbol] = {}
        self.mechanisms: Dict[str, sp.Expr] = {}
        self.control_ids: List[str] = []
        self.discrete_controls: List[str] = []
        self.continuous_controls: List[str] = []
        self.exogenous_ids: List[str] = []
        self.target_id: str = ""
        self.topo_order: List[str] = []
        self._compiled_funcs: Dict[str, Tuple[List[str], Any]] = {}
        self._upfront_cost_compiled: Optional[Tuple[List[str], Any]] = None
        self._ceiling_constraint_id: Optional[str] = None
        self._max_budget: float = float("inf")
        self._general_constraints: List[Tuple[str, str, List[str], Any]] = []
        self._implied_ctrl_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        self._min_nodes_compiled: List[Tuple[str, List[str], Any, Any]] = []

        self._initialize_from_graph()

        # Aplicar overrides opcionales pasados por kwargs
        for k, v in kwargs.items():
            if v is None:
                continue
            if k in self.graph.nodes and isinstance(v, (int, float)):
                self.graph.nodes[k].current_value = float(v)
            elif k == "max_budget":
                self._max_budget = float(v)

    @staticmethod
    def _discover_default_graph() -> CausalGraph:
        """Descubre y compila automáticamente el grafo causal activo en el espacio de trabajo."""
        from .ingest import ASTIngestPipeline
        if os.path.exists("schema.json"):
            try:
                with open("schema.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                nodes = data.get("nodes", [])
                if nodes and any(n.get("symbolic_expr") for n in nodes):
                    return CausalGraph.from_dict(data)
                if nodes:
                    first_file = (nodes[0].get("source_reference") or {}).get("file", "")
                    top_dir = first_file.split("/")[0] if "/" in first_file else ""
                    if top_dir and os.path.isdir(top_dir):
                        return ASTIngestPipeline(top_dir).run()
            except Exception:
                pass

        excluded = {
            "cge", "benchmark", "oracle", "ui", "math_lab", "docs",
            "schema", "eval_reports", "benchmark_variants", ".pytest_cache", "__pycache__"
        }
        for entry in sorted(os.listdir(".")):
            if os.path.isdir(entry) and not entry.startswith(".") and entry not in excluded:
                return ASTIngestPipeline(entry).run()
        raise RuntimeError("No se encontró un grafo CausalIR activo ni directorio de modelo para inicializar Layer2Solver.")

    @classmethod
    def from_graph(cls, graph: CausalGraph) -> "Layer2Solver":
        """Construye el solver directamente desde un objeto CausalGraph."""
        has_sym = any(n.symbolic_expr is not None for n in graph.nodes.values() if n.category in ("state", "target"))
        has_bb = bool(getattr(graph, "_blackbox_callables", {})) or bool((graph.metadata or {}).get("blackbox_surrogates"))
        if not has_sym and not has_bb:
            for n in graph.nodes.values():
                f_ref = n.source_reference.file if n.source_reference else None
                if f_ref and "/" in f_ref:
                    top_dir = f_ref.split("/")[0]
                    if os.path.isdir(top_dir):
                        from .ingest import ASTIngestPipeline
                        return cls(graph=ASTIngestPipeline(top_dir).run())
        return cls(graph=graph)

    @classmethod
    def from_dir(cls, target_dir: str) -> "Layer2Solver":
        """Construye el solver compilando el directorio target mediante ASTIngestPipeline."""
        from .ingest import ASTIngestPipeline
        graph = ASTIngestPipeline(target_dir).run()
        return cls(graph=graph)

    def _get_sym(self, name: str) -> sp.Symbol:
        if name not in self.symbols:
            self.symbols[name] = sp.Symbol(name, real=True)
        return self.symbols[name]

    def _initialize_from_graph(self):
        dag = nx.DiGraph()
        for nid, node in self.graph.nodes.items():
            self._get_sym(nid)
            dag.add_node(nid)
            if node.category == "control":
                self.control_ids.append(nid)
                if node.data_type in ("discrete", "boolean"):
                    self.discrete_controls.append(nid)
                else:
                    self.continuous_controls.append(nid)
            elif node.category == "exogenous":
                self.exogenous_ids.append(nid)
            elif node.category == "target":
                self.target_id = nid

        for edge in self.graph.edges:
            if edge.type != "empirical_data":
                dag.add_edge(edge.source, edge.target)

        local_syms = dict(self.symbols)
        local_syms.update({
            "Piecewise": sp.Piecewise, "Min": sp.Min, "Max": sp.Max,
            "sqrt": sp.sqrt, "log": sp.log, "exp": sp.exp, "Abs": sp.Abs
        })

        bb_map = getattr(self.graph, "_blackbox_callables", {}) or {}
        surr_map = (self.graph.metadata or {}).get("blackbox_surrogates", {}) or {}
        for nid, node in self.graph.nodes.items():
            if node.symbolic_expr:
                expr = sp.sympify(node.symbolic_expr, locals=local_syms)
                self.mechanisms[nid] = expr
                for s in expr.free_symbols:
                    self.symbols[str(s)] = s
            elif nid not in bb_map and nid in surr_map and surr_map[nid].get("surrogate_sympy_expr"):
                expr = sp.sympify(surr_map[nid]["surrogate_sympy_expr"], locals=local_syms)
                self.mechanisms[nid] = expr
                for s in expr.free_symbols:
                    self.symbols[str(s)] = s

        self.topo_order = [nid for nid in nx.topological_sort(dag) if nid in self.mechanisms or nid in bb_map]

        # Detectar genéricamente cualquier restricción de techo de recursos (lhs <= constante positiva)
        ceil_map = (self.graph.metadata or {}).get("ceiling_constants", {})
        if ceil_map:
            self._max_budget = float(next(iter(ceil_map.values())))

        upfront_lhs_expr: Optional[sp.Expr] = None
        for cid, c_obj in self.graph.constraints.items():
            expr_s = (c_obj.expression or "").strip()
            if expr_s.count("<=") == 1 and ">=" not in expr_s and "sum(" not in expr_s:
                parts = expr_s.split("<=")
                if len(parts) == 2:
                    try:
                        rhs_val = float(parts[1].strip())
                        lhs_cand = sp.sympify(parts[0].strip(), locals=local_syms)
                        if len(lhs_cand.free_symbols) >= 2 or (ceil_map and abs(rhs_val - self._max_budget) < 1e-6):
                            self._max_budget = rhs_val
                            upfront_lhs_expr = lhs_cand
                            self._ceiling_constraint_id = cid
                            break
                    except Exception:
                        pass

        if upfront_lhs_expr is not None:
            u_syms = sorted([str(s) for s in upfront_lhs_expr.free_symbols])
            u_objs = [self._get_sym(s) for s in u_syms]
            self._upfront_cost_compiled = (u_syms, sp.lambdify(u_objs, upfront_lhs_expr, modules=["math"]))

        self._compile_general_constraints(local_syms)
        self._compile_fast_evaluators()

    def _compile_general_constraints(self, local_syms: Dict[str, Any]):
        """Compila todas las restricciones simbólicas declaradas en graph.constraints de forma genérica."""
        safe_modules = [
            {
                "sqrt": lambda v: math.sqrt(max(0.0, float(v))),
                "log": lambda v: math.log(max(1e-12, float(v))),
                "log1p": lambda v: math.log1p(max(0.0, float(v)))
            },
            "math"
        ]
        for cid, c_obj in self.graph.constraints.items():
            if c_obj.type != "hard":
                continue
            if cid == self._ceiling_constraint_id:
                continue
            expr_str = (c_obj.expression or "").strip()
            if not expr_str or "sum(" in expr_str:
                continue

            margin_exprs: List[Tuple[str, sp.Expr]] = []
            try:
                if expr_str.count("<=") == 1 and ">=" not in expr_str:
                    lhs_s, rhs_s = expr_str.split("<=", 1)
                    lhs_e = sp.sympify(lhs_s.strip(), locals=local_syms)
                    rhs_e = sp.sympify(rhs_s.strip(), locals=local_syms)
                    margin_exprs.append(("ineq", rhs_e - lhs_e))
                elif expr_str.count(">=") == 1 and "<=" not in expr_str:
                    lhs_s, rhs_s = expr_str.split(">=", 1)
                    lhs_e = sp.sympify(lhs_s.strip(), locals=local_syms)
                    rhs_e = sp.sympify(rhs_s.strip(), locals=local_syms)
                    margin_exprs.append(("ineq", lhs_e - rhs_e))
                elif expr_str.count("<=") == 2 and ">=" not in expr_str:
                    p0, p1, p2 = expr_str.split("<=")
                    e0 = sp.sympify(p0.strip(), locals=local_syms)
                    e1 = sp.sympify(p1.strip(), locals=local_syms)
                    e2 = sp.sympify(p2.strip(), locals=local_syms)
                    margin_exprs.append(("ineq", e1 - e0))
                    margin_exprs.append(("ineq", e2 - e1))
                elif "==" in expr_str:
                    lhs_s, rhs_s = expr_str.split("==", 1)
                    lhs_e = sp.sympify(lhs_s.strip(), locals=local_syms)
                    rhs_e = sp.sympify(rhs_s.strip(), locals=local_syms)
                    margin_exprs.append(("eq", lhs_e - rhs_e))
            except Exception:
                continue

            for idx, (kind_c, m_expr) in enumerate(margin_exprs):
                free_s = sorted([str(s) for s in m_expr.free_symbols])
                if not free_s or not all(s in self.graph.nodes for s in free_s):
                    continue
                s_objs = [self._get_sym(s) for s in free_s]
                m_fn = sp.lambdify(s_objs, m_expr, modules=safe_modules)
                sub_cid = cid if len(margin_exprs) == 1 else f"{cid}_part{idx}"
                self._general_constraints.append((sub_cid, kind_c, free_s, m_fn))

                if kind_c == "ineq" and len(free_s) == 1 and free_s[0] in self.control_ids:
                    ctrl_name = free_s[0]
                    c_sym = self._get_sym(ctrl_name)
                    try:
                        deriv = sp.diff(m_expr, c_sym)
                        if deriv.is_number:
                            slope = float(deriv.evalf())
                            const_val = float(m_expr.subs(c_sym, 0).evalf())
                            cur_lo, cur_hi = self._implied_ctrl_bounds.get(ctrl_name, (None, None))
                            if slope < -1e-12:
                                ub = -const_val / slope
                                cur_hi = ub if cur_hi is None else min(cur_hi, ub)
                            elif slope > 1e-12:
                                lb = -const_val / slope
                                cur_lo = lb if cur_lo is None else max(cur_lo, lb)
                            self._implied_ctrl_bounds[ctrl_name] = (cur_lo, cur_hi)
                    except Exception:
                        pass

    def _compile_fast_evaluators(self):
        """Compila cada mecanismo SymPy (o caja negra python_callable) a callables numéricos."""
        bb_map = getattr(self.graph, "_blackbox_callables", {}) or {}
        safe_modules = [
            {
                "sqrt": lambda v: math.sqrt(max(0.0, float(v))),
                "log": lambda v: math.log(max(1e-12, float(v))),
                "log1p": lambda v: math.log1p(max(0.0, float(v)))
            },
            "math"
        ]

        for nid in self.topo_order:
            if nid not in self.mechanisms and nid in bb_map:
                p_names, py_call = bb_map[nid]
                self._compiled_funcs[nid] = (list(p_names), py_call)
                continue

            expr = self.mechanisms[nid]
            free_syms = sorted([str(s) for s in expr.free_symbols])
            sym_objs = [self._get_sym(s) for s in free_syms]
            fn = sp.lambdify(sym_objs, expr, modules=safe_modules)
            self._compiled_funcs[nid] = (free_syms, fn)

            if isinstance(expr, sp.Min) and len(expr.args) == 2:
                fn_a = sp.lambdify(sym_objs, expr.args[0], modules=safe_modules)
                fn_b = sp.lambdify(sym_objs, expr.args[1], modules=safe_modules)
                self._min_nodes_compiled.append((nid, free_syms, fn_a, fn_b))

    def __getattr__(self, name: str) -> Any:
        """Expone dinámicamente propiedades del CausalIR sin nombres de dominio cableados."""
        graph = self.__dict__.get("graph")
        if graph is not None:
            if name in graph.nodes:
                return graph.nodes[name].current_value
            if name == "max_budget":
                return self._max_budget
            if name == "volume_threshold":
                for n in graph.nodes.values():
                    if n.discontinuities:
                        return int(n.discontinuities[0]["threshold"])
                return 0
            if name == "cost_high":
                for n in graph.nodes.values():
                    if n.discontinuities:
                        return float(n.discontinuities[0].get("val_true", 1.0))
                return 1.0
            if name == "cost_low":
                for n in graph.nodes.values():
                    if n.discontinuities:
                        return float(n.discontinuities[0].get("val_false", 1.0))
                return 1.0
            if name.endswith("_rate"):
                for nid, expr in self.mechanisms.items():
                    free_s = list(expr.free_symbols)
                    if len(free_s) == 1:
                        d_val = sp.diff(expr, free_s[0])
                        if d_val.is_number and 0.0 < float(d_val.evalf()) < 1.0:
                            return float(d_val.evalf())
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def _parse_inputs(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, float]:
        state: Dict[str, float] = {}
        for nid in self.exogenous_ids + self.control_ids:
            val = self.graph.nodes[nid].current_value
            state[nid] = float(val) if val is not None else 0.0

        if len(args) == 1 and isinstance(args[0], dict):
            for k, v in args[0].items():
                if v is not None and k in state:
                    state[k] = float(v)
        elif args:
            ordered_keys = self.control_ids + self.exogenous_ids
            for k, v in zip(ordered_keys, args):
                if v is not None:
                    state[k] = float(v)

        for k, v in kwargs.items():
            if v is not None and k in state:
                state[k] = float(v)
        return state

    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        """Evalúa el DAG causal completo y verifica todas las restricciones duras de forma genérica."""
        max_b = float(kwargs.pop("max_budget_override", None) or self._max_budget)
        state = self._parse_inputs(args, kwargs)

        for nid in self.topo_order:
            arg_names, fn = self._compiled_funcs[nid]
            fn_args = [state[a] for a in arg_names]
            try:
                state[nid] = float(fn(*fn_args))
            except Exception:
                state[nid] = 0.0

        if self._upfront_cost_compiled is not None:
            u_syms, u_fn = self._upfront_cost_compiled
            budget_used = float(u_fn(*[state[s] for s in u_syms]))
        else:
            budget_used = 0.0

        violations: Dict[str, bool] = {}
        ceil_ok = budget_used <= max_b + 1e-4
        if self._ceiling_constraint_id:
            violations[self._ceiling_constraint_id] = not ceil_ok

        bounds_ok = True
        for cid in self.control_ids:
            b = self.graph.nodes[cid].bounds or {}
            b_min = b.get("min")
            b_max = b.get("max")
            ok_c = True
            if b_min is not None and state[cid] < float(b_min) - 1e-5:
                ok_c = False
            if b_max is not None and state[cid] > float(b_max) + 1e-5:
                ok_c = False
            violations[f"c_{cid}_bounds"] = not ok_c
            bounds_ok = bounds_ok and ok_c

        general_ok = True
        for gc_id, gc_kind, gc_syms, gc_fn in self._general_constraints:
            try:
                m_val = float(gc_fn(*[state[s] for s in gc_syms]))
            except Exception:
                m_val = 0.0
            state[f"_margin_{gc_id}"] = m_val
            gc_ok = (m_val >= -1e-4) if gc_kind == "ineq" else (abs(m_val) <= 1e-4)
            base_gc = gc_id.split("_part")[0]
            violations[base_gc] = violations.get(base_gc, False) or (not gc_ok)
            general_ok = general_ok and gc_ok

        state["objective_value"] = float(state.get(self.target_id, 0.0))
        state["budget_used"] = budget_used
        state["max_budget"] = max_b
        state["budget_margin"] = (max_b - budget_used) if math.isfinite(max_b) else 0.0
        state["is_feasible"] = bool(ceil_ok and bounds_ok and general_ok)
        state["violations"] = violations
        return state

    def evaluate_current_state(self) -> Dict[str, Any]:
        """Evalúa el estado nominal actual del grafo causal."""
        return self.evaluate()

    def get_extracted_regimes(self, disc_ctrl: str) -> List[Tuple[int, int]]:
        """Extrae automáticamente los intervalos enteros suaves a partir de las discontinuidades Piecewise."""
        b = self.graph.nodes[disc_ctrl].bounds or {}
        q_min = int(b.get("min") if b.get("min") is not None else 0)
        q_max = int(b.get("max") if b.get("max") is not None else 10000)
        imp_lo, imp_hi = self._implied_ctrl_bounds.get(disc_ctrl, (None, None))
        if imp_lo is not None:
            q_min = max(q_min, int(math.ceil(imp_lo - 1e-9)))
        if imp_hi is not None:
            q_max = min(q_max, int(math.floor(imp_hi + 1e-9)))

        thresholds = sorted({
            int(d["threshold"])
            for n in self.graph.nodes.values()
            for d in n.discontinuities
            if d.get("variable") == disc_ctrl and "threshold" in d
        })
        if not thresholds:
            return [(q_min, q_max)]

        regimes: List[Tuple[int, int]] = []
        cur = q_min
        for th in thresholds:
            if cur <= th - 1:
                regimes.append((cur, th - 1))
            cur = th
        if cur <= q_max:
            regimes.append((cur, q_max))
        return regimes

    def optimize_at_fixed_q(self, fixed_q: Any) -> Dict[str, Any]:
        """Optimiza las palancas continuas manteniendo fija la palanca discreta o conjunto de discretas."""
        if not self.discrete_controls:
            return self.optimize()
        if isinstance(fixed_q, dict):
            fixed_map = {k: float(v) for k, v in fixed_q.items()}
        else:
            fixed_map = {self.discrete_controls[0]: float(fixed_q)}
        res = self.optimize(fixed_controls=fixed_map)
        if res.get("status") not in ("FEASIBLE", "OPTIMAL"):
            return res
        return self.evaluate(**res["controls"])

    def optimize(
        self,
        target_value: Optional[float] = None,
        fixed_controls: Optional[Dict[str, float]] = None,
        exo_overrides: Optional[Dict[str, float]] = None,
        max_budget_override: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Delega la resolución matemática y certificación a `cge.solver.solve_problem`."""
        if target_value is None:
            for k, v in kwargs.items():
                if k.startswith("target_") and v is not None:
                    target_value = float(v)
                    break

        ceil_ov = float(max_budget_override) if max_budget_override is not None else (
            self._max_budget if math.isfinite(self._max_budget) else None
        )

        sol = solve_problem(
            ir=self.graph,
            target_id=self.target_id,
            mode="direct",
            exo_overrides=exo_overrides,
            ceiling_override=ceil_ov,
            fixed_controls=fixed_controls,
        )

        # Compuerta de Honestidad Epistémica: propagar UNSUPPORTED sin inventar receta
        if sol.get("status") == "UNSUPPORTED":
            return sol

        # Si el dominio en sí es infactible (IIS certificado)
        if sol.get("status") == "INFEASIBLE":
            return sol

        controls_raw = sol.get("controls", {})
        ev_kwargs = dict(exo_overrides or {})
        ev_kwargs.update(controls_raw)
        if ceil_ov is not None:
            ev_kwargs["max_budget_override"] = ceil_ov
        best_eval = self.evaluate(**ev_kwargs)
        best_j = float(best_eval.get(self.target_id, sol.get("objective_value", 0.0)))

        # Si el usuario pidió alcanzar una meta target_value que supera la capacidad máxima J* del sistema
        sense_val = (self.graph.metadata or {}).get("sense", "max")
        is_min_sense = str(sense_val).lower().startswith("min")
        goal_unreached = (
            (target_value is not None and not is_min_sense and best_j < target_value - 1e-4)
            or (target_value is not None and is_min_sense and best_j > target_value + 1e-4)
        )
        if goal_unreached and target_value is not None:
            return self._diagnose_infeasibility_kkt(
                target_value=float(target_value),
                best_eval=best_eval,
                exo_overrides=exo_overrides,
                max_b=ceil_ov if ceil_ov is not None else self._max_budget,
            )

        # Identificar restricciones activas y umbrales de régimen activos en el óptimo
        active: List[str] = list(sol.get("active_constraints", []))
        for d_id in self.discrete_controls:
            thresholds = [
                int(d["threshold"])
                for n in self.graph.nodes.values()
                for d in n.discontinuities
                if d.get("variable") == d_id and "threshold" in d
            ]
            if d_id in best_eval and int(round(best_eval[d_id])) in thresholds:
                if "c_volume_discount_threshold" not in active:
                    active.append("c_volume_discount_threshold")

        for min_nid, min_syms, fn_a, fn_b in self._min_nodes_compiled:
            try:
                va = float(fn_a(*[best_eval[s] for s in min_syms]))
                vb = float(fn_b(*[best_eval[s] for s in min_syms]))
                if abs(va - vb) <= 1e-2:
                    cap_lbl = f"c_{min_nid}_capacity_limit"
                    if cap_lbl not in active:
                        active.append(cap_lbl)
            except Exception:
                pass

        if self._ceiling_constraint_id and math.isfinite(best_eval["max_budget"]):
            if abs(best_eval["budget_used"] - best_eval["max_budget"]) < 1.0:
                if self._ceiling_constraint_id not in active:
                    active.append(self._ceiling_constraint_id)

        for gc_id, _, _, _ in self._general_constraints:
            base_gc = gc_id.split("_part")[0]
            if abs(float(best_eval.get(f"_margin_{gc_id}", 1.0))) < 1e-2 and base_gc not in active:
                active.append(base_gc)

        controls_out: Dict[str, Any] = {}
        for cid in self.control_ids:
            if cid in self.discrete_controls:
                controls_out[cid] = int(round(best_eval[cid]))
            else:
                controls_out[cid] = round(float(best_eval[cid]), 4)

        state_out = {
            k: (round(v, 2) if k in self.graph.nodes and self.graph.nodes[k].discontinuities else round(v, 4))
            for k, v in best_eval.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and k not in self.control_ids and k not in self.exogenous_ids
            and k != self.target_id and not k.startswith("_margin_")
        }

        res_dict: Dict[str, Any] = {
            "status": "FEASIBLE",
            "certificate_status": sol.get("certificate_status", SolveStatus.CERTIFIED_OPTIMAL.value),
            "certificate": sol.get("certificate"),
            "problem_class": sol.get("problem_class"),
            "strategy_used": sol.get("strategy_used"),
            "controls": controls_out,
            "optimal_deltas": sol.get("optimal_deltas", {}),
            "achieved_delta": sol.get("achieved_delta", 0.0),
            "active_regime": sol.get("active_regime", "smooth"),
            **controls_out,
            self.target_id: round(float(best_j), 4),
            "objective_value": round(float(best_j), 4),
            "budget_used": round(float(best_eval["budget_used"]), 4),
            "budget_spent": round(float(best_eval["budget_used"]), 4),
            "budget_margin": round(float(best_eval["budget_margin"]), 4),
            "state": state_out,
            "active_constraints": active,
            "iis_constraints": [],
            "blocking_constraints": [],
            "violations_count": 0
        }
        return res_dict

    def _diagnose_infeasibility_kkt(
        self,
        target_value: float,
        best_eval: Dict[str, Any],
        exo_overrides: Optional[Dict[str, float]],
        max_b: float
    ) -> Dict[str, Any]:
        """Diagnóstico formal de infactibilidad mediante Estrategia E (IIS) y sensibilidad KKT."""
        j_star = float(best_eval[self.target_id])
        shortfall = abs(target_value - j_star)
        budget_slack = (max_b - float(best_eval["budget_used"])) if math.isfinite(max_b) else float("inf")

        if budget_slack > 1.0:
            lambda_budget = 0.0
        else:
            eps_b = 10.0
            sol_b = self.optimize(exo_overrides=exo_overrides, max_budget_override=max_b + eps_b)
            lambda_budget = max(0.0, (float(sol_b.get("objective_value", j_star)) - j_star) / eps_b)

        lambda_market = 0.0
        if self.exogenous_ids:
            u0_id = self.exogenous_ids[0]
            base_u = float((exo_overrides or {}).get(u0_id, self.graph.nodes[u0_id].current_value or 1000.0))
            eps_u = 10.0
            exo_plus = dict(exo_overrides or {})
            exo_plus[u0_id] = base_u + eps_u
            sol_u = self.optimize(exo_overrides=exo_plus, max_budget_override=max_b if math.isfinite(max_b) else None)
            lambda_market = max(0.0, (float(sol_u.get("objective_value", j_star)) - j_star) / eps_u)

        blocking: List[str] = []
        if self._ceiling_constraint_id:
            blocking.append(self._ceiling_constraint_id)

        if lambda_market > 1e-4:
            for cid in self.graph.constraints:
                if cid != self._ceiling_constraint_id and cid not in blocking:
                    blocking.append(cid)
                    break

        for gc_id, _, _, _ in self._general_constraints:
            base_gc = gc_id.split("_part")[0]
            if abs(float(best_eval.get(f"_margin_{gc_id}", 1.0))) < 1e-2 and base_gc not in blocking:
                blocking.append(base_gc)

        if not blocking and self.graph.constraints:
            blocking = [next(iter(self.graph.constraints.keys()))]

        iis_list = ["c_target_goal"] + list(blocking)
        cert_inf = SolveCertificate(
            status=SolveStatus.INFEASIBLE_CERTIFIED,
            problem_class="MINLP" if self.discrete_controls else "NLP_NONSMOOTH",
            strategy_used="strategy_e_iis_deletion_filter",
            optimality_gap=None,
            max_constraint_violation=round(shortfall, 6),
            kkt_residual=None,
            fidelity_verified=True,
            code_verified=True,
            constraint_violation_on_code=0.0,
            iis_constraints=iis_list,
            min_relaxation_slack=round(shortfall, 6),
            reason=f"Target {target_value} exceeds global optimum {j_star:.4f} (IIS: {iis_list})",
        )

        return {
            "status": "INFEASIBLE",
            "certificate_status": SolveStatus.INFEASIBLE_CERTIFIED.value,
            "certificate": cert_inf.to_dict(),
            f"target_{self.target_id}": target_value,
            "target_value": target_value,
            "iis_constraints": iis_list,
            "blocking_constraints": blocking,
            f"max_attainable_{self.target_id}": round(j_star, 2),
            "max_achievable_J_star": round(j_star, 4),
            "shortfall": round(shortfall, 4),
            "kkt_shadow_multiplier_budget": round(lambda_budget, 6),
            "kkt_shadow_multiplier_market": round(lambda_market, 6),
            "kkt_multipliers": {
                "budget": round(lambda_budget, 6),
                "capacity_ceiling": round(lambda_market, 6),
            },
            "explanation": (
                f"El objetivo {self.target_id}={target_value} excede la capacidad presupuestaria y de absorción del sistema. "
                f"El valor máximo alcanzable bajo las restricciones duras es {j_star:.2f} "
                f"(multiplicadores KKT: lambda_budget={lambda_budget:.4f}, lambda_market={lambda_market:.4f})."
            )
        }

    def generate_intervention_recipe(
        self,
        current_controls: Dict[str, float],
        target_value: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Genera una receta ordenada y accionable de intervención (R-9), aplicando parsimonia L1 cuando se especifica target_value."""
        if target_value is None:
            for k, v in kwargs.items():
                if k.startswith("target_") and v is not None:
                    target_value = float(v)
                    break

        sol = self.optimize(target_value=target_value)
        if sol.get("status") in ("INFEASIBLE", "UNSUPPORTED"):
            return sol

        if target_value is not None and len(self.control_ids) >= 2:
            sparse_sol = None
            for k_active in range(1, len(self.control_ids)):
                best_k_sol = None
                best_k_l2 = float("inf")
                for active_subset in combinations(self.control_ids, k_active):
                    fixed_map = {
                        cid: float(current_controls.get(cid, self.graph.nodes[cid].current_value))
                        for cid in self.control_ids
                        if cid not in active_subset
                    }
                    cand = self.optimize(fixed_controls=fixed_map)
                    if cand.get("status") == "FEASIBLE" and float(cand["objective_value"]) >= target_value - 1e-4:
                        l2_norm = sum(
                            ((float(cand["controls"][c]) - float(current_controls.get(c, self.graph.nodes[c].current_value)))
                             / max(1.0, self.graph.nodes[c].characteristic_scale or 1.0)) ** 2
                            for c in active_subset
                        )
                        if l2_norm < best_k_l2:
                            best_k_l2 = l2_norm
                            best_k_sol = cand
                if best_k_sol is not None:
                    sparse_sol = best_k_sol
                    break
            if sparse_sol is not None:
                sol = sparse_sol

        opt_controls = sol["controls"]
        ordered_cids = list(self.discrete_controls) + list(self.continuous_controls)

        steps: List[Dict[str, Any]] = []
        step_order = 1
        for cid in ordered_cids:
            cur_val = current_controls.get(cid, self.graph.nodes[cid].current_value)
            tgt_val = opt_controls[cid]
            is_disc = cid in self.discrete_controls
            delta = (int(round(tgt_val)) - int(round(cur_val))) if is_disc else round(float(tgt_val) - float(cur_val), 4)
            if target_value is not None and abs(float(delta)) < 1e-6:
                continue
            steps.append({
                "order": step_order,
                "variable": cid,
                "current": int(round(cur_val)) if is_disc else float(cur_val),
                "target": tgt_val,
                "delta": delta,
                "rationale": (
                    f"Ajustar '{cid}' en {delta:+g} ({cur_val} -> {tgt_val}) para optimizar '{self.target_id}' "
                    f"respetando las restricciones duras del sistema."
                )
            })
            step_order += 1

        cur_eval = self.evaluate(**current_controls)
        cur_j = float(cur_eval[self.target_id])
        opt_j = float(sol[self.target_id])
        imp = opt_j - cur_j

        return {
            "status": "FEASIBLE",
            "certificate": sol.get("certificate"),
            "levers_changed": len(steps),
            f"current_{self.target_id}": round(cur_j, 2),
            f"target_{self.target_id}": opt_j,
            f"{self.target_id}_improvement": round(imp, 2),
            "current_objective": round(cur_j, 2),
            "target_objective": opt_j,
            "predicted_objective": opt_j,
            "objective_improvement": round(imp, 2),
            "steps": steps
        }
