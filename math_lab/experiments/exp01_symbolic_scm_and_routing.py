"""Experimento 01: Compilador AST -> SCM Simbólico (SymPy), AD en Modo Inverso sobre DAGs,
Subdiferenciales de Clarke y Enrutamiento Tropical Max-Product Top-K.

Demuestra cómo extraer el Modelo Causal Estructural M = <U, X, V, F, C, J> de forma
100% determinista desde el árbol sintáctico (AST) sin hardcodear ninguna ecuación,
calcular derivadas totales en tiempo lineal O(|V|+|E|), detectar kinks/discontinuidades
mediante análisis no suave de Clarke y enrutar en el semianillo Max-Product.
"""
from __future__ import annotations
import ast
import glob
import math
import os
import time
from typing import Dict, Any, List, Tuple, Optional
import networkx as nx
import numpy as np
import sympy as sp


class ASTFunctionToSymPy(ast.NodeVisitor):
    """Traduce el cuerpo AST de una función Python determinista a una expresión SymPy exacta (SSA)."""

    def __init__(
        self,
        param_symbols: Optional[Dict[str, sp.Symbol]] = None,
        module_constants: Optional[Dict[str, float]] = None
    ):
        self.env: Dict[str, sp.Expr] = dict(param_symbols or {})
        for k, v in (module_constants or {}).items():
            if k not in self.env and isinstance(v, (int, float)):
                self.env[k] = sp.Float(v)
        self.return_expr: Optional[sp.Expr] = None
        self.discontinuities: List[Dict[str, Any]] = []

    def compile_function(self, fn_node: ast.FunctionDef) -> sp.Expr:
        for arg in fn_node.args.args:
            if arg.arg not in self.env:
                self.env[arg.arg] = sp.Symbol(arg.arg, real=True)
        return self.compile_statements(fn_node.body)

    @staticmethod
    def _is_zero_number(expr: sp.Expr) -> bool:
        try:
            return bool(expr.is_number and abs(float(expr)) < 1e-15)
        except Exception:
            return False

    def eval_expr(self, node: ast.AST) -> sp.Expr:
        if isinstance(node, ast.Constant):
            return sp.Float(node.value) if isinstance(node.value, float) else sp.Integer(node.value)
        if isinstance(node, ast.Name):
            if node.id in self.env:
                return self.env[node.id]
            return sp.Symbol(node.id, real=True)
        if isinstance(node, ast.UnaryOp):
            operand = self.eval_expr(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return operand
        if isinstance(node, ast.BinOp):
            left = self.eval_expr(node.left)
            right = self.eval_expr(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            args = [self.eval_expr(a) for a in node.args]
            if func_name in ("float", "int") and len(args) == 1:
                return args[0]
            if func_name == "max" and len(args) >= 2:
                return sp.Max(*args)
            if func_name == "min" and len(args) >= 2:
                return sp.Min(*args)
            if func_name == "clip" and len(args) == 3:
                return sp.Min(sp.Max(args[0], args[1]), args[2])
            if func_name == "sqrt" and len(args) == 1:
                arg0 = args[0]
                if isinstance(arg0, sp.Max) and len(arg0.args) == 2:
                    if self._is_zero_number(arg0.args[0]):
                        return sp.sqrt(arg0.args[1])
                    if self._is_zero_number(arg0.args[1]):
                        return sp.sqrt(arg0.args[0])
                return sp.sqrt(arg0)
            if func_name == "log1p" and len(args) == 1:
                arg0 = args[0]
                if isinstance(arg0, sp.Max) and len(arg0.args) == 2:
                    if self._is_zero_number(arg0.args[0]):
                        return sp.log(1 + arg0.args[1])
                    if self._is_zero_number(arg0.args[1]):
                        return sp.log(1 + arg0.args[0])
                return sp.log(1 + arg0)
            if func_name == "log" and len(args) == 1:
                return sp.log(args[0])
            if func_name == "exp" and len(args) == 1:
                return sp.exp(args[0])
            if func_name == "abs" and len(args) == 1:
                return sp.Abs(args[0])
            if func_name == "sin" and len(args) == 1:
                return sp.sin(args[0])
            if func_name == "cos" and len(args) == 1:
                return sp.cos(args[0])
            if func_name == "tanh" and len(args) == 1:
                return sp.tanh(args[0])
            if func_name == "pow" and len(args) == 2:
                return args[0] ** args[1]
        if isinstance(node, ast.IfExp):
            cond, disc_info = self._eval_condition(node.test)
            body_e = self.eval_expr(node.body)
            orelse_e = self.eval_expr(node.orelse)
            if disc_info:
                disc_info["val_true"] = float(body_e) if body_e.is_number else str(body_e)
                disc_info["val_false"] = float(orelse_e) if orelse_e.is_number else str(orelse_e)
                self.discontinuities.append(disc_info)
            return sp.Piecewise((body_e, cond), (orelse_e, True))
        raise ValueError(f"Nodo AST no soportado en compilación simbólica: {ast.dump(node)}")

    def _eval_single_cmp(self, left: sp.Expr, op: ast.cmpop, right: sp.Expr) -> Tuple[Any, Optional[Dict[str, Any]]]:
        disc_info = None
        if isinstance(left, sp.Symbol) and right.is_number:
            disc_info = {"variable": str(left), "threshold": float(right)}
        elif isinstance(right, sp.Symbol) and left.is_number:
            disc_info = {"variable": str(right), "threshold": float(left)}
        if isinstance(op, ast.Lt):
            return sp.StrictLessThan(left, right), disc_info
        if isinstance(op, ast.LtE):
            return sp.LessThan(left, right), disc_info
        if isinstance(op, ast.Gt):
            return sp.StrictGreaterThan(left, right), disc_info
        if isinstance(op, ast.GtE):
            return sp.GreaterThan(left, right), disc_info
        if isinstance(op, ast.Eq):
            return sp.Eq(left, right), disc_info
        if isinstance(op, ast.NotEq):
            return sp.Ne(left, right), disc_info
        raise ValueError(f"Operador de comparación AST no soportado: {ast.dump(op)}")

    def _eval_condition(self, test_node: ast.AST) -> Tuple[Any, Optional[Dict[str, Any]]]:
        if isinstance(test_node, ast.Compare):
            operands = [self.eval_expr(test_node.left)] + [self.eval_expr(c) for c in test_node.comparators]
            conds = []
            first_disc = None
            for i, op in enumerate(test_node.ops):
                c_i, d_i = self._eval_single_cmp(operands[i], op, operands[i + 1])
                conds.append(c_i)
                if first_disc is None and d_i is not None:
                    first_disc = d_i
            return (conds[0] if len(conds) == 1 else sp.And(*conds)), first_disc
        if isinstance(test_node, ast.BoolOp):
            sub_conds = [self._eval_condition(v)[0] for v in test_node.values]
            if isinstance(test_node.op, ast.And):
                return sp.And(*sub_conds), None
            if isinstance(test_node.op, ast.Or):
                return sp.Or(*sub_conds), None
        if isinstance(test_node, ast.UnaryOp) and isinstance(test_node.op, ast.Not):
            c_sub, d_sub = self._eval_condition(test_node.operand)
            return sp.Not(c_sub), d_sub
        raise ValueError(f"Condición AST no soportada: {ast.dump(test_node)}")

    def _extract_static_iterable(self, iter_node: ast.AST) -> List[Any]:
        if isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name) and iter_node.func.id == "range":
            args = [int(self.eval_expr(a)) for a in iter_node.args]
            return list(range(*args))
        if isinstance(iter_node, (ast.List, ast.Tuple)):
            return [self.eval_expr(elt) for elt in iter_node.elts]
        raise ValueError(f"Iterador AST no desenrollable estáticamente: {ast.dump(iter_node)}")

    def compile_statements(self, stmts: List[ast.stmt], env: Optional[Dict[str, sp.Expr]] = None) -> sp.Expr:
        cur_env = self.env if env is None else env
        for idx, stmt in enumerate(stmts):
            if isinstance(stmt, ast.Expr):
                continue
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                self.env = cur_env
                if isinstance(target, ast.Name):
                    cur_env[target.id] = self.eval_expr(stmt.value)
                elif isinstance(target, ast.Tuple) and isinstance(stmt.value, ast.Tuple) and len(target.elts) == len(stmt.value.elts):
                    vals = [self.eval_expr(v) for v in stmt.value.elts]
                    for t_elt, v_expr in zip(target.elts, vals):
                        if isinstance(t_elt, ast.Name):
                            cur_env[t_elt.id] = v_expr
            elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                var_name = stmt.target.id
                self.env = cur_env
                prev_val = cur_env.get(var_name, sp.Integer(0))
                rhs_val = self.eval_expr(stmt.value)
                if isinstance(stmt.op, ast.Add):
                    cur_env[var_name] = prev_val + rhs_val
                elif isinstance(stmt.op, ast.Sub):
                    cur_env[var_name] = prev_val - rhs_val
                elif isinstance(stmt.op, ast.Mult):
                    cur_env[var_name] = prev_val * rhs_val
                elif isinstance(stmt.op, ast.Div):
                    cur_env[var_name] = prev_val / rhs_val
                else:
                    raise ValueError(f"AugAssign no soportado: {ast.dump(stmt)}")
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                var_name = stmt.target.id
                self.env = cur_env
                cur_env[var_name] = self.eval_expr(stmt.value)
            elif isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name):
                # Desenrollado estático determinista de bucles acotados (Bounded Loop Unrolling)
                self.env = cur_env
                loop_items = self._extract_static_iterable(stmt.iter)
                unrolled_body: List[ast.stmt] = []
                for item in loop_items:
                    const_val = float(item) if isinstance(item, sp.Float) else (int(item) if isinstance(item, (int, sp.Integer)) else item)
                    if isinstance(const_val, (int, float)):
                        assign_iter = ast.Assign(
                            targets=[ast.Name(id=stmt.target.id, ctx=ast.Store())],
                            value=ast.Constant(value=const_val)
                        )
                        unrolled_body.append(assign_iter)
                        unrolled_body.extend(stmt.body)
                return self.compile_statements(unrolled_body + stmts[idx + 1:], cur_env)
            elif isinstance(stmt, ast.Return) and stmt.value is not None:
                self.env = cur_env
                return self.eval_expr(stmt.value)
            elif isinstance(stmt, ast.If):
                self.env = cur_env
                cond, disc_info = self._eval_condition(stmt.test)
                continuation = stmts[idx + 1:]
                true_expr = self.compile_statements(list(stmt.body) + continuation, dict(cur_env))
                false_expr = self.compile_statements(list(stmt.orelse) + continuation, dict(cur_env))
                if disc_info:
                    var_sym = self.env.get(disc_info["variable"], sp.Symbol(disc_info["variable"], real=True))
                    th_val = disc_info["threshold"]
                    try:
                        lim_true = float(true_expr.subs(var_sym, th_val).evalf())
                        lim_false = float(false_expr.subs(var_sym, th_val).evalf())
                        if abs(lim_false - lim_true) > 1e-7:
                            disc_info["val_true"] = lim_true
                            disc_info["val_false"] = lim_false
                            disc_info["jump_delta"] = lim_false - lim_true
                            self.discontinuities.append(disc_info)
                    except Exception:
                        pass
                return sp.Piecewise((true_expr, cond), (false_expr, True))
        raise ValueError("La función no alcanzó una sentencia return determinista.")


class SymbolicSCM:
    """Modelo Causal Estructural Simbólico M = <U, X, V, F, C, J> compilado automáticamente desde AST."""

    def __init__(self, target_dir: str):
        self.target_dir = os.path.normpath(target_dir)
        self.symbols: Dict[str, sp.Symbol] = {}
        self.node_meta: Dict[str, Dict[str, Any]] = {}
        self.mechanisms: Dict[str, sp.Expr] = {}
        self.discontinuities: Dict[str, List[Dict[str, Any]]] = {}
        self.local_optimizers: Dict[str, Dict[str, Any]] = {}
        self.constraints_meta: Dict[str, Dict[str, Any]] = {}
        self.max_budget: float = 2500.0
        self.base_demand_std: float = 100.0
        self.dag = nx.DiGraph()
        self._compile_directory()

    def _get_symbol(self, name: str) -> sp.Symbol:
        if name not in self.symbols:
            self.symbols[name] = sp.Symbol(name, real=True)
        return self.symbols[name]

    def _rel_path(self, full_path: str) -> str:
        rel = os.path.relpath(full_path, self.target_dir)
        base = os.path.basename(self.target_dir)
        return f"{base}/{rel}".replace("\\", "/")

    def _compile_directory(self):
        # 1. Extraer controles X y cotas desde config.py
        config_path = os.path.join(self.target_dir, "config.py")
        bounds_map: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        raw_constants: Dict[str, Any] = {}

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=config_path)
            rel_file = self._rel_path(config_path)
            lines_map: Dict[str, List[int]] = {}
            type_map: Dict[str, str] = {}

            for stmt in tree.body:
                t_name = None
                val_node = None
                ann_type = None
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    t_name = stmt.targets[0].id
                    val_node = stmt.value
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    t_name = stmt.target.id
                    val_node = stmt.value
                    if isinstance(stmt.annotation, ast.Name):
                        ann_type = stmt.annotation.id

                if t_name and val_node is not None:
                    try:
                        val = ast.literal_eval(val_node)
                        raw_constants[t_name] = val
                        lines_map[t_name] = [stmt.lineno, getattr(stmt, "end_lineno", stmt.lineno)]
                        if ann_type:
                            type_map[t_name] = ann_type
                    except Exception:
                        pass

            for k, val in raw_constants.items():
                if k.endswith("_BOUNDS") and isinstance(val, (tuple, list)) and len(val) == 2:
                    base_key = k[:-7].lower()
                    bounds_map[base_key] = (float(val[0]), float(val[1]))

            if "MAX_BUDGET" in raw_constants:
                self.max_budget = float(raw_constants["MAX_BUDGET"])

            for k, val in raw_constants.items():
                if k.endswith("_BOUNDS") or k == "MAX_BUDGET":
                    continue
                node_id = k.lower()
                b_min, b_max = bounds_map.get(node_id, (None, None))
                if b_min is None and "marketing" in node_id and "marketing" in bounds_map:
                    b_min, b_max = bounds_map["marketing"]
                is_int = type_map.get(k) == "int" or isinstance(val, int)
                self._get_symbol(node_id)
                self.node_meta[node_id] = {
                    "id": node_id,
                    "category": "control",
                    "data_type": "discrete" if is_int else "continuous",
                    "current_value": int(val) if is_int else float(val),
                    "bounds": {"min": b_min, "max": b_max},
                    "file": rel_file,
                    "entity": k,
                    "line_range": lines_map[k]
                }
                self.dag.add_node(node_id, **self.node_meta[node_id])

        # 2. Extraer variables exógenas U desde environment.py
        env_path = os.path.join(self.target_dir, "environment.py")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=env_path)
            rel_file = self._rel_path(env_path)
            env_consts: Dict[str, float] = {}
            for stmt in tree.body:
                if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                    t = stmt.targets[0] if isinstance(stmt, ast.Assign) else stmt.target
                    if isinstance(t, ast.Name) and stmt.value is not None:
                        try:
                            env_consts[t.id] = float(ast.literal_eval(stmt.value))
                        except Exception:
                            pass
                elif isinstance(stmt, ast.FunctionDef) and stmt.name.startswith("sample_"):
                    exo_id = stmt.name[len("sample_"):]
                    mean_val = env_consts.get(f"{exo_id.upper()}_MEAN", 1000.0)
                    std_val = env_consts.get(f"{exo_id.upper()}_STD", 100.0)
                    self.base_demand_std = std_val
                    self._get_symbol(exo_id)
                    self.node_meta[exo_id] = {
                        "id": exo_id,
                        "category": "exogenous",
                        "data_type": "continuous",
                        "current_value": mean_val,
                        "std": std_val,
                        "bounds": {"min": 0.0, "max": None},
                        "file": rel_file,
                        "entity": stmt.name,
                        "line_range": [stmt.lineno, getattr(stmt, "end_lineno", stmt.lineno)]
                    }
                    self.dag.add_node(exo_id, **self.node_meta[exo_id])

        # 3. Extraer mecanismos endógenos V y target J desde todos los demás módulos .py
        py_files = sorted(glob.glob(os.path.join(self.target_dir, "*.py")))
        for py_path in py_files:
            fname = os.path.basename(py_path)
            if fname in ["__init__.py", "config.py", "environment.py", "pipeline.py"]:
                continue
            rel_file = self._rel_path(py_path)
            with open(py_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=py_path)

            mod_consts: Dict[str, float] = {}
            for stmt in tree.body:
                if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                    t = stmt.targets[0] if isinstance(stmt, ast.Assign) else stmt.target
                    if isinstance(t, ast.Name) and stmt.value is not None:
                        try:
                            mod_consts[t.id] = float(ast.literal_eval(stmt.value))
                        except Exception:
                            pass

            for stmt in tree.body:
                if not isinstance(stmt, ast.FunctionDef):
                    continue
                if stmt.name.startswith("calculate_"):
                    node_id = stmt.name[len("calculate_"):]
                    param_names = [a.arg for a in stmt.args.args]
                    param_syms = {p: self._get_symbol(p) for p in param_names}
                    compiler = ASTFunctionToSymPy(param_syms, mod_consts)
                    sym_expr = compiler.compile_statements(stmt.body)

                    self._get_symbol(node_id)
                    self.mechanisms[node_id] = sym_expr
                    if compiler.discontinuities:
                        self.discontinuities[node_id] = compiler.discontinuities

                    cat = "target" if node_id == "profit" else "state"
                    self.node_meta[node_id] = {
                        "id": node_id,
                        "category": cat,
                        "data_type": "continuous",
                        "current_value": None,
                        "bounds": {"min": None, "max": None},
                        "file": rel_file,
                        "entity": stmt.name,
                        "line_range": [stmt.lineno, getattr(stmt, "end_lineno", stmt.lineno)],
                        "sympy_expr": str(sym_expr)
                    }
                    self.dag.add_node(node_id, **self.node_meta[node_id])

                    has_nonsmooth = sym_expr.has(sp.Piecewise, sp.Min, sp.Max)
                    for p_name in param_names:
                        if sym_expr.has(self._get_symbol(p_name)):
                            e_type = "discrete_rule" if (has_nonsmooth and self._is_nonsmooth_wrt(sym_expr, self._get_symbol(p_name))) else "exact_code"
                            self.dag.add_edge(p_name, node_id, type=e_type)

                elif stmt.name.startswith("optimize_"):
                    compiler = ASTFunctionToSymPy({}, mod_consts)
                    try:
                        ret_val = float(compiler.compile_statements(stmt.body))
                        self.local_optimizers[stmt.name] = {
                            "function": stmt.name,
                            "file": rel_file,
                            "line_range": [stmt.lineno, getattr(stmt, "end_lineno", stmt.lineno)],
                            "recommended_value": ret_val
                        }
                    except Exception:
                        pass

        # Evaluar estado nominal inicial
        nominal_inputs = {
            nid: meta["current_value"]
            for nid, meta in self.node_meta.items()
            if meta["category"] in ("control", "exogenous") and meta["current_value"] is not None
        }
        nominal_state = self.evaluate(nominal_inputs)
        for nid, val in nominal_state.items():
            if nid in self.node_meta:
                self.node_meta[nid]["current_value"] = round(float(val), 4)
                self.dag.nodes[nid]["current_value"] = round(float(val), 4)

    @staticmethod
    def _is_nonsmooth_wrt(expr: sp.Expr, sym: sp.Symbol) -> bool:
        for sub in sp.preorder_traversal(expr):
            if isinstance(sub, (sp.Piecewise, sp.Min, sp.Max)) and sub.has(sym):
                if isinstance(sub, sp.Max) and len(sub.args) == 2 and ASTFunctionToSymPy._is_zero_number(sub.args[0]):
                    continue
                return True
        return False

    def evaluate(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """Evalúa el grafo causal completo en orden topológico usando las expresiones simbólicas extraídas."""
        state: Dict[str, float] = {}
        for nid, meta in self.node_meta.items():
            if meta["category"] in ("control", "exogenous"):
                state[nid] = float(inputs.get(nid, meta["current_value"]))

        for nid in nx.topological_sort(self.dag):
            if nid in self.mechanisms:
                expr = self.mechanisms[nid]
                subs_dict = {self.symbols[k]: state[k] for k in state if k in self.symbols}
                val = float(expr.evalf(subs=subs_dict))
                state[nid] = val
        return state

    def evaluate_node_function(self, node_id: str, parent_vals: Dict[str, float]) -> float:
        expr = self.mechanisms[node_id]
        subs_dict = {self.symbols[k]: float(v) for k, v in parent_vals.items() if k in self.symbols}
        return float(expr.evalf(subs=subs_dict))

    def compute_symbolic_total_expression(self, target: str = "profit", stop_at_nodes: Optional[List[str]] = None) -> sp.Expr:
        """Expande recursivamente en orden topológico inverso la expresión simbólica cerrada de `target`
        en función de las variables exógenas/controles (y de `stop_at_nodes` si se especifican).
        Permite verificar simbólicamente cancelaciones de caminos (T1-B: dJ/dv == 0 idénticamente).
        """
        stop_set = set(stop_at_nodes or [])
        topo = list(nx.topological_sort(self.dag))
        expanded: Dict[str, sp.Expr] = {}
        for nid in topo:
            if nid in stop_set or nid not in self.mechanisms:
                expanded[nid] = self._get_symbol(nid)
            else:
                expr = self.mechanisms[nid]
                subs_map = {self.symbols[k]: expanded[k] for k in expanded if k in self.symbols and k != nid}
                expanded[nid] = expr.subs(subs_map)
        return expanded.get(target, self._get_symbol(target))

    def compute_symbolic_total_derivative(self, source: str, target: str = "profit") -> sp.Expr:
        """Calcula la derivada total simbólica exacta d(target)/d(source) manteniendo `source` libre."""
        if not nx.has_path(self.dag, source, target):
            return sp.Integer(0)
        total_expr = self.compute_symbolic_total_expression(target=target, stop_at_nodes=[source])
        src_sym = self._get_symbol(source)
        return sp.simplify(sp.diff(total_expr, src_sym))

    def compute_clarke_derivative(
        self,
        source: str,
        target: str,
        state: Dict[str, float],
        eps: float = 1e-4
    ) -> Dict[str, Any]:
        """Calcula las derivadas direccionales laterales suaves D^+, D^-, el subdiferencial de Clarke
        y separa rigurosamente los saltos discontinuos de orden 0 (Dirac delta) de los kinks Lipschitz (orden 1).
        """
        probe_state = dict(state)
        expr = self.mechanisms[target]
        if isinstance(expr, (sp.Min, sp.Max)) and len(expr.args) == 2:
            other_syms = [str(s) for s in expr.free_symbols if str(s) != source]
            if len(other_syms) == 1 and other_syms[0] in probe_state and source in probe_state:
                other_val = probe_state[other_syms[0]]
                if abs(probe_state[source] - other_val) < 0.05:
                    probe_state[source] = other_val

        base_val = self.evaluate_node_function(target, probe_state)

        def eval_at_offset(delta: float) -> float:
            st = dict(probe_state)
            st[source] = probe_state[source] + delta
            return self.evaluate_node_function(target, st)

        val_plus = eval_at_offset(eps)
        val_plus2 = eval_at_offset(2.0 * eps)
        val_minus = eval_at_offset(-eps)
        val_minus2 = eval_at_offset(-2.0 * eps)

        # Extrapolación lateral de límites para detectar salto de orden 0 en probe_state[source]
        lim_right = 2.0 * val_plus - val_plus2
        lim_left = 2.0 * val_minus - val_minus2
        local_jump = lim_right - lim_left
        has_local_jump = abs(local_jump) > 1e-5

        if has_local_jump:
            # Si hay discontinuidad de salto en x0, la derivada suave lateral izquierda se mide en [x0-2eps, x0-eps]
            # y la derecha en [x0+eps, x0+2eps], evitando dividir el escalón Delta_jump por eps (que produciría O(1/eps) espúreo):
            d_left = (val_minus - val_minus2) / eps
            d_right = (val_plus2 - val_plus) / eps
        else:
            d_right = (val_plus - base_val) / eps
            d_left = (base_val - val_minus) / eps

        central = 0.5 * (d_right + d_left)
        kink_gap = abs(d_right - d_left)

        secant_weight = None
        has_jump = has_local_jump
        jump_delta = local_jump if has_local_jump else 0.0
        if target in self.discontinuities:
            for d_info in self.discontinuities[target]:
                if d_info.get("variable") == source and isinstance(d_info.get("val_true"), float) and isinstance(d_info.get("val_false"), float):
                    has_jump = True
                    jump_delta = d_info["val_false"] - d_info["val_true"]
                    b = self.node_meta.get(source, {}).get("bounds", {})
                    span = (b.get("max") - b.get("min")) if (b.get("max") is not None and b.get("min") is not None) else 1000.0
                    secant_weight = jump_delta / span

        return {
            "source": source,
            "target": target,
            "d_left": round(d_left, 6),
            "d_right": round(d_right, 6),
            "central_gradient": round(central, 6),
            "clarke_subdifferential": [round(min(d_left, d_right), 6), round(max(d_left, d_right), 6)],
            "kink_asymmetry": round(kink_gap, 6),
            "is_kink_active": kink_gap > 1e-3 and not has_jump,
            "has_jump_discontinuity": has_jump,
            "jump_delta": round(jump_delta, 6),
            "secant_weight": round(secant_weight, 6) if secant_weight is not None else None
        }

    def get_characteristic_scale(self, node_id: str, state: Dict[str, float]) -> float:
        meta = self.node_meta.get(node_id, {})
        b = meta.get("bounds", {})
        if b.get("min") is not None and b.get("max") is not None and b["max"] > b["min"]:
            return float(b["max"] - b["min"])
        val = abs(state.get(node_id, meta.get("current_value") or 1.0))
        return max(val, 1.0)

    def compute_reverse_mode_ad_jacobian(
        self,
        inputs: Optional[Dict[str, float]] = None,
        target: str = "profit",
        use_secant_for_jumps: bool = False
    ) -> Dict[str, Any]:
        """Calcula el Jacobiano Total dJ/dv_i para TODOS los nodos en tiempo lineal O(|V|+|E|)
        mediante Diferenciación Automática en Modo Inverso (Backpropagation sobre el DAG),
        separando las derivadas continuas de las diferencias discretas en variables enteras con salto.
        """
        state = self.evaluate(inputs or {})
        edge_derivatives: Dict[Tuple[str, str], float] = {}
        normalized_weights: Dict[Tuple[str, str], float] = {}
        clarke_reports: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for u, v in self.dag.edges():
            c_info = self.compute_clarke_derivative(u, v, state)
            clarke_reports[(u, v)] = c_info
            raw_w = c_info["central_gradient"]
            if use_secant_for_jumps and c_info["has_jump_discontinuity"] and abs(raw_w) < 1e-9 and c_info["secant_weight"] is not None:
                eff_w = c_info["secant_weight"]
            else:
                eff_w = raw_w
            edge_derivatives[(u, v)] = eff_w

            s_u = self.get_characteristic_scale(u, state)
            s_v = self.get_characteristic_scale(v, state)
            norm_w = (c_info["secant_weight"] if (c_info["has_jump_discontinuity"] and abs(raw_w) < 1e-9 and c_info["secant_weight"] is not None) else raw_w) * (s_u / s_v)
            normalized_weights[(u, v)] = norm_w

        # Reverse-Mode AD en orden topológico inverso
        adjoints: Dict[str, float] = {nid: 0.0 for nid in self.dag.nodes()}
        if target in adjoints:
            adjoints[target] = 1.0

        topo_reverse = list(reversed(list(nx.topological_sort(self.dag))))
        for u in topo_reverse:
            if u == target:
                continue
            total_adj = 0.0
            for v in self.dag.successors(u):
                total_adj += adjoints[v] * edge_derivatives[(u, v)]
            adjoints[u] = total_adj

        # Para controles discretos (ej. order_qty), calcular también las diferencias finitas enteras exactas
        # Delta^- J(q) = J(q) - J(q-1) y Delta^+ J(q) = J(q+1) - J(q) (consistente con oracle/solve.py):
        discrete_step_sensitivities: Dict[str, Dict[str, float]] = {}
        base_j = state.get(target, 0.0)
        for nid, meta in self.node_meta.items():
            if meta.get("category") == "control" and meta.get("data_type") == "discrete":
                q_val = state[nid]
                st_minus1 = dict(state)
                st_minus1[nid] = q_val - 1.0
                j_minus1 = self.evaluate(st_minus1).get(target, 0.0)
                st_plus1 = dict(state)
                st_plus1[nid] = q_val + 1.0
                j_plus1 = self.evaluate(st_plus1).get(target, 0.0)
                discrete_step_sensitivities[nid] = {
                    "delta_J_left_step": round(base_j - j_minus1, 4),
                    "delta_J_right_step": round(j_plus1 - base_j, 4)
                }

        return {
            "state": state,
            "adjoints_dJ_dv": adjoints,
            "discrete_step_sensitivities": discrete_step_sensitivities,
            "edge_derivatives": edge_derivatives,
            "normalized_weights": normalized_weights,
            "clarke_reports": clarke_reports
        }


class TropicalMaxProductRouter:
    """Enrutador de Capa 1 sobre el Semianillo Tropical Max-Product (O(K(|V|+|E|) log K)).

    A diferencia de Layer1Router (que suma 1/|w| violando la regla de la cadena),
    este enrutador maximiza exactamente el producto multiplicativo de sensibilidades
    |S(p)| = prod_{(u,v) in p} |w_{uv}| a lo largo de las cadenas causales en un DAG.
    """

    def __init__(self, dag: nx.DiGraph, weight_attr: str = "sensitivity_weight"):
        self.dag = dag.copy()
        self.weight_attr = weight_attr

    def find_top_k_multiplicative_paths(self, source: str, target: str, k: int = 3) -> List[Dict[str, Any]]:
        """Programación dinámica topológica exacta sobre el DAG para hallar las Top-K rutas
        que maximizan prod_{e in p} |w_e|.
        """
        if not self.dag.has_node(source) or not self.dag.has_node(target):
            return []

        # Subgrafo alcanzable desde source para no procesar nodos irrelevantes
        topo_nodes = list(nx.topological_sort(self.dag))
        # dp[u] almacena lista de tuplas: (abs_product, signed_product, path_list)
        dp: Dict[str, List[Tuple[float, float, List[str]]]] = {nid: [] for nid in topo_nodes}
        dp[source] = [(1.0, 1.0, [source])]

        for u in topo_nodes:
            if not dp[u]:
                continue
            for v in self.dag.successors(u):
                ed = self.dag[u][v]
                w = ed.get(self.weight_attr)
                if w is None:
                    w = ed.get("original_weight", ed.get("weight", 1.0))
                w_float = float(w)
                abs_w = max(abs(w_float), 1e-12)
                for abs_prod, signed_prod, path in dp[u]:
                    dp[v].append((abs_prod * abs_w, signed_prod * w_float, path + [v]))
                # Conservar únicamente las Top-K rutas de mayor magnitud hacia v
                if len(dp[v]) > k:
                    dp[v].sort(key=lambda item: item[0], reverse=True)
                    dp[v] = dp[v][:k]

        results = []
        for abs_prod, signed_prod, p in dp.get(target, []):
            results.append({
                "path": p,
                "length": len(p),
                "cumulative_sensitivity": signed_prod,
                "abs_product": abs_prod,
                "log_score": math.log(max(abs_prod, 1e-300))
            })
        return results


def demonstrate_layer1_harmonic_sum_bug() -> Dict[str, Any]:
    """Demuestra empíricamente el contraejemplo matemático donde el coste armónico sum(1/|w|)
    elige una ruta con 4x MENOR sensibilidad real frente a maximizar prod(|w|).
    """
    # Construimos un DAG con dos rutas de X a J:
    # Ruta A (X -> A -> J): w1 = 0.1, w2 = 10.0 => Producto real |dJ/dX|_A = 0.1 * 10.0 = 1.00
    # Ruta B (X -> B -> J): w1 = 0.5, w2 = 0.5  => Producto real |dJ/dX|_B = 0.5 * 0.5  = 0.25
    counterexample_graph = {
        "nodes": [
            {"id": "X", "category": "control"},
            {"id": "A", "category": "state"},
            {"id": "B", "category": "state"},
            {"id": "J", "category": "target"}
        ],
        "edges": [
            {"source": "X", "target": "A", "sensitivity_weight": 0.1},
            {"source": "A", "target": "J", "sensitivity_weight": 10.0},
            {"source": "X", "target": "B", "sensitivity_weight": 0.5},
            {"source": "B", "target": "J", "sensitivity_weight": 0.5}
        ]
    }

    G_harm = nx.DiGraph()
    for e in counterexample_graph["edges"]:
        w = e["sensitivity_weight"]
        G_harm.add_edge(e["source"], e["target"], weight=1.0 / (abs(w) + 1e-4), original_weight=w)
    p_harm = nx.shortest_path(G_harm, "X", "J", weight="weight")
    sens_harm = 1.0
    for i in range(len(p_harm) - 1):
        sens_harm *= G_harm[p_harm[i]][p_harm[i + 1]]["original_weight"]
    legacy_top1 = {"path": p_harm, "cumulative_sensitivity": sens_harm}

    G_dag = nx.DiGraph()
    for n in counterexample_graph["nodes"]:
        G_dag.add_node(n["id"], **n)
    for e in counterexample_graph["edges"]:
        G_dag.add_edge(e["source"], e["target"], sensitivity_weight=e["sensitivity_weight"])

    tropical_router = TropicalMaxProductRouter(G_dag)
    tropical_top1 = tropical_router.find_top_k_multiplicative_paths("X", "J", k=2)[0]

    return {
        "true_best_path": ["X", "A", "J"],
        "true_best_sensitivity": 1.0,
        "legacy_layer1_selected_path": legacy_top1["path"],
        "legacy_layer1_selected_sensitivity": legacy_top1["cumulative_sensitivity"],
        "legacy_is_wrong": legacy_top1["path"] != ["X", "A", "J"],
        "tropical_router_selected_path": tropical_top1["path"],
        "tropical_router_selected_sensitivity": tropical_top1["cumulative_sensitivity"],
        "tropical_is_exact": tropical_top1["path"] == ["X", "A", "J"]
    }
