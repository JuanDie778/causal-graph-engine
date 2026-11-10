"""Pipeline de Ingesta Simbólica AST y Deducción Automática del Modelo Causal (IR v1.1)."""
from __future__ import annotations
import ast
import glob
import math
import os
import re
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
import networkx as nx
import sympy as sp
import yaml

from .schema import CausalGraph, Node, Edge, Constraint, SourceReference
from .empirical import estimate_structural_parameters_from_csv
from .fidelity import certify_fidelity
from .universal_compiler import (
    TabularASTToSymPy,
    SQLLineageTranspiler,
    NormativeDocTranspiler,
    discover_missing_mechanisms_from_csv,
    synthesize_blackbox_analytical_surrogate,
)


class ASTFunctionToSymPy(ast.NodeVisitor):
    """Compilador simbólico que traduce funciones AST de Python puro a expresiones SymPy exactas (SSA)."""

    def __init__(
        self,
        param_symbols: Optional[Dict[str, Any]] = None,
        module_constants: Optional[Dict[str, Any]] = None,
        helper_ast_map: Optional[Dict[str, Tuple[ast.FunctionDef, Dict[str, Any]]]] = None
    ):
        self.env: Dict[str, Any] = dict(param_symbols or {})
        self.module_constants: Dict[str, Any] = dict(module_constants or {})
        self.helper_ast_map = helper_ast_map or {}
        for k, v in self.module_constants.items():
            if k not in self.env:
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    self.env[k] = sp.Integer(v) if isinstance(v, int) else sp.Float(v)
                elif isinstance(v, (list, tuple, dict)):
                    self.env[k] = self._convert_static_container(v)
        self.discontinuities: List[Dict[str, Any]] = []
        self.assert_constraints: List[Dict[str, str]] = []
        self._loop_counter: int = 0

    @classmethod
    def _convert_static_container(cls, obj: Any) -> Any:
        if isinstance(obj, bool):
            return sp.Integer(1 if obj else 0)
        if isinstance(obj, int):
            return sp.Integer(obj)
        if isinstance(obj, float):
            return sp.Float(obj)
        if isinstance(obj, str):
            return obj
        if isinstance(obj, list):
            return [cls._convert_static_container(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(cls._convert_static_container(x) for x in obj)
        if isinstance(obj, dict):
            return {k: cls._convert_static_container(v) for k, v in obj.items()}
        return obj

    def compile_function(self, fn_node: ast.FunctionDef) -> sp.Expr:
        for arg in fn_node.args.args:
            if arg.arg != "self" and arg.arg not in self.env:
                self.env[arg.arg] = sp.Symbol(arg.arg, real=True)
        return self.compile_statements(fn_node.body)

    @staticmethod
    def _is_zero_number(expr: Any) -> bool:
        try:
            return bool(isinstance(expr, sp.Expr) and expr.is_number and abs(float(expr)) < 1e-15)
        except Exception:
            return False

    def _eval_comprehension(self, elt_node: ast.AST, generators: List[ast.comprehension]) -> List[Any]:
        saved_env = dict(self.env)

        def _rec(gen_idx: int, local_env: Dict[str, Any]) -> List[Any]:
            if gen_idx >= len(generators):
                self.env = local_env
                return [self.eval_expr(elt_node)]
            gen = generators[gen_idx]
            self.env = local_env
            items = self._extract_static_iterable(gen.iter)
            acc: List[Any] = []
            for item in items:
                next_env = dict(local_env)
                self._bind_target(gen.target, item, next_env)
                self.env = next_env
                cond_ok = True
                for if_clause in gen.ifs:
                    c_val, _ = self._eval_condition(if_clause)
                    if c_val == sp.false or c_val is False:
                        cond_ok = False
                        break
                if cond_ok:
                    acc.extend(_rec(gen_idx + 1, next_env))
            return acc

        try:
            return _rec(0, dict(self.env))
        finally:
            self.env = saved_env

    def _bind_target(self, target: ast.AST, val: Any, env: Dict[str, Any]) -> None:
        if isinstance(target, ast.Name):
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                env[target.id] = sp.Integer(val) if isinstance(val, int) else sp.Float(val)
            else:
                env[target.id] = val
        elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(val, (tuple, list)) and len(target.elts) == len(val):
            for t_elt, v_elt in zip(target.elts, val):
                self._bind_target(t_elt, v_elt, env)
        else:
            raise ValueError(f"Asignación de destino no soportada: {ast.dump(target)}")

    def eval_expr(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return sp.Integer(1 if node.value else 0)
            if isinstance(node.value, float):
                return sp.Float(node.value)
            if isinstance(node.value, int):
                return sp.Integer(node.value)
            if isinstance(node.value, str):
                return node.value
            raise ValueError(f"Constante no numérica: {node.value!r}")
        if isinstance(node, ast.List):
            return [self.eval_expr(e) for e in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self.eval_expr(e) for e in node.elts)
        if isinstance(node, ast.Dict):
            return {self.eval_expr(k): self.eval_expr(v) for k, v in zip(node.keys, node.values) if k is not None}
        if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
            return self._eval_comprehension(node.elt, node.generators)
        if isinstance(node, ast.Subscript):
            container = self.eval_expr(node.value)
            idx_val = self.eval_expr(node.slice)
            if isinstance(container, (list, tuple)):
                if isinstance(idx_val, (int, sp.Integer)):
                    return container[int(idx_val)]
                if isinstance(idx_val, sp.Expr) and idx_val.is_number:
                    return container[int(round(float(idx_val)))]
            elif isinstance(container, dict):
                if idx_val in container:
                    return container[idx_val]
                if isinstance(idx_val, (int, sp.Integer)) and int(idx_val) in container:
                    return container[int(idx_val)]
                if str(idx_val) in container:
                    return container[str(idx_val)]
            raise ValueError(f"Subscript fuera del subconjunto simbólico: {ast.dump(node)}")
        if isinstance(node, ast.Name):
            if node.id in self.env:
                return self.env[node.id]
            return sp.Symbol(node.id, real=True)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            if node.attr in self.env:
                return self.env[node.attr]
            return sp.Symbol(node.attr, real=True)
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
            if func_name == "sum" and len(node.args) >= 1:
                items = self.eval_expr(node.args[0])
                start_val = self.eval_expr(node.args[1]) if len(node.args) >= 2 else sp.Integer(0)
                if isinstance(items, (list, tuple)):
                    res_sum = start_val
                    for it in items:
                        res_sum = res_sum + it
                    return res_sum
            if func_name == "len" and len(node.args) == 1:
                cont = self.eval_expr(node.args[0])
                if isinstance(cont, (list, tuple, dict)):
                    return sp.Integer(len(cont))
            args = [self.eval_expr(a) for a in node.args]
            for kw in node.keywords:
                if kw.arg is not None:
                    args.append(self.eval_expr(kw.value))
            if func_name in ("float", "int") and len(args) == 1:
                return args[0]
            if func_name in ("max", "maximum") and len(args) == 1 and isinstance(args[0], (list, tuple)):
                return sp.Max(*args[0])
            if func_name in ("min", "minimum") and len(args) == 1 and isinstance(args[0], (list, tuple)):
                return sp.Min(*args[0])
            if func_name in ("max", "maximum") and len(args) >= 2:
                return sp.Max(*args)
            if func_name in ("min", "minimum") and len(args) >= 2:
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
            if func_name in ("abs", "fabs") and len(args) == 1:
                return sp.Abs(args[0])
            if func_name == "sin" and len(args) == 1:
                return sp.sin(args[0])
            if func_name == "cos" and len(args) == 1:
                return sp.cos(args[0])
            if func_name == "tanh" and len(args) == 1:
                return sp.tanh(args[0])
            if func_name == "pow" and len(args) == 2:
                return args[0] ** args[1]
            # Inlining automático de funciones auxiliares (MR2: funciones partidas)
            if func_name in self.helper_ast_map:
                h_node, h_consts = self.helper_ast_map[func_name]
                h_params = [a.arg for a in h_node.args.args if a.arg != "self"]
                if len(h_params) == len(args):
                    sub_env = {p: val for p, val in zip(h_params, args)}
                    sub_compiler = ASTFunctionToSymPy({}, h_consts, self.helper_ast_map)
                    sub_compiler.env.update(sub_env)
                    res_expr = sub_compiler.compile_statements(h_node.body, sub_compiler.env)
                    self.discontinuities.extend(sub_compiler.discontinuities)
                    self.assert_constraints.extend(sub_compiler.assert_constraints)
                    return res_expr
        if isinstance(node, ast.IfExp):
            cond, disc_info = self._eval_condition(node.test)
            if cond == sp.true:
                return self.eval_expr(node.body)
            if cond == sp.false:
                return self.eval_expr(node.orelse)
            body_e = self.eval_expr(node.body)
            orelse_e = self.eval_expr(node.orelse)
            pw_expr = sp.Piecewise((body_e, cond), (orelse_e, True))
            self._record_discontinuity(disc_info, body_e, orelse_e, pw_expr)
            return pw_expr
        raise ValueError(f"Nodo AST fuera del subconjunto simbólico: {ast.dump(node)}")

    def _record_discontinuity(
        self,
        disc_info: Optional[Dict[str, Any]],
        true_expr: sp.Expr,
        false_expr: sp.Expr,
        pw_expr: Optional[sp.Expr] = None
    ):
        if not disc_info:
            return
        th_val = disc_info.get("threshold", 0.0)
        if abs(th_val) <= 1e-9:
            return
        var_name = disc_info["variable"]
        var_sym = self.env.get(var_name, sp.Symbol(var_name, real=True))
        if not isinstance(var_sym, sp.Symbol):
            var_sym = sp.Symbol(var_name, real=True)
        try:
            target_pw = pw_expr if pw_expr is not None else sp.Piecewise((true_expr, var_sym < th_val), (false_expr, True))
            lim_below = float(target_pw.subs(var_sym, th_val - 1e-5).evalf())
            lim_above = float(target_pw.subs(var_sym, th_val + 1e-5).evalf())
            if abs(lim_above - lim_below) > 1e-7:
                entry = dict(disc_info)
                entry["val_true"] = lim_below
                entry["val_false"] = lim_above
                entry["jump_delta"] = lim_above - lim_below
                if not any(
                    d.get("variable") == var_name and abs(float(d.get("threshold", 0.0)) - float(th_val)) < 1e-7
                    for d in self.discontinuities
                ):
                    self.discontinuities.append(entry)
        except Exception:
            pass

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
        raise ValueError(f"Comparador AST no soportado: {ast.dump(op)}")

    def _eval_condition(self, test_node: ast.AST) -> Tuple[Any, Optional[Dict[str, Any]]]:
        if isinstance(test_node, ast.Constant) and isinstance(test_node.value, bool):
            return (sp.true if test_node.value else sp.false), None
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
            eval_pairs = [self._eval_condition(v) for v in test_node.values]
            sub_conds = [p[0] for p in eval_pairs]
            first_disc = next((p[1] for p in eval_pairs if p[1] is not None), None)
            if isinstance(test_node.op, ast.And):
                return sp.And(*sub_conds), first_disc
            if isinstance(test_node.op, ast.Or):
                return sp.Or(*sub_conds), first_disc
        if isinstance(test_node, ast.UnaryOp) and isinstance(test_node.op, ast.Not):
            c_sub, d_sub = self._eval_condition(test_node.operand)
            return sp.Not(c_sub), d_sub
        raise ValueError(f"Condición AST no soportada: {ast.dump(test_node)}")

    def _record_assert_constraint(self, stmt: ast.Assert, cur_env: Dict[str, Any]) -> None:
        self.env = cur_env
        msg_str = ""
        if isinstance(stmt.msg, ast.Constant) and isinstance(stmt.msg.value, str):
            msg_str = stmt.msg.value.strip()

        def _extract_cmp(test_node: ast.AST) -> None:
            if isinstance(test_node, ast.BoolOp) and isinstance(test_node.op, ast.And):
                for v in test_node.values:
                    _extract_cmp(v)
                return
            if not isinstance(test_node, ast.Compare):
                return
            operands = [self.eval_expr(test_node.left)] + [self.eval_expr(c) for c in test_node.comparators]
            for i, op in enumerate(test_node.ops):
                lhs, rhs = operands[i], operands[i + 1]
                if not (isinstance(lhs, sp.Expr) and isinstance(rhs, sp.Expr)):
                    continue
                if isinstance(op, (ast.LtE, ast.Lt)):
                    expr_s = f"{lhs} <= {rhs}"
                elif isinstance(op, (ast.GtE, ast.Gt)):
                    expr_s = f"{lhs} >= {rhs}"
                elif isinstance(op, ast.Eq):
                    expr_s = f"{lhs} == {rhs}"
                else:
                    continue
                c_id = ""
                if msg_str and re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", msg_str):
                    c_id = msg_str if len(test_node.ops) == 1 else f"{msg_str}_{i+1}"
                self.assert_constraints.append({
                    "id": c_id,
                    "expression": expr_s,
                    "description": msg_str or f"Restricción assert línea {getattr(stmt, 'lineno', 1)}",
                    "lineno": getattr(stmt, "lineno", 1),
                })

        try:
            _extract_cmp(stmt.test)
        except Exception:
            pass

    def _extract_static_iterable(self, iter_node: ast.AST) -> List[Any]:
        if isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name):
            fname = iter_node.func.id
            if fname == "range":
                args = [int(round(float(self.eval_expr(a)))) for a in iter_node.args]
                if len(args) >= 1 and abs(args[0]) <= 500 and (len(args) < 2 or abs(args[1]) <= 500):
                    return [sp.Integer(k) for k in range(*args)]
            elif fname == "enumerate" and len(iter_node.args) >= 1:
                sub_items = self._extract_static_iterable(iter_node.args[0])
                start_idx = int(round(float(self.eval_expr(iter_node.args[1])))) if len(iter_node.args) >= 2 else 0
                return [(sp.Integer(start_idx + i), it) for i, it in enumerate(sub_items)]
            elif fname == "zip" and len(iter_node.args) >= 1:
                sub_lists = [self._extract_static_iterable(a) for a in iter_node.args]
                return [tuple(items) for items in zip(*sub_lists)]
        val = self.eval_expr(iter_node)
        if isinstance(val, (list, tuple)) and len(val) <= 500:
            return list(val)
        raise ValueError("Bucle no desenrollable estáticamente")

    def compile_statements(self, stmts: List[ast.stmt], env: Optional[Dict[str, Any]] = None) -> Any:
        cur_env = self.env if env is None else env
        for idx, stmt in enumerate(stmts):
            if isinstance(stmt, (ast.Expr, ast.Pass, ast.Import, ast.ImportFrom)):
                continue
            if isinstance(stmt, ast.Assert):
                self._record_assert_constraint(stmt, cur_env)
                continue
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                self.env = cur_env
                rhs_val = self.eval_expr(stmt.value)
                self._bind_target(target, rhs_val, cur_env)
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
            elif isinstance(stmt, ast.For):
                self.env = cur_env
                loop_items = self._extract_static_iterable(stmt.iter)
                unrolled_body: List[ast.stmt] = []
                loop_uid = self._loop_counter
                self._loop_counter += 1
                for k_idx, item in enumerate(loop_items):
                    tmp_key = f"__cge_loop_{loop_uid}_{k_idx}__"
                    cur_env[tmp_key] = item
                    assign_iter = ast.Assign(
                        targets=[stmt.target],
                        value=ast.Name(id=tmp_key, ctx=ast.Load())
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
                if cond == sp.true:
                    return self.compile_statements(list(stmt.body) + continuation, dict(cur_env))
                if cond == sp.false:
                    return self.compile_statements(list(stmt.orelse) + continuation, dict(cur_env))
                true_expr = self.compile_statements(list(stmt.body) + continuation, dict(cur_env))
                false_expr = self.compile_statements(list(stmt.orelse) + continuation, dict(cur_env))
                if isinstance(true_expr, dict) and isinstance(false_expr, dict):
                    merged_dict: Dict[Any, Any] = {}
                    for dk in set(true_expr.keys()) | set(false_expr.keys()):
                        tv = true_expr.get(dk, sp.Integer(0))
                        fv = false_expr.get(dk, sp.Integer(0))
                        merged_dict[dk] = tv if tv == fv else sp.Piecewise((tv, cond), (fv, True))
                    return merged_dict
                if true_expr == false_expr:
                    return true_expr
                # Si la condición es un guardia de no-negatividad x <= 0 con valor 0, y false_expr ya es continua en 0:
                if disc_info and abs(disc_info.get("threshold", 1.0)) < 1e-12 and self._is_zero_number(true_expr):
                    return false_expr
                pw_expr = sp.Piecewise((true_expr, cond), (false_expr, True))
                self._record_discontinuity(disc_info, true_expr, false_expr, pw_expr)
                return pw_expr
            else:
                raise ValueError(f"Sentencia AST fuera del subconjunto simbólico: {type(stmt).__name__}")
        raise ValueError("La función no alcanzó una sentencia return determinista.")


def _exec_conservation_stmts(
    stmts: List[ast.stmt],
    env: Dict[str, sp.Expr],
    pos_flows: List[sp.Expr],
    neg_flows: List[sp.Expr],
    pos_methods: Set[str],
    neg_methods: Set[str]
) -> Tuple[List[Tuple[str, sp.Expr]], int, int]:
    """Ejecuta simbólicamente sentencias AST sobre un objeto con invariante de conservación (ramas if/else)."""
    compiler = ASTFunctionToSymPy({}, {})
    compiler.env = dict(env)
    p_count = 0
    n_count = 0

    for idx, stmt in enumerate(stmts):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            try:
                compiler.env[stmt.targets[0].id] = compiler.eval_expr(stmt.value)
            except Exception:
                pass
        elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
            var_name = stmt.target.id
            prev = compiler.env.get(var_name, sp.Integer(0))
            try:
                rhs = compiler.eval_expr(stmt.value)
                if isinstance(stmt.op, ast.Add):
                    compiler.env[var_name] = prev + rhs
                elif isinstance(stmt.op, ast.Sub):
                    compiler.env[var_name] = prev - rhs
                elif isinstance(stmt.op, ast.Mult):
                    compiler.env[var_name] = prev * rhs
                elif isinstance(stmt.op, ast.Div):
                    compiler.env[var_name] = prev / rhs
            except Exception:
                pass
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            attr_name = getattr(call.func, "attr", "")
            if (attr_name in pos_methods or attr_name in neg_methods) and len(call.args) >= 1:
                arg_expr = compiler.eval_expr(call.args[0])
                if attr_name in pos_methods:
                    pos_flows.append(sp.sympify(arg_expr))
                    p_count += 1
                else:
                    neg_flows.append(sp.sympify(arg_expr))
                    n_count += 1
        elif isinstance(stmt, ast.If):
            cond_str = ast.unparse(stmt.test) if hasattr(ast, "unparse") else "cond"
            rest = stmts[idx + 1:]
            t_res, tp1, tn1 = _exec_conservation_stmts(
                list(stmt.body) + rest, dict(compiler.env), list(pos_flows), list(neg_flows), pos_methods, neg_methods
            )
            f_res, tp2, tn2 = _exec_conservation_stmts(
                list(stmt.orelse) + rest, dict(compiler.env), list(pos_flows), list(neg_flows), pos_methods, neg_methods
            )
            branches = (
                [(f"{cond_str} -> {b_name}", r) for b_name, r in t_res]
                + [(f"not({cond_str}) -> {b_name}", r) for b_name, r in f_res]
            )
            return branches, p_count + tp1 + tp2, n_count + tn1 + tn2

    total_p = sum(pos_flows, sp.Integer(0))
    total_n = sum(neg_flows, sp.Integer(0))
    res = sp.simplify(total_p - total_n)
    return [("main", res)], p_count, n_count


def verify_conservation_function_symbolically(
    fn_node: ast.FunctionDef,
    rel_file: str,
    pos_methods: Set[str],
    neg_methods: Set[str],
    constraint_id: str = "c_ledger"
) -> Dict[str, Any]:
    """Verifica simbólicamente si una función conserva el invariante de suma cero en todas sus ramas."""
    param_syms: Dict[str, sp.Expr] = {}
    for arg in fn_node.args.args:
        param_syms[arg.arg] = sp.Symbol(arg.arg, positive=True, real=True)

    pos_list: List[sp.Expr] = []
    neg_list: List[sp.Expr] = []
    branch_residuals, p_calls, n_calls = _exec_conservation_stmts(
        fn_node.body, param_syms, pos_list, neg_list, pos_methods, neg_methods
    )

    all_balanced = all((r == 0 or sp.simplify(r) == 0) for _, r in branch_residuals)
    primary_res = branch_residuals[0][1] if len(branch_residuals) == 1 else next(
        (r for _, r in branch_residuals if r != 0), sp.Integer(0)
    )

    return {
        "function": fn_node.name,
        "file": rel_file,
        "constraint_id": constraint_id,
        "positive_calls": p_calls,
        "negative_calls": n_calls,
        "symbolic_residual": str(primary_res),
        "branch_residuals": {b: str(r) for b, r in branch_residuals},
        "is_balanced": bool(all_balanced),
        "line_range": [fn_node.lineno, getattr(fn_node, "end_lineno", fn_node.lineno)]
    }


class ASTIngestPipeline:
    """Extractor automático estático-simbólico que compila cualquier repositorio compatible a CausalIR (schema.json)
    con cero nombres de dominio cableados.
    """

    _STOCHASTIC_FUNCS = {"gauss", "normal", "normalvariate", "uniform", "lognormvariate", "triangular"}
    _VERB_PREFIXES = ("calculate_", "compute_", "calc_", "eval_", "derive_", "estimate_", "predict_", "get_stochastic_", "sample_", "draw_", "sim_")
    _POS_LEDGER_METHODS = {"record_credit", "add_credit", "credit", "record_inflow", "inflow", "deposit"}
    _NEG_LEDGER_METHODS = {"record_debit", "add_debit", "debit", "record_outflow", "outflow", "withdraw"}

    def __init__(self, target_dir: str):
        self.target_dir = os.path.normpath(target_dir)
        self.graph = CausalGraph(system_id="cge_ingested_system", version="1.1.0")
        self._symbols: Dict[str, sp.Symbol] = {}
        self._mechanisms: Dict[str, sp.Expr] = {}
        self._blackbox_callables: Dict[str, Tuple[List[str], Callable[..., float]]] = {}
        self._discontinuities: Dict[str, List[Dict[str, Any]]] = {}

    def _get_sym(self, name: str) -> sp.Symbol:
        if name not in self._symbols:
            self._symbols[name] = sp.Symbol(name, real=True)
        return self._symbols[name]

    def _get_relative_path(self, full_path: str) -> str:
        rel = os.path.relpath(full_path, self.target_dir)
        base_name = os.path.basename(self.target_dir)
        return f"{base_name}/{rel}".replace("\\", "/")

    @staticmethod
    def _humanize_label(node_id: str, category: str, is_dead: bool = False) -> str:
        words = [w.capitalize() for w in node_id.replace("-", "_").split("_") if w]
        base = " ".join(words)
        if category == "target":
            return f"{base} (Target J)"
        if is_dead:
            return f"{base} (Nodo Muerto T1)"
        return base

    @staticmethod
    def _strip_verb_prefix(fn_name: str) -> str:
        for pref in ("calculate_", "compute_", "calc_", "eval_", "derive_", "estimate_", "predict_", "sample_", "draw_", "sim_", "get_nominal_", "get_"):
            if fn_name.startswith(pref) and len(fn_name) > len(pref):
                return fn_name[len(pref):]
        return fn_name

    def _eval_ast_const_expr(self, node: ast.AST, consts: Dict[str, float]) -> Optional[float]:
        try:
            val = ast.literal_eval(node)
            if isinstance(val, (int, float)):
                return float(val)
        except Exception:
            pass
        if isinstance(node, ast.Name) and node.id in consts:
            return float(consts[node.id])
        try:
            comp = ASTFunctionToSymPy({}, consts)
            expr = comp.eval_expr(node)
            if expr.is_number:
                return float(expr.evalf())
        except Exception:
            pass
        return None

    def run(self) -> CausalGraph:
        py_files = sorted(
            p for p in glob.glob(os.path.join(self.target_dir, "**", "*.py"), recursive=True)
            if os.path.basename(p) != "__init__.py" and "tests" not in os.path.relpath(p, self.target_dir).replace("\\", "/").split("/")
        )

        # 0. Cargar especificación declarativa opcional cge.yaml / cge.yml si existe
        cge_spec: Optional[Dict[str, Any]] = None
        yaml_path_rel = "cge.yaml"
        for y_name in ("cge.yaml", "cge.yml"):
            y_cand = os.path.join(self.target_dir, y_name)
            if os.path.exists(y_cand):
                yaml_path_rel = y_name
                try:
                    with open(y_cand, "r", encoding="utf-8") as yf:
                        loaded = yaml.safe_load(yf)
                        if isinstance(loaded, dict):
                            cge_spec = loaded
                            break
                except Exception:
                    pass

        yaml_controls: Dict[str, Dict[str, Any]] = {}
        if cge_spec and "controls" in cge_spec:
            raw_ctrls = cge_spec["controls"]
            if isinstance(raw_ctrls, dict):
                for cid, cinfo in raw_ctrls.items():
                    yaml_controls[cid] = cinfo if isinstance(cinfo, dict) else {}
            elif isinstance(raw_ctrls, list):
                for item in raw_ctrls:
                    if isinstance(item, dict) and "id" in item:
                        yaml_controls[item["id"]] = item

        yaml_exogenous: Dict[str, Dict[str, Any]] = {}
        if cge_spec and "exogenous" in cge_spec:
            raw_exo = cge_spec["exogenous"]
            if isinstance(raw_exo, dict):
                for uid, uinfo in raw_exo.items():
                    yaml_exogenous[uid] = uinfo if isinstance(uinfo, dict) else {}
            elif isinstance(raw_exo, list):
                for item in raw_exo:
                    if isinstance(item, dict) and "id" in item:
                        yaml_exogenous[item["id"]] = item

        # 1. Parsear AST de todos los módulos y recolectar constantes globales, clases y funciones
        raw_modules: List[Tuple[str, str, ast.Module, Dict[str, Any], str]] = []
        global_mod_consts: Dict[str, Any] = {}
        all_constants: Dict[str, Dict[str, Any]] = {}
        bounds_map: Dict[str, Tuple[float, float]] = {}
        ceiling_constants: Dict[str, float] = {}
        has_conservation_class = False
        inferred_sense = str((cge_spec or {}).get("sense", "max")).lower()
        module_declared_constraints: List[Dict[str, str]] = []

        if cge_spec:
            for b_key in ("max_budget", "ceiling", "budget_limit", "MAX_BUDGET"):
                if b_key in cge_spec and isinstance(cge_spec[b_key], (int, float)):
                    ceiling_constants["MAX_BUDGET"] = float(cge_spec[b_key])

        for py_path in py_files:
            rel_file = self._get_relative_path(py_path)
            with open(py_path, "r", encoding="utf-8") as f:
                src_code = f.read()
            tree = ast.parse(src_code, filename=py_path)
            mod_consts: Dict[str, Any] = {}

            for stmt in tree.body:
                t_name = None
                val_node = None
                ann_type = None
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    t_name = stmt.targets[0].id
                    val_node = stmt.value
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                    t_name = stmt.target.id
                    val_node = stmt.value
                    if isinstance(stmt.annotation, ast.Name):
                        ann_type = stmt.annotation.id

                if t_name and val_node is not None:
                    try:
                        lit = ast.literal_eval(val_node)
                    except Exception:
                        num_consts = {k: v for k, v in {**global_mod_consts, **mod_consts}.items() if isinstance(v, (int, float))}
                        lit = self._eval_ast_const_expr(val_node, num_consts)
                    if t_name in ("SENSE", "OBJECTIVE_SENSE") and isinstance(lit, str):
                        inferred_sense = "min" if lit.strip().lower().startswith("min") else "max"
                    elif t_name == "CONSTRAINTS" and isinstance(lit, (dict, list)):
                        if isinstance(lit, dict):
                            for cid_k, cexpr_v in lit.items():
                                if isinstance(cexpr_v, str):
                                    module_declared_constraints.append({"id": str(cid_k), "expression": cexpr_v, "description": str(cid_k)})
                        elif isinstance(lit, list):
                            for idx_c, item_c in enumerate(lit):
                                if isinstance(item_c, str):
                                    module_declared_constraints.append({"id": f"c_decl_{idx_c+1}", "expression": item_c, "description": item_c})
                                elif isinstance(item_c, dict) and "expression" in item_c:
                                    module_declared_constraints.append({
                                        "id": str(item_c.get("id", f"c_decl_{idx_c+1}")),
                                        "expression": str(item_c["expression"]),
                                        "description": str(item_c.get("description", item_c["expression"])),
                                    })
                    elif isinstance(lit, (tuple, list)) and len(lit) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in lit) and any(t_name.endswith(s) for s in ("_BOUNDS", "_LIMITS", "_RANGE")):
                        for suf in ("_BOUNDS", "_LIMITS", "_RANGE"):
                            if t_name.endswith(suf):
                                b_key = t_name[:-len(suf)].lower()
                                bounds_map[b_key] = (float(lit[0]), float(lit[1]))
                    elif isinstance(lit, (int, float)) and not isinstance(lit, bool):
                        mod_consts[t_name] = float(lit)
                        global_mod_consts[t_name] = float(lit)
                        lines = [stmt.lineno, getattr(stmt, "end_lineno", stmt.lineno)]
                        if t_name.startswith("MAX_") or t_name.endswith("_LIMIT"):
                            ceiling_constants[t_name] = float(lit)
                        else:
                            all_constants[t_name] = {
                                "name": t_name,
                                "value": lit,
                                "ann_type": ann_type,
                                "file": rel_file,
                                "line_range": lines
                            }
                    elif isinstance(lit, (list, tuple, dict)):
                        mod_consts[t_name] = lit
                        global_mod_consts[t_name] = lit
                elif isinstance(stmt, ast.ClassDef):
                    m_names = {s.name for s in stmt.body if isinstance(s, ast.FunctionDef)}
                    if (m_names & self._POS_LEDGER_METHODS and m_names & self._NEG_LEDGER_METHODS) or "is_balanced" in m_names:
                        has_conservation_class = True

            raw_modules.append((py_path, rel_file, tree, mod_consts, src_code))

        # Emparejar constantes escalares <NAME>_MIN y <NAME>_MAX en bounds_map (Regla D1.2)
        for k_min, v_min in global_mod_consts.items():
            if k_min.endswith("_MIN") and isinstance(v_min, (int, float)):
                prefix = k_min[:-4]
                k_max = f"{prefix}_MAX"
                if k_max in global_mod_consts and isinstance(global_mod_consts[k_max], (int, float)) and prefix.lower() not in bounds_map:
                    bounds_map[prefix.lower()] = (float(v_min), float(global_mod_consts[k_max]))

        # Fusionar constantes globales para soportar imports de constantes entre módulos hermanos
        parsed_modules: List[Tuple[str, str, ast.Module, Dict[str, Any]]] = [
            (py_path, rel_file, tree, {**global_mod_consts, **mod_consts})
            for py_path, rel_file, tree, mod_consts, _ in raw_modules
        ]

        # 2. Recolectar todas las definiciones de funciones (y métodos de clases de simulación)
        all_functions: List[Tuple[str, str, ast.FunctionDef, Dict[str, float], str]] = []
        for (py_path, rel_file, tree, mod_consts), (_, _, _, _, src_code) in zip(parsed_modules, raw_modules):
            for stmt in tree.body:
                if isinstance(stmt, ast.FunctionDef):
                    all_functions.append((py_path, rel_file, stmt, mod_consts, src_code))
                elif isinstance(stmt, ast.ClassDef):
                    m_names = {s.name for s in stmt.body if isinstance(s, ast.FunctionDef)}
                    is_ledger_cls = (m_names & self._POS_LEDGER_METHODS and m_names & self._NEG_LEDGER_METHODS) or "is_balanced" in m_names
                    if not is_ledger_cls:
                        for sub in stmt.body:
                            if isinstance(sub, ast.FunctionDef) and not sub.name.startswith("__"):
                                all_functions.append((py_path, rel_file, sub, mod_consts, src_code))

        # 3. Identificar funciones orquestadoras / pipelines para mapeo de nombres de salida y parámetros
        pipeline_call_to_var: Dict[str, str] = {}
        pipeline_fn_names: Set[str] = set()
        all_param_names: Set[str] = set()

        for _, _, fn_node, _, _ in all_functions:
            for a in fn_node.args.args:
                if a.arg != "self":
                    all_param_names.add(a.arg)
            if TabularASTToSymPy.is_tabular_function(fn_node):
                pipeline_fn_names.add(fn_node.name)
            returns_dict = any(isinstance(n, ast.Return) and isinstance(n.value, ast.Dict) for n in ast.walk(fn_node))
            call_assigns = []
            seq_assigns = []
            for st in fn_node.body:
                if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                    seq_assigns.append(st.targets[0].id)
                    if isinstance(st.value, ast.Call):
                        c_name = getattr(st.value.func, "id", getattr(st.value.func, "attr", ""))
                        if c_name not in ("min", "max", "sqrt", "log", "log1p", "exp", "abs", "float", "int", "Random"):
                            call_assigns.append((c_name, st.targets[0].id))
            if returns_dict or len(call_assigns) >= 4 or len(seq_assigns) >= 5:
                pipeline_fn_names.add(fn_node.name)
                for c_name, v_name in call_assigns:
                    pipeline_call_to_var[c_name] = v_name

        # Recolectar nombres de columnas referenciadas en scripts tabulares DataFrame y consultas SQL
        for _, _, tree, _ in parsed_modules:
            for sub in ast.walk(tree):
                if isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Constant) and isinstance(sub.slice.value, str):
                    all_param_names.add(sub.slice.value)
        sql_files_all = sorted(glob.glob(os.path.join(self.target_dir, "**", "*.sql"), recursive=True))
        for sql_f in sql_files_all:
            try:
                with open(sql_f, "r", encoding="utf-8") as sf:
                    sql_content = sf.read()
                for tok in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", sql_content):
                    all_param_names.add(tok.lower())
            except Exception:
                pass

        # 4. Identificar nodos Exógenos U (desde cge.yaml o funciones con generadores estocásticos) y funciones de conservación (T2)
        exogenous_fn_names: Set[str] = set()
        getter_fn_names: Set[str] = set()
        conservation_fn_names: Set[str] = set()
        conservation_checks: List[Dict[str, Any]] = []

        if yaml_exogenous:
            for uid, uinfo in yaml_exogenous.items():
                dist_spec = uinfo.get("distribution")
                if isinstance(dist_spec, dict):
                    mean_val = float(dist_spec.get("mean", uinfo.get("mean", 1000.0)))
                    std_val = float(dist_spec.get("std", uinfo.get("std", 100.0)))
                    family = dist_spec.get("family", "normal")
                else:
                    mean_val = float(uinfo.get("mean", 1000.0))
                    std_val = float(uinfo.get("std", 100.0))
                    family = uinfo.get("family", "normal")
                dist = {"family": family, "mean": mean_val, "std": std_val}
                raw_b = uinfo.get("bounds", [0.0, None])
                b_min = raw_b[0] if isinstance(raw_b, (list, tuple)) and len(raw_b) >= 1 else (uinfo.get("bounds", {}).get("min", 0.0) if isinstance(uinfo.get("bounds"), dict) else 0.0)
                b_max = raw_b[1] if isinstance(raw_b, (list, tuple)) and len(raw_b) >= 2 else (uinfo.get("bounds", {}).get("max", None) if isinstance(uinfo.get("bounds"), dict) else None)
                init_val = float(uinfo.get("initial", uinfo.get("current_value", global_mod_consts.get(uid, global_mod_consts.get(uid.upper(), global_mod_consts.get(f"{uid.upper()}_MEAN", mean_val))))))
                self._get_sym(uid)
                self.graph.add_node(Node(
                    id=uid,
                    label=self._humanize_label(uid, "exogenous"),
                    category="exogenous",
                    data_type="continuous",
                    source_reference=SourceReference(
                        file=yaml_path_rel,
                        entity=uid,
                        line_range=[1, 1]
                    ),
                    current_value=init_val,
                    bounds={"min": b_min, "max": b_max},
                    distribution=dist,
                    characteristic_scale=max(mean_val, 1.0)
                ))

        for _, rel_file, fn_node, mod_consts, _ in all_functions:
            if fn_node.name in pipeline_fn_names:
                continue

            # Verificar si es una función de transacción sobre clase con invariante de conservación (T2)
            calls_in_fn = [n for n in ast.walk(fn_node) if isinstance(n, ast.Call)]
            called_attrs = {getattr(c.func, "attr", "") for c in calls_in_fn}
            if (called_attrs & self._POS_LEDGER_METHODS) or (called_attrs & self._NEG_LEDGER_METHODS):
                conservation_fn_names.add(fn_node.name)
                check_res = verify_conservation_function_symbolically(
                    fn_node, rel_file, self._POS_LEDGER_METHODS, self._NEG_LEDGER_METHODS, "c_ledger"
                )
                conservation_checks.append(check_res)
                continue

            if yaml_exogenous:
                continue

            # Verificar si contiene llamada a generador estocástico (U)
            stoch_call = None
            for c in calls_in_fn:
                c_attr = getattr(c.func, "attr", getattr(c.func, "id", ""))
                if c_attr in self._STOCHASTIC_FUNCS:
                    stoch_call = (c_attr, c)
                    break

            if stoch_call is not None or fn_node.name.startswith("sample_"):
                exogenous_fn_names.add(fn_node.name)
                u_id = pipeline_call_to_var.get(fn_node.name) or self._strip_verb_prefix(fn_node.name)
                mean_val = float(mod_consts.get(f"{u_id.upper()}_MEAN", 1000.0))
                std_val = float(mod_consts.get(f"{u_id.upper()}_STD", 100.0))
                s_type = "gauss"
                if stoch_call is not None:
                    s_type, s_node = stoch_call
                    if len(s_node.args) >= 2:
                        v0 = self._eval_ast_const_expr(s_node.args[0], mod_consts)
                        v1 = self._eval_ast_const_expr(s_node.args[1], mod_consts)
                        if v0 is not None and v1 is not None:
                            if s_type == "uniform":
                                mean_val = 0.5 * (v0 + v1)
                                std_val = (v1 - v0) / math.sqrt(12.0)
                            else:
                                mean_val = v0
                                std_val = v1
                min_bound: Optional[float] = 0.0
                self._get_sym(u_id)
                self.graph.add_node(Node(
                    id=u_id,
                    label=self._humanize_label(u_id, "exogenous"),
                    category="exogenous",
                    data_type="continuous",
                    source_reference=SourceReference(
                        file=rel_file,
                        entity=fn_node.name,
                        line_range=[fn_node.lineno, getattr(fn_node, "end_lineno", fn_node.lineno)]
                    ),
                    current_value=float(mean_val),
                    bounds={"min": min_bound, "max": None},
                    distribution={"family": "normal" if s_type != "uniform" else "uniform", "mean": float(mean_val), "std": float(std_val)},
                    characteristic_scale=max(float(mean_val), 1.0)
                ))

        # D1.1 Condición 2: Parámetros de entrada raíz asociados a constantes _MEAN / _STD / _VAR (solo si no provistos en cge.yaml)
        if not yaml_exogenous:
            for p_name in sorted(all_param_names):
                if p_name in self.graph.nodes:
                    continue
                mean_key = f"{p_name.upper()}_MEAN"
                std_key = f"{p_name.upper()}_STD"
                var_key = f"{p_name.upper()}_VAR"
                if mean_key in all_constants or std_key in all_constants or var_key in all_constants:
                    m_info = all_constants.get(mean_key) or all_constants.get(std_key) or all_constants.get(var_key)
                    mean_val = float(all_constants[mean_key]["value"]) if mean_key in all_constants else 1000.0
                    if std_key in all_constants:
                        std_val = float(all_constants[std_key]["value"])
                    elif var_key in all_constants:
                        std_val = math.sqrt(max(0.0, float(all_constants[var_key]["value"])))
                    else:
                        std_val = 100.0
                    self._get_sym(p_name)
                    self.graph.add_node(Node(
                        id=p_name,
                        label=self._humanize_label(p_name, "exogenous"),
                        category="exogenous",
                        data_type="continuous",
                        source_reference=SourceReference(
                            file=m_info["file"],
                            entity=mean_key if mean_key in all_constants else p_name.upper(),
                            line_range=m_info["line_range"]
                        ),
                        current_value=float(mean_val),
                        bounds={"min": 0.0, "max": None},
                        distribution={"family": "normal", "mean": float(mean_val), "std": float(std_val)},
                        characteristic_scale=max(float(mean_val), 1.0)
                    ))

        # Identificar getters nominales de U (ej. get_nominal_*)
        for _, _, fn_node, _, _ in all_functions:
            stripped = self._strip_verb_prefix(fn_node.name)
            if stripped in self.graph.nodes or fn_node.name.startswith("get_nominal_"):
                getter_fn_names.add(fn_node.name)

        # 5. Identificar Palancas de Control X desde cge.yaml o desde constantes globales con cotas
        if yaml_controls:
            for cid, cinfo in yaml_controls.items():
                c_type = cinfo.get("type", "continuous")
                is_int = (c_type == "discrete")
                raw_b = cinfo.get("bounds", [0, 1000])
                b_min = raw_b[0] if isinstance(raw_b, (list, tuple)) and len(raw_b) >= 1 else (cinfo.get("bounds", {}).get("min", 0.0) if isinstance(cinfo.get("bounds"), dict) else 0.0)
                b_max = raw_b[1] if isinstance(raw_b, (list, tuple)) and len(raw_b) >= 2 else (cinfo.get("bounds", {}).get("max", 1000.0) if isinstance(cinfo.get("bounds"), dict) else 1000.0)
                init_val = cinfo.get("initial", cinfo.get("current_value", cinfo.get("value", global_mod_consts.get(cid, global_mod_consts.get(cid.upper(), global_mod_consts.get(cid.lower(), b_min))))))
                scale = float(b_max - b_min) if (b_max is not None and b_min is not None and b_max > b_min) else max(abs(float(init_val)), 1.0)
                b_dict = {
                    "min": int(b_min) if is_int and b_min is not None else (float(b_min) if b_min is not None else None),
                    "max": int(b_max) if is_int and b_max is not None else (float(b_max) if b_max is not None else None)
                }
                self._get_sym(cid)
                self.graph.add_node(Node(
                    id=cid,
                    label=self._humanize_label(cid, "control"),
                    category="control",
                    data_type="discrete" if is_int else "continuous",
                    source_reference=SourceReference(
                        file=yaml_path_rel,
                        entity=cid,
                        line_range=[1, 1]
                    ),
                    current_value=int(init_val) if is_int else float(init_val),
                    bounds=b_dict,
                    characteristic_scale=scale
                ))
        else:
            for c_name, c_info in all_constants.items():
                if c_name.endswith(("_MEAN", "_STD", "_VAR", "_MIN", "_MAX", "_COEFF", "_THRESHOLD")):
                    continue
                cand_id = c_name.lower()
                if cand_id in self.graph.nodes:
                    continue

                # Buscar cota asociada en bounds_map por coincidencia exacta o prefijo
                b_tuple: Optional[Tuple[float, float]] = bounds_map.get(cand_id)
                if b_tuple is None:
                    for b_key, b_val in bounds_map.items():
                        if cand_id.startswith(b_key + "_") or b_key.startswith(cand_id + "_"):
                            b_tuple = b_val
                            break

                if b_tuple is None and c_name.endswith("_RATE"):
                    continue

                if b_tuple is not None or cand_id in all_param_names:
                    val = c_info["value"]
                    is_int = (c_info["ann_type"] == "int") or isinstance(val, int)
                    b_dict = (
                        {"min": int(b_tuple[0]) if is_int else float(b_tuple[0]), "max": int(b_tuple[1]) if is_int else float(b_tuple[1])}
                        if b_tuple is not None
                        else {"min": None, "max": None}
                    )
                    scale = float(b_tuple[1] - b_tuple[0]) if (b_tuple is not None and b_tuple[1] > b_tuple[0]) else max(abs(float(val)), 1.0)
                    self._get_sym(cand_id)
                    self.graph.add_node(Node(
                        id=cand_id,
                        label=self._humanize_label(cand_id, "control"),
                        category="control",
                        data_type="discrete" if is_int else "continuous",
                        source_reference=SourceReference(
                            file=c_info["file"],
                            entity=c_name,
                            line_range=c_info["line_range"]
                        ),
                        current_value=int(val) if is_int else float(val),
                        bounds=b_dict,
                        characteristic_scale=scale
                    ))

            # Soportar palancas definidas mediante <NAME>_BOUNDS + parámetros de función con valor por defecto (sin constante <NAME> separada)
            param_defaults_map: Dict[str, Tuple[Any, Optional[str], str, List[int]]] = {}
            for _, rel_file, fn_node, mod_consts, _ in all_functions:
                non_self_args = [a for a in fn_node.args.args if a.arg != "self"]
                defs = list(fn_node.args.defaults or [])
                offset = len(non_self_args) - len(defs)
                num_consts = {k: v for k, v in mod_consts.items() if isinstance(v, (int, float))}
                for idx_a, arg_obj in enumerate(non_self_args):
                    ann_s = arg_obj.annotation.id if isinstance(arg_obj.annotation, ast.Name) else None
                    def_val: Any = None
                    if idx_a >= offset:
                        d_node = defs[idx_a - offset]
                        try:
                            def_val = ast.literal_eval(d_node)
                        except Exception:
                            def_val = self._eval_ast_const_expr(d_node, num_consts)
                    if arg_obj.arg not in param_defaults_map or def_val is not None:
                        param_defaults_map[arg_obj.arg] = (
                            def_val,
                            ann_s,
                            rel_file,
                            [fn_node.lineno, getattr(fn_node, "end_lineno", fn_node.lineno)],
                        )

            for b_key, b_tuple in bounds_map.items():
                if b_key in self.graph.nodes:
                    continue
                if b_key in param_defaults_map or b_key in all_param_names:
                    def_val, ann_s, rel_file_p, l_range = param_defaults_map.get(
                        b_key, (None, None, parsed_modules[0][1] if parsed_modules else "model.py", [1, 1])
                    )
                    is_int = (ann_s == "int") or (isinstance(def_val, int) and not isinstance(def_val, bool))
                    if def_val is None or not isinstance(def_val, (int, float)):
                        mid = 0.5 * (b_tuple[0] + b_tuple[1])
                        def_val = int(round(mid)) if is_int else float(mid)
                    b_dict = {
                        "min": int(round(b_tuple[0])) if is_int else float(b_tuple[0]),
                        "max": int(round(b_tuple[1])) if is_int else float(b_tuple[1]),
                    }
                    scale = float(b_tuple[1] - b_tuple[0]) if b_tuple[1] > b_tuple[0] else max(abs(float(def_val)), 1.0)
                    self._get_sym(b_key)
                    self.graph.add_node(Node(
                        id=b_key,
                        label=self._humanize_label(b_key, "control"),
                        category="control",
                        data_type="discrete" if is_int else "continuous",
                        source_reference=SourceReference(
                            file=rel_file_p,
                            entity=b_key.upper(),
                            line_range=l_range,
                        ),
                        current_value=int(round(def_val)) if is_int else float(def_val),
                        bounds=b_dict,
                        characteristic_scale=scale,
                    ))

        # 6. Clasificar el resto de funciones entre Optimizadores Locales (T4), Auxiliares (Helpers) y Mecanismos (V / J)
        candidate_fn_list = [
            item for item in all_functions
            if item[2].name not in pipeline_fn_names
            and item[2].name not in exogenous_fn_names
            and item[2].name not in getter_fn_names
            and item[2].name not in conservation_fn_names
        ]

        # Mapa de funciones candidatas para inlining si una función es auxiliar interna (incluyendo entre módulos)
        helper_ast_map: Dict[str, Tuple[ast.FunctionDef, Dict[str, Any]]] = {
            fn_node.name: (fn_node, mod_consts)
            for _, _, fn_node, mod_consts, _ in candidate_fn_list
        }

        # Espacio de nombres compartido para ejecución aislada de cajas negras (D3.2) con soporte de helpers entre módulos
        import random as _py_random
        project_exec_ns: Dict[str, Any] = {"math": math, "random": _py_random}
        project_exec_ns.update(global_mod_consts)
        for py_path, _, fn_node, _, _ in candidate_fn_list:
            try:
                exec(compile(ast.Module(body=[fn_node], type_ignores=[]), filename=py_path, mode="exec"), project_exec_ns)
            except Exception:
                pass

        # Identificar cuáles funciones son llamadas únicamente dentro de otras funciones de mecanismo (helpers internos)
        called_by_other_mechanisms: Set[str] = set()
        for _, _, fn_node, _, _ in candidate_fn_list:
            for sub in ast.walk(fn_node):
                if isinstance(sub, ast.Call):
                    callee = getattr(sub.func, "id", getattr(sub.func, "attr", ""))
                    if callee in helper_ast_map and callee != fn_node.name:
                        called_by_other_mechanisms.add(callee)

        local_optimizers: Dict[str, Dict[str, Any]] = {}
        mechanism_meta: Dict[str, Dict[str, Any]] = {}
        extracted_assert_constraints: List[Dict[str, str]] = []
        control_ids = [nid for nid, n in self.graph.nodes.items() if n.category == "control"]

        for py_path, rel_file, fn_node, mod_consts, src_code in candidate_fn_list:
            # Si es una función tabular/DataFrame, se procesa a nivel de columnas en el Paso 6B
            if TabularASTToSymPy.is_tabular_function(fn_node):
                continue
            # Si es un helper privado o llamado por otro mecanismo y no llamado directamente por el pipeline:
            if fn_node.name in called_by_other_mechanisms and fn_node.name not in pipeline_call_to_var:
                continue

            fn_params = [a.arg for a in fn_node.args.args if a.arg != "self"]
            # Detectar Optimizador Local (T4): sin argumentos del grafo causal y devuelve constante escalar
            if not any(p in self.graph.nodes or p in all_param_names for p in fn_params) or len(fn_params) == 0 or (
                len(fn_params) == 1 and fn_node.args.defaults and fn_params[0] not in self.graph.nodes
            ):
                try:
                    comp_loc = ASTFunctionToSymPy({}, mod_consts, helper_ast_map)
                    expr_loc = comp_loc.compile_statements(fn_node.body)
                    if isinstance(expr_loc, sp.Expr) and expr_loc.is_number:
                        rec_val = float(expr_loc.evalf())
                        # Emparejar con la palanca de control correspondiente por tokens y tipo/rango
                        local_tokens = set(fn_node.name.lower().split("_"))
                        for sub in ast.walk(fn_node):
                            if isinstance(sub, ast.Name):
                                local_tokens.update(sub.id.lower().split("_"))
                        best_ctrl = None
                        best_score = -1
                        for cid in control_ids:
                            c_node = self.graph.nodes[cid]
                            c_tokens = set(cid.lower().split("_"))
                            score = 10 * len(local_tokens & c_tokens)
                            if c_node.data_type == "discrete" and float(rec_val).is_integer():
                                score += 3
                            b_min = c_node.bounds.get("min")
                            b_max = c_node.bounds.get("max")
                            if b_min is not None and b_max is not None and b_min <= rec_val <= b_max:
                                score += 2
                            if score > best_score:
                                best_score = score
                                best_ctrl = cid
                        if best_ctrl is not None:
                            is_disc = self.graph.nodes[best_ctrl].data_type == "discrete"
                            local_optimizers[fn_node.name] = {
                                "function": fn_node.name,
                                "file": rel_file,
                                "line_range": [fn_node.lineno, getattr(fn_node, "end_lineno", fn_node.lineno)],
                                "control_id": best_ctrl,
                                "recommended_value": int(round(rec_val)) if is_disc else float(rec_val)
                            }
                            continue
                except Exception:
                    pass

            # Es una función de mecanismo (V o J)
            node_id = pipeline_call_to_var.get(fn_node.name) or self._strip_verb_prefix(fn_node.name)
            param_syms = {p: self._get_sym(p) for p in fn_params}
            self._get_sym(node_id)

            try:
                compiler = ASTFunctionToSymPy(param_syms, mod_consts, helper_ast_map)
                sym_expr = compiler.compile_statements(fn_node.body)
                if not isinstance(sym_expr, sp.Expr):
                    raise ValueError("La función no devolvió una expresión simbólica escalar.")
                self._mechanisms[node_id] = sym_expr
                if compiler.discontinuities:
                    self._discontinuities[node_id] = compiler.discontinuities
                if compiler.assert_constraints:
                    extracted_assert_constraints.extend(compiler.assert_constraints)
                engine_type = "sympy_expr"
                conf = 1.0
                expr_str = str(sym_expr)
            except Exception:
                # Fallback de Caja Negra (D3.2): ejecución aislada y diferencias finitas
                ns: Dict[str, Any] = dict(project_exec_ns)
                ns.update(mod_consts)
                exec(compile(ast.Module(body=[fn_node], type_ignores=[]), filename=py_path, mode="exec"), ns)
                py_callable = ns[fn_node.name]
                self._blackbox_callables[node_id] = (fn_params, py_callable)
                engine_type = "python_callable"
                conf = 0.8
                expr_str = f"blackbox:{fn_node.name}({', '.join(fn_params)})"

            mechanism_meta[node_id] = {
                "id": node_id,
                "params": fn_params,
                "file": rel_file,
                "entity": fn_node.name,
                "line_range": [fn_node.lineno, getattr(fn_node, "end_lineno", fn_node.lineno)],
                "engine": engine_type,
                "confidence": conf,
                "symbolic_expr": expr_str
            }

        # Soporte para funciones monolíticas o fusionadas en el pipeline (MR2: funciones juntadas en una)
        if pipeline_fn_names:
            for py_path, rel_file, fn_node, mod_consts, _ in all_functions:
                if fn_node.name not in pipeline_fn_names:
                    continue
                if TabularASTToSymPy.is_tabular_function(fn_node):
                    continue
                fn_params = [a.arg for a in fn_node.args.args if a.arg != "self"]
                param_syms = {p: self._get_sym(p) for p in fn_params}
                comp = ASTFunctionToSymPy(param_syms, mod_consts, helper_ast_map)
                added_from_this_fn = 0
                had_stmt_failure = False

                for idx_st, st in enumerate(fn_node.body):
                    if isinstance(st, ast.Assert):
                        comp._record_assert_constraint(st, comp.env)
                        continue
                    if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                        v_id = st.targets[0].id
                        # Si es una llamada directa a una función ya extraída como nodo, mantener el símbolo sin re-extraer
                        if isinstance(st.value, ast.Call):
                            c_name = getattr(st.value.func, "id", getattr(st.value.func, "attr", ""))
                            if c_name in pipeline_call_to_var or c_name in exogenous_fn_names or c_name in getter_fn_names:
                                comp.env[v_id] = self._get_sym(v_id)
                                continue
                        try:
                            prev_disc_len = len(comp.discontinuities)
                            expr_v = comp.eval_expr(st.value)
                            new_discs = comp.discontinuities[prev_disc_len:]
                            if not isinstance(expr_v, sp.Expr) or (expr_v.is_number and len(expr_v.free_symbols) == 0):
                                comp.env[v_id] = expr_v
                            elif v_id.startswith("_") or v_id.endswith(("_factor", "_lift", "_boost", "_mult")):
                                comp.env[v_id] = expr_v
                            elif v_id not in self.graph.nodes and v_id not in mechanism_meta:
                                self._get_sym(v_id)
                                self._mechanisms[v_id] = expr_v
                                if new_discs:
                                    self._discontinuities[v_id] = list(new_discs)
                                used_p = sorted([str(s) for s in expr_v.free_symbols if str(s) != v_id])
                                mechanism_meta[v_id] = {
                                    "id": v_id,
                                    "params": used_p,
                                    "file": rel_file,
                                    "entity": fn_node.name,
                                    "line_range": [st.lineno, getattr(st, "end_lineno", st.lineno)],
                                    "engine": "sympy_expr",
                                    "confidence": 1.0,
                                    "symbolic_expr": str(expr_v)
                                }
                                added_from_this_fn += 1
                                # Vincular v_id a su propio símbolo para preservar las aristas causales hacia nodos hijos
                                comp.env[v_id] = self._get_sym(v_id)
                        except Exception:
                            had_stmt_failure = True
                    elif isinstance(st, (ast.AugAssign, ast.For, ast.If)):
                        # Si hay bucles, acumuladores o bloques if/else intermedios, compilar el sufijo restante
                        try:
                            ret_val = comp.compile_statements(fn_node.body[idx_st:], comp.env)
                            if isinstance(ret_val, dict):
                                for dk, dv in ret_val.items():
                                    dk_str = str(dk)
                                    if isinstance(dv, sp.Expr) and len(dv.free_symbols) > 0 and dk_str not in self.graph.nodes and dk_str not in mechanism_meta:
                                        self._get_sym(dk_str)
                                        self._mechanisms[dk_str] = dv
                                        used_p = sorted([str(s) for s in dv.free_symbols if str(s) != dk_str])
                                        mechanism_meta[dk_str] = {
                                            "id": dk_str,
                                            "params": used_p,
                                            "file": rel_file,
                                            "entity": fn_node.name,
                                            "line_range": [st.lineno, getattr(fn_node, "end_lineno", st.lineno)],
                                            "engine": "sympy_expr",
                                            "confidence": 1.0,
                                            "symbolic_expr": str(dv),
                                        }
                                        added_from_this_fn += 1
                            elif isinstance(ret_val, sp.Expr) and not isinstance(ret_val, sp.Symbol):
                                ret_id = self._strip_verb_prefix(fn_node.name)
                                last_ret = fn_node.body[-1]
                                if isinstance(last_ret, ast.Return) and isinstance(last_ret.value, ast.Name):
                                    ret_id = last_ret.value.id
                                if ret_id not in self.graph.nodes and ret_id not in mechanism_meta:
                                    self._get_sym(ret_id)
                                    self._mechanisms[ret_id] = ret_val
                                    if comp.discontinuities:
                                        self._discontinuities[ret_id] = list(comp.discontinuities)
                                    used_p = sorted([str(s) for s in ret_val.free_symbols if str(s) != ret_id])
                                    mechanism_meta[ret_id] = {
                                        "id": ret_id,
                                        "params": used_p,
                                        "file": rel_file,
                                        "entity": fn_node.name,
                                        "line_range": [st.lineno, getattr(fn_node, "end_lineno", st.lineno)],
                                        "engine": "sympy_expr",
                                        "confidence": 1.0,
                                        "symbolic_expr": str(ret_val),
                                    }
                                    added_from_this_fn += 1
                            break
                        except Exception:
                            had_stmt_failure = True
                            break
                    elif isinstance(st, ast.Return) and st.value is not None:
                        try:
                            ret_v = comp.eval_expr(st.value)
                            if isinstance(ret_v, dict):
                                for dk, dv in ret_v.items():
                                    dk_str = str(dk)
                                    if isinstance(dv, sp.Expr) and not isinstance(dv, sp.Symbol) and len(dv.free_symbols) > 0 and dk_str not in self.graph.nodes and dk_str not in mechanism_meta:
                                        self._get_sym(dk_str)
                                        self._mechanisms[dk_str] = dv
                                        used_p = sorted([str(s) for s in dv.free_symbols if str(s) != dk_str])
                                        mechanism_meta[dk_str] = {
                                            "id": dk_str,
                                            "params": used_p,
                                            "file": rel_file,
                                            "entity": fn_node.name,
                                            "line_range": [st.lineno, getattr(st, "end_lineno", st.lineno)],
                                            "engine": "sympy_expr",
                                            "confidence": 1.0,
                                            "symbolic_expr": str(dv),
                                        }
                                        added_from_this_fn += 1
                            elif isinstance(ret_v, sp.Expr) and not isinstance(ret_v, sp.Symbol):
                                ret_id = self._strip_verb_prefix(fn_node.name)
                                if ret_id not in self.graph.nodes and ret_id not in mechanism_meta:
                                    self._get_sym(ret_id)
                                    self._mechanisms[ret_id] = ret_v
                                    used_p = sorted([str(s) for s in ret_v.free_symbols if str(s) != ret_id])
                                    mechanism_meta[ret_id] = {
                                        "id": ret_id,
                                        "params": used_p,
                                        "file": rel_file,
                                        "entity": fn_node.name,
                                        "line_range": [st.lineno, getattr(st, "end_lineno", st.lineno)],
                                        "engine": "sympy_expr",
                                        "confidence": 1.0,
                                        "symbolic_expr": str(ret_v),
                                    }
                                    added_from_this_fn += 1
                        except Exception:
                            had_stmt_failure = True

                if comp.assert_constraints:
                    extracted_assert_constraints.extend(comp.assert_constraints)

                # Si la función monolítica falló simbólicamente (p. ej. bucle opaco o mutación) y no es mero orquestador
                if had_stmt_failure and not pipeline_call_to_var:
                    node_id = self._strip_verb_prefix(fn_node.name)
                    if node_id not in self.graph.nodes:
                        self._get_sym(node_id)
                        ns = dict(project_exec_ns)
                        ns.update(mod_consts)
                        try:
                            exec(compile(ast.Module(body=[fn_node], type_ignores=[]), filename=py_path, mode="exec"), ns)
                            py_callable = ns[fn_node.name]
                            self._blackbox_callables[node_id] = (fn_params, py_callable)
                        except Exception:
                            pass
                        self._mechanisms.pop(node_id, None)
                        mechanism_meta[node_id] = {
                            "id": node_id,
                            "params": fn_params,
                            "file": rel_file,
                            "entity": fn_node.name,
                            "line_range": [fn_node.lineno, getattr(fn_node, "end_lineno", fn_node.lineno)],
                            "engine": "python_callable",
                            "confidence": 0.8,
                            "symbolic_expr": f"blackbox:{fn_node.name}({', '.join(fn_params)})",
                        }

        # 6B. Fase 7 (Parte A): Extraer lineaje causal a nivel de columnas de pipelines tabulares (Pandas/NumPy/Polars)
        source_modalities: Set[str] = {"python_ast"} if mechanism_meta else set()
        tab_compiler = TabularASTToSymPy(known_symbols=self._symbols, module_constants=global_mod_consts)
        for (_, rel_file, tree, mod_consts) in parsed_modules:
            tab_compiler.module_constants.update(mod_consts)
            tab_compiler.extract_from_statements(tree.body, rel_file, entity_name="module_tabular")
            for stmt in tree.body:
                if isinstance(stmt, ast.FunctionDef) and TabularASTToSymPy.is_tabular_function(stmt):
                    tab_compiler.extract_from_statements(stmt.body, rel_file, entity_name=stmt.name)

        for col_id, col_meta in tab_compiler.extracted_columns.items():
            if col_id not in self.graph.nodes and col_id not in mechanism_meta:
                self._get_sym(col_id)
                self._mechanisms[col_id] = col_meta["sympy_obj"]
                if col_meta.get("discontinuities"):
                    self._discontinuities[col_id] = list(col_meta["discontinuities"])
                mechanism_meta[col_id] = col_meta
                source_modalities.add("tabular_dataframe")

        # 6C. Fase 7 (Parte B): Extraer lineaje causal a nivel de columnas y restricciones desde consultas SQL
        sql_compiler = SQLLineageTranspiler(known_symbols=self._symbols, module_constants=global_mod_consts)
        sql_compiler.col_env.update(tab_compiler.col_env)
        for sql_path in sql_files_all:
            rel_sql = self._get_relative_path(sql_path)
            try:
                with open(sql_path, "r", encoding="utf-8") as sf:
                    sql_txt = sf.read()
                sql_compiler.parse_sql_text(sql_txt, rel_sql, base_line=1)
            except Exception:
                pass

        # Soportar también consultas SQL embebidas como cadenas literales en módulos Python
        for (_, rel_file, tree, _) in parsed_modules:
            for sub in ast.walk(tree):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    if re.search(r"\bSELECT\b.+\bFROM\b", sub.value, flags=re.IGNORECASE | re.DOTALL):
                        try:
                            sql_compiler.parse_sql_text(sub.value, rel_file, base_line=getattr(sub, "lineno", 1))
                        except Exception:
                            pass

        for col_id, col_meta in sql_compiler.extracted_columns.items():
            if col_id not in self.graph.nodes and col_id not in mechanism_meta:
                self._get_sym(col_id)
                self._mechanisms[col_id] = col_meta["sympy_obj"]
                if col_meta.get("discontinuities"):
                    self._discontinuities[col_id] = list(col_meta["discontinuities"])
                mechanism_meta[col_id] = col_meta
                source_modalities.add("sql_query")

        # 6D. Fase 8 (Parte A): Extraer mecanismos explícitos documentados en manuales (.md / .txt)
        doc_files_all = sorted(
            p for p in (
                glob.glob(os.path.join(self.target_dir, "**", "*.md"), recursive=True)
                + glob.glob(os.path.join(self.target_dir, "**", "*.txt"), recursive=True)
            )
            if "tests" not in os.path.relpath(p, self.target_dir).replace("\\", "/").split("/")
        )
        known_tokens_pre = set(self.graph.nodes.keys()) | set(mechanism_meta.keys()) | set(all_param_names)
        norm_pre = NormativeDocTranspiler(known_tokens_pre, self._symbols)
        for doc_path in doc_files_all:
            rel_doc = self._get_relative_path(doc_path)
            try:
                res_doc_pre = norm_pre.parse_normative_file(doc_path, rel_doc, self._mechanisms, self.graph.nodes)
                for m_id, m_meta in res_doc_pre["mechanisms"].items():
                    if m_id not in self.graph.nodes and m_id not in mechanism_meta:
                        self._get_sym(m_id)
                        self._mechanisms[m_id] = m_meta["sympy_obj"]
                        mechanism_meta[m_id] = m_meta
                        source_modalities.add("normative_manual")
            except Exception:
                pass

        # 6E. Fase 9 (Parte A): Descubrir ecuaciones estructurales desde CSV para nodos referenciados sin fórmula explícita
        discovered_empirical_estimates: List[Dict[str, Any]] = []
        referenced_input_syms: Set[str] = set()
        for expr in self._mechanisms.values():
            for s in expr.free_symbols:
                referenced_input_syms.add(str(s))
        missing_mech_nodes = {
            s for s in referenced_input_syms
            if s not in self.graph.nodes and s not in mechanism_meta
        }
        all_csv_files = sorted(glob.glob(os.path.join(self.target_dir, "**", "*.csv"), recursive=True))
        if missing_mech_nodes and all_csv_files:
            for csv_file in all_csv_files:
                rel_csv = self._get_relative_path(csv_file)
                try:
                    disc_mechs, disc_emp_meta = discover_missing_mechanisms_from_csv(
                        csv_file, rel_csv, missing_mech_nodes, self.graph.nodes, self._symbols
                    )
                    for m_id, m_meta in disc_mechs.items():
                        if m_id not in self.graph.nodes and m_id not in mechanism_meta:
                            self._get_sym(m_id)
                            self._mechanisms[m_id] = m_meta["sympy_obj"]
                            mechanism_meta[m_id] = m_meta
                            missing_mech_nodes.discard(m_id)
                            source_modalities.add("empirical_csv_surrogate")
                    discovered_empirical_estimates.extend(disc_emp_meta)
                except Exception:
                    pass

        # 7. Construir DAG topológico para deducir automáticamente Target J, Estado V y Nodos Muertos T1-A
        dag = nx.DiGraph()
        for nid in self.graph.nodes:
            dag.add_node(nid)
        for nid in mechanism_meta:
            dag.add_node(nid)

        for nid, m_info in mechanism_meta.items():
            if nid in self._mechanisms:
                expr = self._mechanisms[nid]
                free_names = {str(s) for s in expr.free_symbols}
                parents = [p for p in m_info["params"] if p in free_names or p in dag.nodes]
                for s_name in sorted(free_names):
                    if s_name in dag.nodes and s_name not in parents and s_name != nid:
                        parents.append(s_name)
            else:
                parents = [p for p in m_info["params"] if p in dag.nodes]

            for p in parents:
                if p != nid and p in dag.nodes:
                    dag.add_edge(p, nid)

        # Seleccionar el nodo Target J automáticamente o desde cge.yaml
        sink_candidates = [nid for nid in mechanism_meta if dag.out_degree(nid) == 0]
        if not sink_candidates:
            sink_candidates = list(mechanism_meta.keys())

        if cge_spec and cge_spec.get("target") and (cge_spec["target"] in mechanism_meta or cge_spec["target"] in dag.nodes):
            target_id = str(cge_spec["target"])
        else:
            target_id = max(
                sink_candidates,
                key=lambda nid: (
                    len(nx.ancestors(dag, nid)),
                    len([a for a in nx.ancestors(dag, nid) if a in control_ids])
                )
            )

        # 8. Evaluar estado nominal en orden topológico y registrar nodos V y J
        nominal_state: Dict[str, float] = {
            nid: float(n.current_value)
            for nid, n in self.graph.nodes.items()
            if n.current_value is not None
        }

        topo_order = list(nx.topological_sort(dag))
        for nid in topo_order:
            if nid in self._mechanisms:
                expr = self._mechanisms[nid]
                subs_map = {self._symbols[k]: nominal_state[k] for k in nominal_state if k in self._symbols}
                nominal_state[nid] = float(expr.evalf(subs=subs_map))
            elif nid in self._blackbox_callables:
                p_names, fn_call = self._blackbox_callables[nid]
                args = [nominal_state[p] for p in p_names]
                nominal_state[nid] = float(fn_call(*args))

        for nid, m_info in mechanism_meta.items():
            cat = "target" if nid == target_id else "state"
            has_path_to_j = (nid == target_id) or nx.has_path(dag, nid, target_id)
            val = nominal_state.get(nid, 0.0)
            b_min, b_max = self._infer_node_bounds(nid)
            disc_list = self._discontinuities.get(nid, [])

            self.graph.add_node(Node(
                id=nid,
                label=self._humanize_label(nid, cat, is_dead=not has_path_to_j),
                category=cat,
                data_type="continuous",
                source_reference=SourceReference(
                    file=m_info["file"],
                    entity=m_info["entity"],
                    line_range=m_info["line_range"],
                    column_name=m_info.get("column_name")
                ),
                current_value=round(float(val), 2),
                bounds={"min": b_min, "max": b_max},
                characteristic_scale=max(abs(float(val)), 1.0),
                symbolic_expr=m_info["symbolic_expr"] if m_info["engine"] == "sympy_expr" else None,
                discontinuities=disc_list,
                is_dead=not has_path_to_j
            ))

        # 9. Calcular derivadas parciales exactas / subdiferenciales de Clarke y registrar aristas
        for nid, m_info in mechanism_meta.items():
            for u in dag.predecessors(nid):
                e_type, weight, elast = self._compute_edge_properties(u, nid, nominal_state)
                self.graph.add_edge(Edge(
                    source=u,
                    target=nid,
                    type=e_type,
                    sensitivity_weight=weight,
                    confidence=m_info["confidence"],
                    exact_mechanism={
                        "engine": "python_callable" if m_info["engine"] == "python_callable" else "sympy_expr",
                        "definition": f"{nid} = {m_info['symbolic_expr']}"
                    },
                    dimensionless_elasticity=elast
                ))

        # 9B. Fase 9 (Parte B): Síntesis de subrogados analíticos locales para funciones de caja negra / modelos ML
        blackbox_surrogates: Dict[str, Dict[str, Any]] = {}
        for bb_nid, (bb_params, bb_call) in self._blackbox_callables.items():
            try:
                blackbox_surrogates[bb_nid] = synthesize_blackbox_analytical_surrogate(
                    bb_nid, bb_params, bb_call, nominal_state, self._symbols
                )
                source_modalities.add("blackbox_surrogate")
            except Exception:
                pass

        # 10. Extraer restricciones duras C declaradas en cge.yaml, SQL CHECK, manuales normativos o automáticamente (D4)
        if cge_spec and "constraints" in cge_spec:
            c_data = cge_spec["constraints"]
            if isinstance(c_data, dict):
                for cid, c_info in c_data.items():
                    c_type = c_info.get("type", "hard") if isinstance(c_info, dict) else "hard"
                    c_expr = c_info.get("expression", "") if isinstance(c_info, dict) else str(c_info)
                    c_desc = c_info.get("description", "") if isinstance(c_info, dict) else ""
                    self.graph.add_constraint(Constraint(id=cid, type=c_type, expression=c_expr, description=c_desc))
            elif isinstance(c_data, list):
                for item in c_data:
                    if isinstance(item, dict):
                        cid = item.get("id", "c_custom")
                        c_type = item.get("type", "hard")
                        c_expr = item.get("expression", "")
                        c_desc = item.get("description", "")
                        self.graph.add_constraint(Constraint(id=cid, type=c_type, expression=c_expr, description=c_desc))

        for m_c in module_declared_constraints:
            cid_m = m_c["id"]
            if cid_m not in self.graph.constraints:
                self.graph.add_constraint(Constraint(
                    id=cid_m,
                    type="hard",
                    expression=m_c["expression"],
                    description=m_c["description"],
                ))

        for idx_ac, a_c in enumerate(extracted_assert_constraints):
            cid_a = a_c["id"] or f"c_assert_{idx_ac+1}"
            if cid_a in self.graph.constraints:
                cid_a = f"{cid_a}_L{a_c.get('lineno', idx_ac+1)}"
            if cid_a not in self.graph.constraints:
                self.graph.add_constraint(Constraint(
                    id=cid_a,
                    type="hard",
                    expression=a_c["expression"],
                    description=a_c["description"],
                ))

        for sql_c in sql_compiler.extracted_constraints:
            if sql_c["id"] not in self.graph.constraints:
                self.graph.add_constraint(Constraint(
                    id=sql_c["id"],
                    type=sql_c["type"],
                    expression=sql_c["expression"],
                    description=sql_c["description"],
                ))
                source_modalities.add("sql_query")

        self._extract_constraints_automatically(target_id, ceiling_constants, has_conservation_class)

        # 11. Extraer evidencias empíricas desde data/*.csv (T7) y combinar con subrogados empíricos descubiertos
        empirical_estimates: List[Dict[str, Any]] = list(discovered_empirical_estimates)
        for emp_d in discovered_empirical_estimates:
            src_col = emp_d["source"]
            dst_col = emp_d["target"]
            if src_col in self.graph.nodes and dst_col in self.graph.nodes:
                self.graph.add_edge(Edge(
                    source=src_col,
                    target=dst_col,
                    type="empirical_data",
                    sensitivity_weight=emp_d["local_sensitivity_weight"],
                    confidence=emp_d["confidence"],
                    exact_mechanism={
                        "engine": "statistical_regression",
                        "definition": f"discovered {emp_d['selected_basis']} structural mechanism (beta={emp_d['estimated_coefficient']}) from {emp_d['file']}"
                    }
                ))

        data_csvs = sorted(glob.glob(os.path.join(self.target_dir, "data", "*.csv")))
        for csv_file in data_csvs:
            rel_csv = self._get_relative_path(csv_file)
            if any(e.get("file") == rel_csv for e in empirical_estimates):
                continue
            try:
                est = estimate_structural_parameters_from_csv(csv_file)
                empirical_estimates.append({**est, "file": rel_csv})
                src_col = est["source"]
                dst_col = est["target"]
                if src_col in self.graph.nodes and dst_col in self.graph.nodes:
                    self.graph.add_edge(Edge(
                        source=src_col,
                        target=dst_col,
                        type="empirical_data",
                        sensitivity_weight=est["local_sensitivity_weight"],
                        confidence=est["confidence"],
                        exact_mechanism={
                            "engine": "statistical_regression",
                            "definition": f"estimated {est['selected_basis']} lift (beta={est['estimated_coefficient']}) from {rel_csv}"
                        }
                    ))
                    source_modalities.add("empirical_csv")
            except Exception:
                pass

        # 12. Escanear conflictos Documentación vs Código (T3 / D5) y extraer restricciones normativas de manuales (Fase 8)
        doc_discrepancies = self._scan_doc_code_discrepancies()
        normative_constraints_meta: List[Dict[str, Any]] = []
        norm_transpiler = NormativeDocTranspiler(set(self.graph.nodes.keys()), self._symbols)
        for doc_path in doc_files_all:
            rel_doc = self._get_relative_path(doc_path)
            try:
                norm_res = norm_transpiler.parse_normative_file(doc_path, rel_doc, self._mechanisms, self.graph.nodes)
                for nc in norm_res["constraints"]:
                    normative_constraints_meta.append(nc)
                    cid_to_add = nc["id"]
                    if cid_to_add in self.graph.constraints:
                        l_no = (nc.get("line_range") or [1])[0]
                        cid_to_add = f"{cid_to_add}_L{l_no}"
                    if cid_to_add not in self.graph.constraints:
                        self.graph.add_constraint(Constraint(
                            id=cid_to_add,
                            type=nc["type"],
                            expression=nc["expression"],
                            description=nc["description"],
                        ))
                        source_modalities.add("normative_manual")
                for ndisc in norm_res["discrepancies"]:
                    if not any(
                        d.get("node") == ndisc.get("node")
                        and d.get("doc_file") == ndisc.get("doc_file")
                        and d.get("subtype") == ndisc.get("subtype")
                        and d.get("doc_value") == ndisc.get("doc_value")
                        for d in doc_discrepancies
                    ):
                        doc_discrepancies.append(ndisc)
                        source_modalities.add("normative_manual")
            except Exception:
                pass

        self.graph._blackbox_callables = dict(self._blackbox_callables)
        self.graph.metadata = {
            "target_dir": self.target_dir,
            "target_id": target_id,
            "sense": inferred_sense,
            "ceiling_constants": ceiling_constants,
            "local_optimizers": local_optimizers,
            "conservation_checks": conservation_checks,
            "doc_discrepancies": doc_discrepancies,
            "empirical_estimates": empirical_estimates,
            "normative_constraints": normative_constraints_meta,
            "blackbox_surrogates": blackbox_surrogates,
            "source_modalities": sorted(source_modalities),
        }

        # 13. Fase 10 (Certificado 1): Verificar fidelidad punto a punto entre el código original y el CausalIR
        try:
            certify_fidelity(self.graph, source_dir=self.target_dir)
        except Exception:
            pass

        return self.graph

    def _infer_node_bounds(self, nid: str) -> Tuple[Optional[float], Optional[float]]:
        if nid not in self._mechanisms:
            return (None, None)
        expr = self._mechanisms[nid]
        if nid in self._discontinuities and self._discontinuities[nid]:
            vals = []
            for d in self._discontinuities[nid]:
                if isinstance(d.get("val_true"), (int, float)):
                    vals.append(float(d["val_true"]))
                if isinstance(d.get("val_false"), (int, float)):
                    vals.append(float(d["val_false"]))
            if vals:
                return (min(vals), max(vals))
        if isinstance(expr, sp.Min) and len(expr.args) == 2:
            for a in expr.args:
                s_name = str(a)
                if s_name in self.graph.nodes and self.graph.nodes[s_name].bounds.get("max") is not None:
                    return (0.0, float(self.graph.nodes[s_name].bounds["max"]))
            return (0.0, None)
        # Si es objetivo neto con restas (ingreso - costes), sin cota inferior0
        if isinstance(expr, sp.Add) and any(
            isinstance(arg, sp.Mul) and any(a.is_number and float(a) < 0 for a in arg.args)
            for arg in expr.args
        ):
            return (None, None)
        return (0.0, None)

    def _compute_edge_properties(
        self,
        u: str,
        v: str,
        nominal_state: Dict[str, float]
    ) -> Tuple[str, float, float]:
        """Determina el tipo de arista, la derivada parcial exacta en el punto nominal y la elasticidad adimensional."""
        if v not in self._mechanisms:
            # Diferencias finitas centrales sobre caja negra
            p_names, fn_call = self._blackbox_callables[v]
            eps = 1e-4
            args_plus = [nominal_state[p] + (eps if p == u else 0.0) for p in p_names]
            args_minus = [nominal_state[p] - (eps if p == u else 0.0) for p in p_names]
            grad = (float(fn_call(*args_plus)) - float(fn_call(*args_minus))) / (2.0 * eps)
            s_u = self.graph.nodes[u].characteristic_scale or 1.0
            s_v = self.graph.nodes[v].characteristic_scale or 1.0
            return "exact_code", round(grad, 3), round(grad * (s_u / s_v), 4)

        expr = self._mechanisms[v]
        u_sym = self._get_sym(u)

        # Verificar si es regla discreta (Piecewise con salto o Min entre dos variables no nulas)
        is_discrete_rule = False
        if v in self._discontinuities:
            for d in self._discontinuities[v]:
                if d.get("variable") == u:
                    is_discrete_rule = True
                    jump_delta = float(d.get("jump_delta", 0.0))
                    s_u = self.graph.nodes[u].characteristic_scale or 1.0
                    s_v = self.graph.nodes[v].characteristic_scale or 1.0
                    return "discrete_rule", round(jump_delta, 3), round((jump_delta / s_u) * (s_u / s_v), 4)

        for sub in sp.preorder_traversal(expr):
            if isinstance(sub, sp.Min) and sub.has(u_sym):
                is_discrete_rule = True
                break

        # Para funciones tipo Max(0, u - w) que en el estado nominal están justo en 0 (ej. holding_cost),
        # usar la derivada de la rama activa no nula para representar el mecanismo estructural:
        expr_for_diff = expr
        for sub in list(sp.preorder_traversal(expr)):
            if isinstance(sub, sp.Max) and len(sub.args) == 2 and any(ASTFunctionToSymPy._is_zero_number(a) for a in sub.args):
                nz = [a for a in sub.args if not ASTFunctionToSymPy._is_zero_number(a)][0]
                expr_for_diff = expr_for_diff.subs(sub, nz)

        subs_map = {self._symbols[k]: nominal_state[k] for k in nominal_state if k in self._symbols}
        try:
            if isinstance(expr, sp.Min) and len(expr.args) == 2:
                # Evaluar qué rama del Min está activa en el estado nominal
                a0, a1 = expr.args[0], expr.args[1]
                v0 = float(a0.evalf(subs=subs_map))
                v1 = float(a1.evalf(subs=subs_map))
                active_branch = a0 if v0 <= v1 else a1
                grad = float(sp.diff(active_branch, u_sym).evalf(subs=subs_map))
            else:
                grad = float(sp.diff(expr_for_diff, u_sym).evalf(subs=subs_map))
        except Exception:
            eps = 1e-4
            s_plus = dict(subs_map)
            s_plus[u_sym] = nominal_state[u] + eps
            s_minus = dict(subs_map)
            s_minus[u_sym] = nominal_state[u] - eps
            grad = (float(expr.evalf(subs=s_plus)) - float(expr.evalf(subs=s_minus))) / (2.0 * eps)

        e_type = "discrete_rule" if is_discrete_rule else "exact_code"
        s_u = self.graph.nodes[u].characteristic_scale or 1.0
        s_v = self.graph.nodes[v].characteristic_scale or 1.0
        return e_type, round(grad, 3), round(grad * (s_u / s_v), 4)

    def _extract_constraints_automatically(
        self,
        target_id: str,
        ceiling_constants: Dict[str, float],
        has_conservation_class: bool
    ):
        """Extrae automáticamente las restricciones duras C desde la topología y expresiones simbólicas (D4)."""
        exo_ids = {nid for nid, n in self.graph.nodes.items() if n.category == "exogenous"}
        dag = nx.DiGraph()
        for e in self.graph.edges:
            if e.type != "empirical_data":
                dag.add_edge(e.source, e.target)

        # 1. Restricción presupuestaria / techo de recursos (c_budget)
        if ceiling_constants and target_id in self._mechanisms:
            ceil_val = next(iter(ceiling_constants.values()))
            budget_limit = int(ceil_val) if float(ceil_val).is_integer() else ceil_val
            j_expr = self._mechanisms[target_id]

            # Identificar términos de desembolso directo (aquellos términos negativos en J sin ancestros exógenos U)
            upfront_terms: List[sp.Expr] = []
            for term in sp.Add.make_args(j_expr):
                coeff, rest = term.as_coeff_Mul()
                if float(coeff) < 0:
                    positive_term = -term
                    depends_on_u = False
                    for sym in positive_term.free_symbols:
                        s_id = str(sym)
                        if s_id in exo_ids or (s_id in dag and (nx.ancestors(dag, s_id) & exo_ids)):
                            depends_on_u = True
                            break
                    if not depends_on_u:
                        upfront_terms.append(positive_term)

            if upfront_terms and "c_budget" not in self.graph.constraints:
                lhs_expr = " + ".join(str(t) for t in upfront_terms)
                self.graph.add_constraint(Constraint(
                    id="c_budget",
                    type="hard",
                    expression=f"{lhs_expr} <= {budget_limit}",
                    description=f"Presupuesto máximo de desembolso directo de {budget_limit}"
                ))

        # 2. Invariante de conservación contable (c_ledger)
        if has_conservation_class and "c_ledger" not in self.graph.constraints:
            self.graph.add_constraint(Constraint(
                id="c_ledger",
                type="hard",
                expression="sum(credits) - sum(debits) == 0",
                description="Invariante contable de partida doble en el libro mayor"
            ))

        # 3. Invariantes físicas de truncamiento / capacidad (c_stock)
        for nid, expr in self._mechanisms.items():
            if isinstance(expr, sp.Min) and len(expr.args) == 2:
                a0, a1 = str(expr.args[0]), str(expr.args[1])
                cap_var = a1 if (a1 in self.graph.nodes and self.graph.nodes[a1].category == "control") else a0
                if "c_stock" not in self.graph.constraints:
                    self.graph.add_constraint(Constraint(
                        id="c_stock",
                        type="hard",
                        expression=f"{cap_var} - {nid} >= 0",
                        description=f"El flujo {nid} no puede exceder la capacidad {cap_var}"
                    ))
                break

        # 4. Cotas explícitas de control continuo principal
        for nid, node in self.graph.nodes.items():
            if node.category == "control" and node.data_type == "continuous":
                b_min = node.bounds.get("min")
                b_max = node.bounds.get("max")
                if b_min is not None and b_max is not None and b_min > 0:
                    self.graph.add_constraint(Constraint(
                        id=f"c_{nid}_bounds",
                        type="hard",
                        expression=f"{b_min} <= {nid} <= {b_max}",
                        description=f"Rango admisible de la palanca {nid}"
                    ))
                    break

    def _scan_doc_code_discrepancies(self) -> List[Dict[str, Any]]:
        """Escanea archivos Markdown en el directorio y compara tasas porcentuales documentadas
        frente a mecanismos lineales f_v(u) = r * u (0 < r < 1) extraídos del AST (D5 / T3).
        """
        md_files = sorted(glob.glob(os.path.join(self.target_dir, "*.md")))
        if not md_files:
            return []

        discrepancies: List[Dict[str, Any]] = []
        # Encontrar nodos con mecanismo lineal de tasa 0 < r < 1 respecto a su padre
        rate_nodes: List[Tuple[str, float]] = []
        for nid, expr in self._mechanisms.items():
            free_s = list(expr.free_symbols)
            if len(free_s) == 1:
                deriv = sp.diff(expr, free_s[0])
                if deriv.is_number:
                    r_val = float(deriv.evalf())
                    if 0.0 < r_val < 1.0:
                        rate_nodes.append((nid, r_val))

        if not rate_nodes:
            return []

        for md_path in md_files:
            rel_doc = self._get_relative_path(md_path)
            with open(md_path, "r", encoding="utf-8") as f:
                md_text = f.read()

            pct_matches = re.findall(r"(\d+(?:\.\d+)?)%", md_text)
            if not pct_matches:
                continue

            doc_rate = float(pct_matches[0]) / 100.0
            for nid, code_rate in rate_nodes:
                if abs(doc_rate - code_rate) > 1e-4:
                    node_obj = self.graph.nodes[nid]
                    discrepancies.append({
                        "id": "T3",
                        "code": "DOC_CODE_DISCREPANCY",
                        "severity": "critical",
                        "node": nid,
                        "doc_file": rel_doc,
                        "code_file": node_obj.source_reference.file,
                        "function": node_obj.source_reference.entity,
                        "doc_value": round(doc_rate, 4),
                        "code_value": round(code_rate, 4),
                        "details": {
                            "reason": (
                                f"La documentación '{rel_doc}' especifica {doc_rate*100:.1f}% pero la función "
                                f"'{node_obj.source_reference.entity}' en '{node_obj.source_reference.file}' "
                                f"aplica {code_rate*100:.1f}% sobre '{nid}'."
                            )
                        }
                    })
        return discrepancies


def run_ingestion_pipeline(target_dir: str, output_schema_path: str = "schema.json") -> CausalGraph:
    """Ejecuta el pipeline de ingesta simbólica automática y guarda el CausalIR en output_schema_path."""
    pipeline = ASTIngestPipeline(target_dir)
    graph = pipeline.run()
    graph.save_to_file(output_schema_path)
    return graph


def compile_source_to_ir(
    source_path_or_code: str,
    entrypoint: Optional[str] = None,
    cge_spec: Optional[Dict[str, Any]] = None,
) -> CausalGraph:
    """Compila un directorio, ruta de archivo .py o cadena de código Python a un CausalGraph con Certificado de Fidelidad."""
    import shutil
    import tempfile

    if os.path.isdir(source_path_or_code):
        return ASTIngestPipeline(source_path_or_code).run()

    tmp_dir = tempfile.mkdtemp(prefix="cge_src_compile_")
    try:
        if os.path.isfile(source_path_or_code):
            shutil.copy2(source_path_or_code, os.path.join(tmp_dir, os.path.basename(source_path_or_code)))
        else:
            with open(os.path.join(tmp_dir, "model_source.py"), "w", encoding="utf-8") as f:
                f.write(source_path_or_code)
        if cge_spec:
            with open(os.path.join(tmp_dir, "cge.yaml"), "w", encoding="utf-8") as yf:
                yaml.safe_dump(cge_spec, yf)
        graph = ASTIngestPipeline(tmp_dir).run()
        if not os.path.isfile(source_path_or_code):
            graph.metadata["inline_source_code"] = source_path_or_code
            graph.metadata["target_dir"] = source_path_or_code
        else:
            graph.metadata["target_dir"] = source_path_or_code
        if entrypoint:
            certify_fidelity(graph, source_path_or_code=source_path_or_code, entrypoint=entrypoint)
        return graph
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

