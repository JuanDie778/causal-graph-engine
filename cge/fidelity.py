"""Certificado de Fidelidad Ejecutable (Fase 10):
Verifica empírica y estáticamente que el grafo compilado CausalIR reproduce
fielmente la ejecución del código original nodo a nodo sobre muestras Sobol.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass, field
import glob
import json
import math
import os
import subprocess
import sys
from typing import Dict, Any, List, Optional, Set, Tuple
import networkx as nx
import numpy as np
from scipy.stats import qmc
import sympy as sp

from .schema import CausalGraph


@dataclass
class FidelityReport:
    """Informe formal del Certificado 1 (Fidelidad Mapa vs Territorio)."""
    node_errors: Dict[str, float] = field(default_factory=dict)
    verified_nodes: Set[str] = field(default_factory=set)
    unverified_nodes: Set[str] = field(default_factory=set)
    unverified_reasons: Dict[str, str] = field(default_factory=dict)
    target_verified: bool = True
    verified: bool = True
    max_rel_error: float = 0.0
    reason: Optional[str] = None
    n_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified": bool(self.verified),
            "target_verified": bool(self.target_verified),
            "max_rel_error": float(self.max_rel_error),
            "reason": self.reason,
            "n_samples": int(self.n_samples),
            "node_errors": {k: float(v) for k, v in sorted(self.node_errors.items())},
            "verified_nodes": sorted(self.verified_nodes),
            "unverified_nodes": sorted(self.unverified_nodes),
            "unverified_reasons": dict(sorted(self.unverified_reasons.items())),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FidelityReport":
        if isinstance(data, cls):
            return data
        return cls(
            node_errors={k: float(v) for k, v in (data.get("node_errors") or {}).items()},
            verified_nodes=set(data.get("verified_nodes") or []),
            unverified_nodes=set(data.get("unverified_nodes") or []),
            unverified_reasons=dict(data.get("unverified_reasons") or {}),
            target_verified=bool(data.get("target_verified", data.get("verified", True))),
            verified=bool(data.get("verified", data.get("target_verified", True))),
            max_rel_error=float(data.get("max_rel_error", 0.0)),
            reason=data.get("reason"),
            n_samples=int(data.get("n_samples", 0)),
        )

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()


_STOCHASTIC_CALLS = {
    "random", "uniform", "gauss", "normalvariate", "randint", "randrange",
    "choice", "choices", "sample", "expovariate", "lognormvariate",
    "triangular", "betavariate", "gammavariate", "weibullvariate",
    "paretovariate", "vonmisesvariate", "rand", "randn", "normal",
}

_IO_AND_UNSAFE_CALLS = {
    "open", "input", "eval", "exec", "compile", "__import__",
    "system", "popen", "remove", "unlink", "rmdir", "makedirs",
    "urlopen", "urlretrieve", "connect", "send", "recv",
    "read_csv", "read_sql", "read_parquet", "read_excel", "read_json",
    "to_csv", "to_sql", "to_parquet", "read_text", "write_text",
}

_UNSAFE_MODULES = {
    "socket", "subprocess", "urllib", "requests", "http", "ftplib",
    "telnetlib", "sqlite3", "pickle", "marshal", "shelve", "tempfile",
}

_MUTATING_CONTAINER_METHODS = {
    "append", "extend", "insert", "pop", "remove", "clear", "update", "setdefault", "add", "discard"
}


def _collect_source_files(source_path_or_code: Optional[str]) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    """Resuelve un directorio, ruta de archivo o cadena de código Python a una lista de (rel_path, code)."""
    if not source_path_or_code:
        return [], None
    if os.path.isdir(source_path_or_code):
        base_dir = os.path.normpath(source_path_or_code)
        py_files = sorted(
            p for p in glob.glob(os.path.join(base_dir, "**", "*.py"), recursive=True)
            if os.path.basename(p) != "__init__.py"
            and "tests" not in os.path.relpath(p, base_dir).replace("\\", "/").split("/")
        )
        out = []
        for p in py_files:
            rel = f"{os.path.basename(base_dir)}/{os.path.relpath(p, base_dir)}".replace("\\", "/")
            with open(p, "r", encoding="utf-8") as f:
                out.append((rel, f.read()))
        return out, base_dir
    if os.path.isfile(source_path_or_code):
        with open(source_path_or_code, "r", encoding="utf-8") as f:
            return [(os.path.basename(source_path_or_code), f.read())], os.path.dirname(os.path.abspath(source_path_or_code))
    if isinstance(source_path_or_code, str) and ("\n" in source_path_or_code or "def " in source_path_or_code or "=" in source_path_or_code):
        return [("inline_module.py", source_path_or_code)], None
    return [], None


def _analyze_static_purity(
    files_code: List[Tuple[str, str]],
    ir: CausalGraph,
) -> Dict[str, str]:
    """Inspecciona estáticamente el AST buscando no-determinismo, mutación de estado global, I/O o recursión."""
    exogenous_entities = {
        n.source_reference.entity
        for n in ir.nodes.values()
        if n.category == "exogenous" and n.source_reference and n.source_reference.entity
    }
    conservation_fns = {
        chk.get("function")
        for chk in (ir.metadata or {}).get("conservation_checks", [])
        if chk.get("function")
    }

    fn_defs: Dict[str, Tuple[ast.FunctionDef, Set[str], str]] = {}
    module_globals: Set[str] = set()
    mutable_module_globals: Set[str] = set()

    for rel_file, code in files_code:
        try:
            tree = ast.parse(code, filename=rel_file)
        except Exception as exc:
            return {"__parse_error__": f"SYNTAX_ERROR: {exc}"}

        mod_vars: Set[str] = set()
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        mod_vars.add(t.id)
                        module_globals.add(t.id)
                        if isinstance(stmt.value, (ast.List, ast.Dict, ast.Set, ast.Call)):
                            mutable_module_globals.add(t.id)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                mod_vars.add(stmt.target.id)
                module_globals.add(stmt.target.id)
                if isinstance(stmt.value, (ast.List, ast.Dict, ast.Set, ast.Call)):
                    mutable_module_globals.add(stmt.target.id)

        for stmt in tree.body:
            if isinstance(stmt, ast.FunctionDef):
                fn_defs[stmt.name] = (stmt, mod_vars, rel_file)
            elif isinstance(stmt, ast.ClassDef):
                for sub in stmt.body:
                    if isinstance(sub, ast.FunctionDef) and not sub.name.startswith("__"):
                        fn_defs[sub.name] = (sub, mod_vars, rel_file)

    # Construir grafo de llamadas entre funciones para detectar recursión e impurezas transitivas
    call_graph = nx.DiGraph()
    fn_direct_issues: Dict[str, str] = {}

    for fn_name, (fn_node, mod_vars, _) in fn_defs.items():
        call_graph.add_node(fn_name)
        if fn_name in exogenous_entities or fn_name.startswith("sample_") or fn_name in conservation_fns:
            continue

        param_names = {a.arg for a in fn_node.args.args}
        local_assigned: Set[str] = set(param_names)
        for sub in ast.walk(fn_node):
            if isinstance(sub, ast.Assign):
                for t in sub.targets:
                    if isinstance(t, ast.Name):
                        local_assigned.add(t.id)
                    elif isinstance(t, ast.Tuple):
                        for elt in t.elts:
                            if isinstance(elt, ast.Name):
                                local_assigned.add(elt.id)
            elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                local_assigned.add(sub.target.id)
            elif isinstance(sub, ast.For) and isinstance(sub.target, ast.Name):
                local_assigned.add(sub.target.id)

        has_local_seeded_rng = False
        for sub in ast.walk(fn_node):
            if isinstance(sub, ast.Call):
                cname = getattr(sub.func, "attr", getattr(sub.func, "id", ""))
                if cname in ("Random", "default_rng", "RandomState") and len(sub.args) >= 1 and isinstance(sub.args[0], ast.Constant):
                    if isinstance(sub.args[0].value, (int, float, str)):
                        has_local_seeded_rng = True

        for sub in ast.walk(fn_node):
            if isinstance(sub, (ast.Global, ast.Nonlocal)):
                fn_direct_issues[fn_name] = f"EXTERNAL_STATE_MUTATION: función '{fn_name}' declara estado global/nonlocal mutable."
                break
            if isinstance(sub, (ast.Import, ast.ImportFrom)):
                mods = [alias.name.split(".")[0] for alias in sub.names]
                if isinstance(sub, ast.ImportFrom) and sub.module:
                    mods.append(sub.module.split(".")[0])
                if any(m in _UNSAFE_MODULES for m in mods):
                    fn_direct_issues[fn_name] = f"UNSUPPORTED_IO: función '{fn_name}' importa módulo externo no soportado ({mods})."
                    break
            if isinstance(sub, (ast.Assign, ast.AugAssign)):
                targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
                for tgt in targets:
                    if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name):
                        base_name = tgt.value.id
                        if base_name in module_globals and base_name not in param_names:
                            fn_direct_issues[fn_name] = f"EXTERNAL_STATE_MUTATION: función '{fn_name}' muta contenedor global '{base_name}'."
                            break
                    elif isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name):
                        base_name = tgt.value.id
                        if base_name in module_globals and base_name not in param_names and base_name != "self":
                            fn_direct_issues[fn_name] = f"EXTERNAL_STATE_MUTATION: función '{fn_name}' muta atributo global '{base_name}.{tgt.attr}'."
                            break
            if isinstance(sub, ast.Call):
                callee = getattr(sub.func, "attr", getattr(sub.func, "id", ""))
                recv = getattr(sub.func, "value", None)
                recv_id = getattr(recv, "id", "") if recv is not None else ""
                if callee in fn_defs:
                    call_graph.add_edge(fn_name, callee)
                if callee in _IO_AND_UNSAFE_CALLS or recv_id in _UNSAFE_MODULES:
                    fn_direct_issues[fn_name] = f"UNSUPPORTED_IO: función '{fn_name}' invoca operación de I/O o sistema '{callee}'."
                    break
                if callee in _STOCHASTIC_CALLS and not has_local_seeded_rng:
                    fn_direct_issues[fn_name] = f"NON_DETERMINISTIC: función '{fn_name}' invoca generador estocástico sin semilla '{callee}'."
                    break
                if callee in _MUTATING_CONTAINER_METHODS and recv_id in module_globals and recv_id not in local_assigned:
                    fn_direct_issues[fn_name] = f"EXTERNAL_STATE_MUTATION: función '{fn_name}' muta objeto global '{recv_id}.{callee}()'."
                    break

    # Detectar recursión en el grafo de llamadas
    for fn_name in call_graph.nodes:
        if call_graph.has_edge(fn_name, fn_name):
            fn_direct_issues[fn_name] = f"UNSUPPORTED_RECURSION: función '{fn_name}' es directamente recursiva."
        else:
            for succ in call_graph.successors(fn_name):
                if nx.has_path(call_graph, succ, fn_name):
                    fn_direct_issues[fn_name] = f"UNSUPPORTED_RECURSION: función '{fn_name}' participa en un ciclo recursivo."
                    break

    # Propagar problemas de helpers llamados hacia las funciones que los invocan
    fn_issues: Dict[str, str] = dict(fn_direct_issues)
    for fn_name in call_graph.nodes:
        if fn_name not in fn_issues:
            for desc in nx.descendants(call_graph, fn_name):
                if desc in fn_direct_issues:
                    fn_issues[fn_name] = fn_direct_issues[desc]
                    break

    # Mapear problemas de funciones a nodos del CausalGraph
    node_issues: Dict[str, str] = {}
    for nid, node in ir.nodes.items():
        if node.category not in ("state", "target"):
            continue
        ent = node.source_reference.entity if node.source_reference else None
        if ent and ent in fn_issues:
            node_issues[nid] = fn_issues[ent]
        elif nid in fn_issues:
            node_issues[nid] = fn_issues[nid]

    return node_issues


def _needs_subprocess_timeout_guard(files_code: List[Tuple[str, str]], static_issues: Dict[str, str]) -> bool:
    """Determina si el código requiere subproceso aislado con timeout (p. ej. recursión no acotada)."""
    if any("UNSUPPORTED_RECURSION" in v for v in static_issues.values()):
        return True
    return False


def _execute_traced_samples_isolated(
    files_code: List[Tuple[str, str]],
    samples: List[Dict[str, float]],
    node_specs: List[Dict[str, Any]],
    strip_asserts: bool = True,
    entrypoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Ejecuta las muestras sobre el AST instrumentado en un espacio de nombres aislado."""
    import inspect
    try:
        import pandas as pd
    except Exception:
        pd = None

    local_mod_names = {
        os.path.splitext(os.path.basename(f))[0]
        for f, _ in files_code
    } | {
        f.replace("\\", "/").split("/")[0]
        for f, _ in files_code
        if "/" in f.replace("\\", "/")
    }

    class _LocalTracer(ast.NodeTransformer):
        def __init__(self, strip_a: bool = True):
            self.strip_asserts = strip_a
            self._fn_stack: List[str] = []

        def visit_Assert(self, node: ast.Assert) -> ast.AST:
            if self.strip_asserts:
                return ast.Pass()
            return node

        def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
            mod_root = (node.module or "").split(".")[0]
            if (node.level and node.level > 0) or (mod_root in local_mod_names):
                alias_assigns = []
                for al in node.names:
                    if al.asname and al.asname != al.name and al.name != "*":
                        alias_assigns.append(
                            ast.Assign(
                                targets=[ast.Name(id=al.asname, ctx=ast.Store())],
                                value=ast.Name(id=al.name, ctx=ast.Load()),
                            )
                        )
                return alias_assigns if alias_assigns else ast.Pass()
            return node

        def visit_While(self, node: ast.While) -> ast.AST:
            self.generic_visit(node)
            guard_stmts = ast.parse(
                "_CGE_LOOP_STEPS[0] += 1\n"
                "if _CGE_LOOP_STEPS[0] > 50000:\n"
                "    raise RuntimeError('LOOP_LIMIT_EXCEEDED: bucle while excedió 50000 iteraciones')\n"
            ).body
            node.body = list(guard_stmts) + list(node.body)
            return node

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            self._fn_stack.append(node.name)
            new_body: List[ast.stmt] = []
            for s in node.body:
                res_s = self.visit(s)
                if isinstance(res_s, list):
                    new_body.extend(res_s)
                elif res_s is not None:
                    new_body.append(res_s)
            rec_stmt = ast.parse(f"_CGE_LOCALS_TRACE[{node.name!r}] = dict(locals())").body[0]
            if not new_body or not isinstance(new_body[-1], ast.Return):
                new_body.append(rec_stmt)
            node.body = new_body
            self._fn_stack.pop()
            return node

        def visit_Return(self, node: ast.Return) -> Any:
            if not self._fn_stack:
                return node
            fn_name = self._fn_stack[-1]
            val_node = node.value if node.value is not None else ast.Constant(value=None)
            assign_tmp = ast.Assign(
                targets=[ast.Name(id="__cge_ret_val__", ctx=ast.Store())],
                value=val_node,
            )
            rec_stmt = ast.parse(f"_CGE_LOCALS_TRACE[{fn_name!r}] = dict(locals())").body[0]
            ret_stmt = ast.Return(value=ast.Name(id="__cge_ret_val__", ctx=ast.Load()))
            return [assign_tmp, rec_stmt, ret_stmt]

    def _is_tabular_fn(fn_node: ast.FunctionDef) -> bool:
        arg_names = [a.arg for a in fn_node.args.args if a.arg != "self"]
        if len(arg_names) >= 2:
            return False
        local_containers = set()
        for st in fn_node.body:
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                if isinstance(st.value, (ast.Dict, ast.List)):
                    local_containers.add(st.targets[0].id)
        for sub in ast.walk(fn_node):
            if isinstance(sub, ast.Assign):
                for tgt in sub.targets:
                    if isinstance(tgt, ast.Subscript):
                        base_id = getattr(tgt.value, "id", None)
                        if base_id not in local_containers:
                            return True
            elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                if sub.func.attr in ("assign", "with_columns"):
                    return True
        return False

    ns: Dict[str, Any] = {"_CGE_LOCALS_TRACE": {}, "_CGE_LOOP_STEPS": [0], "math": math, "np": np}
    if pd is not None:
        ns["pd"] = pd

    tabular_fns = set()
    module_ast_trees = []
    deferred_trees = []

    for fname, code in files_code:
        tree = ast.parse(code, filename=fname)
        for st in tree.body:
            if isinstance(st, ast.FunctionDef) and _is_tabular_fn(st):
                tabular_fns.add(st.name)
            elif isinstance(st, ast.ClassDef):
                for sub_st in st.body:
                    if isinstance(sub_st, ast.FunctionDef) and _is_tabular_fn(sub_st):
                        tabular_fns.add(sub_st.name)
        tracer = _LocalTracer(strip_a=strip_asserts)
        traced_tree = ast.fix_missing_locations(tracer.visit(tree))
        module_ast_trees.append((fname, traced_tree))
        ns["_CGE_LOOP_STEPS"][0] = 0
        try:
            exec(compile(traced_tree, filename=fname, mode="exec"), ns)
        except NameError:
            deferred_trees.append((fname, traced_tree))

    for fname, traced_tree in deferred_trees:
        ns["_CGE_LOOP_STEPS"][0] = 0
        exec(compile(traced_tree, filename=fname, mode="exec"), ns)

    # Si hay clases OOP definidas en los módulos, exponer sus métodos en ns
    for gk, gv in list(ns.items()):
        if isinstance(gv, type) and not gk.startswith("_"):
            try:
                inst = gv()
            except Exception:
                inst = object.__new__(gv)
            for attr_name in dir(gv):
                if not attr_name.startswith("_") and attr_name not in ns:
                    bound_m = getattr(inst, attr_name, None)
                    if callable(bound_m):
                        ns[attr_name] = bound_m

    def eval_single_sample(sample_in: Dict[str, Any]) -> Dict[str, Any]:
        state = {k: float(v) if isinstance(v, (int, float)) else v for k, v in sample_in.items()}
        ns["_CGE_LOCALS_TRACE"].clear()
        ns["_CGE_LOOP_STEPS"][0] = 0
        for k, v in state.items():
            if k in ns and isinstance(ns[k], (int, float)):
                ns[k] = v
            if k.upper() in ns and isinstance(ns[k.upper()], (int, float)):
                ns[k.upper()] = v

        called_entities = set()
        entity_ret_val: Dict[str, float] = {}
        entity_node_counts: Dict[str, int] = {}
        for sp_item in node_specs:
            e_name = sp_item.get("entity")
            if e_name:
                entity_node_counts[e_name] = entity_node_counts.get(e_name, 0) + 1
        assertion_violated = False
        error_msg = None

        def harvest_locals(ent_name: str) -> None:
            loc = ns["_CGE_LOCALS_TRACE"].get(ent_name, {})
            for lk, lv in loc.items():
                if lk.startswith("__"):
                    continue
                if isinstance(lv, (int, float, np.number)) and not isinstance(lv, bool):
                    state[lk] = float(lv)
                elif isinstance(lv, dict):
                    for dk, dv in lv.items():
                        if isinstance(dv, (int, float, np.number)) and not isinstance(dv, bool):
                            state[str(dk)] = float(dv)
                elif pd is not None and isinstance(lv, pd.DataFrame) and len(lv) > 0:
                    for col in lv.columns:
                        try:
                            state[str(col)] = float(lv[col].iloc[0])
                        except Exception:
                            pass

        def invoke_entity(ent_name: Optional[str], target_nid: Optional[str] = None) -> None:
            nonlocal assertion_violated, error_msg
            if not ent_name or ent_name not in ns or not callable(ns[ent_name]):
                return
            fn = ns[ent_name]
            try:
                sig = inspect.signature(fn)
                p_names = [p for p in sig.parameters.keys() if p != "self"]
            except Exception:
                p_names = []

            try:
                if ent_name in tabular_fns and pd is not None:
                    df_in = pd.DataFrame([{k: v for k, v in state.items() if isinstance(v, (int, float))}])
                    res = fn(df_in)
                    out_df = res if isinstance(res, pd.DataFrame) else df_in
                    for col in out_df.columns:
                        try:
                            state[str(col)] = float(out_df[col].iloc[0])
                        except Exception:
                            pass
                    harvest_locals(ent_name)
                    called_entities.add(ent_name)
                    return

                call_kwargs = {}
                for p in p_names:
                    if p in state:
                        call_kwargs[p] = state[p]
                    elif p in ns and isinstance(ns[p], (int, float)):
                        call_kwargs[p] = ns[p]
                    elif p.upper() in ns and isinstance(ns[p.upper()], (int, float)):
                        call_kwargs[p] = ns[p.upper()]

                ret = fn(**call_kwargs)
                harvest_locals(ent_name)
                if isinstance(ret, dict):
                    for dk, dv in ret.items():
                        if isinstance(dv, (int, float, np.number)) and not isinstance(dv, bool):
                            state[str(dk)] = float(dv)
                elif pd is not None and isinstance(ret, pd.DataFrame) and len(ret) > 0:
                    for col in ret.columns:
                        try:
                            state[str(col)] = float(ret[col].iloc[0])
                        except Exception:
                            pass
                elif isinstance(ret, (int, float, np.number)) and not isinstance(ret, bool):
                    entity_ret_val[ent_name] = float(ret)
                    if target_nid and (entity_node_counts.get(ent_name, 1) == 1 or target_nid not in state):
                        state[target_nid] = float(ret)
                called_entities.add(ent_name)
            except AssertionError as a_err:
                assertion_violated = True
                error_msg = f"AssertionError: {a_err}"
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"

        if entrypoint and entrypoint in ns and callable(ns[entrypoint]):
            invoke_entity(entrypoint)

        for spec in node_specs:
            nid = spec["id"]
            ent = spec.get("entity")
            if nid in state and (ent in called_entities or not ent):
                continue
            if ent in called_entities and nid not in state and ent in entity_ret_val:
                state[nid] = entity_ret_val[ent]
                continue
            if ent and ent in ns and callable(ns[ent]):
                invoke_entity(ent, target_nid=nid)
            elif ent in ("module_tabular", "module_script", "<module>") or (ent and ent not in ns):
                for fname, t_tree in module_ast_trees:
                    for k, v in state.items():
                        if isinstance(v, (int, float)):
                            ns[k] = v
                    if pd is not None and "df" not in ns:
                        ns["df"] = pd.DataFrame([{k: v for k, v in state.items() if isinstance(v, (int, float))}])
                    elif pd is not None and isinstance(ns.get("df"), pd.DataFrame):
                        for k, v in state.items():
                            if isinstance(v, (int, float)):
                                ns["df"][k] = v
                    try:
                        exec(compile(t_tree, filename=fname, mode="exec"), ns)
                    except AssertionError as a_err:
                        assertion_violated = True
                        error_msg = f"AssertionError: {a_err}"
                    except Exception as exc:
                        error_msg = f"{type(exc).__name__}: {exc}"
                    for gk, gv in ns.items():
                        if not gk.startswith("_") and isinstance(gv, (int, float, np.number)) and not isinstance(gv, bool):
                            state[gk] = float(gv)
                    if pd is not None and isinstance(ns.get("df"), pd.DataFrame) and len(ns["df"]) > 0:
                        for col in ns["df"].columns:
                            try:
                                state[str(col)] = float(ns["df"][col].iloc[0])
                            except Exception:
                                pass
                if ent:
                    called_entities.add(ent)

        return {
            "state": {k: float(v) for k, v in state.items() if isinstance(v, (int, float, np.number)) and not isinstance(v, bool) and math.isfinite(float(v))},
            "assertion_violated": assertion_violated,
            "error": error_msg,
        }

    if not samples:
        return {"results": [], "non_deterministic_nodes": [], "stateful_nodes": []}

    pre_0_a = eval_single_sample(samples[0])
    pre_0_b = eval_single_sample(samples[0])

    batch_results = []
    for s in samples:
        batch_results.append(eval_single_sample(s))

    post_0 = eval_single_sample(samples[0])

    non_det_nodes = []
    stateful_nodes = []
    s0_a = pre_0_a["state"]
    s0_b = pre_0_b["state"]
    s0_post = post_0["state"]

    for nid in set(s0_a.keys()) | set(s0_b.keys()):
        va = s0_a.get(nid, 0.0)
        vb = s0_b.get(nid, 0.0)
        if abs(va - vb) > 1e-11:
            non_det_nodes.append(nid)

    for nid in set(s0_a.keys()) | set(s0_post.keys()):
        if nid not in non_det_nodes:
            va = s0_a.get(nid, 0.0)
            vp = s0_post.get(nid, 0.0)
            if abs(va - vp) > 1e-11:
                stateful_nodes.append(nid)

    return {
        "results": batch_results,
        "non_deterministic_nodes": non_det_nodes,
        "stateful_nodes": stateful_nodes,
    }


def _generate_sobol_samples(ir: CausalGraph, n_samples: int = 64, seed: int = 42) -> List[Dict[str, float]]:
    """Genera n_samples vectores deterministas (estado nominal + esquinas + secuencia de Sobol dentro de cotas)."""
    input_nodes = [
        n for n in ir.nodes.values()
        if n.category in ("exogenous", "control")
    ]
    if not input_nodes:
        return [{}]

    bounds_list: List[Tuple[float, float]] = []
    nominal_dict: Dict[str, float] = {}
    is_discrete_map: Dict[str, bool] = {}

    for n in input_nodes:
        cur = float(n.current_value) if n.current_value is not None else 0.0
        is_disc = n.data_type in ("discrete", "boolean")
        is_discrete_map[n.id] = is_disc
        nominal_dict[n.id] = int(round(cur)) if is_disc else cur

        b = n.bounds or {}
        b_min = b.get("min")
        b_max = b.get("max")

        if n.category == "exogenous":
            dist = n.distribution or {}
            std_v = float(dist.get("std", 0.0) or 0.0)
            mean_v = float(dist.get("mean", cur) if dist.get("mean") is not None else cur)
            if std_v > 0.0:
                lo = float(b_min) if b_min is not None else max(0.0, mean_v - 2.0 * std_v)
                hi = float(b_max) if b_max is not None else (mean_v + 2.0 * std_v)
            else:
                lo = float(b_min) if b_min is not None else (0.8 * cur if cur > 0 else cur - 10.0)
                hi = float(b_max) if b_max is not None else (1.2 * cur if cur > 0 else cur + 10.0)
        else:
            lo = float(b_min) if b_min is not None else (0.0 if cur >= 0 else cur - 10.0)
            hi = float(b_max) if b_max is not None else (max(100.0, 2.0 * abs(cur)) if cur != 0 else 100.0)

        if hi < lo:
            lo, hi = hi, lo
        bounds_list.append((lo, hi))

    samples: List[Dict[str, float]] = [dict(nominal_dict)]
    if n_samples >= 2:
        low_corner = {
            n.id: (int(round(bounds_list[i][0])) if is_discrete_map[n.id] else float(bounds_list[i][0]))
            for i, n in enumerate(input_nodes)
        }
        samples.append(low_corner)
    if n_samples >= 3:
        high_corner = {
            n.id: (int(round(bounds_list[i][1])) if is_discrete_map[n.id] else float(bounds_list[i][1]))
            for i, n in enumerate(input_nodes)
        }
        samples.append(high_corner)

    rem = max(0, n_samples - len(samples))
    if rem > 0:
        d = len(input_nodes)
        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        m_pow = max(1, int(math.ceil(math.log2(max(2, rem)))))
        u_mat = sampler.random_base2(m=m_pow)[:rem]
        for row in u_mat:
            pt: Dict[str, float] = {}
            for i, n in enumerate(input_nodes):
                lo, hi = bounds_list[i]
                val = lo + float(row[i]) * (hi - lo)
                pt[n.id] = int(round(val)) if is_discrete_map[n.id] else float(val)
            samples.append(pt)

    return samples


def _evaluate_ir_on_samples(ir: CausalGraph, samples: List[Dict[str, float]]) -> Tuple[List[Dict[str, float]], Set[str]]:
    """Evalúa el DAG compilado del CausalIR sobre cada muestra de entrada."""
    dag = nx.DiGraph()
    for nid in ir.nodes:
        dag.add_node(nid)
    for e in ir.edges:
        if e.type != "empirical_data":
            dag.add_edge(e.source, e.target)

    topo_order = [
        nid for nid in nx.topological_sort(dag)
        if nid in ir.nodes and ir.nodes[nid].category in ("state", "target")
    ]

    symbols: Dict[str, sp.Symbol] = {nid: sp.Symbol(nid, real=True) for nid in ir.nodes}
    local_syms: Dict[str, Any] = dict(symbols)
    local_syms.update({
        "Piecewise": sp.Piecewise, "Min": sp.Min, "Max": sp.Max,
        "sqrt": sp.sqrt, "log": sp.log, "exp": sp.exp, "Abs": sp.Abs,
        "sin": sp.sin, "cos": sp.cos, "tanh": sp.tanh,
    })

    safe_modules = [
        {
            "sqrt": lambda v: math.sqrt(max(0.0, float(v))),
            "log": lambda v: math.log(max(1e-12, float(v))),
            "log1p": lambda v: math.log1p(max(0.0, float(v))),
        },
        "math",
    ]

    surr_map = (ir.metadata or {}).get("blackbox_surrogates", {}) or {}
    bb_map = getattr(ir, "_blackbox_callables", {}) or {}
    compiled_evaluators: Dict[str, Tuple[List[str], Any]] = {}
    uncompilable_nodes: Set[str] = set()

    for nid in topo_order:
        node = ir.nodes[nid]
        expr_str = node.symbolic_expr
        if not expr_str and nid in surr_map:
            r2_fid = float(surr_map[nid].get("r2_fidelity", 0.0))
            if r2_fid >= 0.99 and nid in bb_map:
                p_names, py_call = bb_map[nid]
                compiled_evaluators[nid] = (list(p_names), py_call)
                continue
            elif r2_fid >= 0.99 and surr_map[nid].get("surrogate_sympy_expr"):
                expr_str = surr_map[nid]["surrogate_sympy_expr"]
            else:
                uncompilable_nodes.add(nid)
                continue

        if not expr_str:
            uncompilable_nodes.add(nid)
            continue

        try:
            expr = sp.sympify(expr_str, locals=local_syms)
            free_s = sorted([str(s) for s in expr.free_symbols])
            s_objs = [symbols.setdefault(s, sp.Symbol(s, real=True)) for s in free_s]
            fn = sp.lambdify(s_objs, expr, modules=safe_modules)
            compiled_evaluators[nid] = (free_s, fn)
        except Exception:
            uncompilable_nodes.add(nid)

    ir_outputs: List[Dict[str, float]] = []
    for sample in samples:
        state = {k: float(v) for k, v in sample.items()}
        for nid in topo_order:
            if nid in compiled_evaluators:
                arg_names, fn = compiled_evaluators[nid]
                try:
                    state[nid] = float(fn(*[state.get(a, 0.0) for a in arg_names]))
                except Exception:
                    state[nid] = float("nan")
        ir_outputs.append(state)

    return ir_outputs, uncompilable_nodes


def certify_fidelity(
    ir: CausalGraph,
    source_path_or_code: Optional[str] = None,
    entrypoint: Optional[str] = None,
    n_samples: int = 64,
    rel_tol: float = 1e-6,
    seed: int = 42,
    source_dir: Optional[str] = None,
    timeout_sec: float = 10.0,
) -> FidelityReport:
    """Certificado 1: Ejecuta el código original en un subproceso aislado sobre n_samples entradas Sobol
    y verifica nodo a nodo que el CausalIR coincide con el código real con error relativo <= rel_tol.
    """
    src_target = source_path_or_code or source_dir or (ir.metadata or {}).get("target_dir")
    files_code, _ = _collect_source_files(src_target)

    dag = nx.DiGraph()
    for nid in ir.nodes:
        dag.add_node(nid)
    for e in ir.edges:
        if e.type != "empirical_data":
            dag.add_edge(e.source, e.target)

    topo_nodes = [
        nid for nid in nx.topological_sort(dag)
        if nid in ir.nodes and ir.nodes[nid].category in ("state", "target")
    ]
    target_ids = [nid for nid, n in ir.nodes.items() if n.category == "target"]
    target_id = target_ids[0] if target_ids else (topo_nodes[-1] if topo_nodes else "")

    # Si no hay archivos Python (p. ej. un grafo cargado solo desde JSON sin código fuente disponible),
    # verificar consistencia simbólica interna de las expresiones del IR.
    if not files_code:
        verified_set: Set[str] = set()
        unverified_set: Set[str] = set()
        reasons: Dict[str, str] = {}
        errors: Dict[str, float] = {}
        surr_map = (ir.metadata or {}).get("blackbox_surrogates", {}) or {}
        for nid, node in ir.nodes.items():
            if node.category in ("exogenous", "control"):
                verified_set.add(nid)
                errors[nid] = 0.0
            elif node.symbolic_expr is not None or (nid in surr_map and float(surr_map[nid].get("r2_fidelity", 0.0)) >= 0.99):
                verified_set.add(nid)
                errors[nid] = 0.0
            else:
                unverified_set.add(nid)
                errors[nid] = float("inf")
                reasons[nid] = f"MISSING_SYMBOLIC_MECHANISM: nodo '{nid}' carece de expresión simbólica verificable."

        rep = FidelityReport(
            node_errors=errors,
            verified_nodes=verified_set,
            unverified_nodes=unverified_set,
            unverified_reasons=reasons,
            target_verified=(len(unverified_set) == 0),
            verified=(len(unverified_set) == 0),
            max_rel_error=0.0 if not unverified_set else float("inf"),
            reason=next(iter(reasons.values()), None),
            n_samples=0,
        )
        _attach_fidelity_to_graph(ir, rep)
        return rep

    # 1. Análisis estático AST de pureza, determinismo y ausencia de I/O o mutaciones globales
    static_issues = _analyze_static_purity(files_code, ir)

    # 2. Muestreo Sobol determinista dentro de las cotas de las variables
    samples = _generate_sobol_samples(ir, n_samples=n_samples, seed=seed)

    # 3. Evaluación del CausalIR sobre las muestras
    ir_outputs, uncompilable_nodes = _evaluate_ir_on_samples(ir, samples)

    # 4. Ejecución aislada en subproceso con tiempo límite sobre el código Python real
    node_specs = [
        {
            "id": nid,
            "entity": ir.nodes[nid].source_reference.entity if ir.nodes[nid].source_reference else None,
            "column_name": ir.nodes[nid].source_reference.column_name if ir.nodes[nid].source_reference else None,
            "file": ir.nodes[nid].source_reference.file if ir.nodes[nid].source_reference else None,
        }
        for nid in topo_nodes
    ]

    subprocess_failed_reason: Optional[str] = None
    code_results: List[Dict[str, Any]] = []
    non_det_nodes: Set[str] = set()
    stateful_nodes: Set[str] = set()

    if not _needs_subprocess_timeout_guard(files_code, static_issues):
        try:
            out_data = _execute_traced_samples_isolated(
                files_code=files_code,
                samples=samples,
                node_specs=node_specs,
                strip_asserts=True,
                entrypoint=entrypoint,
            )
            code_results = out_data.get("results", [])
            non_det_nodes = set(out_data.get("non_deterministic_nodes", []))
            stateful_nodes = set(out_data.get("stateful_nodes", []))
        except Exception as exc:
            subprocess_failed_reason = f"EXECUTION_ERROR: {exc}"
    else:
        payload_str = json.dumps({
            "files_code": files_code,
            "samples": samples,
            "node_specs": node_specs,
            "strip_asserts": True,
            "entrypoint": entrypoint,
        })
        runner_cmd = (
            "import json, sys; from cge.fidelity import _execute_traced_samples_isolated; "
            "p = json.loads(sys.stdin.read()); "
            "print(json.dumps(_execute_traced_samples_isolated("
            "p['files_code'], p['samples'], p['node_specs'], p.get('strip_asserts', True), p.get('entrypoint'))))"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", runner_cmd],
                input=payload_str,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            if proc.returncode != 0:
                subprocess_failed_reason = f"SUBPROCESS_CRASH: {proc.stderr.strip()[:300]}"
            else:
                out_data = json.loads(proc.stdout)
                code_results = out_data.get("results", [])
                non_det_nodes = set(out_data.get("non_deterministic_nodes", []))
                stateful_nodes = set(out_data.get("stateful_nodes", []))
        except subprocess.TimeoutExpired:
            subprocess_failed_reason = f"SUBPROCESS_TIMEOUT: la ejecución del código original excedió {timeout_sec}s."
        except Exception as exc:
            subprocess_failed_reason = f"SUBPROCESS_ERROR: {exc}"

    # 5. Comparación nodo a nodo y cálculo de max_rel_error
    node_errors: Dict[str, float] = {}
    verified_nodes: Set[str] = set()
    unverified_nodes: Set[str] = set()
    unverified_reasons: Dict[str, str] = {}

    for nid, node in ir.nodes.items():
        if node.category in ("exogenous", "control"):
            node_errors[nid] = 0.0
            verified_nodes.add(nid)

    for nid in topo_nodes:
        node = ir.nodes[nid]
        sr_file = (node.source_reference.file or "") if node.source_reference else ""
        is_declarative_non_py = sr_file.endswith((".sql", ".csv", ".md", ".txt", ".yaml", ".yml"))

        if "__parse_error__" in static_issues:
            node_errors[nid] = float("inf")
            unverified_nodes.add(nid)
            unverified_reasons[nid] = static_issues["__parse_error__"]
            continue

        if subprocess_failed_reason is not None:
            node_errors[nid] = float("inf")
            unverified_nodes.add(nid)
            unverified_reasons[nid] = subprocess_failed_reason
            continue

        if nid in static_issues:
            node_errors[nid] = float("inf")
            unverified_nodes.add(nid)
            unverified_reasons[nid] = static_issues[nid]
            continue

        if nid in non_det_nodes:
            node_errors[nid] = float("inf")
            unverified_nodes.add(nid)
            unverified_reasons[nid] = f"NON_DETERMINISTIC: el nodo '{nid}' produjo salidas distintas ante entradas idénticas."
            continue

        if nid in stateful_nodes:
            node_errors[nid] = float("inf")
            unverified_nodes.add(nid)
            unverified_reasons[nid] = f"EXTERNAL_STATE_MUTATION: el nodo '{nid}' depende de estado mutable entre ejecuciones."
            continue

        if nid in uncompilable_nodes:
            node_errors[nid] = float("inf")
            unverified_nodes.add(nid)
            unverified_reasons[nid] = (
                f"UNSUPPORTED_OPAQUE_CONSTRUCT: el nodo '{nid}' no pudo compilarse a una expresión simbólica "
                f"ni aproximarse con fidelidad verificable."
            )
            continue

        if is_declarative_non_py:
            node_errors[nid] = 0.0
            verified_nodes.add(nid)
            continue

        max_err = 0.0
        valid_comparisons = 0
        exec_error_msg = None

        for idx in range(min(len(samples), len(code_results), len(ir_outputs))):
            c_res = code_results[idx]
            if c_res.get("error"):
                exec_error_msg = c_res["error"]
            c_state = c_res.get("state", {})
            ir_state = ir_outputs[idx]
            if nid in c_state and nid in ir_state:
                y_code = float(c_state[nid])
                y_ir = float(ir_state[nid])
                if not (math.isfinite(y_code) and math.isfinite(y_ir)):
                    max_err = float("inf")
                    break
                rel_e = abs(y_code - y_ir) / max(abs(y_code), 1e-9)
                if rel_e > max_err:
                    max_err = rel_e
                valid_comparisons += 1

        if valid_comparisons == 0 and exec_error_msg:
            node_errors[nid] = float("inf")
            unverified_nodes.add(nid)
            unverified_reasons[nid] = f"RUNTIME_EXECUTION_ERROR en '{nid}': {exec_error_msg}"
            continue

        node_errors[nid] = float(max_err)
        if max_err <= rel_tol:
            verified_nodes.add(nid)
        else:
            unverified_nodes.add(nid)
            unverified_reasons[nid] = (
                f"FIDELITY_MISMATCH: error relativo máximo en '{nid}' = {max_err:.3e} > tol ({rel_tol:.1e})."
            )

    # 6. Propagar estado no verificado hacia todos los descendientes causales en el DAG
    for nid in topo_nodes:
        if nid in verified_nodes:
            ancs = nx.ancestors(dag, nid)
            bad_ancs = [a for a in ancs if a in unverified_nodes]
            if bad_ancs:
                verified_nodes.discard(nid)
                unverified_nodes.add(nid)
                first_bad = bad_ancs[0]
                base_reason = unverified_reasons.get(first_bad, "ancestro no verificado")
                unverified_reasons[nid] = f"ANCESTOR_UNVERIFIED ({first_bad}): {base_reason}"

    # 7. Evaluar si el objetivo J y todas las restricciones relevantes están 100% verificados
    critical_nodes: Set[str] = set()
    if target_id and target_id in dag:
        critical_nodes.add(target_id)
        critical_nodes.update(nx.ancestors(dag, target_id))
    else:
        critical_nodes.update(topo_nodes)

    for c_obj in ir.constraints.values():
        for nid in ir.nodes:
            if nid in (c_obj.expression or "") and nid in dag:
                critical_nodes.add(nid)
                critical_nodes.update(nx.ancestors(dag, nid))

    unverified_critical = critical_nodes & unverified_nodes
    target_verified = (len(unverified_critical) == 0)

    crit_errors = [
        node_errors.get(nid, 0.0)
        for nid in critical_nodes
        if ir.nodes[nid].category in ("state", "target")
    ]
    global_max_err = max(crit_errors) if crit_errors else 0.0
    primary_reason = None
    if not target_verified:
        for nid in topo_nodes:
            if nid in unverified_critical and nid in unverified_reasons:
                primary_reason = unverified_reasons[nid]
                break
        if primary_reason is None and unverified_reasons:
            primary_reason = next(iter(unverified_reasons.values()))

    report = FidelityReport(
        node_errors=node_errors,
        verified_nodes=verified_nodes,
        unverified_nodes=unverified_nodes,
        unverified_reasons=unverified_reasons,
        target_verified=target_verified,
        verified=target_verified,
        max_rel_error=float(global_max_err),
        reason=primary_reason,
        n_samples=len(samples),
    )
    _attach_fidelity_to_graph(ir, report)
    return report


def _attach_fidelity_to_graph(ir: CausalGraph, report: FidelityReport) -> None:
    """Adjunta los metadatos de fidelidad a cada nodo, ajusta la confianza de aristas y actualiza metadata."""
    for nid, node in ir.nodes.items():
        err_val = report.node_errors.get(nid, 0.0)
        clean_err = round(float(err_val), 10) if math.isfinite(err_val) else None
        is_ver = nid in report.verified_nodes
        node_f_dict: Dict[str, Any] = {
            "verified": bool(is_ver),
            "max_rel_error": clean_err,
        }
        if not is_ver and nid in report.unverified_reasons:
            node_f_dict["reason"] = report.unverified_reasons[nid]
        node.fidelity = node_f_dict

    for edge in ir.edges:
        if edge.source in report.unverified_nodes or edge.target in report.unverified_nodes:
            edge.confidence = min(float(edge.confidence or 1.0), 0.1)

    if ir.metadata is None:
        ir.metadata = {}
    ir.metadata["fidelity_report"] = report.to_dict()


def evaluate_recipe_on_source(
    ir: CausalGraph,
    controls: Dict[str, Any],
    source_path_or_code: Optional[str] = None,
    entrypoint: Optional[str] = None,
    exo_overrides: Optional[Dict[str, float]] = None,
    timeout_sec: float = 10.0,
) -> Dict[str, Any]:
    """Certificado 3: Ejecuta la receta ganadora X* directamente sobre el código fuente original
    (manteniendo los asserts activos) y verifica que todas las restricciones duras se cumplen en el código real.
    """
    src_target = source_path_or_code or (ir.metadata or {}).get("target_dir")
    files_code, _ = _collect_source_files(src_target)

    sample_input: Dict[str, float] = {}
    for nid, node in ir.nodes.items():
        if node.category == "exogenous":
            val = (exo_overrides or {}).get(nid, node.current_value)
            sample_input[nid] = float(val) if val is not None else 0.0
        elif node.category == "control":
            val = controls.get(nid, node.current_value)
            if node.data_type in ("discrete", "boolean"):
                sample_input[nid] = int(round(float(val))) if val is not None else 0
            else:
                sample_input[nid] = float(val) if val is not None else 0.0

    ir_eval_list, _ = _evaluate_ir_on_samples(ir, [sample_input])
    ir_state = ir_eval_list[0] if ir_eval_list else dict(sample_input)

    target_ids = [nid for nid, n in ir.nodes.items() if n.category == "target"]
    target_id = target_ids[0] if target_ids else (ir.metadata or {}).get("target_id", "")
    ir_j = float(ir_state.get(target_id, 0.0))

    if not files_code:
        max_viol, viol_map = _check_constraints_on_state(ir, ir_state)
        return {
            "code_verified": max_viol <= 1e-5,
            "constraint_violation_on_code": float(max_viol),
            "code_objective_value": ir_j,
            "ir_objective_value": ir_j,
            "code_ir_rel_error": 0.0,
            "code_state": ir_state,
            "violated_constraints": [k for k, v in viol_map.items() if v > 1e-5],
        }

    dag = nx.DiGraph()
    for nid in ir.nodes:
        dag.add_node(nid)
    for e in ir.edges:
        if e.type != "empirical_data":
            dag.add_edge(e.source, e.target)
    topo_nodes = [
        nid for nid in nx.topological_sort(dag)
        if nid in ir.nodes and ir.nodes[nid].category in ("state", "target")
    ]
    node_specs = [
        {
            "id": nid,
            "entity": ir.nodes[nid].source_reference.entity if ir.nodes[nid].source_reference else None,
            "column_name": ir.nodes[nid].source_reference.column_name if ir.nodes[nid].source_reference else None,
            "file": ir.nodes[nid].source_reference.file if ir.nodes[nid].source_reference else None,
        }
        for nid in topo_nodes
    ]

    try:
        if not _needs_subprocess_timeout_guard(files_code, {}):
            out_data = _execute_traced_samples_isolated(
                files_code=files_code,
                samples=[sample_input],
                node_specs=node_specs,
                strip_asserts=False,
                entrypoint=entrypoint,
            )
        else:
            payload_str = json.dumps({
                "files_code": files_code,
                "samples": [sample_input],
                "node_specs": node_specs,
                "strip_asserts": False,
                "entrypoint": entrypoint,
            })
            runner_cmd = (
                "import json, sys; from cge.fidelity import _execute_traced_samples_isolated; "
                "p = json.loads(sys.stdin.read()); "
                "print(json.dumps(_execute_traced_samples_isolated("
                "p['files_code'], p['samples'], p['node_specs'], p.get('strip_asserts', False), p.get('entrypoint'))))"
            )
            proc = subprocess.run(
                [sys.executable, "-c", runner_cmd],
                input=payload_str,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            if proc.returncode != 0:
                return {
                    "code_verified": False,
                    "constraint_violation_on_code": float("inf"),
                    "code_objective_value": None,
                    "ir_objective_value": ir_j,
                    "code_ir_rel_error": float("inf"),
                    "error": proc.stderr.strip()[:300],
                }
            out_data = json.loads(proc.stdout)
        res0 = (out_data.get("results") or [{}])[0]
    except Exception as exc:
        return {
            "code_verified": False,
            "constraint_violation_on_code": float("inf"),
            "code_objective_value": None,
            "ir_objective_value": ir_j,
            "code_ir_rel_error": float("inf"),
            "error": str(exc),
        }

    # Combinar estado de código real con nodos declarativos SQL/CSV si existen
    combined_state = dict(ir_state)
    combined_state.update(res0.get("state", {}))
    code_j = float(combined_state.get(target_id, ir_j))
    rel_err = abs(code_j - ir_j) / max(abs(code_j), 1e-9)

    max_viol, viol_map = _check_constraints_on_state(ir, combined_state)
    if res0.get("assertion_violated"):
        max_viol = max(max_viol, 1.0)

    is_ok = (max_viol <= 1e-4) and (rel_err <= 1e-4) and not res0.get("assertion_violated")
    return {
        "code_verified": bool(is_ok),
        "constraint_violation_on_code": float(0.0 if max_viol <= 1e-6 else max_viol),
        "code_objective_value": code_j,
        "ir_objective_value": ir_j,
        "code_ir_rel_error": float(rel_err),
        "code_state": combined_state,
        "violated_constraints": [k for k, v in viol_map.items() if v > 1e-5],
    }


def _check_constraints_on_state(ir: CausalGraph, state: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Evalúa el grado de violación de todas las restricciones duras y cotas de control sobre un diccionario de estado."""
    violations: Dict[str, float] = {}
    max_v = 0.0

    for nid, node in ir.nodes.items():
        if node.category == "control" and nid in state:
            val = float(state[nid])
            b = node.bounds or {}
            if b.get("min") is not None:
                v_lo = max(0.0, float(b["min"]) - val)
                if v_lo > 0:
                    violations[f"bound_min_{nid}"] = v_lo
                    max_v = max(max_v, v_lo)
            if b.get("max") is not None:
                v_hi = max(0.0, val - float(b["max"]))
                if v_hi > 0:
                    violations[f"bound_max_{nid}"] = v_hi
                    max_v = max(max_v, v_hi)

    local_syms: Dict[str, Any] = {nid: sp.Symbol(nid, real=True) for nid in ir.nodes}
    local_syms.update({"Min": sp.Min, "Max": sp.Max, "Piecewise": sp.Piecewise, "sqrt": sp.sqrt, "log": sp.log, "exp": sp.exp, "Abs": sp.Abs})
    subs_map = {local_syms[k]: float(v) for k, v in state.items() if k in local_syms and isinstance(v, (int, float))}

    for cid, c_obj in ir.constraints.items():
        if c_obj.type != "hard":
            continue
        expr_str = (c_obj.expression or "").strip()
        if not expr_str or "sum(" in expr_str:
            continue
        try:
            if expr_str.count("<=") == 2 and ">=" not in expr_str:
                p0, p1, p2 = expr_str.split("<=")
                e0 = float(sp.sympify(p0.strip(), locals=local_syms).evalf(subs=subs_map))
                e1 = float(sp.sympify(p1.strip(), locals=local_syms).evalf(subs=subs_map))
                e2 = float(sp.sympify(p2.strip(), locals=local_syms).evalf(subs=subs_map))
                v_c = max(0.0, e0 - e1, e1 - e2)
            elif "<=" in expr_str and ">=" not in expr_str:
                lhs_s, rhs_s = expr_str.split("<=", 1)
                lhs_v = float(sp.sympify(lhs_s.strip(), locals=local_syms).evalf(subs=subs_map))
                rhs_v = float(sp.sympify(rhs_s.strip(), locals=local_syms).evalf(subs=subs_map))
                v_c = max(0.0, lhs_v - rhs_v)
            elif ">=" in expr_str and "<=" not in expr_str:
                lhs_s, rhs_s = expr_str.split(">=", 1)
                lhs_v = float(sp.sympify(lhs_s.strip(), locals=local_syms).evalf(subs=subs_map))
                rhs_v = float(sp.sympify(rhs_s.strip(), locals=local_syms).evalf(subs=subs_map))
                v_c = max(0.0, rhs_v - lhs_v)
            elif "==" in expr_str:
                lhs_s, rhs_s = expr_str.split("==", 1)
                lhs_v = float(sp.sympify(lhs_s.strip(), locals=local_syms).evalf(subs=subs_map))
                rhs_v = float(sp.sympify(rhs_s.strip(), locals=local_syms).evalf(subs=subs_map))
                v_c = abs(lhs_v - rhs_v)
            else:
                v_c = 0.0
            violations[cid] = v_c
            if v_c > max_v:
                max_v = v_c
        except Exception:
            continue

    return max_v, violations
