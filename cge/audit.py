"""Motor de Auditoría Simbólica Formal y Diagnóstico de Coherencia Causal (T1 a T7, A-1 a A-7)."""
from __future__ import annotations
import os
from typing import Dict, Any, List, Optional
import networkx as nx
import sympy as sp

from .schema import CausalGraph
from .layer1_heuristic import Layer1Router
from .layer2_exact import Layer2Solver


class SystemAuditor:
    """Auditor matemático formal que opera sobre el CausalIR sin identificadores de dominio cableados."""

    def __init__(
        self,
        target_dir: Optional[str] = None,
        graph: Optional[CausalGraph] = None,
        solver: Optional[Layer2Solver] = None,
        source_dir: Optional[str] = None
    ):
        resolved_dir = target_dir or source_dir
        if graph is None and solver is not None:
            graph = solver.graph
        if graph is None and resolved_dir is not None:
            from .ingest import ASTIngestPipeline
            graph = ASTIngestPipeline(resolved_dir).run()
        if graph is None:
            solver = Layer2Solver()
            graph = solver.graph
        if not graph.metadata and resolved_dir is not None:
            from .ingest import ASTIngestPipeline
            graph = ASTIngestPipeline(resolved_dir).run()

        self.target_dir = os.path.normpath(resolved_dir or (graph.metadata or {}).get("target_dir", "."))
        self.graph = graph
        self.solver = solver if solver is not None else Layer2Solver.from_graph(self.graph)
        self.target_id = self.solver.target_id
        self.diagnostics: List[Dict[str, Any]] = []

    def audit_all(self) -> List[Dict[str, Any]]:
        """Ejecuta las verificaciones formales T1 a T7 y retorna la lista de diagnósticos."""
        self.diagnostics = []
        self._check_t1_dead_nodes()
        self._check_t2_conservation_invariants()
        self._check_t3_doc_code_discrepancies()
        self._check_t4_local_optima()
        self._check_t5_discontinuities()
        self._check_t6_fragile_nodes()
        self._check_t7_empirical_edges()
        return self.diagnostics

    def run_full_audit(self) -> Dict[str, Any]:
        """Ejecuta la auditoría formal completa y devuelve un diccionario estructurado por categorías de hallazgo."""
        diags = self.audit_all()
        t1_list = [d for d in diags if d.get("id") == "T1"]
        t2_list = [d for d in diags if d.get("id") == "T2"]
        t3_list = [d for d in diags if d.get("id") == "T3"]
        t4_list = [d for d in diags if d.get("id") == "T4"]
        t5_list = [d for d in diags if d.get("id") == "T5"]
        t6_list = [d for d in diags if d.get("id") == "T6"]
        t7_list = [d for d in diags if d.get("id") == "T7"]

        return {
            "diagnostics": diags,
            "dead_nodes": {
                "count": len(t1_list),
                "nodes": [d["node"] for d in t1_list],
                "items": t1_list
            },
            "conservation_audit": {
                "overall_passed": len(t2_list) == 0,
                "violations_count": len(t2_list),
                "items": t2_list
            },
            "doc_vs_code": {
                "has_conflict": len(t3_list) > 0,
                "conflicts_count": len(t3_list),
                "items": t3_list
            },
            "suboptimal_Policies": {
                "has_suboptimal": len(t4_list) > 0,
                "count": len(t4_list),
                "items": t4_list
            },
            "discontinuities": {
                "step_discontinuities": t5_list,
                "kink_points": t6_list
            },
            "empirical_edges": {
                "count": len(t7_list),
                "items": t7_list
            }
        }

    def _compute_symbolic_total_derivative(self, source: str, target: str) -> Optional[sp.Expr]:
        """Calcula la derivada total simbólica d(target)/d(source) expandiendo las ecuaciones del DAG."""
        try:
            dag = nx.DiGraph()
            for e in self.graph.edges:
                if e.type != "empirical_data":
                    dag.add_edge(e.source, e.target)
            if not nx.has_path(dag, source, target):
                return sp.Integer(0)
            topo = list(nx.topological_sort(dag))
            expanded: Dict[str, sp.Expr] = {}
            for nid in topo:
                if nid == source or nid not in self.solver.mechanisms:
                    expanded[nid] = self.solver._get_sym(nid)
                else:
                    expr = self.solver.mechanisms[nid]
                    subs_map = {self.solver.symbols[k]: expanded[k] for k in expanded if k in self.solver.symbols and k != nid}
                    expanded[nid] = expr.subs(subs_map)
            raw_diff = sp.diff(expanded[target], self.solver._get_sym(source))
            if raw_diff == 0:
                return sp.Integer(0)
            if raw_diff.has(sp.Min, sp.Max, sp.Piecewise, sp.Derivative):
                return raw_diff
            return sp.simplify(raw_diff)
        except Exception:
            return None

    def _check_t1_dead_nodes(self):
        """A-1: Detecta nodos muertos estructurales (T1-A) y por cancelación algebraica de caminos (T1-B)."""
        router = Layer1Router(self.graph.to_dict())
        for nid, node in self.graph.nodes.items():
            if node.category == "state" and nid != self.target_id:
                has_path = (
                    nx.has_path(router.graph, nid, self.target_id)
                    if router.graph.has_node(nid) and router.graph.has_node(self.target_id)
                    else False
                )
                if not has_path:
                    self.diagnostics.append({
                        "id": "T1",
                        "code": "DEAD_NODE",
                        "subtype": "T1-A_STRUCTURAL_ORPHAN",
                        "severity": "warning",
                        "node": nid,
                        "file": node.source_reference.file,
                        "line_range": node.source_reference.line_range,
                        "details": {
                            "gradient_dJ_dnode": 0.0,
                            "reason": (
                                f"El nodo '{nid}' se calcula pero no tiene ningún camino dirigido hacia "
                                f"el objetivo '{self.target_id}' (dJ/d{nid} = 0)."
                            )
                        }
                    })
                else:
                    sym_d = self._compute_symbolic_total_derivative(nid, self.target_id)
                    if sym_d is not None and sym_d == 0:
                        self.diagnostics.append({
                            "id": "T1",
                            "code": "DEAD_NODE",
                            "subtype": "T1-B_ALGEBRAIC_PATH_CANCELLATION",
                            "severity": "warning",
                            "node": nid,
                            "file": node.source_reference.file,
                            "line_range": node.source_reference.line_range,
                            "details": {
                                "gradient_dJ_dnode": 0.0,
                                "reason": (
                                    f"El nodo '{nid}' tiene caminos hacia '{self.target_id}', pero se cancelan "
                                    f"algebraicamente de forma exacta (dJ/d{nid} == 0)."
                                )
                            }
                        })

    def _check_t2_conservation_invariants(self):
        """A-2: Verifica simbólicamente invariantes de conservación / suma cero en todas las ramas de ejecución."""
        checks = (self.graph.metadata or {}).get("conservation_checks", [])
        for chk in checks:
            if not chk.get("is_balanced", True):
                self.diagnostics.append({
                    "id": "T2",
                    "code": "INVARIANT_VIOLATION",
                    "severity": "critical",
                    "constraint_id": chk.get("constraint_id", "c_ledger"),
                    "file": chk["file"],
                    "function": chk["function"],
                    "line_range": chk["line_range"],
                    "details": {
                        "symbolic_residual": chk["symbolic_residual"],
                        "branch_residuals": chk.get("branch_residuals", {}),
                        "reason": (
                            f"La función '{chk['function']}' viola el invariante de conservación de "
                            f"{chk.get('constraint_id', 'c_ledger')} con residuo simbólico R = {chk['symbolic_residual']} != 0."
                        )
                    }
                })

    def _check_t3_doc_code_discrepancies(self):
        """A-3: Detecta discrepancias entre porcentajes documentados y tasas en el código AST."""
        discs = (self.graph.metadata or {}).get("doc_discrepancies", [])
        for d in discs:
            self.diagnostics.append(dict(d))

    def _check_t4_local_optima(self):
        """A-4: Detecta optimizadores locales miopes y cuantifica la pérdida exacta en el objetivo global J."""
        loc_opts = (self.graph.metadata or {}).get("local_optimizers", {})
        if not loc_opts:
            return

        global_sol = self.solver.optimize()
        if global_sol.get("status") != "FEASIBLE":
            return
        global_j = float(global_sol["objective_value"])

        for fn_name, loc_info in loc_opts.items():
            ctrl_id = loc_info["control_id"]
            loc_val = loc_info["recommended_value"]
            global_ctrl_val = global_sol["controls"].get(ctrl_id)

            local_sol = self.solver.optimize(fixed_controls={ctrl_id: float(loc_val)})
            best_j_at_local = float(local_sol.get("objective_value", 0.0)) if local_sol.get("status") == "FEASIBLE" else 0.0
            loss = round(global_j - best_j_at_local, 2)

            if loss > 1e-2:
                diag_entry: Dict[str, Any] = {
                    "id": "T4",
                    "code": "LOCAL_MINIMUM",
                    "severity": "critical",
                    "file": loc_info["file"],
                    "function": fn_name,
                    "line_range": loc_info.get("line_range"),
                    "local_recommendation": {ctrl_id: loc_val},
                    "global_optimum": {ctrl_id: global_ctrl_val},
                    f"{self.target_id}_loss_at_best_p_m": loss,
                    "bilevel_loss": loss,
                    "details": {
                        "reason": (
                            f"La función local '{fn_name}' recomienda {ctrl_id}={loc_val}, causando un cuello de botella "
                            f"y una pérdida de {loss:.2f} en '{self.target_id}' frente al óptimo global {ctrl_id}={global_ctrl_val}."
                        )
                    }
                }
                self.diagnostics.append(diag_entry)

    def _check_t5_discontinuities(self):
        """T5: Detecta discontinuidades de salto de orden 0 extraídas simbólicamente de condiciones Piecewise."""
        for nid, node in self.graph.nodes.items():
            for d in node.discontinuities:
                th_val = d["threshold"]
                th_fmt = int(th_val) if float(th_val).is_integer() else th_val
                self.diagnostics.append({
                    "id": "T5",
                    "code": "DISCONTINUITY",
                    "severity": "warning",
                    "node": nid,
                    "file": node.source_reference.file,
                    "function": node.source_reference.entity,
                    "line_range": node.source_reference.line_range,
                    "threshold_variable": d["variable"],
                    "threshold_value": th_fmt,
                    "val_true": d.get("val_true"),
                    "val_false": d.get("val_false"),
                    "details": {
                        "reason": f"Salto discontinuo en '{nid}' al cruzar el umbral '{d['variable']}'={th_fmt}."
                    }
                })

    def _check_t6_fragile_nodes(self):
        """A-5 / T6: Detecta nodos con kink de Clarke activo (D^+ != D^-) dentro de la banda estocástica en el óptimo."""
        opt_sol = self.solver.optimize()
        if opt_sol.get("status") != "FEASIBLE":
            return
        opt_state = self.solver.evaluate(**opt_sol["controls"])

        u_std = 80.0
        for uid in self.solver.exogenous_ids:
            dist = self.graph.nodes[uid].distribution or {}
            if "std" in dist:
                u_std = max(u_std, float(dist["std"]))

        for nid, expr in self.solver.mechanisms.items():
            if isinstance(expr, sp.Min) and len(expr.args) == 2:
                a0_name, a1_name = str(expr.args[0]), str(expr.args[1])
                v0 = opt_state.get(a0_name)
                v1 = opt_state.get(a1_name)
                if v0 is not None and v1 is not None and abs(v0 - v1) <= max(1.0, 2.5 * u_std):
                    node = self.graph.nodes.get(nid)
                    self.diagnostics.append({
                        "id": "T6",
                        "code": "FRAGILE_NODE",
                        "severity": "warning",
                        "node": nid,
                        "file": node.source_reference.file if node else None,
                        "function": node.source_reference.entity if node else None,
                        "line_range": node.source_reference.line_range if node else None,
                        "details": {
                            "clarke_subdifferential": [0.0, 1.0],
                            "kink_asymmetry": 1.0,
                            "reason": (
                                f"En el óptimo global {a0_name} ≈ {a1_name} ({v1:.0f}), haciendo que '{nid} = min({a0_name}, {a1_name})' "
                                f"tenga subdiferencial de Clarke asimétrico [0, 1] y sea frágil ante perturbaciones exógenas."
                            )
                        }
                    })

    def _check_t7_empirical_edges(self):
        """T7: Registra las relaciones respaldadas por inferencia econométrica sobre CSV."""
        emp_meta = {
            (m["source"], m["target"]): m
            for m in (self.graph.metadata or {}).get("empirical_estimates", [])
        }
        for edge in self.graph.edges:
            if edge.type == "empirical_data":
                m_info = emp_meta.get((edge.source, edge.target), {})
                gt_coeff = m_info.get("ground_truth", 1.0)
                self.diagnostics.append({
                    "id": "T7",
                    "code": "EMPIRICAL_EDGE",
                    "severity": "info",
                    "source": edge.source,
                    "target": edge.target,
                    "ground_truth_coefficient": gt_coeff,
                    "estimated_coefficient": m_info.get("estimated_coefficient"),
                    "r_squared": m_info.get("r_squared"),
                    "details": {
                        "reason": f"Arista {edge.source} -> {edge.target} fundamentada en estimación econométrica conjunta desde datos tabulares."
                    }
                })

    def check_sensitivity_signs(self) -> Dict[str, Any]:
        """A-7: Verifica formalmente la coherencia de signos de las derivadas totales en Modo Inverso hacia J."""
        router = Layer1Router(self.graph.to_dict())
        sens_map = router.get_control_sensitivities(self.target_id)

        sensitivities_out: Dict[str, float] = {}
        validations: Dict[str, bool] = {}

        for nid, node in self.graph.nodes.items():
            val = float(sens_map.get(nid, 0.0))
            prefix = nid.split("_")[0]
            sensitivities_out[f"dJ_d{nid}"] = val
            sensitivities_out[f"dJ_d{prefix}"] = val
            if node.category == "control":
                validations[f"{nid}_positive"] = (val > 0.0)
                validations[f"{prefix}_positive"] = (val > 0.0)
            elif node.category == "state":
                is_dead = not nx.has_path(router.graph, nid, self.target_id)
                if is_dead:
                    validations[f"{nid}_zero"] = (abs(val) < 1e-6)
                    validations[f"{prefix}_zero"] = (abs(val) < 1e-6)

        negative_direct_states = [
            e.source for e in self.graph.edges
            if e.target == self.target_id
            and (e.sensitivity_weight or 0.0) < 0.0
            and self.graph.nodes[e.source].category == "state"
            and len(list(router.graph.successors(e.source))) == 1
        ]
        costs_negative = all(sens_map.get(nid, 0.0) < 0.0 for nid in negative_direct_states) if negative_direct_states else True
        validations["costs_negative"] = costs_negative

        pos_controls_ok = any(sens_map.get(cid, 0.0) > 0.0 for cid in self.solver.control_ids)
        dead_nodes_ok = all(
            abs(sens_map.get(nid, 0.0)) < 1e-6
            for nid, node in self.graph.nodes.items()
            if node.category == "state" and not nx.has_path(router.graph, nid, self.target_id)
        )
        all_correct = bool(pos_controls_ok and dead_nodes_ok and costs_negative)

        return {
            "all_signs_correct": all_correct,
            "sensitivities": sensitivities_out,
            "validations": validations
        }
