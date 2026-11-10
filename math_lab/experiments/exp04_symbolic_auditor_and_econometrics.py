"""Experimento 04: Auditoría Simbólica Formal (T1-T6) e Inferencia Econométrica Estructural Conjunta (T7).

Supera las heurísticas sintácticas de cge/audit.py y la trampa de cge/empirical.py mediante:
1. Ejecución simbólica de invariantes de suma cero (SymbolicLedger con SymPy),
2. Detección algebraica de cancelación de caminos (violación de fidelidad T1-B) y saturación de régimen (T1-C),
3. Cálculo exacto del Precio de la Anarquía Bilevel (T4) y Fragilidad de Clarke (T6),
4. Estimación econométrica conjunta de (alpha, beta) y selección de base funcional por BIC desde CSV (T7).
"""
from __future__ import annotations
import ast
import csv
import math
import os
from typing import Dict, Any, List, Optional
import networkx as nx
import numpy as np
import sympy as sp

from .exp01_symbolic_scm_and_routing import SymbolicSCM, ASTFunctionToSymPy
from .exp02_universal_minlp_solver import UniversalLayer2Solver


class SymbolicLedger:
    """Estado contable simbólico para verificar leyes de conservación sum(credits) - sum(debits) == 0
    sobre expresiones algebraicas arbitrarias (sin contar llamadas por nombre).
    """

    def __init__(self):
        self.credits: List[sp.Expr] = []
        self.debits: List[sp.Expr] = []

    def record_credit(self, amount: sp.Expr):
        self.credits.append(sp.sympify(amount))

    def record_debit(self, amount: sp.Expr):
        self.debits.append(sp.sympify(amount))

    def conservation_residual(self) -> sp.Expr:
        """Retorna R = simplify(sum(credits) - sum(debits)). Si R == 0, el invariante se cumple."""
        total_c = sum(self.credits, sp.Integer(0))
        total_d = sum(self.debits, sp.Integer(0))
        return sp.simplify(total_c - total_d)


def _exec_ledger_stmts(
    stmts: List[ast.stmt],
    env: Dict[str, sp.Expr],
    credits: List[sp.Expr],
    debits: List[sp.Expr]
) -> Tuple[List[Tuple[str, sp.Expr]], int, int]:
    """Ejecuta simbólicamente una lista de sentencias AST sobre un Ledger ramificado (soportando if/else)."""
    compiler = ASTFunctionToSymPy({}, {})
    compiler.env = dict(env)
    c_count = 0
    d_count = 0

    for idx, stmt in enumerate(stmts):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            compiler.env[stmt.targets[0].id] = compiler.eval_expr(stmt.value)
        elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
            var_name = stmt.target.id
            prev = compiler.env.get(var_name, sp.Integer(0))
            rhs = compiler.eval_expr(stmt.value)
            if isinstance(stmt.op, ast.Add):
                compiler.env[var_name] = prev + rhs
            elif isinstance(stmt.op, ast.Sub):
                compiler.env[var_name] = prev - rhs
            elif isinstance(stmt.op, ast.Mult):
                compiler.env[var_name] = prev * rhs
            elif isinstance(stmt.op, ast.Div):
                compiler.env[var_name] = prev / rhs
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            attr_name = getattr(call.func, "attr", "")
            if attr_name in ("record_credit", "record_debit") and len(call.args) >= 1:
                arg_expr = compiler.eval_expr(call.args[0])
                if attr_name == "record_credit":
                    credits.append(sp.sympify(arg_expr))
                    c_count += 1
                else:
                    debits.append(sp.sympify(arg_expr))
                    d_count += 1
        elif isinstance(stmt, ast.If):
            cond_str = ast.unparse(stmt.test) if hasattr(ast, "unparse") else "cond"
            rest = stmts[idx + 1:]
            t_res, tc1, td1 = _exec_ledger_stmts(list(stmt.body) + rest, dict(compiler.env), list(credits), list(debits))
            f_res, tc2, td2 = _exec_ledger_stmts(list(stmt.orelse) + rest, dict(compiler.env), list(credits), list(debits))
            branches = [(f"{cond_str} -> {b_name}", r) for b_name, r in t_res] + [(f"not({cond_str}) -> {b_name}", r) for b_name, r in f_res]
            return branches, c_count + tc1 + tc2, d_count + td1 + td2

    total_c = sum(credits, sp.Integer(0))
    total_d = sum(debits, sp.Integer(0))
    res = sp.simplify(total_c - total_d)
    return [("main", res)], c_count, d_count


def verify_ledger_function_symbolically(code_str: str, fn_name: str) -> Dict[str, Any]:
    """Ejecuta simbólicamente una función que opera sobre un objeto Ledger en AST (incluyendo ramas if/else)
    y verifica si conserva el invariante de suma cero para todo monto simbólico `amount > 0`.
    """
    tree = ast.parse(code_str)
    target_fn: Optional[ast.FunctionDef] = None
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == fn_name:
            target_fn = stmt
            break
    if target_fn is None:
        raise ValueError(f"Función {fn_name} no encontrada en el AST.")

    param_syms: Dict[str, sp.Expr] = {}
    for arg in target_fn.args.args:
        if arg.arg != "ledger":
            param_syms[arg.arg] = sp.Symbol(arg.arg, positive=True, real=True)

    credits_list: List[sp.Expr] = []
    debits_list: List[sp.Expr] = []
    branch_residuals, credit_call_count, debit_call_count = _exec_ledger_stmts(
        target_fn.body, param_syms, credits_list, debits_list
    )

    all_branches_balanced = all((r == 0 or sp.simplify(r) == 0) for _, r in branch_residuals)
    primary_residual = branch_residuals[0][1] if len(branch_residuals) == 1 else next(
        (r for _, r in branch_residuals if r != 0), sp.Integer(0)
    )
    naive_call_count_balanced = (credit_call_count == debit_call_count)

    return {
        "function": fn_name,
        "credit_calls": credit_call_count,
        "debit_calls": debit_call_count,
        "symbolic_credits": str(sum(credits_list, sp.Integer(0))),
        "symbolic_debits": str(sum(debits_list, sp.Integer(0))),
        "symbolic_residual": str(primary_residual),
        "branch_residuals": {b: str(r) for b, r in branch_residuals},
        "is_balanced_symbolically": bool(all_branches_balanced),
        "naive_auditor_says_balanced": bool(naive_call_count_balanced),
        "line_range": [target_fn.lineno, getattr(target_fn, "end_lineno", target_fn.lineno)]
    }


def demonstrate_t2_symbolic_vs_naive_counterexamples() -> Dict[str, Any]:
    """Demuestra tres casos donde el auditor sintáctico de cge/audit.py falla (falso positivo y falsos negativos)
    mientras que el verificador simbólico acierta con exactitud matemática (incluyendo ramas if/else).
    """
    sample_code = '''
def valid_split_vat_transaction(ledger, amount):
    """VÁLIDA: divide el débito en base imponible (80%) + IVA (20%). Suma = 1.0 * amount."""
    ledger.record_credit(amount, "total_invoice")
    ledger.record_debit(0.80 * amount, "net_base")
    ledger.record_debit(0.20 * amount, "vat_20pct")

def buggy_scaled_debit_transaction(ledger, amount):
    """INVÁLIDA: 1 llamada a credit y 1 llamada a debit, pero por el doble del importe."""
    ledger.record_credit(amount, "credit_entry")
    ledger.record_debit(2.0 * amount, "double_amount_bug")

def conditional_branch_leak_transaction(ledger, amount, fee_mode):
    """INVÁLIDA EN UNA RAMA: balanceada cuando fee_mode <= 0, pero con fuga de 0.05*amount cuando fee_mode > 0."""
    if fee_mode > 0:
        ledger.record_credit(amount, "credit_branch")
        ledger.record_debit(1.05 * amount, "unbalanced_fee_branch")
    else:
        ledger.record_credit(amount, "credit_ok")
        ledger.record_debit(amount, "debit_ok")
'''
    res_split = verify_ledger_function_symbolically(sample_code, "valid_split_vat_transaction")
    res_scaled = verify_ledger_function_symbolically(sample_code, "buggy_scaled_debit_transaction")
    res_branch = verify_ledger_function_symbolically(sample_code, "conditional_branch_leak_transaction")

    with open("golden_case/finance.py", "r", encoding="utf-8") as f:
        golden_fin_code = f.read()
    res_golden = verify_ledger_function_symbolically(golden_fin_code, "process_refund")

    return {
        "split_vat_valid_case": res_split,
        "scaled_debit_bug_case": res_scaled,
        "conditional_branch_leak_case": res_branch,
        "golden_case_process_refund": res_golden
    }


def detect_algebraic_path_cancellation_demo() -> Dict[str, Any]:
    """Demuestra la detección simbólica de Nodos Muertos por Cancelación de Caminos (T1-B: Violación de Fidelidad),
    donde nx.has_path(G, 'v_cancel', 'J') es True pero dJ/dv_cancel == 0 idénticamente.
    """
    x = sp.Symbol("x", real=True)
    v_cancel = sp.Symbol("v_cancel", real=True)
    m1_expr = 2 * v_cancel + x
    m2_expr = -2 * v_cancel + 3 * x
    j_expr = m1_expr + m2_expr

    G = nx.DiGraph()
    G.add_edges_from([("x", "m1"), ("x", "m2"), ("v_cancel", "m1"), ("v_cancel", "m2"), ("m1", "J"), ("m2", "J")])

    has_topological_path = nx.has_path(G, "v_cancel", "J")
    total_derivative_expr = sp.simplify(
        sp.diff(j_expr, sp.Symbol("m1")) * sp.diff(m1_expr, v_cancel)
        + sp.diff(j_expr, sp.Symbol("m2")) * sp.diff(m2_expr, v_cancel)
    )
    is_faithfulness_dead_node = (total_derivative_expr == 0)

    return {
        "node": "v_cancel",
        "has_topological_path_nx": has_topological_path,
        "symbolic_total_derivative_dJ_dv": str(total_derivative_expr),
        "detected_as_dead_by_topology_only": not has_topological_path,
        "detected_as_dead_by_symbolic_jacobian": bool(is_faithfulness_dead_node)
    }


class FormalSymbolicAuditor:
    """Auditor matemático formal sobre SymbolicSCM (cubriendo T1-A, T1-B, T1-C, T2, T3, T4, T5, T6 y T7)."""

    def __init__(self, target_dir: str):
        self.target_dir = os.path.normpath(target_dir)
        self.scm = SymbolicSCM(self.target_dir)
        self.solver = UniversalLayer2Solver(self.scm)

    def run_full_formal_audit(self) -> Dict[str, Any]:
        opt_sol = self.solver.optimize()
        opt_controls = opt_sol["controls"]
        opt_inputs = {
            "price": opt_controls["price"],
            "marketing_budget": opt_controls["marketing_budget"],
            "order_qty": float(opt_controls["order_qty"]),
            "base_demand": float(self.scm.node_meta["base_demand"]["current_value"])
        }

        nom_ad = self.scm.compute_reverse_mode_ad_jacobian()
        opt_ad = self.scm.compute_reverse_mode_ad_jacobian(opt_inputs)

        diagnostics = []

        # T1: Nodos muertos (T1-A Huérfano Estructural, T1-B Cancelación Algebraica de Caminos, T1-C Saturación de Régimen)
        for nid, meta in self.scm.node_meta.items():
            if meta["category"] == "state":
                has_path = nx.has_path(self.scm.dag, nid, "profit")
                adj_nom = nom_ad["adjoints_dJ_dv"].get(nid, 0.0)
                adj_opt = opt_ad["adjoints_dJ_dv"].get(nid, 0.0)
                if not has_path:
                    diagnostics.append({
                        "id": "T1",
                        "subtype": "T1-A_STRUCTURAL_ORPHAN",
                        "node": nid,
                        "dJ_dnode_nominal": adj_nom,
                        "dJ_dnode_optimal": adj_opt,
                        "file": meta["file"],
                        "line_range": meta["line_range"]
                    })
                elif abs(adj_nom) < 1e-9 and abs(adj_opt) < 1e-9:
                    sym_deriv = self.scm.compute_symbolic_total_derivative(nid, "profit")
                    if sym_deriv == 0:
                        diagnostics.append({
                            "id": "T1-B",
                            "subtype": "T1-B_FAITHFULNESS_PATH_CANCELLATION",
                            "node": nid,
                            "symbolic_total_derivative": str(sym_deriv),
                            "dJ_dnode_nominal": adj_nom,
                            "dJ_dnode_optimal": adj_opt,
                            "file": meta["file"],
                            "line_range": meta["line_range"],
                            "explanation": f"El nodo '{nid}' tiene caminos dirigidos hacia 'profit', pero sus efectos se cancelan algebraicamente de forma exacta (dJ/d{nid} == 0)."
                        })
                elif abs(adj_nom) < 1e-9 and abs(adj_opt) > 1e-9:
                    diagnostics.append({
                        "id": "T1-C",
                        "subtype": "REGIME_SATURATED_AT_NOMINAL",
                        "node": nid,
                        "dJ_dnode_nominal": adj_nom,
                        "dJ_dnode_optimal": adj_opt,
                        "explanation": f"En el estado nominal X0, '{nid}' tiene derivada parcial nula por saturación de capacidad, pero en el óptimo se activa ({adj_opt:.4f})."
                    })

        # T2: Verificación simbólica de Ledger
        fin_path = os.path.join(self.target_dir, "finance.py")
        if os.path.exists(fin_path):
            with open(fin_path, "r", encoding="utf-8") as f:
                fin_code = f.read()
            fin_tree = ast.parse(fin_code)
            for stmt in fin_tree.body:
                if isinstance(stmt, ast.FunctionDef) and any(a.arg == "ledger" for a in stmt.args.args):
                    ver = verify_ledger_function_symbolically(fin_code, stmt.name)
                    if not ver["is_balanced_symbolically"]:
                        diagnostics.append({
                            "id": "T2",
                            "function": stmt.name,
                            "symbolic_residual": ver["symbolic_residual"],
                            "line_range": ver["line_range"]
                        })

        # T3: Discrepancia simbólica entre especificación documental (README.md) y derivada simbólica df_fees/drevenue
        readme_path = os.path.join(self.target_dir, "README.md")
        if os.path.exists(readme_path) and "fees" in self.scm.mechanisms and "revenue" in self.scm.symbols:
            import re
            with open(readme_path, "r", encoding="utf-8") as f:
                readme_txt = f.read()
            m_pct = re.search(r"(\d+(?:\.\d+)?)%\s*(?:sobre|en|de|\s|$)", readme_txt)
            if m_pct:
                doc_rate = float(m_pct.group(1)) / 100.0
                code_rate = float(sp.diff(self.scm.mechanisms["fees"], self.scm.symbols["revenue"]).evalf())
                if abs(doc_rate - code_rate) > 1e-5:
                    diagnostics.append({
                        "id": "T3",
                        "node": "fees",
                        "doc_value": round(doc_rate, 4),
                        "code_value": round(code_rate, 4),
                        "relative_discrepancy": round(abs(code_rate - doc_rate) / max(doc_rate, 1e-9), 4)
                    })

        # T4: Pérdida Bilevel / Precio de la Anarquía para optimizadores locales extraídos del AST
        for fn_name, loc_info in self.scm.local_optimizers.items():
            q_loc = int(round(loc_info["recommended_value"]))
            constrained_sol = self.solver.optimize(fixed_controls={"order_qty": float(q_loc)})
            bilevel_loss = round(opt_sol["profit"] - constrained_sol["profit"], 4)
            diagnostics.append({
                "id": "T4",
                "function": fn_name,
                "local_order_qty": q_loc,
                "global_optimal_order_qty": opt_controls["order_qty"],
                "max_profit_at_local_q": constrained_sol["profit"],
                "global_optimal_profit": opt_sol["profit"],
                "bilevel_profit_loss": bilevel_loss
            })

        # T5: Discontinuidades Piecewise extraídas simbólicamente
        for nid, disc_list in self.scm.discontinuities.items():
            for d in disc_list:
                diagnostics.append({
                    "id": "T5",
                    "node": nid,
                    "threshold_variable": d["variable"],
                    "threshold_value": d["threshold"],
                    "jump_delta": round(float(d["val_false"]) - float(d["val_true"]), 4)
                })

        # T6: Fragilidad de Clarke (asimetría de subgradiente D^+ != D^- en el punto óptimo)
        fragile_nodes_detected = set()
        for (u, v), c_rep in opt_ad["clarke_reports"].items():
            if c_rep["is_kink_active"] and v not in fragile_nodes_detected:
                fragile_nodes_detected.add(v)
                diagnostics.append({
                    "id": "T6",
                    "node": v,
                    "trigger_parent": u,
                    "clarke_subdifferential": c_rep["clarke_subdifferential"],
                    "kink_asymmetry": c_rep["kink_asymmetry"]
                })

        # T7: Inferencia Econométrica Estructural Conjunta si existe CSV histórico
        csv_path = os.path.join(self.target_dir, "data", "sales_history.csv")
        if os.path.exists(csv_path):
            est = JointStructuralEconometricEstimator.estimate_from_csv(csv_path)
            diagnostics.append({
                "id": "T7",
                "source": "marketing_budget",
                "target": "demand",
                "selected_basis": est["selected_functional_basis_by_bic"],
                "estimated_alpha": est["estimated_alpha_price_elasticity"],
                "estimated_beta": est["estimated_beta_marketing_lift"],
                "condition_number": est["design_matrix_condition_number"],
                "derived_confidence": est["derived_edge_confidence"]
            })

        return {
            "optimal_solution": opt_sol,
            "diagnostics": diagnostics
        }


class JointStructuralEconometricEstimator:
    """Estimador econométrico conjunto (T7) que estima simultáneamente la elasticidad de precio (alpha)
    y el coeficiente de marketing (beta) + selecciona la forma funcional óptima mediante BIC desde el CSV,
    SIN usar ninguna constante hardcodeada del código fuente.
    """

    @staticmethod
    def estimate_from_csv(csv_path: str = "golden_case/data/sales_history.csv") -> Dict[str, Any]:
        d0_list, p_list, m_list, d_list = [], [], [], []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d0_list.append(float(row["base_demand"]))
                p_list.append(float(row["price"]))
                m_list.append(float(row["marketing_budget"]))
                d_list.append(float(row["demand"]))

        D0 = np.array(d0_list)
        P = np.array(p_list)
        M = np.array(m_list)
        Y = np.array(d_list)
        n = len(Y)

        # 1. Selección de modelo estructural por Criterio de Información Bayesiano (BIC)
        # Comparamos 3 hipótesis para el mecanismo de marketing f(m): sqrt(m), lineal m, log(1+m)
        # Modelo estructural general: D = theta_1 * D0 + theta_2 * (D0 * P) + beta * g(M)
        basis_candidates = {
            "sqrt_m": np.sqrt(np.maximum(0.0, M)),
            "linear_m": M,
            "log1p_m": np.log1p(np.maximum(0.0, M))
        }

        bic_scores: Dict[str, float] = {}
        fits: Dict[str, Any] = {}

        for b_name, g_m in basis_candidates.items():
            X_mat = np.column_stack([D0, D0 * P, g_m])
            theta_hat, _, _, _ = np.linalg.lstsq(X_mat, Y, rcond=None)
            y_pred = X_mat @ theta_hat
            residuals = Y - y_pred
            rss = float(np.sum(residuals ** 2))
            sigma2 = max(rss / n, 1e-12)
            k_params = X_mat.shape[1]
            bic = n * math.log(sigma2) + k_params * math.log(n)
            bic_scores[b_name] = round(bic, 2)
            fits[b_name] = (X_mat, theta_hat, residuals, rss)

        best_basis = min(bic_scores, key=bic_scores.get)
        X_best, theta_best, res_best, rss_best = fits[best_basis]

        # Recuperación de parámetros estructurales sin usar 0.15 ni 3.0 hardcodeados:
        # Si D = D0 * (1 - alpha * (P - 3)) + beta * sqrt(M) = (1 + 3*alpha)*D0 - alpha*(D0*P) + beta*sqrt(M)
        # También podemos estimar directamente sobre (D - D0) = -alpha * D0 * (P - P_ref) + beta * sqrt(M)
        # Del ajuste conjunto libre X_best = [D0, D0*P, sqrt(M)]:
        theta_1, theta_2, beta_hat = float(theta_best[0]), float(theta_best[1]), float(theta_best[2])
        alpha_hat = -theta_2
        implied_p_ref = (theta_1 - 1.0) / max(alpha_hat, 1e-9)

        # Diagnóstico de identificabilidad: Número de condición de la matriz de diseño estandarizada
        X_norm = X_best / np.linalg.norm(X_best, axis=0, keepdims=True)
        cond_number = float(np.linalg.cond(X_norm))

        # Matriz de covarianza robusta HC3 (MacKinnon-White) para errores estándar e intervalos al 95%
        xtx_inv = np.linalg.inv(X_best.T @ X_best)
        hat_diag = np.einsum("ij,jk,ik->i", X_best, xtx_inv, X_best)
        hc3_weights = (res_best / np.maximum(1.0 - hat_diag, 1e-4)) ** 2
        meat = (X_best * hc3_weights[:, None]).T @ X_best
        cov_hc3 = xtx_inv @ meat @ xtx_inv
        se_vec = np.sqrt(np.maximum(0.0, np.diag(cov_hc3)))

        se_alpha = float(se_vec[1])
        se_beta = float(se_vec[2])

        tss = float(np.sum((Y - np.mean(Y)) ** 2))
        r_squared = 1.0 - (rss_best / max(tss, 1e-12))
        derived_confidence = float(r_squared * max(0.0, 1.0 - se_beta / max(abs(beta_hat), 1e-9)))

        return {
            "sample_size": n,
            "selected_functional_basis_by_bic": best_basis,
            "bic_comparison": bic_scores,
            "estimated_alpha_price_elasticity": round(alpha_hat, 5),
            "alpha_95pct_ci": [round(alpha_hat - 1.96 * se_alpha, 5), round(alpha_hat + 1.96 * se_alpha, 5)],
            "ground_truth_alpha": 0.15,
            "estimated_beta_marketing_lift": round(beta_hat, 5),
            "beta_95pct_ci": [round(beta_hat - 1.96 * se_beta, 5), round(beta_hat + 1.96 * se_beta, 5)],
            "ground_truth_beta": 8.0,
            "implied_reference_price": round(implied_p_ref, 4),
            "design_matrix_condition_number": round(cond_number, 2),
            "is_structurally_identifiable": cond_number < 100.0,
            "r_squared": round(r_squared, 6),
            "derived_edge_confidence": round(derived_confidence, 4)
        }
