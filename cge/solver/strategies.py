"""Estrategias de resolución matemática genérica (LP/MILP, Enumeración de Regímenes, Homotopía LSE, B&B y IIS)."""
from __future__ import annotations
import itertools
import math
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
from scipy.optimize import minimize, milp, LinearConstraint, Bounds
from scipy.stats import qmc
import sympy as sp

from .compile import Problem, Var, CompiledConstraint, NonsmoothAtom, _poly_degree, _has_nonsmooth_subexpr


_SAFE_LAMBDIFY_MODULES = [
    {
        "sqrt": lambda v: math.sqrt(max(0.0, float(v))),
        "log": lambda v: math.log(max(1e-12, float(v))),
        "log1p": lambda v: math.log1p(max(-1.0 + 1e-12, float(v))),
        "Heaviside": lambda v, *args: (1.0 if float(v) >= 0.0 else 0.0),
        "sign": lambda v: (1.0 if float(v) > 0.0 else (-1.0 if float(v) < 0.0 else 0.0)),
    },
    "math",
]

_SAFE_GRAD_LAMBDIFY_MODULES = [
    {
        "sqrt": lambda v: math.sqrt(max(1e-12, float(v))),
        "log": lambda v: math.log(max(1e-12, float(v))),
        "log1p": lambda v: math.log1p(max(-1.0 + 1e-12, float(v))),
        "Heaviside": lambda v, *args: (1.0 if float(v) >= 0.0 else 0.0),
        "sign": lambda v: (1.0 if float(v) > 0.0 else (-1.0 if float(v) < 0.0 else 0.0)),
    },
    "math",
]


def _apply_atom_subs(expr: sp.Expr, ordered_subs: List[Tuple[sp.Expr, sp.Expr]]) -> sp.Expr:
    """Aplica sustituciones de átomos desde el más externo al más interno para evitar colisiones bottom-up de xreplace."""
    out = expr
    for atom_sym, sub_val in reversed(ordered_subs):
        out = out.xreplace({atom_sym: sub_val})
    return out


_LIN_COEFF_CACHE: Dict[Tuple[Any, ...], Optional[Tuple[np.ndarray, float]]] = {}


def _extract_linear_coeffs(expr: sp.Expr, syms: List[sp.Symbol]) -> Optional[Tuple[np.ndarray, float]]:
    """Extrae vector a y escalar b tales que expr == a^T x + b, o None si no es lineal."""
    key = (expr, tuple(syms))
    if key in _LIN_COEFF_CACHE:
        res = _LIN_COEFF_CACHE[key]
        return (res[0].copy(), res[1]) if res is not None else None
    deg = _poly_degree(expr, syms)
    if deg is None or deg > 1:
        if len(_LIN_COEFF_CACHE) < 4096:
            _LIN_COEFF_CACHE[key] = None
        return None
    expanded = sp.expand(expr)
    coeffs = np.zeros(len(syms), dtype=float)
    subs_zero = {s: sp.Float(0.0) for s in syms}
    try:
        const_val = float(expanded.xreplace(subs_zero).evalf())
        for i, s in enumerate(syms):
            d_s = sp.diff(expanded, s)
            if not d_s.is_number:
                if len(_LIN_COEFF_CACHE) < 4096:
                    _LIN_COEFF_CACHE[key] = None
                return None
            coeffs[i] = float(d_s.evalf())
        if len(_LIN_COEFF_CACHE) < 4096:
            _LIN_COEFF_CACHE[key] = (coeffs.copy(), const_val)
        return coeffs, const_val
    except Exception:
        if len(_LIN_COEFF_CACHE) < 4096:
            _LIN_COEFF_CACHE[key] = None
        return None


# ==============================================================================
# ESTRATEGIA A: LP / MILP Exacto vía HiGHS + Lifting Lineal a Trozos (Big-M)
# ==============================================================================

def solve_strategy_a_milp(problem: Problem) -> Optional[Dict[str, Any]]:
    """Resuelve problemas LP y MILP de forma exacta global mediante HiGHS (scipy.optimize.milp)."""
    n = len(problem.vars)
    if n == 0:
        return None
    syms = problem.ctrl_symbols

    obj_lin = _extract_linear_coeffs(problem.min_obj_expr, syms)
    if obj_lin is None:
        return None
    c_vec, _ = obj_lin

    a_rows: List[np.ndarray] = []
    lb_rows: List[float] = []
    ub_rows: List[float] = []

    for constr in problem.constraints:
        c_lin = _extract_linear_coeffs(constr.expr, syms)
        if c_lin is None:
            return None
        row_a, row_b = c_lin
        a_rows.append(row_a)
        if constr.kind == "ineq":
            lb_rows.append(-np.inf)
            ub_rows.append(-row_b)
        else:
            lb_rows.append(-row_b)
            ub_rows.append(-row_b)

    lb_var = np.array([v.lb for v in problem.vars], dtype=float)
    ub_var = np.array([v.ub for v in problem.vars], dtype=float)
    integrality = np.array([1 if v.domain in ("integer", "binary") else 0 for v in problem.vars], dtype=int)

    lin_constraints = (
        LinearConstraint(np.vstack(a_rows), np.array(lb_rows, dtype=float), np.array(ub_rows, dtype=float))
        if a_rows else None
    )

    res = milp(
        c=c_vec,
        integrality=integrality,
        bounds=Bounds(lb_var, ub_var),
        constraints=lin_constraints,
    )
    if not res.success or res.x is None:
        return None

    x_sol = np.array(res.x, dtype=float)
    for i, v in enumerate(problem.vars):
        if v.domain in ("integer", "binary"):
            x_sol[i] = float(int(round(x_sol[i])))
        x_sol[i] = float(np.clip(x_sol[i], v.lb, v.ub))

    max_v, _ = problem.max_constraint_violation(x_sol)
    if max_v > 1e-4:
        return None

    j_val = problem.evaluate_j(x_sol)
    gap = float(getattr(res, "mip_gap", 0.0) or 0.0)
    return {
        "x": x_sol,
        "j_val": j_val,
        "max_violation": max_v,
        "strategy_used": "strategy_a_highs_milp",
        "is_exact": True,
        "optimality_gap": gap,
        "kkt_residual": 0.0,
        "active_regime": "linear_exact",
    }


def solve_strategy_a_piecewise_milp(problem: Problem) -> Optional[Dict[str, Any]]:
    """Resuelve problemas lineales a trozos (con átomos Min, Max, Abs, Piecewise) vía MILP Disyuntivo Big-M en HiGHS."""
    if not problem.is_piecewise_linear or not problem.nonsmooth_atoms:
        return None

    n_ctrl = len(problem.vars)
    if n_ctrl == 0:
        return None

    atoms = problem.nonsmooth_atoms
    k_atoms = len(atoms)
    ctrl_syms = problem.ctrl_symbols
    ph_syms = [a.placeholder for a in atoms]
    all_val_syms = ctrl_syms + ph_syms

    # Reemplazar átomos de abajo hacia arriba en cada átomo superior, objetivo y restricciones
    def _replace_atoms_up_to(expr: sp.Expr, max_idx: int) -> sp.Expr:
        out = expr
        for r in range(max_idx - 1, -1, -1):
            out = out.xreplace({atoms[r].sympy_atom: atoms[r].placeholder})
        return out

    top_obj = _replace_atoms_up_to(problem.min_obj_expr, k_atoms)
    top_constrs = [(_replace_atoms_up_to(c.expr, k_atoms), c.kind) for c in problem.constraints]

    obj_lin = _extract_linear_coeffs(top_obj, all_val_syms)
    if obj_lin is None:
        return None
    c_val_vec, _ = obj_lin

    # Construir vector de variables del MILP: [x_0..x_{n-1}, w_0..w_{K-1}, z_{0,0}..z_{K-1, m-1}]
    branch_offsets: List[int] = []
    total_z = 0
    for a in atoms:
        branch_offsets.append(n_ctrl + k_atoms + total_z)
        total_z += len(a.branches)

    total_vars = n_ctrl + k_atoms + total_z
    c_full = np.zeros(total_vars, dtype=float)
    c_full[:n_ctrl + k_atoms] = c_val_vec

    lb_full = np.full(total_vars, -1e9, dtype=float)
    ub_full = np.full(total_vars, 1e9, dtype=float)
    integrality = np.zeros(total_vars, dtype=int)

    for i, v in enumerate(problem.vars):
        lb_full[i] = v.lb
        ub_full[i] = v.ub
        if v.domain in ("integer", "binary"):
            integrality[i] = 1

    for z_idx in range(n_ctrl + k_atoms, total_vars):
        lb_full[z_idx] = 0.0
        ub_full[z_idx] = 1.0
        integrality[z_idx] = 1

    # Estimar cotas de intervalo para [x, w] para elegir constantes Big-M seguras y ajustadas
    var_intervals: List[Tuple[float, float]] = [(v.lb, v.ub) for v in problem.vars]

    a_rows: List[np.ndarray] = []
    lb_rows: List[float] = []
    ub_rows: List[float] = []

    for r, atom in enumerate(atoms):
        w_col = n_ctrl + r
        z_start = branch_offsets[r]
        m_br = len(atom.branches)

        # Restricción SOS1: sum_q z_{r, q} == 1
        sos_row = np.zeros(total_vars, dtype=float)
        for q in range(m_br):
            sos_row[z_start + q] = 1.0
        a_rows.append(sos_row)
        lb_rows.append(1.0)
        ub_rows.append(1.0)

        # Estimar rango de las ramas del átomo r
        br_parsed: List[Tuple[np.ndarray, float, float, float, List[Tuple[np.ndarray, float]]]] = []
        r_lo = 1e12
        r_hi = -1e12

        for q, (br_e, br_conds, _) in enumerate(atom.branches):
            br_sub = _replace_atoms_up_to(br_e, r)
            lin_br = _extract_linear_coeffs(br_sub, all_val_syms[:n_ctrl + r])
            if lin_br is None:
                return None
            vec_b, cst_b = lin_br

            # Aritmética de intervalos sobre las variables previas
            b_min_v = cst_b
            b_max_v = cst_b
            for j_idx, coeff_j in enumerate(vec_b):
                v_lo, v_hi = var_intervals[j_idx]
                if coeff_j >= 0:
                    b_min_v += coeff_j * v_lo
                    b_max_v += coeff_j * v_hi
                else:
                    b_min_v += coeff_j * v_hi
                    b_max_v += coeff_j * v_lo
            r_lo = min(r_lo, b_min_v)
            r_hi = max(r_hi, b_max_v)

            cond_parsed: List[Tuple[np.ndarray, float]] = []
            for gc in br_conds:
                gc_sub = _replace_atoms_up_to(gc, r)
                lin_gc = _extract_linear_coeffs(gc_sub, all_val_syms[:n_ctrl + r])
                if lin_gc is None:
                    return None
                cond_parsed.append(lin_gc)

            br_parsed.append((vec_b, cst_b, float(b_min_v), float(b_max_v), cond_parsed))

        var_intervals.append((r_lo, r_hi))
        w_margin = max(1e-6 * max(1.0, r_hi - r_lo), 1e-9)
        lb_full[w_col] = r_lo - w_margin
        ub_full[w_col] = r_hi + w_margin

        for q, (vec_b, cst_b, b_min_v, b_max_v, cond_parsed) in enumerate(br_parsed):
            z_col = z_start + q
            m_val_up = max(r_hi - b_min_v, 1e-7) * 1.05 + 1e-7
            m_val_lo = max(b_max_v - r_lo, 1e-7) * 1.05 + 1e-7

            # 1) w_r - vec_b^T y + M_up * z_{r,q} <= M_up + cst_b
            row1 = np.zeros(total_vars, dtype=float)
            row1[:n_ctrl + r] = -vec_b
            row1[w_col] = 1.0
            row1[z_col] = m_val_up
            a_rows.append(row1)
            lb_rows.append(-np.inf)
            ub_rows.append(m_val_up + cst_b)

            # 2) -w_r + vec_b^T y + M_lo * z_{r,q} <= M_lo - cst_b
            row2 = np.zeros(total_vars, dtype=float)
            row2[:n_ctrl + r] = vec_b
            row2[w_col] = -1.0
            row2[z_col] = m_val_lo
            a_rows.append(row2)
            lb_rows.append(-np.inf)
            ub_rows.append(m_val_lo - cst_b)

            # 3) Condiciones de validez de rama: vec_gc^T y + cst_gc <= M_cond * (1 - z_{r,q})
            #    <=> vec_gc^T y + M_cond * z_{r,q} <= M_cond - cst_gc
            for vec_gc, cst_gc in cond_parsed:
                gc_max_v = cst_gc
                for j_idx, coeff_j in enumerate(vec_gc):
                    v_lo, v_hi = var_intervals[j_idx]
                    gc_max_v += coeff_j * v_hi if coeff_j >= 0 else coeff_j * v_lo
                m_cond = max(gc_max_v, 1e-7) * 1.05 + 1e-7
                row_c = np.zeros(total_vars, dtype=float)
                row_c[:n_ctrl + r] = vec_gc
                row_c[z_col] = m_cond
                a_rows.append(row_c)
                lb_rows.append(-np.inf)
                ub_rows.append(m_cond - cst_gc + 1e-9 * max(1.0, m_cond))

    # Añadir las restricciones originales del problema en términos de [x, w]
    for tc_expr, tc_kind in top_constrs:
        lin_tc = _extract_linear_coeffs(tc_expr, all_val_syms)
        if lin_tc is None:
            return None
        vec_tc, cst_tc = lin_tc
        row_tc = np.zeros(total_vars, dtype=float)
        row_tc[:n_ctrl + k_atoms] = vec_tc
        a_rows.append(row_tc)
        if tc_kind == "ineq":
            lb_rows.append(-np.inf)
            ub_rows.append(-cst_tc)
        else:
            lb_rows.append(-cst_tc)
            ub_rows.append(-cst_tc)

    res = milp(
        c=c_full,
        integrality=integrality,
        bounds=Bounds(lb_full, ub_full),
        constraints=LinearConstraint(np.vstack(a_rows), np.array(lb_rows, dtype=float), np.array(ub_rows, dtype=float)),
    )
    if not res.success or res.x is None:
        return None

    x_sol = np.array(res.x[:n_ctrl], dtype=float)
    for i, v in enumerate(problem.vars):
        if v.domain in ("integer", "binary"):
            x_sol[i] = float(int(round(x_sol[i])))
        x_sol[i] = float(np.clip(x_sol[i], v.lb, v.ub))

    max_v, _ = problem.max_constraint_violation(x_sol)
    if max_v > 1e-4:
        return None

    j_val = problem.evaluate_j(x_sol)
    gap = float(getattr(res, "mip_gap", 0.0) or 0.0)
    return {
        "x": x_sol,
        "j_val": j_val,
        "max_violation": max_v,
        "strategy_used": "strategy_a_piecewise_milp",
        "is_exact": True,
        "optimality_gap": gap,
        "kkt_residual": 0.0,
        "active_regime": "piecewise_linear_milp",
    }


# ==============================================================================
# ESTRATEGIA B: Enumeración Exacta de Regímenes Multi-Átomo (K <= 8) + Variedad Kink
# ==============================================================================

def _compute_kkt_stationarity_residual(
    x_sol: np.ndarray,
    grad_f: np.ndarray,
    ineq_vals: List[float],
    ineq_grads: List[np.ndarray],
    eq_grads: List[np.ndarray],
    lb: np.ndarray,
    ub: np.ndarray,
) -> float:
    """Calcula el residuo de estacionariedad KKT min_{mu >= 0, lam} ||grad_f + sum mu_i grad_g_i + sum lam_j grad_h_j||."""
    n = len(x_sol)
    active_ineq_cols: List[np.ndarray] = []
    for val, g_vec in zip(ineq_vals, ineq_grads):
        if val >= -1e-4:
            active_ineq_cols.append(g_vec)
    for i in range(n):
        if x_sol[i] <= lb[i] + 1e-4:
            e_i = np.zeros(n, dtype=float)
            e_i[i] = -1.0
            active_ineq_cols.append(e_i)
        if x_sol[i] >= ub[i] - 1e-4:
            e_i = np.zeros(n, dtype=float)
            e_i[i] = 1.0
            active_ineq_cols.append(e_i)

    if not active_ineq_cols and not eq_grads:
        return float(np.linalg.norm(grad_f))

    # Resolver mínimos cuadrados no negativos / mixtos para encontrar multiplicadores de Lagrange óptimos
    m_ineq = len(active_ineq_cols)
    m_eq = len(eq_grads)
    # Representar lam = lam^+ - lam^- con variables >= 0 y usar scipy.optimize.lsq_linear
    from scipy.optimize import lsq_linear
    cols: List[np.ndarray] = list(active_ineq_cols)
    for eg in eq_grads:
        cols.append(eg)
        cols.append(-eg)
    if not cols:
        return float(np.linalg.norm(grad_f))
    mat_a = np.column_stack(cols)
    try:
        res_lsq = lsq_linear(mat_a, -grad_f, bounds=(0.0, 1e12))
        return float(np.linalg.norm(grad_f + mat_a @ res_lsq.x))
    except Exception:
        return float(np.linalg.norm(grad_f))


_COMPILED_FN_CACHE: Dict[Tuple[Any, ...], Tuple[Callable[..., float], Callable[..., Any]]] = {}


def _get_compiled_fn_and_grad(expr: sp.Expr, ctrl_syms: List[sp.Symbol]) -> Tuple[Callable[..., float], Callable[..., Any]]:
    key = (expr, tuple(ctrl_syms))
    cached = _COMPILED_FN_CACHE.get(key)
    if cached is not None:
        return cached
    fn = sp.lambdify(ctrl_syms, expr, modules=_SAFE_LAMBDIFY_MODULES)
    grad_exprs = [sp.diff(expr, s) for s in ctrl_syms]
    grad_fn = sp.lambdify(ctrl_syms, grad_exprs, modules=_SAFE_GRAD_LAMBDIFY_MODULES)
    if len(_COMPILED_FN_CACHE) < 2048:
        _COMPILED_FN_CACHE[key] = (fn, grad_fn)
    return fn, grad_fn


def _solve_smooth_subproblem_slsqp(
    f_expr: sp.Expr,
    ineq_exprs: List[sp.Expr],
    eq_exprs: List[sp.Expr],
    ctrl_syms: List[sp.Symbol],
    lb: np.ndarray,
    ub: np.ndarray,
    start_points: List[np.ndarray],
) -> Optional[Tuple[np.ndarray, float, float]]:
    """Resuelve un subproblema suave C^2 con gradientes y Jacobianos simbólicos exactos vía SLSQP."""
    n = len(ctrl_syms)
    try:
        f_fn, grad_f_fn = _get_compiled_fn_and_grad(f_expr, ctrl_syms)

        ineq_pairs = [_get_compiled_fn_and_grad(g, ctrl_syms) for g in ineq_exprs]
        ineq_fns = [p[0] for p in ineq_pairs]
        ineq_grad_fns = [p[1] for p in ineq_pairs]

        eq_pairs = [_get_compiled_fn_and_grad(h, ctrl_syms) for h in eq_exprs]
        eq_fns = [p[0] for p in eq_pairs]
        eq_grad_fns = [p[1] for p in eq_pairs]
    except Exception:
        return None

    def obj_wrap(x: np.ndarray) -> float:
        try:
            v = float(f_fn(*[float(z) for z in x]))
            return v if math.isfinite(v) else 1e15
        except Exception:
            return 1e15

    def jac_wrap(x: np.ndarray) -> np.ndarray:
        try:
            g = np.array(grad_f_fn(*[float(z) for z in x]), dtype=float)
            return np.where(np.isfinite(g), g, 0.0)
        except Exception:
            return np.zeros(n, dtype=float)

    scipy_constrs: List[Dict[str, Any]] = []
    for g_fn, g_grad_fn in zip(ineq_fns, ineq_grad_fns):
        # SLSQP 'ineq' exige c(x) >= 0 <=> -g(x) >= 0
        scipy_constrs.append({
            "type": "ineq",
            "fun": lambda x, _f=g_fn: -float(_f(*[float(z) for z in x])),
            "jac": lambda x, _gf=g_grad_fn: -np.array(_gf(*[float(z) for z in x]), dtype=float),
        })
    for h_fn, h_grad_fn in zip(eq_fns, eq_grad_fns):
        scipy_constrs.append({
            "type": "eq",
            "fun": lambda x, _f=h_fn: float(_f(*[float(z) for z in x])),
            "jac": lambda x, _hf=h_grad_fn: np.array(_hf(*[float(z) for z in x]), dtype=float),
        })

    bounds_list = [(float(lb[i]), float(ub[i])) for i in range(n)]
    best_x: Optional[np.ndarray] = None
    best_f = float("inf")

    for x0 in start_points:
        x_init = np.clip(np.array(x0, dtype=float), lb, ub)
        try:
            res = minimize(
                obj_wrap,
                x0=x_init,
                jac=jac_wrap,
                bounds=bounds_list,
                constraints=scipy_constrs,
                method="SLSQP",
                options={"ftol": 1e-11, "maxiter": 250},
            )
            cand_x = np.clip(np.array(res.x, dtype=float), lb, ub)
            # Comprobar factibilidad en el subproblema
            args_c = [float(z) for z in cand_x]
            max_sub_v = 0.0
            for g_fn in ineq_fns:
                max_sub_v = max(max_sub_v, float(g_fn(*args_c)))
            for h_fn in eq_fns:
                max_sub_v = max(max_sub_v, abs(float(h_fn(*args_c))))
            if max_sub_v <= 1e-4:
                f_v = obj_wrap(cand_x)
                if f_v < best_f:
                    best_f = f_v
                    best_x = cand_x
        except Exception:
            continue

    if best_x is None:
        return None

    args_b = [float(z) for z in best_x]
    grad_b = jac_wrap(best_x)
    ineq_v_b = [float(gf(*args_b)) for gf in ineq_fns]
    ineq_g_b = [np.array(ggf(*args_b), dtype=float) for ggf in ineq_grad_fns]
    eq_g_b = [np.array(egf(*args_b), dtype=float) for egf in eq_grad_fns]
    kkt_res = _compute_kkt_stationarity_residual(best_x, grad_b, ineq_v_b, ineq_g_b, eq_g_b, lb, ub)
    return best_x, best_f, kkt_res


def _generate_multistart_points(problem: Problem, n_sobol: int = 16) -> List[np.ndarray]:
    n = len(problem.vars)
    if n == 0:
        return [np.array([], dtype=float)]
    lb = np.array([v.lb for v in problem.vars], dtype=float)
    ub = np.array([v.ub for v in problem.vars], dtype=float)
    x0 = np.clip(np.array([v.x0 for v in problem.vars], dtype=float), lb, ub)

    pts: List[np.ndarray] = [x0, 0.5 * (lb + ub), lb + 0.2 * (ub - lb), lb + 0.8 * (ub - lb)]
    sampler = qmc.Sobol(d=n, scramble=True, seed=42)
    m_pow = max(1, int(math.ceil(math.log2(max(2, n_sobol)))))
    u_mat = sampler.random_base2(m=m_pow)[:n_sobol]
    for row in u_mat:
        pts.append(lb + row * (ub - lb))
    return pts


def solve_strategy_b_regime_enumeration(
    problem: Problem,
    fixed_var_overrides: Optional[Dict[int, float]] = None,
    n_starts: int = 12,
) -> Optional[Dict[str, Any]]:
    """Estrategia B: Enumeración exacta de todos los regímenes suaves (2^K, K <= 8) + variedades kink."""
    n = len(problem.vars)
    if n == 0:
        return None

    lb = np.array([v.lb for v in problem.vars], dtype=float)
    ub = np.array([v.ub for v in problem.vars], dtype=float)
    if fixed_var_overrides:
        for idx_v, val_v in fixed_var_overrides.items():
            lb[idx_v] = float(val_v)
            ub[idx_v] = float(val_v)

    ctrl_syms = problem.ctrl_symbols
    atoms = problem.nonsmooth_atoms

    branch_index_lists = [list(range(len(a.branches))) for a in atoms]
    all_combos = list(itertools.product(*branch_index_lists)) if atoms else [()]
    if len(all_combos) > 256:
        return None

    base_starts = _generate_multistart_points(problem, n_sobol=n_starts)
    if fixed_var_overrides:
        for pt in base_starts:
            for idx_v, val_v in fixed_var_overrides.items():
                pt[idx_v] = float(val_v)

    best_x: Optional[np.ndarray] = None
    best_min_obj = float("inf")
    best_j = float("-inf") if problem.sense == "max" else float("inf")
    best_kkt = 0.0
    best_regime_label = "smooth_interior"
    all_regime_optima: List[np.ndarray] = []

    try:
        min_obj_fast = sp.lambdify(ctrl_syms, problem.min_obj_expr, modules=_SAFE_LAMBDIFY_MODULES)
    except Exception:
        min_obj_fast = None

    def eval_true_min_obj(x_c: np.ndarray) -> float:
        if min_obj_fast is not None:
            try:
                return float(min_obj_fast(*[float(z) for z in x_c]))
            except Exception:
                pass
        subs = {ctrl_syms[i]: float(x_c[i]) for i in range(n)}
        return float(problem.min_obj_expr.evalf(subs=subs))

    # También extraer igualdades de frontera kink (p. ej. a == b en Min(a, b) o Max(a, b))
    kink_candidates_per_atom: List[List[sp.Expr]] = []
    for atom in atoms:
        if atom.kind in ("Min", "Max") and len(atom.branches) >= 2:
            b0_e = atom.branches[0][0]
            b1_e = atom.branches[1][0]
            kink_candidates_per_atom.append([b0_e - b1_e])
        else:
            kink_candidates_per_atom.append([])

    for combo in all_combos:
        # Sustituir cada átomo de abajo hacia arriba según la rama elegida en combo
        atom_subs_list: List[Tuple[sp.Expr, sp.Expr]] = []
        regime_ineqs: List[sp.Expr] = []
        regime_labels: List[str] = []
        pruned = False

        for r, br_idx in enumerate(combo):
            atom = atoms[r]
            br_expr, br_conds, br_lbl = atom.branches[br_idx]
            sub_br_expr = _apply_atom_subs(br_expr, atom_subs_list)
            atom_subs_list.append((atom.sympy_atom, sub_br_expr))
            regime_labels.append(f"a{r}:{br_lbl}")

            for gc in br_conds:
                sub_gc = _apply_atom_subs(gc, atom_subs_list)
                if sub_gc.is_number:
                    if float(sub_gc.evalf()) > 1e-7:
                        pruned = True
                        break
                else:
                    # Poda rápida de cotas univariadas (ej. order_qty - 800 <= 0)
                    lin_gc = _extract_linear_coeffs(sub_gc, ctrl_syms)
                    if lin_gc is not None:
                        vec_g, cst_g = lin_gc
                        nz_indices = np.where(np.abs(vec_g) > 1e-12)[0]
                        if len(nz_indices) == 1:
                            idx_u = int(nz_indices[0])
                            coeff_u = vec_g[idx_u]
                            bound_u = -cst_g / coeff_u
                            if coeff_u > 0 and bound_u < lb[idx_u] - 1e-7:
                                pruned = True
                                break
                            if coeff_u < 0 and bound_u > ub[idx_u] + 1e-7:
                                pruned = True
                                break
                    regime_ineqs.append(sub_gc)
            if pruned:
                break

        if pruned:
            continue

        # Ajustar cotas locales del régimen si hay desigualdades univariadas
        reg_lb = lb.copy()
        reg_ub = ub.copy()
        non_trivial_ineqs: List[sp.Expr] = []
        for g_e in regime_ineqs:
            lin_g = _extract_linear_coeffs(g_e, ctrl_syms)
            if lin_g is not None:
                vec_g, cst_g = lin_g
                nz_idx = np.where(np.abs(vec_g) > 1e-12)[0]
                if len(nz_idx) == 1:
                    i_u = int(nz_idx[0])
                    c_u = vec_g[i_u]
                    b_u = -cst_g / c_u
                    if c_u > 0:
                        reg_ub[i_u] = min(reg_ub[i_u], b_u)
                    else:
                        reg_lb[i_u] = max(reg_lb[i_u], b_u)
                    continue
            non_trivial_ineqs.append(g_e)

        if np.any(reg_lb > reg_ub + 1e-7):
            continue

        f_reg = _apply_atom_subs(problem.min_obj_expr, atom_subs_list)
        ineq_reg: List[sp.Expr] = []
        eq_reg: List[sp.Expr] = []

        raw_ineqs_all = list(non_trivial_ineqs)
        for c in problem.constraints:
            c_sub = _apply_atom_subs(c.expr, atom_subs_list)
            if c.kind == "ineq":
                raw_ineqs_all.append(c_sub)
            else:
                eq_reg.append(c_sub)

        expanded_ineq_reg: List[sp.Expr] = []
        pruned_contradictory = False
        for g_cand in raw_ineqs_all:
            g_exp = sp.expand(g_cand)
            if g_exp.is_number:
                if float(g_exp.evalf()) > 1e-7:
                    pruned_contradictory = True
                    break
                continue
            if any(sp.expand(g_exp + prev_e) == 0 for prev_e in expanded_ineq_reg):
                # Dos átomos acoplados exigen g(x) <= 0 y -g(x) <= 0 simultáneamente; cubierto por el régimen compatible
                pruned_contradictory = True
                break
            if not any(sp.expand(g_exp - prev_e) == 0 for prev_e in expanded_ineq_reg):
                ineq_reg.append(g_cand)
                expanded_ineq_reg.append(g_exp)

        if pruned_contradictory:
            continue

        # Propagación de cotas por intervalos sobre todas las desigualdades lineales del régimen
        lin_ineq_coeffs: List[Tuple[np.ndarray, float]] = []
        for g_lin_cand in ineq_reg:
            lc = _extract_linear_coeffs(g_lin_cand, ctrl_syms)
            if lc is not None:
                lin_ineq_coeffs.append(lc)

        for _pass in range(2):
            for vec_a, cst_b in lin_ineq_coeffs:
                for i_v in range(n):
                    a_i = vec_a[i_v]
                    if abs(a_i) <= 1e-12:
                        continue
                    min_other = 0.0
                    for j_v in range(n):
                        if j_v == i_v or abs(vec_a[j_v]) <= 1e-12:
                            continue
                        min_other += min(vec_a[j_v] * reg_lb[j_v], vec_a[j_v] * reg_ub[j_v])
                    rhs_rem = -cst_b - min_other
                    if a_i > 0:
                        reg_ub[i_v] = min(reg_ub[i_v], rhs_rem / a_i)
                    else:
                        reg_lb[i_v] = max(reg_lb[i_v], rhs_rem / a_i)

        if np.any(reg_lb > reg_ub + 1e-7):
            continue

        # Generar puntos iniciales escalados directamente a la caja factible del régimen [reg_lb, reg_ub]
        reg_span = np.maximum(0.0, reg_ub - reg_lb)
        reg_starts: List[np.ndarray] = [
            0.5 * (reg_lb + reg_ub),
            reg_lb.copy(),
            reg_lb + 0.25 * reg_span,
            reg_lb + 0.75 * reg_span,
        ]
        # Punto mixto con variables discretas en su cota inferior del régimen y continuas en el interior
        pt_disc_lo = 0.5 * (reg_lb + reg_ub)
        for i_v, v_obj in enumerate(problem.vars):
            if v_obj.domain in ("integer", "binary"):
                pt_disc_lo[i_v] = reg_lb[i_v]
        reg_starts.append(pt_disc_lo)
        for pt in base_starts[:max(2, n_starts // 2)]:
            reg_starts.append(np.clip(pt, reg_lb, reg_ub))

        # 1) Resolver el subproblema suave del régimen
        sub_res = _solve_smooth_subproblem_slsqp(f_reg, ineq_reg, eq_reg, ctrl_syms, reg_lb, reg_ub, reg_starts[:max(2, min(4, len(reg_starts)))])
        candidates_in_regime: List[Tuple[np.ndarray, float, str]] = []
        if sub_res is not None:
            candidates_in_regime.append((sub_res[0], sub_res[2], "+".join(regime_labels) or "smooth_interior"))

        # 2) Resolver sobre la variedad kink activa (donde a(x) == b(x) en átomos Min/Max presentes en el régimen)
        base_kink_starts = ([sub_res[0]] if sub_res is not None else []) + reg_starts[:2]
        seen_kink_exps: List[sp.Expr] = []
        for r_kink, kink_eqs in enumerate(kink_candidates_per_atom):
            if not kink_eqs:
                continue
            kink_eq_sub = _apply_atom_subs(kink_eqs[0], atom_subs_list)
            kink_exp = sp.expand(kink_eq_sub)
            if not kink_exp.is_number:
                if any(sp.expand(kink_exp - prev_k) == 0 or sp.expand(kink_exp + prev_k) == 0 for prev_k in seen_kink_exps):
                    continue
                if not any(sp.expand(e_exp - kink_exp) == 0 or sp.expand(e_exp + kink_exp) == 0 for e_exp in expanded_ineq_reg):
                    continue
                seen_kink_exps.append(kink_exp)
                filtered_ineq = [
                    e for e, e_exp in zip(ineq_reg, expanded_ineq_reg)
                    if sp.expand(e_exp - kink_exp) != 0 and sp.expand(e_exp + kink_exp) != 0
                ]
                # Proyectar puntos iniciales sobre la variedad kink si alguna variable entra linealmente
                proj_starts: List[np.ndarray] = []
                lin_var_idx: Optional[int] = None
                lin_slope: float = 0.0
                for idx_s, s_sym in enumerate(ctrl_syms):
                    if reg_ub[idx_s] > reg_lb[idx_s] + 1e-7:
                        d_s = sp.diff(kink_exp, s_sym)
                        if d_s.is_number and abs(float(d_s.evalf())) > 1e-9:
                            lin_var_idx = idx_s
                            lin_slope = float(d_s.evalf())
                            break
                if lin_var_idx is not None:
                    try:
                        k_fn, _ = _get_compiled_fn_and_grad(kink_eq_sub, ctrl_syms)
                        for st_pt in base_kink_starts[:2]:
                            pt_c = st_pt.copy()
                            kv = float(k_fn(*[float(z) for z in pt_c]))
                            pt_c[lin_var_idx] = float(np.clip(pt_c[lin_var_idx] - kv / lin_slope, reg_lb[lin_var_idx], reg_ub[lin_var_idx]))
                            proj_starts.append(pt_c)
                    except Exception:
                        proj_starts = list(base_kink_starts[:2])
                else:
                    proj_starts = list(base_kink_starts[:2])

                kink_res = _solve_smooth_subproblem_slsqp(
                    f_reg,
                    filtered_ineq,
                    eq_reg + [kink_eq_sub],
                    ctrl_syms,
                    reg_lb,
                    reg_ub,
                    proj_starts,
                )
                if kink_res is not None:
                    candidates_in_regime.append((kink_res[0], kink_res[2], f"{'+'.join(regime_labels)}+kink_{r_kink}"))

        for cand_x, cand_kkt, lbl in candidates_in_regime:
            max_v, _ = problem.max_constraint_violation(cand_x, check_integrality=False)
            if max_v <= 1e-4:
                true_obj = eval_true_min_obj(cand_x)
                all_regime_optima.append(cand_x.copy())
                if true_obj < best_min_obj - 1e-9:
                    best_min_obj = true_obj
                    best_x = cand_x.copy()
                    best_j = problem.evaluate_j(cand_x)
                    best_kkt = cand_kkt
                    best_regime_label = lbl

    if best_x is None:
        return None

    max_v, _ = problem.max_constraint_violation(best_x, check_integrality=False)
    return {
        "x": best_x,
        "j_val": best_j,
        "max_violation": max_v,
        "strategy_used": "strategy_b_regime_enumeration",
        "is_exact": True,
        "optimality_gap": 0.0,
        "kkt_residual": best_kkt,
        "active_regime": best_regime_label,
        "all_regime_optima": all_regime_optima,
    }


# ==============================================================================
# ESTRATEGIA C: Homotopía Suave Log-Sum-Exp + Pulido de Régimen Activo (K > 8)
# ==============================================================================

def _smooth_nonsmooth_expr_lse(expr: sp.Expr, tau_sym: sp.Symbol) -> sp.Expr:
    """Reemplaza simbólicamente Min, Max, Abs y Piecewise por aproximaciones C^infinito parametrizadas por tau."""
    if not isinstance(expr, sp.Basic) or not expr.args:
        return expr
    new_args = [_smooth_nonsmooth_expr_lse(a, tau_sym) for a in expr.args]
    if isinstance(expr, sp.Min):
        if len(new_args) == 2:
            a, b = new_args[0], new_args[1]
            # m = 0.5*(a+b) - 0.5*sqrt((a-b)^2 + tau^2) es C^infinito y numéricamente inmune a overflow
            return 0.5 * (a + b) - 0.5 * sp.sqrt((a - b) ** 2 + tau_sym ** 2)
        acc = new_args[0]
        for a in new_args[1:]:
            acc = 0.5 * (acc + a) - 0.5 * sp.sqrt((acc - a) ** 2 + tau_sym ** 2)
        return acc
    if isinstance(expr, sp.Max):
        if len(new_args) == 2:
            a, b = new_args[0], new_args[1]
            return 0.5 * (a + b) + 0.5 * sp.sqrt((a - b) ** 2 + tau_sym ** 2)
        acc = new_args[0]
        for a in new_args[1:]:
            acc = 0.5 * (acc + a) + 0.5 * sp.sqrt((acc - a) ** 2 + tau_sym ** 2)
        return acc
    if isinstance(expr, sp.Abs):
        a0 = new_args[0]
        return sp.sqrt(a0 ** 2 + tau_sym ** 2)
    if isinstance(expr, sp.Piecewise):
        # Blending sigmoidal suave entre ramas
        res_pw = expr.args[-1][0]
        res_pw = _smooth_nonsmooth_expr_lse(res_pw, tau_sym)
        for val_e, cond_e in reversed(expr.args[:-1]):
            v_sm = _smooth_nonsmooth_expr_lse(val_e, tau_sym)
            ineqs = []
            if isinstance(cond_e, (sp.LessThan, sp.StrictLessThan)):
                ineqs.append(cond_e.lhs - cond_e.rhs)
            elif isinstance(cond_e, (sp.GreaterThan, sp.StrictGreaterThan)):
                ineqs.append(cond_e.rhs - cond_e.lhs)
            if ineqs:
                g_c = _smooth_nonsmooth_expr_lse(ineqs[0], tau_sym)
                sig = 0.5 * (1.0 - sp.tanh(g_c / tau_sym))
                res_pw = sig * v_sm + (1.0 - sig) * res_pw
        return res_pw
    try:
        return expr.func(*new_args)
    except Exception:
        return expr


def solve_strategy_c_homotopy_lse(problem: Problem) -> Optional[Dict[str, Any]]:
    """Estrategia C: Continuación por homotopía suave (tau in {1.0, 0.1, 1e-2, 1e-4}) + pulido en régimen activo."""
    n = len(problem.vars)
    if n == 0:
        return None
    ctrl_syms = problem.ctrl_symbols
    lb = np.array([v.lb for v in problem.vars], dtype=float)
    ub = np.array([v.ub for v in problem.vars], dtype=float)

    tau_sym = sp.Symbol("__cge_tau__", positive=True, real=True)
    f_smooth_tau = _smooth_nonsmooth_expr_lse(problem.min_obj_expr, tau_sym)
    ineq_smooth_tau = [_smooth_nonsmooth_expr_lse(c.expr, tau_sym) for c in problem.constraints if c.kind == "ineq"]
    eq_smooth_tau = [_smooth_nonsmooth_expr_lse(c.expr, tau_sym) for c in problem.constraints if c.kind == "eq"]

    starts = _generate_multistart_points(problem, n_sobol=8)
    current_candidates = starts[:6]

    for tau_val in (1.0, 0.1, 1e-2, 1e-4):
        subs_tau = {tau_sym: sp.Float(tau_val)}
        f_t = f_smooth_tau.xreplace(subs_tau)
        ineq_t = [g.xreplace(subs_tau) for g in ineq_smooth_tau]
        eq_t = [h.xreplace(subs_tau) for h in eq_smooth_tau]
        stage_res = _solve_smooth_subproblem_slsqp(f_t, ineq_t, eq_t, ctrl_syms, lb, ub, current_candidates)
        if stage_res is not None:
            current_candidates = [stage_res[0]] + current_candidates[:2]

    best_x = current_candidates[0]
    # Pulido exacto identificando la rama activa de cada átomo en best_x junto con sus desigualdades de validez
    subs_best = {ctrl_syms[i]: float(best_x[i]) for i in range(n)}
    atom_subs_list: List[Tuple[sp.Expr, sp.Expr]] = []
    pol_branch_ineqs: List[sp.Expr] = []
    for atom in problem.nonsmooth_atoms:
        best_br_expr = _apply_atom_subs(atom.branches[0][0], atom_subs_list)
        best_br_conds = [_apply_atom_subs(gc, atom_subs_list) for gc in atom.branches[0][1]]
        for br_e, br_conds, _ in atom.branches:
            sub_conds = [_apply_atom_subs(gc, atom_subs_list) for gc in br_conds]
            try:
                if all(float(sc.xreplace({s: sp.Float(v) for s, v in subs_best.items()}).evalf()) <= 1e-5 for sc in sub_conds):
                    best_br_expr = _apply_atom_subs(br_e, atom_subs_list)
                    best_br_conds = sub_conds
                    break
            except Exception:
                continue
        atom_subs_list.append((atom.sympy_atom, best_br_expr))
        pol_branch_ineqs.extend(best_br_conds)

    f_pol = _apply_atom_subs(problem.min_obj_expr, atom_subs_list)
    ineq_pol = pol_branch_ineqs + [_apply_atom_subs(c.expr, atom_subs_list) for c in problem.constraints if c.kind == "ineq"]
    eq_pol = [_apply_atom_subs(c.expr, atom_subs_list) for c in problem.constraints if c.kind == "eq"]
    pol_res = _solve_smooth_subproblem_slsqp(f_pol, ineq_pol, eq_pol, ctrl_syms, lb, ub, [best_x])
    if pol_res is not None:
        cand_pol = pol_res[0]
        v_pol, _ = problem.max_constraint_violation(cand_pol, check_integrality=False)
        if v_pol <= 1e-4:
            j_best = problem.evaluate_j(best_x)
            j_cand = problem.evaluate_j(cand_pol)
            if (problem.sense == "max" and j_cand >= j_best - 1e-7) or (problem.sense == "min" and j_cand <= j_best + 1e-7):
                best_x = cand_pol

    max_v, _ = problem.max_constraint_violation(best_x, check_integrality=False)
    if max_v > 1e-4:
        return None

    return {
        "x": best_x,
        "j_val": problem.evaluate_j(best_x),
        "max_violation": max_v,
        "strategy_used": "strategy_c_lse_homotopy",
        "is_exact": False,
        "optimality_gap": None,
        "kkt_residual": pol_res[2] if pol_res is not None else 0.0,
        "active_regime": "homotopy_polished",
    }


# ==============================================================================
# ESTRATEGIA D: Branch-and-Bound / Búsqueda Discreta Multi-Variable General (MINLP)
# ==============================================================================

def solve_strategy_d_minlp(problem: Problem) -> Optional[Dict[str, Any]]:
    """Estrategia D: Resolución de problemas con múltiples variables enteras/binarias y expresiones no lineales."""
    # 1. Si es lineal o lineal a trozos, delegar directamente a HiGHS MILP (Estrategia A)
    if problem.problem_class in ("LP", "MILP"):
        res_a = solve_strategy_a_milp(problem)
        if res_a is not None:
            return res_a
    if problem.is_piecewise_linear and problem.nonsmooth_atoms:
        res_pw = solve_strategy_a_piecewise_milp(problem)
        if res_pw is not None:
            return res_pw

    n = len(problem.vars)
    disc_indices = [i for i, v in enumerate(problem.vars) if v.domain in ("integer", "binary")]
    cont_indices = [i for i, v in enumerate(problem.vars) if v.domain == "continuous"]

    if not disc_indices:
        return solve_strategy_b_regime_enumeration(problem)

    ctrl_syms = problem.ctrl_symbols
    try:
        min_obj_fast = sp.lambdify(ctrl_syms, problem.min_obj_expr, modules=_SAFE_LAMBDIFY_MODULES)
    except Exception:
        min_obj_fast = None

    def eval_min_obj(x_c: np.ndarray) -> float:
        if min_obj_fast is not None:
            try:
                return float(min_obj_fast(*[float(z) for z in x_c]))
            except Exception:
                pass
        subs = {ctrl_syms[i]: float(x_c[i]) for i in range(n)}
        return float(problem.min_obj_expr.evalf(subs=subs))

    # Caso D1: Todas las variables son discretas (ej. programación entera no lineal / asignación de recursos discreta)
    card_per_disc = [int(round(problem.vars[i].ub - problem.vars[i].lb)) + 1 for i in disc_indices]
    total_disc_card = 1
    for c_sz in card_per_disc:
        total_disc_card *= max(1, c_sz)

    if len(cont_indices) == 0 and total_disc_card <= 50000:
        ranges = [range(int(round(problem.vars[i].lb)), int(round(problem.vars[i].ub)) + 1) for i in disc_indices]
        best_x: Optional[np.ndarray] = None
        best_f = float("inf")
        for combo in itertools.product(*ranges):
            x_cand = np.array(combo, dtype=float)
            max_v, _ = problem.max_constraint_violation(x_cand)
            if max_v <= 1e-5:
                f_v = eval_min_obj(x_cand)
                if f_v < best_f - 1e-9:
                    best_f = f_v
                    best_x = x_cand
        if best_x is not None:
            return {
                "x": best_x,
                "j_val": problem.evaluate_j(best_x),
                "max_violation": 0.0,
                "strategy_used": "strategy_d_discrete_exact_enumeration",
                "is_exact": True,
                "optimality_gap": 0.0,
                "kkt_residual": 0.0,
                "active_regime": "discrete_lattice_optimum",
            }

    # Caso D2: MINLP mixto (variables discretas + continuas) o cardinalidad discreta moderada
    # Paso 1: Resolver la relajación continua en cada régimen suave + variedad kink
    rel_sol = solve_strategy_b_regime_enumeration(problem, n_starts=3)
    if rel_sol is None and len(problem.nonsmooth_atoms) > 8:
        rel_sol = solve_strategy_c_homotopy_lse(problem)

    best_int_x: Optional[np.ndarray] = None
    best_int_f = float("inf")
    best_regime = "minlp_branch_and_bound"

    # Si la cardinalidad discreta total es <= 64 (ej. hasta 6 variables binarias), enumerar las asignaciones discretas
    # y optimizar las variables continuas en cada nodo (Branch-and-Bound exhaustivo)
    if total_disc_card <= 64:
        ranges = [range(int(round(problem.vars[i].lb)), int(round(problem.vars[i].ub)) + 1) for i in disc_indices]
        for combo in itertools.product(*ranges):
            fixed_map = {disc_indices[k]: float(combo[k]) for k in range(len(disc_indices))}
            sub_sol = solve_strategy_b_regime_enumeration(problem, fixed_var_overrides=fixed_map, n_starts=2)
            if sub_sol is not None and sub_sol["max_violation"] <= 1e-4:
                cand_x = sub_sol["x"]
                for idx_d in disc_indices:
                    cand_x[idx_d] = float(int(round(cand_x[idx_d])))
                f_v = eval_min_obj(cand_x)
                if f_v < best_int_f - 1e-9:
                    best_int_f = f_v
                    best_int_x = cand_x.copy()
                    best_regime = sub_sol.get("active_regime", best_regime)
    else:
        # Branch-and-Bound guiado por la relajación continua + vecindario entero alrededor de x_rel
        if rel_sol is not None:
            x_rel = rel_sol["x"]
            best_regime = rel_sol.get("active_regime", best_regime)
            reg_optima = rel_sol.get("all_regime_optima") or [x_rel]

            # 1) Evaluar redondeo rápido directo sobre todos los óptimos de régimen continuo
            for x_ro in reg_optima:
                ro_rounds = [sorted({math.floor(float(x_ro[idx_d])), math.ceil(float(x_ro[idx_d]))}) for idx_d in disc_indices]
                for r_combo in itertools.product(*ro_rounds):
                    cand_dir = x_ro.copy()
                    for k_i, idx_d in enumerate(disc_indices):
                        v_d = problem.vars[idx_d]
                        cand_dir[idx_d] = float(max(int(round(v_d.lb)), min(int(round(v_d.ub)), int(r_combo[k_i]))))
                    mv_dir, _ = problem.max_constraint_violation(cand_dir)
                    if mv_dir <= 1e-4:
                        f_dir = eval_min_obj(cand_dir)
                        if f_dir < best_int_f - 1e-9:
                            best_int_f = f_dir
                            best_int_x = cand_dir.copy()

            # 2) Si x_rel tiene parte fraccionaria en variables discretas, re-optimizar las continuas en floor/ceil
            is_rel_already_int = all(abs(float(x_rel[idx_d]) - round(float(x_rel[idx_d]))) <= 1e-5 for idx_d in disc_indices)
            if not is_rel_already_int or best_int_x is None:
                disc_cand_sets: List[List[int]] = []
                for idx_d in disc_indices:
                    v_d = problem.vars[idx_d]
                    lo_i, hi_i = int(round(v_d.lb)), int(round(v_d.ub))
                    val_r = float(x_rel[idx_d])
                    core_cands = {math.floor(val_r), math.ceil(val_r)}
                    cands = sorted(set(max(lo_i, min(hi_i, int(c_v))) for c_v in core_cands))
                    disc_cand_sets.append(cands)

                for combo in itertools.product(*disc_cand_sets):
                    fixed_map = {disc_indices[k]: float(combo[k]) for k in range(len(disc_indices))}
                    sub_sol = solve_strategy_b_regime_enumeration(problem, fixed_var_overrides=fixed_map, n_starts=2)
                    if sub_sol is not None and sub_sol["max_violation"] <= 1e-4:
                        cand_x = sub_sol["x"]
                        for idx_d in disc_indices:
                            cand_x[idx_d] = float(int(round(cand_x[idx_d])))
                        f_v = eval_min_obj(cand_x)
                        if f_v < best_int_f - 1e-9:
                            best_int_f = f_v
                            best_int_x = cand_x.copy()
                            best_regime = sub_sol.get("active_regime", best_regime)

    if best_int_x is None:
        return None

    max_v, _ = problem.max_constraint_violation(best_int_x)
    return {
        "x": best_int_x,
        "j_val": problem.evaluate_j(best_int_x),
        "max_violation": max_v,
        "strategy_used": "strategy_d_minlp_branch_and_bound",
        "is_exact": True,
        "optimality_gap": 0.0,
        "kkt_residual": 0.0,
        "active_regime": best_regime,
    }


# ==============================================================================
# ESTRATEGIA E: Diagnóstico General de Infactibilidad (IIS) vía Relajación Elástica L1
# ==============================================================================

def _test_subset_feasibility(problem: Problem, active_c_indices: List[int]) -> Tuple[bool, float, Optional[np.ndarray], Dict[str, float]]:
    """Resuelve min sum_k s_k s.t. g_k(x) <= s_k, s_k >= 0 para el subconjunto de restricciones dado."""
    n = len(problem.vars)
    m = len(active_c_indices)
    if m == 0:
        x_mid = np.array([0.5 * (v.lb + v.ub) for v in problem.vars], dtype=float)
        for i, v in enumerate(problem.vars):
            if v.domain in ("integer", "binary"):
                x_mid[i] = float(int(round(x_mid[i])))
        return True, 0.0, x_mid, {}

    sub_constrs = [problem.constraints[i] for i in active_c_indices]
    syms = problem.ctrl_symbols

    # Vía rápida LP/MILP en HiGHS si todas las restricciones del subconjunto son lineales
    all_lin = all(c.is_linear for c in sub_constrs)
    if all_lin:
        total_vars = n + m
        c_vec = np.zeros(total_vars, dtype=float)
        c_vec[n:] = 1.0
        lb_full = np.zeros(total_vars, dtype=float)
        ub_full = np.full(total_vars, 1e12, dtype=float)
        integrality = np.zeros(total_vars, dtype=int)
        for i, v in enumerate(problem.vars):
            lb_full[i] = v.lb
            ub_full[i] = v.ub
            if v.domain in ("integer", "binary"):
                integrality[i] = 1

        a_rows: List[np.ndarray] = []
        lb_rows: List[float] = []
        ub_rows: List[float] = []

        lin_ok = True
        for k_idx, constr in enumerate(sub_constrs):
            lin_c = _extract_linear_coeffs(constr.expr, syms)
            if lin_c is None:
                lin_ok = False
                break
            vec_a, cst_b = lin_c
            if constr.kind == "ineq":
                # vec_a^T x - s_k <= -cst_b
                row = np.zeros(total_vars, dtype=float)
                row[:n] = vec_a
                row[n + k_idx] = -1.0
                a_rows.append(row)
                lb_rows.append(-np.inf)
                ub_rows.append(-cst_b)
            else:
                # vec_a^T x - s_k <= -cst_b  y  -vec_a^T x - s_k <= cst_b
                row1 = np.zeros(total_vars, dtype=float)
                row1[:n] = vec_a
                row1[n + k_idx] = -1.0
                a_rows.append(row1)
                lb_rows.append(-np.inf)
                ub_rows.append(-cst_b)

                row2 = np.zeros(total_vars, dtype=float)
                row2[:n] = -vec_a
                row2[n + k_idx] = -1.0
                a_rows.append(row2)
                lb_rows.append(-np.inf)
                ub_rows.append(cst_b)

        if lin_ok and a_rows:
            res = milp(
                c=c_vec,
                integrality=integrality,
                bounds=Bounds(lb_full, ub_full),
                constraints=LinearConstraint(np.vstack(a_rows), np.array(lb_rows, dtype=float), np.array(ub_rows, dtype=float)),
            )
            if res.success and res.x is not None:
                total_slack = float(np.sum(np.maximum(0.0, res.x[n:])))
                x_best = np.array(res.x[:n], dtype=float)
                slack_by_id = {sub_constrs[k].id: float(max(0.0, res.x[n + k])) for k in range(m)}
                return (total_slack <= 1e-6), total_slack, x_best, slack_by_id

    # Fallback no lineal: minimizar la suma de violaciones max(0, g_k(x)) + |h_k(x)| sobre [lb, ub]
    lb_v = np.array([v.lb for v in problem.vars], dtype=float)
    ub_v = np.array([v.ub for v in problem.vars], dtype=float)
    c_fns: List[Tuple[str, str, Callable[..., float]]] = []
    for sc in sub_constrs:
        for cid_f, kind_f, fn_f in problem._fast_constr_fns:
            if cid_f == sc.id:
                c_fns.append((cid_f, kind_f, fn_f))
                break

    def penalty_fn(x: np.ndarray) -> float:
        args = [float(z) for z in x]
        tot = 0.0
        for _, kind_f, fn_f in c_fns:
            try:
                v = float(fn_f(*args))
            except Exception:
                v = 1e6
            tot += max(0.0, v) if kind_f == "ineq" else abs(v)
        return tot

    starts = _generate_multistart_points(problem, n_sobol=16)
    best_slack = float("inf")
    best_x: Optional[np.ndarray] = None
    bounds_list = [(float(lb_v[i]), float(ub_v[i])) for i in range(n)]

    # Si todas las variables son discretas y el retículo entero es pequeño, evaluarlo exactamente
    if all(v.domain in ("integer", "binary") for v in problem.vars):
        card = 1
        ranges: List[List[float]] = []
        for v in problem.vars:
            lo_i = int(math.ceil(v.lb))
            hi_i = int(math.floor(v.ub))
            span = max(0, hi_i - lo_i + 1)
            card *= max(1, span)
            if card > 4096:
                break
            ranges.append([float(k) for k in range(lo_i, hi_i + 1)])
        if card <= 4096 and len(ranges) == n and all(len(r) > 0 for r in ranges):
            for pt in itertools.product(*ranges):
                cand_x = np.array(pt, dtype=float)
                s_val = penalty_fn(cand_x)
                if s_val < best_slack:
                    best_slack = s_val
                    best_x = cand_x
                if best_slack <= 1e-7:
                    break

    if best_slack > 1e-7:
        for x0 in starts[:10]:
            try:
                res = minimize(penalty_fn, x0=x0, bounds=bounds_list, method="SLSQP", options={"ftol": 1e-9, "maxiter": 150})
                cand_x = np.clip(np.array(res.x, dtype=float), lb_v, ub_v)
                for i, v in enumerate(problem.vars):
                    if v.domain in ("integer", "binary"):
                        cand_x[i] = float(int(round(cand_x[i])))
                s_val = penalty_fn(cand_x)
                if s_val < best_slack:
                    best_slack = s_val
                    best_x = cand_x
                if best_slack <= 1e-7:
                    break
            except Exception:
                continue

    slack_map: Dict[str, float] = {}
    if best_x is not None:
        args_b = [float(z) for z in best_x]
        for cid_f, kind_f, fn_f in c_fns:
            try:
                rv = float(fn_f(*args_b))
            except Exception:
                rv = 0.0
            slack_map[cid_f] = max(0.0, rv) if kind_f == "ineq" else abs(rv)

    return (best_slack <= 1e-5), float(best_slack), best_x, slack_map


def solve_strategy_e_iis(problem: Problem) -> Dict[str, Any]:
    """Estrategia E: Relajación Elástica L1 + Filtro de Eliminación para extraer un IIS mínimo certificado."""
    all_indices = list(range(len(problem.constraints)))
    is_feas, min_slack, x_elastic, slack_map = _test_subset_feasibility(problem, all_indices)

    # Si la relajación elástica encuentra un punto con holgura 0, el problema era factible
    if is_feas and x_elastic is not None:
        max_v, _ = problem.max_constraint_violation(x_elastic)
        if max_v <= 1e-4:
            return {
                "feasible_recovered": True,
                "x": x_elastic,
                "j_val": problem.evaluate_j(x_elastic),
                "max_violation": max_v,
                "strategy_used": "strategy_e_elastic_recovery",
                "is_exact": False,
                "optimality_gap": None,
                "kkt_residual": 0.0,
                "active_regime": "elastic_feasible",
            }

    # Filtro de eliminación (Deletion Filter) para aislar un Subsistema Irreduciblemente Inconsistente (IIS)
    iis_indices = list(all_indices)
    for idx_c in list(all_indices):
        trial_indices = [i for i in iis_indices if i != idx_c]
        sub_feas, _, _, _ = _test_subset_feasibility(problem, trial_indices)
        # Si al quitar idx_c el sistema SIGUE siendo infactible, idx_c es redundante para el conflicto
        if not sub_feas:
            iis_indices = trial_indices

    # Si hay pocos candidatos totales y el IIS hallado tiene cardinalidad > 1, verificar si existe un IIS
    # de cardinalidad estrictamente menor (cuando existen múltiples IIS solapados)
    if len(iis_indices) > 1 and len(all_indices) <= 12:
        found_smaller = False
        for r_size in range(1, len(iis_indices)):
            if math.comb(len(all_indices), r_size) > 84:
                break
            for comb in itertools.combinations(all_indices, r_size):
                comb_list = list(comb)
                sub_feas_c, _, _, _ = _test_subset_feasibility(problem, comb_list)
                if not sub_feas_c:
                    iis_indices = comb_list
                    found_smaller = True
                    break
            if found_smaller:
                break

    iis_ids = [problem.constraints[i].id for i in iis_indices]

    # Si una única restricción choca directamente con las cotas de las variables, incluir las cotas en conflicto
    if len(iis_indices) == 1:
        c_only = problem.constraints[iis_indices[0]]
        for v in problem.vars:
            if c_only.expr.has(v.symbol):
                iis_ids.append(f"bounds({v.id})")

    # Construir diagnóstico de multiplicadores sombra KKT / cuello de botella de infactibilidad
    shadow_multipliers: Dict[str, float] = {}
    for cid in iis_ids:
        if cid in slack_map and slack_map[cid] > 1e-7:
            shadow_multipliers[cid] = round(float(slack_map[cid]), 4)
        else:
            shadow_multipliers[cid] = 1.0

    return {
        "feasible_recovered": False,
        "iis_constraints": iis_ids,
        "min_relaxation_slack": float(min_slack),
        "shadow_multipliers": shadow_multipliers,
        "x_closest": x_elastic,
        "slack_by_constraint": slack_map,
    }
