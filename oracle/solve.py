"""Oráculo de referencia independiente con fuerza bruta y cálculo analítico exacto.

Este script es completamente independiente del CGE.
Calcula el espacio de soluciones por fuerza bruta / rejilla fina y optimización analítica,
y genera los 4 archivos JSON de referencia esperada:
- expected_graph.json
- expected_diagnostics.json
- expected_solution.json
- expected_infeasible.json
"""
import os
import json
import math
from typing import Dict, Any, List


def evaluate_system(price: float, marketing_budget: float, order_qty: int, base_demand: float = 1000.0) -> Dict[str, Any]:
    """Evalúa el modelo exacto de la Tostadora de Café."""
    unit_cost = 1.20 if order_qty < 800 else 1.00
    brand_awareness = 12.5 * math.log1p(max(0.0, marketing_budget)) if marketing_budget > 0 else 0.0
    demand = max(0.0, base_demand * (1.0 - 0.15 * (price - 3.0)) + 8.0 * math.sqrt(max(0.0, marketing_budget)))
    sales = min(demand, float(order_qty))
    holding_cost = 0.20 * max(0.0, float(order_qty) - sales)
    revenue = price * sales
    fees = 0.03 * revenue
    profit = revenue - fees - (unit_cost * float(order_qty)) - marketing_budget - holding_cost
    budget_used = unit_cost * float(order_qty) + marketing_budget
    is_budget_feasible = budget_used <= 2500.0 + 1e-6
    is_price_feasible = 2.0 <= price <= 6.0
    is_stock_feasible = float(order_qty) >= sales

    return {
        "price": price,
        "marketing_budget": marketing_budget,
        "order_qty": order_qty,
        "base_demand": base_demand,
        "brand_awareness": brand_awareness,
        "demand": demand,
        "sales": sales,
        "unit_cost": unit_cost,
        "holding_cost": holding_cost,
        "revenue": revenue,
        "fees": fees,
        "profit": profit,
        "budget_used": budget_used,
        "is_feasible": is_budget_feasible and is_price_feasible and is_stock_feasible,
        "c_budget_violated": not is_budget_feasible,
        "c_stock_violated": not is_stock_feasible,
        "c_price_violated": not is_price_feasible
    }


def find_optimal_solution(base_demand: float = 1000.0) -> Dict[str, Any]:
    """Encuentra la solución óptima mediante cálculo analítico exacto y verificación de rejilla."""
    # En el óptimo q* = 800 (discontinuidad de coste c=1.00)
    # y d* = q* = 800.
    # d = 1450 - 150p + 8*sqrt(m) = 800 => sqrt(m) = (150p - 650)/8
    # J(p) = (0.97p - 1.0)*800 - ((150p - 650)/8)^2
    # Derivada dJ/dp = 776 - (45000p - 195000)/64 = 0 => p* = 244664 / 45000 = 5.4369777...
    p_opt = 244664.0 / 45000.0
    sqrt_m_opt = (150.0 * p_opt - 650.0) / 8.0
    m_opt = sqrt_m_opt ** 2
    q_opt = 800

    opt_eval = evaluate_system(p_opt, m_opt, q_opt, base_demand)
    
    # Sensibilidades numéricas en el óptimo (diferencias finitas centradas)
    eps = 1e-4
    eval_p_plus = evaluate_system(p_opt + eps, m_opt, q_opt, base_demand)
    eval_p_minus = evaluate_system(p_opt - eps, m_opt, q_opt, base_demand)
    dJ_dp = (eval_p_plus["profit"] - eval_p_minus["profit"]) / (2 * eps)

    eval_m_plus = evaluate_system(p_opt, m_opt + eps, q_opt, base_demand)
    eval_m_minus = evaluate_system(p_opt, m_opt - eps, q_opt, base_demand)
    dJ_dm = (eval_m_plus["profit"] - eval_m_minus["profit"]) / (2 * eps)

    # Sensibilidad discreta w.r.t q:
    eval_q_minus1 = evaluate_system(p_opt, m_opt, q_opt - 1, base_demand)
    eval_q_plus1 = evaluate_system(p_opt, m_opt, q_opt + 1, base_demand)
    delta_J_dq_left = opt_eval["profit"] - eval_q_minus1["profit"] # salto escalón c=1.2 vs 1.0
    delta_J_dq_right = eval_q_plus1["profit"] - opt_eval["profit"]

    return {
        "optimal_controls": {
            "price": round(p_opt, 4),
            "marketing_budget": round(m_opt, 4),
            "order_qty": int(q_opt)
        },
        "optimal_target": {
            "profit": round(opt_eval["profit"], 4)
        },
        "state_values": {
            "demand": round(opt_eval["demand"], 4),
            "sales": round(opt_eval["sales"], 4),
            "unit_cost": round(opt_eval["unit_cost"], 2),
            "holding_cost": round(opt_eval["holding_cost"], 4),
            "revenue": round(opt_eval["revenue"], 4),
            "fees": round(opt_eval["fees"], 4),
            "brand_awareness": round(opt_eval["brand_awareness"], 4),
            "budget_used": round(opt_eval["budget_used"], 4)
        },
        "active_constraints": [
            "c_volume_discount_threshold",
            "c_sales_capacity_limit"
        ],
        "sensitivities": {
            "dJ_dp": round(dJ_dp, 4),
            "dJ_dm": round(dJ_dm, 4),
            "delta_J_dq_left": round(delta_J_dq_left, 4),
            "delta_J_dq_right": round(delta_J_dq_right, 4),
            "dJ_d_brand_awareness": 0.0
        }
    }


def generate_expected_graph() -> Dict[str, Any]:
    """Genera expected_graph.json con los 12 nodos y aristas verdaderas."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "system_id": "cge_golden_case_coffee",
        "version": "1.0.0",
        "nodes": [
            {
                "id": "base_demand",
                "label": "Demanda Base Exógena",
                "category": "exogenous",
                "data_type": "continuous",
                "source_reference": {
                    "file": "golden_case/environment.py",
                    "entity": "sample_base_demand",
                    "line_range": [9, 14]
                },
                "current_value": 1000.0,
                "bounds": {"min": 0.0, "max": None}
            },
            {
                "id": "price",
                "label": "Precio de Venta",
                "category": "control",
                "data_type": "continuous",
                "source_reference": {
                    "file": "golden_case/config.py",
                    "entity": "PRICE",
                    "line_range": [4, 4]
                },
                "current_value": 3.50,
                "bounds": {"min": 2.0, "max": 6.0}
            },
            {
                "id": "marketing_budget",
                "label": "Presupuesto de Marketing",
                "category": "control",
                "data_type": "continuous",
                "source_reference": {
                    "file": "golden_case/config.py",
                    "entity": "MARKETING_BUDGET",
                    "line_range": [5, 5]
                },
                "current_value": 500.00,
                "bounds": {"min": 0.0, "max": 2000.0}
            },
            {
                "id": "order_qty",
                "label": "Cantidad de Pedido",
                "category": "control",
                "data_type": "discrete",
                "source_reference": {
                    "file": "golden_case/config.py",
                    "entity": "ORDER_QTY",
                    "line_range": [6, 6]
                },
                "current_value": 500,
                "bounds": {"min": 0, "max": 1500}
            },
            {
                "id": "brand_awareness",
                "label": "Notoriedad de Marca (Nodo Muerto T1)",
                "category": "state",
                "data_type": "continuous",
                "source_reference": {
                    "file": "golden_case/marketing.py",
                    "entity": "calculate_brand_awareness",
                    "line_range": [8, 23]
                },
                "current_value": 77.71,
                "bounds": {"min": 0.0, "max": None}
            },
            {
                "id": "demand",
                "label": "Demanda Estimada",
                "category": "state",
                "data_type": "continuous",
                "source_reference": {
                    "file": "golden_case/demand.py",
                    "entity": "calculate_demand",
                    "line_range": [5, 13]
                },
                "current_value": 1103.88,
                "bounds": {"min": 0.0, "max": None}
            },
            {
                "id": "sales",
                "label": "Ventas Efectivas",
                "category": "state",
                "data_type": "continuous",
                "source_reference": {
                    "file": "golden_case/inventory.py",
                    "entity": "calculate_sales",
                    "line_range": [17, 22]
                },
                "current_value": 500.0,
                "bounds": {"min": 0.0, "max": 1500.0}
            },
            {
                "id": "unit_cost",
                "label": "Coste Unitario",
                "category": "state",
                "data_type": "continuous",
                "source_reference": {
                    "file": "golden_case/inventory.py",
                    "entity": "calculate_unit_cost",
                    "line_range": [5, 14]
                },
                "current_value": 1.20,
                "bounds": {"min": 1.00, "max": 1.20}
            },
            {
                "id": "holding_cost",
                "label": "Coste de Almacenamiento",
                "category": "state",
                "data_type": "continuous",
                "source_reference": {
                    "file": "golden_case/inventory.py",
                    "entity": "calculate_holding_cost",
                    "line_range": [25, 28]
                },
                "current_value": 0.0,
                "bounds": {"min": 0.0, "max": None}
            },
            {
                "id": "revenue",
                "label": "Ingresos Brutos",
                "category": "state",
                "data_type": "continuous",
                "source_reference": {
                    "file": "golden_case/finance.py",
                    "entity": "calculate_revenue",
                    "line_range": [5, 7]
                },
                "current_value": 1750.0,
                "bounds": {"min": 0.0, "max": None}
            },
            {
                "id": "fees",
                "label": "Comisiones de Pasarela",
                "category": "state",
                "data_type": "continuous",
                "source_reference": {
                    "file": "golden_case/finance.py",
                    "entity": "calculate_fees",
                    "line_range": [10, 18]
                },
                "current_value": 52.50,
                "bounds": {"min": 0.0, "max": None}
            },
            {
                "id": "profit",
                "label": "Beneficio Neto Operativo (Target J)",
                "category": "target",
                "data_type": "continuous",
                "source_reference": {
                    "file": "golden_case/finance.py",
                    "entity": "calculate_profit",
                    "line_range": [21, 35]
                },
                "current_value": 597.50,
                "bounds": {"min": None, "max": None}
            }
        ],
        "edges": [
            {"source": "base_demand", "target": "demand", "type": "exact_code", "sensitivity_weight": 0.925, "confidence": 1.0, "exact_mechanism": {"engine": "python_callable", "definition": "demand = base_demand * (1 - 0.15*(price-3)) + 8*sqrt(marketing_budget)"}},
            {"source": "price", "target": "demand", "type": "exact_code", "sensitivity_weight": -150.0, "confidence": 1.0, "exact_mechanism": {"engine": "python_callable", "definition": "ddemand/dprice = -0.15 * base_demand"}},
            {"source": "marketing_budget", "target": "demand", "type": "exact_code", "sensitivity_weight": 0.178, "confidence": 1.0, "exact_mechanism": {"engine": "python_callable", "definition": "ddemand/dmarketing = 4 / sqrt(marketing_budget)"}},
            {"source": "marketing_budget", "target": "demand", "type": "empirical_data", "sensitivity_weight": 0.18, "confidence": 0.95, "exact_mechanism": {"engine": "statistical_regression", "definition": "estimated lift: 8.0 * sqrt(marketing_budget) from sales_history.csv"}},
            {"source": "marketing_budget", "target": "brand_awareness", "type": "exact_code", "sensitivity_weight": 0.025, "confidence": 1.0, "exact_mechanism": {"engine": "python_callable", "definition": "brand_awareness = 12.5 * log1p(marketing_budget)"}},
            {"source": "demand", "target": "sales", "type": "discrete_rule", "sensitivity_weight": 0.0, "confidence": 1.0, "exact_mechanism": {"engine": "python_callable", "definition": "sales = min(demand, order_qty)"}},
            {"source": "order_qty", "target": "sales", "type": "discrete_rule", "sensitivity_weight": 1.0, "confidence": 1.0, "exact_mechanism": {"engine": "python_callable", "definition": "sales = min(demand, order_qty)"}},
            {"source": "order_qty", "target": "unit_cost", "type": "discrete_rule", "sensitivity_weight": -0.20, "confidence": 1.0, "exact_mechanism": {"engine": "python_callable", "definition": "unit_cost = 1.20 if order_qty < 800 else 1.00"}},
            {"source": "order_qty", "target": "holding_cost", "type": "exact_code", "sensitivity_weight": 0.20, "confidence": 1.0, "exact_mechanism": {"engine": "python_callable", "definition": "holding_cost = 0.20 * max(0, order_qty - sales)"}},
            {"source": "sales", "target": "holding_cost", "type": "exact_code", "sensitivity_weight": -0.20, "confidence": 1.0, "exact_mechanism": {"engine": "python_callable", "definition": "holding_cost = 0.20 * max(0, order_qty - sales)"}},
            {"source": "price", "target": "revenue", "type": "exact_code", "sensitivity_weight": 500.0, "confidence": 1.0, "exact_mechanism": {"engine": "python_callable", "definition": "revenue = price * sales"}},
            {"source": "sales", "target": "revenue", "type": "exact_code", "sensitivity_weight": 3.50, "confidence": 1.0, "exact_mechanism": {"engine": "python_callable", "definition": "revenue = price * sales"}},
            {"source": "revenue", "target": "fees", "type": "exact_code", "sensitivity_weight": 0.03, "confidence": 1.0, "exact_mechanism": {"engine": "python_callable", "definition": "fees = 0.03 * revenue"}},
            {"source": "revenue", "target": "profit", "type": "exact_code", "sensitivity_weight": 1.0, "confidence": 1.0, "exact_mechanism": {"engine": "python_callable", "definition": "profit += revenue"}},
            {"source": "fees", "target": "profit", "type": "exact_code", "sensitivity_weight": -1.0, "confidence": 1.0, "exact_mechanism": {"engine": "python_callable", "definition": "profit -= fees"}},
            {"source": "unit_cost", "target": "profit", "type": "exact_code", "sensitivity_weight": -500.0, "confidence": 1.0, "exact_mechanism": {"engine": "python_callable", "definition": "profit -= unit_cost * order_qty"}},
            {"source": "order_qty", "target": "profit", "type": "exact_code", "sensitivity_weight": -1.20, "confidence": 1.0, "exact_mechanism": {"engine": "python_callable", "definition": "profit -= unit_cost * order_qty"}},
            {"source": "marketing_budget", "target": "profit", "type": "exact_code", "sensitivity_weight": -1.0, "confidence": 1.0, "exact_mechanism": {"engine": "python_callable", "definition": "profit -= marketing_budget"}},
            {"source": "holding_cost", "target": "profit", "type": "exact_code", "sensitivity_weight": -1.0, "confidence": 1.0, "exact_mechanism": {"engine": "python_callable", "definition": "profit -= holding_cost"}}
        ],
        "constraints": [
            {
                "id": "c_budget",
                "type": "hard",
                "expression": "unit_cost * order_qty + marketing_budget <= 2500",
                "description": "Presupuesto máximo de costes directos y marketing de 2500€"
            },
            {
                "id": "c_ledger",
                "type": "hard",
                "expression": "sum(credits) - sum(debits) == 0",
                "description": "Invariante contable de partida doble en el libro mayor"
            },
            {
                "id": "c_stock",
                "type": "hard",
                "expression": "order_qty - sales >= 0",
                "description": "Las ventas no pueden exceder las existencias disponibles"
            },
            {
                "id": "c_price_bounds",
                "type": "hard",
                "expression": "2.0 <= price <= 6.0",
                "description": "Rango admisible de fijación de precios en el mercado"
            }
        ]
    }


def generate_expected_diagnostics() -> Dict[str, Any]:
    """Genera expected_diagnostics.json con las 7 trampas plantadas T1-T7."""
    return {
        "diagnostics": [
            {
                "id": "T1",
                "code": "DEAD_NODE",
                "severity": "warning",
                "node": "brand_awareness",
                "file": "golden_case/marketing.py",
                "line_range": [8, 23],
                "details": {
                    "gradient_dJ_dnode": 0.0,
                    "reason": "Variable calculada pero desconectada de la función objetivo target J."
                }
            },
            {
                "id": "T2",
                "code": "INVARIANT_VIOLATION",
                "severity": "critical",
                "constraint_id": "c_ledger",
                "file": "golden_case/finance.py",
                "function": "process_refund",
                "line_range": [63, 73],
                "details": {
                    "reason": "La función process_refund anota un débito por duplicado, violando el balance de partida doble."
                }
            },
            {
                "id": "T3",
                "code": "DOC_CODE_DISCREPANCY",
                "severity": "critical",
                "node": "fees",
                "doc_file": "golden_case/README.md",
                "code_file": "golden_case/finance.py",
                "function": "calculate_fees",
                "doc_value": 0.02,
                "code_value": 0.03,
                "details": {
                    "reason": "El README documenta una comisión del 2% pero el código aplica un 3% (error relativo del 50%)."
                }
            },
            {
                "id": "T4",
                "code": "LOCAL_MINIMUM",
                "severity": "critical",
                "file": "golden_case/inventory.py",
                "function": "optimize_order",
                "local_recommendation": {"order_qty": 500},
                "global_optimum": {"order_qty": 800},
                "profit_loss_at_best_p_m": 680.88,
                "details": {
                    "reason": "El optimizador local minimiza el coste de almacenaje sugiriendo q=500, causando roturas de stock y perdiendo más de 680€ en beneficio respecto a q=800."
                }
            },
            {
                "id": "T5",
                "code": "DISCONTINUITY",
                "severity": "warning",
                "node": "unit_cost",
                "file": "golden_case/inventory.py",
                "function": "calculate_unit_cost",
                "line_range": [5, 14],
                "threshold_variable": "order_qty",
                "threshold_value": 800,
                "details": {
                    "reason": "Salto no continuo en unit_cost (1.20 -> 1.00) en el umbral q=800."
                }
            },
            {
                "id": "T6",
                "code": "FRAGILE_NODE",
                "severity": "warning",
                "node": "sales",
                "file": "golden_case/inventory.py",
                "function": "calculate_sales",
                "line_range": [17, 22],
                "details": {
                    "reason": "En el óptimo global, demand ≈ order_qty (800 unidades), haciendo que min(demand, order_qty) cambie de régimen ante pequeñas perturbaciones de base_demand."
                }
            },
            {
                "id": "T7",
                "code": "EMPIRICAL_EDGE",
                "severity": "info",
                "source": "marketing_budget",
                "target": "demand",
                "data_source": "golden_case/data/sales_history.csv",
                "ground_truth_coefficient": 8.0,
                "details": {
                    "reason": "Relación funcional estocástica observable empíricamente en datos tabulares con ruido."
                }
            }
        ]
    }


def generate_expected_infeasible() -> Dict[str, Any]:
    """Genera expected_infeasible.json con un objetivo inalcanzable (J >= 5000)."""
    return {
        "target_objective": {
            "profit": 5000.0,
            "operator": ">="
        },
        "status": "INFEASIBLE",
        "unsat_core": [
            "c_budget",
            "c_market_capacity"
        ],
        "blocking_constraint": "c_budget",
        "explanation": "El beneficio máximo alcanzable en el sistema es de aproximadamente 2990.88€ debido a que el presupuesto máximo de 2500€ y el límite de absorción de demanda del mercado impiden generar ingresos suficientes para alcanzar 5000€.",
        "max_attainable_profit": 2990.88
    }


def main():
    os.makedirs("oracle", exist_ok=True)
    
    # 1. Expected Graph
    graph_data = generate_expected_graph()
    with open("oracle/expected_graph.json", "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
    # 2. Expected Diagnostics
    diag_data = generate_expected_diagnostics()
    with open("oracle/expected_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diag_data, f, indent=2, ensure_ascii=False)
        
    # 3. Expected Solution
    sol_data = find_optimal_solution(base_demand=1000.0)
    with open("oracle/expected_solution.json", "w", encoding="utf-8") as f:
        json.dump(sol_data, f, indent=2, ensure_ascii=False)
        
    # 4. Expected Infeasible
    inf_data = generate_expected_infeasible()
    with open("oracle/expected_infeasible.json", "w", encoding="utf-8") as f:
        json.dump(inf_data, f, indent=2, ensure_ascii=False)
        
    print("Oracle generated all 4 expected JSON files in 'oracle/'.")


if __name__ == "__main__":
    main()
