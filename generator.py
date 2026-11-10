"""Generador de variantes sintéticas (M1) y mundos procedurales multi-dominio (M2/M3) con Oráculo Genérico."""
import os
import math
import json
import random
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from scipy.optimize import minimize
import yaml


class GenericWorldOracle:
    """Oráculo matemático independiente para cualquier variante M1 o mundo procedural M2/M3."""

    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec

    def evaluate_world(self, p: float, m: float, q: float, u_val: Optional[float] = None) -> Dict[str, float]:
        """Evalúa las ecuaciones estructurales exactas del mundo en (p, m, q, u)."""
        s = self.spec
        u = float(s["u_mean"] if u_val is None else u_val)
        p_ref = float(s.get("p_ref", 3.0))
        elasticity = float(s["elasticity"])
        m_coeff = float(s["m_coeff"])
        vol_thresh = float(s["volume_threshold"])
        cost_high = float(s["cost_high"])
        cost_low = float(s["cost_low"])
        hold_rate = float(s.get("hold_rate", 0.20))
        fee_rate = float(s["code_fee_rate"])
        clean = bool(s.get("clean", False))

        demand = max(0.0, u * (1.0 - elasticity * (p - p_ref)) + m_coeff * math.sqrt(max(0.0, m)))
        unit_cost = cost_low if (clean or q >= vol_thresh) else cost_high
        sales = min(demand, float(q))
        holding_cost = hold_rate * max(0.0, float(q) - sales)
        revenue = p * sales
        fees = fee_rate * revenue
        objective = revenue - fees - (unit_cost * float(q)) - m - holding_cost
        return {
            "u": u,
            "demand": demand,
            "unit_cost": unit_cost,
            "sales": sales,
            "holding_cost": holding_cost,
            "revenue": revenue,
            "fees": fees,
            "objective": objective,
            "budget_used": (unit_cost * float(q)) + m,
        }

    def solve_global_optimum(self) -> Dict[str, Any]:
        """Resuelve el óptimo global exacto por partición de regímenes (Q < threshold vs Q >= threshold)."""
        s = self.spec
        p_bounds = tuple(s["p_bounds"])
        m_bounds = tuple(s["m_bounds"])
        q_bounds = tuple(s["q_bounds"])
        max_budget = float(s["max_budget"])
        vol_thresh = float(s["volume_threshold"])
        clean = bool(s.get("clean", False))

        p0, m0, q0 = float(s["p_init"]), float(s["m_init"]), float(s["q_init"])
        base_eval = self.evaluate_world(p0, m0, q0)
        baseline_obj = base_eval["objective"]

        best_obj = -1e18
        best_x = (p0, m0, q0)

        regimes: List[Tuple[float, Tuple[float, float]]] = []
        if clean:
            regimes.append((float(s["cost_low"]), (float(q_bounds[0]), float(q_bounds[1]))))
        else:
            if q_bounds[0] < vol_thresh:
                regimes.append((float(s["cost_high"]), (float(q_bounds[0]), min(float(q_bounds[1]), vol_thresh - 1e-4))))
            if q_bounds[1] >= vol_thresh:
                regimes.append((float(s["cost_low"]), (max(float(q_bounds[0]), vol_thresh), float(q_bounds[1]))))

        for c_unit, (q_lo, q_hi) in regimes:
            if q_lo > q_hi:
                continue

            # En el óptimo sin exceso de inventario, q = demand(p, m).
            # También probamos optimización 3D completa + restricción en la variedad kink q = demand(p, m).
            def neg_obj_kink(pm: np.ndarray) -> float:
                pv, mv = float(pm[0]), float(pm[1])
                u = float(s["u_mean"])
                p_ref = float(s.get("p_ref", 3.0))
                dem = max(0.0, u * (1.0 - float(s["elasticity"]) * (pv - p_ref)) + float(s["m_coeff"]) * math.sqrt(max(0.0, mv)))
                qv = min(max(dem, q_lo), q_hi)
                sales = min(dem, qv)
                hold = float(s.get("hold_rate", 0.20)) * max(0.0, qv - sales)
                rev = pv * sales
                fees = float(s["code_fee_rate"]) * rev
                return -(rev - fees - c_unit * qv - mv - hold)

            def budget_ineq_kink(pm: np.ndarray) -> float:
                pv, mv = float(pm[0]), float(pm[1])
                u = float(s["u_mean"])
                p_ref = float(s.get("p_ref", 3.0))
                dem = max(0.0, u * (1.0 - float(s["elasticity"]) * (pv - p_ref)) + float(s["m_coeff"]) * math.sqrt(max(0.0, mv)))
                qv = min(max(dem, q_lo), q_hi)
                return max_budget - (c_unit * qv + mv)

            for p_frac in [0.2, 0.4, 0.6, 0.8, 0.95]:
                for m_frac in [0.02, 0.1, 0.25, 0.5]:
                    p_start = p_bounds[0] + p_frac * (p_bounds[1] - p_bounds[0])
                    m_start = m_bounds[0] + m_frac * (m_bounds[1] - m_bounds[0])
                    res = minimize(
                        neg_obj_kink,
                        x0=np.array([p_start, m_start], dtype=float),
                        bounds=[p_bounds, m_bounds],
                        constraints=[{"type": "ineq", "fun": budget_ineq_kink}],
                        method="SLSQP",
                        options={"ftol": 1e-11, "maxiter": 300},
                    )
                    pv, mv = float(res.x[0]), float(res.x[1])
                    u = float(s["u_mean"])
                    p_ref = float(s.get("p_ref", 3.0))
                    dem = max(0.0, u * (1.0 - float(s["elasticity"]) * (pv - p_ref)) + float(s["m_coeff"]) * math.sqrt(max(0.0, mv)))
                    qv = min(max(dem, q_lo), q_hi)
                    if c_unit * qv + mv <= max_budget + 1e-5:
                        ev = self.evaluate_world(pv, mv, qv)
                        if ev["objective"] > best_obj:
                            best_obj = ev["objective"]
                            best_x = (pv, mv, qv)

        names = s["node_names"]
        return {
            "baseline_objective": baseline_obj,
            "optimal_objective": best_obj,
            "improvement": best_obj - baseline_obj,
            "optimal_controls": {
                names["p"]: best_x[0],
                names["m"]: best_x[1],
                names["q"]: best_x[2],
            },
            "expected_categories": {
                names["u"]: "exogenous",
                names["p"]: "control",
                names["m"]: "control",
                names["q"]: "control",
                names["demand"]: "state",
                names["unit_cost"]: "state",
                names["sales"]: "state",
                names["holding_cost"]: "state",
                names["revenue"]: "state",
                names["fees"]: "state",
                names["objective"]: "target",
            },
            "expected_dead_nodes": [] if clean else [names["dead"]],
        }


class GoldenCaseGenerator:
    """Genera variantes del Golden Case (numéricas y estructurales M1) con su propio oráculo."""

    def __init__(self, base_seed: int = 100):
        self.base_seed = base_seed

    def generate_variant(self, variant_id: int, output_dir: str, clean: bool = False) -> Dict[str, Any]:
        """Genera una variante completa con su propio oráculo y código (compatible con N5)."""
        rng = random.Random(self.base_seed + variant_id)

        base_demand_mean = round(rng.uniform(800.0, 1200.0), 1)
        price_elasticity = round(rng.uniform(0.10, 0.20), 3)
        marketing_coeff = round(rng.uniform(6.0, 10.0), 2)
        volume_threshold = rng.choice([600, 700, 800, 900])
        cost_high = round(rng.uniform(1.15, 1.35), 2)
        cost_low = 1.00
        code_fee_rate = round(rng.uniform(0.025, 0.045), 3)
        doc_fee_rate = code_fee_rate if clean else 0.02
        max_budget = round(rng.uniform(2200.0, 2800.0), 0)

        os.makedirs(output_dir, exist_ok=True)
        tests_dir = os.path.join(output_dir, "tests")
        os.makedirs(tests_dir, exist_ok=True)

        # 1. config.py
        config_code = f'''"""Configuración para variante {variant_id}."""
PRICE: float = 3.50
MARKETING_BUDGET: float = 500.00
ORDER_QTY: int = 500
PRICE_BOUNDS = (2.0, 6.0)
MARKETING_BOUNDS = (0.0, 2000.0)
ORDER_QTY_BOUNDS = (0, 1500)
MAX_BUDGET: float = {max_budget}
'''
        with open(os.path.join(output_dir, "config.py"), "w", encoding="utf-8") as f:
            f.write(config_code)

        # 2. environment.py
        env_code = f'''"""Entorno exógeno para variante {variant_id}."""
import random
BASE_DEMAND_MEAN: float = {base_demand_mean}
BASE_DEMAND_STD: float = 80.0

def sample_base_demand(seed=None) -> float:
    if seed is not None:
        rng = random.Random(seed)
        return max(0.0, rng.gauss(BASE_DEMAND_MEAN, BASE_DEMAND_STD))
    return max(0.0, random.gauss(BASE_DEMAND_MEAN, BASE_DEMAND_STD))

def get_nominal_base_demand() -> float:
    return BASE_DEMAND_MEAN
'''
        with open(os.path.join(output_dir, "environment.py"), "w", encoding="utf-8") as f:
            f.write(env_code)

        # 3. demand.py
        demand_code = f'''"""Demanda para variante {variant_id}."""
import math

def calculate_demand(base_demand: float, price: float, marketing_budget: float) -> float:
    price_factor = 1.0 - {price_elasticity} * (price - 3.0)
    marketing_lift = {marketing_coeff} * math.sqrt(max(0.0, marketing_budget))
    return max(0.0, base_demand * price_factor + marketing_lift)
'''
        with open(os.path.join(output_dir, "demand.py"), "w", encoding="utf-8") as f:
            f.write(demand_code)

        # 4. inventory.py
        if clean:
            inv_code = f'''"""Inventario sano para variante {variant_id}."""
def calculate_unit_cost(order_qty: int) -> float:
    return {cost_low}

def calculate_sales(demand: float, order_qty: int) -> float:
    return min(demand, float(order_qty))

def calculate_holding_cost(order_qty: int, sales: float) -> float:
    return 0.20 * max(0.0, float(order_qty) - sales)
'''
        else:
            inv_code = f'''"""Inventario con trampas para variante {variant_id}."""
def calculate_unit_cost(order_qty: int) -> float:
    if order_qty < {volume_threshold}:
        return {cost_high}
    return {cost_low}

def calculate_sales(demand: float, order_qty: int) -> float:
    return min(demand, float(order_qty))

def calculate_holding_cost(order_qty: int, sales: float) -> float:
    return 0.20 * max(0.0, float(order_qty) - sales)

def optimize_order() -> int:
    return 500
'''
        with open(os.path.join(output_dir, "inventory.py"), "w", encoding="utf-8") as f:
            f.write(inv_code)

        # 5. finance.py
        if clean:
            fin_code = f'''"""Finanzas sin bugs para variante {variant_id}."""
def calculate_revenue(price: float, sales: float) -> float:
    return price * sales

def calculate_fees(revenue: float) -> float:
    return {code_fee_rate} * revenue

def calculate_profit(revenue: float, fees: float, unit_cost: float, order_qty: int, marketing_budget: float, holding_cost: float) -> float:
    return revenue - fees - (unit_cost * float(order_qty)) - marketing_budget - holding_cost

class Ledger:
    def __init__(self):
        self.entries = []
    def record_credit(self, amount: float, desc: str = ""):
        self.entries.append({{"type": "credit", "amount": float(amount)}})
    def record_debit(self, amount: float, desc: str = ""):
        self.entries.append({{"type": "debit", "amount": float(amount)}})
    def is_balanced(self) -> bool:
        credits = sum(e["amount"] for e in self.entries if e["type"] == "credit")
        debits = sum(e["amount"] for e in self.entries if e["type"] == "debit")
        return abs(credits - debits) < 1e-6

def process_refund(ledger: Ledger, amount: float):
    ledger.record_credit(amount, "refund_issued")
    ledger.record_debit(amount, "refund_cash")
'''
        else:
            fin_code = f'''"""Finanzas con trampas para variante {variant_id}."""
def calculate_revenue(price: float, sales: float) -> float:
    return price * sales

def calculate_fees(revenue: float) -> float:
    return {code_fee_rate} * revenue

def calculate_profit(revenue: float, fees: float, unit_cost: float, order_qty: int, marketing_budget: float, holding_cost: float) -> float:
    return revenue - fees - (unit_cost * float(order_qty)) - marketing_budget - holding_cost

class Ledger:
    def __init__(self):
        self.entries = []
    def record_credit(self, amount: float, desc: str = ""):
        self.entries.append({{"type": "credit", "amount": float(amount)}})
    def record_debit(self, amount: float, desc: str = ""):
        self.entries.append({{"type": "debit", "amount": float(amount)}})
    def is_balanced(self) -> bool:
        credits = sum(e["amount"] for e in self.entries if e["type"] == "credit")
        debits = sum(e["amount"] for e in self.entries if e["type"] == "debit")
        return abs(credits - debits) < 1e-6

def process_refund(ledger: Ledger, amount: float):
    ledger.record_credit(amount, "refund_issued")
    ledger.record_debit(amount, "refund_cash")
    ledger.record_debit(amount, "refund_tax_adjustment")
'''
        with open(os.path.join(output_dir, "finance.py"), "w", encoding="utf-8") as f:
            f.write(fin_code)

        # 6. marketing.py
        if clean:
            mkt_code = f'''"""Marketing para variante limpia {variant_id} (sin nodos muertos)."""
# En la variante limpia, no se generan métricas muertas desacopladas de J.
'''
        else:
            mkt_code = f'''"""Marketing con nodo muerto para variante {variant_id}."""
import math

def calculate_brand_awareness(marketing_budget: float) -> float:
    return 12.5 * math.log1p(max(0.0, marketing_budget))
'''
        with open(os.path.join(output_dir, "marketing.py"), "w", encoding="utf-8") as f:
            f.write(mkt_code)

        # 7. README.md
        readme_code = f'''# Variante {variant_id}

- Comisión pasarela: {doc_fee_rate * 100:.1f}%
- Presupuesto máximo: {max_budget}
'''
        with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_code)

        return {
            "variant_id": variant_id,
            "clean": clean,
            "base_demand_mean": base_demand_mean,
            "price_elasticity": price_elasticity,
            "marketing_coeff": marketing_coeff,
            "volume_threshold": volume_threshold,
            "code_fee_rate": code_fee_rate,
            "doc_fee_rate": doc_fee_rate,
            "cost_high": cost_high,
            "cost_low": cost_low,
            "max_budget": max_budget,
        }

    def generate_m1_structural_variant(
        self,
        variant_id: int,
        output_dir: str,
        style: str = "renamed",
        clean: bool = False,
    ) -> Dict[str, Any]:
        """
        Genera una variante estructural M1 del caso base donde cambian:
        - "renamed": todos los nombres de variables, funciones y archivos son distintos.
        - "split_helpers": las fórmulas se dividen en funciones auxiliares privadas (_helper) llamadas en cadena.
        - "merged_monolith": todo el sistema se fusiona en un único archivo monolítico con orden reordenado.
        - "oop_service": toda la lógica causal está encapsulada como métodos de una clase de servicio.
        """
        rng = random.Random(self.base_seed + 500 + variant_id)
        u_mean = round(rng.uniform(850.0, 1150.0), 1)
        elasticity = round(rng.uniform(0.11, 0.19), 3)
        m_coeff = round(rng.uniform(6.5, 9.5), 2)
        vol_thresh = rng.choice([650, 750, 850])
        cost_high = round(rng.uniform(1.18, 1.32), 2)
        cost_low = 1.00
        code_fee_rate = round(rng.uniform(0.028, 0.042), 3)
        doc_fee_rate = code_fee_rate if clean else 0.015
        max_budget = round(rng.uniform(2300.0, 2700.0), 0)

        if os.path.exists(output_dir):
            for root, dirs, files in os.walk(output_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        os.makedirs(output_dir, exist_ok=True)

        node_names = {
            "u": "inflow_baseline",
            "p": "unit_tariff",
            "m": "campaign_outlay",
            "q": "batch_quota",
            "demand": "projected_requests",
            "unit_cost": "procurement_rate",
            "sales": "fulfilled_units",
            "holding_cost": "storage_penalty",
            "revenue": "gross_receipts",
            "fees": "platform_levy",
            "objective": "net_surplus",
            "dead": "vanity_score",
        }

        params_header = f'''"""Parámetros del sistema para variante estructural M1-{variant_id} ({style})."""
UNIT_TARIFF: float = 3.50
CAMPAIGN_OUTLAY: float = 500.00
BATCH_QUOTA: int = 500
UNIT_TARIFF_BOUNDS = (2.0, 6.0)
CAMPAIGN_OUTLAY_BOUNDS = (0.0, 2000.0)
BATCH_QUOTA_BOUNDS = (0, 1500)
MAX_BUDGET: float = {max_budget}
INFLOW_BASELINE_MEAN: float = {u_mean}
INFLOW_BASELINE_STD: float = 80.0
'''

        env_funcs = '''import random

def sample_inflow_baseline(seed=None) -> float:
    if seed is not None:
        rng = random.Random(seed)
        return max(0.0, rng.gauss(INFLOW_BASELINE_MEAN, INFLOW_BASELINE_STD))
    return max(0.0, random.gauss(INFLOW_BASELINE_MEAN, INFLOW_BASELINE_STD))
'''

        if style == "split_helpers":
            mech_code = f'''import math

def _tariff_multiplier(unit_tariff: float) -> float:
    return 1.0 - {elasticity} * (unit_tariff - 3.0)

def _campaign_boost(campaign_outlay: float) -> float:
    return {m_coeff} * math.sqrt(max(0.0, campaign_outlay))

def compute_projected_requests(inflow_baseline: float, unit_tariff: float, campaign_outlay: float) -> float:
    return max(0.0, inflow_baseline * _tariff_multiplier(unit_tariff) + _campaign_boost(campaign_outlay))

def _tier_cost_lookup(batch_quota: int) -> float:
    if batch_quota < {vol_thresh}:
        return {cost_high}
    return {cost_low}

def compute_procurement_rate(batch_quota: int) -> float:
    return {cost_low} if {clean} else _tier_cost_lookup(batch_quota)

def compute_fulfilled_units(projected_requests: float, batch_quota: int) -> float:
    return min(projected_requests, float(batch_quota))

def compute_storage_penalty(batch_quota: int, fulfilled_units: float) -> float:
    return 0.20 * max(0.0, float(batch_quota) - fulfilled_units)

def compute_gross_receipts(unit_tariff: float, fulfilled_units: float) -> float:
    return unit_tariff * fulfilled_units

def compute_platform_levy(gross_receipts: float) -> float:
    return {code_fee_rate} * gross_receipts

def compute_net_surplus(
    gross_receipts: float,
    platform_levy: float,
    procurement_rate: float,
    batch_quota: int,
    campaign_outlay: float,
    storage_penalty: float,
) -> float:
    return gross_receipts - platform_levy - (procurement_rate * float(batch_quota)) - campaign_outlay - storage_penalty
'''
        elif style == "oop_service":
            mech_code = f'''import math

class OperationsService:
    """Servicio OOP que encapsula las ecuaciones estructurales."""

    @staticmethod
    def compute_projected_requests(inflow_baseline: float, unit_tariff: float, campaign_outlay: float) -> float:
        factor = 1.0 - {elasticity} * (unit_tariff - 3.0)
        lift = {m_coeff} * math.sqrt(max(0.0, campaign_outlay))
        return max(0.0, inflow_baseline * factor + lift)

    @staticmethod
    def compute_procurement_rate(batch_quota: int) -> float:
        if {not clean} and batch_quota < {vol_thresh}:
            return {cost_high}
        return {cost_low}

    @staticmethod
    def compute_fulfilled_units(projected_requests: float, batch_quota: int) -> float:
        return min(projected_requests, float(batch_quota))

    @staticmethod
    def compute_storage_penalty(batch_quota: int, fulfilled_units: float) -> float:
        return 0.20 * max(0.0, float(batch_quota) - fulfilled_units)

    @staticmethod
    def compute_gross_receipts(unit_tariff: float, fulfilled_units: float) -> float:
        return unit_tariff * fulfilled_units

    @staticmethod
    def compute_platform_levy(gross_receipts: float) -> float:
        return {code_fee_rate} * gross_receipts

    @staticmethod
    def compute_net_surplus(
        gross_receipts: float,
        platform_levy: float,
        procurement_rate: float,
        batch_quota: int,
        campaign_outlay: float,
        storage_penalty: float,
    ) -> float:
        return gross_receipts - platform_levy - (procurement_rate * float(batch_quota)) - campaign_outlay - storage_penalty
'''
        else:
            mech_code = f'''import math

def compute_projected_requests(inflow_baseline: float, unit_tariff: float, campaign_outlay: float) -> float:
    factor = 1.0 - {elasticity} * (unit_tariff - 3.0)
    lift = {m_coeff} * math.sqrt(max(0.0, campaign_outlay))
    return max(0.0, inflow_baseline * factor + lift)

def compute_procurement_rate(batch_quota: int) -> float:
    if {not clean} and batch_quota < {vol_thresh}:
        return {cost_high}
    return {cost_low}

def compute_fulfilled_units(projected_requests: float, batch_quota: int) -> float:
    return min(projected_requests, float(batch_quota))

def compute_storage_penalty(batch_quota: int, fulfilled_units: float) -> float:
    return 0.20 * max(0.0, float(batch_quota) - fulfilled_units)

def compute_gross_receipts(unit_tariff: float, fulfilled_units: float) -> float:
    return unit_tariff * fulfilled_units

def compute_platform_levy(gross_receipts: float) -> float:
    return {code_fee_rate} * gross_receipts

def compute_net_surplus(
    gross_receipts: float,
    platform_levy: float,
    procurement_rate: float,
    batch_quota: int,
    campaign_outlay: float,
    storage_penalty: float,
) -> float:
    return gross_receipts - platform_levy - (procurement_rate * float(batch_quota)) - campaign_outlay - storage_penalty
'''

        traps_extra = ""
        if not clean:
            traps_extra = '''
def compute_vanity_score(campaign_outlay: float) -> float:
    return 14.0 * math.log1p(max(0.0, campaign_outlay))

def plan_fixed_quota() -> int:
    return 500

class JournalBook:
    def __init__(self):
        self.entries = []
    def add_credit(self, val: float, tag: str = ""):
        self.entries.append({"type": "credit", "amount": float(val)})
    def add_debit(self, val: float, tag: str = ""):
        self.entries.append({"type": "debit", "amount": float(val)})

def settle_reversal(book: JournalBook, val: float):
    book.add_credit(val, "rev_credit")
    book.add_debit(val, "rev_debit_1")
    book.add_debit(val, "rev_debit_2")
'''
        else:
            traps_extra = '''
class JournalBook:
    def __init__(self):
        self.entries = []
    def add_credit(self, val: float, tag: str = ""):
        self.entries.append({"type": "credit", "amount": float(val)})
    def add_debit(self, val: float, tag: str = ""):
        self.entries.append({"type": "debit", "amount": float(val)})

def settle_reversal(book: JournalBook, val: float):
    book.add_credit(val, "rev_credit")
    book.add_debit(val, "rev_debit_1")
'''

        if style == "merged_monolith":
            with open(os.path.join(output_dir, "monolith_system.py"), "w", encoding="utf-8") as f:
                f.write(params_header + "\n" + env_funcs + "\n" + mech_code + "\n" + traps_extra)
        else:
            with open(os.path.join(output_dir, "settings_mod.py"), "w", encoding="utf-8") as f:
                f.write(params_header + "\n" + env_funcs)
            with open(os.path.join(output_dir, "core_mechanisms.py"), "w", encoding="utf-8") as f:
                f.write(mech_code + "\n" + traps_extra)

        readme_code = f'''# M1 Structural Variant {variant_id} ({style})

- Tasa de plataforma: {doc_fee_rate * 100:.1f}%
- Presupuesto operativo máximo: {max_budget}
'''
        with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_code)

        spec = {
            "variant_id": variant_id,
            "style": style,
            "clean": clean,
            "u_mean": u_mean,
            "u_std": 80.0,
            "p_ref": 3.0,
            "elasticity": elasticity,
            "m_coeff": m_coeff,
            "volume_threshold": vol_thresh,
            "cost_high": cost_high,
            "cost_low": cost_low,
            "hold_rate": 0.20,
            "code_fee_rate": code_fee_rate,
            "doc_fee_rate": doc_fee_rate,
            "max_budget": max_budget,
            "p_init": 3.50,
            "m_init": 500.0,
            "q_init": 500,
            "p_bounds": (2.0, 6.0),
            "m_bounds": (0.0, 2000.0),
            "q_bounds": (0, 1500),
            "node_names": node_names,
        }
        oracle = GenericWorldOracle(spec)
        spec["oracle"] = oracle.solve_global_optimum()
        return spec


class LogisticsMultiStageOracle:
    """Oráculo analítico exacto independiente para Mundo Heterogéneo A (Logística Multi-Etapa)."""

    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec

    def evaluate_world(
        self,
        p: float,
        m: float,
        q1: float,
        q2: float,
        u1_val: Optional[float] = None,
        u2_val: Optional[float] = None,
    ) -> Dict[str, float]:
        s = self.spec
        u1 = float(s["u1_mean"] if u1_val is None else u1_val)
        u2 = float(s["u2_mean"] if u2_val is None else u2_val)
        p_ref = float(s.get("p_ref", 3.0))
        el1 = float(s["elasticity1"])
        el2 = float(s["elasticity2"])
        beta1 = float(s["beta1"])
        vol_thresh = float(s["volume_threshold"])
        cost_high = float(s["cost_high"])
        cost_low = float(s["cost_low"])
        cost2 = float(s.get("cost2", 0.90))
        gamma1 = float(s.get("gamma1", 0.0006))
        gamma2 = float(s.get("gamma2", 0.0005))
        h1 = float(s.get("hold_rate1", 0.20))
        h2 = float(s.get("hold_rate2", 0.15))
        fee_rate = float(s["code_fee_rate"])
        clean = bool(s.get("clean", False))

        d1 = max(0.0, u1 * (1.0 - el1 * (p - p_ref)) + beta1 * math.log1p(max(0.0, m)))
        d2 = max(0.0, u2 * (1.0 - el2 * (p - p_ref)))
        s1 = min(d1, float(q1))
        s2 = min(s1 + d2, float(q2))
        c_unit1 = cost_low if (clean or q1 >= vol_thresh) else cost_high
        c1 = c_unit1 * float(q1) + gamma1 * float(q1) * float(q1)
        c2 = cost2 * float(q2) + gamma2 * float(q2) * float(q2)
        empty_mile = h1 * max(0.0, float(q1) - s1) + h2 * max(0.0, float(q2) - s2)
        revenue = p * s1 + (p + 0.80) * s2
        fees = fee_rate * revenue
        objective = revenue - fees - c1 - c2 - m - empty_mile
        budget_used = c1 + c2 + m
        return {
            "u1": u1,
            "u2": u2,
            "d1": d1,
            "d2": d2,
            "s1": s1,
            "s2": s2,
            "c1": c1,
            "c2": c2,
            "empty_mile": empty_mile,
            "revenue": revenue,
            "fees": fees,
            "objective": objective,
            "budget_used": budget_used,
        }

    def solve_global_optimum(self) -> Dict[str, Any]:
        s = self.spec
        p_bounds = tuple(s["p_bounds"])
        m_bounds = tuple(s["m_bounds"])
        q1_bounds = tuple(s["q1_bounds"])
        q2_bounds = tuple(s["q2_bounds"])
        max_budget = float(s["max_budget"])
        vol_thresh = float(s["volume_threshold"])
        clean = bool(s.get("clean", False))

        p0 = float(s["p_init"])
        m0 = float(s["m_init"])
        q1_0 = float(s["q1_init"])
        q2_0 = float(s["q2_init"])
        base_eval = self.evaluate_world(p0, m0, q1_0, q2_0)
        baseline_obj = base_eval["objective"]

        best_obj = -1e18
        best_x = (p0, m0, q1_0, q2_0)

        regimes: List[Tuple[float, float]] = []
        if clean:
            regimes.append((float(q1_bounds[0]), float(q1_bounds[1])))
        else:
            if q1_bounds[0] < vol_thresh:
                regimes.append((float(q1_bounds[0]), min(float(q1_bounds[1]), vol_thresh - 1e-4)))
            if q1_bounds[1] >= vol_thresh:
                regimes.append((max(float(q1_bounds[0]), vol_thresh), float(q1_bounds[1])))

        for q1_lo, q1_hi in regimes:
            if q1_lo > q1_hi:
                continue

            def neg_obj_cont(x: np.ndarray) -> float:
                ev = self.evaluate_world(float(x[0]), float(x[1]), float(x[2]), float(x[3]))
                return -ev["objective"]

            def budget_cons_cont(x: np.ndarray) -> float:
                ev = self.evaluate_world(float(x[0]), float(x[1]), float(x[2]), float(x[3]))
                return max_budget - ev["budget_used"]

            q1_init = 0.5 * (q1_lo + q1_hi)
            q2_init = 0.5 * (q2_bounds[0] + q2_bounds[1])
            res_cont = minimize(
                neg_obj_cont,
                np.array([p0, m0, q1_init, q2_init], dtype=float),
                bounds=[p_bounds, m_bounds, (q1_lo, q1_hi), q2_bounds],
                constraints=[{"type": "ineq", "fun": budget_cons_cont}],
                method="SLSQP",
                options={"ftol": 1e-11, "maxiter": 120}
            )

            q1_star = float(res_cont.x[2])
            q2_star = float(res_cont.x[3])

            cands_q1 = {int(round(q1_star)) + d for d in (-3, -2, -1, 0, 1, 2, 3)}
            if not clean:
                cands_q1.update({int(vol_thresh), int(vol_thresh) - 1})
            cands_q1 = sorted(c for c in cands_q1 if q1_lo <= c <= q1_hi)

            cands_q2 = sorted(c for c in {int(round(q2_star)) + d for d in (-3, -2, -1, 0, 1, 2, 3)} if q2_bounds[0] <= c <= q2_bounds[1])

            for q1_cand in cands_q1:
                for q2_cand in cands_q2:
                    def neg_obj_pm(pm: np.ndarray) -> float:
                        ev = self.evaluate_world(float(pm[0]), float(pm[1]), float(q1_cand), float(q2_cand))
                        return -ev["objective"]

                    def budget_cons_pm(pm: np.ndarray) -> float:
                        ev = self.evaluate_world(float(pm[0]), float(pm[1]), float(q1_cand), float(q2_cand))
                        return max_budget - ev["budget_used"]

                    res_pm = minimize(
                        neg_obj_pm,
                        np.array([float(res_cont.x[0]), float(res_cont.x[1])], dtype=float),
                        bounds=[p_bounds, m_bounds],
                        constraints=[{"type": "ineq", "fun": budget_cons_pm}],
                        method="SLSQP",
                        options={"ftol": 1e-12, "maxiter": 80}
                    )
                    ev_cand = self.evaluate_world(float(res_pm.x[0]), float(res_pm.x[1]), float(q1_cand), float(q2_cand))
                    if ev_cand["budget_used"] <= max_budget + 1e-4 and ev_cand["objective"] > best_obj:
                        best_obj = ev_cand["objective"]
                        best_x = (float(res_pm.x[0]), float(res_pm.x[1]), float(q1_cand), float(q2_cand))

        names = s["node_names"]
        return {
            "baseline_objective": baseline_obj,
            "optimal_objective": best_obj,
            "improvement": best_obj - baseline_obj,
            "optimal_controls": {
                names["p"]: best_x[0],
                names["m"]: best_x[1],
                names["q1"]: int(round(best_x[2])),
                names["q2"]: int(round(best_x[3])),
            },
            "expected_categories": {
                names["u1"]: "exogenous",
                names["u2"]: "exogenous",
                names["p"]: "control",
                names["m"]: "control",
                names["q1"]: "control",
                names["q2"]: "control",
                names["d1"]: "state",
                names["d2"]: "state",
                names["s1"]: "state",
                names["s2"]: "state",
                names["c1"]: "state",
                names["c2"]: "state",
                names["hold"]: "state",
                names["revenue"]: "state",
                names["fees"]: "state",
                names["objective"]: "target",
            },
            "expected_dead_nodes": [] if clean else [names["dead"]],
            "has_step_discontinuity": not clean,
            "has_kinks": True,
        }


class SaaSMultiProductOracle:
    """Oráculo analítico exacto independiente para Mundo Heterogéneo B (SaaS B2B Multi-Producto)."""

    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec

    def evaluate_world(
        self,
        p1: float,
        p2: float,
        m: float,
        u1_val: Optional[float] = None,
        u2_val: Optional[float] = None,
    ) -> Dict[str, float]:
        s = self.spec
        u1 = float(s["u1_mean"] if u1_val is None else u1_val)
        u2 = float(s["u2_mean"] if u2_val is None else u2_val)
        p1_ref = float(s.get("p1_ref", 25.0))
        p2_ref = float(s.get("p2_ref", 50.0))
        el1 = float(s["elasticity1"])
        el2 = float(s["elasticity2"])
        beta12 = float(s["cross_beta12"])
        beta21 = float(s["cross_beta21"])
        m1 = float(s["m_coeff1"])
        m2 = float(s["m_coeff2"])
        c1 = float(s["c_unit1"])
        c2 = float(s["c_unit2"])
        tier_thresh = float(s.get("tier_scale", 800.0))
        kappa = float(s.get("tier_quad_coeff", 0.001))
        fee_rate = float(s["code_fee_rate"])

        d1 = max(0.0, u1 * (1.0 - el1 * (p1 - p1_ref)) + beta12 * (p2 - p2_ref) + m1 * math.log1p(max(0.0, m)))
        d2 = max(0.0, u2 * (1.0 - el2 * (p2 - p2_ref)) + beta21 * (p1 - p1_ref) + m2 * math.log1p(max(0.0, m)))
        total_d = d1 + d2
        tier_excess = max(0.0, total_d - tier_thresh)
        infra_cost = c1 * d1 + c2 * d2 + kappa * tier_excess * tier_excess
        revenue = p1 * d1 + p2 * d2
        fees = fee_rate * revenue
        objective = revenue - fees - infra_cost - m
        budget_used = infra_cost + m
        return {
            "u1": u1,
            "u2": u2,
            "d1": d1,
            "d2": d2,
            "infra_cost": infra_cost,
            "revenue": revenue,
            "fees": fees,
            "objective": objective,
            "budget_used": budget_used,
        }

    def solve_global_optimum(self) -> Dict[str, Any]:
        s = self.spec
        p1_bounds = tuple(s["p1_bounds"])
        p2_bounds = tuple(s["p2_bounds"])
        m_bounds = tuple(s["m_bounds"])
        max_budget = float(s["max_budget"])

        p1_0, p2_0, m0 = float(s["p1_init"]), float(s["p2_init"]), float(s["m_init"])
        base_eval = self.evaluate_world(p1_0, p2_0, m0)
        baseline_obj = base_eval["objective"]

        def neg_obj(x: np.ndarray) -> float:
            ev = self.evaluate_world(float(x[0]), float(x[1]), float(x[2]))
            return -ev["objective"]

        def budget_cons(x: np.ndarray) -> float:
            ev = self.evaluate_world(float(x[0]), float(x[1]), float(x[2]))
            return max_budget - ev["budget_used"]

        best_obj = -1e18
        best_x = (p1_0, p2_0, m0)

        for p1_frac in (0.3, 0.7):
            for p2_frac in (0.3, 0.7):
                for m_frac in (0.2, 0.6):
                    x0 = np.array([
                        p1_bounds[0] + p1_frac * (p1_bounds[1] - p1_bounds[0]),
                        p2_bounds[0] + p2_frac * (p2_bounds[1] - p2_bounds[0]),
                        m_bounds[0] + m_frac * (m_bounds[1] - m_bounds[0]),
                    ], dtype=float)
                    res = minimize(
                        neg_obj,
                        x0,
                        bounds=[p1_bounds, p2_bounds, m_bounds],
                        constraints=[{"type": "ineq", "fun": budget_cons}],
                        method="SLSQP",
                        options={"ftol": 1e-12, "maxiter": 150}
                    )
                    ev = self.evaluate_world(float(res.x[0]), float(res.x[1]), float(res.x[2]))
                    if ev["budget_used"] <= max_budget + 1e-4 and ev["objective"] > best_obj:
                        best_obj = ev["objective"]
                        best_x = (float(res.x[0]), float(res.x[1]), float(res.x[2]))

        names = s["node_names"]
        return {
            "baseline_objective": baseline_obj,
            "optimal_objective": best_obj,
            "improvement": best_obj - baseline_obj,
            "optimal_controls": {
                names["p1"]: best_x[0],
                names["p2"]: best_x[1],
                names["m"]: best_x[2],
            },
            "expected_categories": {
                names["u1"]: "exogenous",
                names["u2"]: "exogenous",
                names["p1"]: "control",
                names["p2"]: "control",
                names["m"]: "control",
                names["d1"]: "state",
                names["d2"]: "state",
                names["infra"]: "state",
                names["revenue"]: "state",
                names["fees"]: "state",
                names["objective"]: "target",
            },
            "expected_dead_nodes": [] if s.get("clean", False) else [names["dead"]],
            "has_step_discontinuity": False,
            "has_kinks": True,
        }


class ProceduralWorldGenerator:
    """
    Generador de Mundos Procedurales Multi-Dominio Heterogéneos Reales (M2 y M3).
    Rompe totalmente el isomorfismo con el caso base cafetería:
    - fleet_logistics: Mundo Heterogéneo A (Logística Multi-Etapa: 2 variables discretas Q1, Q2, 2 cuellos de botella min en serie, costes convexos cuadráticos, log1p).
    - saas_cloud: Mundo Heterogéneo B (SaaS B2B Multi-Producto: puramente continuo P1, P2, M, demandas cruzadas canibalizadas, infra tier continuo sin Heaviside).
    - energy_microgrid: Micro-red solar y despacho de baterías.
    """

    DOMAIN_TEMPLATES: Dict[str, Dict[str, Any]] = {
        "saas_cloud": {
            "title": "Plataforma Cloud SaaS & Licenciamiento de Nodos",
            "node_names": {
                "u1": "organic_leads",
                "u2": "partner_referrals",
                "p1": "seat_tariff",
                "p2": "enterprise_tariff",
                "m": "ad_campaigns",
                "d1": "active_signups",
                "d2": "enterprise_signups",
                "infra": "infra_tier_cost",
                "revenue": "gross_arr",
                "fees": "gateway_commission",
                "objective": "net_ebitda",
                "dead": "social_impressions",
            },
            "files": ("cloud_params.py", "acquisition_engine.py", "cluster_ops.py", "billing_ledger.py"),
            "subopt_fn": "lock_seat_tariff",
            "tx_fn": "reconcile_credit_note",
            "csv_name": "signups_history.csv",
        },
        "fleet_logistics": {
            "title": "Flota de Transporte Frigorífico Multi-Etapa",
            "node_names": {
                "u1": "freight_requests",
                "u2": "transfer_requests",
                "p": "freight_rate",
                "m": "broker_incentive",
                "q1": "fleet_stage1_capacity",
                "q2": "fleet_stage2_capacity",
                "d1": "stage1_demand",
                "d2": "stage2_demand",
                "s1": "delivered_stage1",
                "s2": "delivered_stage2",
                "c1": "stage1_cost",
                "c2": "stage2_cost",
                "hold": "empty_mile_cost",
                "revenue": "gross_haulage",
                "fees": "customs_levy",
                "objective": "operating_margin",
                "dead": "driver_app_score",
            },
            "files": ("fleet_config.py", "dispatch_demand.py", "capacity_routing.py", "freight_settlement.py"),
            "subopt_fn": "schedule_fleet_stage1_capacity",
            "tx_fn": "process_claim_voucher",
            "csv_name": "dispatch_history.csv",
        },
        "energy_microgrid": {
            "title": "Micro-Red Solar Industrial & Despacho de Baterías",
            "node_names": {
                "u": "baseline_load_mwh",
                "p": "tariff_mwh",
                "m": "demand_response_rebate",
                "q": "committed_storage_mwh",
                "demand": "grid_demand_mwh",
                "unit_cost": "storage_om_rate",
                "sales": "dispatched_mwh",
                "holding_cost": "curtailment_penalty",
                "revenue": "gross_utility_rev",
                "fees": "transmission_toll",
                "objective": "net_grid_yield",
                "dead": "green_pr_index",
            },
            "files": ("grid_settings.py", "load_forecast.py", "storage_dispatch.py", "utility_accounting.py"),
            "subopt_fn": "lock_storage_commitment",
            "tx_fn": "settle_imbalance_credit",
            "csv_name": "grid_telemetry.csv",
        },
    }

    def __init__(self, base_seed: int = 2000):
        self.base_seed = base_seed

    def generate_world(
        self,
        world_id: int,
        output_dir: str,
        domain: Optional[str] = None,
        clean: bool = False,
        include_csv: bool = True,
    ) -> Dict[str, Any]:
        """Genera un mundo procedural completo en `output_dir` y devuelve su especificación + solución del oráculo."""
        rng = random.Random(self.base_seed + world_id * 37)
        domain_keys = list(self.DOMAIN_TEMPLATES.keys())
        if domain is None:
            domain = domain_keys[world_id % len(domain_keys)]

        if os.path.exists(output_dir):
            for root, dirs, files in os.walk(output_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        os.makedirs(output_dir, exist_ok=True)

        if domain == "fleet_logistics":
            return self._generate_fleet_logistics(world_id, output_dir, rng, clean, include_csv)
        elif domain == "saas_cloud":
            return self._generate_saas_cloud(world_id, output_dir, rng, clean, include_csv)
        else:
            return self._generate_energy_microgrid(world_id, output_dir, rng, clean, include_csv)

    def _generate_fleet_logistics(
        self,
        world_id: int,
        output_dir: str,
        rng: random.Random,
        clean: bool,
        include_csv: bool,
    ) -> Dict[str, Any]:
        """Mundo Heterogéneo A: Logística Multi-Etapa con 2 discretas, 2 min seriales y costes convexos."""
        tmpl = self.DOMAIN_TEMPLATES["fleet_logistics"]
        names = tmpl["node_names"]

        u1_mean = round(rng.uniform(700.0, 1100.0), 1)
        u1_std = round(rng.uniform(50.0, 90.0), 1)
        u2_mean = round(rng.uniform(400.0, 700.0), 1)
        u2_std = round(rng.uniform(30.0, 60.0), 1)
        p_ref = round(rng.uniform(3.0, 4.0), 2)
        p_init = round(p_ref + rng.uniform(0.2, 0.6), 2)
        p_bounds = (round(p_ref - 1.0, 1), round(p_ref + 3.0, 1))
        m_init = 350.0
        m_bounds = (0.0, 1500.0)
        q1_init = 450
        q1_bounds = (100, 750)
        q2_init = 600
        q2_bounds = (150, 950)

        elasticity1 = round(rng.uniform(0.08, 0.15), 3)
        elasticity2 = round(rng.uniform(0.06, 0.12), 3)
        beta1 = round(rng.uniform(20.0, 35.0), 2)
        vol_thresh = rng.choice([400, 500, 600])
        cost_high = round(rng.uniform(1.25, 1.45), 2)
        cost_low = round(rng.uniform(0.95, 1.05), 2)
        cost2 = round(rng.uniform(0.85, 0.95), 2)
        gamma1 = 0.0006
        gamma2 = 0.0005
        hold_rate1 = round(rng.uniform(0.15, 0.25), 2)
        hold_rate2 = round(rng.uniform(0.12, 0.20), 2)
        code_fee_rate = round(rng.uniform(0.025, 0.040), 3)
        doc_fee_rate = code_fee_rate if clean else 0.015
        max_budget = round(rng.uniform(2500.0, 3200.0), 0)

        f_cfg, f_dem, f_ops, f_fin = tmpl["files"]

        # 1. fleet_config.py
        cfg_code = f'''"""Configuración y entorno exógeno — {tmpl['title']} (Mundo {world_id})."""
import random

FREIGHT_RATE: float = {p_init}
BROKER_INCENTIVE: float = {m_init}
FLEET_STAGE1_CAPACITY: int = {q1_init}
FLEET_STAGE2_CAPACITY: int = {q2_init}

FREIGHT_RATE_BOUNDS = ({p_bounds[0]}, {p_bounds[1]})
BROKER_INCENTIVE_BOUNDS = ({m_bounds[0]}, {m_bounds[1]})
FLEET_STAGE1_CAPACITY_BOUNDS = ({q1_bounds[0]}, {q1_bounds[1]})
FLEET_STAGE2_CAPACITY_BOUNDS = ({q2_bounds[0]}, {q2_bounds[1]})

MAX_BUDGET: float = {max_budget}

FREIGHT_REQUESTS_MEAN: float = {u1_mean}
FREIGHT_REQUESTS_STD: float = {u1_std}
TRANSFER_REQUESTS_MEAN: float = {u2_mean}
TRANSFER_REQUESTS_STD: float = {u2_std}

def sample_freight_requests(seed=None) -> float:
    if seed is not None:
        rng = random.Random(seed)
        return max(0.0, rng.gauss(FREIGHT_REQUESTS_MEAN, FREIGHT_REQUESTS_STD))
    return max(0.0, random.gauss(FREIGHT_REQUESTS_MEAN, FREIGHT_REQUESTS_STD))

def sample_transfer_requests(seed=None) -> float:
    if seed is not None:
        rng = random.Random(seed + 1 if seed is not None else None)
        return max(0.0, rng.gauss(TRANSFER_REQUESTS_MEAN, TRANSFER_REQUESTS_STD))
    return max(0.0, random.gauss(TRANSFER_REQUESTS_MEAN, TRANSFER_REQUESTS_STD))
'''
        with open(os.path.join(output_dir, f_cfg), "w", encoding="utf-8") as f:
            f.write(cfg_code)

        # 2. dispatch_demand.py
        dead_fn = ""
        if not clean:
            dead_fn = f'''
def compute_{names['dead']}(broker_incentive: float) -> float:
    return 14.0 * math.log1p(max(0.0, broker_incentive))
'''
        dem_code = f'''"""Módulo de demanda logística multi-etapa — {tmpl['title']}."""
import math

def compute_stage1_demand(freight_requests: float, freight_rate: float, broker_incentive: float) -> float:
    price_factor = 1.0 - {elasticity1} * (freight_rate - {p_ref})
    incentive_boost = {beta1} * math.log1p(max(0.0, broker_incentive))
    return max(0.0, freight_requests * price_factor + incentive_boost)

def compute_stage2_demand(transfer_requests: float, freight_rate: float) -> float:
    price_factor = 1.0 - {elasticity2} * (freight_rate - {p_ref})
    return max(0.0, transfer_requests * price_factor)
{dead_fn}'''
        with open(os.path.join(output_dir, f_dem), "w", encoding="utf-8") as f:
            f.write(dem_code)

        # 3. capacity_routing.py
        subopt_block = ""
        if not clean:
            subopt_block = f'''
def {tmpl['subopt_fn']}() -> int:
    return 180
'''
        if clean:
            c1_body = f"    return {cost_low} * float(fleet_stage1_capacity) + {gamma1} * float(fleet_stage1_capacity) * float(fleet_stage1_capacity)"
        else:
            c1_body = f"""    if fleet_stage1_capacity < {vol_thresh}:
        c_base = {cost_high}
    else:
        c_base = {cost_low}
    return c_base * float(fleet_stage1_capacity) + {gamma1} * float(fleet_stage1_capacity) * float(fleet_stage1_capacity)"""

        ops_code = f'''"""Módulo de despacho de capacidad y cuellos de botella seriales — {tmpl['title']}."""

def compute_delivered_stage1(stage1_demand: float, fleet_stage1_capacity: int) -> float:
    return min(stage1_demand, float(fleet_stage1_capacity))

def compute_delivered_stage2(delivered_stage1: float, stage2_demand: float, fleet_stage2_capacity: int) -> float:
    return min(delivered_stage1 + stage2_demand, float(fleet_stage2_capacity))

def compute_stage1_cost(fleet_stage1_capacity: int) -> float:
{c1_body}

def compute_stage2_cost(fleet_stage2_capacity: int) -> float:
    return {cost2} * float(fleet_stage2_capacity) + {gamma2} * float(fleet_stage2_capacity) * float(fleet_stage2_capacity)

def compute_empty_mile_cost(
    fleet_stage1_capacity: int,
    delivered_stage1: float,
    fleet_stage2_capacity: int,
    delivered_stage2: float
) -> float:
    return {hold_rate1} * max(0.0, float(fleet_stage1_capacity) - delivered_stage1) + {hold_rate2} * max(0.0, float(fleet_stage2_capacity) - delivered_stage2)
{subopt_block}'''
        with open(os.path.join(output_dir, f_ops), "w", encoding="utf-8") as f:
            f.write(ops_code)

        # 4. freight_settlement.py
        tx_extra_debit = "" if clean else '    ledger.record_debit(amount, "duplicate_fuel_voucher")\n'
        fin_code = f'''"""Módulo de liquidación y margen operativo — {tmpl['title']}."""

def compute_gross_haulage(freight_rate: float, delivered_stage1: float, delivered_stage2: float) -> float:
    return freight_rate * delivered_stage1 + (freight_rate + 0.80) * delivered_stage2

def compute_customs_levy(gross_haulage: float) -> float:
    return {code_fee_rate} * gross_haulage

def compute_operating_margin(
    gross_haulage: float,
    customs_levy: float,
    stage1_cost: float,
    stage2_cost: float,
    broker_incentive: float,
    empty_mile_cost: float
) -> float:
    return gross_haulage - customs_levy - stage1_cost - stage2_cost - broker_incentive - empty_mile_cost

class AccountLedger:
    def __init__(self):
        self.entries = []
    def record_credit(self, amount: float, memo: str = ""):
        self.entries.append({{"type": "credit", "amount": float(amount)}})
    def record_debit(self, amount: float, memo: str = ""):
        self.entries.append({{"type": "debit", "amount": float(amount)}})

def {tmpl['tx_fn']}(ledger: AccountLedger, amount: float):
    ledger.record_credit(amount, "freight_settle_credit")
    ledger.record_debit(amount, "freight_settle_debit")
{tx_extra_debit}'''
        with open(os.path.join(output_dir, f_fin), "w", encoding="utf-8") as f:
            f.write(fin_code)

        # 5. README.md
        readme_code = f'''# {tmpl['title']} (World {world_id})

## Reglas de Negocio y Parámetros
- Tasa aduanera documentada: {doc_fee_rate * 100:.1f}%
- Presupuesto operativo máximo: {max_budget}
'''
        with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_code)

        # 6. cge.yaml
        yaml_spec = {
            "target": "operating_margin",
            "controls": {
                "freight_rate": {"type": "continuous", "bounds": [p_bounds[0], p_bounds[1]]},
                "broker_incentive": {"type": "continuous", "bounds": [m_bounds[0], m_bounds[1]]},
                "fleet_stage1_capacity": {"type": "discrete", "bounds": [q1_bounds[0], q1_bounds[1]]},
                "fleet_stage2_capacity": {"type": "discrete", "bounds": [q2_bounds[0], q2_bounds[1]]},
            },
            "exogenous": {
                "freight_requests": {"mean": u1_mean, "std": u1_std},
                "transfer_requests": {"mean": u2_mean, "std": u2_std},
            },
            "constraints": {
                "c_budget": {
                    "type": "hard",
                    "expression": f"stage1_cost + stage2_cost + broker_incentive <= {max_budget}",
                    "description": "Presupuesto operativo máximo de transporte"
                }
            }
        }
        with open(os.path.join(output_dir, "cge.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(yaml_spec, f, sort_keys=False)

        # 7. CSV empírico
        if include_csv:
            data_dir = os.path.join(output_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            csv_path = os.path.join(data_dir, tmpl["csv_name"])
            lines = ["day_id,season_index,freight_rate,broker_incentive,freight_requests,stage1_demand,delivered_stage1"]
            for day in range(1, 61):
                s_idx = round(0.85 + 0.35 * ((day % 12) / 11.0), 3)
                pv = round(rng.uniform(p_bounds[0] + 0.2, p_bounds[1] - 0.3), 2)
                mv = round(100.0 + 800.0 * (s_idx - 0.85) + rng.uniform(0.0, 200.0), 2)
                uv = round(u1_mean * s_idx + rng.gauss(0.0, 15.0), 2)
                dem = round(max(0.0, uv * (1.0 - elasticity1 * (pv - p_ref)) + beta1 * math.log1p(mv)), 2)
                sls = round(min(dem, float(q1_init)), 2)
                lines.append(f"{day},{s_idx},{pv},{mv},{uv},{dem},{sls}")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

        spec = {
            "world_id": world_id,
            "domain": "fleet_logistics",
            "clean": clean,
            "u1_mean": u1_mean,
            "u1_std": u1_std,
            "u2_mean": u2_mean,
            "u2_std": u2_std,
            "p_ref": p_ref,
            "elasticity1": elasticity1,
            "elasticity2": elasticity2,
            "beta1": beta1,
            "volume_threshold": vol_thresh,
            "cost_high": cost_high,
            "cost_low": cost_low,
            "cost2": cost2,
            "gamma1": gamma1,
            "gamma2": gamma2,
            "hold_rate1": hold_rate1,
            "hold_rate2": hold_rate2,
            "code_fee_rate": code_fee_rate,
            "doc_fee_rate": doc_fee_rate,
            "max_budget": max_budget,
            "p_init": p_init,
            "m_init": m_init,
            "q1_init": q1_init,
            "q2_init": q2_init,
            "p_bounds": p_bounds,
            "m_bounds": m_bounds,
            "q1_bounds": q1_bounds,
            "q2_bounds": q2_bounds,
            "node_names": names,
        }
        oracle = LogisticsMultiStageOracle(spec)
        spec["oracle"] = oracle.solve_global_optimum()
        return spec

    def _generate_saas_cloud(
        self,
        world_id: int,
        output_dir: str,
        rng: random.Random,
        clean: bool,
        include_csv: bool,
    ) -> Dict[str, Any]:
        """Mundo Heterogéneo B: SaaS B2B Multi-Producto puramente continuo con demandas cruzadas canibalizadas."""
        tmpl = self.DOMAIN_TEMPLATES["saas_cloud"]
        names = tmpl["node_names"]

        u1_mean = round(rng.uniform(500.0, 800.0), 1)
        u1_std = round(rng.uniform(30.0, 60.0), 1)
        u2_mean = round(rng.uniform(300.0, 500.0), 1)
        u2_std = round(rng.uniform(20.0, 45.0), 1)
        p1_ref = round(rng.uniform(22.0, 28.0), 2)
        p1_init = round(p1_ref + rng.uniform(2.0, 5.0), 2)
        p1_bounds = (15.0, 45.0)
        p2_ref = round(rng.uniform(45.0, 55.0), 2)
        p2_init = round(p2_ref + rng.uniform(4.0, 8.0), 2)
        p2_bounds = (30.0, 90.0)
        m_init = 400.0
        m_bounds = (0.0, 2500.0)

        elasticity1 = round(rng.uniform(0.06, 0.10), 3)
        elasticity2 = round(rng.uniform(0.04, 0.08), 3)
        cross_beta12 = round(rng.uniform(1.2, 2.2), 2)
        cross_beta21 = round(rng.uniform(0.8, 1.6), 2)
        m_coeff1 = round(rng.uniform(18.0, 28.0), 2)
        m_coeff2 = round(rng.uniform(12.0, 20.0), 2)
        c_unit1 = round(rng.uniform(1.10, 1.35), 2)
        c_unit2 = round(rng.uniform(1.85, 2.20), 2)
        tier_scale = round(rng.uniform(700.0, 900.0), 0)
        tier_quad_coeff = 0.001
        code_fee_rate = round(rng.uniform(0.025, 0.040), 3)
        doc_fee_rate = code_fee_rate if clean else 0.015
        max_budget = round(rng.uniform(3200.0, 4200.0), 0)

        f_cfg, f_dem, f_ops, f_fin = tmpl["files"]

        # 1. cloud_params.py
        cfg_code = f'''"""Configuración y entorno exógeno — {tmpl['title']} (Mundo {world_id})."""
import random

SEAT_TARIFF: float = {p1_init}
ENTERPRISE_TARIFF: float = {p2_init}
AD_CAMPAIGNS: float = {m_init}

SEAT_TARIFF_BOUNDS = ({p1_bounds[0]}, {p1_bounds[1]})
ENTERPRISE_TARIFF_BOUNDS = ({p2_bounds[0]}, {p2_bounds[1]})
AD_CAMPAIGNS_BOUNDS = ({m_bounds[0]}, {m_bounds[1]})

MAX_BUDGET: float = {max_budget}

ORGANIC_LEADS_MEAN: float = {u1_mean}
ORGANIC_LEADS_STD: float = {u1_std}
PARTNER_REFERRALS_MEAN: float = {u2_mean}
PARTNER_REFERRALS_STD: float = {u2_std}

def sample_organic_leads(seed=None) -> float:
    if seed is not None:
        rng = random.Random(seed)
        return max(0.0, rng.gauss(ORGANIC_LEADS_MEAN, ORGANIC_LEADS_STD))
    return max(0.0, random.gauss(ORGANIC_LEADS_MEAN, ORGANIC_LEADS_STD))

def sample_partner_referrals(seed=None) -> float:
    if seed is not None:
        rng = random.Random(seed + 1 if seed is not None else None)
        return max(0.0, rng.gauss(PARTNER_REFERRALS_MEAN, PARTNER_REFERRALS_STD))
    return max(0.0, random.gauss(PARTNER_REFERRALS_MEAN, PARTNER_REFERRALS_STD))
'''
        with open(os.path.join(output_dir, f_cfg), "w", encoding="utf-8") as f:
            f.write(cfg_code)

        # 2. acquisition_engine.py
        dead_fn = ""
        if not clean:
            dead_fn = f'''
def compute_{names['dead']}(ad_campaigns: float) -> float:
    return 15.0 * math.log1p(max(0.0, ad_campaigns))
'''
        dem_code = f'''"""Módulo de adquisición multi-producto con canibalización — {tmpl['title']}."""
import math

def compute_active_signups(organic_leads: float, seat_tariff: float, enterprise_tariff: float, ad_campaigns: float) -> float:
    own_factor = 1.0 - {elasticity1} * (seat_tariff - {p1_ref})
    cross_effect = {cross_beta12} * (enterprise_tariff - {p2_ref})
    promo_lift = {m_coeff1} * math.log1p(max(0.0, ad_campaigns))
    return max(0.0, organic_leads * own_factor + cross_effect + promo_lift)

def compute_enterprise_signups(partner_referrals: float, enterprise_tariff: float, seat_tariff: float, ad_campaigns: float) -> float:
    own_factor = 1.0 - {elasticity2} * (enterprise_tariff - {p2_ref})
    cross_effect = {cross_beta21} * (seat_tariff - {p1_ref})
    promo_lift = {m_coeff2} * math.log1p(max(0.0, ad_campaigns))
    return max(0.0, partner_referrals * own_factor + cross_effect + promo_lift)
{dead_fn}'''
        with open(os.path.join(output_dir, f_dem), "w", encoding="utf-8") as f:
            f.write(dem_code)

        # 3. cluster_ops.py
        subopt_block = ""
        if not clean:
            subopt_block = f'''
def {tmpl['subopt_fn']}() -> float:
    return 16.0
'''
        ops_code = f'''"""Módulo de operaciones de cluster y costes continuos por tier — {tmpl['title']}."""

def compute_infra_tier_cost(active_signups: float, enterprise_signups: float) -> float:
    total_signups = active_signups + enterprise_signups
    scale_excess = max(0.0, total_signups - {tier_scale})
    return {c_unit1} * active_signups + {c_unit2} * enterprise_signups + {tier_quad_coeff} * scale_excess * scale_excess
{subopt_block}'''
        with open(os.path.join(output_dir, f_ops), "w", encoding="utf-8") as f:
            f.write(ops_code)

        # 4. billing_ledger.py
        tx_extra_debit = "" if clean else '    ledger.record_debit(amount, "duplicate_tier_adjustment")\n'
        fin_code = f'''"""Módulo financiero de facturación y EBITDA — {tmpl['title']}."""

def compute_gross_arr(seat_tariff: float, active_signups: float, enterprise_tariff: float, enterprise_signups: float) -> float:
    return seat_tariff * active_signups + enterprise_tariff * enterprise_signups

def compute_gateway_commission(gross_arr: float) -> float:
    return {code_fee_rate} * gross_arr

def compute_net_ebitda(
    gross_arr: float,
    gateway_commission: float,
    infra_tier_cost: float,
    ad_campaigns: float
) -> float:
    return gross_arr - gateway_commission - infra_tier_cost - ad_campaigns

class AccountLedger:
    def __init__(self):
        self.entries = []
    def record_credit(self, amount: float, memo: str = ""):
        self.entries.append({{"type": "credit", "amount": float(amount)}})
    def record_debit(self, amount: float, memo: str = ""):
        self.entries.append({{"type": "debit", "amount": float(amount)}})

def {tmpl['tx_fn']}(ledger: AccountLedger, amount: float):
    ledger.record_credit(amount, "note_credit")
    ledger.record_debit(amount, "note_debit")
{tx_extra_debit}'''
        with open(os.path.join(output_dir, f_fin), "w", encoding="utf-8") as f:
            f.write(fin_code)

        # 5. README.md
        readme_code = f'''# {tmpl['title']} (World {world_id})

## Reglas de Negocio y Parámetros
- Tasa de pasarela documentada: {doc_fee_rate * 100:.1f}%
- Presupuesto operativo máximo: {max_budget}
'''
        with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_code)

        # 6. cge.yaml
        yaml_spec = {
            "target": "net_ebitda",
            "controls": {
                "seat_tariff": {"type": "continuous", "bounds": [p1_bounds[0], p1_bounds[1]]},
                "enterprise_tariff": {"type": "continuous", "bounds": [p2_bounds[0], p2_bounds[1]]},
                "ad_campaigns": {"type": "continuous", "bounds": [m_bounds[0], m_bounds[1]]},
            },
            "exogenous": {
                "organic_leads": {"mean": u1_mean, "std": u1_std},
                "partner_referrals": {"mean": u2_mean, "std": u2_std},
            },
            "constraints": {
                "c_budget": {
                    "type": "hard",
                    "expression": f"infra_tier_cost + ad_campaigns <= {max_budget}",
                    "description": "Presupuesto máximo de infraestructura y marketing"
                }
            }
        }
        with open(os.path.join(output_dir, "cge.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(yaml_spec, f, sort_keys=False)

        # 7. CSV empírico
        if include_csv:
            data_dir = os.path.join(output_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            csv_path = os.path.join(data_dir, tmpl["csv_name"])
            lines = ["day_id,season_index,seat_tariff,ad_campaigns,organic_leads,active_signups,enterprise_signups"]
            for day in range(1, 61):
                s_idx = round(0.85 + 0.35 * ((day % 12) / 11.0), 3)
                pv = round(rng.uniform(p1_bounds[0] + 1.0, p1_bounds[1] - 1.0), 2)
                mv = round(150.0 + 900.0 * (s_idx - 0.85) + rng.uniform(0.0, 250.0), 2)
                uv = round(u1_mean * s_idx + rng.gauss(0.0, 10.0), 2)
                dem = round(max(0.0, uv * (1.0 - elasticity1 * (pv - p1_ref)) + m_coeff1 * math.log1p(mv)), 2)
                ent_d = round(max(0.0, u2_mean * (1.0 - elasticity2 * (p2_init - p2_ref)) + cross_beta21 * (pv - p1_ref)), 2)
                lines.append(f"{day},{s_idx},{pv},{mv},{uv},{dem},{ent_d}")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

        spec = {
            "world_id": world_id,
            "domain": "saas_cloud",
            "clean": clean,
            "u1_mean": u1_mean,
            "u1_std": u1_std,
            "u2_mean": u2_mean,
            "u2_std": u2_std,
            "p1_ref": p1_ref,
            "p2_ref": p2_ref,
            "elasticity1": elasticity1,
            "elasticity2": elasticity2,
            "cross_beta12": cross_beta12,
            "cross_beta21": cross_beta21,
            "m_coeff1": m_coeff1,
            "m_coeff2": m_coeff2,
            "c_unit1": c_unit1,
            "c_unit2": c_unit2,
            "tier_scale": tier_scale,
            "tier_quad_coeff": tier_quad_coeff,
            "code_fee_rate": code_fee_rate,
            "doc_fee_rate": doc_fee_rate,
            "max_budget": max_budget,
            "p1_init": p1_init,
            "p2_init": p2_init,
            "m_init": m_init,
            "p1_bounds": p1_bounds,
            "p2_bounds": p2_bounds,
            "m_bounds": m_bounds,
            "node_names": names,
        }
        oracle = SaaSMultiProductOracle(spec)
        spec["oracle"] = oracle.solve_global_optimum()
        return spec

    def _generate_energy_microgrid(
        self,
        world_id: int,
        output_dir: str,
        rng: random.Random,
        clean: bool,
        include_csv: bool,
    ) -> Dict[str, Any]:
        """Micro-Red Solar y Baterías usando plantilla de flujo energético."""
        tmpl = self.DOMAIN_TEMPLATES["energy_microgrid"]
        names = tmpl["node_names"]

        u_mean = round(rng.uniform(900.0, 1400.0), 1)
        u_std = round(rng.uniform(60.0, 110.0), 1)
        p_ref = round(rng.uniform(3.0, 4.0), 2)
        p_init = round(p_ref + rng.uniform(0.3, 0.7), 2)
        p_bounds = (round(p_ref - 1.0, 1), round(p_ref + 3.5, 1))
        m_init = 500.0
        m_bounds = (0.0, 2200.0)
        q_init = 550
        q_bounds = (0, 1800)

        elasticity = round(rng.uniform(0.10, 0.18), 3)
        m_coeff = round(rng.uniform(6.5, 10.5), 2)
        vol_thresh = rng.choice([700, 800, 900, 1000])
        cost_high = round(rng.uniform(1.20, 1.40), 2)
        cost_low = round(rng.uniform(0.95, 1.05), 2)
        hold_rate = round(rng.uniform(0.15, 0.25), 2)
        code_fee_rate = round(rng.uniform(0.028, 0.045), 3)
        doc_fee_rate = code_fee_rate if clean else 0.015
        max_budget = round(rng.uniform(2400.0, 3100.0), 0)

        f_cfg, f_dem, f_ops, f_fin = tmpl["files"]
        U_UPPER = names["u"].upper()
        P_UPPER = names["p"].upper()
        M_UPPER = names["m"].upper()
        Q_UPPER = names["q"].upper()

        # 1. Archivo de configuración y entorno exógeno
        cfg_code = f'''"""Configuración y entorno exógeno — {tmpl['title']} (Mundo {world_id})."""
import random

{P_UPPER}: float = {p_init}
{M_UPPER}: float = {m_init}
{Q_UPPER}: int = {q_init}

{P_UPPER}_BOUNDS = ({p_bounds[0]}, {p_bounds[1]})
{M_UPPER}_BOUNDS = ({m_bounds[0]}, {m_bounds[1]})
{Q_UPPER}_BOUNDS = ({q_bounds[0]}, {q_bounds[1]})

MAX_BUDGET: float = {max_budget}

{U_UPPER}_MEAN: float = {u_mean}
{U_UPPER}_STD: float = {u_std}

def sample_{names['u']}(seed=None) -> float:
    if seed is not None:
        rng = random.Random(seed)
        return max(0.0, rng.gauss({U_UPPER}_MEAN, {U_UPPER}_STD))
    return max(0.0, random.gauss({U_UPPER}_MEAN, {U_UPPER}_STD))
'''
        with open(os.path.join(output_dir, f_cfg), "w", encoding="utf-8") as f:
            f.write(cfg_code)

        # 2. Archivo de demanda y nodo muerto (T1)
        dead_fn = ""
        if not clean:
            dead_fn = f'''
def compute_{names['dead']}({names['m']}: float) -> float:
    return 15.0 * math.log1p(max(0.0, {names['m']}))
'''
        dem_code = f'''"""Módulo de demanda y adquisición — {tmpl['title']}."""
import math

def compute_{names['demand']}({names['u']}: float, {names['p']}: float, {names['m']}: float) -> float:
    price_factor = 1.0 - {elasticity} * ({names['p']} - {p_ref})
    promo_lift = {m_coeff} * math.sqrt(max(0.0, {names['m']}))
    return max(0.0, {names['u']} * price_factor + promo_lift)
{dead_fn}'''
        with open(os.path.join(output_dir, f_dem), "w", encoding="utf-8") as f:
            f.write(dem_code)

        # 3. Archivo de operaciones, escalón T5, kink T6 y subóptimo T4
        subopt_block = ""
        if not clean:
            subopt_block = f'''
def {tmpl['subopt_fn']}() -> int:
    return {q_init}
'''
        if clean:
            cost_body = f"    return {cost_low}"
        else:
            cost_body = f"    if {names['q']} < {vol_thresh}:\n        return {cost_high}\n    return {cost_low}"

        ops_code = f'''"""Módulo de capacidad y costos operativos — {tmpl['title']}."""

def compute_{names['unit_cost']}({names['q']}: int) -> float:
{cost_body}

def compute_{names['sales']}({names['demand']}: float, {names['q']}: int) -> float:
    return min({names['demand']}, float({names['q']}))

def compute_{names['holding_cost']}({names['q']}: int, {names['sales']}: float) -> float:
    return {hold_rate} * max(0.0, float({names['q']}) - {names['sales']})
{subopt_block}'''
        with open(os.path.join(output_dir, f_ops), "w", encoding="utf-8") as f:
            f.write(ops_code)

        # 4. Archivo financiero y conservación contable T2
        tx_extra_debit = "" if clean else '    ledger.record_debit(amount, "duplicate_tax_entry")\n'
        fin_code = f'''"""Módulo financiero y contable — {tmpl['title']}."""

def compute_{names['revenue']}({names['p']}: float, {names['sales']}: float) -> float:
    return {names['p']} * {names['sales']}

def compute_{names['fees']}({names['revenue']}: float) -> float:
    return {code_fee_rate} * {names['revenue']}

def compute_{names['objective']}(
    {names['revenue']}: float,
    {names['fees']}: float,
    {names['unit_cost']}: float,
    {names['q']}: int,
    {names['m']}: float,
    {names['holding_cost']}: float,
) -> float:
    return {names['revenue']} - {names['fees']} - ({names['unit_cost']} * float({names['q']})) - {names['m']} - {names['holding_cost']}

class AccountLedger:
    def __init__(self):
        self.entries = []
    def record_credit(self, amount: float, memo: str = ""):
        self.entries.append({{"type": "credit", "amount": float(amount)}})
    def record_debit(self, amount: float, memo: str = ""):
        self.entries.append({{"type": "debit", "amount": float(amount)}})

def {tmpl['tx_fn']}(ledger: AccountLedger, amount: float):
    ledger.record_credit(amount, "credit_reversal")
    ledger.record_debit(amount, "debit_reversal")
{tx_extra_debit}'''
        with open(os.path.join(output_dir, f_fin), "w", encoding="utf-8") as f:
            f.write(fin_code)

        # 5. README.md
        readme_code = f'''# {tmpl['title']} (World {world_id})

## Reglas de Negocio y Parámetros
- Comisión / Tasa operativa documentada: {doc_fee_rate * 100:.1f}%
- Presupuesto operativo máximo: {max_budget}
'''
        with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_code)

        # 6. CSV empírico opcional (T7)
        if include_csv:
            data_dir = os.path.join(output_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            csv_path = os.path.join(data_dir, tmpl["csv_name"])
            lines = [f"day_id,season_index,{names['p']},{names['m']},{names['u']},{names['demand']},{names['sales']}"]
            for day in range(1, 61):
                s_idx = round(0.85 + 0.35 * ((day % 12) / 11.0), 3)
                pv = round(rng.uniform(p_bounds[0] + 0.2, p_bounds[1] - 0.3), 2)
                mv = round(150.0 + 900.0 * (s_idx - 0.85) + rng.uniform(0.0, 250.0), 2)
                uv = round(u_mean * s_idx + rng.gauss(0.0, 15.0), 2)
                dem = round(max(0.0, uv * (1.0 - elasticity * (pv - p_ref)) + m_coeff * math.sqrt(mv)), 2)
                sls = round(min(dem, float(q_init + 250)), 2)
                lines.append(f"{day},{s_idx},{pv},{mv},{uv},{dem},{sls}")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

        spec = {
            "world_id": world_id,
            "domain": "energy_microgrid",
            "clean": clean,
            "u_mean": u_mean,
            "u_std": u_std,
            "p_ref": p_ref,
            "elasticity": elasticity,
            "m_coeff": m_coeff,
            "volume_threshold": vol_thresh,
            "cost_high": cost_high,
            "cost_low": cost_low,
            "hold_rate": hold_rate,
            "code_fee_rate": code_fee_rate,
            "doc_fee_rate": doc_fee_rate,
            "max_budget": max_budget,
            "p_init": p_init,
            "m_init": m_init,
            "q_init": q_init,
            "p_bounds": p_bounds,
            "m_bounds": m_bounds,
            "q_bounds": q_bounds,
            "node_names": names,
        }
        oracle = GenericWorldOracle(spec)
        spec["oracle"] = oracle.solve_global_optimum()
        return spec


def mutate_constant(file_path: str, old_val: str, new_val: str):
    """Prueba de mutación (G-3): altera una constante en el código fuente."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    if old_val not in content:
        raise ValueError(f"Value '{old_val}' not found in {file_path}")
    mutated = content.replace(old_val, new_val, 1)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(mutated)


class MultiSourceWorldOracle:
    """Oráculo matemático exacto independiente para Mundos Multi-Fuente (Fases 7-9: Pandas/NumPy + SQL + Manual Normativo + CSV)."""

    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec

    def evaluate_world(
        self,
        rate_val: float,
        boost_val: float,
        slots_val: int,
        u_val: Optional[float] = None,
    ) -> Dict[str, float]:
        s = self.spec
        u = float(s["u_mean"] if u_val is None else u_val)
        p_ref = float(s["p_ref"])
        elasticity = float(s["elasticity"])
        m_coeff = float(s["m_coeff"])
        vol_thresh = int(s["volume_threshold"])
        cost_high = float(s["cost_high"])
        cost_low = float(s["cost_low"])
        hold_rate = float(s["hold_rate"])
        fee_rate = float(s["code_fee_rate"])
        clean = bool(s.get("clean", False))

        raw_inflow_demand = max(0.0, u * (1.0 - elasticity * (rate_val - p_ref)) + m_coeff * math.sqrt(max(0.0, boost_val)))
        unit_expense = cost_low if (clean or slots_val >= vol_thresh) else cost_high
        served_volume = min(raw_inflow_demand, float(slots_val))
        idle_penalty = hold_rate * max(0.0, float(slots_val) - served_volume)
        gross_intake = rate_val * served_volume
        protocol_levy = fee_rate * gross_intake
        net_yield_score = gross_intake - protocol_levy - (unit_expense * float(slots_val)) - boost_val - idle_penalty
        budget_used = (unit_expense * float(slots_val)) + boost_val
        return {
            "exo_inflow": u,
            "raw_inflow_demand": raw_inflow_demand,
            "unit_expense": unit_expense,
            "served_volume": served_volume,
            "idle_penalty": idle_penalty,
            "gross_intake": gross_intake,
            "protocol_levy": protocol_levy,
            "net_yield_score": net_yield_score,
            "budget_used": budget_used,
        }

    def solve_global_optimum(self) -> Dict[str, Any]:
        s = self.spec
        p_min, p_max_code = float(s["p_bounds"][0]), float(s["p_bounds"][1])
        norm_cap = float(s.get("normative_rate_cap", p_max_code))
        p_max_eff = min(p_max_code, norm_cap)
        m_min, m_max = float(s["m_bounds"][0]), float(s["m_bounds"][1])
        q_min_code, q_max = int(s["q_bounds"][0]), int(s["q_bounds"][1])
        sql_min_slots = int(s.get("sql_min_slots", q_min_code))
        q_min_eff = max(q_min_code, sql_min_slots)
        max_b = float(s["max_budget"])
        vol_thresh = int(s["volume_threshold"])
        clean = bool(s.get("clean", False))

        base_ev = self.evaluate_world(float(s["p_init"]), float(s["m_init"]), int(s["q_init"]))
        baseline_obj = base_ev["net_yield_score"]

        best_obj = -1e18
        best_controls = (float(s["p_init"]), float(s["m_init"]), int(s["q_init"]))

        regimes: List[Tuple[float, int, int]] = []
        if clean:
            regimes.append((float(s["cost_low"]), q_min_eff, q_max))
        else:
            if q_min_eff < vol_thresh:
                regimes.append((float(s["cost_high"]), q_min_eff, min(q_max, vol_thresh - 1)))
            if q_max >= vol_thresh:
                regimes.append((float(s["cost_low"]), max(q_min_eff, vol_thresh), q_max))

        cand_qs: set = {q_min_eff, q_max}
        if q_min_eff <= vol_thresh <= q_max:
            cand_qs.add(vol_thresh)
            if vol_thresh - 1 >= q_min_eff:
                cand_qs.add(vol_thresh - 1)

        u = float(s["u_mean"])
        p_ref = float(s["p_ref"])
        elasticity = float(s["elasticity"])
        m_coeff = float(s["m_coeff"])
        hold_rate = float(s["hold_rate"])
        fee_rate = float(s["code_fee_rate"])

        for c_q, q_lo, q_hi in regimes:
            q_hi_feas = min(q_hi, int(math.floor(max_b / max(c_q, 1e-6))))
            if q_hi_feas < q_lo:
                continue
            cand_qs.add(q_lo)
            cand_qs.add(q_hi_feas)

            def neg_obj_kink(pm: np.ndarray, cq=c_q, qlo=q_lo, qhi=q_hi_feas) -> float:
                pv, mv = float(pm[0]), float(pm[1])
                dem = max(0.0, u * (1.0 - elasticity * (pv - p_ref)) + m_coeff * math.sqrt(max(0.0, mv)))
                qv = min(max(dem, float(qlo)), float(qhi))
                srv = min(dem, qv)
                idle = hold_rate * max(0.0, qv - srv)
                gross = pv * srv
                levy = fee_rate * gross
                return -(gross - levy - cq * qv - mv - idle)

            def bud_ineq_kink(pm: np.ndarray, cq=c_q, qlo=q_lo, qhi=q_hi_feas) -> float:
                pv, mv = float(pm[0]), float(pm[1])
                dem = max(0.0, u * (1.0 - elasticity * (pv - p_ref)) + m_coeff * math.sqrt(max(0.0, mv)))
                qv = min(max(dem, float(qlo)), float(qhi))
                return max_b - (cq * qv + mv)

            for p_frac in (0.3, 0.7, 1.0):
                for m_frac in (0.15, 0.5):
                    x0 = np.array([
                        p_min + p_frac * (p_max_eff - p_min),
                        m_min + m_frac * (m_max - m_min),
                    ], dtype=float)
                    res_k = minimize(
                        neg_obj_kink,
                        x0,
                        bounds=[(p_min, p_max_eff), (m_min, m_max)],
                        constraints=[{"type": "ineq", "fun": bud_ineq_kink}],
                        method="SLSQP",
                        options={"ftol": 1e-11, "maxiter": 100},
                    )
                    pv_k, mv_k = float(res_k.x[0]), float(res_k.x[1])
                    dem_k = max(0.0, u * (1.0 - elasticity * (pv_k - p_ref)) + m_coeff * math.sqrt(max(0.0, mv_k)))
                    qv_k = min(max(dem_k, float(q_lo)), float(q_hi_feas))
                    for delta_q in (-1, 0, 1):
                        q_int_cand = int(math.floor(qv_k)) + delta_q
                        if q_lo <= q_int_cand <= q_hi_feas:
                            cand_qs.add(q_int_cand)

        for q in sorted(cand_qs):
            if q < q_min_eff or q > q_max:
                continue
            c_q = float(s["cost_low"]) if (clean or q >= vol_thresh) else float(s["cost_high"])
            rem_b = max_b - c_q * q
            if rem_b < m_min - 1e-9:
                continue
            m_hi = min(m_max, rem_b)
            for p_frac in (0.3, 0.7, 1.0):
                for m_frac in (0.2, 0.7):
                    x0 = np.array([
                        p_min + p_frac * (p_max_eff - p_min),
                        m_min + m_frac * (m_hi - m_min),
                    ], dtype=float)
                    res = minimize(
                        lambda x, qv=q: -self.evaluate_world(float(x[0]), float(x[1]), qv)["net_yield_score"],
                        x0,
                        bounds=[(p_min, p_max_eff), (m_min, m_hi)],
                        method="SLSQP",
                        options={"ftol": 1e-12, "maxiter": 80},
                    )
                    ev = self.evaluate_world(float(res.x[0]), float(res.x[1]), q)
                    if ev["budget_used"] <= max_b + 1e-4 and ev["net_yield_score"] > best_obj:
                        best_obj = ev["net_yield_score"]
                        best_controls = (float(res.x[0]), float(res.x[1]), int(q))

        return {
            "baseline_objective": baseline_obj,
            "optimal_objective": best_obj,
            "improvement": best_obj - baseline_obj,
            "optimal_controls": {
                "ctrl_rate": round(best_controls[0], 4),
                "ctrl_boost": round(best_controls[1], 4),
                "ctrl_slots": int(best_controls[2]),
            },
            "expected_categories": {
                "exo_inflow": "exogenous",
                "ctrl_rate": "control",
                "ctrl_boost": "control",
                "ctrl_slots": "control",
                "raw_inflow_demand": "state",
                "unit_expense": "state",
                "served_volume": "state",
                "idle_penalty": "state",
                "gross_intake": "state",
                "protocol_levy": "state",
                "net_yield_score": "target",
            },
            "has_step_discontinuity": True,
            "has_kinks": True,
            "normative_rate_cap": norm_cap,
            "sql_min_slots": sql_min_slots,
        }


class MultiSourceWorldGenerator:
    """
    Generador de Mundos Multi-Fuente (Fases 7-9):
    Combina en un mismo proyecto:
    1. Código tabular vectorizado Pandas/NumPy (`pipeline_tabular.py` con `df["col"] = ...`, `np.where`, `np.maximum`).
    2. Consultas SQL declarativas y restricciones DDL (`warehouse_settlement.sql` con `SELECT ... AS ...`, `LEAST`, `CHECK`).
    3. Manual operativo normativo (`manual_normativo.md` con reglas duras `[RULE ...]`, cotas en lenguaje natural y discrepancias T3).
    4. Datos observacionales empíricos (`data/telemetry_observations.csv`) de donde se descubre automáticamente el mecanismo `raw_inflow_demand`.
    """

    def __init__(self, base_seed: int = 7000):
        self.base_seed = base_seed

    def generate_multisource_world(
        self,
        world_id: int,
        output_dir: str,
        clean: bool = False,
        discover_demand_from_csv: bool = True,
    ) -> Dict[str, Any]:
        rng = random.Random(self.base_seed + world_id * 53)

        if os.path.exists(output_dir):
            for root, dirs, files in os.walk(output_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        os.makedirs(output_dir, exist_ok=True)

        u_mean = round(rng.uniform(950.0, 1250.0), 1)
        u_std = round(rng.uniform(60.0, 95.0), 1)
        p_ref = 3.0
        elasticity = round(rng.uniform(0.11, 0.16), 3)
        m_coeff = round(rng.uniform(7.0, 9.5), 2)
        vol_thresh = rng.choice([650, 750, 850])
        cost_high = round(rng.uniform(1.20, 1.35), 2)
        cost_low = 1.00
        hold_rate = 0.20
        code_fee_rate = round(rng.uniform(0.030, 0.042), 3)
        doc_fee_rate = code_fee_rate if clean else 0.015
        max_budget = round(rng.uniform(2300.0, 2700.0), 0)

        p_bounds = (2.0, 6.5)
        normative_rate_cap = round(rng.uniform(4.20, 4.65), 2)
        # En modo no-clean, el valor nominal viola la cota del manual normativo (T3: NORMATIVE_BOUND_VIOLATION)
        p_init = round(normative_rate_cap - 0.30, 2) if clean else round(normative_rate_cap + 0.55, 2)
        m_init = 450.0
        m_bounds = (0.0, 2000.0)
        q_init = 520
        q_bounds = (100, 1500)
        sql_min_slots = 200

        # 1. pipeline_tabular.py (Configuración + Pipeline Vectorizado Pandas/NumPy)
        if not discover_demand_from_csv:
            demand_tabular_line = (
                f'    df["raw_inflow_demand"] = np.maximum(0.0, df["exo_inflow"] * (1.0 - {elasticity} * (df["ctrl_rate"] - {p_ref})) '
                f'+ {m_coeff} * np.sqrt(np.maximum(0.0, df["ctrl_boost"])))\n'
            )
        else:
            demand_tabular_line = ""

        dead_tabular_line = ""
        if not clean:
            dead_tabular_line = '    df["vanity_index"] = 11.5 * np.log1p(np.maximum(0.0, df["ctrl_boost"]))\n'

        unit_exp_line = (
            f'    df["unit_expense"] = 0.0 * df["ctrl_slots"] + {cost_low}\n'
            if clean
            else f'    df["unit_expense"] = np.where(df["ctrl_slots"] < {vol_thresh}, {cost_high}, {cost_low})\n'
        )
        subopt_block = ""
        if not clean:
            subopt_block = f"\ndef recommend_fixed_slots() -> int:\n    return {q_init}\n"
        tx_extra_debit = "" if clean else '    ledger.record_debit(amount, "duplicate_protocol_withholding")\n'

        py_code = f'''"""Configuración y Pipeline Tabular Vectorizado (Pandas/NumPy) — Mundo Multi-Fuente {world_id}."""
import random
import numpy as np

CTRL_RATE: float = {p_init}
CTRL_BOOST: float = {m_init}
CTRL_SLOTS: int = {q_init}

CTRL_RATE_BOUNDS = ({p_bounds[0]}, {p_bounds[1]})
CTRL_BOOST_BOUNDS = ({m_bounds[0]}, {m_bounds[1]})
CTRL_SLOTS_BOUNDS = ({q_bounds[0]}, {q_bounds[1]})

MAX_BUDGET: float = {max_budget}

EXO_INFLOW_MEAN: float = {u_mean}
EXO_INFLOW_STD: float = {u_std}


def sample_exo_inflow(seed=None) -> float:
    if seed is not None:
        rng = random.Random(seed)
        return max(0.0, rng.gauss(EXO_INFLOW_MEAN, EXO_INFLOW_STD))
    return max(0.0, random.gauss(EXO_INFLOW_MEAN, EXO_INFLOW_STD))


def transform_operations_dataframe(df):
    """Pipeline vectorizado estilo DataFrame con lineaje a nivel de columna."""
{demand_tabular_line}{unit_exp_line}    df["idle_penalty"] = {hold_rate} * np.maximum(0.0, df["ctrl_slots"] - df["served_volume"])
{dead_tabular_line}    return df
{subopt_block}

class SettlementLedger:
    def __init__(self):
        self.entries = []
    def record_credit(self, amount: float, memo: str = ""):
        self.entries.append({{"type": "credit", "amount": float(amount)}})
    def record_debit(self, amount: float, memo: str = ""):
        self.entries.append({{"type": "debit", "amount": float(amount)}})


def reconcile_settlement_voucher(ledger: SettlementLedger, amount: float):
    ledger.record_credit(amount, "settlement_credit")
    ledger.record_debit(amount, "settlement_debit")
{tx_extra_debit}'''
        with open(os.path.join(output_dir, "pipeline_tabular.py"), "w", encoding="utf-8") as f:
            f.write(py_code)

        # 2. warehouse_settlement.sql (Consultas SQL declarativas + DDL CHECK)
        sql_code = f'''-- Consultas de liquidación y restricciones DDL para Mundo Multi-Fuente {world_id}
CREATE TABLE settlement_staging (
    ctrl_slots INT,
    ctrl_rate FLOAT,
    CONSTRAINT c_sql_slots_floor CHECK (ctrl_slots >= {sql_min_slots})
);

SELECT
    LEAST(raw_inflow_demand, ctrl_slots) AS served_volume,
    ctrl_rate * served_volume AS gross_intake,
    {code_fee_rate} * gross_intake AS protocol_levy,
    gross_intake - protocol_levy - (unit_expense * ctrl_slots) - ctrl_boost - idle_penalty AS net_yield_score
FROM settlement_staging;
'''
        with open(os.path.join(output_dir, "warehouse_settlement.sql"), "w", encoding="utf-8") as f:
            f.write(sql_code)

        # 3. manual_normativo.md (Manual operativo y regulatorio con reglas simbólicas y en lenguaje natural)
        md_code = f'''# Manual Operativo y Normativo del Sistema (Mundo {world_id})

## 1. Regulación Tarifaria y Cotas Operativas
- En ningún caso `ctrl_rate` superará {normative_rate_cap} por disposición del comité regulatorio.
- [RULE c_norm_rate_cap]: ctrl_rate <= {normative_rate_cap}
- `ctrl_slots` no puede ser menor que {sql_min_slots}.

## 2. Parámetros Financieros Documentados
- Tasa de liquidación de protocolo: {doc_fee_rate * 100:.1f}%
- Presupuesto operativo máximo: {max_budget}
'''
        with open(os.path.join(output_dir, "manual_normativo.md"), "w", encoding="utf-8") as f:
            f.write(md_code)

        # 4. data/telemetry_observations.csv (Observaciones empíricas para descubrimiento causal híbrido Fase 9)
        data_dir = os.path.join(output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        csv_path = os.path.join(data_dir, "telemetry_observations.csv")
        csv_lines = ["day_id,season_index,ctrl_rate,ctrl_boost,exo_inflow,raw_inflow_demand,served_volume"]
        for day in range(1, 81):
            s_idx = round(0.85 + 0.35 * ((day % 12) / 11.0), 4)
            pv = round(rng.uniform(p_bounds[0] + 0.1, p_bounds[1] - 0.1), 4)
            mv = round(100.0 + 900.0 * (s_idx - 0.85) + rng.uniform(0.0, 500.0), 4)
            uv = round(u_mean + rng.uniform(-u_std, u_std), 4)
            dem = round(
                max(0.0, uv * (1.0 - elasticity * (pv - p_ref)) + m_coeff * math.sqrt(max(0.0, mv))),
                4,
            )
            srv = round(min(dem, float(q_init + 250)), 4)
            csv_lines.append(f"{day},{s_idx},{pv},{mv},{uv},{dem},{srv}")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("\n".join(csv_lines) + "\n")

        spec = {
            "world_id": world_id,
            "clean": clean,
            "discover_demand_from_csv": discover_demand_from_csv,
            "u_mean": u_mean,
            "u_std": u_std,
            "p_ref": p_ref,
            "elasticity": elasticity,
            "m_coeff": m_coeff,
            "volume_threshold": vol_thresh,
            "cost_high": cost_high,
            "cost_low": cost_low,
            "hold_rate": hold_rate,
            "code_fee_rate": code_fee_rate,
            "doc_fee_rate": doc_fee_rate,
            "max_budget": max_budget,
            "normative_rate_cap": normative_rate_cap,
            "sql_min_slots": sql_min_slots,
            "p_init": p_init,
            "m_init": m_init,
            "q_init": q_init,
            "p_bounds": p_bounds,
            "m_bounds": m_bounds,
            "q_bounds": q_bounds,
        }
        oracle = MultiSourceWorldOracle(spec)
        spec["oracle"] = oracle.solve_global_optimum()
        return spec

