"""Thermal generation unit data model and cost functions."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ThermalGenerator:
    """Thermal generator unit with quadratic fuel cost characteristic:

        C_i(P_i) = a_i * P_i^2 + b_i * P_i + c_i

    Marginal incremental cost:
        MC_i(P_i) = 2 * a_i * P_i + b_i
    """
    id: str = ""
    gen_id: str = ""
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    p_min: float = 0.0
    p_max: float = 0.0

    def __post_init__(self):
        if not self.id and self.gen_id:
            self.id = self.gen_id
        elif not self.gen_id and self.id:
            self.gen_id = self.id

    def fuel_cost(self, power: float) -> float:
        """Evaluate generation fuel cost in EUR/h for output power MW."""
        return self.a * (power ** 2) + self.b * power + self.c

    def marginal_cost(self, power: float) -> float:
        """Evaluate incremental fuel cost dC/dP in EUR/MWh."""
        return 2.0 * self.a * power + self.b
