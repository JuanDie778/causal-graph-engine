"""Variables exógenas de entorno (U)."""
import random
from typing import Optional

BASE_DEMAND_MEAN: float = 1000.0
BASE_DEMAND_STD: float = 100.0


def sample_base_demand(seed: Optional[int] = None) -> float:
    """Genera una muestra de la demanda exógena base D_0 ~ N(1000, 100^2)."""
    if seed is not None:
        rng = random.Random(seed)
        return max(0.0, rng.gauss(BASE_DEMAND_MEAN, BASE_DEMAND_STD))
    return max(0.0, random.gauss(BASE_DEMAND_MEAN, BASE_DEMAND_STD))


def get_nominal_base_demand() -> float:
    """Retorna el valor nominal esperado de la demanda base."""
    return BASE_DEMAND_MEAN
