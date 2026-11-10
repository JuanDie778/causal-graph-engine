"""Módulo de marketing y notoriedad de marca."""
import math
import logging

logger = logging.getLogger(__name__)


def calculate_brand_awareness(marketing_budget: float) -> float:
    """Calcula el índice de notoriedad de marca (Brand Awareness Score).
    
    Trampa T1: Nodo muerto.
    Esta métrica se calcula a partir de marketing_budget y se guarda en registros/telemetría,
    pero NINGÚN otro componente del sistema de demanda, costes o ingresos la consume.
    Por tanto, su impacto causal directo e indirecto en el beneficio neto J es nulo:
    dJ / d(brand_awareness) == 0.
    """
    if marketing_budget <= 0.0:
        score = 0.0
    else:
        score = 12.5 * math.log1p(marketing_budget)
    
    logger.info("Brand awareness score computed: %.2f", score)
    return score
