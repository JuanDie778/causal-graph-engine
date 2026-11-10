"""Nivel N3: Benchmark de Incertidumbre, Datos Empíricos y Monte Carlo (R-10, T7)."""
import pytest
from cge.monte_carlo import run_monte_carlo_simulation
from cge.empirical import estimate_marketing_coefficient_from_csv


def test_r10_monte_carlo_expected_profit():
    """R-10: Con 1.000 muestras de D0, evalúa J esperado y probabilidad de quiebre de stock (Error <= 5%)."""
    res = run_monte_carlo_simulation(n_samples=1000, seed=42)
    print(f"\nR-10 Monte Carlo: E[J]={res['mean_profit']}€, std={res['std_profit']}€, P(quiebre)={res['stockout_probability']}")
    
    # El valor nominal analítico con ruido simétrico está en torno a 2852€
    theoretical_mean = 2852.0
    rel_error_mean = abs(res["mean_profit"] - theoretical_mean) / theoretical_mean
    assert rel_error_mean <= 0.05, f"Error relativo en media ({rel_error_mean}) excede el 5%"

    # En el óptimo q=800 y E[d]=800, la probabilidad de rotura de stock por simetría normal es ~50%
    assert 0.45 <= res["stockout_probability"] <= 0.55, f"Probabilidad de rotura {res['stockout_probability']} fuera de [0.45, 0.55]"


def test_t7_empirical_csv_consistency():
    """T7: Comprueba la consistencia entre datos tabulares históricos y el modelo estocástico."""
    est = estimate_marketing_coefficient_from_csv("golden_case/data/sales_history.csv")
    assert est["passes_threshold"], "El coeficiente empírico no coincide con el valor verdadero de 8.0"
    assert est["sample_size"] == 365, "El conjunto de datos debe contener exactamente 365 días"
