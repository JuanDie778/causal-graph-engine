"""Configuración de palancas de decisión (variables de control X)."""

# Palancas de decisión X
PRICE: float = 3.50                  # p in [2.0, 6.0]
MARKETING_BUDGET: float = 500.00     # m in [0.0, 2000.0]
ORDER_QTY: int = 500                 # q in {0, ..., 1500}

# Límites de las variables de control
PRICE_BOUNDS = (2.0, 6.0)
MARKETING_BOUNDS = (0.0, 2000.0)
ORDER_QTY_BOUNDS = (0, 1500)

# Restricción presupuestaria máxima
MAX_BUDGET: float = 2500.00
