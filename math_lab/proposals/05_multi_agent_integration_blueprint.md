# Propuesta Matemática 05: Blueprint de Integración Multi-Agente y Hoja de Ruta Iterativa para el Repositorio CGE

**Autor:** Agente de Arquitectura Matemática (CGE Math Lab)  
**Destinatarios:** Coordinador, Agente de Ingesta AST (`cge/ingest.py`), Agente de Motores (`cge/layer1_heuristic.py`, `cge/layer2_exact.py`), Agente de Auditoría (`cge/audit.py`), Agente de UI/Copiloto (`cge/copilot.py`, `ui/`).

---

## 1. Mapa de Reemplazo Quirúrgico (`cge/` $\leftarrow$ `math_lab/experiments/`)

Para transformar `cge/` de un MVP hardcodeado ("Cartón Piedra") en un motor matemático generalista sin romper el contrato `schema/canonical_schema.json` ni los 41 criterios del benchmark `run_benchmark.py`:

| Módulo Actual en `cge/` | Patología Matemática Detectada | Prototipo Verificado en `math_lab/` | Acción de Integración para los Agentes |
| :--- | :--- | :--- | :--- |
| [`cge/ingest.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/cge/ingest.py) | `if "demand" in fn_name:` con pesos y valores nominales estáticos hardcodeados. | [`SymbolicSCM`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/experiments/exp01_symbolic_scm_and_routing.py) (`exp01`) | Usar `ASTFunctionToSymPy` para serializar cada función como expresión `sympy_expr` dentro de `exact_mechanism.definition` en `schema.json` y poblar `sensitivity_weight` mediante `compute_reverse_mode_ad_jacobian()`. |
| [`cge/layer1_heuristic.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/cge/layer1_heuristic.py) | Coste aditivo `1/(abs(w)+1e-4)` que minimiza la suma armónica en lugar de maximizar el producto de la cadena $\prod \|w_e\|$. | [`TropicalMaxProductRouter`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/experiments/exp01_symbolic_scm_and_routing.py) (`exp01`) | Sustituir `find_top_k_paths` por programación dinámica topológica sobre el Semianillo Tropical Max-Product usando elasticidades adimensionales normalizadas $\tilde{w}_{ij} = \frac{\partial v_j}{\partial v_i}\frac{S_i}{S_j}$. |
| [`cge/layer2_exact.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/cge/layer2_exact.py) | Ecuaciones del café fijas en código, 11 candidatos de $q$ fijos y `target_profit > 3500.0` hardcodeado. | [`UniversalLayer2Solver`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/experiments/exp02_universal_minlp_solver.py) (`exp02`) | Compilar el DAG desde `schema.json` con `sympy.lambdify`, extraer umbrales de `Piecewise` para partición automática de regímenes + `SLSQP`, y diagnosticar infactibilidad para cualquier $J_{\text{target}} > J^*$ con precios sombra KKT. |
| [`cge/monte_carlo.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/cge/monte_carlo.py) | Evaluación pasiva sin optimización estocástica (pierde $138.80€$ por la Brecha de Jensen en el kink $d=q$). | [`StochasticCVaROptimizer`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/experiments/exp03_stochastic_cvar_optimizer.py) (`exp03`) | Integrar cuadratura de Gauss-Hermite ($N_q=32$, $<1\text{ ms}$) y optimización conjunta Media-$\text{CVaR}_\alpha$ de Rockafellar-Uryasev para ofrecer recetas robustas con buffer óptimo de demanda. |
| [`cge/audit.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/cge/audit.py) | T1 sólo mira `nx.has_path`; T2 sólo cuenta llamadas `len(debit) != len(credit)`; T4 y T6 tienen constantes fijas. | [`FormalSymbolicAuditor`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/experiments/exp04_symbolic_auditor_and_econometrics.py) (`exp04`) | Incorporar `SymbolicLedger` para T2, derivada simbólica total para T1-A/B/C, optimización bilevel restringida para T4 y asimetría de Clarke $|D^+ - D^-| > 0$ para T6. |
| [`cge/empirical.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/cge/empirical.py) | Resta `d0*(1 - 0.15*(p-3))` hardcodeado antes de estimar $\beta$. | [`JointStructuralEconometricEstimator`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/experiments/exp04_symbolic_auditor_and_econometrics.py) (`exp04`) | Estimar conjuntamente $(\hat{\alpha}, \hat{\beta})$, seleccionar base funcional por BIC, calcular número de condición $\kappa(\tilde{X})$ e intervalos de confianza HC3. |

---

## 2. Extensión Recomendada para `schema/canonical_schema.json` (Versión 1.1)

Para que el `schema.json` sea verdaderamente autosuficiente (permitiendo que `Layer2Solver` reconstruya y optimice el sistema leyendo únicamente `schema.json` sin volver a abrir los `.py`), proponemos enriquecer las propiedades opcionales de `nodes` y `edges` manteniendo compatibilidad 100% hacia atrás:

1. **En `nodes[*]`:**
   * `"distribution"`: `{"family": "normal", "mean": 1000.0, "std": 100.0}` para nodos `"exogenous"` ($U$).
   * `"characteristic_scale"`: $S_i = x_i^{\max} - x_i^{\min}$ para adimensionalización automática.
   * `"clarke_kink_index"`: $\mathcal{K}_i(X^*) = |D^+ f_i - D^- f_i|$ en el punto óptimo.
2. **En `edges[*].exact_mechanism`:**
   * `"engine": "sympy_expr"`,
   * `"definition": "Piecewise((1.2, order_qty < 800), (1.0, True))"` (expresión parseable directamente por `sympy.sympify()`).
   * `"dimensionless_elasticity"`: $\tilde{w}_{ij}$.

---

## 3. Plan de Iteraciones Futuras para el Agente Matemático

En próximas iteraciones, estamos listos para expandir `math_lab/` hacia:
* **Iteración 2 (Sistemas con Ciclos y Dinámica Temporal — Grafos de Estado en Tiempo Discreto / MDP):** Cuando el repositorio ingerido contenga bucles de retroalimentación temporal ($v_t = f(v_{t-1}, x_t, u_t)$, ej. inventario multi-periodo con *carry-over* de stock entre días), desenrollar el grafo sobre un horizonte $T$ (*Time-Expanded DAG*) o resolver la ecuación de Bellman / Control Predictivo Basado en Modelos (MPC).
* **Iteración 3 (Verificación SMT Exacta con Z3 sobre Ramas Discretas Complejas):** Traducción automática del AST Python a cláusulas SMT-LIB2 (`QF_NRA` / `QF_LIRA`) para demostrar formalmente la alcanzabilidad o imposibilidad de estados lógicos booleanos/categóricos arbitrarios.
* **Iteración 4 (Descubrimiento Causal Híbrido Código + Logs de Producción):** Combinar la estructura parcial del AST con algoritmos de descubrimiento causal sobre trazas observacionales (PC / NOTEARS con restricciones estructurales dadas por el código) cuando existan microservicios externos o cajas negras sin código fuente disponible.
