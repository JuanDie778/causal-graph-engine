# Laboratorio de Ingeniería e Investigación Matemática — CGE (`math_lab/`)

> **Propósito:** Centro de diseño teórico, auditoría forense, experimentación algorítmica y prototipado matemático riguroso para hacer que el **Decision Intelligence & Causal Graph Engine (CGE)** sea matemáticamente real, generalizable y libre de heurísticas hardcodeadas.

---

## 1. Diagnóstico Forense del Estado Inicial (`cge/`)

Aunque la batería inicial `run_benchmark.py` reportaba `6/6 PASS (41 criterios)`, la auditoría matemática del código fuente reveló que la implementación inicial de `cge/` operaba bajo un **Efecto "Cartón Piedra" (Hardcoded Mockup)** acoplado al caso nominal de la Tostadora de Café (`golden_case`):

1. **Ingesta AST Estática Falsa ([`cge/ingest.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/cge/ingest.py)):**
   No extraía ecuaciones simbólicas $\mathcal{F}$ desde el árbol sintáctico; buscaba subcadenas como `if "demand" in fn_name:` e inyectaba pesos fijos (`sensitivity_weight=-150.0`) y valores nominales estáticos.
2. **Error Algebraico y Dimensional en el Enrutador de Capa 1 ([`cge/layer1_heuristic.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/cge/layer1_heuristic.py)):**
   Definía el coste de arista como $c_{ij} = \frac{1}{|w_{ij}| + 10^{-4}}$ y ejecutaba Dijkstra ($\min \sum c_{ij}$). Por la regla de la cadena multivariable en un DAG, la sensibilidad a lo largo de una ruta es el **producto** $\prod_{(i\to j)\in p} \frac{\partial v_j}{\partial v_i}$, jamás la suma armónica. Demostramos con un contraejemplo constructivo que `Layer1Router` elige rutas con **4 veces menor impacto causal real** sobre $J$. Además, mezclar derivadas con distintas unidades físicas ($[\text{ud}/€]$ vs $[€/€]$) rompe la invariancia de escala.
3. **Optimizador de Capa 2 con Plantilla Fija y Pérdida de Parámetros en el Grafo ([`cge/layer2_exact.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/cge/layer2_exact.py)):**
   Tiene codificada a mano la forma algebraica de la cafetería en `evaluate()`, una rejilla discreta de paso 50 en `order_qty` y una lista estática `"blocking_constraints": ["c_budget", "c_market_capacity"]` sin verificar holgura complementaria KKT ($\lambda_{\text{budget}}^* = 0.0$ cuando el presupuesto tiene holgura).
   * Cuando `generator.py` crea variantes con distintos umbrales (`600, 700, 800, 900`) y elasticidades, `Layer2Solver.from_graph(graph)` olvida extraer `volume_threshold`, `marketing_coeff` y `holding_cost_rate`, **alucinando el beneficio en 10/10 variantes** (hasta **$217.57€$** de error), mientras que el constructor por defecto se equivoca por **$135.30€$** de media (y pierde **$+309.14€$** de beneficio medio real) y `from_dir` pierde optimalidad cuando el entero óptimo $q^*$ cae fuera de la rejilla de paso 50.
4. **Ceguera ante la Desigualdad de Jensen bajo Incertidumbre ([`cge/monte_carlo.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/cge/monte_carlo.py)):**
   El óptimo determinista sitúa el sistema exactamente sobre la cresta no diferenciable (*kink*) $d^* = q^* = 800$ de $\text{sales} = \min(d, q)$. Por la concavidad de $\min(\cdot, q)$, las fluctuaciones simétricas $D_0 \sim \mathcal{N}(1000, 100^2)$ provocan una pérdida sistemática de **$-138.54€$** (Brecha de Jensen) y un **50% de probabilidad de rotura de stock**, sin que el motor re-optimizara jamás bajo incertidumbre.
5. **Auditoría Sintáctica Frágil ([`cge/audit.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/cge/audit.py)) y Trampa Empírica ([`cge/empirical.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/cge/empirical.py)):**
   * En `T2`, contar llamadas `len(debit_calls) != len(credit_calls)` produce falsos positivos en asientos contables divididos (ej. Base Imponible 80% + IVA 20%) y falsos negativos cuando se pasa un importe multiplicado (`2 * amount`) o en ramas condicionales `if/else` asimétricas.
   * En `T7`, `cge/empirical.py` restaba `d0 * (1.0 - 0.15 * (p - 3.0))` usando el `0.15` verdadero hardcodeado antes de estimar $\beta = 8.0$.

---

## 2. Resultados Empíricos Clave Verificados en `math_lab/`

Todos los resultados están serializados y verificados en [`experimental_results.json`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/experimental_results.json):

| Experimento | Hallazgo Matemático / Resultado Empírico | Motor Actual (`cge/`) | Nuevo Motor (`math_lab/`) |
| :--- | :--- | :--- | :--- |
| **Exp 01 (`exp01`)** | Selección de ruta causal óptima ($\|S(A)\|=1.0$ vs $\|S(B)\|=0.25$) | Elige la ruta **incorrecta** `X->B->J` ($0.25$) por sumar $\sum 1/\|w\|$ | Elige la ruta **exacta** `X->A->J` ($1.00$) en el semianillo Max-Product |
| **Exp 01 (`exp01`)** | Detección del *kink* y separación de saltos de orden 0 en $q^*=800$ | Regla hardcodeada `if edge.target == "sales"` | Subdiferencial de Clarke $\partial_C \text{sales} = [0.0, 1.0]$ + sensibilidad de salto $\Delta_q^- J = +164.07€$ sin explosión de Dirac |
| **Exp 02 (`exp02`)** | Óptimo global en `golden_case` y en rama interior $q=1000$ | $J^* = 2990.88€$ (con plantilla escrita a mano) | $J^* = 2990.8807€$ ($q^*=800$) y $J^*(q=1000) = 2754.7755€$ (**0 fórmulas hardcodeadas**) |
| **Exp 02 (`exp02`)** | Generalización real en las variantes de `generator.py` | `from_graph` y constructor por defecto alucinan en **10/10 variantes** | Compila el AST de cada variante con **error de alucinación $0.00€$** y gana hasta **+$1124.17€$** |
| **Exp 02 (`exp02`)** | Diagnóstico KKT de infactibilidad en $J_{\text{target}} = 3100€$ | Devuelve lista estática `["c_budget", "c_market_capacity"]` aunque el presupuesto sobra | Verifica holgura complementaria ($\lambda_{\text{budget}}^* = 0.0$, $\lambda_{D_0}^* = 3.2539€/\text{ud}$) + relajación $\ell_1$ |
| **Exp 02 (`exp02`)** | Intervención inversa escasa ($\ell_1$) para $J_{\text{target}} = 1325€$ | Salta siempre al óptimo global moviendo las 3 variables ($\|\Delta X\|_0 = 3$) | Halla intervención **1-sparse ($\|\Delta X\|_0 = 1$)**: mueve sólo `order_qty` a $800$ ($J=1416€$) |
| **Exp 03 (`exp03`)** | Integración de $\mathbb{E}[J(X, U)]$ con partición dinámica en $u_{\text{kink}}(X)$ | Monte Carlo 1000 muestras: $2858.89€$ (error muestral $+6.55€$) | Cuadratura Gauss-Legendre dividida dinámicamente en $u_{\text{kink}}$: **error $= 0.000000€$** en $X_{\text{det}}^*$ y $X_{\text{stoch}}^*$ |
| **Exp 03 (`exp03`)** | Optimización estocástica directa $\max_X \mathbb{E}_U[J(X, U)]$ y $\text{CVaR}_{5\%}$ | No existe (se queda en $\mathbb{E}[J]=2852.34€$, rotura de stock $50.0\%$) | Halla $(p^*=5.7721, m^*=384.01, q^*=800)$: **$\mathbb{E}[J] = 2925.09€$ (+$72.75€$/día)**, rotura **$15.3\%$**, y $\text{CVaR}_{5\%}$ **+$99.76€$** con $q$ global |
| **Exp 04 (`exp04`)** | Verificación simbólica de invariantes contables (T2) y T1-B | Falla en asiento con IVA dividido, débito $2\times$ y ramas `if/else` | `SymbolicLedger` verifica $\mathcal{R}(\text{amount}) \equiv 0$ en todas las ramas y `FormalSymbolicAuditor` detecta T1-B |
| **Exp 04 (`exp04`)** | Estimación econométrica conjunta desde CSV (T7) | Hace trampa restando `0.15` hardcodeado antes de estimar $\beta$ | Estima conjuntamente $\hat{\alpha}=0.15000, \hat{\beta}=7.99999$, elige $\sqrt{m}$ por BIC ($\text{BIC}=-4063.6$) y verifica $\kappa(\tilde{X})=12.27 < 100$ |
| **Exp 05 (`exp05`)** | Escalabilidad en cadenas con $M=6$ kinks ($2^6 = 64$ regímenes) | Enumeración de regímenes requiere $2^M = 64$ subproblemas NLP | Homotopía Log-Sum-Exp + Epígrafe MPCC halla el óptimo global ($28742.96€$, error $0.0001€$) en **3 llamadas NLP** ($21.3\times$ menos) |
| **Exp 05 (`exp05`)** | Grafos con ciclos temporales de inventario ($I_t = \max(0, I_{t-1}+q_t-s_t)$) | Falla por ciclo estático (`is_dag == False`) o política miope ($2686.47€$) | Desenrolla el **Time-Expanded DAG $\mathcal{G}_T$** (`is_dag == True`) y gana **+$217.18€$** mediante Batch-Ordering MPC ($2903.65€$) |
| **Exp 05 (`exp05`)** | Inferencia causal bajo endogeneidad / confusión ($M_t \leftarrow W_t \to D_t$) | OLS sufre un **sesgo de $+21.39\%$** ($\hat{\beta}_{\text{OLS}}=9.7115$) | Test de Hausman-Wu ($t=67.77$) detecta la endogeneidad; **2SLS/IV** baja el error a **$1.58\%$** y **Back-Door** a **$0.21\%$** |

---

## 3. Arquitectura del Laboratorio (`math_lab/`)

### Documentación Teórica y Propuestas (`math_lab/proposals/`)
1. [**Propuesta 01 — Compilación AST $\to$ SCM Simbólico, Reverse-Mode AD en DAGs, Subdiferenciales de Clarke y Enrutamiento Tropical Max-Product**](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/proposals/01_scm_jacobian_and_routing.md)
2. [**Propuesta 02 — Optimización Híbrida Simbólico-Numérica (MINLP por Partición Automática de Regímenes), Control Inverso Escaso ($\ell_1+\ell_2$) y Diagnóstico KKT/IIS**](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/proposals/02_hybrid_minlp_and_inverse_opt.md)
3. [**Propuesta 03 — Optimización Estocástica Directa, Teorema de la Brecha de Jensen en Kinks, Cuadratura Compuesta Dinámica y Optimización Media-$\text{CVaR}_\alpha$ de Rockafellar-Uryasev**](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/proposals/03_stochastic_and_risk_optimization.md)
4. [**Propuesta 04 — Auditoría Formal Simbólica (T1–T6: SymbolicLedger, Violación de Fidelidad, Precio de la Anarquía Bilevel) e Inferencia Econométrica Estructural Conjunta (T7)**](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/proposals/04_formal_audit_and_verification.md)
5. [**Propuesta 05 — Blueprint de Integración Multi-Agente y Hoja de Ruta para Iteraciones Futuras**](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/proposals/05_multi_agent_integration_blueprint.md)
6. [**Propuesta 06 — Escalabilidad MPCC ($2^M$ Kinks), Grafos con Ciclos Temporales (MPC sobre DAG Expandido) y Econometría Causal bajo Endogeneidad (2SLS / IV & Back-Door)**](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/proposals/06_mpcc_kinks_temporal_mdp_and_iv_econometrics.md)

### Prototipos Matemáticos Ejecutables (`math_lab/experiments/`)
* [`exp01_symbolic_scm_and_routing.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/experiments/exp01_symbolic_scm_and_routing.py): `ASTFunctionToSymPy`, `SymbolicSCM`, `TropicalMaxProductRouter`.
* [`exp02_universal_minlp_solver.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/experiments/exp02_universal_minlp_solver.py): `UniversalLayer2Solver`, `compare_legacy_vs_universal_on_variants`.
* [`exp03_stochastic_cvar_optimizer.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/experiments/exp03_stochastic_cvar_optimizer.py): `StochasticCVaROptimizer`, `run_jensen_and_cvar_experiment`.
* [`exp04_symbolic_auditor_and_econometrics.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/experiments/exp04_symbolic_auditor_and_econometrics.py): `SymbolicLedger`, `FormalSymbolicAuditor`, `JointStructuralEconometricEstimator`.
* [`exp05_advanced_edge_cases_mpcc_mdp_iv.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/experiments/exp05_advanced_edge_cases_mpcc_mdp_iv.py): `MultiKinkSupplyChainBenchmark`, `TemporalCyclicSCMAndMPCSolver`, `CausalEndogeneityIV2SLSEstimator`.
* [`test_math_lab.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/tests/test_math_lab.py): Suite de verificación automatizada (`pytest`).
* [`experimental_results.json`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/experimental_results.json): Resultados numéricos reproducibles generados por [`run_all_experiments.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/run_all_experiments.py).

---

## 4. Comandos de Verificación

```powershell
# 1. Ejecutar la suite de pruebas matemáticas de math_lab
python -m pytest math_lab/tests/test_math_lab.py -v

# 2. Regenerar el informe JSON completo de todos los experimentos
python -m math_lab.run_all_experiments

# 3. Verificar que el benchmark original del repositorio (N0-N5) sigue 100% intacto
python run_benchmark.py
```
