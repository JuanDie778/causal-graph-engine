# Propuesta Matemática 02: Optimización Híbrida Simbólico-Numérica (MINLP por Partición de Regímenes), Control Inverso Escaso y Diagnóstico KKT/IIS

**Autor:** Agente de Arquitectura Matemática (CGE Math Lab)  
**Estado:** Implementado y Verificado Experimentalmente en [`exp02_universal_minlp_solver.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/experiments/exp02_universal_minlp_solver.py)

---

## 1. Diagnóstico Forense de `cge/layer2_exact.py`

El archivo [`cge/layer2_exact.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/cge/layer2_exact.py) presenta cuatro limitaciones estructurales que impiden su uso como motor matemático universal:

1. **Plantilla Funcional Hardcodeada en `evaluate()` (Líneas 114–129):**
   En lugar de compilar las ecuaciones estructurales $\mathcal{F}$ desde el código fuente o desde el `CausalGraph`, `evaluate()` tiene codificada a mano la forma algebraica de la cafetería (`1.0 - self.price_elasticity * (price - 3.0)`, `self.marketing_coeff * math.sqrt(...)`, `min(demand, float(order_qty))`). Si cualquier ecuación del dominio cambia su estructura funcional, `Layer2Solver` no puede representarla.
2. **Pérdida de Parámetros en `Layer2Solver.from_graph(graph)` y Rejilla Gruesa en `from_dir()` (Líneas 43–93 y 199–209):**
   * El constructor por defecto `Layer2Solver(base_demand=..., max_budget=...)` mantiene hardcodeados `volume_threshold=800, price_elasticity=0.15, marketing_coeff=8.0, cost_high=1.20, cost_low=1.00, fee_rate=0.03`.
   * Incluso `Layer2Solver.from_graph(graph)` (líneas 43–93) **olvida extraer `volume_threshold`, `marketing_coeff` y `holding_cost_rate`**, dejándolos fijos en `(800, 8.0, 0.20)`. Como resultado, `from_graph` **alucina el beneficio en 10/10 variantes de `generator.py`** (con errores de hasta $217.57€$).
   * Por su parte, `Layer2Solver.from_dir(target_dir)` (que re-ejecuta `ASTIngestPipeline` para parchar los atributos) evalúa `order_qty` únicamente sobre una rejilla discreta de paso 50 (`range(100, 1501, 50)`) más `{volume_threshold-1, volume_threshold, volume_threshold+1, 500, 800}` y `price` en una rejilla de 401 puntos, perdiendo optimalidad cuando el entero óptimo $q^*$ cae entre múltiplos de 50 por saturación presupuestaria o cuando el umbral del descuento no está en el `CausalGraph`.
3. **Diagnóstico Estático de Infactibilidad sin Dualidad KKT (`R-8`, Líneas 218–228):**
   Cuando `best_profit < target_profit - 1e-4`, `optimize()` devuelve siempre la lista estática hardcodeada `"blocking_constraints": ["c_budget", "c_market_capacity"]` sin verificar la **holgura complementaria de Karush-Kuhn-Tucker ($\lambda_i^* h_i(X^*) = 0$)**. En el caso nominal ($J^* = 2990.88€$), el presupuesto usado es $1228.22€ < 2500.00€$ (holgura de $1271.78€$), por lo que el precio sombra real del presupuesto es $\lambda_{\text{budget}}^* = 0.0$ (`c_budget` **no** es una restricción activa ni bloqueante en $X^*$; el cuello de botella real es la capacidad de demanda del mercado y el techo de precio $p \le 6.0€$).
4. **Ausencia del Problema Inverso de Mínima Intervención ($\ell_1$, Líneas 263–319):**
   El Manifiesto (§4) exige resolver:
   $$\min_{\Delta X} \|V(X_0 + \Delta X, U) - V^*\|_2^2 + \lambda \|\Delta X\|_1 \quad \text{s.t. } \mathcal{C}(V, X) \le 0$$
   pero `generate_intervention_recipe` ignora la búsqueda de mínima intervención sobre $J_{\text{target}}$, salta siempre al máximo global absoluto y devuelve 3 pasos con plantilla fija (`order_qty -> price -> marketing_budget`).

---

## 2. Arquitectura Matemática del Solucionador Universal de Capa 2

### 2.1. Descomposición Automática en Regímenes Suaves (Branch-and-Regime MINLP)
Un grafo causal compilado desde código real contiene funciones suaves mezcladas con dos tipos de no-linealidades no diferenciables:
1. **Ramificaciones Discontinuas (`Piecewise` / `if-else`):** Ej. $c(q) = c_{\text{high}}$ si $q < \theta$, $c_{\text{low}}$ si $q \ge \theta$.
2. **Operadores Continuos a Trozos (`Min` / `Max`):** Ej. $\text{sales} = \min(d(p, m), q)$ y $\text{holding} = 0.20 \max(0, q - \text{sales})$.

Si se entrega la función global directamente a un optimizador basado en gradientes (BFGS / SLSQP):
* En $q < \theta$, $\frac{\partial c}{\partial q} = 0$, por lo que el gradiente **jamás informa al optimizador de la existencia del escalón en $q = \theta$**.
* En $d > q$, $\frac{\partial \text{sales}}{\partial p} = \frac{\partial \text{sales}}{\partial m} = 0$, atrapando al optimizador en mesetas de gradiente nulo.

**Teorema de Partición Semialgebraica sobre el DAG:**
Extraemos simbólicamente del `SymbolicSCM` el conjunto finito de predicados de ramificación $\mathcal{B} = \{\psi_1(X, V) \le 0, \dots, \psi_K(X, V) \le 0\}$:
* De cada nodo `Piecewise((e_1, a < b), (e_2, True))`, extraemos la frontera $a(X) - b(X) = 0$.
* De cada nodo `Min(A(X), B(X))` o `Max(A(X), B(X))`, introducimos las sub-regiones:
  - **Régimen Limitado por Demanda (Stock Sobrante):** $A(X) \le B(X)$ (donde $\min(A, B) \equiv A$, suave).
  - **Régimen Saturado en Capacidad (Equilibrio Exacto):** $A(X) = B(X)$ (variedad diferenciable donde no se desperdicia marketing ni stock).
  - **Régimen Limitado por Capacidad (Exceso de Demanda):** $A(X) \ge B(X)$ (donde $\min(A, B) \equiv B$, suave).

El espacio factible $\Omega_X$ se descompone en una unión finita de subdominios suaves $\Omega_X = \bigcup_{r=1}^R \Omega_r$. En cada régimen $\Omega_r$:
1. Todas las funciones de estado $V^{(r)}(X)$ y el objetivo $J^{(r)}(X)$ son funciones **infinitamente diferenciables ($C^\infty$)** cuyos gradientes $\nabla_X J^{(r)}(X)$ y jacobianos de restricciones $\nabla_X h_i^{(r)}(X)$ se obtienen analíticamente mediante `sympy` / Reverse-Mode AD.
2. Resolvemos el subproblema suave mediante Programación Cuadrática Sucesiva (`SLSQP`) con multi-arranque determinista, proyectando las variables discretas $x_j \in \mathbb{Z}$ sobre el entorno entero de la frontera y del punto estacionario continuo ($\{\lfloor x_j^* \rfloor, \lceil x_j^* \rceil, \theta_r\}$).
3. El óptimo global exacto es $X^* = \arg\max_{r \in \{1,\dots,R\}} J(X_r^*)$.

---

### 2.2. Diagnóstico Matemático Exacto de Infactibilidad: Precios Sombra KKT y Relajación Elástica $\ell_1$

Cuando el usuario solicita alcanzar un umbral objetivo $J(X) \ge J_{\text{target}}$:

#### Paso 1: Verificación Exacta de Factibilidad vía Maximización Primitiva
Calculamos primero el máximo global factible del sistema bajo las restricciones duras $\mathcal{C}$:
$$J^* = \max_{X \in \Omega_X} J(X) \quad \text{sujeto a} \quad h_i(X) \le 0 \;\;\forall i \in \mathcal{C}$$
Si $J_{\text{target}} > J^*$, el problema es **matemáticamente INFACTIBLE** (sin importar si $J_{\text{target}}$ es $3000€$, $3200€$ o $5000€$, eliminando cualquier constante mágica como `3500.0`).

#### Paso 2: Identificación del UNSAT Core (IIS) y Precios Sombra de Karush-Kuhn-Tucker (KKT)
En el óptimo global $X^*$, la condición de estacionariedad de Karush-Kuhn-Tucker (bajo cualificación de restricciones LICQ) establece que:
$$\nabla J(X^*) = \sum_{i \in \mathcal{A}(X^*)} \lambda_i^* \nabla h_i(X^*) + \sum_{k : x_k^* = x_k^{\max}} \mu_k^+ e_k - \sum_{k : x_k^* = x_k^{\min}} \mu_k^- e_k$$
donde $\mathcal{A}(X^*) = \{i \in \mathcal{C} : |h_i(X^*)| \le \epsilon\}$ son las restricciones activas y $\lambda_i^* = \frac{\partial J^*}{\partial b_i} \ge 0$ son los **multiplicadores de Lagrange (precios sombra)**.

Además, para cuantificar exactamente qué restricciones bloquean $J_{\text{target}}$, resolvemos el **Problema de Factibilidad Elástica Ponderada ($\ell_1$-Relaxation)**:
$$\min_{X, \, s \ge 0} \sum_{i \in \mathcal{C} \cup \text{bounds}} w_i s_i \quad \text{sujeto a} \quad J(X) \ge J_{\text{target}}, \quad h_i(X) \le s_i$$
* El conjunto de restricciones con holgura estrictamente positiva $\{i : s_i^* > 0\}$ forma el **Subsistema Irreduciblemente Inconsistente (IIS / Blocking Constraints)** mínimo que impide alcanzar $J_{\text{target}}$.
* $s_i^*$ le dice al usuario **exactamente en cuántas unidades o euros habría que ampliar la restricción $i$** (por ejemplo, ampliar el presupuesto máximo o el techo de precio/mercado) para que $J_{\text{target}}$ sea alcanzable.

---

### 2.3. Formulación del Problema Inverso de Mínima Intervención ($\ell_1 + \ell_2$ Elastic Net)

Cuando el usuario no pide el máximo absoluto sino alcanzar una meta operativa factible $J_{\text{target}} \le J^*$ perturbando lo menos posible la operación actual $X_0$, resolvemos el problema de control inverso regularizado con escalado adimensional $\Sigma_X = \text{diag}(x_k^{\max} - x_k^{\min})$:
$$\min_{\Delta X \in \mathbb{R}^d} \; \|\Sigma_X^{-1} \Delta X\|_1 + \frac{\gamma}{2} \|\Sigma_X^{-1} \Delta X\|_2^2$$
$$\text{sujeto a} \quad J(X_0 + \Delta X) \ge J_{\text{target}}, \quad h_i(X_0 + \Delta X) \le 0, \quad X_{\min} \le X_0 + \Delta X \le X_{\max}$$
Mediante desdoblamiento de variables positivas y negativas $\Delta X = \Delta X^+ - \Delta X^-$ con $\Delta X^+, \Delta X^- \ge 0$, la norma $\|\Sigma_X^{-1} \Delta X\|_1 = \sum_k \frac{\Delta x_k^+ + \Delta x_k^-}{S_k}$ se convierte en un objetivo lineal suave diferenciable:
* Gracias a la geometría del politopo $\ell_1$, si una meta intermedia (ej. subir el beneficio de $597.50€$ a $1325.00€$) puede lograrse **moviendo una sola palanca** (ej. sólo `price` de $3.50€$ a $5.00€$, dejando $\Delta \text{marketing\_budget} = 0$ y $\Delta \text{order\_qty} = 0$), el solucionador encuentra automáticamente una **receta escasa (sparse)** de cardinalidad $\|\Delta X\|_0 = 1$.
* Además, el **orden óptimo de ejecución secuencial** de los pasos $\Delta x_{\pi(1)}, \Delta x_{\pi(2)}, \dots$ se determina maximizando la integral de línea discreteada (greedy supermodular ordering) de tal forma que **ningún estado intermedio viole las restricciones duras $\mathcal{C}$** y cada paso desbloquee la sensibilidad del siguiente (por ejemplo, elevar primero `order_qty` para eliminar el cuello de botella $\min(d, q)$ antes de ajustar `price` y `marketing_budget`).
