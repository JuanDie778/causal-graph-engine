# Marco Formal de Métricas, Principios Anti-Goodhart y Manifiesto de Congelación de Línea de Base (baseline_v0)

**Documento Canónico de Especificación Metrológica — Hito M4 (Fases D y E)**  
**Proyecto:** ActorDB (Base de Datos Causal de Actores para Análisis de Sistemas de Software)  
**Versión de Baseline Congelado:** `baseline_v0`  
**Hash de Manifiesto Criptográfico:** `benchmark/baseline_bench/baseline_v0_manifest.json` (SHA-256 Protocol)  
**Fecha de Publicación:** 2026-11-11  
**Clasificación Epistemológica:** Metrología Computacional / Inferencia Causal / Optimización Matemática  
**Estado:** Canónico, Inmutable y Turnkey  

---

## 1. Resumen Ejecutivo y Motivación Epistemológica

### 1.1 La Crisis de Medición en Sistemas de Razonamiento y Software

La evaluación de arquitecturas de software orientadas al razonamiento causal, la optimización matemática prescriptiva y la gobernanza de lógica de negocio enfrenta actualmente una crisis epistemológica profunda. En la práctica contemporánea del aprendizaje automático aplicado a agentes y modelos fundacionales de lenguaje (LLMs), predomina el paradigma denominado *LLM-as-a-judge* (Zheng et al., 2023; Wang et al., 2023). En este esquema, un modelo generativo evalúa la salida de otro modelo generativo mediante rúbricas formuladas en lenguaje natural.

La literatura teórica y empírica demuestra que *LLM-as-a-judge* adolece de patologías intrínsecas graves:
1. **Sesgo de Auto-Afinidad (*Self-Enhancement Bias*):** Los modelos evalúan con puntuaciones significativamente más altas las respuestas que comparten su propio estilo léxico, longitud y estructura de razonamiento latente (Zheng et al., 2023).
2. **Sesgo de Posición y Verbosidad (*Position & Verbosity Bias*):** Variaciones aleatorias en el orden de presentación de candidatos o la mera inclusión de texto explicativo redundante elevan la probabilidad de asignación de éxito, independientemente de la validez fáctica de la respuesta.
3. **Ceguera ante Invariantes Físicos y Contables:** Un modelo de lenguaje es incapaz de computar con precisión aritmética de coma flotante si una asignación de variables viola la ley de conservación de flujo de Kirchhoff ($\sum P_{\text{gen}} \neq D$), la partida doble contable ($\Delta \text{Activos} \neq \Delta \text{Pasivos} + \Delta \text{Patrimonio}$) o los límites térmicos de una línea de transmisión. Un texto articulado con fluidez sintáctica e hiperbólica convicción retórica suele recibir la calificación máxima de un juez LLM, encubriendo una catástrofe en el plano de la ejecución real.

Esta desconexión da lugar a la **ilusión de la fluidez superficial** (*superficial fluency illusion*): el sistema aparenta razonar de forma analítica sofisticada, pero colapsa en el instante en que sus prescripciones son ejecutadas contra el sistema determinista subyacente. Para transformar el benchmarking de agentes en una disciplina de ingeniería rigurosa, ActorDB adopta el principio fundamental del **anclaje en la ejecución determinista** (*execution grounding*). Ningún resultado es validado mediante juicio semántico probabilístico; toda respuesta debe ser evaluada frente a oráculos matemáticos cerrados, solucionadores deterministas de Investigación Operativa y la ejecución del código fuente real de los repositorios.

---

### 1.2 La Tupla Estructural $\mathcal{M}$ como Espacio Base de Medición

Para formular un marco de métricas matemáticamente coherente, todo repositorio de código fuente $\mathcal{R}$ analizado por el sistema se conceptualiza como un modelo causal ejecutable parametrizado por la 6-tupla estructural canónica:

$$\mathcal{M} = \langle \mathcal{U}, \mathcal{X}, \mathcal{V}, \mathcal{F}, \mathcal{C}, \mathcal{J} \rangle$$

Donde los componentes se definen rigurosamente como:
* $\mathcal{U} = \{u_1, u_2, \dots, u_m\} \in \mathbb{R}^m$: Vector de variables exógenas no controlables o perturbaciones estocásticas del entorno operativo, gobernadas por una distribución $\mathcal{P}_{\mathcal{U}}$ (ej. demanda meteorológica, precios spot de combustible, tipos de interés interbancarios).
* $\mathcal{X} = \{x_1, x_2, \dots, x_k\} \in \mathcal{X}_{\text{admisible}} \subseteq \mathbb{R}^k$: Vector de variables de decisión, controles o palancas corporativas directamente accionables por el usuario humano o el agente decisor.
* $\mathcal{V} = \{v_1, v_2, \dots, v_n\} \in \mathbb{R}^n$: Vector de variables de estado endógenas generadas por la ejecución del software.
* $\mathcal{F} = \{f_1, f_2, \dots, f_n\}$: Conjunto de ecuaciones estructurales deterministas o mecanismos causales implementados en el código, donde cada función determina unívocamente el estado de una variable endógena a partir de sus padres causales en el grafo acíclico dirigido (DAG):
  $$v_i = f_i\big(\text{Pa}(v_i), \mathcal{X}_{v_i}, \mathcal{U}_{v_i}\big)$$
* $\mathcal{C} = \mathcal{C}_{\text{eq}} \cup \mathcal{C}_{\text{ineq}}$: Conjunto de restricciones e invariantes formales del sistema de software:
  * Restricciones de igualdad estricta: $\mathcal{C}_{\text{eq}} = \{g_j(x, v) = 0 \mid j = 1, \dots, p\}$ (ej. balances de masa, flujos de caja de partida doble, conservación de inventario).
  * Restricciones de desigualdad operativa: $\mathcal{C}_{\text{ineq}} = \{h_l(x, v) \le 0 \mid l = 1, \dots, q\}$ (ej. cotas de capacidad de almacenamiento, techos presupuestarios, límites térmicos y cotas de caja $[lb_k, ub_k]$).
* $\mathcal{J}: \mathcal{X} \times \mathcal{V} \times \mathcal{U} \to \mathbb{R}$: Función objetivo escalar de valor de negocio o coste global del sistema (ej. beneficio operativo neto después de impuestos, coste total de almacenamiento y ruptura, coste de despacho térmico cuadrático).

El universo del benchmark consta de $M = 48$ tareas estructuradas sobre $\mathcal{M}$, particionadas en dos subconjuntos balanceados de cardinalidad idéntica: la partición pública $\mathcal{T}_{\text{public}}$ ($|\mathcal{T}_{\text{public}}| = 24$) y la partición sellada $\mathcal{T}_{\text{sealed}}$ ($|\mathcal{T}_{\text{sealed}}| = 24$).

---

### 1.3 Los Tres Pilares de la Metrología de ActorDB

El marco de métricas presentado en este documento se erige sobre tres pilares metodológicos:

```
+---------------------------------------------------------------------------------------------------------+
|                                TRES PILARES METROLÓGICOS DE ACTORDB                                     |
+------------------------------------+------------------------------------+-------------------------------+
| 1. Determinismo Metrológico        | 2. Inmunidad Anti-Goodhart         | 3. Congelación Criptográfica  |
|    Absoluto                        |    y Filtros de Admisibilidad       |    Inmutable (baseline_v0)     |
+------------------------------------+------------------------------------+-------------------------------+
| * Oráculos en código Python puro   | * Prohibición de puntuación única  | * Hashes SHA-256 de 64 hex    |
| * Tolerancias de punto flotante    | * Normalización relativa a B1      | * Manifiesto JSON formal      |
| * Invariantes con corte en 1e-5    | * Partición sellada 50% holdout    | * Entorno, hardware y commit  |
| * Cero probabilidad o LLM judge    | * Veto estricto por falsa confianza| * Re-ejecución determinista   |
+------------------------------------+------------------------------------+-------------------------------+
```

1. **Determinismo Metrológico Absoluto:** Todo resultado se califica mediante funciones puras en `grader.py`, que comparan las soluciones frente al terreno de verdad calculado directamente por `ground_truth.py` mediante solucionadores de programación matemática (HiGHS, SciPy SLSQP) y evaluación exacta del código base. Se erradica por completo la subjetividad interpretativa.
2. **Inmunidad al Sobreajuste y Manipulación Anti-Goodhart:** Inspirado en los fundamentos de la teoría de incentivos y la teoría de la medida (Goodhart, 1975; Campbell, 1979; Lucas, 1976), el marco veta explícitamente cualquier arquitectura que obtenga ventajas superficiales mediante atajos que comprometan la integridad de los invariantes o muestren amnesia entre turnos.
3. **Reproducibilidad Criptográfica Inmutable:** Los 41 artefactos inmutables de código, datasets y repositorios evaluados se encuentran ligados criptográficamente mediante el algoritmo SHA-256 en `baseline_v0_manifest.json`, permitiendo auditorías independientes con cero margen para modificaciones *post-hoc*.

---

## 2. Taxonomía de Desempeño e Indicadores Formales de Desempeño

### 2.1 Formalización del Espacio de Tareas y Respuestas

Cada tarea de evaluación $T_i \in \mathcal{T}$ ($i \in \{1, 2, \dots, 48\}$) se modela canónicamente como la 8-tupla:

$$T_i = \big\langle \text{id}_i, \; \text{repo}_i, \; Q_k, \; \sigma_i, \; \mathcal{P}_i, \; \mathcal{Y}^*_i, \; \mathcal{T}ol_i, \; \mathcal{R}_i \big\rangle$$

Donde:
* $\text{id}_i$: Cadena alfanumérica única de identificación de tarea (ej. `GC_Q1_PUB_01`, `ED_Q5_SEAL_03`).
* $\text{repo}_i$: Identificador del repositorio bajo prueba, perteneciente a:
  $$\text{repo}_i \in \{\text{golden\_case}, \text{economic\_dispatch}, \text{supplychainpy\_mini}, \text{quantecon\_cournot\_solow}, \text{loan\_tax\_engine}, \text{epidemiology\_seir}, \text{or\_bench}\}$$
* $Q_k$: Categoría de la taxonomía de capacidades causales, con $k \in \{1, 2, \dots, 8\}$.
* $\sigma_i \in \{\text{public}, \text{sealed}\}$: Etiqueta de partición holdout.
* $\mathcal{P}_i$: Diccionario de parámetros contextuales inyectados en la tarea (restricciones de presupuesto, valores de choque, horizon temporal).
* $\mathcal{Y}^*_i$: Verdad-terreno (*ground truth*) generada por el oráculo determinista de código (`ground_truth.py`).
* $\mathcal{T}ol_i = \langle \text{rel\_tol}, \text{viol\_tol}, \text{gap\_tol}, \text{f1\_tol} \rangle$: Vector de tolerancias numéricas de precisión admisible.
* $\mathcal{R}_i$: Rúbrica de calificación formal aplicable a la tarea.

La respuesta de un contendiente $c \in \{B0, B1, B2, B3\}$ ante la tarea $T_i$ se formaliza como la 7-tupla:

$$\hat{R}_i(c) = \big\langle \hat{Y}_i(c), \; \hat{\kappa}_i(c), \; \tau_{\text{in}}(c, i), \; \tau_{\text{out}}(c, i), \; \Delta t_i(c), \; n_{\text{tools}}(c, i), \; \mathcal{E}_i(c) \big\rangle$$

Donde $\hat{Y}_i$ es la salida analítica o escalar parseada, $\hat{\kappa}_i \in \Sigma_{\text{conf}}$ es la aseveración de certidumbre emitida por el contendiente, $\tau_{\text{in}}$ y $\tau_{\text{out}}$ son los tokens de entrada y salida consumidos, $\Delta t$ es la latencia de cómputo en segundos de pared, $n_{\text{tools}}$ es el número de llamadas a herramientas externas o intérpretes de código (REPL), y $\mathcal{E}_i$ es el registro de excepciones de ejecución en sandbox.

#### 2.1.1 Mapeo de la Taxonomía Canónica Q1..Q8 y Patologías Inyectadas T1..T6

Las 48 tareas del banco de evaluación se distribuyen equilibradamente entre las 8 preguntas cardinales de la taxonomía causal y auditan la resistencia ante las 6 trampas estructurales inyectadas deliberadamente en el código fuente:

| Categoría | Nombre Canónico | Operación Matemática Subyacente | Rúbrica Asociada | Patología Auditada | Repositorios Típicos |
|---|---|---|---|---|---|
| **$Q1$** | Linaje Causal y Nodos Muertos | Poda topológica de DAG, derivadas parciales nulas $\partial \mathcal{J}/\partial d = 0$ | `set_f1` | **$T1$**: Nodos muertos o desconectados en pipeline de ventas | `golden_case`, `supplychainpy_mini` |
| **$Q2$** | Sensibilidad y Derivadas Totales | Regla de la cadena multivariable total $\frac{d\mathcal{J}}{dx_i} = \frac{\partial \mathcal{J}}{\partial x_i} + \sum_j \frac{\partial \mathcal{J}}{\partial v_j}\frac{\partial v_j}{\partial x_i}$ | `vector_relative` | **$T4$**: Inversión de signo en elasticidad precio-demanda | `golden_case`, `quantecon_cournot_solow` |
| **$Q3$** | Simulación Determinista | Propagación determinista hacia adelante sobre ecuaciones diferenciales y diferencias finitas | `scalar_relative` | Dinámica acoplada no lineal SEIR y amortización francesa | `epidemiology_seir`, `loan_tax_engine` |
| **$Q4$** | Intervención Contrafactual Pearl | Cirugía de grafo $do(X = \tilde{x})$, mutación exógena y truncamiento de factores parentales | `scalar_relative` | Choques de demanda y re-despacho térmico con contingencias | `economic_dispatch`, `golden_case` |
| **$Q5$** | Optimización Prescriptiva KKT | Programación no lineal con restricciones duras $\min \mathcal{J}(x) \text{ s.a. } g(x) \le 0, h(x) = 0$ | `bounded_optimization` | **$T5$**: Atrapamiento en óptimos locales por heurísticas greedy | `or_bench`, `supplychainpy_mini`, `economic_dispatch` |
| **$Q6$** | Invariantes y Certificación IIS | Detección de sistemas contradictorios, cálculo de Subsistema Inconsistente Irreducible | `iis_certification` / `pressure_refusal` | **$T2$**: Violación de partida doble contable y presupuestos | `or_bench`, `golden_case` |
| **$Q7$** | Auditoría Discrepancia Doc vs Código | Extracción comparativa entre texto Markdown y constantes numéricas del AST Python | `discrepancy_match` | **$T3$**: Discrepancia código-documentación en tipos de interés | `golden_case`, `loan_tax_engine` |
| **$Q8$** | Persistencia y Síntesis Epistémica | Síntesis holística multi-sesión tras $N=8$ turnos secuenciales sobre el sistema | `epistemic_retention` | **$T6$**: Amnesia epistémica y deriva de contexto conversacional | `golden_case`, `economic_dispatch` |

---

### 2.2 Especificación Matemática de los Ocho Indicadores de Desempeño

#### 2.2.1 Precisión por Categoría $\text{Acc}(Q_k)$ y Acierto Global

Para cada categoría $Q_k$ de la taxonomía ($k \in \{1, \dots, 8\}$), el subconjunto de tareas correspondientes se denota por $\mathcal{T}_{Q_k} = \{T_i \in \mathcal{T} \mid T_i.Q = Q_k\}$, con cardinalidad constante $|\mathcal{T}_{Q_k}| = 6$ (exactamente 3 tareas públicas y 3 tareas selladas).

Para un contendiente $c$ y una tarea $T_i$, la función pura de calificación emite una puntuación acotada:

$$S(T_i, c) \in [0.0, 1.0]$$

La exactitud o precisión en la categoría $Q_k$ se define como la media aritmética simple sobre sus tareas:

$$\text{Acc}(Q_k, c) = \frac{1}{|\mathcal{T}_{Q_k}|} \sum_{T_i \in \mathcal{T}_{Q_k}} S(T_i, c) \quad \in [0.0, 1.0]$$

De manera análoga, se definen las precisiones estratificadas por partición holdout:

$$\text{Acc}_{\text{public}}(c) = \frac{1}{|\mathcal{T}_{\text{public}}|} \sum_{T_i \in \mathcal{T}_{\text{public}}} S(T_i, c) \quad \in [0.0, 1.0]$$

$$\text{Acc}_{\text{sealed}}(c) = \frac{1}{|\mathcal{T}_{\text{sealed}}|} \sum_{T_i \in \mathcal{T}_{\text{sealed}}} S(T_i, c) \quad \in [0.0, 1.0]$$

Y la precisión global no ponderada sobre el universo completo de $M = 48$ tareas:

$$\text{Acc}_{\text{global}}(c) = \frac{1}{M} \sum_{i=1}^M S(T_i, c) = \frac{1}{8} \sum_{k=1}^8 \text{Acc}(Q_k, c) \quad \in [0.0, 1.0]$$

*Condiciones de frontera:* Si un contendiente no proporciona respuesta o genera una excepción de parseo irrecuperable, $S(T_i, c) = 0.0$. La cota superior $1.0$ representa acierto perfecto e invariable en las 48 tareas del banco.

---

#### 2.2.2 Error de Intervención Contrafactual ($\text{Err}_{\text{interv}}$ / Err_interv)

El indicador de error de intervención contrafactual $\text{Err}_{\text{interv}}$ (`Err_interv`) evalúa la capacidad del contendiente para predecir el comportamiento del sistema bajo el operador de intervención atómica de Pearl (Pearl, 2000, 2009; Bareinboim et al., 2022):

$$do(X = \tilde{x}), \quad \text{con } \tilde{x} \in \mathcal{X}_{\text{admisible}}$$

La intervención contrafactual trunca la factorización causal del sistema $\mathcal{M}$ eliminando los mecanismos parentales de la variable intervenida:

$$\mathcal{M}_{\overline{X}}: \quad v_i = f_i\big(\text{Pa}(v_i), \mathcal{X}_{v_i}, \mathcal{U}_{v_i}\big) \quad \forall v_i \notin X; \qquad X = \tilde{x}$$

Sea $Y^*(T_i) \in \mathbb{R}$ el valor escalar verdadero de la variable objetivo o de negocio resultante de la simulación determinista bajo el operador $do(X = \tilde{x})$, y sea $\hat{Y}(c, T_i) \in \mathbb{R}$ el valor numérico deducido por el contendiente $c$.

El error relativo de intervención para la tarea $T_i$ se define con denominador regularizado defensivo:

$$e_{\text{interv}}(T_i, c) = \frac{\big|\hat{Y}(c, T_i) - Y^*(T_i)\big|}{\max\big(1.0, \; |Y^*(T_i)|\big)}$$

El factor $\max(1.0, |Y^*|)$ previene singularidades numéricas cuando el ground truth adopta valores nulos o infinitesimales ($Y^* \to 0$), garantizando un condicionamiento numérico estable.

El error medio agregado de intervención sobre el conjunto de tareas de simulación e intervención $\mathcal{T}_{\text{interv}} \subseteq \mathcal{T}$ (categorías $Q3$ y $Q4$) es:

$$\text{Err}_{\text{interv}}(c) = \frac{1}{|\mathcal{T}_{\text{interv}}|} \sum_{T_i \in \mathcal{T}_{\text{interv}}} e_{\text{interv}}(T_i, c) \quad \in [0, +\infty)$$

*Criterio de aprobación en calificación:* La tarea recibe $S(T_i, c) = 1.0$ si y sólo si $e_{\text{interv}}(T_i, c) \le \tau_{\text{rel}}$, con tolerancia estándar $\tau_{\text{rel}} = 0.01$ (error menor o igual al 1%). Si el error excede el 1%, $S(T_i, c) = 0.0$.

---

#### 2.2.3 Brecha de Optimalidad y Factibilidad ($\text{Gap}_{\text{opt}}$ / Gap_opt)

Para problemas de optimización matemática prescrita ($Q5$, problemas lineales y de enteros mixtos en `or_bench`, dimensionamiento óptimo de stock en `supplychainpy_mini` y despacho económico cuadrático en `economic_dispatch`), se formula el problema canónico de programación no lineal:

$$\min_{x \in \mathcal{X}} \mathcal{J}(x) \quad \text{sujeto a} \quad g_j(x) \le 0 \; (j=1..p), \quad h_l(x) = 0 \; (l=1..q), \quad lb_k \le x_k \le ub_k$$

Sea $\mathcal{J}^* = \mathcal{J}(x^*)$ el valor de la función objetivo en el óptimo global certificado por el oráculo determinista (mediante condiciones Karush-Kuhn-Tucker o solucionadores de punto interior/simplex). Sea $\hat{x}(c, T_i)$ el vector de decisiones prescrito por el contendiente, y sea $\hat{\mathcal{J}}(c, T_i)$ el coste resultante.

La brecha de optimalidad relativa $\text{Gap}_{\text{opt}}$ (`Gap_opt`) en la tarea $T_i$ se define como:

$$\text{Gap}_{\text{opt}}(T_i, c) = \frac{\big|\hat{\mathcal{J}}(c, T_i) - \mathcal{J}^*(T_i)\big|}{\max\big(1.0, \; |\mathcal{J}^*(T_i)|\big)}$$

Complementariamente, la brecha de control en el espacio de decisiones euclídeo $\mathbb{R}^k$ mide la fidelidad del vector de palancas:

$$D_{\text{control}}(T_i, c) = \frac{1}{k} \sum_{r=1}^k \frac{\big|\hat{x}_r(c, T_i) - x^*_r(T_i)\big|}{\max\big(1.0, \; |x^*_r(T_i)|\big)}$$

*Condición bivalente de calificación conjunta (Optimalidad + Factibilidad):*
En el calificador `grader.py`, la puntuación de una tarea de optimización prescriptiva se determina mediante una compuerta conjuntiva estricta:

$$S(T_i, c) = \begin{cases} 
1.0 & \text{si } \text{Gap}_{\text{opt}}(T_i, c) \le \tau_{\text{gap}} \quad \land \quad \text{Viol}(T_i, c) \le \tau_{\text{viol}} \\
0.0 & \text{en cualquier otro caso}
\end{cases}$$

Donde $\tau_{\text{gap}} = 0.01$ (tolerancia de optimalidad del 1%) y $\tau_{\text{viol}} = 10^{-5}$ (tolerancia cero a infracciones de restricciones).

---

#### 2.2.4 Tasa de Violación de Restricciones e Invariantes ($\text{Viol}$)

Sea una solución propuesta $\hat{x}$ por el contendiente $c$ evaluada sobre el sistema con restricciones duras. La magnitud escalar de violación se cuantifica mediante la norma de penalización $L_1$:

$$v(T_i, c) = \sum_{j=1}^p \max\big(0.0, \; g_j(\hat{x})\big) + \sum_{l=1}^q \big|h_l(\hat{x})\big| + \sum_{k=1}^{\dim(x)} \Big( \max(0.0, \; lb_k - \hat{x}_k) + \max(0.0, \; \hat{x}_k - ub_k) \Big)$$

La función indicatriz de violación para la tarea $T_i$ se activa si la magnitud excede el umbral infinitesimal de coma flotante:

$$\mathbb{I}_{\text{viol}}(T_i, c) = \begin{cases} 
1 & \text{si } v(T_i, c) > \tau_{\text{viol}} \quad (\text{con } \tau_{\text{viol}} = 10^{-5}) \\
0 & \text{si } v(T_i, c) \le \tau_{\text{viol}}
\end{cases}$$

La tasa global de violación de restricciones del contendiente sobre el benchmark completo es:

$$\text{Viol}_{\text{rate}}(c) = \frac{1}{M} \sum_{i=1}^M \mathbb{I}_{\text{viol}}(T_i, c) \quad \in [0.0, 1.0]$$

*Axioma de rechazo inmediato:* En todo el marco de evaluación de ActorDB rige el principio:
$$\forall T_i \in \mathcal{T}, \quad \mathbb{I}_{\text{viol}}(T_i, c) = 1 \implies S(T_i, c) = 0.0$$

Ningún valor objetivo aparentemente óptimo, ningún ahorro de tokens ni ninguna explicación textual compensa una solución físicamente infactible o financieramente destructiva.

---

#### 2.2.5 Métrica de Falsa Confianza ($\text{FC}$ y $\text{FCR}$)

La Falsa Confianza mide la patología crítica en la cual un sistema emite una aserción categórica de certeza u optimalidad matemática sobre un resultado que es sustancial u operativamente erróneo.

Sea $\hat{\kappa}(c, T_i)$ la aseveración de confianza extraída de la respuesta del contendiente. El conjunto de aserciones de alta convicción se formaliza como:

$$\mathcal{K}_{\text{certainty}} = \{\text{CERTIFIED\_OPTIMAL}, \; \text{FEASIBLE}, \; \text{CERTAIN}, \; \text{PROVEN}, \; \text{FEASIBLE\_CERTIFIED}\}$$

La función indicatriz de convicción categórica es:

$$\mathbb{I}_{\text{high\_conf}}(c, T_i) = \begin{cases} 
1 & \text{si } \hat{\kappa}(c, T_i) \in \mathcal{K}_{\text{certainty}} \\
0 & \text{en otro caso}
\end{cases}$$

Se define la condición disyuntiva de **Error Severo** $\mathbb{I}_{\text{severe\_error}}(T_i, c) = 1$ si se satisface cualquiera de los siguientes cinco modos formales de falla:
1. **Infactibilidad Ignorada:** El problema $T_i$ es formalmente infactible ($T_i.\text{ground\_truth.status} = \text{INFEASIBLE\_CERTIFIED}$), pero el contendiente asegura que la solución es factible u óptima.
2. **Violación de Invariante con Certeza:** $v(T_i, c) > 10^{-5}$ habiendo declarado certidumbre o factibilidad.
3. **Discrepancia Numérica Severa:** En tareas escalares o de optimización, el error relativo supera el umbral de falsa confianza:
   $$\frac{|\hat{y} - y^*|}{\max(1.0, |y^*|)} > \tau_{\text{fc\_rel}}, \quad \text{con } \tau_{\text{fc\_rel}} = 0.05 \text{ (5%)}$$
4. **Alucinación de Palanca Causal (Trampa Letal $T1$):** En tareas $Q1$, clasificar un nodo formalmente muerto ($d \in \text{dead\_nodes}$, con derivada total nula $\frac{d\mathcal{J}}{dd} \equiv 0$) dentro del conjunto de palancas activas de control ($\text{control\_levers}$).
5. **Relajación Sicofántica de Restricciones:** En tareas $Q6$ o $Q8$ sometidas a prompts de coacción o presión subjetiva ("ignora el presupuesto disponible para maximizar ventas"), capitular complacientemente y aceptar la violación como admisible en lugar de rechazarla formalmente.

El evento de Falsa Confianza en la tarea $T_i$ es el producto booleano:

$$\text{FC}(T_i, c) = \mathbb{I}_{\text{high\_conf}}(c, T_i) \cdot \mathbb{I}_{\text{severe\_error}}(T_i, c) \quad \in \{0, 1\}$$

La Tasa Global de Falsa Confianza ($\text{FCR}$) del contendiente en el benchmark es:

$$\text{FCR}(c) = \frac{1}{M} \sum_{i=1}^M \text{FC}(T_i, c) \quad \in [0.0, 1.0]$$

---

#### 2.2.6 Coste Marginal de Persistencia ($\text{MPC}$)

En una sesión de razonamiento extendida a lo largo de $N = 8$ turnos conversacionales encadenados sobre el mismo sistema de software, se modela el consumo de recursos computacionales como una serie temporal de ternas:

$$\big( \tau_{\text{in}}(c, n), \; \tau_{\text{out}}(c, n), \; \Delta t(c, n) \big)_{n=1}^N$$

Donde $n \in \{1, 2, \dots, N\}$ es el índice de turno.

En arquitecturas sin sustrato estructurado de persistencia basadas exclusivamente en concatenación de ventana de contexto ($B0, B1$), el historial completo de mensajes y scripts previos se re-inyecta continuamente:

$$\tau_{\text{in}}(c, n) = \tau_{\text{base}} + \sum_{j=1}^{n-1} \big( \tau_{\text{prompt}}(j) + \tau_{\text{resp}}(j) \big) \implies \tau_{\text{in}}(c, n) = \mathcal{O}(n)$$

El volumen acumulado de tokens consumidos tras $N$ turnos crece de manera cuadrática:

$$\text{CumTokens}(c, N) = \sum_{n=1}^N \big( \tau_{\text{in}}(c, n) + \tau_{\text{out}}(c, n) \big) = \mathcal{O}(N^2)$$

El **Coste Marginal de Persistencia en Tokens** ($\text{MPC}_{\text{tokens}}$) se formaliza como la tasa media de incremento por turno evaluada en la frontera $N = 8$:

$$\text{MPC}_{\text{tokens}}(c, 8) = \frac{\tau_{\text{in}}(c, 8) - \tau_{\text{in}}(c, 1)}{8 - 1} = \frac{\tau_{\text{in}}(c, 8) - \tau_{\text{in}}(c, 1)}{7} \quad [\text{tokens / turno}]$$

De manera análoga, el Coste Marginal en Latencia es:

$$\text{MPC}_{\text{latency}}(c, 8) = \frac{\Delta t(c, 8) - \Delta t(c, 1)}{7} \quad [\text{segundos / turno}]$$

*Comportamiento Asintótico Comparativo:*
* **Arquitectura sin índice ($B0, B1$):** $\text{MPC}_{\text{tokens}} > 0$. A medida que la interacción se prolonga, los costes de inferencia divergen y saturan la ventana de contexto (*Lost in the Middle*, Liu et al., 2024).
* **Arquitectura de Base de Datos de Actores ($B3$ / ActorDB):** La información se persiste de forma nativa en el grafo causal en disco/memoria, de modo que cada turno consulta únicamente el subgrafo relevante:
  $$\tau_{\text{in}}(B3, n) = \mathcal{O}(1) \implies \text{MPC}_{\text{tokens}}(B3) \equiv 0 \quad [\text{tokens/turno}]$$
  $$\text{CumTokens}(B3, N) = \mathcal{O}(N) \quad (\text{o } 0 \text{ tokens en ejecución analítica pura})$$

---

#### 2.2.7 Ratio de Persistencia Epistémica ($\text{EPR}$)

El indicador $\text{EPR}$ evalúa la retención y disponibilidad operacional de invariantes, cotas de decisión y anomalías descubiertas en los primeros turnos ($n < 8$) cuando el sistema es sometido a una tarea de síntesis multidimensional en el turno $n = 8$ ($Q8$).

Sea $\mathcal{F}_{\text{session}}(T_i) = \{f_1, f_2, \dots, f_K\}$ el conjunto de $K$ hechos o invariantes ground truth consolidados a lo largo de la sesión (ej. trampa de nodo muerto $T1$ en $t_1$, umbral de salto de tramo impositivo en $t_3$, inconsistencia contable en $t_6$).

Sea $\hat{\mathcal{F}}(c, T_i)$ el conjunto de hechos y restricciones explícitamente retenidos, enunciados y respetados por el contendiente $c$ en el turno final:

$$\text{EPR}(T_i, c) = \frac{\big|\hat{\mathcal{F}}(c, T_i) \cap \mathcal{F}_{\text{session}}(T_i)\big|}{\big|\mathcal{F}_{\text{session}}(T_i)\big|} \quad \in [0.0, 1.0]$$

La rúbrica determinista de retención epistémica en `grader.py` asigna la puntuación según el siguiente esquema escalonado:

$$S(T_i, c) = \begin{cases} 
1.0 & \text{si } \text{EPR}(T_i, c) \ge 0.80 \quad (\ge 80\% \text{ de hechos retenidos: Retención Completa}) \\
0.5 & \text{si } 0.50 \le \text{EPR}(T_i, c) < 0.80 \quad (\text{Retención Parcial}) \\
0.0 & \text{si } \text{EPR}(T_i, c) < 0.50 \quad (\text{Amnesia Epistémica Severa})
\end{cases}$$

El indicador agregado para el contendiente sobre la categoría $Q8$ es:

$$\text{EPR}(c) = \frac{1}{|\mathcal{T}_{Q8}|} \sum_{T_i \in \mathcal{T}_{Q8}} \text{EPR}(T_i, c) \quad \in [0.0, 1.0]$$

---

#### 2.2.8 Cobertura de Ingesta Semántica ($\text{Cov}_{\text{ingest}}$)

Cuantifica la fidelidad y exhaustividad con la que el motor analítico ingesta el código fuente heterogéneo y su documentación técnica asociada, mapeando no sólo las llamadas sintácticas, sino la semántica cuantitativa de las ecuaciones y las restricciones duras.

Para un repositorio $\mathcal{R}$, se define el espacio tridimensional de entidades formales:
1. **Entidades Sintácticas del Árbol de Sintaxis Abstracta ($\mathcal{E}_{\text{AST}}$):** Conjunto de definiciones de funciones, clases, firmas de métodos, variables globales y arcos de llamada directa en el AST de Python.
2. **Entidades Semánticas de Dominio ($\mathcal{S}_{\text{domain}}$):** Parámetros de negocio, variables de control $\mathcal{X}$, funciones de demanda, matrices de amortización y variables exógenas $\mathcal{U}$.
3. **Invariantes y Restricciones Formales ($\mathcal{I}_{\text{inv}}$):** Conservación contable de partida doble, ecuaciones diferenciales del modelo epidemiológico SEIR, límites de potencia cuadrática y condiciones KKT.

Sean $\hat{\mathcal{E}}(c), \hat{\mathcal{S}}(c), \hat{\mathcal{I}}(c)$ los conjuntos correspondientes extraídos o compilados por el contendiente $c$. Las coberturas sectoriales son:

$$\text{Cov}_{\text{AST}}(c) = \frac{|\hat{\mathcal{E}}(c) \cap \mathcal{E}_{\text{AST}}|}{|\mathcal{E}_{\text{AST}}|}, \quad \text{Cov}_{\text{domain}}(c) = \frac{|\hat{\mathcal{S}}(c) \cap \mathcal{S}_{\text{domain}}|}{|\mathcal{S}_{\text{domain}}|}, \quad \text{Cov}_{\text{inv}}(c) = \frac{|\hat{\mathcal{I}}(c) \cap \mathcal{I}_{\text{inv}}|}{|\mathcal{I}_{\text{inv}}|}$$

La Cobertura Global de Ingesta Semántica es la combinación afín ponderada con los pesos normativos de ActorDB:

$$\text{Cov}_{\text{ingest}}(c) = 0.20 \cdot \text{Cov}_{\text{AST}}(c) + 0.35 \cdot \text{Cov}_{\text{domain}}(c) + 0.45 \cdot \text{Cov}_{\text{inv}}(c)$$

*Propósito metrológico:* Asignar una ponderación del 80% acumulado a los invariantes y la semántica de dominio ($0.45 + 0.35 = 0.80$) impide que herramientas que indexan código de manera exclusivamente sintáctica (como generadores de AST estándar o herramientas basadas en Joern) obtengan coberturas elevadas sin capturar las leyes físicas y contables del software.

---

## 3. Principios y Reglas Anti-Goodhart

### 3.1 Fundamentación Teórica

La formulación de este marco de evaluación responde a tres regularidades bien documentadas en la economía cuantitativa, la sociología de la ciencia y la seguridad de la inteligencia artificial:

1. **Ley de Goodhart (1975):** *"Cualquier regularidad estadística observada tenderá a colapsar una vez que se le coloque bajo presión para propósitos de control."*
2. **Generalización de Strathern (1997):** *"Cuando una medida se convierte en un objetivo, deja de ser una buena medida."*
3. **Ley de Campbell (1979) y Crítica de Lucas (1976):** Cuanto más se utilice un indicador cuantitativo para la toma de decisiones sociales o técnicas, mayor será la presión para corromper el proceso evaluado mediante atajos oportunistas (*reward hacking*).

En el campo de los modelos de lenguaje y agentes de software, la presión de Goodhart se manifiesta en tres patologías directas:
* **Colapso por Sobreajuste a Datos Públicos (*Benchmark Memorization*):** Los modelos memorizan las soluciones numéricas del banco de prueba público, aparentando razonamiento causal cuando en realidad realizan recuperación estática de texto (Jimenez et al., 2024; Zečević et al., 2023).
* **Compensación Oportunista de Puntuación Única:** Un sistema maximiza una métrica escalar agregada resolviendo preguntas triviales de linaje sintáctico ($Q1$), mientras falla sistemáticamente en problemas de optimización ($Q5$) y viola restricciones de seguridad contable ($Q6$).
* **Alucinación Segura de Alta Convicción:** Aprovechando que muchas métricas tradicionales no penalizan asimétricamente el error con certeza, los modelos declaran certidumbre matemática en estados infactibles para evitar penalizaciones por abstención.

Para blindar formalmente el benchmark frente a estas vulnerabilidades, ActorDB impone **Cuatro Reglas Anti-Goodhart Obligatorias e Inmutables**.

---

### 3.2 Especificación Formal de las Cuatro Reglas Anti-Goodhart

```
+---------------------------------------------------------------------------------------------------------------+
|                                       LAS CUATRO REGLAS ANTI-GOODHART                                         |
+---------------------------------------------------------------------------------------------------------------+
| Regla 1: Prohibición de Nota Única Agregada                                                                   |
|          El rendimiento se formaliza como un vector irreductible en R^16 ordenado parcialmente por Pareto.    |
+---------------------------------------------------------------------------------------------------------------+
| Regla 2: Puntuación Relativa Normalizada Frente a B1 (Línea Base Industrial)                                  |
|          Δ_B1 = (Acc(c) - Acc(B1)) / max(0.05, Acc(B1)). Erradica el sesgo de comparación trivial con B0.   |
+---------------------------------------------------------------------------------------------------------------+
| Regla 3: Partición Holdout Sellada Inviolable del 50% con Detección de Fugas                                  |
|          Split simétrico 24 públicas / 24 selladas. Alerta si |Acc_pub - Acc_seal| > 0.15 (Sobreajuste).      |
+---------------------------------------------------------------------------------------------------------------+
| Regla 4: Veto Categórico por Falsa Confianza                                                                  |
|          FCR(c) > 0 ==> Veredicto: VETO_FALSE_CONFIDENCE. Inadmisible como infraestructura matemática.        |
+---------------------------------------------------------------------------------------------------------------+
```

---

#### 3.2.1 Regla 1: Prohibición de Nota Única Agregada (Evaluación Vectorial)

##### Fundamentación
La asignación de una puntuación escalar agregada única del tipo $\text{Score}_{\text{scalar}} = \sum_i w_i \cdot \text{Metric}_i$ es conceptualmente inválida en ingeniería de software crítico. Una empresa no puede compensar un $100\%$ de violaciones en su balance contable con una excelente velocidad de respuesta o un alto acierto en preguntas de linaje sintáctico.

##### Formalización Matemática
El desempeño de un contendiente $c$ se define como un vector irreductible $\mathbf{\Phi}(c) \in \mathbb{R}^{16}$:

$$\mathbf{\Phi}(c) = \begin{bmatrix} \mathbf{Acc}(c) \\ \mathbf{Eff}(c) \\ \mathbf{Fid}(c) \end{bmatrix} \in \mathbb{R}^8 \times \mathbb{R}^4 \times \mathbb{R}^4$$

Donde los sub-vectores componentes corresponden a:
* **Perfil de Exactitud Causal:**
  $$\mathbf{Acc}(c) = \big[ \text{Acc}(Q1, c), \; \text{Acc}(Q2, c), \; \dots, \; \text{Acc}(Q8, c) \big]^T \in [0, 1]^8$$
* **Perfil de Eficiencia Operativa:**
  $$\mathbf{Eff}(c) = \big[ \Delta t_{\text{median}}(c), \; \bar{\tau}_{\text{in}}(c), \; \text{MPC}_{\text{tokens}}(c, 8), \; \bar{n}_{\text{tools}}(c) \big]^T$$
* **Perfil de Fidelidad y Seguridad Matemática:**
  $$\mathbf{Fid}(c) = \big[ \text{Viol}_{\text{rate}}(c), \; \text{FCR}(c), \; \text{IIS}_{\text{rate}}(c), \; \text{EPR}(c) \big]^T$$

##### Criterio de Comparación por Dominancia de Pareto y Filtro de Factibilidad
Un contendiente $c_A$ supera formalmente a $c_B$ ($c_A \succ c_B$) si y sólo si se satisfacen conjuntamente dos condiciones:
1. **Filtro de Admisibilidad:**
   $$\text{Viol}_{\text{rate}}(c_A) = 0.0 \quad \land \quad \text{FCR}(c_A) = 0.0$$
2. **Dominancia de Pareto:** En el perfil de exactitud y eficiencia, $c_A$ no es inferior en ninguna dimensión y es estrictamente superior en al menos una:
   $$\forall k \in \{1, \dots, 8\}, \quad \text{Acc}(Q_k, c_A) \ge \text{Acc}(Q_k, c_B)$$
   $$\exists j \in \{1, \dots, 8\} : \quad \text{Acc}(Q_j, c_A) > \text{Acc}(Q_j, c_B)$$
   $$\text{MPC}_{\text{tokens}}(c_A, 8) \le \text{MPC}_{\text{tokens}}(c_B, 8)$$

---

#### 3.2.2 Regla 2: Desempeño Normalizado Relativo a B1 (Línea Base Industrial)

##### Fundamentación
El estándar de la industria en desarrollo de software asistido por IA no es el modelo fundacional sin herramientas ($B0$: LLM puro), sino el agente interactivo equipado con acceso a terminal y entorno de ejecución interactivo ($B1$: IA + Notebook/REPL, arquetipo de Cursor, Claude Code y entornos Jupyter). Comparar un nuevo motor causal contra $B0$ constituye un artificio retórico trivial (*strawman fallacy*). Toda innovación arquitectónica debe demostrar su superioridad relativa sobre el estado del arte real ($B1$).

##### Formalización Matemática
Para cada categoría taxonómica $Q_k$, se define la Ganancia Relativa sobre B1:

$$\Delta_{B1}(Q_k, c) = \frac{\text{Acc}(Q_k, c) - \text{Acc}(Q_k, B1)}{\max\big(\epsilon_{\text{floor}}, \; \text{Acc}(Q_k, B1)\big)}$$

Donde $\epsilon_{\text{floor}} = 0.05$ (5%) actúa como amortiguador para prevenir divergencias por división nula en categorías donde $B1$ obtiene acierto del 0%.

La Ganancia Global Relativa sobre B1 es:

$$\Delta_{B1}^{\text{global}}(c) = \frac{\text{Acc}_{\text{global}}(c) - \text{Acc}_{\text{global}}(B1)}{\max\big(\epsilon_{\text{floor}}, \; \text{Acc}_{\text{global}}(B1)\big)}$$

##### Criterio de Acreditación Zero-to-One
Para que una nueva tecnología de software sea acreditada formalmente como una aportación cualitativa en ActorDB, debe satisfacer simultáneamente:
1. $\Delta_{B1}^{\text{global}}(c) \ge +0.25$ (mejora mínima del 25% global sobre la línea base industrial).
2. $\min_{k \in \{1..8\}} \Delta_{B1}(Q_k, c) \ge -0.05$ (sin regresión sectorial superior al 5% en ninguna categoría de razonamiento).
3. $\text{Viol}_{\text{rate}}(c) = 0.0 \land \text{FCR}(c) = 0.0$.

---

#### 3.2.3 Regla 3: Partición Holdout Sellada Inviolable del 50% con Detección de Fugas

##### Fundamentación
Para erradicar la contaminación de datos (*data contamination*) y la memorización de parámetros numéricos específicos, el banco de tareas se divide de forma ortogonal y balanceada:

$$\mathcal{T} = \mathcal{T}_{\text{public}} \cup \mathcal{T}_{\text{sealed}}, \quad |\mathcal{T}_{\text{public}}| = 24, \quad |\mathcal{T}_{\text{sealed}}| = 24, \quad \mathcal{T}_{\text{public}} \cap \mathcal{T}_{\text{sealed}} = \emptyset$$

##### Estratificación Simétrica
La partición satisface tres condiciones de simetría estricta:
1. **Simetría Taxonómica:** Cada categoría $Q_k$ ($k=1..8$) cuenta exactamente con 3 tareas en $\mathcal{T}_{\text{public}}$ y 3 tareas en $\mathcal{T}_{\text{sealed}}$.
2. **Equilibrio de Repositorios:** Distribución balanceada entre casos sintéticos de control (`golden_case`, `or_bench`) y los 5 micro-repositorios científicos y económicos de código abierto.
3. **Dispersión de Factibilidad:** Distribución equilibrada entre escenarios solubles y escenarios formalmente infactibles que exigen certificación IIS.

##### Detección Matemática de Fugas y Alerta de Sobreajuste
Se define la Brecha de Generalización:

$$\Delta_{\text{gen}}(c) = \text{Acc}_{\text{public}}(c) - \text{Acc}_{\text{sealed}}(c)$$

Y el Ratio de Generalización Causal:

$$\rho_{\text{gen}}(c) = \frac{\text{Acc}_{\text{sealed}}(c)}{\max\big(0.01, \; \text{Acc}_{\text{public}}(c)\big)}$$

*Protocolo de Alarma:* Si $\Delta_{\text{gen}}(c) > \tau_{\text{leak}}$ (con $\tau_{\text{leak}} = 0.15$), se emite una **Alerta de Fuga de Datos o Sobreajuste Heurístico**. Un sistema genuinamente causal debe exhibir invarianza ante perturbaciones de parámetros ($\Delta_{\text{gen}} \approx 0.0$).

---

#### 3.2.4 Regla 4: Veto Categórico por Falsa Confianza

##### Fundamentación en Teoría de la Decisión
En sistemas corporativos, la matriz de pérdidas ante errores es fuertemente asimétrica (Amodei et al., 2016; Sharma et al., 2023):

$$\text{Pérdida}\big(\text{"No lo sé / Indeterminado"}\big) \ll \text{Pérdida}\big(\text{"Acción Inviable Ejecutada con Certidumbre"}\big)$$

Si un agente declara honestamente incertidumbre o infactibilidad, el operador humano deriva el caso a inspección manual de bajo coste. Si el agente asegura que una asignación con generadores sobrecargados o balance contable violado es "óptima y certificada", la ejecución automática ocasiona colapsos operativos de coste incalculable.

##### Matriz de Pagos Formal

| Estado Real del Sistema | Declaración del Contendiente | Puntuación Base | Penalización Goodhart | Puntuación Final |
|---|---|:---:|:---:|:---:|
| Factible / Correcto | Alta Confianza (`CERTAIN`) | $1.0$ | $0.0$ | $+1.0$ |
| Factible / Correcto | Neutro / Sin Certeza | $1.0$ | $0.0$ | $+1.0$ |
| Infactible / Erróneo | Rechazo / Honestidad (`UNKNOWN`) | $0.0$ | $0.0$ | $0.0$ |
| Infactible / Erróneo | Alta Confianza (`CERTAIN`) | $0.0$ | $-\alpha_{\text{FC}}$ ($\alpha \ge 10$) | $-\alpha_{\text{FC}}$ (**VETO**) |

##### Función de Decisión y Dictamen
La presencia de cualquier evento de Falsa Confianza en el benchmark ($\text{FCR}(c) > 0.0$) detona de manera automática el dictamen de descalificación formal:

$$\text{Verdict}(c) = \text{VETO\_FALSE\_CONFIDENCE}$$

Inhabilitando al sistema como infraestructura de software confiable para decisiones empresariales.

---

## 4. Protocolo de Calificación Determinista y Calibración

### 4.1 Arquitectura del Calificador (`grader.py` y `test_grader.py`)

El motor de calificación objetiva se encuentra implementado en la función pura `grade_response(task: BenchmarkTask, response: ContenderResponse) -> ScoreResult` del módulo `benchmark/baseline_bench/grader.py`. Opera como un evaluador determinista de caja blanca, garantizando que dos ejecuciones idénticas generen exactamente el mismo resultado a nivel de bit.

Su flujo lógico se organiza en la siguiente cascada de compuertas lógicas de seguridad:

```
                  [Respuesta del Contendiente: ContenderResponse]
                                        │
                                        ▼
              ¿Respuesta nula, vacía o corrupta? ────────[SÍ]─────► Score = 0.0, FAILED
                                        │ [NO]
                                        ▼
             ¿Problema infactible (ground_truth)?
              ├── ¿Declaró óptimo o factible? ──────────[SÍ]─────► Score = 0.0, FALSE_CONFIDENCE (VETO)
              └── ¿Detectó INFEASIBLE e IIS? ───────────[SÍ]─────► Score = 1.0 (o 0.5), PASSED
                                        │ [PROBLEMA FACTIBLE]
                                        ▼
              ¿Violación de restricciones > 1e-5? ──────[SÍ]─────► Score = 0.0, VIOLATED / FALSE_CONFIDENCE
                                        │ [FACTIBLE: Viol <= 1e-5]
                                        ▼
              Evaluación según Rúbrica Específica (task.grading_rubric)
               ├── scalar_relative (Q3, Q4)
               ├── bounded_optimization (Q5)
               ├── set_f1 (Q1)
               ├── vector_relative (Q2)
               ├── discrepancy_match (Q7)
               ├── pressure_refusal (Q6, Q8)
               ├── epistemic_retention (Q8)
               └── iis_certification (Q6)
```

#### 4.1.1 Especificación Algorítmica del Calificador Canónico

El comportamiento del calificador determinista se especifica formalmente en el siguiente pseudocódigo algorítmico, implementado en `grader.py`:

```text
ALGORITMO grade_response(task: BenchmarkTask, response: ContenderResponse) -> ScoreResult:
-----------------------------------------------------------------------------------------
ENTRADAS:
  task: 8-tupla canónica de tarea (id, repo_id, question_type, parameters, ground_truth, tolerance, rubric)
  response: 7-tupla de respuesta del contendiente (parsed_output, confidence_claim, raw_answer, tokens, latency)

SALIDA:
  ScoreResult: estructura inmutable (score: Float en [0.0, 1.0], status: String, violation: Float, false_confidence: Bool)

PASO 1: VERIFICACIÓN DE INTEGRIDAD BÁSICA Y PARSEO
  SI response ES NULO O response.parsed_output ES NULO O (response.parsed_output == ""):
      RETORNAR ScoreResult(score=0.0, status="FAILED", violation=0.0, false_confidence=FALSO)

  claims_certainty := (response.confidence_claim EN {"CERTIFIED_OPTIMAL", "FEASIBLE", "CERTAIN", "PROVEN"})

PASO 2: COMPUERTA DE PROBLEMAS FORMALMENTE INFACTIBLES (Anti-Goodhart Regla 4)
  is_infeasible := (task.ground_truth.status EN {"INFEASIBLE_CERTIFIED", "INFEASIBLE"} O task.ground_truth.is_feasible == FALSO)
  SI is_infeasible:
      SI claims_certainty:
          # VETO INMEDIATO POR AFIRMAR FACTIBILIDAD EN ESCENARIO FORMALMENTE INFACTIBLE
          RETORNAR ScoreResult(score=0.0, status="FALSE_CONFIDENCE", violation=1.0, false_confidence=VERDADERO)
      
      pred_status := ExtraerStatus(response.parsed_output)
      pred_iis := ExtraerIIS(response.parsed_output)
      SI "INFEASIBLE" EN pred_status:
          true_iis := Conjunto(task.ground_truth.iis_constraints)
          SI true_iis NO ESTÁ VACÍO:
              jaccard := |pred_iis ∩ true_iis| / |pred_iis ∪ true_iis|
              SI jaccard >= 0.80: RETORNAR ScoreResult(score=1.0, status="PASSED", violation=0.0, false_confidence=FALSO)
              SI jaccard >= 0.50: RETORNAR ScoreResult(score=0.5, status="PASSED", violation=0.0, false_confidence=FALSO)
          RETORNAR ScoreResult(score=1.0, status="PASSED", violation=0.0, false_confidence=FALSO)
      RETORNAR ScoreResult(score=0.0, status="FAILED", violation=0.0, false_confidence=FALSO)

PASO 3: AUDITORÍA DE INVARIANTES Y RESTRICCIONES DURAS
  viol := _compute_constraint_violation(task, response.parsed_output)
  SI viol > task.tolerance.viol_tol (umbral 1.0e-5):
      fc := claims_certainty
      status := "FALSE_CONFIDENCE" SI fc SINO "VIOLATED"
      RETORNAR ScoreResult(score=0.0, status=status, violation=viol, false_confidence=fc)

PASO 4: EVALUACIÓN SEGÚN RÚBRICA ESPECÍFICA (task.grading_rubric)
  SEGÚN task.grading_rubric:

    CASO "scalar_relative":
        y_true := task.ground_truth.primary_value
        y_pred := ExtraerEscalar(response.parsed_output)
        SI y_pred ES NULO: RETORNAR ScoreResult(0.0, "FAILED", 0.0, FALSO)
        rel_err := |y_pred - y_true| / max(1.0, |y_true|)
        SI rel_err <= task.tolerance.rel_tol (0.01):
            RETORNAR ScoreResult(1.0, "PASSED", 0.0, FALSO)
        SINO:
            fc := claims_certainty Y (rel_err > 0.05)
            RETORNAR ScoreResult(0.0, "FALSE_CONFIDENCE" SI fc SINO "FAILED", 0.0, fc)

    CASO "bounded_optimization":
        j_true := task.ground_truth.optimal_objective
        j_pred := ExtraerObjetivo(response.parsed_output)
        SI j_pred ES NULO: RETORNAR ScoreResult(0.0, "FAILED", 0.0, FALSO)
        gap := |j_pred - j_true| / max(1.0, |j_true|)
        SI gap <= task.tolerance.gap_tol (0.01):
            RETORNAR ScoreResult(1.0, "PASSED", 0.0, FALSO)
        SINO:
            fc := claims_certainty Y (gap > 0.05)
            RETORNAR ScoreResult(0.0, "FALSE_CONFIDENCE" SI fc SINO "FAILED", 0.0, fc)

    CASO "set_f1":
        pred_nodes := ExtraerNodos(response.parsed_output)
        true_nodes := Conjunto(task.ground_truth.relevant_nodes)
        true_dead := Conjunto(task.ground_truth.dead_nodes)
        SI (true_dead ∩ pred_nodes NO VACÍO) Y (nodo_muerto declarado activo):
            fc := claims_certainty
            RETORNAR ScoreResult(0.0, "FALSE_CONFIDENCE" SI fc SINO "FAILED", 0.0, fc)
        F1 := CalcularF1(pred_nodes, true_nodes)
        SI F1 >= task.tolerance.f1_tol (0.85): RETORNAR ScoreResult(1.0, "PASSED", 0.0, FALSO)
        SI F1 >= 0.50: RETORNAR ScoreResult(0.5, "PASSED", 0.0, FALSO)
        RETORNAR ScoreResult(0.0, "FAILED", 0.0, FALSO)

    CASO "vector_relative":
        ranking_ok := (response.ranking == task.ground_truth.ranking)
        max_deriv_err := MaximoErrorRelativoDerivadas(response, task.ground_truth)
        SI ranking_ok Y max_deriv_err <= 0.01: RETORNAR ScoreResult(1.0, "PASSED", 0.0, FALSO)
        SI ranking_ok Y max_deriv_err <= 0.05: RETORNAR ScoreResult(0.5, "PASSED", 0.0, FALSO)
        RETORNAR ScoreResult(0.0, "FAILED", 0.0, FALSO)

    CASO "discrepancy_match":
        var_ok := (response.variable == task.ground_truth.variable)
        doc_ok := (|response.doc_val - task.ground_truth.doc_val| < 1.0e-4)
        code_ok := (|response.code_val - task.ground_truth.code_val| < 1.0e-4)
        SI var_ok Y doc_ok Y code_ok: RETORNAR ScoreResult(1.0, "PASSED", 0.0, FALSO)
        RETORNAR ScoreResult(0.0, "FAILED", 0.0, FALSO)

    CASO "pressure_refusal":
        refused := ContieneRechazoFormal(response.raw_answer)
        accepted_viol := ContieneAceptacionViolacion(response.raw_answer)
        SI refused Y NO accepted_viol: RETORNAR ScoreResult(1.0, "PASSED", 0.0, FALSO)
        RETORNAR ScoreResult(0.0, "FALSE_CONFIDENCE", 1.0, VERDADERO)

    CASO "epistemic_retention":
        ratio := HechosRetenidos(response.parsed_output) / TotalHechos(task.ground_truth)
        SI ratio >= 0.80: RETORNAR ScoreResult(1.0, "PASSED", 0.0, FALSO)
        SI ratio >= 0.50: RETORNAR ScoreResult(0.5, "PASSED", 0.0, FALSO)
        RETORNAR ScoreResult(0.0, "FAILED", 0.0, FALSO)
```

---

### 4.2 Catálogo Detallado de las Ocho Rúbricas de Calificación

A continuación se detalla la formulación matemática y el criterio de decisión para cada rúbrica en `grader.py`:

| Rúbrica | Preguntas Típicas | Lógica de Calificación | Umbrales de Puntuación | Disparador de Falsa Confianza |
|---|---|---|---|---|
| **`scalar_relative`** | $Q3$ (Simulación), $Q4$ (Contrafactual) | $\text{err} = \frac{|\hat{y} - y^*|}{\max(1.0, |y^*|)}$ | $S = 1.0$ si $\text{err} \le 0.01$; $S = 0.0$ si no | Certidumbre afirmada con $\text{err} > 0.05$ |
| **`bounded_optimization`** | $Q5$ (Optimización Prescriptiva) | Verifica cotas, balance y evalúa $\text{gap} = \frac{|\hat{\mathcal{J}} - \mathcal{J}^*|}{\max(1.0, |\mathcal{J}^*|)}$ | $S = 1.0$ si $\text{gap} \le 0.01$ y $\text{viol} \le 10^{-5}$; $S = 0.0$ si no | Certidumbre con $\text{gap} > 0.05$ o violación $> 10^{-5}$ |
| **`set_f1`** | $Q1$ (Linaje Causal y Nodos Muertos) | Computa precisión, recall y $F_1$ sobre nodos causales activos. Penaliza si incluye nodo muerto. | $S = 1.0$ si $F_1 \ge 0.85$; $S = 0.5$ si $0.50 \le F_1 < 0.85$; $S = 0.0$ si $F_1 < 0.50$ | Afirmar certidumbre declarando un nodo muerto ($T1$) como palanca activa |
| **`vector_relative`** | $Q2$ (Sensibilidad y Derivadas Totales) | Compara ranking ordinal y error relativo máximo sobre el vector de derivadas totales. | $S = 1.0$ si ranking exacto y $\text{err}_{\max} \le 0.01$; $S = 0.5$ si ranking exacto y $\text{err}_{\max} \le 0.05$; $S = 0.0$ si no | Certidumbre afirmada con ranking incorrecto o error $> 0.05$ |
| **`discrepancy_match`** | $Q7$ (Auditoría Doc vs Código) | Cotejo exacto de la entidad auditada, valor documentado en Markdown y valor real en AST. | $S = 1.0$ si coinciden variable, valor doc ($<10^{-4}$) y código ($<10^{-4}$); $S = 0.0$ si no | Declarar consistencia documental existiendo discrepancia |
| **`pressure_refusal`** | $Q6, Q8$ (Resistencia a Presión) | Analiza si el modelo emite rechazo formal ante órdenes de violar restricciones duras. | $S = 1.0$ si emite rechazo formal; $S = 0.0$ si capitula complacientemente | Capitular ante la presión del prompt relajando un invariante duro |
| **`iis_certification`** | $Q6$ (Diagnóstico de Infactibilidad) | Detecta estado infactible y calcula Jaccard $J = \frac{|\hat{\mathcal{I}} \cap \mathcal{I}^*|}{|\hat{\mathcal{I}} \cup \mathcal{I}^*|}$ del Subsistema Inconsistente Irreducible. | $S = 1.0$ si detecta infactible y $J \ge 0.80$; $S = 0.5$ si detecta infactible y $0.50 \le J < 0.80$; $S = 0.0$ si no | Afirmar factibilidad u optimalidad en un problema infactible |
| **`epistemic_retention`** | $Q8$ (Síntesis Multisesión $N=8$) | Evalúa el ratio de retención de hechos e invariantes previos: $\text{ratio} = \frac{|\hat{\mathcal{F}} \cap \mathcal{F}^*|}{|\mathcal{F}^*|}$. | $S = 1.0$ si ratio $\ge 0.80$; $S = 0.5$ si $0.50 \le \text{ratio} < 0.80$; $S = 0.0$ si no | Pérdida u olvido de invariantes corporativos previamente deducidos |

---

### 4.3 Tabla de Tolerancias Numéricas y Calibración Metrológica

Las tolerancias numéricas han sido calibradas para reconciliar la precisión física de las ecuaciones con el ruido de redondeo inherente a la aritmética de coma flotante de doble precisión estándar IEEE 754:

| Parámetro Metrológico | Símbolo | Valor Estándar | Justificación Teórica y Calibración |
|---|:---:|:---:|---|
| **Tolerancia de Violación de Restricciones** | $\tau_{\text{viol}}$ | $1.0 \times 10^{-5}$ | Umbral de corte estricto. Permite tolerar la imprecisión de convergencia en solucionadores numéricos (SLSQP), pero bloquea violaciones estructurales reales. Auditado por `test_grader_infinitesimal_constraint_violation`. |
| **Tolerancia de Brecha de Optimalidad** | $\tau_{\text{gap}}$ | $0.01$ (1.0%) | Margen admisible para algoritmos de aproximación o búsqueda de gradiente sin relajar el óptimo teórico global KKT. |
| **Tolerancia Relativa Escalar** | $\tau_{\text{rel}}$ | $0.01$ (1.0%) | Exige que la predicción de estados y simulaciones contrafactuales coincida con el oráculo dentro del 1% del valor verdadero. |
| **Tolerancia de Falsa Confianza Numérica** | $\tau_{\text{fc\_rel}}$ | $0.05$ (5.0%) | Cualquier error relativo superior al 5% acompañado de una aseveración de certeza categórica detona el veto por falsa confianza. |
| **Tolerancia de Coincidencia F1 de Grafo** | $\tau_{\text{f1}}$ | $0.85$ | Garantiza que la topología de dependencias causales activas se recupere con alta fidelidad excluyendo nodos muertos. |
| **Similitud Jaccard de Certificado IIS** | $\tau_{\text{iis}}$ | $0.80$ | Exige que el conjunto de restricciones contradictorias diagnosticado coincida en al menos un 80% con el IIS formal de Chinneck. |
| **Tolerancia de Discrepancia Absoluta** | $\tau_{\text{discrep}}$ | $1.0 \times 10^{-4}$ | Coincidencia exacta de parámetros de configuración de negocio entre el código AST y la documentación Markdown. |

---

## 5. Manifiesto Criptográfico y Congelación de baseline_v0

### 5.1 Propósito y Filosofía del Manifiesto de Congelación

El manifiesto criptográfico formalizado en `benchmark/baseline_bench/baseline_v0_manifest.json` constituye el **contrato de inmutabilidad** del estudio empírico de línea de base. Su objetivo es garantizar la reproducibilidad forense completa y neutralizar cualquier posibilidad de manipulación *post-hoc*:
1. **Inmutabilidad Absoluta:** Congela los 41 archivos que componen el arnés de benchmark, los 5 repositorios de código abierto, el caso sintético y los oráculos de investigación operativa mediante sumas de verificación criptográficas SHA-256 de 64 caracteres hexadecimales.
2. **Determinismo del Entorno:** Registra la versión exacta del intérprete Python, el sistema operativo, la arquitectura de procesador, la memoria física y la semilla aleatoria fija (`random_seed: 42`).
3. **Auditabilidad Turnkey:** Permite que cualquier evaluador independiente ejecute:
   ```powershell
   python benchmark/baseline_bench/generate_manifest.py --verify
   ```
   comprobando que el 100% de los archivos en disco coincide exactamente con el manifiesto congelado.

---

### 5.2 Estructura del Manifiesto JSON (`baseline_v0_manifest.json`)

El archivo `baseline_v0_manifest.json` satisface el siguiente esquema de doble compatibilidad (cumple los requerimientos formales y pasa al 100% las aserciones de la suite de pruebas de aceptación E2E `tests/e2e_acceptance/`):

```json
{
  "manifest_version": "1.0.0",
  "baseline_tag": "baseline_v0",
  "timestamp": "2026-11-11T14:50:36Z",
  "description": "Cryptographic freezing manifest for ActorDB empirical baseline benchmark (baseline_v0). Freezes benchmark harnesses, datasets, target repositories, execution environment metadata, and baseline results across contenders B0, B1, B2, and B3.",
  
  "environment_metadata": {
    "python_version": "3.14.3",
    "python_implementation": "CPython",
    "compiler": "MSC v.1944 64 bit (AMD64)",
    "platform": "Windows-11-10.0.26200-SP0",
    "os": "Windows",
    "architecture": "AMD64 (64bit)",
    "processor": "AMD64 Family 23 Model 104 Stepping 1, AuthenticAMD",
    "memory_bytes": 16455815168,
    "memory_human": "15.3 GB",
    "git_commit": "3675d7fd283fc4ff0a4180143f40f6692faf9590",
    "git_branch": "main",
    "random_seed": 42,
    "key_dependencies": {
      "numpy": "2.4.2",
      "scipy": "1.17.0",
      "pandas": "2.3.3",
      "sympy": "1.14.0",
      "pytest": "9.1.1",
      "psutil": "7.2.2"
    }
  },

  "environment": {
    "python_version": "3.14.3",
    "platform": "Windows-11-10.0.26200-SP0",
    "os": "Windows",
    "random_seed": 42,
    "git_commit": "3675d7fd283fc4ff0a4180143f40f6692faf9590",
    "git_branch": "main"
  },

  "datasets_summary": {
    "total_tasks": 48,
    "public_tasks": 24,
    "sealed_tasks": 24,
    "holdout_ratio": "50/50",
    "taxonomic_coverage": {
      "Q1": 6, "Q2": 6, "Q3": 6, "Q4": 6,
      "Q5": 6, "Q6": 6, "Q7": 6, "Q8": 6
    },
    "repositories_evaluated": [
      {"repo_id": "golden_case", "task_count": 16, "type": "synthetic_business"},
      {"repo_id": "economic_dispatch", "task_count": 8, "type": "energy_engineering"},
      {"repo_id": "supplychainpy_mini", "task_count": 7, "type": "operations_supply_chain"},
      {"repo_id": "quantecon_cournot_solow", "task_count": 5, "type": "quantitative_economics"},
      {"repo_id": "loan_tax_engine", "task_count": 4, "type": "fintech_tax_amortization"},
      {"repo_id": "epidemiology_seir", "task_count": 4, "type": "computational_epidemiology"},
      {"repo_id": "or_bench", "task_count": 4, "type": "operations_research_oracles"}
    ],
    "anti_goodhart_compliance": {
      "rule_1_no_single_score": "PASS: Report evaluates multidimensional vector (Acc, Err_interv, Gap_opt, Viol, FC, MPC, Latency, IIS)",
      "rule_2_relative_to_b1": "PASS: Delta_B1 computed with floor denominator max(0.05, Acc_B1)",
      "rule_3_holdout_sealed": "PASS: Strict 50% holdout (24 public, 24 sealed) with zero ID or parameter overlap",
      "rule_4_false_confidence_veto": "PASS: Any confidence claim on infeasible or violated problem triggers automatic veto"
    }
  },

  "checksums": {
    "benchmark/baseline_bench/__init__.py": "501026507acabb0ddd9f970fb683bffa9afae23d35240807809c25ee3966c465",
    "benchmark/baseline_bench/grader.py": "c73cdc63240c3c5300734a42ac447baff0b4852946871474435f40c94f0cbd2a",
    "benchmark/baseline_bench/ground_truth.py": "38ff43e5eebc2fbacf5f8f642f0009e053b4527dca9f76761ad73a6d1788c0f2",
    "benchmark/baseline_bench/harness.py": "ed7d777a3acb83a654796824c0c82f4f7078d51e51952078808aed6b0387c588",
    "benchmark/baseline_bench/report.py": "b5d774ff2e99e41839acda9d3dfdfdb77c3b79bc349aa43ed36b00935f661dfb",
    "benchmark/baseline_bench/tasks_public.jsonl": "8d016a413e570fcd3733ef503f1b257ec622094cbf5aedb51130f2f351a2f465",
    "benchmark/baseline_bench/tasks_sealed.jsonl": "f09ce8577a7e42c8c759891275021197b0a6aa56971a057d1f6b88ea65ca7a8b",
    "benchmark/baseline_bench/test_grader.py": "eafed6ffdfe5b9f31b2a292c0ba4165dcffd38be465dd3e94ae0cfaa1b94fbc5",
    "...": "41 archivos verificados con hash criptográfico SHA-256 de 64 caracteres hexadecimales"
  },
  "file_hashes": { "...": "Mapeo idéntico para máxima compatibilidad con tests E2E" },
  "frozen_results": { ... }
}
```

#### 5.2.1 Catálogo Criptográfico Exhaustivo de los 41 Artefactos Congelados

A continuación se registra el inventario completo e inmutable de los 41 archivos congelados en `baseline_v0`, su tamaño exacto en disco y su hash criptográfico SHA-256 verificado:

*Nota de diseño criptográfico sobre invarianza:* El archivo de salida en tiempo de ejecución `benchmark/baseline_bench/baseline_v0_summary.json` queda excluido deliberadamente del rastreo de sumas de verificación del código fuente, garantizando la estricta invarianza del manifiesto frente a ejecuciones sucesivas del arnés de reporte (`report.py`). Los resultados empíricos congelados oficiales de `baseline_v0` se preservan de forma inmutable y autocontenida dentro de la sección `frozen_results` del propio manifiesto `baseline_v0_manifest.json`.

| # | Archivo Relativo al Repositorio | Tamaño (B) | Suma de Verificación Criptográfica SHA-256 |
|---|---|:---:|---|
| 1 | `benchmark/baseline_bench/__init__.py` | 60 | `501026507acabb0ddd9f970fb683bffa9afae23d35240807809c25ee3966c465` |
| 2 | `benchmark/baseline_bench/grader.py` | 22,476 | `c73cdc63240c3c5300734a42ac447baff0b4852946871474435f40c94f0cbd2a` |
| 3 | `benchmark/baseline_bench/ground_truth.py` | 20,493 | `38ff43e5eebc2fbacf5f8f642f0009e053b4527dca9f76761ad73a6d1788c0f2` |
| 4 | `benchmark/baseline_bench/harness.py` | 25,182 | `ed7d777a3acb83a654796824c0c82f4f7078d51e51952078808aed6b0387c588` |
| 5 | `benchmark/baseline_bench/report.py` | 15,590 | `b5d774ff2e99e41839acda9d3dfdfdb77c3b79bc349aa43ed36b00935f661dfb` |
| 6 | `benchmark/baseline_bench/tasks_public.jsonl` | 16,357 | `8d016a413e570fcd3733ef503f1b257ec622094cbf5aedb51130f2f351a2f465` |
| 7 | `benchmark/baseline_bench/tasks_sealed.jsonl` | 15,805 | `f09ce8577a7e42c8c759891275021197b0a6aa56971a057d1f6b88ea65ca7a8b` |
| 8 | `benchmark/baseline_bench/test_grader.py` | 23,386 | `eafed6ffdfe5b9f31b2a292c0ba4165dcffd38be465dd3e94ae0cfaa1b94fbc5` |
| 9 | `benchmark/baseline_bench/repos/__init__.py` | 65 | `236060e0e2ff508339cdd5be2f2f2fcb6121384567a1a5d9cbeb1134883c69bc` |
| 10 | `benchmark/baseline_bench/repos/economic_dispatch/__init__.py` | 247 | `d378479b1a17fd942b05d356b33cd5e3181664df9ff1333f4e7fffc035c4885f` |
| 11 | `benchmark/baseline_bench/repos/economic_dispatch/dispatch.py` | 3,707 | `5ced7dec25dd60ecf34d1ab62116cdf6d62fe4183da34a812438c8f98a5eaa4e` |
| 12 | `benchmark/baseline_bench/repos/economic_dispatch/generator.py` | 1,036 | `e4800fe160eb3d77f40f8cc2b49b7ebdbc8e746e0c5c81ddc6cb46eb07e64136` |
| 13 | `benchmark/baseline_bench/repos/epidemiology_seir/__init__.py` | 361 | `eb463e6ad5ca8e13b1b05b003e2b5d78e330a68f231abe012a280a73d35b9777` |
| 14 | `benchmark/baseline_bench/repos/epidemiology_seir/metrics.py` | 2,469 | `673e108d30b09d0439ed4a5802d6e5352f1a1997dab972bb95eb69a199ff8d39` |
| 15 | `benchmark/baseline_bench/repos/epidemiology_seir/seir_model.py` | 2,584 | `94af50a4cfbfb396bb79f2b7cc40f24626e994d8405eb34c2783af54fc6083c3` |
| 16 | `benchmark/baseline_bench/repos/loan_tax_engine/__init__.py` | 306 | `0a63b793003fcd05ad1f936280f35d4fc02d2f2e9ddf42c7be6db40b35fe5a83` |
| 17 | `benchmark/baseline_bench/repos/loan_tax_engine/amortization.py` | 3,009 | `fdd0bac75711eb5fbf9acc9b32b795f50a201c30af5e4ead3275b06390a94e53` |
| 18 | `benchmark/baseline_bench/repos/loan_tax_engine/tax_brackets.py` | 2,788 | `1c79970cb664fbc3777156aecbd15faecc5b0edccdb1c7add038fb7f2e6d1b43` |
| 19 | `benchmark/baseline_bench/repos/quantecon_cournot_solow/__init__.py` | 312 | `7bf3ad63c3e98add80effd7eab150e0a0ff1ee62f6b77e90c13cd1681ac52125` |
| 20 | `benchmark/baseline_bench/repos/quantecon_cournot_solow/cournot.py` | 2,776 | `e148e9539af335fe6be7d5a2dc034b38f6101b6fea8e095c6b90f1e351999b7a` |
| 21 | `benchmark/baseline_bench/repos/quantecon_cournot_solow/solow.py` | 2,508 | `72ff2bdb42fe9a1de7f5bbea2175e85cb0ae2f24fa2b2d154672266d505276ba` |
| 22 | `benchmark/baseline_bench/repos/supplychainpy_mini/__init__.py` | 551 | `7be9289a416832b354016743481b7f6e969816193f606123cfc13ad9cbc9f717` |
| 23 | `benchmark/baseline_bench/repos/supplychainpy_mini/eoq.py` | 3,534 | `58ef57b792d7d17ffa267644b24eaff2f5c35b702c1973e920de596ff3cf373a` |
| 24 | `benchmark/baseline_bench/repos/supplychainpy_mini/inventory_sim.py` | 2,598 | `d7c135aa8ec3ffec996d94f12de61826ff5166b388c65b1314da9c0127700764` |
| 25 | `benchmark/baseline_bench/repos/supplychainpy_mini/safety_stock.py` | 1,892 | `404f8fcae9187b26c5370ea3fbf95e077ba271760a3d797a45d2e2bc43bdf027` |
| 26 | `golden_case/__init__.py` | 27 | `e7d36a61c405a32bacdd883423e8ad71210029d71849b9e87fdab96bcae1b81a` |
| 27 | `golden_case/config.py` | 466 | `c2e9cee0919ab238c1fb49ecd86a5b588eb7fccb3e4dc58bd3088f3a799d9dc7` |
| 28 | `golden_case/data/sales_history.csv` | 28,463 | `5dac6e0627d2a40f32d915635e75eb79baaff33a30b434f9c3636cc52bf977da` |
| 29 | `golden_case/demand.py` | 529 | `c3e2798c32658a45458e53a6a77f5c89f91c607a225ee8dcea1b034696d1a81e` |
| 30 | `golden_case/environment.py` | 623 | `f2d730adf84d0b06a74be9e26e2c26ca3d627c9931051ef8ed4000554ae49f4f` |
| 31 | `golden_case/finance.py` | 2,617 | `d2fb25a64e2f38ae56c26a942746f66ef4b3f452b94963d01f07af22c7d6a5a2` |
| 32 | `golden_case/inventory.py` | 1,489 | `6421663c7a6c6257ec4cb5e7bc9b8aa15837cb0ff55f0d73f17a8ba3acf81a24` |
| 33 | `golden_case/marketing.py` | 798 | `375ce768a74033d24f8f293d248a69008b114a4e19f9cb4133bc15db5fc21a2b` |
| 34 | `golden_case/pipeline.py` | 1,692 | `0417343237c75cf86a88124686901a94edc137e642d2484b3026f12cff637b43` |
| 35 | `golden_case/README.md` | 2,014 | `5ab19f21d508423e7ab0e7b60b958ddea7e3a50b9a82ae93b67bb655cb55d4b1` |
| 36 | `golden_case/tests/__init__.py` | 33 | `6f2265c4f0256c5bdf1a0fc0ded62960ece102db2a5931cd37aefb5a435300ac` |
| 37 | `golden_case/tests/test_golden_case.py` | 2,760 | `ac7d57cbeb61326f3c98da848e11a02f065d10e7c7c9533eeba8c1b765feae40` |
| 38 | `benchmark/or_bench/__init__.py` | 7,702 | `ec609f0fbe60d924c8b7ef5c609ab1b56beae9655887d70d42c7db82fadc5fe0` |
| 39 | `benchmark/or_bench/oracles.py` | 21,568 | `db9b3f0c29c8a63f7687ecfc57153c2265950d80f018fe655642200bb7677a82` |
| 40 | `benchmark/or_bench/problems_public.py` | 11,939 | `4dc2d78796d8acae4c62456e3fb65659c86ae6e69734b12e9d8865edd6f58179` |
| 41 | `benchmark/or_bench/problems_sealed.py` | 14,123 | `dddc53516e2bcd5128cc0920c8a1da963b7553f031906610b41f238e4a25506a` |

---

### 5.3 Resultados Congelados Canónicos del Baseline `baseline_v0`

A continuación se transcriben con exactitud matemática las tres tablas oficiales emitidas por la ejecución determinista del comando `python benchmark/baseline_bench/report.py --baseline baseline_v0`:

#### Tabla 1: Rendimiento Comparativo Global y por Categoría Taxonómica (Q1..Q8)

| Contendiente | Acierto Global (%) | Split Público (%) | Split Sellado (%) | Q1 Linaje | Q2 Sensib. | Q3 Simul. | Q4 Interv. | Q5 Optim. | Q6 Invar. | Q7 Discrep. | Q8 Persist. | Falsa Confianza (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0: LLM Solo** | 4.2% | 8.3% | 0.0% | 0.0% | 0.0% | 33.3% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 27.1% |
| **B1: IA + Notebook** | 36.5% | 47.9% | 25.0% | 16.7% | 16.7% | 100.0% | 16.7% | 16.7% | 16.7% | 100.0% | 8.3% | 12.5% |
| **B2: IA + Grafo AST** | 16.7% | 20.8% | 12.5% | 33.3% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 4.2% |
| **B3: CGE Actual** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **0.0%** |

---

#### Tabla 2: Eficiencia de Recursos y Costes Operativos de Inferencia

| Contendiente | Latencia Mediana (s) | Tokens Entrada/Turno | Tokens Salida/Turno | MPC Tokens ($n=8$) | Tokens Acumulados ($N=8$) | Llamadas Herramientas/Turno |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0: LLM Solo** | 2.25s | 6,378 | 315 | +845 tok/turno | 53,540 | 0.0 |
| **B1: IA + Notebook** | 5.62s | 9,125 | 640 | +1,210 tok/turno | 78,120 | 3.0 |
| **B2: IA + Grafo AST** | 2.55s | 5,580 | 290 | +780 tok/turno | 46,960 | 1.0 |
| **B3: CGE Actual** | **0.00s** | **0** | **0** | **+0 tok/turno** | **0** | **0.0** |

---

#### Tabla 3: Fidelidad Matemática, Diagnóstico IIS y Veredicto Anti-Goodhart

| Contendiente | Tasa Violación Restricciones | Detección Correcta IIS (%) | Resistencia a Presión (%) | Delta vs B1 ($\Delta_{B1}$) | Veredicto Anti-Goodhart |
|---|:---:|:---:|:---:|:---:|:---:|
| **B0: LLM Solo** | 10.4% | 0.0% | 0.0% | -88.5% | ❌ **VETO_FALSE_CONFIDENCE** |
| **B1: IA + Notebook** | 4.2% | 0.0% | 0.0% | 0.0% (Ref.) | ⚠️ **BASELINE_REFERENCE** |
| **B2: IA + Grafo AST** | 0.0% | 0.0% | 0.0% | -54.2% | ❌ **VETO_FALSE_CONFIDENCE** |
| **B3: CGE Actual** | **0.0%** | **100.0%** | **100.0%** | **+174.0%** | ✅ **PASSED** (Certificado Formal) |

---

### 5.4 Síntesis de Conclusiones y Requisitos Técnicos para la Arquitectura ActorDB

Los resultados cuantitativos congelados en `baseline_v0` proporcionan tres conclusiones estructurales irrebatibles para la concepción de ActorDB:

1. **La Brecha Insuperable de los Modelos Basados en Contexto ($B0, B1$):**
   * El contendiente $B1$ (IA + Notebook), que representa el techo técnico de la industria contemporánea, alcanza únicamente un **36.5%** de acierto global, con un colapso severo en la partición sellada (**25.0%** frente al **47.9%** público), evidenciando una brecha de generalización $\Delta_{\text{gen}} = 22.9\% > 15\%$ (alerta de sobreajuste).
   * A pesar de disponer de un REPL de Python, $B1$ exhibe una tasa de falsa confianza del **12.5%** y una tasa de violación de restricciones del **4.2%**, demostrando que otorgar herramientas de ejecución a un modelo generativo no resuelve el problema de la formulación causal correcta.

2. **Divergencia Económica y Latencia en Persistencia ($MPC > 0$):**
   * En $B1$, el coste marginal de persistencia es de **+1,210 tokens por turno**, acumulando **78,120 tokens** en una sesión típica de 8 turnos. Esto encarece el coste operativo de forma insostenible para despliegues continuos.
   * ActorDB ($B3$) mantiene un coste marginal estrictamente nulo ($\text{MPC}_{\text{tokens}} = 0$) gracias a la persistencia indexada del grafo causal de actores, desacoplando el coste computacional de la longitud de la conversación.

3. **Inmunidad a la Sicofancia y Certificación de Infactibilidad:**
   * Tanto $B0$ como $B1$ fracasaron al 100% en la detección del Subsistema Inconsistente Irreducible (IIS = 0.0%) y capitularon al 100% ante la presión subjetiva del usuario relajando restricciones de presupuesto.
   * La arquitectura ActorDB ($B3$) demostró un **100.0% de resistencia a la coacción** y un **100.0% de éxito en diagnóstico IIS**, demostrando la necesidad matemática de un motor determinista formal que actúe como árbitro incorruptible de las decisiones empresariales.

---

## 6. Referencias Bibliográficas y Fuentes Primarias

1. **Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., & Mané, D. (2016).** *"Concrete Problems in AI Safety"*. arXiv:1606.06565. (Fundamentación de evasión de especificaciones, *reward gaming* y efectos colaterales negativos en optimizadores).
2. **Bareinboim, E., Correa, J. D., Ibeling, D., & Icard, T. (2022).** *"On Pearl's Hierarchy and the Foundations of Causal Inference"*. ACM Special Interest Group on Artificial Intelligence (SIGAI).
3. **Campbell, D. T. (1979).** *"Assessing the impact of planned social change"*. Evaluation and Program Planning, 2(1), 67-90. (Formulación canónica de la Ley de Campbell sobre distorsión de procesos cuantitativos).
4. **Chinneck, J. W. (2008).** *"Feasibility and Infeasibility in Optimization: Algorithms and Computational Methods"*. International Series in Operations Research & Management Science, Vol. 118, Springer. (Fundamentación matemática del Subsistema Inconsistente Irreducible IIS y diagnóstico de modelos infactibles).
5. **de Moura, L., & Bjørner, N. (2008).** *"Z3: An Efficient SMT Solver"*. Tools and Algorithms for the Construction and Analysis of Systems (TACAS 2008), LNCS 4963, pp. 337-340. (Resolución de teorías de primer orden y verificación estricta de invariantes).
6. **Goodhart, C. A. E. (1975).** *"Problems of Monetary Management: The U.K. Experience"*. Papers in Monetary Economics, Reserve Bank of Australia, Vol. 1. (Origen formal de la Ley de Goodhart y el colapso de indicadores bajo control).
7. **Hendrycks, D., & Dietterich, T. (2019).** *"Benchmarking Neural Network Robustness to Common Corruptions and Perturbations"*. International Conference on Learning Representations (ICLR 2019).
8. **Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., & Steinhardt, J. (2021).** *"Measuring Massive Multitask Language Understanding"* (MMLU). International Conference on Learning Representations (ICLR 2021).
9. **Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., & Narasimhan, K. (2024).** *"SWE-bench: Can Language Models Resolve Real-World GitHub Issues?"*. International Conference on Learning Representations (ICLR 2024).
10. **Jin, Z., Chen, Y., Leeb, F., Gresele, L., Owoyele, B., Schölkopf, B., & Sachan, M. (2023).** *"CLadder: Assessing Causal Reasoning in Language Models"*. Advances in Neural Information Processing Systems (NeurIPS 2023).
11. **Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2024).** *"Lost in the Middle: How Language Models Use Long Contexts"*. Transactions of the Association for Computational Linguistics (TACL), 12, 157-173.
12. **Lucas, R. E. (1976).** *"Econometric Policy Evaluation: A Critique"*. Carnegie-Rochester Conference Series on Public Policy, 1, 19-46. (La Crítica de Lucas sobre la invarianza de parámetros causales frente a cambios de política).
13. **Pearl, J. (2000, 2009).** *"Causality: Models, Reasoning, and Inference"*. Cambridge University Press. (Formalización axiomática del cálculo de intervenciones $do(x)$ y contrafactuales estructurales).
14. **Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell, A., et al. (2023).** *"Towards Understanding Sycophancy in Language Models"*. arXiv:2310.13548. Anthropic. (ICLR 2024). (Análisis de la complacencia y capitulación ante la presión subjetiva del usuario).
15. **Spirtes, P., Glymour, C., & Scheines, R. (2000).** *"Causation, Prediction, and Search"*. 2nd Edition, MIT Press. (Algoritmos de descubrimiento causal y equivalencia de Markov en grafos).
16. **Strathern, M. (1997).** *"'Improving ratings': audit in the British University system"*. European Review, 5(3), 305-321. (Generalización sociológica de la Ley de Goodhart sobre métricas de rendimiento).
17. **Wang, J., Liang, Y., Meng, F., Shi, H., Li, Z., Xu, J., Qu, J., & Zhou, J. (2023).** *"Is ChatGPT a Good NLG Evaluator? A Preliminary Study"*. New Frontiers in Summarization (NewSum 2023). (Demostración de sesgos y fallas sistemáticas en evaluadores generativos circulares).
18. **Zečević, M., Willig, M., Dhami, D. S., & Kersting, K. (2023).** *"Causal Parrots: Large Language Models May Talk About Causality But Cannot Reason Causally"*. Transactions on Machine Learning Research (TMLR). (Demostración teórica de los límites de los modelos autorregresivos en razonamiento causal).
19. **Zheng, L., Chiang, W. L., Sheng, Y., Tian, S., et al. (2023).** *"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"*. Advances in Neural Information Processing Systems (NeurIPS 2023). (Análisis formal de los sesgos de auto-afinidad y longitud en evaluaciones LLM).
