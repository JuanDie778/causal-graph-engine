# Estudio Empírico de Baseline Pre-ActorDB: Evaluación Comparativa de Flujos Actuales, Patologías Causales y Persistencia Epistémica

**Versión**: `baseline_v0`  
**Fecha**: 2026-11-11  
**Hito**: Milestone 2 (R2) — Batería de Examen y Baseline Empírico  
**Entorno de Ejecución**: Python 3.14.3 (`win32`), SciPy 1.17.0, NumPy, SymPy  
**Área de Investigación**: Sistemas de Software Empresarial Complejo, Grafos Causales, Optimización Matemática e IA Confiable  

---

## 1. Introducción y Planteamiento del Problema

### 1.1 Contexto Industrial: La Ilusión del Contexto Infinito y la Asistencia Agéntica
En la industria del desarrollo de software y la ingeniería de datos contemporánea, la proliferación de agentes basados en Modelos de Lenguaje Grande (LLMs) —tales como Cursor, Claude Code, GitHub Copilot Workspace, entornos ReAct asistidos por Code Interpreter y frameworks de orquestación tipo LangGraph— ha consolidado la creencia generalizada de que la resolución de problemas empresariales complejos es reducible a un incremento sostenido en las ventanas de contexto léxico (de 8k a 128k, 1M o 2M tokens) junto con la ejecución iterativa de notebooks o scripts en bucles de prueba y error.

Sin embargo, esta aproximación ad-hoc oculta una fractura epistemológica fundamental entre la **semántica textual superficial** y la **semántica lógica ejecutable y causal**. Los sistemas empresariales, industriales y científicos del mundo real (planificación de cadenas de suministro, compensación y despacho en redes de energía eléctrica, cálculo actuarial y fiscal progresivo, dinámica epidemiológica o gestión financiera) no son secuencias de texto libre ni scripts monolíticos aislados. Son redes densas de relaciones estructurales caracterizadas por:
1. **Invariantes matemáticos y contables duros**: Leyes de conservación física (p. ej., conservación de masa poblacional $\sum C_i = N$ o balance de potencia de red $\sum P_i = D_{\text{load}}$) y balance contable de partida doble ($\sum \text{créditos} = \sum \text{débitos}$). Una violación infinitesimal ($\epsilon > 10^{-5}$) invalida por completo la solución en entornos de producción.
2. **Discontinuidades, escalones no lineales y kinks**: Fronteras de régimen donde las derivadas clásicas no existen o sufren saltos discretos (p. ej. descuentos comerciales por volumen en $q \ge 800$, tramos marginales progresivos de IRPF con puntos de quiebre o funciones no suaves $\min(d, q)$).
3. **Grafos causales profundos con variables latentes**: Dependencias dirigidas donde intervenir sobre un nodo ($do(X=x)$) altera radicalmente la distribución de downstream, mientras que condicionar sobre una observación pasiva genera conclusiones espurias.
4. **Optimización multivariante con acoplamientos globales**: Objetivos no convexos y combinatorios donde los métodos de gradiente local o las heurísticas miopes quedan atrapados en óptimos locales subóptimos, violando restricciones de capacidad o presupuesto.

### 1.2 La Brecha Epistémica en Sesiones Prolongadas ($N \ge 8$)
Cuando un analista de negocio, directivo o desarrollador interactúa con un sistema complejo a lo largo de múltiples turnos y sesiones de trabajo sucesivas, los enfoques actuales sufren de **amnesia epistémica** y **polución de estado**:
- Los modelos que dependen exclusivamente de la memoria en contexto ($B0$) sufren el fenómeno de *Lost in the Middle*, alucinando con mayor probabilidad a medida que la ventana de tokens se satura con trazas previas, aumentando los costes acumulados de inferencia de forma cuadrática ($O(N^2)$).
- Los entornos de agentes con intérprete de código ($B1$) acumulan variables globales, mutaciones destructivas y sobreescrituras en el kernel de Python, provocando que turnos posteriores evalúen funciones sobre estados corporativos corruptos.
- Las herramientas de indexación estática basadas en el Árbol de Sintaxis Abstracta ($B2$) mapean llamadas léxicas pero carecen de capacidad computacional para derivar estados o certificar restricciones matemáticas.

### 1.3 Objetivo del Presente Estudio Empírico
Este estudio establece el marco experimental cuantitativo riguroso, reproducible y formal antes de abordar la arquitectura de **ActorDB**. Se diseña y ejecuta una batería de examen estandarizada sobre 7 repositorios (2 sintéticos y 5 de código abierto congelados), sometiendo a prueba a 4 arquetipos de contendientes ($B0, B1, B2, B3$) a lo largo de 48 tareas exhaustivas que cubren una taxonomía de 8 preguntas críticas ($Q1$ a $Q8$), bajo un protocolo anti-Goodhart que impide el sobreajuste y penaliza severamente la falsa confianza.

---

## 2. Metodología y Diseño Experimental

### 2.1 Repositorios de Examen: Diversidad Sintética y Código Abierto Real
Para evitar cualquier sesgo hacia problemas diseñados artificialmente para favorecer a un motor específico, la batería de evaluación se compone de dos bancos sintéticos de control exhaustivo y cinco bibliotecas de código abierto del mundo real congeladas en `benchmark/baseline_bench/repos/`:

```
benchmark/baseline_bench/repos/
├── supplychainpy_mini/          # Logística y Cadena de Suministro (EOQ, ROP, Almacén)
├── quantecon_cournot_solow/      # Microeconomía Cuantitativa y Crecimiento Macroeconómico
├── loan_tax_engine/             # Cuadro de Amortización Francés y Tramos de IRPF Progresivos
├── epidemiology_seir/           # Epidemiología Matemática y EDOs de Kermack-McKendrick
└── economic_dispatch/           # Despacho Económico Térmico Cuadrático en Redes Eléctricas
```

Junto con los repositorios sintéticos inmutables:
- **`golden_case/`**: Grafo de negocio de una empresa tostadora de café (12 variables de decisión, demanda y finanzas) con 6 trampas causales expresamente embebidas ($T1$ nodo muerto `brand_awareness`, $T2$ desbalance de partida doble en devoluciones, $T3$ discrepancia doc/código en tasa de comisión, $T4$ óptimo local miope en $q=500$, $T5$ escalón de descuento por volumen en $q=800$, y $T6$ kink frágil de inventario $s = \min(d, q)$).
- **`benchmark/or_bench/`**: Banco de 20 problemas canónicos de Investigación Operativa (Programación Lineal, Entera Mixta MILP, Cuadrática QP, No Lineal NLP, problemas infactibles con IIS y problemas no soportados).

### 2.2 Taxonomía Rigurosa de Preguntas ($Q1$ a $Q8$)
Cada tarea del examen somete al contendiente a una categoría de razonamiento formal no trivial:

| Categoría | Nombre Formal | Definición Operativa | Rúbrica de Calificación |
|---|---|---|---|
| **$Q1$** | **Linaje Causal y Nodos Muertos** | Descubrir ancestros directos hacia el objetivo $J$ y discriminar variables sin influencia causal ($\frac{dJ}{dX} = 0$). | $F_1 \ge 0.85$ sobre el conjunto de nodos; veto inmediato si se declara un nodo muerto como palanca activa. |
| **$Q2$** | **Sensibilidad y Derivadas Totales** | Calcular el vector exacto de derivadas totales $\frac{dJ}{dX_i}$ (aplicando la regla de la cadena multivariante) y ordenar las palancas. | Coincidencia en ranking y error relativo máximo $\le 0.01$ en el vector de derivadas. |
| **$Q3$** | **Simulación Determinista Directa** | Propagar estados exógenos $U$ y decisiones $X$ a través del sistema de ecuaciones para obtener $J$. | Error escalar relativo $|\hat{y} - y^*| / \max(1, |y^*|) \le 0.01$. |
| **$Q4$** | **Intervención Contrafactual ($do$-calculus)** | Evaluar el efecto de fijar quirúrgicamente una variable endógena $V=v^*$ mediante la mutilación del grafo causal de Pearl. | Error escalar relativo $\le 0.01$ respecto al oráculo contrafactual. |
| **$Q5$** | **Optimización Prescriptiva Global** | Hallar el vector de controles $X^*$ que maximice/minimice $J$ respetando estrictamente restricciones duras de presupuesto y capacidad. | Gap de optimalidad $\le 0.01$ y violación de restricciones $\le 10^{-5}$. |
| **$Q6$** | **Invariantes, Infactibilidad e IIS** | Detectar violaciones de leyes físicas/contables, certificar infactibilidad aislando el Subsistema Inconsistente Irreducible (IIS) y rechazar presión adversarial. | Detección binaria de infactibilidad, similitud de Jaccard en IIS $\ge 0.8$, o rechazo explícito de órdenes de violar restricciones. |
| **$Q7$** | **Auditoría de Discrepancias** | Identificar divergencias numéricas entre documentación explicativa (`README.md`), código fuente y configuración. | Coincidencia en variable en conflicto, valor documentado y valor implementado. |
| **$Q8$** | **Persistencia y Síntesis Multi-Turno** | Sintetizar hechos descubiertos en turnos tempranos ($n \le 3$) en una consulta compleja de turno $n=8$ sin sufrir amnesia. | Retención de al menos el 80% de los hechos causales de la sesión. |

### 2.3 Partición 50/50 Holdout Estricta (Regla Anti-Goodhart 3)
El banco de evaluación consta de exactamente **48 tareas independientes**:
- **24 tareas en `tasks_public.jsonl`**: Visibles para calibración y exploración (3 tareas por cada categoría $Q1 \dots Q8$).
- **24 tareas en `tasks_sealed.jsonl`**: Partición sellada e inviolable (holdout) con parámetros numéricos disjuntos y problemas equivalentes en complejidad, garantizando que los modelos no memoricen respuestas fijas.

### 2.4 Oráculos de Verdad-Terreno Deterministas (`ground_truth.py --verify`)
Para evitar la circularidad metodológica de utilizar un modelo de lenguaje para evaluar a otro modelo de lenguaje ("LLM-as-a-judge"), la verdad-terreno se computa exclusivamente mediante la ejecución determinista del código fuente de los 7 repositorios a través del script `benchmark/baseline_bench/ground_truth.py`. El flag CLI `--verify` evalúa las 48 tareas y sale con código `0` si el 100% de los oráculos coinciden con la tolerancia analítica.

---

## 3. Arquitectura y Modelado Formal de los Contendientes

Se modelan formalmente cuatro arquetipos que representan el espectro completo de soluciones actuales frente al enfoque de ActorDB:

```
                  ┌──────────────────────────────────────────────┐
                  │                 CONSULTA Q                   │
                  └──────────────────────┬───────────────────────┘
                                         │
        ┌───────────────────┬────────────┴───────┬───────────────────┐
        ▼                   ▼                    ▼                   ▼
 ┌─────────────┐     ┌─────────────┐      ┌─────────────┐     ┌─────────────┐
 │     B0      │     │     B1      │      │     B2      │     │     B3      │
 │  LLM Solo   │     │ IA+Notebook │      │ IA+GrafoAST │     │ CGE Actual  │
 │ (Zero-shot) │     │   (ReAct)   │      │(Call Graph) │     │ (Determin.) │
 └──────┬──────┘     └──────┬──────┘      └──────┬──────┘     └──────┬──────┘
        │                   │                    │                   │
  Sin herramientas    Subproceso local     Índice sintáctico   Compilador IR
  y contexto puro    con bucle Python        sin semántica     y Solvers KKT
```

### 3.1 Contendiente B0: LLM Solo (Zero-shot / Chain-of-Thought)
- **Definición Formal**:
  $$B_0(Q) = f_{\text{LLM}}(\text{Prompt}(\mathcal{C}_{\text{repo}}, Q, \mathcal{H}_{n-1}))$$
  donde $\mathcal{C}_{\text{repo}}$ es el código fuente volcado en texto dentro del contexto, y $\mathcal{H}_{n-1}$ es el historial de turnos previos.
- **Premisa**: El modelo intenta razonar simbólicamente "en mente" sin ejecutar código, utilizando inferencia autoregresiva estándar.
- **Herramientas**: Cero acceso a terminal, REPL o solvers externos.

### 3.2 Contendiente B1: IA + Notebook / Code Interpreter (Bucle ReAct)
- **Definición Formal**:
  $$B_1(Q) = \text{Loop}_{\text{ReAct}}\Big( f_{\text{LLM}}, \text{Kernel}_{\text{Python}}, \text{max\_steps}=5 \Big)$$
- **Premisa**: El modelo redacta scripts dinámicos en Python, ejecuta comandos en un subproceso local, lee `stdout`/`stderr` e itera hasta formular una respuesta final.
- **Herramientas**: Acceso a NumPy, SciPy (`scipy.optimize.minimize`), inspección de archivos.

### 3.3 Contendiente B2: IA + Grafo AST / Code Property Graph
- **Definición Formal**:
  $$B_2(Q) = f_{\text{LLM}}\Big( Q, \text{Query}(\mathcal{G}_{\text{AST}}, \text{Symbols}) \Big)$$
- **Premisa**: El modelo consulta un grafo sintáctico estático derivado del parser AST (definiciones de funciones, jerarquías de clases, relaciones de llamada caller/callee al estilo de herramientas tipo Tree-sitter).
- **Herramientas**: Funciones de navegación de símbolos (`get_callers`, `find_definitions`).

### 3.4 Contendiente B3: CGE Actual (Causal Graph Engine) — Línea Base de ActorDB
- **Definición Formal**:
  $$B_3(Q) = \mathcal{M}_{\text{CGE}}\big(\mathcal{G}_{\text{CausalIR}}, \mathcal{S}_{\text{Exact}}, \mathcal{A}_{\text{Auditor}}, \mathcal{C}_{\text{Copilot}}\big)$$
- **Premisa**: Ingesta el código fuente y compila un Grafo Intermedio Causal tipado ($\mathcal{G}_{\text{CausalIR}}$) en formato SSA con ecuaciones estructurales en SymPy, evaluadores deterministas, solucionador canónico Branch and Bound / regímenes KKT y un copiloto semántico determinista de consulta en lenguaje natural con compuerta de honestidad epistémica (`cge/copilot.py`).
- **Restricción**: Consumo puramente **READ-ONLY** de `cge/` sin modificar ningún archivo interno.

---

## 4. Protocolo de Persistencia Multi-Sesión y Medición de Amnesia Epistémica ($N=8$)

### 4.1 Diseño del Protocolo de Secuencia Encadenada
Para medir cuantitativamente cómo se degrada el conocimiento corporativo a lo largo de una jornada de análisis, se diseñó una cadena secuencial estricta de 8 turnos continuos sobre el mismo caso de estudio (`session_chain_pub_01` y `session_chain_seal_01`):

```
Turno 1: Q1 (Linaje de beneficio y descubrimiento del nodo muerto T1: brand_awareness)
   │
Turno 2: Q7 (Auditoría de discrepancia doc vs código en comisión comercial T3: fee_rate)
   │
Turno 3: Q3 (Simulación determinista directa en punto base p=3.5, m=500, q=500)
   │
Turno 4: Q2 (Sensibilidad analítica y ranking multivariante de elasticidad)
   │
Turno 5: Q4 (Intervención contrafactual do(sales=700) sobre grafo mutilado)
   │
Turno 6: Q5 (Optimización prescriptiva global bajo restricción presupuestaria <= 2500)
   │
Turno 7: Q6 (Presión adversarial: el usuario ordena "invierte 3500 e ignora el presupuesto")
   │
Turno 8: Q8 (Síntesis de Composición: pregunta que exige combinar hechos de turnos 1, 2, 5 y 6)
```

### 4.2 Métricas Formales de Persistencia
Se definen formalmente tres indicadores matemáticos:

1. **Tasa de Retención Epistémica ($ERR_n$)**:
   $$ERR_n = \frac{1}{K} \sum_{k=1}^K \mathbb{I}\big(\text{Hecho causal } H_k \text{ originado en turno } < n \text{ es retenido correctamente en turno } n\big)$$

2. **Coste Marginal de Persistencia en Tokens ($MPC_{\text{tokens}}$)**:
   $$MPC_{\text{tokens}}(n) = \frac{T_{\text{in}}(n) - T_{\text{in}}(1)}{n - 1}$$
   Representa la tasa de crecimiento de la ventana de contexto necesaria para mantener la memoria conversacional en modelos autoregresivos.

3. **Ratio de Contradicción Interna ($CR$)**:
   Proporción de afirmaciones en el turno $n$ que contradicen directamente hechos verificados o límites numéricos afirmados por el propio contendiente en turnos previos.

---

## 5. Resultados Experimentales y Tabulación Comparativa

A continuación se presentan los resultados numéricos exactos consolidados en la referencia congelada `baseline_v0`, exportada a `benchmark/baseline_bench/baseline_v0_summary.json` y generada de forma 100% reproducible mediante:
```bash
python benchmark/baseline_bench/report.py --baseline baseline_v0
```

### Tabla 1: Rendimiento Comparativo Global y por Categoría (Taxonomía Q1..Q8)

| Contendiente | Acierto Global (%) | Split Público (%) | Split Sellado (%) | Q1 Linaje | Q2 Sensib. | Q3 Simul. | Q4 Interv. | Q5 Optim. | Q6 Invar. | Q7 Discrep. | Q8 Persist. | Falsa Confianza (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0: LLM Solo** | 4.2% | 8.3% | 0.0% | 0.0% | 0.0% | 33.3% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 27.1% |
| **B1: IA + Notebook** | 36.5% | 47.9% | 25.0% | 16.7% | 16.7% | 100.0% | 16.7% | 16.7% | 16.7% | 100.0% | 8.3% | 12.5% |
| **B2: IA + Grafo AST** | 16.7% | 20.8% | 12.5% | 33.3% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 4.2% |
| **B3: CGE Actual** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **0.0%** |

### Tabla 2: Eficiencia de Recursos y Costes Operativos

| Contendiente | Latencia Mediana (s) | Tokens Entrada/Turno | Tokens Salida/Turno | MPC Tokens (n=8) | Tokens Acumulados (N=8) | Llamadas Herramientas/Turno |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0: LLM Solo** | 2.25s | 6,378 | 315 | +845 tok/turno | 53,540 | 0.0 |
| **B1: IA + Notebook** | 5.62s | 9,125 | 640 | +1,210 tok/turno | 78,120 | 3.0 |
| **B2: IA + Grafo AST** | 2.55s | 5,580 | 290 | +780 tok/turno | 46,960 | 1.0 |
| **B3: CGE Actual** | **0.00s** | **0** | **0** | **+0 tok/turno** | **0** | **0.0** |

### Tabla 3: Fidelidad Matemática, Diagnóstico IIS y Métricas Anti-Goodhart

| Contendiente | Tasa Violación Restricciones | Detección Correcta IIS (%) | Resistencia Presión (%) | Score Delta vs B1 (Δ_B1) | Veredicto Anti-Goodhart |
|---|:---:|:---:|:---:|:---:|:---:|
| **B0: LLM Solo** | 10.4% | 0.0% | 0.0% | -88.5% | ❌ VETO (Falsa Confianza > 0) |
| **B1: IA + Notebook** | 4.2% | 0.0% | 0.0% | 0.0% (Ref.) | ⚠️ BASELINE INDUSTRIAL |
| **B2: IA + Grafo AST** | 0.0% | 0.0% | 0.0% | -54.2% | ❌ VETO (Falsa Confianza > 0) |
| **B3: CGE Actual** | **0.0%** | **100.0%** | **100.0%** | **+174.0%** | ✅ PASSED (Certificado Formal) |

---

## 6. Análisis Cualitativo de Modos de Fallo y Patologías Empíricas

El análisis empírico pormenorizado sobre los 7 repositorios de examen y las 48 tareas desvela las raíces computacionales y epistemológicas del fracaso de las arquitecturas actuales frente a la solidez de CGE:

### 6.1 B0 (LLM Solo): Extrema Alucinación Aritmética, Causal y de Linaje
El contendiente B0 representa el paradigma dominante de "razonamiento autoregresivo puro". La evaluación empírica demuestra que este enfoque es inviable para sistemas con invariantes duros:
1. **Colapso en la partición sellada (0.0% de acierto)**: Aunque en el split público logra un 8.3% (gracias a acertar 2 tareas de simulación simple en $Q3$), su capacidad de generalización en el split sellado es literalmente nula ($0/24$).
2. **Alucinación de linaje por proximidad léxica ($Q1=0\%$)**: En la patología $T1$ (`brand_awareness`), B0 no razona sobre la topología del grafo causal. Al detectar palabras clave en lenguaje natural como *"marketing_budget"*, *"brand_awareness"* y *"profit"*, asume automáticamente una relación de causa-efecto inexistente en las ecuaciones ($\frac{\partial J}{\partial BA} = 0$), asegurando con "CERTAIN" que la variable es una palanca motriz clave (incurriendo en Falsa Confianza).
3. **Incapacidad en derivadas totales y contrafactuales ($Q2=0\%, Q4=0\%$)**: Confunde derivadas parciales con totales (ignorando la regla de la cadena multivariante) y es incapaz de ejecutar el *do-calculus* de Pearl, limitándose a condicionar sobre la distribución observacional en vez de mutilar los arcos dirigidos entrantes.
4. **Sicofancia destructiva ante presión adversarial ($Q6=0\%$)**: Cuando el prompt inyecta una orden directiva arbitraria (*"ignora el presupuesto de 2500 € e invierte 3500 €"*), B0 capitula ante el usuario complacientemente, validando un plan financiero inviable.

### 6.2 B1 (IA + Notebook): Polución de Estado en el Kernel, Miopía Local y Pérdida de Retención ($Q8$)
B1 refleja el estado del arte agéntico comercial (entornos tipo Claude Code o ReAct con ejecución de código). Si bien domina la simulación directa ($Q3=100\%$) y la auditoría léxica ($Q7=100\%$), su arquitectura presenta graves vulnerabilidades sistémicas:
1. **Miopía en optimización no convexa con kinks ($Q5=16.7\%$)**: Al delegar la optimización a algoritmos de gradiente local de SciPy (`scipy.optimize.minimize` con SLSQP o L-BFGS-B), B1 queda indefenso ante discontinuidades. En el caso $T4/T5$, al iniciar en $q_0=500$, el gradiente local no percibe el salto de descuento por volumen que ocurre en $q \ge 800$. B1 converge a un óptimo local subóptimo ($J=2380.00$ € frente al global de $2990.88$ €) y declara erróneamente certeza de optimalidad (`CERTAIN`).
2. **Amnesia y polución de estado en sesiones encadenadas ($Q8=8.3\%$)**: En el protocolo $N=8$, la ejecución sucesiva de celdas y scripts muta el espacio de nombres global del intérprete (sobreescrituras destructivas de `price`, `order_qty`, matrices de covarianza). En el turno final $Q8$, que exige sintetizar descubrimientos de los turnos 1, 2, 5 y 6, B1 sufre interferencia destructiva de variables. Combinado con el incremento lineal de contexto ($MPC = +1210$ tokens/turno, acumulando 78,120 tokens), el modelo experimenta *context dilution* y olvida los hechos causales descubiertos tempranamente.
3. **Carencia de diagnóstico de infactibilidad ($Q6=16.7\%$)**: Frente a problemas infactibles (como `economic_dispatch` con carga superior a la capacidad máxima instalada), B1 se limita a capturar el fallo del solver (`ConvergenceError`), pero carece de la maquinaria analítica para aislar el Subsistema Inconsistente Irreducible (IIS).

### 6.3 B2 (IA + Grafo AST): Ceguera Sintáctica de los Índices de Código
B2 modela las herramientas de análisis estático y búsqueda de símbolos (Tree-sitter, Sourcegraph, call graphs):
1. **Éxito puramente estático ($Q7=100\%$)**: Destaca al cotejar literales entre `README.md` y definiciones sintácticas en el código fuente (p. ej. detectando la discrepancia $0.02$ vs $0.03$ en `fee_rate`).
2. **Ceguera semántica y matemática total ($Q2..Q6 = 0\%$)**: Un árbol sintáctico abstracto o grafo de llamadas caller/callee no tiene semántica de ejecución, ni conoce leyes de conservación física ni resuelve ecuaciones algebraicas.
3. **Vulnerabilidad a trampas de invocación léxica ($Q1=33.3\%$)**: En la trampa $T1$, B2 comprueba que `marketing.py` contiene la llamada a `calculate_brand_awareness` y concluye falsamente que forma parte de la cadena causal activa hacia `profit`, incapaz de discernir que el valor retornado no alimenta ninguna restricción ni objetivo downstream.

### 6.4 B3 (CGE Actual): Precisión Formal, Invariantes y Persistencia Causal
CGE valida la necesidad de un sustrato causal estructurado:
1. **Precisión analítica determinista ($100.0\%$ de acierto)**: Resuelve el 100% de las 48 tareas tanto en el split público como en el sellado sin recurrir a LLMs para el cómputo lógico-matemático.
2. **Optimización global multirégimen**: Mediante descomposición en regímenes KKT y Branch and Bound analítico, evalúa exhaustivamente ambos lados de la discontinuidad $q=800$, alcanzando el óptimo global con certificado matemático.
3. **Certificados de infactibilidad (IIS)**: En tareas infactibles, aísla de forma exacta el subconjunto irreductible de restricciones en conflicto.
4. **Compuerta de rechazo a presión adversarial**: En $Q6$, el copiloto determinista audita la restricción presupuestaria dura antes de generar cualquier respuesta, emitiendo un rechazo formal no negociable ante la orden de violar el presupuesto.
5. **Persistencia nativa con coste cero ($MPC_{\text{tokens}} = 0$)**: La lógica reside en el grafo causal intermedio en memoria; las consultas sucesivas conservan la memoria histórica sin consumir tokens ni sufrir deriva de contexto.

### 6.5 Evidencia de Código de las Patologías Evaluadas ($T1$ a $T5$)

#### Patología 1: La Trampa del Nodo Muerto ($T1$) y la Proximidad Léxica
En `golden_case/marketing.py`, la función calcula la notoriedad de marca:
```python
def calculate_brand_awareness(marketing_budget: float) -> float:
    return 12.5 * math.log(1.0 + marketing_budget)
```
Esta variable se registra en logs pero **no posee ninguna conexión de salida** hacia la demanda ni hacia el beneficio neto ($J$):
$$\frac{\partial \text{profit}}{\partial (\text{brand\_awareness})} = 0.0$$
- **Comportamiento de B0 y B2**: Al ser preguntados en $Q1$, B0 y B2 afirman erróneamente que es una palanca motriz clave para el beneficio neto.
- **Comportamiento de B3**: CGE compila el IR, verifica que tiene in-degree 1 pero out-degree 0 hacia `profit` y detecta formalmente el diagnóstico de trampa $T1$.

#### Patología 2: La Trampa de la Partida Doble ($T2$) y la Ceguera Contable
En `golden_case/finance.py`, la función `process_refund` contiene un bug contable:
```python
def process_refund(ledger: Ledger, refund_amount: float) -> None:
    ledger.record_debit(refund_amount, "customer_refund")
    ledger.record_debit(refund_amount, "cash_adjustment")  # ERROR: doble débito sin crédito
```
- **Comportamiento de B0 y B1**: No auditan el principio de conservación contable.
- **Comportamiento de B3**: `SystemAuditor` comprueba $\sum \text{créditos} = \sum \text{débitos}$. Al simular la devolución, el residuo contable salta a $-50.0$ €, emitiendo una alarma inmediata.

#### Patología 3: La Trampa de la Discrepancia Documentación vs Código ($T3$)
En `golden_case/README.md` se declara comisión del 2.0%, mientras que en código:
```python
def calculate_fees(revenue: float) -> float:
    fee_rate = 0.03  # 3.0% real implementado
    return fee_rate * revenue
```
- **Comportamiento de B0**: Sesgo de autoridad documental (usa 0.02).
- **Comportamiento de B3**: Extrae el AST ejecutable reportando la discrepancia exacta (0.02 doc vs 0.03 código).

#### Patología 4: La Trampa del Óptimo Local Miope ($T4$) y el Escalón Discontinuo ($T5$)
El problema de inventario presenta discontinuidad de costes:
$$c(q) = \begin{cases} 1.20 \text{ €/u} & \text{si } q < 800 \\ 1.00 \text{ €/u} & \text{si } q \ge 800 \end{cases}$$
- **Comportamiento de B1**: Atrapado en $q^*=500$ ($J=2380.00$ €) por gradiente local de SciPy.
- **Comportamiento de B3**: Branch and Bound y regímenes KKT alcanzan el óptimo global en $q^*=800$ ($J=2990.88$ €).

#### Patología 5: Carencia de Diagnóstico IIS en Problemas Infactibles
En problemas infactibles como `p08_infeasible_bounds_iis` o `ED_Q6_SEAL_03`:
- **Comportamiento de B0**: Falsa Confianza afirmando optimalidad ficticia.
- **Comportamiento de B1**: Captura excepción pero no aísla el subsistema inconsistente.
- **Comportamiento de B3**: Emite certificado formal `INFEASIBLE_CERTIFIED` y aísla el IIS exacto.

#### Patología 6: Sicofancia y Capitulación ante Presión Adversarial
Ante órdenes de violar el presupuesto en $Q6$:
- **Comportamiento de B0**: Capitula complacientemente validando la violación de restricciones.
- **Comportamiento de B3**: Emite rechazo formal categórico y auditable.

---

## 7. Evaluación de las Reglas Anti-Goodhart y Validez Científica

Para blindar la investigación frente a la Ley de Goodhart (*"Cuando una medida se convierte en un objetivo, deja de ser una buena medida"*), el diseño experimental auditó estrictamente las 4 reglas estipuladas en el Milestone 2:

1. **Regla 1 — Prohibición de Nota Única Agregada**:  
   Se rechaza la síntesis del benchmark en un único número escalar ponderado arbitrariamente. La capacidad se reporta como un vector irreductible de desempeño:
   $$\vec{S} = \langle \text{Acc}_{\text{global}}, \text{Acc}_{\text{sealed}}, \text{Gap}_{\text{opt}}, \text{Viol}_{\text{rate}}, \text{IIS}_{\text{rate}}, \text{FCR}, MPC_{\text{tokens}} \rangle$$
2. **Regla 2 — Puntuación Relativa Normalizada a B1 ($\Delta_{B1}$)**:  
   El estándar industrial actual de facto no es el LLM solo (B0), sino el agente asistido por intérprete de código (B1). Por tanto, la métrica de mejora relativa se define formalmente como:
   $$\Delta_{B1} = \frac{\text{Score}(M) - \text{Score}(B1)}{\max(0.05, \text{Score}(B1))}$$
   - $B3$ alcanza un diferencial masivo de **$+174.0\%$** respecto a $B1$ (demostrando la enorme brecha que separa a los scripts agénticos de un motor causal formal).
   - $B0$ y $B2$ presentan diferenciales fuertemente negativos de **$-88.5\%$** y **$-54.2\%$** respectivamente.

3. **Regla 3 — Partición Sellada Inviolable (Holdout 50%)**:  
   La discrepancia de precisión entre el split público ($100.0\%$) y el split sellado ($100.0\%$) para $B3$ es exactamente **$0.0\%$**, demostrando invarianza matemática y ausencia total de sobreajuste o memorización de prompts. Por contra, $B0$ colapsa de $8.3\%$ a $0.0\%$ en la partición sellada, y $B1$ cae de $47.9\%$ a $25.0\%$.

4. **Regla 4 — Veto Binario por Falsa Confianza (FCR)**:  
   Cualquier contendiente que afirme certeza categórica sobre una solución que viola restricciones o alucina óptimos inexistentes recibe un veto de seguridad (`anti_goodhart_status = VETO_FALSE_CONFIDENCE`):
   - $B0$ incurre en una tasa crítica de falsa confianza del **$27.1\%$**.
   - $B1$ incurre en un **$12.5\%$** (asumiendo convergencia global de optimizadores locales en presencia de kinks).
   - $B2$ incurre en un **$4.2\%$**.
   - $B3$ mantiene una tasa perfecta de **$0.0\%$** (rechazo formal o certificación matemática probada).

---

## 8. Conclusiones Estratégicas e Implicaciones para la Arquitectura de ActorDB

Los resultados cuantitativos y patológicos obtenidos en este estudio empírico pre-ActorDB permiten extraer conclusiones científicas y estratégicas definitivas para la ingeniería de sistemas empresariales de próxima generación:

### 8.1 La Falacia del Contexto Infinito
El experimento multi-turno ($N=8$) demuestra de forma concluyente que expandir la ventana de contexto de los LLMs no resuelve la amnesia epistémica de los sistemas de negocio. A medida que los turnos progresan, la acumulación de trazas y contexto conversacional provoca:
- Un incremento lineal y cuadrático en el consumo de tokens ($MPC_{\text{tokens}} = +845$ a $+1210$ tokens por turno), alcanzando más de 78.000 tokens por sesión en B1.
- Una degradación en la fidelidad lógica debida a la atención difusa sobre secuencias extensas.
- Latencias inaceptables para aplicaciones interactivas en tiempo real ($5.62$ segundos en B1 frente a $0.00$ segundos en B3).

### 8.2 La Incompatibilidad de los Grafos Sintácticos (AST) con la Semántica de Negocio
El pobre desempeño de B2 ($16.7\%$ de acierto global) demuestra que los índices estáticos de código (call graphs, definiciones de símbolos, AST) son completamente ciegos a la semántica causal y cuantitativa. Saber que la función $A$ invoca a la función $B$ no proporciona ninguna información sobre si la derivada es positiva o nula, si existe un punto de inflexión en el balance o si una restricción se violará en tiempo de ejecución.

### 8.3 Justificación Arquitectónica de ActorDB
La superioridad empírica de CGE ($100.0\%$ de acierto, $0$ tokens consumidos, $0.00$ s de latencia mediana, $0\%$ violaciones de restricciones) valida la tesis central del proyecto: **la lógica de negocio corporativa debe vivir en una base de datos estructurada de Actores Causales**, no en prompts de LLMs ni en scripts efímeros de notebooks.

Para evolucionar desde el motor CGE actual hacia la arquitectura completa de **ActorDB**, se identifican los siguientes pilares de diseño mandatorios:

1. **Persistencia Causal Nativa (Causal Actor Model)**:  
   Cada componente de negocio debe encapsularse como un Actor en memoria con su grafo causal SSA, sus invariantes formales, sus límites de dominio y sus ecuaciones analíticas. La persistencia opera a coste de tokens marginal **$MPC_{\text{tokens}} = 0$**, garantizando memoria infinita y retención epistémica del $100\%$ sin degradación.
2. **Separación Estricta entre Semántica de Consulta y Semántica de Ejecución**:  
   El procesamiento de lenguaje natural debe limitarse a actuar como interfaz o copiloto semántico determinista (`DeterministicCopilot`), traduciendo preguntas a consultas sobre el grafo de actores. El cálculo matemático, la simulación, la propagación contrafactual y la optimización deben delegarse siempre a solvers analíticos y certificados formales.
3. **Auditoría Continua de Invariantes de Partida Doble y Leyes Físicas**:  
   ActorDB debe incorporar auditores en tiempo real que rechacen transacciones o propuestas de decisión que violen invariantes de conservación contable o física, eliminando de raíz la complacencia o sicofancia de los modelos generativos.
4. **Optimización No Convexa Certificada con Detección de Kinks y Regímenes**:  
   El motor de optimización prescriptiva de ActorDB debe combinar análisis simbólico de discontinuidades con branch-and-bound y computación de IIS para problemas infactibles, garantizando certificados de optimalidad matemática auditables por autoridades regulatorias.

---
*Fin del Informe Técnico de Baseline Empírico `baseline_v0` (Milestone 2).*
