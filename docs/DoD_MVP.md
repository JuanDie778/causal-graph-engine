# Definition of Done — MVP del Causal Graph Engine (CGE)

## 0. Filosofía del DoD

El MVP se da por terminado cuando el CGE, **sin ninguna ayuda manual**, recibe un repositorio cuya verdad ya conocemos de antemano (el *Golden Case*) y reconstruye esa verdad: el grafo, los diagnósticos y la solución óptima.

Tres principios:

1. **Sabemos la respuesta antes de preguntar.** El Golden Case se construye a mano, con trampas puestas a propósito. Su solución la calcula un **Oráculo independiente** (fuerza bruta) que no comparte código con el CGE.
2. **Cada intuición del Manifiesto tiene una prueba.** Si una frase del Manifiesto no tiene un criterio verificable aquí, o la frase sobra o falta una prueba.
3. **Que no pueda memorizar la respuesta.** Un generador crea variantes del Golden Case con otros parámetros. Si el CGE solo acierta en el caso original, no cuenta como hecho.

---

## 1. El Golden Case: "Tostadora de Café"

Un negocio pequeño, pero que contiene **todas** las piezas de la tupla $\mathcal{M} = \langle U, X, V, \mathcal{F}, \mathcal{C}, J \rangle$ y **todas** las patologías que el Manifiesto dice detectar.

### 1.1 Estructura del repositorio

```
golden_case/
├── README.md              # Documentación (contiene 1 mentira intencionada)
├── config.py              # Palancas X
├── environment.py         # Exógenas U
├── demand.py              # Mecanismo de demanda
├── inventory.py           # Ventas, stock, descuento por volumen, optimizador LOCAL
├── finance.py             # Ingresos, comisiones, beneficio, libro mayor
├── marketing.py           # Contiene un nodo muerto
├── data/sales_history.csv # 365 días generados con el modelo real + ruido
└── tests/                 # pytest: sirven para la validación por ejecución
```

### 1.2 Modelo verdadero (lo que el CGE debe redescubrir)

| Símbolo | Nodo | Tipo | Definición verdadera |
|---|---|---|---|
| $D_0$ | `base_demand` | **U** | $\mathcal{N}(1000, 100^2)$ |
| $p$ | `price` | **X** | $p \in [2, 6]$ |
| $m$ | `marketing_budget` | **X** | $m \in [0, 2000]$ |
| $q$ | `order_qty` | **X** | $q \in \{0, \dots, 1500\}$ (entero) |
| $d$ | `demand` | V | $d = D_0 \cdot (1 - 0.15(p-3)) + 8\sqrt{m}$ |
| $s$ | `sales` | V | $s = \min(d, q)$ ← **no lineal / discreto** |
| $c$ | `unit_cost` | V | $c = 1.2$ si $q < 800$, si no $1.0$ ← **escalón** |
| $R$ | `revenue` | V | $R = p \cdot s$ |
| $f$ | `fees` | V | $f = 0.03 R$ (el README dice 2%) |
| $h$ | `holding_cost` | V | $h = 0.2(q - s)$ |
| $b$ | `brand_awareness` | V | Se calcula y se registra en logs, **nunca afecta a J** |
| $J$ | `profit` | **target** | $J = R - f - c\,q - m - h$ |

### 1.3 Restricciones $\mathcal{C}$

| ID | Tipo | Expresión |
|---|---|---|
| `c_budget` | hard | $c \cdot q + m \le 2500$ |
| `c_ledger` | hard | $\sum \text{créditos} - \sum \text{débitos} = 0$ |
| `c_stock` | hard | $q - s \ge 0$ |
| `c_price_bounds` | hard | $2 \le p \le 6$ |

### 1.4 Trampas plantadas (conocidas por construcción)

| # | Trampa | Dónde | Qué prueba del Manifiesto |
|---|---|---|---|
| T1 | **Nodo muerto**: `brand_awareness` | `marketing.py` | §3.2.2 — derivada nula $\partial J / \partial b = 0$ |
| T2 | **Violación de invariante**: la rama de reembolsos anota el débito dos veces | `finance.py::process_refund` | §3.2.1 — comprobación de invariantes |
| T3 | **Doc ≠ Código**: el README dice comisión del 2%, el código aplica el 3% | `README.md` vs `finance.py` | §6 Paso 3 — diagnóstico intención vs. comportamiento |
| T4 | **Mínimo local**: `inventory.optimize_order()` minimiza solo el coste de almacenaje y recomienda $q=500$, lo que provoca roturas de stock | `inventory.py` | §3.2.3 — mínimos locales |
| T5 | **Discontinuidad**: el descuento por volumen a partir de $q \ge 800$ | `inventory.py` | Capa 1 con lógica no derivable |
| T6 | **Nodo frágil**: `sales` cuando $d \approx q$ (el `min` cambia de rama según el ruido de $D_0$) | `inventory.py` | §3.2.2 — fragilidad ante U |
| T7 | **Arista empírica**: el efecto del marketing también está en el CSV con ruido | `data/` | Distinguir aristas `exact_code` de `empirical_data` |

### 1.5 El Oráculo (la verdad de referencia)

`oracle/solve.py` es independiente del CGE. Recorre por fuerza bruta todo el espacio $(p, m, q)$ en una rejilla fina, aplica las restricciones y guarda:

- `expected_graph.json`: nodos, categorías y aristas verdaderas.
- `expected_diagnostics.json`: T1–T7, con archivo y líneas.
- `expected_solution.json`: $X^*$, $J^*$, y qué restricciones están activas en el óptimo.
- `expected_infeasible.json`: un objetivo imposible (p. ej. $J \ge 5000$) y la restricción que lo bloquea.

---

## 2. Criterios de aceptación

Cada criterio es **binario** (pasa / no pasa) y se comprueba con un test automático en `benchmark/`.

### Etapa 1 — Ingesta y estructuración

| ID | Criterio | Umbral |
|---|---|---|
| I-1 | El pipeline se ejecuta con un solo comando sobre `golden_case/`, sin etiquetado manual | Sí/No |
| I-2 | Genera un `schema.json` que valida contra el JSON Schema canónico | 100% |
| I-3 | **Recall de nodos** frente a `expected_graph.json` | 100% (12/12) |
| I-4 | **Precisión de aristas** (sin aristas inventadas) | ≥ 95% |
| I-5 | **Recall de aristas** | 100% de las aristas hacia J |
| I-6 | Clasificación U / X / V / target correcta | 100% |
| I-7 | **Trazabilidad**: cada nodo y arista tiene `file` + `line_range` o la referencia de una columna del CSV, y es correcta | 100% |
| I-8 | Tipo de arista correcto (`exact_code` / `discrete_rule` / `empirical_data`) | 100% en T5 y T7 |
| I-9 | Determinismo: 3 ejecuciones dan un `schema.json` idéntico (ignorando timestamps) | Idénticos |
| I-10 | Coeficiente empírico de marketing estimado desde el CSV | Error ≤ 10% sobre el real |

### Etapa 2 — Auditoría y diagnóstico

| ID | Criterio | Umbral |
|---|---|---|
| A-1 | Detecta T1 (nodo muerto) señalando `brand_awareness` | Sí |
| A-2 | Detecta T2 (rompe `c_ledger`) señalando la función y las líneas exactas | Sí |
| A-3 | Detecta T3 (README dice 2%, código hace 3%) | Sí |
| A-4 | Detecta T4: explica que el $q$ local (500) es globalmente subóptimo y cuantifica la pérdida en J | Sí, ±2% |
| A-5 | Marca `sales` como frágil (T6) en el óptimo | Sí |
| A-6 | **Falsos positivos**: diagnósticos que no están en `expected_diagnostics.json` | ≤ 1 |
| A-7 | Sensibilidades $\partial J / \partial x_k$ de la Capa 1 con el signo correcto | 100% |

### Etapa 3 — Resolución (motor de dos capas)

| ID | Criterio | Umbral |
|---|---|---|
| R-1 | La Capa 1 devuelve Top-K rutas $X \to J$ y la ruta verdadera está entre ellas (K=3) | Sí |
| R-2 | Latencia de la Capa 1 en el Golden Case | < 50 ms |
| R-3 | Latencia de la Capa 1 en un grafo sintético de 5.000 nodos | < 50 ms (p95) |
| R-4 | La Capa 2 encuentra un $X$ con $J \ge 0.99 \cdot J^*$ del Oráculo | Sí |
| R-5 | La solución **nunca** viola una restricción hard (comprobado de nuevo por el Oráculo) | 0 violaciones |
| R-6 | Identifica correctamente las restricciones activas en el óptimo | 100% |
| R-7 | Cruza correctamente el escalón de $q=800$ (T5) si el óptimo está al otro lado | Sí |
| R-8 | **Problema imposible**: devuelve `INFEASIBLE` y nombra la restricción que lo bloquea (núcleo UNSAT), sin inventarse una solución | Sí |
| R-9 | Entrega una receta legible: qué variables mover, en qué orden y cuánto ($\Delta X$) | Sí |
| R-10 | **Robustez ante U**: con 1.000 muestras de $D_0$, informa del J esperado y de la probabilidad de violar `c_stock` | Error ≤ 5% sobre el Oráculo |

### Etapa 4 — Visualización

| ID | Criterio | Umbral |
|---|---|---|
| V-1 | La UI renderiza el `schema.json` con colores por categoría U/X/V/J | Sí |
| V-2 | Clic en un nodo → muestra el código fuente exacto y sus datos / distribución | Sí |
| V-3 | Los diagnósticos T1–T6 aparecen como alertas visuales sobre los nodos | Sí |
| V-4 | La ruta óptima de la Capa 2 se resalta en el grafo | Sí |
| V-5 | La UI no tiene lógica propia: si cambia `schema.json`, la UI cambia; editar desde la UI genera una mutación del schema validada | Sí |

### Etapa 5 — Copiloto agéntico

Banco fijo de **10 preguntas** con respuestas conocidas (en `benchmark/copilot_questions.json`). Ejemplos:

- "¿Qué palancas afectan al beneficio?" → `{price, marketing_budget, order_qty}`
- "¿Por qué `brand_awareness` no importa?" → nodo muerto, $\partial J / \partial b = 0$
- "¿Qué pasa si subo el precio a 5?" → valor calculado por el motor, no estimado por el LLM
- "¿Puedo ganar 5.000?" → `INFEASIBLE` por `c_budget`

| ID | Criterio | Umbral |
|---|---|---|
| C-1 | Respuestas correctas | ≥ 9/10 |
| C-2 | Cada afirmación cita los IDs de nodo/arista del grafo | 100% |
| C-3 | **Cero alucinación numérica**: todo número sale de una herramienta determinista | 100% |
| C-4 | El copiloto resalta en la UI los nodos de los que habla | Sí |

---

## 3. Prueba anti-memorización (generalización)

Lo que demuestra que el CGE *entiende* y no está hecho a medida del ejemplo.

| ID | Criterio | Umbral |
|---|---|---|
| G-1 | `generator.py` crea 20 variantes del Golden Case (coeficientes, umbrales y nombres de variables distintos; trampas movidas de archivo) | — |
| G-2 | Las etapas 1–3 pasan en las 20 variantes | ≥ 18/20 |
| G-3 | **Prueba de mutación**: cambiar una constante del código (p. ej. la comisión 0.03 → 0.05) cambia el schema y la solución en la dirección y magnitud que dice el Oráculo | Sí |
| G-4 | **Variante sana**: un Golden Case sin trampas → 0 diagnósticos | 0 falsos positivos |

---

## 4. Reglas de Oro → verificación

| Regla del Manifiesto | Cómo se comprueba |
|---|---|
| Prioridad del determinismo | I-9 + C-3. Hay un registro de qué aristas salen del LLM; en el Golden Case deben ser **0** |
| Estado inmutable fuera del grafo | V-5 |
| Trazabilidad estricta | I-7 |
| Nunca violar restricciones hard | R-5 + R-8 |

---

## 5. Definición final de "HECHO"

El MVP está **terminado** cuando:

1. `make benchmark` ejecuta todos los criterios de §2–§4 y produce un informe.
2. **Todos** los criterios marcados como 100%, Sí o 0 pasan. Los que tienen umbral numérico llegan a su umbral.
3. G-2 y G-4 pasan: el sistema generaliza y no inventa problemas.
4. Se puede grabar una demo de 5 minutos que recorra las 4 etapas sobre una variante **nunca vista**.

---

## 6. Niveles para iterar

Para no atacarlo todo a la vez, el benchmark se puede pasar por niveles:

| Nivel | Qué se activa | Objetivo |
|---|---|---|
| **N0** | Schema escrito a mano + Oráculo + Capas 1 y 2 (R-1…R-9) | Validar el motor matemático aislado |
| **N1** | + Ingesta AST (I-*) sin CSV ni trampas | Validar la extracción |
| **N2** | + Trampas T1–T6 + auditoría (A-*) | Validar el diagnóstico |
| **N3** | + CSV, U estocástica, T7, R-10 | Validar incertidumbre y datos empíricos |
| **N4** | + UI + Copiloto (V-*, C-*) | Validar la experiencia |
| **N5** | + Generador y mutación (G-*) | Validar la generalización → **MVP HECHO** |
