# Especificación del Modelo Interno (IR) y Reglas Automáticas (F0)

**Propósito:** definir el contrato único (`CausalIR` / `schema.json` v1.1) y las reglas 100% automáticas para que el motor descubra, audite y optimice cualquier sistema dentro del alcance del MVP sin configuración manual ni nombres cableados.

---

## 1. Principio de Aislamiento (Cero Fugas de Dominio)

1. **Los extractores** (`cge/ingest.py`, `cge/empirical.py`) son los únicos módulos que leen archivos `.py`, `.csv` o `.md` del proyecto analizado y producen un objeto `CausalIR` serializable a `schema.json`.
2. **Los motores** (`cge/layer1_heuristic.py`, `cge/layer2_exact.py`, `cge/audit.py`, `cge/monte_carlo.py`, `cge/copilot.py`) **solo reciben `CausalIR` / `schema.json`**.
3. **Prohibición estricta:** ningún archivo dentro de `cge/` puede contener literales de nombres de variables, funciones o constantes de ningún caso de prueba (`price`, `order_qty`, `marketing_budget`, `base_demand`, `brand_awareness`, `profit`, `sales`, `demand`, `unit_cost`, `fees`, `holding_cost`).

---

## 2. Deducción 100% Automática de la Tupla $\mathcal{M} = \langle U, X, V, \mathcal{F}, \mathcal{C}, J \rangle$ (D2)

Sin pedirle al usuario un archivo de configuración, el extractor clasifica cada nodo combinando **topología del DAG** + **firma sintáctica AST**:

| Categoría | Regla de detección automática |
|---|---|
| **$U$ (Exógena)** | Nodo raíz ($\text{in-degree} = 0$) cuyo valor proviene de un generador estocástico (`random.*`, `np.random.*`, `gauss`, `normal`, `uniform`) o de una columna de datos sin padres causales. Su distribución $(\mu, \sigma)$ se extrae de los argumentos de la llamada estocástica o de la media/desviación del CSV. |
| **$X$ (Control / Palanca)** | Nodo raíz ($\text{in-degree} = 0$) determinista: constantes configurables de módulo o argumentos de entrada del pipeline que no dependen de otras funciones y poseen cotas (`*_BOUNDS`, `assert a <= x <= b`, o rango observado). Su tipo (`discrete` vs `continuous`) se deduce de `int` vs `float` en la anotación de tipo o literal inicial. |
| **$V$ (Estado Endógeno)** | Todo nodo con $\text{in-degree} > 0$ calculado por una función del sistema que no sea el objetivo final $J$. |
| **$J$ (Target / Objetivo)** | El nodo sumidero principal ($\text{out-degree} = 0$) devuelto por el pipeline principal o con mayor centralidad de ancestros desde $X$ y $U$ (desempatando por semántica de valor neto: ingreso menos costes, utilidad, score o métrica final). |
| **Nodo Muerto (T1-A)** | Cualquier otro nodo con $\text{in-degree} > 0$ que no tenga camino dirigido hacia $J$ ($\mathcal{P}(v \to J) = \emptyset$). Se clasifica como `state` y el auditor lo marca automáticamente como `DEAD_NODE` sin necesidad de conocer su nombre. |

---

## 3. Subconjunto de Código Soportado y Fallback de Caja Negra (D3)

### 3.1. Caja Blanca (Compilación Simbólica Exacta vía `ASTFunctionToSymPy`)
Se traduce automáticamente a ecuaciones `sympy` exactas:
- Asignaciones secuenciales con plegado SSA (`a = expr1; b = expr2(a); return expr3(b)`).
- Aritmética (`+`, `-`, `*`, `/`, `**`), `math.sqrt`, `math.log`, `math.exp`, `abs`.
- Operadores continuos por tramos: `min(a, b)` $\to$ `Min(a, b)`, `max(a, b)` $\to$ `Max(a, b)`.
- Condicionales `if cond: return e1 else: return e2` y expresiones ternarias $\to$ `Piecewise((e1, cond), (e2, True))`.
- Bucles `for` estáticos desenrollables sobre rangos constantes pequeños.

### 3.2. Caja Negra (Fallback por Ejecución y Diferencias Finitas)
Si una función contiene estructuras fuera del subconjunto (llamadas a librerías externas, bucles dinámicos complejos, objetos complejos):
1. El nodo se registra igualmente en el grafo (usando sus argumentos como padres).
2. Su mecanismo se marca como `"engine": "python_callable"` con `"confidence": 0.8`.
3. Sus derivadas parciales $\frac{\partial v_j}{\partial v_i}$ se estiman numéricamente ejecutando la función en un entorno aislado mediante diferencias finitas centrales:
   $$\frac{\partial f_j}{\partial v_i}(z_0) \approx \frac{f_j(z_0 + h e_i) - f_j(z_0 - h e_i)}{2h}$$
4. Nunca se oculta ni se inventa una fórmula: la UI y el Copiloto muestran que ese nodo se evalúa como caja negra ejecutable.

---

## 4. Extracción Automática de Restricciones $\mathcal{C}$ (D4)

El extractor reúne todas las restricciones duras (`type: "hard"`) desde cuatro fuentes automáticas:

1. **Cotas explícitas de control ($X_{\min} \le X \le X_{\max}$):** tuplas `*_BOUNDS = (min, max)` o comparaciones en validadores.
2. **Aserciones y guardas en código (`assert cond`, `if not cond: raise ...`) y funciones de validación/presupuesto:** cualquier desigualdad simbólica que involucre variables del grafo y constantes de techo (ej. `unit_cost * order_qty + marketing_budget <= MAX_BUDGET`).
3. **Invariantes físicas de no-negatividad de flujo/stock:** de todo operador de truncamiento `sales = Min(demand, cap)` o `Max(0, cap - sales)` se registra la restricción estructural `cap - sales >= 0`.
4. **Invariantes de conservación en clases con estado (`Ledger` / balances):** clases con métodos de acumulación opuestos (`credit`/`debit`, `inflow`/`outflow`, `add`/`subtract`) e inspección de métodos `is_balanced()` (`abs(a - b) == 0`).

---

## 5. Jerarquía de Evidencia y Detección de Conflictos (D5)

Cuando dos fuentes describen el mismo parámetro o mecanismo, el sistema aplica esta prioridad estricta:

$$\text{1. Ejecución / Código AST} \;>\; \text{2. Datos Empíricos (CSV)} \;>\; \text{3. Documentación (README / Docstrings)}$$

- **Regla de conflicto Documento vs. Código (T3):** el extractor escanea porcentajes y constantes asociadas a conceptos en `README.md`/docstrings y los compara con las constantes literales de la expresión simbólica AST del nodo correspondiente. Si difieren ($\text{doc} \neq \text{code}$), **el motor ejecuta con el valor del código**, pero emite automáticamente el diagnóstico `DOC_CODE_MISMATCH`.
- **Regla de conflicto Código vs. Datos (T7):** para cada relación con soporte en `data/*.csv`, el estimador econométrico ajusta conjuntamente la ecuación estructural sobre el CSV y registra una arista paralela `type: "empirical_data"` con su confianza estadística $R^2(1 - \text{SE}/|\hat{\theta}|)$.

---

## 6. Selector Automático de Solvers en Capa 2 (D6)

`Layer2Solver` inspecciona las expresiones simbólicas del `CausalIR` y elige automáticamente la estrategia:

| Estructura detectada en `CausalIR` | Estrategia seleccionada |
|---|---|
| Todas las ecuaciones son diferenciables $C^\infty$ | `SLSQP` multi-arranque con gradientes analíticos de Reverse-Mode AD |
| Contiene $M \le 6$ puntos de cambio (`Piecewise`, `Min`, `Max`) | **Partición Automática de Regímenes**: extrae las fronteras simbólicas $A(X) = B(X)$ y $x_k = \theta$, resuelve `SLSQP` suave en cada régimen factible y evalúa el entorno entero de las variables discretas |
| Contiene $M > 6$ puntos de cambio | **Continuación Homotópica Log-Sum-Exp ($\text{LSE}_\tau$) + Epígrafe MPCC** (3 llamadas NLP en lugar de $2^M$) |
| Nodos $U$ con $\sigma_U > 0$ y modo estocástico / riesgo activo | **Cuadratura Compuesta partida en el kink / SAA + Rockafellar-Uryasev $\text{CVaR}_\alpha$** |
| Meta $J_{\text{target}} > J^*$ (imposible) | Devuelve `INFEASIBLE` + **Relajación Elástica $\ell_1$ y Precios Sombra KKT** ($\lambda_i^* > 0$) para identificar qué restricción exacta bloquea la meta |
| Meta $J_{\text{target}} \le J^*$ desde estado actual $X_0$ | **Intervención Inversa Escasa ($\ell_1 + \ell_2$)** que minimiza el número de palancas movidas y ordena los pasos sin violar $\mathcal{C}$ en ningún punto intermedio |

---

## 7. Esquema Canónico Enriquecido (`schema.json` v1.1)

Compatible al 100% con el JSON Schema actual (`schema/canonical_schema.json`), añadiendo campos opcionales que hacen que `schema.json` sea **autosuficiente** (ningún motor necesita volver a abrir los `.py`):

```json
{
  "nodes": [
    {
      "id": "string",
      "label": "string",
      "category": "exogenous | control | state | target",
      "data_type": "continuous | discrete | boolean | categorical",
      "source_reference": {"file": "string", "entity": "string", "line_range": [1, 10]},
      "current_value": 0.0,
      "bounds": {"min": 0.0, "max": 100.0},
      "distribution": {"family": "normal", "mean": 1000.0, "std": 100.0},
      "characteristic_scale": 100.0,
      "symbolic_expr": "Piecewise((1.2, order_qty < 800), (1.0, True))",
      "clarke_kink_index": 0.0
    }
  ],
  "edges": [
    {
      "source": "string",
      "target": "string",
      "type": "exact_code | discrete_rule | empirical_data",
      "sensitivity_weight": 0.0,
      "dimensionless_elasticity": 0.0,
      "confidence": 1.0,
      "exact_mechanism": {
        "engine": "sympy_expr | python_callable | smt_clause",
        "definition": "string"
      }
    }
  ],
  "constraints": [
    {
      "id": "string",
      "type": "hard | soft",
      "expression": "unit_cost * order_qty + marketing_budget - 2500 <= 0",
      "description": "string"
    }
  ]
}
```
