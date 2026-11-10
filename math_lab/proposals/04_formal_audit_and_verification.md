# Propuesta Matemática 04: Auditoría Formal Simbólica (T1–T6) e Inferencia Econométrica Estructural (T7)

**Autor:** Agente de Arquitectura Matemática (CGE Math Lab)  
**Estado:** Implementado y Verificado Experimentalmente en [`exp04_symbolic_auditor_and_econometrics.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/experiments/exp04_symbolic_auditor_and_econometrics.py)

---

## 1. Diagnóstico Forense de `cge/audit.py` y `cge/empirical.py`

1. **T1 (`_check_t1_dead_nodes`) sólo comprueba conectividad topológica (`nx.has_path`):**
   Un nodo puede tener caminos dirigidos hacia $J$ y aun así tener derivada total $\frac{dJ}{dv_i} = 0$ por dos razones matemáticas no detectadas actualmente:
   * **Muerte por Régimen Activo (Saturación):** En el punto nominal $X_0 = (p=3.5, m=500, q=500)$, la demanda es $d = 1103.88 > q = 500$. Como $\text{sales} = \min(d, q) = 500$, localmente $\frac{\partial \text{sales}}{\partial d} = 0$, lo que bloquea el efecto marginal de `demand` sobre `sales` hasta que no se aumente $q$ o se suba el precio.
   * **Violación de Fidelidad / Cancelación Algebraica Exacta:** Si un nodo $v$ influye en $J$ por dos rutas cuyos efectos se anulan idénticamente ($\sum_{p \in (v \to J)} \prod_{e \in p} w_e \equiv 0$), `nx.has_path` dice `True`, pero $\frac{dJ}{dv} \equiv 0$.
2. **T2 (`_check_t2_ledger_invariants`) cuenta llamadas AST por nombre (`len(debit_calls) != len(credit_calls)`):**
   Si una transacción válida divide un crédito de $a$ en dos débitos de $0.8a$ y $0.2a$ (como el desglose base imponible + IVA: `record_credit(a); record_debit(0.8*a); record_debit(0.2*a)`), el contador ingenuo `1 != 2` lanza un **Falso Positivo crítico** porque no evalúa la suma simbólica de los argumentos. Y a la inversa, si hace `record_credit(a); record_debit(2*a)`, tiene 1 llamada a cada uno (`1 == 1`) y el auditor actual **no detecta el bug**.
3. **T4 (`_check_t4_local_optima`) tiene hardcodeado `q_local = 500` y una rejilla de 4 puntos.**
4. **T6 (`_check_t6_fragile_nodes`) hace `if edge.target == "sales":`**, sin evaluar jamás una derivada ni una varianza.
5. **T7 (`cge/empirical.py`) hace trampa con la ecuación verdadera:**
   En la línea 21 de [`cge/empirical.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/cge/empirical.py#L21):
   ```python
   price_effect = d0 * (1.0 - 0.15 * (p - 3.0))
   marketing_lift = d - price_effect
   ```
   ¡Usa el coeficiente verdadero `0.15` hardcodeado dentro del estimador empírico para despejar `marketing_lift`! En un caso real donde se desconoce la ecuación o en cualquiera de las 20 variantes de `generator.py` donde la elasticidad no es `0.15`, este estimador falla sistemáticamente.

---

## 2. Formalización Matemática de las Reglas de Auditoría (T1–T6)

### 2.1. Taxonomía Tripartita de Nodos Muertos (T1)
Para todo nodo de estado $v_i \in V \setminus \{J\}$, evaluamos tanto la conectividad topológica como la expresión simbólica de su adjunto total $\bar{v}_i(X, U) = \frac{dJ}{dv_i}$:
1. **Tipo T1-A (Nodo Estructuralmente Huérfano):** $\mathcal{P}(v_i \to J) = \emptyset \implies \bar{v}_i \equiv 0$ globalmente (ej. `brand_awareness`).
2. **Tipo T1-B (Cancelación Algebraica / Violación de Fidelidad):** $\mathcal{P}(v_i \to J) \neq \emptyset$, pero `sympy.simplify` demuestra que $\sum_{p \in \mathcal{P}(v_i \to J)} \prod_{(a \to b) \in p} \frac{\partial f_b}{\partial v_a} \equiv 0$ en todo el dominio.
3. **Tipo T1-C (Nodo Localmente Silenciado por Saturación):** $\bar{v}_i(X, U) \not\equiv 0$ globalmente, pero en el estado operativo actual $z_0 = (X_0, U_0)$, $\bar{v}_i(z_0) = 0$ debido a que una compuerta $\min/\max/\text{clip}$ aguas abajo está saturada.

### 2.2. Verificación Simbólica de Invariantes de Conservación (T2)
Sea una clase de estado mutable con invariante lineal de conservación $\mathcal{I}(S) = \sum_{k} \alpha_k S_k = 0$ (para un libro mayor de partida doble, $\mathcal{I}(\text{Ledger}) = \sum \text{credits} - \sum \text{debits} = 0$).
Para auditar cualquier función de transición $T(\text{ledger}, a_1, \dots, a_m)$:
1. Instanciamos un **Estado Simbólico Abstracto** `SymbolicLedger` e inyectamos símbolos algebraicos libres estrictamente positivos $a_1, \dots, a_m \in \mathbb{Sym}(\mathbb{R}_{>0})$.
2. Ejecutamos simbólicamente el cuerpo de $T$ (o interpretamos su AST sobre el estado simbólico), acumulando los polinomios/expresiones simbólicas de créditos $\mathcal{C}_{\text{sym}}(a)$ y débitos $\mathcal{D}_{\text{sym}}(a)$.
3. Calculamos el **Residuo Simbólico de Conservación**:
   $$\mathcal{R}_T(a) = \text{simplify}\big(\mathcal{C}_{\text{sym}}(a) - \mathcal{D}_{\text{sym}}(a)\big)$$
   * Si $\mathcal{R}_T(a) \equiv 0$, la función respeta la ley de conservación **para cualquier entrada numérica posible** (incluso si divide un asiento en múltiples apuntes como base + IVA: $a - (0.8a + 0.2a) \equiv 0$).
   * Si $\mathcal{R}_T(a) \neq 0$ (en `process_refund`, $\mathcal{R}_T(\text{amount}) = \text{amount} - 2\cdot\text{amount} = -\text{amount} \neq 0$), se demuestra formalmente la violación del invariante y se reporta la expresión exacta de la fuga contable $\mathcal{R}_T(a)$.

### 2.3. Precio de la Anarquía y Brecha de Desacoplamiento Bilevel (T4)
Cuando un submódulo $S$ contiene una política o heurística local que fija un subconjunto de variables de control $X_S = x_S^{\text{loc}}$ optimizando un objetivo miope $J_S \neq J$, definimos la **Pérdida por Suboptimalidad Local (Bilevel Decoupling Loss)** como:
$$\Delta J_{\text{local}}(x_S^{\text{loc}}) = \max_{X \in \Omega_X, \, \mathcal{C}(X) \le 0} J(X) \;-\; \max_{X_{-S} : (x_S^{\text{loc}}, X_{-S}) \in \Omega_X, \, \mathcal{C}(x_S^{\text{loc}}, X_{-S}) \le 0} J(x_S^{\text{loc}}, X_{-S})$$
Ambos términos se resuelven con el **Solucionador Universal MINLP de Capa 2** (Propuesta 02), primero con $X_S$ libre y luego fijando la restricción de igualdad $X_S = x_S^{\text{loc}}$. En el `golden_case`, para $q_{\text{loc}} = 500$:
* $\max_{p, m, q} J = 2990.8835€$ (en $q^*=800, p^*=5.4370, m^*=428.2140$)
* $\max_{p, m} J(p, m, q=500) = 2310.0000€$ (en $p^*=6.0, m^*=0.0$, donde $d=550 \ge 500$)
* $\Delta J_{\text{local}} = 2990.8835 - 2310.0000 = 680.8835€$, obtenido analíticamente sin rejillas arbitrarias.

### 2.4. Índice de Fragilidad de Clarke y Sensibilidad de Varianza (T6)
Para cada nodo $v_j$ en un punto de operación $z^*$ (por ejemplo, el óptimo determinista $X^*$), definimos dos métricas matemáticas complementarias de fragilidad:
1. **Índice de Asimetría Subdiferencial de Clarke ($\mathcal{K}_j$):**
   $$\mathcal{K}_j(z^*) = \max_{u \in \text{padres}(v_j)} \left| D^+_u f_j(z^*) - D^-_u f_j(z^*) \right|$$
   Si $\mathcal{K}_j(z^*) > 0$, el nodo se encuentra exactamente sobre una cresta no diferenciable (*kink*), lo que garantiza por la Desigualdad de Jensen que las fluctuaciones simétricas de $U$ producirán una pérdida sistemática de primer orden en valor esperado.
2. **Distancia Estandarizada a la Frontera de Conmutación ($\zeta_j$):**
   Para todo nodo con ramas $\min(A, B)$ o $\max(A, B)$, medimos la distancia en unidades de desviación típica exógena $\sigma_{|A-B|}$:
   $$\zeta_j(z^*) = \frac{|A(z^*) - B(z^*)|}{\sqrt{\nabla_U (A - B)^T \Sigma_U \nabla_U (A - B)} + \epsilon}$$
   Cuando $\zeta_j(z^*) < 1.0$ (en el óptimo determinista del `golden_case`, $|d^* - q^*| = 0 \implies \zeta_{\text{sales}} = 0.0$), la probabilidad de conmutación de régimen ante ruido exógeno supera el $15.8\%$ (siendo $50\%$ en $\zeta = 0$), activando automáticamente la alerta `FRAGILE_NODE` (T6).

---

## 3. Inferencia Econométrica Estructural Conjunta e Identificabilidad sobre CSV (T7)

Para estimar relaciones empíricas desde datos históricos tabulares (`sales_history.csv`) **sin hacer trampa usando constantes del código**:

### 3.1. Formulación del Modelo Estructural Lineal en los Parámetros
Dadas las variables padres de `demand` identificadas topológicamente en el DAG ($\text{padres}(\text{demand}) = \{D_0, p, m\}$), postulamos una familia de funciones base candidata $\Phi(D_0, p, m)$:
$$d_t = \theta_1 D_{0,t} + \theta_2 \big(D_{0,t} \cdot p_t\big) + \theta_3 \sqrt{m_t} + \theta_4 m_t + \theta_5 \ln(1 + m_t) + \varepsilon_t$$
Nótese que el modelo estructural real $d_t = D_{0,t}(1 - \alpha(p_t - p_{\text{ref}})) + \beta\sqrt{m_t} = (1 + \alpha p_{\text{ref}}) D_{0,t} - \alpha (D_{0,t} p_t) + \beta \sqrt{m_t}$ es **exactamente lineal en los parámetros estructurales** $\boldsymbol{\theta} = [\theta_1, \theta_2, \theta_3]^T$ donde:
$$\alpha = -\theta_2, \qquad p_{\text{ref}} = \frac{\theta_1 - 1}{\alpha}, \qquad \beta = \theta_3$$

### 3.2. Estimación Conjunta OLS / Selección Escasa (BIC-LASSO) y Diagnóstico de Identificabilidad
Sea $X_{\text{des}} \in \mathbb{R}^{N \times K}$ la matriz de diseño construida sobre las $N = 365$ observaciones históricas:
1. **Diagnóstico de Identificabilidad (Número de Condición):** Calculamos el número de condición espectral de la matriz de diseño normalizada $\tilde{X}$:
   $$\kappa(\tilde{X}) = \frac{\sigma_{\max}(\tilde{X})}{\sigma_{\min}(\tilde{X})}$$
   Si $\kappa(\tilde{X}) < 100$, todos los parámetros causales $(\alpha, \beta)$ son **simultáneamente identificables** a partir de los datos históricos sin necesidad de fijar ninguno desde el código.
2. **Estimador de Mínimos Cuadrados Ordinarios (OLS) y Matriz de Covarianza Heterocedástica (HC3 / White):**
   $$\hat{\boldsymbol{\theta}} = (X_{\text{des}}^T X_{\text{des}})^{-1} X_{\text{des}}^T \mathbf{y}, \qquad \widehat{\text{Cov}}(\hat{\boldsymbol{\theta}}) = (X_{\text{des}}^T X_{\text{des}})^{-1} \left( \sum_{t=1}^N \frac{\hat{\varepsilon}_t^2}{(1 - h_{tt})^2} \mathbf{x}_t \mathbf{x}_t^T \right) (X_{\text{des}}^T X_{\text{des}})^{-1}$$
3. **Métrica de Confianza Causal (`confidence` en `schema.json`):** En lugar de inventar `confidence = 0.95` como constante fija en `cge/ingest.py`, la confianza de la arista empírica se deriva matemáticamente del coeficiente de determinación $R^2$ y del error estándar relativo:
   $$\text{confidence}(e) = R^2 \cdot \max\left(0, \; 1 - \frac{\text{SE}(\hat{\theta}_e)}{|\hat{\theta}_e|}\right)$$
