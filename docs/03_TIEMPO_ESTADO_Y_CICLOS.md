# ESPECIFICACIÓN FUNDACIONAL 03: TIEMPO, ESTADO Y BUCLES DE RETROALIMENTACIÓN
## ActorDB & Causal Logic Engine
**Estado:** Borrador de Trabajo v0 (Draft)  
**Propósito:** Formalizar cómo `ActorDB` representa, audita, simula y optimiza sistemas con memoria intertemporal, retardos causales, máquinas de estados y lazos de retroalimentación cerrados sin perder tratabilidad ni garantías formales. Ancla la cuña *Zero-to-One* en **Causal Diff para CI/CD** (detección de roturas de balance intertemporal, lazos algebraicos no bien planteados y regresiones dinámicas) y el **Oráculo Causal MCP para agentes de IA** (simulaciones contrafactuales temporales, trayectorias óptimas MPC y Bounded Model Checking).

---

## 0. Marco Empírico de Referencia y Cuña *Zero-to-One*

Este documento conceptual formaliza la dinámica temporal y cíclica del motor integrando los hallazgos de la auditoría de estado del arte y el baseline empírico congelado en `docs/baseline/`:
* [**Estado del Arte y Análisis Competitivo (`docs/baseline/01_estado_del_arte.md`)**](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/baseline/01_estado_del_arte.md): Evidencia la incapacidad de los agentes LLM y herramientas actuales de razonamiento estático para gestionar acumulación intertemporal, retardos o lazos algebraicos sin incurrir en alucinaciones aritméticas o sesgos de anticipatividad.
* [**Flujo Actual y Baseline Empírico (`docs/baseline/02_flujo_actual_y_baseline_empirico.md`)**](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/baseline/02_flujo_actual_y_baseline_empirico.md): Audita el comportamiento de los contendientes ante problemas de inventario multi-periodo (`s06_multiperiod_inventory` en OR-Bench) y transiciones de estado secuenciales.
* [**Mapa de Vacíos Tecnológicos y Prior Art (`docs/baseline/03_mapa_de_vacios.md`)**](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/baseline/03_mapa_de_vacios.md): Cataloga los 9 vacíos del estado del arte, destacando la frontera *Zero-to-One* (`GAP-01` a `GAP-04`) frente a problemas explorados (`GAP-05`, `GAP-06`) y tecnologías maduras (`GAP-07` a `GAP-09`).
* [**Marco Formal de Métricas y Grader Determinista (`docs/baseline/04_marco_de_metricas.md`)**](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/baseline/04_marco_de_metricas.md): Establece las reglas de evaluación determinista para trayectorias de simulación, verificación de invariantes temporales y penalización estricta por falsa confianza.

### La Cuña Comercial y Operativa *Zero-to-One* en Tiempo y Ciclos
La gestión temporal y de ciclos en ActorDB habilita capacidades operativas críticas:
1. **Causal Diff para CI/CD en Sistemas Dinámicos:**
   - **Detección de lazos algebraicos instantáneos ($\Lambda(C) = 0$):** Alerta al desarrollador cuando un PR introduce dependencias circulares no demoradas en reglas de negocio o cálculos financieros (`ILL_POSED_CYCLE` o equilibrios múltiples no controlados).
   - **Auditoría de Invariantes Intertemporales:** Verifica que las leyes de conservación de stock, saldo o balance contable se mantengan a lo largo de cualquier horizonte $T$ sin fugas de inventario ni de capital.
   - **Prevención de Miopía Terminal:** Detecta si una modificación en la función objetivo o en las restricciones temporales vacía los almacenes o corta inversiones críticas al final del horizonte de planificación.
2. **Oráculo Causal MCP para Agentes de IA en Escenarios Secuenciales:**
   - **Consultas Contrafactuales Dinámicas:** Permite al agente ejecutar `DO(X[t] = x)` para simular pulsos transitorios o cambios sostenidos de política sin alucinaciones numéricas.
   - **Model Predictive Control (MPC) y Bounded Model Checking (BMC):** Proporciona al agente un oráculo determinista que resuelve trayectorias óptimas secuenciales y verifica propiedades temporales $\forall t \le T: \Phi$ mediante SMT (`PROVE`), entregando contraejemplos paso a paso si la propiedad es falsada (`GAP-07`).

---

## 1. El Problema y Diagnóstico del Motor Base

En [`02_FRONTERA_IMPERATIVA_Y_CONTRATOS.md`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/02_FRONTERA_IMPERATIVA_Y_CONTRATOS.md) resolvimos cómo purificar un objeto mutable en una transición funcional $(s_{\text{in}}, x) \mapsto (s_{\text{out}}, y)$. En la realidad operativa de cualquier empresa, ese estado `s_out` rara vez muere al terminar una llamada: **se convierte en el `s_in` del instante siguiente** (el stock sobrante de hoy es el inventario inicial de mañana; la deuda impagada de este mes devenga intereses el mes próximo; los clientes retenidos hoy conforman la base activa del trimestre siguiente).

Al examinar cómo el código actual (`cge/`) maneja el tiempo y los ciclos, se observan tres carencias estructurales:

1. **Desenrollado Manual No Escalable ([`benchmark/or_bench/problems_sealed.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/benchmark/or_bench/problems_sealed.py#L242-L280)):**
   En el caso `s06_multiperiod_inventory`, un balance de solo $T=4$ periodos obliga a declarar a mano cuatro palancas distintas (`LOT_1`…`LOT_4`), cuatro columnas de estado (`inv_1`…`inv_4`) y ocho restricciones expandidas manualmente. Para un horizonte realista de $T=52$ semanas o $T=365$ días, esa representación manual se vuelve inmanejable para humanos y prohibitiva en tokens para un agente de IA.
2. **Suposición Rígida de Aciclicidad Estática ([`cge/layer1_heuristic.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/cge/layer1_heuristic.py#L73)):**
   El enrutador y los evaluadores comprueban `nx.is_directed_acyclic_graph(self.graph)` sobre el grafo estático de nombres de variables. Si se proyecta `inventario -> ventas -> inventario` olvidando los subíndices temporales, aparece un ciclo dirigido y los algoritmos topológicos dejan de operar.
3. **Ausencia de Primitivas Temporales en la IR ([`docs/IR_SPEC.md`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/IR_SPEC.md)):**
   Todas las aristas actuales son instantáneas ($\text{lag} = 0$). No existe noción nativa de estado inicial, retardo causal, horizonte de planificación ni valor terminal.

---

## 2. El Principio Rector: *"Todo Ciclo Causal o Tiene un Retardo o es un Equilibrio"*

En lugar de prohibir los ciclos en `ActorDB`, dotamos a toda arista dirigida $e = (u \to v) \in E$ de un atributo entero no negativo llamado **retardo causal** $\ell(e) \in \mathbb{Z}_{\ge 0}$, donde $\ell(e) = 0$ indica efecto dentro del mismo paso temporal $t$ y $\ell(e) = k \ge 1$ indica que el valor de $u$ en $t - k$ afecta a $v$ en $t$.

Para cualquier ciclo dirigido elemental $C = (v_1 \to v_2 \to \dots \to v_m \to v_1)$ en el grafo estructural estático, definimos su **peso de retardo de ciclo**:

$$\Lambda(C) = \sum_{e \in C} \ell(e)$$

Esto particiona matemáticamente **todos los ciclos posibles** en exactamente dos clases con significado físico y algorítmico bien definido:

```
                      ¿Aparece un ciclo C en el mapa de actores?
                                          │
                       Calcular retardo total Λ(C) = Σ ℓ(e)
                                          │
                 ┌────────────────────────┴────────────────────────┐
                 ▼                                                 ▼
        CASO A: Λ(C) ≥ 1                                  CASO B: Λ(C) = 0
   CICLO DINÁMICO (CON MEMORIA)                     LAZO ALGEBRAICO (SIMULTÁNEO)
 • Causalidad bien fundada en el tiempo           • Acoplamiento instantáneo v = H(v, x, u)
 • Al expandir t = 1..T se vuelve un DAG exacto   • Representa un equilibrio / punto fijo
 • Soporta AD inversa (BPTT), MILP/NLP y BMC      • Requiere resolver F(v) = 0 y auditar
                                                    existencia y unicidad (det(I - J_H) ≠ 0)
```

* **Teorema Estructural Básico:** El grafo expandido en el tiempo $\mathcal{G}_T$ a lo largo de cualquier horizonte $T \ge 1$ es un **Grafo Acíclico Dirigido (DAG) estricto** si y solo si todo ciclo del grafo compacto cumple $\Lambda(C) \ge 1$.

---

## 3. Taxonomía de Fenómenos Temporales y Cíclicos (`D1`–`D6`)

| Código | Fenómeno Dinámico | Ejemplo en Sistemas de Negocio / Software | Modelo en `ActorDB` |
| :--- | :--- | :--- | :--- |
| **D1** | **Acumuladores y Leyes de Conservación Intertemporal (*Stocks*)** | Inventario con *carry-over*, saldo de tesorería, deuda con interés compuesto, cartera activa de suscriptores. | Ecuación en diferencias $S_t = S_{t-1} + \text{in}_t - \text{out}_t$ con condición inicial explícita $S_0$. |
| **D2** | **Retardos de Propagación y Efectos Diferidos (*Lags*)** | Tiempo de tránsito de proveedores (*lead time* $L$ días), impacto diferido de campañas de marketing, maduración de cohortes. | Aristas con retardo explícito $\ell(e) = k \ge 1$ ($v_t = f(x_{t-k}, \dots)$). |
| **D3** | **Máquinas de Estados Discretos y Ciclo de Vida Lógico** | Estado de un pedido (`CREADO -> PAGADO -> ENVIADO`), cuenta (`ACTIVA -> MOROSA -> BLOQUEADA`). | Transición guardada sobre actores $V_{\text{logic}}$: $q_t = \delta(q_{t-1}, \text{evento}_t)$. |
| **D4** | **Lazos Algebraicos Simultáneos ($\Lambda(C) = 0$)** | Precio de liquidación de mercado donde oferta y demanda se igualan en el mismo instante; comisiones circulares sobre neto/bruto. | Bloque de ecuaciones simultáneas $v_t = H(v_t, x_t, u_t)$ resuelto como punto fijo con diagnóstico de unicidad. |
| **D5** | **Bucles de Retroalimentación con Política (*Closed-Loop Control*)** | Regla automática escrita en el repositorio: *"si `stock[t] < min_threshold`, emitir `order[t] = target_stock - stock[t]`"*. | La regla opera como estado endógeno; sus parámetros estructurales (`min_threshold`, `target_stock`) son las palancas $X$. |
| **D6** | **Tiempo Continuo Puro y Simulación por Eventos Discretos (ODE / DES)** | Ecuaciones diferenciales físicas en tiempo continuo, colas $M/G/k$ a nivel de microsegundo. | **Fuera del alcance nativo inicial:** solo se admite previa discretización temporal fija ($\Delta t$) o encapsulado como oráculo `L4` (según P1). |

---

## 4. Representación Dual: Plantilla Compacta vs. Grafo Expandido ($\mathcal{G}_T$)

Para evitar la duplicación manual de variables por cada periodo, `ActorDB` separa la **declaración intensional** de la **expansión computacional**:

1. **Vista Compacta (Plantilla Intensional — Interfaz para Humanos e IA):**
   Los actores dinámicos se declaran **una única vez** parametrizados sobre un paso discreto $t \in \{1, \dots, T\}$, especificando únicamente:
   * La condición de frontera inicial (`init`, en $t=0$ o para $t \le 0$ si hay retardos de orden $k$).
   * La ley causal de transición de $t-k \to t$.
   * Las invariantes universales cuantificadas (`for all t`).

   *Ejemplo conceptual frente a las 35 líneas repetitivas de `S06`:*
   ```text
   horizon T = 4
   lever   lot[t]       : Real in [0, 110]
   exogenous demand[t]  : Real = [70, 90, 80, 100]
   state   inv[t]       : Real
     init  inv[0] = 20
     law   inv[t] <- inv[t-1] + lot[t] - demand[t]

   invariant stock_bounds[t] : 0 <= inv[t] <= 60
   objective total_cost      : min( sum(t: cost[t]*lot[t] + 1.5*inv[t]) )
   ```

2. **Vista Expandida ($\mathcal{G}_T$ — Estructura Interna de Resolución):**
   Cuando se ejecuta una consulta sobre un horizonte $T$ (libre de lazos $\Lambda=0$), el motor instancia en memoria el DAG desenrollado $\mathcal{G}_T = (V_T, E_T)$:
   $$V_T = V_{\text{init}} \cup \bigcup_{t=1}^T \{v_t : v \in V\}, \qquad E_T = \bigcup_{t=1}^T \bigcup_{(u \to v) \in E} \big\{(u_{t-\ell}, v_t)\big\}$$
   Sobre $\mathcal{G}_T$, **toda la maquinaria matemática de `cge/` y `P1` funciona de forma nativa** sin reescribir solvers.

---

## 5. Semántica del Horizonte y Prevención del "Efecto Fin del Mundo"

Todo problema temporal requiere precisar qué ocurre en la frontera final $t = T$. Si un optimizador maximiza el beneficio acumulado hasta el viernes sin valorar el inventario que queda el viernes por la noche, **vaciará los almacenes a cero y cortará toda inversión de marketing en los últimos periodos** porque el futuro $t > T$ no vale nada para él (*End-of-Horizon Myopia*).

`ActorDB` define **tres regímenes de horizonte**:

1. **Horizonte Finito con Condición o Valor Terminal Explícito ($T < \infty$):**
   La meta cuantitativa acumulada incorpora obligatoriamente o bien una **restricción terminal de sostenibilidad** ($S_T \ge S_{\min}$, ej. no dejar el stock por debajo del nivel de seguridad inicial $S_0$), o bien una **función de valor residual** $\Psi(S_T)$:
   $$J_{1:T} = \sum_{t=1}^T \gamma^{t-1} r_t(S_t, X_t, U_t) + \gamma^T \Psi(S_T)$$
   Si el código analizado optimiza o evalúa un horizonte finito sin condición ni valor terminal sobre un estado conservativo `D1`, el auditor de `ActorDB` emitirá una advertencia estructural de **Miopía de Frontera Terminal**.
2. **Horizonte Deslizante (*Model Predictive Control* — MPC):**
   Modo operativo por defecto para toma de decisiones secuenciales: en el instante actual $t_0$, `SOLVE` optimiza la ventana $\{t_0, \dots, t_0 + H - 1\}$, prescribe ejecutar únicamente la acción inmediata $X_{t_0}^*$, observa la realización real del entorno $U_{t_0}$ y desplaza la ventana a $t_0 + 1$.
3. **Régimen Estacionario / Equilibrio de Largo Plazo ($T \to \infty$):**
   Para consultas estratégicas (*"¿cuál es la tarifa y nivel de reposición sostenibles en estado estacionario?"*), el motor busca un punto de equilibrio invariante en el tiempo $(S^*, X^*)$ tal que $S^* = f(S^*, X^*, \bar{U})$ cumpliendo todas las invariantes $\mathcal{C}$.

---

## 6. Planes en Lazo Abierto vs. Políticas Cerradas Paramétricas

Ante un sistema que evoluciona en el tiempo, decidir "qué hacer" admite dos formulaciones muy distintas que `ActorDB` soporta con una jerarquía clara:

1. **Plan en Lazo Abierto (*Open-Loop Schedule*):**
   Una trayectoria numérica concreta de decisiones $X_{1:T} = (x_1, x_2, \dots, x_T)$ calculada en $t=0$ (por ejemplo, el plan semanal de compras `lot[1..4]`). Es ideal bajo certidumbre o combinado con re-planificación en horizonte deslizante (MPC).
2. **Política Reactiva en Lazo Cerrado (*Closed-Loop Parametric Policy*):**
   Una ley de decisión $x_t = \pi_\theta(S_{t-1}, \text{señales}_t)$ que adapta la acción al estado observado en cada periodo.
   * **Ventaja decisiva en `ActorDB`:** No necesitamos entrenar cajas negras de Aprendizaje por Refuerzo desde cero, porque en los repositorios reales **la estructura de la política ya está escrita en el código del negocio** (ej. una política de inventario $(s, S)$, una regla de reintento exponencial o un escalado automático de capacidad).
   * En ese caso, las decisiones instantáneas $x_t$ pasan a ser actores de estado calculados por la regla, mientras que **los hiperparámetros o umbrales fijos en el tiempo $\theta = (s, S)$ son las verdaderas Palancas $X$**. Optimizar $\theta$ sobre el grafo expandido $\mathcal{G}_T$ (bajo múltiples escenarios de $U_{1:T}$) halla automáticamente la mejor regla reactiva respetando el código real del repositorio.

---

## 7. Invariante Estructural de No-Anticipatividad (Causalidad Temporal Estricta)

Cuando coexisten tiempo ($t = 1, \dots, T$) e incertidumbre exógena ($U_1, \dots, U_T$), aparece una trampa matemática letal: si se genera una muestra del futuro $(u_1, \dots, u_T)$ y se permite al solver elegir libremente $(x_1, \dots, x_T)$ para cada escenario por separado, el optimizador **hará trampa eligiendo $x_1$ sabiendo cuánto valdrá la demanda $u_4$ dentro de cuatro meses** (*Clarividencia Artificial / Lookahead Bias*).

`ActorDB` impone la **Restricción de No-Anticipatividad** como ley estructural inquebrantable:
* Sea $\mathcal{F}_t = \sigma(S_0, U_1, \dots, U_t)$ la información revelada hasta el instante $t$ (o hasta $t-1$ si la decisión $X_t$ se toma antes de observar $U_t$).
* Para cualesquiera dos escenarios $\omega, \omega'$ que compartan la misma historia observada hasta el momento de decidir $X_t$, el motor exige:
  $$X_t(\omega) = X_t(\omega')$$
* En particular:
  * Un **Plan en Lazo Abierto** robusto exige $X_t(\omega) = x_t$ determinista idéntico para todos los escenarios $\omega$.
  * Una **Política Paramétrica** $X_t(\omega) = \pi_\theta(S_{t-1}(\omega))$ cumple la no-anticipatividad **automáticamente por construcción**, ya que $\theta$ es único y $S_{t-1}(\omega)$ solo depende de realizaciones pasadas.

---

## 8. Lazos Algebraicos Simultáneos ($\Lambda(C) = 0$) y Diagnóstico de Equilibrio

Cuando el grafo contiene una componente fuertemente conexa sin ningún retardo temporal ($\Lambda(C) = 0$), los actores implicados $z \in \mathbb{R}^m$ satisfacen un sistema implícito simultáneo dado el resto de entradas $p$:

$$z = H(z, p) \iff R(z, p) \equiv z - H(z, p) = 0$$

Siguiendo la directriz aprobada, `ActorDB` **acepta lazos algebraicos** sometiéndolos a un protocolo riguroso de resolución y diagnóstico:

1. **Clasificación y Resolución del Bloque Simultáneo:**
   * Si $H$ es afín en $z$ ($z = A z + b(p)$), se verifica simbólica o numéricamente la invertibilidad de $(I - A)$ y se resuelve exactamente $z = (I - A)^{-1} b(p)$ (`L0`).
   * Si $H$ es no lineal y diferenciable, se resuelve $R(z, p) = 0$ mediante Newton-Raphson amortiguado / Powell híbrido y se propagan sensibilidades exactas aplicando el **Teorema de la Función Implícita**:
     $$\frac{\partial z^*}{\partial p} = \big(I - J_z H(z^*, p)\big)^{-1} J_p H(z^*, p)$$
2. **Diagnóstico Obligatorio de Bien-Planteamiento (*Well-Posedness Audit*):**
   Un lazo sin retardo puede esconder un bug grave de circularidad en el código o múltiples equilibrios económicos. El auditor clasifica cada bloque $\Lambda(C) = 0$ en uno de tres estados:
   * **`UNIQUE_EQUILIBRIUM`:** Contracción verificada ($\|J_z H\| < 1$ en el dominio) o sistema lineal regular ($\det(I - A) \neq 0$). Conserva garantía `PROVEN` si los mecanismos son `L0`.
   * **`LOCAL_EQUILIBRIUM_NONUNIQUE_RISK`:** Converge localmente con $\det(I - J_z H(z^*)) \neq 0$, pero no se garantiza unicidad global (posibles equilibrios múltiples). Se degrada automáticamente la garantía a `SAMPLED_VERIFIED` alertando del supuesto de selección de equilibrio.
   * **`ILL_POSED_CYCLE` (Contradicción o Divergencia):** No existe punto fijo (ej. `a = b + 1; b = a + 1`) o Jacobiano singular. Se reporta como patología crítica de coherencia lógica del sistema.

---

## 9. Consultas Causales en Presencia del Tiempo

Las primitivas de consulta de `ActorDB` adquieren semántica temporal nativa:

1. **`TRACE(X -> G)` — Función de Respuesta al Impulso Causal (*Impulse Response*):**
   En un sistema dinámico, el impacto de mover una palanca en $t_0$ se reparte en el tiempo. `TRACE` devuelve el perfil temporal de derivadas totales a distintos horizontes $\tau \ge 0$:
   $$\text{IRF}_{X \to V}(\tau) = \frac{d V_{t_0 + \tau}}{d X_{t_0}}$$
   distinguiendo **efectos inmediatos** ($\tau = 0$), **picos diferidos** ($\tau = \ell$) y **efectos acumulados de largo plazo** ($\sum_{\tau=0}^{T-t_0} \text{IRF}(\tau)$).
2. **`DO(X[t] = x)` vs. `DO(X[*] = x)`:**
   Permite distinguir entre un **pulso transitorio** en un periodo concreto (*"hacer una promoción puntual la semana 3"*) y un **cambio permanente de régimen** (*"bajar la tarifa de forma sostenida a partir de la semana 3"*).
3. **`SOLVE` Temporal:**
   Soporta optimizar trayectorias abiertas $X_{1:T}$, pasos de control deslizante MPC o parámetros globales $\theta$ de políticas reactivas.
4. **`PROVE Φ` Temporal — *Bounded Model Checking* (BMC) y *k-Inducción*:**
   Para propiedades de seguridad sobre máquinas de estados `D3` y acumuladores `D1` (p. ej., $\Phi \equiv \forall t: \neg(\text{cuenta\_bloqueada}_t \land \text{retiro\_autorizado}_t)$):
   * **BMC en horizonte $T$:** Desenrolla la relación de transición $T$ pasos en SMT. Si halla una traza violadora $(x_1, \dots, x_k)$, emite **`FALSIFIED(trace_1..k)`** (contraejemplo exacto paso a paso). Si es insatisfacible hasta $T$, garantiza seguridad acotada en $T$ pasos.
   * **$k$-Inducción (Horizonte Ilimitado):** Intenta demostrar el caso base y el paso inductivo en SMT; solo si ambos resultan `UNSAT` se otorga el sello **`PROVEN` para todo $t \in \mathbb{N}$**, manteniendo coherencia absoluta con las reglas de veracidad de [`02_FRONTERA_IMPERATIVA_Y_CONTRATOS.md`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/02_FRONTERA_IMPERATIVA_Y_CONTRATOS.md).

---

## 10. Extracción desde Código Real y Encaje con la Frontera Causal

¿Cómo reconoce el compilador de ingesta que en un repositorio existe estructura temporal sin obligar al usuario a reescribir su código? Siguiendo las tres dimensiones desacopladas en [`02_FRONTERA_IMPERATIVA_Y_CONTRATOS.md`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/02_FRONTERA_IMPERATIVA_Y_CONTRATOS.md):

1. **Bucles Temporales Explícitos en Python (`for t in range(T):`):**
   Cuando `T` es una constante o parámetro de horizonte fijo y el cuerpo actualiza variables acumuladoras (`inv = inv + lot[t] - demand[t]`), el compilador extrae la recurrencia simbólica como ecuación de caja blanca (`White-box`) con procedencia `@extracted` en lugar de desenrollar código repetido ciego. (Si el número de iteraciones depende de una condición numérica sobre variables continuas `while err > 1e-5`, se clasifica como punto fijo implícito o caja gris `@contract` evaluada mediante solver numérico o subrogado en región de confianza).
2. **Operadores de Desplazamiento y Ventana en `pandas` y `SQL`:**
   * En `pandas`: `.shift(k)`, `.diff(k)`, `.cumsum()`, `.cumprod()`, `.rolling(w)` se traducen directamente a aristas con retardo $\ell = k$ o leyes acumulativas `D1` (`White-box`, certificado `PROVEN`).
   * En `SQL`: funciones analíticas `LAG(col, k) OVER (ORDER BY fecha)` y `SUM(col) OVER (ORDER BY fecha ROWS UNBOUNDED PRECEDING)` se transpilan exactamente a la misma representación temporal unificada con procedencia `@extracted`.
3. **Clases con Estado Purificado (Mecanismo `@contract`):**
   Cuando un objeto de dominio (`Wallet`, `InventoryPool`, `LoanAccount`) expone métodos de paso (`step()`, `apply_month()`) purificados como $s_{\text{out}} = \tilde{f}(s_{\text{in}}, x)$, encadenar sus llamadas define automáticamente la transición $S_t = \tilde{f}(S_{t-1}, X_t)$ con la garantía de certificación (`PROVEN` o `SAMPLED_VERIFIED`) heredada del contrato de $\tilde{f}$.

---

## 11. Escalabilidad Computacional en Horizontes Largos

Desenrollar ingenuamente un grafo de $|V|$ nodos durante $T$ periodos produce $|V_T| = O(T \cdot |V|)$ nodos. Para que `ActorDB` siga respondiendo en milisegundos cuando $T$ crece:

1. **Diferenciación Inversa en el Tiempo (*Backpropagation Through Time* — BPTT):**
   Dado que las aristas solo conectan $t-\ell \to t$, la matriz Jacobiana de $\mathcal{G}_T$ es **dispersa por bloques banda**. Calcular todas las sensibilidades exactas $\frac{\partial J_{1:T}}{\partial X_t}$ para todo $t \in \{1, \dots, T\}$ requiere una sola pasada hacia atrás de coste lineal **$O(T \cdot |E|)$**, jamás cuadrático ni cúbico en $T$.
2. **Estructura Separable por Etapas en Solvers:**
   En problemas `LP`/`MILP`/`QP` dinámicos, la matriz de restricciones posee estructura escalonada por bloques (*staircase structure*), que solvers modernos como HiGHS e interior-point explotan en tiempo casi lineal respecto a $T$. Para horizontes muy largos con no linealidades suaves, el horizonte deslizante MPC acota el tamaño de la ventana activa $H \ll T$.

---

## 12. Puentes Directos hacia las Problemáticas P3 y P4

Cerrar conceptualmente **P2** despeja de golpe dos conexiones clave para los siguientes eslabones:

1. **Hacia P3 (Cardinalidad y Actores Indexados):**
   Observemos la notación de plantilla que emergió naturalmente aquí: `lot[t]`, `inv[t]`, `sum(t: ...)`, `for all t`. **El tiempo discreto $t \in \{1, \dots, T\}$ no es más que una dimensión índice ordenada.** En **P3** generalizaremos exactamente este mismo mecanismo de plantillas hacia índices relacionales arbitrarios (`sku`, `tienda`, `segmento`, `ruta`), evitando tener dos sistemas distintos para indexar el tiempo e indexar entidades.
2. **Hacia P4 (Fusión de Lógica Booleana/SMT con Optimización Continua):**
   Las máquinas de estados `D3` ($V_{\text{logic}}$ que cambian de fase a lo largo del tiempo, como un préstamo que pasa a `MOROSO` si el saldo continuo cae por debajo de un umbral durante 3 meses seguidos) acoplan íntimamente condiciones discretas con variables continuas, preparando el terreno para la orquestación híbrida de **P4**.

---

## 13. Mapeo de Vacíos Estructurales (`GAP-01` a `GAP-09`)

El tratamiento de tiempo, estado y lazos de retroalimentación en este documento se vincula directamente a la taxonomía formalizada en [`docs/baseline/03_mapa_de_vacios.md`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/baseline/03_mapa_de_vacios.md):

* **Frontera Zero-to-One Inexplorada:**
  - `GAP-01` (Ingesta Estática Multi-Artefacto a SCM): Extracción de bucles secuenciales, recurrencias de estado y cláusulas temporales desde AST de Python, SQL (`LAG`/`OVER`) y Pandas (`shift`/`rolling`) sin requerir reescritura manual.
  - `GAP-02` (Certificación de Fidelidad Determinista $\le 10^{-6}$): Certificación de gradientes y derivadas intertemporales mediante BPTT exacto $O(T \cdot |E|)$ a lo largo del DAG expandido $\mathcal{G}_T$.
  - `GAP-03` (Copiloto Causal Resistente a Presiones y Premisas Falsas): El oráculo MCP rechaza presiones de optimización miope que violen la no-anticipatividad o destruyan inventarios/capital al final del horizonte ($t=T$).
  - `GAP-04` (Memoria Epistémica Estructurada sin Re-escaneo $O(N)$): Persistencia estructurada de estados intertemporales $S_t$ e historiales de trayectorias, habilitando consultas MPC paso a paso sin re-ingestar el contexto completo.
* **Prototipos Parciales y Tecnologías Maduras Integradas:**
  - `GAP-05` (Text-to-LP / MINLP Dinámico): Generación automática de restricciones de balance intertemporal con estructura escalonada por bloques (*staircase*) para solvers HiGHS/IPOPT.
  - `GAP-06` (Descubrimiento Causal Temporal): Resolución de ambigüedades direccionales aprovechando la flecha del tiempo ($t-k \to t$, $\Lambda(C) \ge 1$) para eliminar simetrías espurias.
  - `GAP-07` a `GAP-09` (Verificación Formal SMT/BMC, AST/CFG y Linaje SQL): Aplicación de Bounded Model Checking (BMC) y $k$-inducción formal para probar invariantes de seguridad sobre acumuladores y máquinas de estados intertemporales.
