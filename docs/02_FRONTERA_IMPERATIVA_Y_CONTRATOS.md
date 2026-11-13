# ESPECIFICACIÓN FUNDACIONAL 02: LA FRONTERA DE LO IMPERATIVO Y LAS CAJAS GRISES (`@contract`)
## ActorDB & Causal Logic Engine
**Estado:** Borrador de Trabajo v0 (Draft)  
**Propósito:** Definir la teoría, taxonomía y reglas formales mediante las cuales `ActorDB` representa, interroga, optimiza y audita actores cuya implementación en el repositorio real no es reducible a una ecuación algebraica de caja blanca. Ancla la domesticación de código imperativo en la cuña *Zero-to-One*: **Causal Diff para CI/CD** y el **Oráculo Causal MCP para agentes de IA**.

---

## 0. Marco Empírico y Cuña *Zero-to-One*

Este documento técnico formaliza el tratamiento de la frontera imperativa integrando la evidencia recopilada en `docs/baseline/`:
* [**Estado del Arte (`docs/baseline/01_estado_del_arte.md`)**](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/baseline/01_estado_del_arte.md): Documenta la brecha entre herramientas de análisis puramente estático (que rechazan construcciones no cerradas) y frameworks de optimización de caja negra (que descartan la estructura del código).
* [**Baseline Empírico (`docs/baseline/02_flujo_actual_y_baseline_empirico.md`)**](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/baseline/02_flujo_actual_y_baseline_empirico.md): Evidencia cómo contendientes sin frontera imperativa graduada sufren o bien de parálisis binaria (0% en funciones complejas) o de extrapolaciones erróneas.
* [**Mapa de Vacíos (`docs/baseline/03_mapa_de_vacios.md`)**](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/baseline/03_mapa_de_vacios.md): Conecta con `GAP-01` (ingesta estática multi-artefacto), `GAP-02` (certificación de fidelidad $\le 10^{-6}$), `GAP-03` (resistencia a falsas premisas) y `GAP-07` (verificación formal).
* [**Marco de Métricas (`docs/baseline/04_marco_de_metricas.md`)**](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/baseline/04_marco_de_metricas.md): Establece la penalización absoluta por falsa confianza cuando un oráculo opaco afirma optimalidad sin prueba formal.

### Integración en la Cuña *Zero-to-One*
1. **Causal Diff en CI/CD:** Cuando un desarrollador modifica una función imperativa o introduce una librería externa en un pull request, el Causal Diff no se rompe: audita si la modificación preserva el contrato (`@contract`), detecta si se introdujeron efectos secundarios impuros (mutaciones $O2$ o llamadas I/O $O3$) y certifica si las invariantes del sistema continúan garantizadas.
2. **Oráculo Causal MCP para Agentes de IA:** Expone herramientas de consulta desacopladas (`TRACE`, `DO`, `SOLVE`, `PROVE`). El agente de IA puede delegar la simulación o la optimización en el oráculo MCP sabiendo de antemano el nivel de garantía del resultado (`PROVEN` vs. `SAMPLED_VERIFIED`), impidiendo que el agente afirme certeza sobre aproximaciones locales.

---

## 1. El Problema y Diagnóstico del Motor Base

Dentro de los 4 Arquetipos fijados en [`01_ALCANCE_Y_ONTOLOGIA_DOMINIO.md`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/01_ALCANCE_Y_ONTOLOGIA_DOMINIO.md), un repositorio real rara vez contiene únicamente funciones matemáticas puras. Coexisten ecuaciones explícitas con objetos mutables en memoria, llamadas a librerías compiladas (`numpy`, `scipy`, motores actuariales en C), modelos predictivos (`model.predict`), bucles con parada dependiente de datos y consultas a servicios externos.

Al auditar cómo el código base actual (`cge/`) enfrenta esta frontera, aparecen **dos limitaciones opuestas** que debemos superar:

1. **Fragilidad Binaria ("Todo o Nada") en [`cge/fidelity.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/cge/fidelity.py#L251-L341):**
   El analizador estático detecta correctamente construcciones no simbólicas (`UNSUPPORTED_DYNAMIC_LOOP`, `UNSUPPORTED_IO`, `UNSUPPORTED_RECURSION`, `EXTERNAL_STATE_MUTATION`), pero aplica una política binaria: basta que **un solo ancestro** en un grafo de 50 actores contenga un bucle dinámico o una llamada opaca para que toda consulta de resolución devuelva `UNSUPPORTED`. El sistema es epistemológicamente honesto, pero pierde operatividad ante el primer roce con software imperativo real.
2. **Extrapolación Silenciosa de Subrogados en [`cge/universal_compiler.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/cge/universal_compiler.py#L1362-L1437) y [`cge/solver/compile.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/cge/solver/compile.py#L406-L433):**
   `synthesize_blackbox_analytical_surrogate` muestrea una caja cerrada en un entorno local de $\pm 25\%$ alrededor del punto nominal $x_0$, ajusta un polinomio con raíz cuadrada y, si $R^2 \ge 0.99$ localmente, lo inyecta en el compilador simbólico como si fuera válido en **todo** el dominio global $[X_{\min}, X_{\max}]$, sin guardar la región de validez ni distinguir su garantía respecto a una ley exacta.

**Meta de P1:** Sustituir esa dicotomía rígida por una **arquitectura graduada de conocimiento causal**, donde ningún bloque imperativo rompa el mapa completo y ninguna aproximación se disfrace jamás de prueba exacta.

---

## 2. Taxonomía de la Opacidad Imperativa (O1–O5)

No todo código "no simbólico" es opaco por la misma razón. `ActorDB` clasifica los obstáculos imperativos en **cinco categorías disjuntas**, cada una con un mecanismo de domesticación específico:

| Código | Categoría de Opacidad | Ejemplos en Repositorios Reales | Naturaleza Matemática | Tratamiento Nativo en `ActorDB` |
| :--- | :--- | :--- | :--- | :--- |
| **O1** | **Cálculo Puro pero Analíticamente Intratable** | Rutinas matriciales `numpy`/`scipy`, árboles GBDT (`xgboost.predict`), búsquedas binarias internas, funciones especiales en C. | Función determinista $y = f(x)$ bien definida, sin efectos laterales, pero sin expresión cerrada en `SymPy`. | Encapsulación como **Contrato (`L2`)**, **Subrogado con Región de Confianza (`L3`)** u **Oráculo Puro (`L4`)**. |
| **O2** | **Mutación de Estado Interno o Compartido** | Métodos `self.balance -= amt`, cachés mutables, acumuladores en diccionarios o listas pasadas por referencia. | Transición de estado $(s_{\text{in}}, x) \mapsto (s_{\text{out}}, y)$ oculta tras efectos laterales en memoria. | **Purificación Funcional (Elevación de Estado):** transformar los atributos mutados en puertos explícitos de entrada/salida. |
| **O3** | **Efectos e Interacciones con el Exterior (I/O)** | Llamadas HTTP (`requests.post`), lecturas/escrituras en BD transaccional, reloj del sistema (`datetime.now()`), envío de pagos. | Interacción no aislada con el mundo físico fuera del proceso; ejecutarla repetidamente es costoso o destructivo. | **Corte Causal de Frontera (`L5`):** aislar la lectura externa como Actor Exógeno $U$ y prohibir la ejecución de escrituras externas en simulación. |
| **O4** | **No-Determinismo Intrínseco** | `random.random()` sin semilla controlada, condiciones de carrera por concurrencia, UUIDs dinámicos en lógica de decisión. | Mapeo estocástico $Y \sim P(\cdot \mid X)$ en lugar de función determinista. | **Elevación Estocástica:** extraer la fuente aleatoria como un puerto exógeno explícito $U \sim P(U)$ o marcar no-determinismo no acotado. |
| **O5** | **Flujo de Control Dinámico No Acotado** | `while error > tol:`, `for i in range(int(order_qty)):`, recursión dependiente de variables de decisión. | El árbol sintáctico depende del valor numérico de los padres; no admite desenrollado estático único. | **Evaluación como Oráculo Acotado** (con *timeout* y contrato `L2`/`L3`) o formulación iterativa temporal (conectado con **P2**). |

---

## 3. Desacoplamiento de la Frontera: Procedencia, Estructura y Capacidad por Consulta

La versión preliminar utilizaba una escala unidimensional simplificada ($L0$ a $L5$). Sin embargo, la auditoría empírica demuestra que colapsar procedencia, computabilidad y garantías formales en una sola dimensión introduce confusión epistemológica. En `ActorDB`, el conocimiento sobre cualquier actor se estructura de forma **ortogonal en tres dimensiones independientes**:

```
                       ESPACIO DE CONOCIMIENTO EPISTÉMICO
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 1. PROCEDENCIA          │ 2. ESTRUCTURA MECANISMO │ 3. CERTIFICACIÓN   │
  │    (Fuente de Verdad)   │    (Grado de Opacidad)  │    (Sello Emitido) │
  ├─────────────────────────┼─────────────────────────┼────────────────────┤
  │ • @extracted            │ • Caja Blanca Simbólica │ • PROVEN           │
  │   (AST código / SQL)    │ • Contrato (@contract)  │   (Dual SMT / MILP)│
  │ • @empirical            │ • Subrogado Local Trust │ • SAMPLED_VERIFIED │
  │   (Datos CSV / ML)      │ • Oráculo Sandbox Puro  │   (Sobol N, ε<=1e-6│
  │ • @normative            │ • Frontera Cortada I/O  │ • UNSUPPORTED      │
  │   (Reglas / Leyes / SLA)│   (Exógena U)           │   (Sin afirmación) │
  └─────────────────────────┴─────────────────────────┴────────────────────┘
```

### 3.1 Dimensión 1: Procedencia de la Verdad (`Provenance`)
* **`@extracted`:** Mecanismo o relación derivada estáticamente de los artefactos de código fuente (`Python AST`, vistas `SQL`, expresiones `SSA`). Su fidelidad se verifica deterministamente frente al intérprete.
* **`@empirical`:** Ley o relación estimada a partir de observaciones (`CSVs`, series temporales, modelos `predict()`), portando explícitamente su matriz de covarianza, $R^2$ y región muestreada.
* **`@normative`:** Invariante o regla inyectada desde documentación técnica (`Markdown`), especificación contractual o regulación legal (ej. ley de presupuesto, invariante contable de partida doble).

### 3.2 Dimensión 2: Estructura Computacional del Mecanismo
* **Caja Blanca Simbólica:** Expresión matemática o lógica cerrada manipulable por `SymPy`/`Z3` ($\text{max\_rel\_error} \le 10^{-6}$).
* **Caja Gris con Contrato (`@contract`):** Código imperativo cuyas fórmulas internas no están cerradas, pero cuyos puertos tipados y propiedades geométricas (monotonía, cotas de imagen, Lipschitz, convexidad) han sido declarados y certificados.
* **Caja Gris con Subrogado Local (`Trust-Region`):** Aproximación diferenciable $\hat{f}(x)$ válida exclusivamente dentro de una región de confianza acotada $\mathcal{R}_v = \{x : \|x - x_0\|_W \le \Delta_v\}$ con error residual $\epsilon_v$.
* **Caja Negra Oráculo:** Código determinista ejecutable exclusivamente como caja cerrada en un *sandbox* seguro con presupuesto finito de evaluaciones $B_{\max}$.
* **Frontera Cortada (Efecto Externo):** Código con I/O o mutación externa irreversible. Se aísla cortando la dependencia y modelando el valor como un Actor Exógeno $U$.

### 3.3 Dimensión 3: Sellos de Certificación (`PROVEN` vs. `SAMPLED_VERIFIED`)
Para evitar incurrir en **Falsa Confianza** (infracción crítica penada con veto según la Regla Anti-Goodhart 4 de `docs/baseline/04_marco_de_metricas.md`), el motor separa rígidamente:
* **`PROVEN` (Demostrado Formalmente):** Garantía universal válida para todo el dominio $\forall x \in \Omega$ obtenida por deducción simbólica, análisis de intervalos cerrado, optimalidad dual global (MILP Branch and Bound con gap $\le \text{gap\_tol}$) o insatisfacibilidad lógica (`UNSAT` en SMT).
* **`SAMPLED_VERIFIED(N, ε)` (Verificado por Muestreo Determinista):** Garantía empírica obtenida tras evaluar $N$ puntos deterministas (muestreo Quasi-Monte Carlo de Sobol, esquinas del dominio y puntos singulares) sobre el código real en sandbox, constatando cero violaciones y error residual $\le \epsilon$.

---

## 4. Evaluación de Capacidad Operativa por Tipo de Consulta

La capacidad de `ActorDB` no es un número escalar: un subgrafo con código imperativo puede ser 100% capaz para responder una traza de linaje causal o una intervención contrafactual puntual, pero requerir algoritmos heurísticos para optimización prescriptiva o emitir un certificado muestreado en verificación.

Por ello, la gobernanza del motor se evalúa vectorialmente según las **cuatro primitivas formales de consulta**:

```
                              TIPO DE CONSULTA
         ┌────────────┬────────────┬─────────────┬─────────────┐
         │   TRACE    │     DO     │    SOLVE    │    PROVE    │
         │  (Linaje)  │ (What-If)  │ (Prescrip.) │(Invariantes)│
         └────────────┴────────────┴─────────────┴─────────────┘
```

1. **`TRACE(target)` (Linaje Causal y Nodos Muertos):**
   - Responde: ¿Qué actores componen la cadena de causas dirigida hacia `target`? ¿Qué variables son nodos muertos ($T1$)?
   - Operabilidad: Requiere únicamente el grafo de dependencias de puertos. Funciona con certeza (`PROVEN`) incluso en presencia de oráculos opacos, siempre que sus entradas y salidas estén tipadas.
2. **`DO(X=x, [interventions])` (Simulación Contrafactual e Intervención de Pearl):**
   - Responde: ¿Cuál es el estado del sistema si forzamos intervenidamente $X=x$ o mutilamos un nodo intermedio $do(V_i = v)$?
   - Operabilidad: Ejecución determinista hacia adelante en orden topológico evaluando código real en sandbox. Funciona con certeza exacta (`PROVEN` o evaluación de código real) sobre cajas blancas, contratos y oráculos.
3. **`SOLVE(target, sense, constraints)` (Optimización Prescriptiva):**
   - Responde: ¿Qué configuración $X^*$ maximiza el objetivo respetando todas las invariantes $\mathcal{C}$?
   - Operabilidad:
     * Si todo el subgrafo es Caja Blanca $\to$ `CERTIFIED_OPTIMAL` (`PROVEN` vía MILP/MINLP Branch and Bound).
     * Si contiene contratos `@contract` $\to$ `BEST_FOUND` guiado por monotonía/cotas, verificado obligatoriamente en código real (`constraint_violation == 0.0`).
     * Si contiene subrogados $\to$ Optimización por Región de Confianza (*Trust-Region*) iterativa con validación en código.
     * Si contiene oráculos puros $\to$ Optimización libre de derivadas (Nelder-Mead / Bayesiana) acotada por presupuesto $B_{\max}$.
4. **`PROVE(invariant)` (Verificación Formal de Seguridad e IIS):**
   - Responde: ¿Se garantiza que la invariante $\mathcal{C}$ se cumple $\forall U, X$? Si es infactible, ¿cuál es el subconjunto mínimo de restricciones irreductible (IIS)?
   - Operabilidad:
     * En Cajas Blancas $\to$ `PROVEN` (`UNSAT`) o `COUNTEREXAMPLE(x*)` (`SAT`).
     * En Cajas Grises / Oráculos $\to$ Búsqueda adversaria guiada de contraejemplos. Si encuentra una violación en el código real emite `FALSIFIED(x*)` (contraejemplo exacto comprobado); si no la encuentra tras $N$ evaluaciones emite `SAMPLED_VERIFIED(N, ε)`, jamás `PROVEN`.

---

## 5. Matriz de Capacidades Desacoplada: Consultas $\times$ Estructura del Mecanismo

| Primitiva de Consulta | Caja Blanca Simbólica (`@extracted`) | Contrato Estructural (`@contract`) | Subrogado Local (`Trust-Region`) | Oráculo Opaco (Sandbox) |
| :--- | :--- | :--- | :--- | :--- |
| **`TRACE`** *(topología y linaje)* | Exacto (`PROVEN`) | Exacto por puertos declarados | Exacto por puertos declarados | Exacto por puertos declarados |
| **`DO(X=x)`** *(simulación puntual)* | Evaluación exacta en código/IR | Exacta (ejecuta código real en sandbox + valida contrato) | Exacta (ejecuta código real, no el subrogado, para el valor final) | Exacta (ejecuta código real en sandbox) |
| **Sensibilidad $\frac{\partial \mathcal{G}}{\partial X}$** | Derivada total / Clarke analítica exacta | Signo garantizado y/o cotas de pendiente $[m_-, m_+]$ | Gradiente analítico del subrogado dentro de $\mathcal{R}_v$ | Diferencias finitas adaptativas locales (`SAMPLED`) |
| **`SOLVE`** *(optimización)* | `CERTIFIED_OPTIMAL` (MILP / KKT global con cota dual) | `BEST_FOUND` guiado por monotonía/cotas + validación en código | Optimización *Trust-Region* iterativa + re-evaluación exacta en código | Búsqueda directa / Bayesiana con presupuesto $B_{\max}$ |
| **`PROVE Φ`** *(invariantes / IIS)* | `PROVEN` (`UNSAT`) o `FALSIFIED(x*)` vía SMT / intervalos | `PROVEN` solo si las cláusulas son demostrables; si no, búsqueda adversaria | No demuestra; busca contraejemplos guiados por gradiente | Búsqueda adversaria (`FALSIFIED(x*)` exacto o `SAMPLED_VERIFIED(N)`) |

---

## 6. Las 4 Técnicas para Domesticar Código Imperativo

### 6.1 Purificación Funcional del Estado Mutable (Resolución de O2)
En programación orientada a objetos es habitual escribir mutaciones *in-place*:
```python
class Wallet:
    def apply_charge(self, amount: float, fee_rate: float) -> None:
        self.balance -= amount * (1.0 + fee_rate)
```
Un analizador ingenuo rechaza `apply_charge` porque devuelve `None` y muta `self.balance`. En `ActorDB`, toda mutación acotada sobre estructuras propias del dominio se somete a **Purificación SSA de Estado**:
1. Cada atributo relevante mutado (`self.balance`) se desdobla en dos puertos explícitos: `balance_in` (padre causal) y `balance_out` (actor resultante).
2. El método impuro se reinterpreta funcionalmente como $(s_{\text{out}}, y) = \tilde{f}(s_{\text{in}}, x)$, permitiendo en muchos casos que un bloque aparentemente imperativo (O2) ascienda directamente a **Caja Blanca `L0`** (como ocurre con las invariantes contables `Ledger`), o en casos complejos a un contrato limpio `L2` sin efectos ocultos.
*(Nota: cuando esta transición se repite cíclicamente a lo largo del tiempo $s_{t+1} = \tilde{f}(s_t, x_t)$, conecta directamente con la **Problemática P2**).*

### 6.2 Corte Causal de Efectos Externos (Resolución de O3 $\to$ Nivel `L5`)
Cuando una función del núcleo de negocio mezcla una consulta I/O con lógica de decisión (ej. lee el tipo de cambio actual por HTTP y calcula el margen):
1. `ActorDB` coloca una **Frontera de Corte (`@boundary`)** en la llamada externa.
2. El valor devuelto por el exterior se registra como un **Actor Exógeno $U$** (con un valor nominal observado o una distribución de incertidumbre $P(U)$).
3. Las operaciones de escritura externa (enviar un correo, ejecutar un cobro en Stripe, hacer `INSERT` en producción) quedan catalogadas como **Efectos Sumidero**: se registra la condición lógica $V_{\text{logic}}$ que dispara el efecto (para poder auditar *"¿cuándo se cobraría al cliente?"*), pero la llamada física queda desconectada en el sandbox.

### 6.3 Contratos Estructurales `@contract` (Nivel `L2`) y Protocolo Híbrido
Un `@contract` permite capturar la **geometría cualitativa** de una caja gris sin necesitar su fórmula exacta. Las 5 propiedades primitivas de mayor valor para los motores de `ActorDB` son:

1. **Cotas de Imagen (`bounds`):** $f(x) \in [a, b]$ para todo $x$ admisible (permite propagar aritmética de intervalos y acotar Big-M en MILP).
2. **Monotonía Direccional (`monotone`):** $\frac{\partial f}{\partial x_i} \ge 0$ (`increasing`) o $\le 0$ (`decreasing`). Si el coste es monótono creciente con el riesgo y un modelo ML de scoring tiene contrato de monotonía, el solver puede podar regiones enteras sin abrir la red neuronal.
3. **Cota de Sensibilidad / Lipschitz (`lipschitz`):** $|f(x) - f(x')| \le L \|x - x'\|$, crucial para acotar cuánto puede variar la salida entre dos puntos muestreados.
4. **Curvatura (`convex` / `concave`):** Garantiza que cualquier óptimo local hallado con gradientes numéricos sobre el oráculo sea óptimo global.
5. **Conservación Relacional (`conserves`):** $\sum_k y_k = \sum_m x_m$ (ej. un repartidor algorítmico complejo que asigna presupuesto entre canales sin crear ni destruir dinero).

**Protocolo Híbrido de Definición de Contratos (Inferencia $\to$ Confirmación $\to$ Vigilancia):**
* **Paso 1 (Inferencia):** El motor sondea el bloque opaco con secuencias Sobol y pares direccionales, proponiendo contratos candidatos (*"En $N=128$ pares, `score` resultó monótono decreciente respecto a `debt_ratio` y acotado en $[0, 1]$"*).
* **Paso 2 (Confirmación):** El humano o el agente revisor acepta la propiedad como parte del contrato del actor (quedando marcada como `SAMPLED_VERIFIED`, o `PROVEN` si proviene de una restricción estructural del modelo, como `sigmoid` $\in (0,1)$ o árboles monótonos restringidos).
* **Paso 3 (Vigilancia Continua):** En cada cambio de código, el motor vuelve a someter el contrato a pruebas adversarias; si una actualización del código rompe la monotonía prometida, salta una alerta inmediata de ruptura de contrato.

### 6.4 Subrogados con Región de Confianza Explícita (Nivel `L3`)
Para corregir la extrapolación ciega de subrogados locales, todo actor `L3` almacena la tupla $(\hat{f}_v, x_0, \Delta_v, \epsilon_v)$:
* Durante una consulta `SOLVE`, el optimizador nunca confía en $\hat{f}_v(x)$ fuera de la bola $\|x - x_0\|_W \le \Delta_v$.
* Si el paso óptimo cae en el borde de $\mathcal{R}_v$, el motor sigue un ciclo clásico de **Optimización por Región de Confianza (*Trust-Region*)**: evalúa el código real en el nuevo punto candidato $x^{(k+1)}$, mide la razón de acuerdo $\rho_k = \frac{J_{\text{real}}(x^{(k+1)}) - J_{\text{real}}(x^{(k)})}{\hat{J}(x^{(k+1)}) - \hat{J}(x^{(k)})}$, recentra el subrogado si el acuerdo es alto ($\rho_k > 0.75$) o contrae $\Delta_v$ si el subrogado diverge.
* Toda receta final se certifica evaluando obligatoriamente el código real (`constraint_violation_on_code == 0.0`).

---

## 7. Principios de Ejecución Segura de Oráculos (`L2`–`L4`)

Dado que interrogar cajas grises exige ejecutar funciones del repositorio analizado, la arquitectura futura de `ActorDB` deberá respetar tres leyes operativas inquebrantables:

1. **Aislamiento de Ejecución (*Sandbox* sin Red ni Escritura):** Toda evaluación de oráculo corre en un entorno aislado con sistema de archivos en solo lectura, red deshabilitada y límite estricto de tiempo por llamada (*timeout*) y memoria.
2. **Presupuesto Explícito de Evaluaciones ($B_{\max}$):** Un oráculo puede tardar $0.05\text{ ms}$ o $500\text{ ms}$ por llamada. El motor mide la latencia basal en la ingesta y asigna un presupuesto máximo de llamadas por consulta (`eval_budget`), eligiendo métodos parsimoniosos (interpolación cuadrática / optimización bayesiana) cuando el oráculo es costoso.
3. **Prohibición Absoluta de Ejecución en `L5`:** Ningún actor marcado con efectos externos `L5` puede ejecutarse jamás dentro de un bucle de muestreo, auditoría u optimización; debe ser sustituido siempre por valores inyectados en su puerto exógeno $U$.

---

## 8. Conexión con el Siguiente Eslabón: Problemática P2 (Tiempo, Estado y Bucles)

Al resolver **P1** mediante la **Purificación de Estado Mutable** ($(s_{\text{in}}, x) \mapsto (s_{\text{out}}, y)$) y detectar la opacidad por **Flujo Dinámico Iterativo (O5)**, queda tendido el puente exacto hacia la **Problemática Crítica P2**:

¿Qué sucede cuando `balance_out` del instante $t$ vuelve a entrar como `balance_in` en el instante $t+1$, formando un bucle cerrado de retroalimentación temporal (inventarios multi-periodo, retención/churn de cohortes, intereses compuestos, colas o máquinas de estados)? Formalizar cómo `ActorDB` representa el **tiempo, la memoria y los ciclos** sin perder tratabilidad matemática es el siguiente paso natural de la hoja de ruta.

---

## 9. Mapeo de Vacíos Tecnológicos (`GAP-01` a `GAP-09`)

El tratamiento desacoplado de la frontera imperativa responde a los vacíos detallados en [`docs/baseline/03_mapa_de_vacios.md`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/baseline/03_mapa_de_vacios.md):

* **`GAP-01` (Ingesta Estática Multi-Artefacto):** La purificación funcional de estado mutable ($O2$) y el aislamiento de puertos $I/O$ ($O3$) permiten que código imperativo complejo sea asimilado en la tupla estructural canónica sin requerir reescritura manual.
* **`GAP-02` (Certificación de Fidelidad Determinista):** Los contratos `@contract` y subrogados locales portan certificados con cota $\le 10^{-6}$ en secuencias Sobol, garantizando que ninguna aproximación se disfrace de verdad universal.
* **`GAP-03` (Resistencia a Falsas Premisas y Sicofancia):** La regla del eslabón más débil y la distinción estricta `PROVEN` vs. `SAMPLED_VERIFIED` evitan que el oráculo MCP emita respuestas complacientes o relaje restricciones ante consultas manipuladoras.
* **`GAP-04` (Memoria Epistémica Estructurada sin Re-escaneo $O(N)$):** La persistencia de contratos `@contract` e índices de frontera imperativa permite responder a consultas causales iterativas en tiempo $O(1)$ sin re-analizar el AST ni saturar la ventana de contexto de los agentes.
* **`GAP-05` (Compilación Text-to-Optimization / MINLP):** Mapeo de bifurcaciones condicionales y contratos discontinuos a formulaciones lineales enteras mixtas (MILP/Big-M) o penalizaciones suaves tratables en IPOPT/HiGHS.
* **`GAP-06` (Descubrimiento Causal e Inferencia de Mecanismos):** Identificación de aristas de datos y dependencias de control en código opaco mediante análisis estático de variables de asignación única (SSA) y muestreo activo en caja negra.
* **`GAP-07` y `GAP-08` (Verificación Formal SMT y Grafos AST/CFG):** Se integran como sustrato para validar formalmente las cláusulas de contratos, garantizar no-anticipatividad y realizar la purificación de variables de asignación única (SSA).
* **`GAP-09` (Linaje SQL y Procedimientos Almacenados):** Extensión del marco de cajas grises y purificación de estado a consultas SQL complejas, vistas materializadas y transformaciones ETL multi-tabla.

