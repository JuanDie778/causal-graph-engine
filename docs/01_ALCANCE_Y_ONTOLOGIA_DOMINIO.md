# ESPECIFICACIÓN FUNDACIONAL 01: ALCANCE DEL DOMINIO Y ONTOLOGÍA DE ACTORES
## ActorDB & Causal Logic Engine
**Estado:** Borrador de Trabajo v0 (Draft)  
**Propósito:** Delimitar con precisión matemática y arquitectónica qué sistemas modela la tecnología, qué constituye un "Actor" dentro de un repositorio real, qué queda fuera de la frontera causal y cómo se unifican los sistemas cuantitativos con los sistemas de reglas lógicas. Ancla la cuña *Zero-to-One* en **Causal Diff para CI/CD** y el **Oráculo Causal MCP para agentes de IA**.

---

## 0. Marco Empírico de Referencia y Cuña *Zero-to-One*

Este documento conceptual formaliza la ontología del motor integrando los hallazgos de la auditoría de estado del arte y el baseline empírico congelado en `docs/baseline/`:
* [**Estado del Arte y Análisis Competitivo (`docs/baseline/01_estado_del_arte.md`)**](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/baseline/01_estado_del_arte.md): Demuestra la ausencia global de herramientas que unifiquen AST de código, inferencia causal de Pearl y optimización matemática MINLP.
* [**Flujo Actual y Baseline Empírico (`docs/baseline/02_flujo_actual_y_baseline_empirico.md`)**](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/baseline/02_flujo_actual_y_baseline_empirico.md): Audita el comportamiento de los contendientes B0, B1, B2 y B3 bajo la batería estandarizada Q1..Q8.
* [**Mapa de Vacíos Tecnológicos y Prior Art (`docs/baseline/03_mapa_de_vacios.md`)**](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/baseline/03_mapa_de_vacios.md): Cataloga los 9 vacíos del estado del arte, destacando la frontera *Zero-to-One* (`GAP-01` a `GAP-04`) frente a problemas explorados (`GAP-05`, `GAP-06`) y tecnologías maduras (`GAP-07` a `GAP-09`).
* [**Marco Formal de Métricas y Grader Determinista (`docs/baseline/04_marco_de_metricas.md`)**](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/baseline/04_marco_de_metricas.md): Establece las 4 Reglas Anti-Goodhart y la penalización estricta por falsa confianza.

### La Cuña Comercial y Operativa *Zero-to-One*
ActorDB no se dispersa como un analizador genérico de código fuente ni como un catálogo pasivo de metadatos. Se enfoca en dos casos de uso de valor inmediato:
1. **Causal Diff para CI/CD:** Motor de análisis estático causal ejecutado en pull requests. Compara el grafo causal antes y después de cada commit ($\Delta \mathcal{M}$), detectando de forma automatizada:
   - Roturas silenciosas de invariantes contables o normativas ($\mathcal{C}$).
   - Introducción de nodos muertos ($T1$) o divergencias documentación-código ($T3$).
   - Regresiones en el espacio factible o degradación del óptimo prescriptivo ($J^*$).
2. **Oráculo Causal MCP para Agentes de IA:** Servidor compatible con el estándar *Model Context Protocol* (MCP) que expone herramientas formales deterministas (`TRACE`, `DO`, `SOLVE`, `PROVE`). Elimina las alucinaciones aritméticas de los LLMs (como se evidencia en las trampas de B0 y B1) al dotar a agentes autónomos de una fuente de verdad matemática ejecutable sin re-escaneos $O(N)$ de contexto (`GAP-04`).

---

## 1. La Tesis de Alcance: Aislar el "Núcleo de Decisión, Reglas y Estado"

El primer dilema fundacional de cualquier tecnología que pretenda indexar la lógica de un proyecto es elegir entre dos extremos peligrosos:
* **El extremo de la irrelevancia práctica:** Modelar únicamente scripts académicos de 10 líneas con fórmulas matemáticas puras, quedando incapacitados para tocar un repositorio de producción real.
* **El extremo del ruido sintáctico:** Intentar convertir *cada línea y variable* de un repositorio de software (componentes visuales de React, routers HTTP, middlewares de logs, formateadores de strings, serializadores JSON) en nodos de la base de datos, produciendo una maraña inútil de miles de variables de plomería sin valor causal ni de negocio.

### 1.1 La Postura Adoptada
**ActorDB puede conectarse a cualquier repositorio de software, datos o negocio, pero aplica un Principio de Aislamiento del Núcleo Causal (*Decision & Policy Core Isolation*):**

```
┌────────────────────────────────────────────────────────────────────────────┐
│                   REPOSITORIO REAL DE UN PROYECTO                          │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ CAPA DE PLOMERÍA Y PRESENTACIÓN (Excluida por ActorDB)               │  │
│  │ • Renderizado UI (HTML/CSS/React), CLI spinners, formato de texto    │  │
│  │ • Transporte HTTP/gRPC, serialización JSON, cursores de conexión     │  │
│  │ • Telemetría pasiva (logging, métricas de consola sin retroacción)   │  │
│  └──────────────────────────────────┬───────────────────────────────────┘  │
│                                     │ Frontera de Relevancia Causal        │
│  ┌──────────────────────────────────▼───────────────────────────────────┐  │
│  │ NÚCLEO DE DECISIÓN, REGLAS Y ESTADO (Indexado y Certificado)         │  │
│  │ • Actores Exógenos (U): Entorno, demanda, shocks, atributos de input │  │
│  │ • Actores Palanca (X): Precios, umbrales, cupos, políticas, flags    │  │
│  │ • Estados Cuantitativos (V_num): Costes, flujos, saldos, métricas    │  │
│  │ • Estados Lógicos (V_logic): Elegibilidad, permisos, estados de paso │  │
│  │ • Invariantes (C): Leyes de conservación, límites normativos, SLAs   │  │
│  │ • Metas del Sistema (J / Φ): Valor a optimizar o estado a garantizar │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

De este modo, la tecnología no compite con un AST Call-Graph de ingeniería de software (como Tree-sitter o Sourcegraph, que indexan quién llama a qué función de interfaz), sino que **extrae la física económica, operativa y lógica que vive dentro del repositorio**.

---

## 2. El Test de Admisión de un Actor (*Actor Admission Test*)

Para que una entidad de un repositorio (variable, constante de configuración, columna de tabla/DataFrame, vista SQL, retorno de función o regla documental) sea admitida como un **Actor** en `ActorDB`, debe superar al menos uno de los siguientes **4 criterios de relevancia causal**:

1. **Criterio de Gobernabilidad (Es una Palanca $X$):** ¿Representa un parámetro, umbral, tarifa, cupo o bandera de política que un humano o un algoritmo puede decidir modificar para alterar el comportamiento del sistema?
2. **Criterio de Exogeneidad Relevante (Es un Contexto $U$):** ¿Representa una condición del mundo exterior o del sujeto evaluado (ej. demanda de mercado, volatilidad, ingresos declarados por un solicitante, latencia de un proveedor) que alimenta las reglas o ecuaciones del sistema sin poder ser controlada directamente?
3. **Criterio de Transformación de Estado (Es un Estado $V_{\text{num}}$ o $V_{\text{logic}}$):** ¿Es el resultado de aplicar una ley de negocio, una transformación de flujo o una compuerta lógica que determina decisiones posteriores, costes, permisos o métricas finales?
4. **Criterio de Frontera o Juicio (Es una Invariante $\mathcal{C}$ o Meta $\mathcal{G}$):** ¿Define un límite inquebrantable (presupuesto, balance suma cero, requisito regulatorio) o una medida del éxito/validez del sistema?

> **Regla de Exclusión Automática (Plomería):** Toda variable temporal que solo transporta, formatea, serializa o registra datos sin intervenir en la estructura causal hacia ningún estado del dominio ($\mathcal{P}(v \to \mathcal{G}) = \emptyset$ y fuera de toda invariante $\mathcal{C}$) se descarta como plomería de transporte, o bien se señala como **Actor Muerto ($T1$)** si consume cómputo de dominio sin efecto.

---

## 3. Ontología Formal de Actores v1 ($\mathcal{M}_{\text{ActorDB}}$)

Ampliando la tupla original del proyecto para abarcar tanto **sistemas matemáticos de optimización** como **motores lógicos de reglas, permisos y estados**, todo proyecto en `ActorDB` queda definido por la estructura séptupla:

$$\mathcal{M}_{\text{ActorDB}} = \big\langle U,\; X,\; V_{\text{num}},\; V_{\text{logic}},\; \mathcal{F},\; \mathcal{C},\; \mathcal{G} \big\rangle$$

### 3.1 Taxonomía de los 6 Tipos de Actores

| Símbolo | Rol en `ActorDB` | Tipo de Dominio Matemático | Ejemplos en Repositorios Reales |
| :--- | :--- | :--- | :--- |
| **$U$** | **Actor Exógeno (`exogenous`)** | Variables aleatorias $\sim P(U)$, series empíricas o atributos de entrada fijos por instancia. | `base_demand`, `market_interest_rate`, `applicant_debt_ratio`, `provider_latency_ms`. |
| **$X$** | **Actor Palanca / Política (`lever`)** | Continuo $[a, b] \subset \mathbb{R}$, Discreto $\{a, \dots, b\} \subset \mathbb{Z}$, Booleano $\{0, 1\}$ o Categórico finito. | `price`, `order_qty`, `min_credit_score_threshold`, `enable_express_routing`, `fee_tier`. |
| **$V_{\text{num}}$** | **Actor de Estado Cuantitativo (`state_num`)** | Escalares o vectores en $\mathbb{R}^n \times \mathbb{Z}^m$ calculados por mecanismos funcionales. | `revenue`, `holding_cost`, `expected_loss`, `queue_wait_time`, `net_margin`. |
| **$V_{\text{logic}}$** | **Actor de Estado Lógico / Compuerta (`state_logic`)** | Booleano $\{\text{True}, \text{False}\}$ o Enumerado de estados finitos $\{s_1, \dots, s_k\}$. | `is_kyc_verified`, `credit_approved`, `sla_breached`, `discount_regime_active`. |
| **$\mathcal{C}$** | **Actor Invariante / Contrato (`invariant`)** | Igualdades $g(V)=0$, desigualdades $h(V,X)\le 0$ y cláusulas lógicas $P(V_{\text{logic}}) \implies Q(V_{\text{logic}})$. | `credits - debits == 0`, `total_spend <= max_budget`, `payout_sent => is_kyc_verified`. |
| **$\mathcal{G}$** | **Actor Meta (`goal`: $J$ o $\Phi$)** | **Dual:** Objetivo escalar $J \in \mathbb{R}$ ($\max / \min$) **o** Meta lógica $\Phi \in \{\text{True}, \text{False}\}$ (alcanzar / garantizar). | $J = \text{profit}$ (maximizar beneficio) o $\Phi = (\text{credit\_approved} \land \text{risk} \le 0.02)$ (encontrar condiciones para aprobar). |

### 3.2 El Salto Conceptual: Metas Duales $\mathcal{G} = (J, \Phi)$
Al incorporar actores de estado lógico ($V_{\text{logic}}$) y metas lógicas ($\Phi$) junto a los objetivos numéricos ($J$), `ActorDB` puede responder dos grandes familias de preguntas con el mismo motor:

1. **Preguntas de Optimización Cuantitativa (sobre $J$):**
   * *"¿Qué combinación de palancas $X$ maximiza el beneficio $J$ sin violar ninguna invariante $\mathcal{C}$?"*
   * *"¿Cuál es la sensibilidad $\frac{\partial J}{\partial X}$ y qué restricción tiene mayor precio sombra KKT?"*
2. **Preguntas de Alcanzabilidad, Contrafactuales y Verificación Lógica (sobre $\Phi$):**
   * **Contrafactual de Cambio Mínimo (*Recourse*):** *"Un usuario o pedido fue rechazado ($\Phi = \text{False}$); ¿cuál es el cambio mínimo en las palancas o atributos $\Delta X$ para que pase a ser aprobado ($\Phi = \text{True}$) respetando todas las normas $\mathcal{C}$?"*
   * **Verificación Formal de Seguridad:** *"¿Existe algún escenario posible del entorno $U \in \Omega_U$ donde se autorice una salida de fondos sin que se cumpla la compuerta de verificación ($\neg \text{verified} \land \text{payout} > 0$)?"*

---

## 4. Los 4 Arquetipos de Sistemas Objetivo

Esta definición de dominio posiciona a `ActorDB` como la tecnología nativa para **cuatro arquetipos de proyectos** donde el enfoque tradicional ("IA + Notebook" o tests unitarios simples) es insuficiente:

### Arquetipo I: Sistemas Operativos, Financieros y de Pricing Algorítmico
* **Qué son:** Repositorios que calculan precios, tarifas, comisiones, presupuestos de adquisición, lotes de inventario, asignación de recursos o carteras.
* **Qué aporta `ActorDB`:** Extrae las ecuaciones exactas, detecta umbrales discontinuos (economías de escala, tramos fiscales), cuantifica la pérdida por incertidumbre (Brecha de Jensen / CVaR) y calcula recetas óptimas certificadas contra el código fuente.

### Arquetipo II: Motores de Reglas, Elegibilidad, Riesgo y Compliance
* **Qué son:** Sistemas de decisión que evalúan decenas de condiciones encadenadas (aprobación de créditos, fraude, suscripción de seguros, permisos empresariales, asignación de niveles/tiers).
* **Qué aporta `ActorDB`:** Modela las compuertas $V_{\text{logic}}$ y los umbrales $X$, detecta reglas contradictorias o ramas lógicas inalcanzables (código muerto lógico), verifica que ninguna ruta viole regulaciones `@normative` y calcula contrafactuales exactos (*"qué umbral o condición bloqueó el estado deseado"*).

### Arquetipo III: Pipelines Híbridos de Datos, SQL y Modelos Predictivos
* **Qué son:** Proyectos donde los datos fluyen desde tablas/CSVs a través de vistas SQL y transformaciones en `pandas` hasta alimentar modelos de demanda, churn o scoring.
* **Qué aporta `ActorDB`:** Unifica en un solo grafo el linaje determinista de las columnas transformadas en SQL/Pandas con las relaciones empíricas (`@empirical`) aprendidas de los datos, alertando cuando hay variables confusoras o cuando una columna calculada no tiene impacto en la decisión final.

### Arquetipo IV: Sistemas Transaccionales con Leyes de Conservación
* **Qué son:** Libros mayores contables (*ledgers*), motores de pagos, liquidaciones, gestión de colas o balances de energía/masa donde nada puede crearse ni destruirse por error de código.
* **Qué aporta `ActorDB`:** Audita simbólicamente todas las ramas de ejecución frente a invariantes de conservación ($g(V) = 0$), cazando fugas de redondeo, comisiones omitidas en ramas condicionales o discrepancias entre la documentación operativa y el código.

---

## 5. Lo que Queda Explícitamente Fuera del Alcance (No-Objetivos)

Para mantener el foco afilado (*Zero to One*), `ActorDB` **NO** intenta modelar:
1. **Diseño visual, animaciones o estado puramente estético de interfaces (DOM/CSS).**
2. **Sistemas operativos de bajo nivel, drivers de hardware o protocolos de red binarios.**
3. **Generación de texto creativo libre sin estructura de estado o decisión.**

---

## 6. Conexión Directa con el Siguiente Paso: Problemática P1 (Cajas Grises `@contract`)

Ahora que sabemos exactamente **qué tipo de sistemas y actores viven dentro de `ActorDB`**, emerge de forma inmediata el primer reto técnico duro de la hoja de ruta (**Problemática P1**):

En cualquiera de los 4 Arquetipos anteriores, una parte de los actores tendrá ecuaciones limpias extraíbles en caja blanca (`@extracted`), pero otra parte estará implementada mediante **código imperativo complejo, llamadas a objetos mutables, librerías externas o modelos de caja negra**. Definir cómo `ActorDB` encapsula, interroga y certifica esos bloques mediante el anclaje **`@contract`** es el siguiente eslabón natural de diseño.

---

## 7. Mapeo de Vacíos Estructurales (`GAP-01` a `GAP-09`)

La ontología aquí definida aborda directamente la taxonomía formalizada en [`docs/baseline/03_mapa_de_vacios.md`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/baseline/03_mapa_de_vacios.md):

* **Frontera Zero-to-One Inexplorada:**
  - `GAP-01` (Ingesta Estática Multi-Artefacto a SCM): Define la extracción de $\mathcal{M}_{\text{ActorDB}}$ unificando Python AST, SQL, CSV y documentación Markdown.
  - `GAP-02` (Certificación de Fidelidad Determinista $\le 10^{-6}$): Establece los certificados de correspondencia entre código fuente y ecuaciones en $\mathcal{F}$.
  - `GAP-03` (Copiloto Causal Resistente a Presiones y Premisas Falsas): Base del oráculo MCP resistente a sicofancia e invariante ante presiones en $\mathcal{C}$.
  - `GAP-04` (Memoria Epistémica Estructurada sin Re-escaneo $O(N)$): Indexación en grafo dirigida a evitar la saturación de contexto de los LLMs.
* **Prototipos Parciales y Tecnologías Maduras Integradas:**
  - `GAP-05` (Text-to-LP): Resuelto mediante compilación causal a solvers MINLP sin alucinación de restricciones.
  - `GAP-06` (Descubrimiento Causal textual): Complementado por análisis de código estático verificable.
  - `GAP-07` a `GAP-09` (Verificación SMT, AST/CFG y Linaje SQL): Reutilizados como capas fundacionales consolidadas para alimentar la frontera semántica.

