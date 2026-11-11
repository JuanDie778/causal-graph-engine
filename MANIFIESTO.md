# MANIFIESTO Y TESIS ESTRATÉGICA DEL PROYECTO
## ActorDB & Causal Graph Engine (CGE)
**Propósito:** Redefinición conceptual, tesis de innovación y base matemática para una nueva tecnología de estructuración lógica y causal de los actores de un sistema.

---

## 1. El Punto de Partida y la Crítica al Status Quo

### 1.1 El Status Quo Actual ("IA + Notebook")
Hoy en día, el flujo de trabajo estándar cuando un analista o ingeniero quiere entender un sistema, diagnosticar un problema o proponer una mejora consiste en abrir un entorno interactivo (Jupyter Notebook, Cursor, Claude Code) junto a un modelo de lenguaje (LLM), cargar archivos CSV o scripts de Python y pedirle a la IA un análisis descriptivo o exploratorio en cuestión de minutos.

Intentar competir contra "IA + Notebook" haciendo simplemente otro dashboard de análisis descriptivo o un optimizador aislado carece de sentido estratégico. Para crear algo verdaderamente transformador (*Zero to One*), debemos mirar dónde fracasa estructuralmente ese *status quo*:

1. **Razonamiento Efímero y Amnesia Epistémica:** Lo que la IA deduce hoy en un notebook desaparece al cerrar la sesión. La comprensión lógica de cómo funciona el sistema no queda indexada, verificada ni estructurada en el repositorio.
2. **Ceguera Sistémica ante la Lógica Distribuida:** En un proyecto real, las reglas que gobiernan la realidad están dispersas entre código imperativo (`Python`), transformaciones tabulares (`pandas`), vistas y restricciones de bases de datos (`SQL`), documentos normativos (`.md`) y tablas históricas (`CSV`). Un LLM leyendo archivos sueltos o un analista mirando solo un CSV ignora restricciones duras, umbrales o invariantes ocultos en otros módulos.
3. **Confusión entre Correlación Pasiva y Mecanismo Ejecutable:** Las bases de datos tradicionales guardan registros pasados (filas), y las bases de datos vectoriales guardan fragmentos de texto por similitud semántica (embeddings). Ninguna de las dos almacena **la estructura lógica y causal viva de los actores del sistema**.

---

## 2. La Nueva Tesis Central: Una Base de Datos de los Actores y su Lógica

La innovación central de este proyecto consiste en crear una nueva categoría tecnológica: **una Base de Datos de los Actores de un Proyecto (`ActorDB`)**.

A diferencia de una base de datos tradicional enfocada en organizar registros tabulares, esta tecnología se enfoca en **indexar, relacionar, auditar y consultar a los actores que componen la lógica de un repositorio o sistema** (variables, parámetros de decisión, perturbaciones del entorno, estados intermedios, restricciones e invariantes):

* **En un repositorio convencional** solo existe el código fuente y, a lo sumo, una base de datos de registros históricos.
* **Con esta tecnología**, el repositorio cuenta además con un **mapa lógico-causal auditable, explícito y matemáticamente verificable** de todos sus actores y de cómo interactúan entre sí.

### 2.1 ¿Por qué cambia las reglas del juego para la IA y los Humanos?
Cuando existe una base de datos formal de los actores y sus relaciones lógicas:
1. **Comprensión Instantánea, Objetiva y sin Alucinaciones para la IA:** Un agente de IA ya no tiene que adivinar la lógica del proyecto leyendo miles de líneas de código a ciegas ni improvisar scripts frágiles. Puede consultar directamente la estructura de actores para entender qué variables existen, qué rol juegan, cómo se propagan las causas y qué límites no se pueden violar.
2. **De Análisis Descriptivo a Razonamiento Estructural y Contrafactual:** Permite responder preguntas que una base de datos de registros no puede contestar: *¿Qué actores están muertos y no afectan a ningún objetivo? ¿Dónde se contradice una regla documentada con el código real? ¿Qué ocurre exactamente en toda la cadena si alteramos un actor?*

---

## 3. Cimientos Matemáticos y Técnicos Disponibles (El Motor Base en `cge/`)

Para que esta visión no sea un simple diagrama pasivo de cajas y flechas, el repositorio ya cuenta con un núcleo matemático y de verificación funcional en [`cge/`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/cge) que sirve como materia prima reciclable:

### 3.1 Formalización Estructural de Actores
Todo sistema analizado dentro de la frontera de relevancia causal de `ActorDB` se representa bajo la tupla extendida (definida en [**`docs/01_ALCANCE_Y_ONTOLOGIA_DOMINIO.md`**](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/01_ALCANCE_Y_ONTOLOGIA_DOMINIO.md)):

$$\mathcal{M}_{\text{ActorDB}} = \big\langle U,\; X,\; V_{\text{num}},\; V_{\text{logic}},\; \mathcal{F},\; \mathcal{C},\; \mathcal{G} \big\rangle$$

* **$U$ (Actores Exógenos):** Factores del entorno, mercado o contexto fuera del control directo.
* **$X$ (Actores Palanca / Política):** Variables continuas, discretas, booleanas o reglas de umbral sobre las que se puede intervenir.
* **$V_{\text{num}}$ (Actores de Estado Cuantitativo):** Magnitudes intermedias, flujos y acumuladores calculados por el sistema.
* **$V_{\text{logic}}$ (Actores de Estado Lógico / Compuerta):** Estados discretos, banderas de elegibilidad, permisos y transiciones condicionales.
* **$\mathcal{F}$ (Mecanismos y Relaciones):** Leyes funcionales, reglas condicionadas con guardas, contratos de caja gris (`@contract`) o relaciones empíricas (`@empirical`) que conectan a los actores.
* **$\mathcal{C}$ (Invariantes y Contratos Normativos):** Condiciones de conservación, límites físicos o normativos ($g(V) = 0, \; h(V, X) \le 0, \; P(V_{\text{logic}}) \implies Q(V_{\text{logic}})$).
* **$\mathcal{G} = (J, \Phi)$ (Metas Duales del Sistema):** Tanto objetivos cuantitativos escalares ($J \in \mathbb{R}$ a maximizar/minimizar) como propiedades o estados lógicos meta ($\Phi \in \{\text{True}, \text{False}\}$ a alcanzar o garantizar).

### 3.2 Capacidades ya Construidas y Verificadas en el Código Base
1. **Ingesta y Compilación Multi-Fuente ([`cge/ingest.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/cge/ingest.py), [`cge/universal_compiler.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/cge/universal_compiler.py)):** Extracción estática desde AST de Python, linaje de columnas en `pandas`, vistas `SQL`, reglas en documentos Markdown y datos tabulares `CSV`.
2. **Certificación Empírica de Fidelidad ([`cge/fidelity.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/cge/fidelity.py)):** Verificación mediante muestreo determinista (Sobol) y análisis estático de pureza que comprueba que la representación lógica reproduce fielmente la ejecución del código original ($\text{error relativo} \le 10^{-6}$).
3. **Motor de Resolución y Certificados Formales ([`cge/solver/`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/cge/solver/__init__.py)):** Compilación simbólica y resolución matemática (LP, MILP, QP, NLP, MINLP e identificación de subsistemas infactibles IIS) con certificados verificables (`CERTIFIED_OPTIMAL`, `INFEASIBLE_CERTIFIED`, `UNSUPPORTED`).
4. **Auditoría de Coherencia ([`cge/audit.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/cge/audit.py)):** Detección de actores muertos ($T1$), violación de invariantes de conservación ($T2$), discrepancias entre documentación y código ($T3$), óptimos locales miopes ($T4$), discontinuidades ($T5$) y fragilidad ante incertidumbre ($T6$).

---

## 4. Hoja de Ruta Conceptual y Decisiones Fundacionales

El avance conceptual del proyecto se gestiona en el documento vivo [**`docs/HOJA_DE_RUTA_CONCEPTUAL.md`**](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/HOJA_DE_RUTA_CONCEPTUAL.md):

1. **[RESUELTO] Alcance del Dominio y Ontología de Actores ([`docs/01_ALCANCE_Y_ONTOLOGIA_DOMINIO.md`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/01_ALCANCE_Y_ONTOLOGIA_DOMINIO.md)):** Aislamiento del Núcleo de Decisión, Reglas y Estado frente al código de plomería, Test de Admisión de Actores y soporte de Metas Duales $\mathcal{G} = (J, \Phi)$.
2. **[RESUELTO] Problemática P1 — La Frontera de lo Imperativo y Cajas Grises ([`docs/02_FRONTERA_IMPERATIVA_Y_CONTRATOS.md`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/02_FRONTERA_IMPERATIVA_Y_CONTRATOS.md)):** Taxonomía de opacidad `O1`–`O5`, espectro epistémico `L0`–`L5` con degradación honesta, distinción `PROVEN` vs `SAMPLED_VERIFIED`, purificación de estado, corte de efectos externos y contratos `@contract`.
3. **[RESUELTO] Problemática P2 — Tiempo, Estado y Bucles de Retroalimentación ([`docs/03_TIEMPO_ESTADO_Y_CICLOS.md`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/03_TIEMPO_ESTADO_Y_CICLOS.md)):** Clasificación de ciclos por retardo ($\Lambda \ge 1$ dinámico vs $\Lambda = 0$ punto fijo algebraico), plantilla intensional con vista expandida $\mathcal{G}_T$, horizontes con valor terminal/MPC, políticas paramétricas, no-anticipatividad y BMC/$k$-inducción.
4. **[SIGUIENTE] Problemática P3 — Cardinalidad y Actores Indexados (Escalar vs. Tensorial/Relacional):** Cómo generalizar el índice temporal hacia conjuntos de entidades arbitrarias (`precio[sku, tienda]`) con restricciones compartidas sin explosión del mapa lógico.
5. **[PENDIENTE] Problemática P4–P5 y Decisiones 2–4:** Orquestación híbrida SMT + optimización continua, protocolo de ingesta asistida por LLM y gramática/persistencia del lenguaje declarativo.
