# HOJA DE RUTA CONCEPTUAL, PROBLEMÁTICAS Y DECISIONES FUNDACIONALES
## Proyecto ActorDB & Causal Logic Engine
**Documento vivo de trabajo**  
*Propósito: Registro formal de la reflexión sobre la nueva tecnología, sus problemáticas críticas y decisiones abiertas para su seguimiento, modificación y descarte/tachado.*

---

## I. Pilares de la Nueva Tecnología (Propuesta Conceptual)

- [ ] **1. Representación Causal-Lógica con Guardas (*Guarded Causal DSL*):**
  Unificar en una misma declaración compacta:
  - Lógica discreta y permisos (estados/guardas para SMT / Z3).
  - Lógica cuantitativa y estocástica (mecanismos continuos, elasticidades, optimización MINLP / CVaR).
  - Lógica relacional indexada (conjuntos de actores parametrizados).

- [x] **2. Separación de Naturalezas Epistémicas y Degradación Honesta (`L0`–`L5`):**
  - Anclajes `@extracted` (`L0`), `@empirical` (`L1`), `@contract` (`L2`), Subrogado con Región de Confianza (`L3`), Oráculo Puro (`L4`), `@boundary` de efectos cortados (`L5`) y leyes `@normative`.
  - Distinción gramatical inquebrantable entre sellos **`PROVEN`** (demostrado simbólicamente) y **`SAMPLED_VERIFIED(N, err)`** (verificado empíricamente sobre $N$ muestras), con propagación por el eslabón más débil $\mathcal{L}_{\text{eff}}(Q) = \min_{v \in \text{Anc}} \mathcal{L}(v)$.

- [ ] **3. Escalabilidad por Zoom Algebraico (Composicionalidad Modular):**
  - Capacidad de condensar algebraicamente subgrafos internos de módulos complejos en "bloques con puertos" para no inundar el contexto de la IA ni al usuario.

- [ ] **4. Lenguaje de Consulta Causal (*Causal Query Language*):**
  - Primitivas formales para agentes de IA y analistas: `TRACE` (rutas e impacto), `DO` (intervenciones de Pearl), `SOLVE` (receta óptima bajo restricciones duras), `AUDIT` (salud lógica e invariantes).

---

## II. Las 5 Problemáticas Críticas a Resolver

> *Nota: Conforme vayamos profundizando y resolviendo cada una, marcaremos `[x]` y documentaremos la solución acordada.*

### [x] P1. La Frontera de lo Imperativo (Cajas Grises y Efectos Secundarios)
- **El desafío:** Gran parte del código real no son funciones matemáticas puras; contiene clases mutables, llamadas de red, ORMs y librerías externas en C.
- **Solución adoptada:** Taxonomía de opacidad `O1`–`O5`, espectro graduado de conocimiento `L0`–`L5` con regla de degradación honesta del eslabón más débil, purificación funcional de estado mutable (`O2`), corte causal de efectos externos hacia actores exógenos `U` (`L5`), protocolo híbrido de contratos verificables `@contract` (`L2`), subrogados con región de confianza explícita (`L3`) y principios de ejecución segura en *sandbox* con presupuesto $B_{\max}$.
- **Estado:** **Resuelta conceptualmente** en [**`docs/02_FRONTERA_IMPERATIVA_Y_CONTRATOS.md`**](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/02_FRONTERA_IMPERATIVA_Y_CONTRATOS.md).

---

### [x] P2. Tiempo, Estado y Bucles de Retroalimentación (Non-DAGs)
- **El desafío:** Muchos sistemas tienen memoria intertemporal ($S_{t+1} = f(S_t, X_t)$), colas, inventarios encadenados o bucles causales cerrados que violan la aciclicidad simple.
- **Solución adoptada:** Clasificación formal de todo ciclo por su retardo total $\Lambda(C) = \sum \ell(e)$ (**ciclos dinámicos** $\Lambda \ge 1$ expandibles a un DAG exacto $\mathcal{G}_T$ con plantilla intensional compacta, frente a **lazos algebraicos** $\Lambda = 0$ resueltos como punto fijo con auditoría de existencia/unicidad); soporte de horizontes con valor terminal, control deslizante MPC, planes abiertos y políticas paramétricas extraídas del código; invariante obligatoria de no-anticipatividad; y tiempo continuo/DES fuera del alcance inicial.
- **Estado:** **Resuelta conceptualmente** en [**`docs/03_TIEMPO_ESTADO_Y_CICLOS.md`**](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/03_TIEMPO_ESTADO_Y_CICLOS.md).

---

### [ ] P3. Cardinalidad y Actores Indexados (Escalar vs. Tensorial/Relacional)
- **El desafío:** Pasar de `precio` (1 escalar) a `precio[producto, tienda]` (decenas de miles de instancias de un mismo tipo de actor sujetas a restricciones compartidas de presupuesto o logística).
- **Riesgo:** Explosión del tamaño del grafo si se expande ingenuamente a miles de nodos individuales en el archivo declarativo.
- **Frente de investigación:** Diseñar la semántica de "clases o plantillas de actores" con dimensiones indexadas sin multiplicar la verbosidad del lenguaje.
- **Estado:** *Pendiente de profundización.*

---

### [ ] P4. Fusión de Lógica Booleana/Simbólica con Optimización Continua
- **El desafío:** Los solvers lógicos (SMT) sufren con operaciones continuas no lineales trascendentes ($\sqrt{x}, \log x$), y los solvers continuos (SLSQP/HiGHS) sufren con lógica booleana arbitraria.
- **Riesgo:** Cuello de botella computacional si no se diseña una orquestación híbrida limpia.
- **Frente de investigación:** Arquitectura en dos pasos: reducción de espacio de búsqueda por SMT / regímenes activos $\to$ optimización continua condicionada.
- **Estado:** *Pendiente de profundización.*

---

### [ ] P5. Protocolo de Ingesta Asistida por LLM sin Falsa Confianza
- **El desafío:** El código fuente crudo muchas veces no declara explícitamente si un parámetro es una palanca controlable ($X$) o una variable exógena ($U$). El LLM puede inferir la semántica de negocio, pero no debe inventar relaciones no respaldadas.
- **Riesgo:** Reintroducir alucinaciones en el mapa lógico.
- **Frente de investigación:** Establecer un protocolo estricto de *"Propuesta Semántica del LLM $\to$ Verificación Determinista en Código y Datos $\to$ Validación/Aprobación del Usuario"*.
- **Estado:** *Pendiente de profundización.*

---

## III. Decisiones Fundacionales Abiertas (Bifurcaciones de Diseño)

### [x] Decisión 1: Alcance del Dominio Inicial
- ~~**Opción A (Amplitud Total):** Intentar modelar cualquier repositorio de software (drivers, frontend React, infraestructura).~~
- **Opción B Evolucionada (Especialización en el Núcleo Causal-Lógico con Aislamiento de Plomería):** `ActorDB` se conecta a repositorios reales, ignora automáticamente la capa de presentación/transporte/plomería y modela el **Núcleo de Decisión, Reglas y Estado** bajo la tupla extendida $\mathcal{M}_{\text{ActorDB}} = \langle U, X, V_{\text{num}}, V_{\text{logic}}, \mathcal{F}, \mathcal{C}, \mathcal{G} \rangle$, soportando tanto metas cuantitativas ($J \in \mathbb{R}$) como metas lógicas de elegibilidad/estado ($\Phi \in \{\text{True}, \text{False}\}$).
- **Elección acordada:** **Opción B Evolucionada** — Documentada formalmente en [**`docs/01_ALCANCE_Y_ONTOLOGIA_DOMINIO.md`**](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/01_ALCANCE_Y_ONTOLOGIA_DOMINIO.md).

---

### [ ] Decisión 2: Fuente de la Verdad Primaria
- **Opción A (Code-First):** El código existente manda; la tecnología genera una radiografía derivada que se sincroniza continuamente.
- **Opción B (Spec-First):** El archivo declarativo es el contrato rector del negocio que audita y condiciona el código.
- **Opción C (Co-evolución Híbrida):** Inicia como Code-First mediante ingesta automática, pero permite enriquecerlo con reglas `@normative` que luego gobiernan las pruebas de integración en CI/CD.
- **Elección acordada:** *Por definir.*

---

### [ ] Decisión 3: Nivel de Expresividad de la Gramática
- **Opción A (Minimalismo Radical):** Muy pocas palabras clave (`actor`, `rule`, `invariant`, `goal`) fáciles de memorizar para un humano y con poquísimos tokens para un LLM.
- **Opción B (Riqueza Tipada Completa):** Tipado exhaustivo, contratos formales estilo Eiffel/Rust, soporte nativo de distribuciones estocásticas complejas y lógica temporal de primer orden.
- **Elección acordada:** *Por definir.*

---

### [ ] Decisión 4: Arquitectura de Persistencia y Runtime
- **Opción A (Archivos Declarativos Planos):** Archivos `.actor` en el repositorio, leídos al vuelo por el CLI o herramientas de agentes (estilo Prisma/Terraform).
- **Opción B (Motor Residente / Daemon Local):** Un servicio en segundo plano que indexa y vigila cambios en tiempo real en los archivos del proyecto.
- **Elección acordada:** *Por definir.*

---

## Historial de Revisiones
- **2026-11-11:** Creación del documento tras la refundación conceptual y limpieza del repositorio.
- **2026-11-11:** Resolución de la **Decisión Fundacional 1** en [`docs/01_ALCANCE_Y_ONTOLOGIA_DOMINIO.md`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/01_ALCANCE_Y_ONTOLOGIA_DOMINIO.md) (Aislamiento del Núcleo de Decisión/Reglas, Test de Admisión de Actores y Metas Duales $\mathcal{G} = (J, \Phi)$).
- **2026-11-11:** Resolución de la **Problemática P1** en [`docs/02_FRONTERA_IMPERATIVA_Y_CONTRATOS.md`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/02_FRONTERA_IMPERATIVA_Y_CONTRATOS.md) (Taxonomía de opacidad `O1`–`O5`, niveles epistémicos `L0`–`L5`, degradación honesta, distinción `PROVEN` vs `SAMPLED_VERIFIED`, purificación de estado y contratos `@contract`).
- **2026-11-11:** Resolución de la **Problemática P2** en [`docs/03_TIEMPO_ESTADO_Y_CICLOS.md`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/03_TIEMPO_ESTADO_Y_CICLOS.md) (Clasificación de ciclos por retardo $\Lambda \ge 1$ vs lazos algebraicos $\Lambda = 0$, plantilla temporal compacta + DAG expandido $\mathcal{G}_T$, horizontes MPC/valor terminal, políticas paramétricas, no-anticipatividad y BMC/$k$-inducción).
