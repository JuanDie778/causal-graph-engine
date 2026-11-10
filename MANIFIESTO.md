# MANIFIESTO Y ESPECIFICACIÓN TÉCNICA DEL PROYECTO
## Decision Intelligence & Causal Graph Engine (CGE)
**Fecha de formalización:** Noviembre 2026  
**Propósito:** Especificación de arquitectura, diseño lógico-matemático y directivas para agentes autónomos.

---

## 1. Declaración de Visión e Intuición Original

### 1.1 La Intuición Base
La intuición fundamental de este proyecto nace el 05-11-26 con una premisa clara:
> *Construir un mapa conceptual interactivo de variables, datos, causas y proyecciones para transformar el análisis de datos en un proceso intuitivo, trazable y matemáticamente objetivo. Una arquitectura donde sea visible cómo se relacionan entre sí las variables, los problemas a los que se enfrenta el sistema, el flujo de causas-consecuencias y el estado objetivo deseado. A partir de esa definición del objetivo, el sistema debe armar el camino óptimo para que la realidad concreta alcance la solución cumpliendo con todas las exigencias impuestas.*

El sistema debe operar como un **"Google Maps multidimensional"**:
* Un mapa de carreteras donde los nodos son variables, datos o estados lógicos.
* Las aristas son relaciones causales, dependencias funcionales y reglas de negocio.
* El tráfico, los cortes viales o los peajes son las restricciones duras y los cuellos de botella del sistema.
* La búsqueda de ruta es un algoritmo de optimización que encuentra la secuencia mínima de intervenciones para ir del estado actual al estado objetivo.

### 1.2 El Salto Cualitativo
A diferencia de los grafos asociativos estáticos (tipo Obsidian o pizarras visuales), este sistema no mapea "conceptos que se parecen", sino **sistemas dinámicos que se ejecutan**. Traduce código, datos tabulares y reglas operativas a una estructura lógica formal que puede ser comprendida, auditada y manipulada tanto por personas como por agentes de inteligencia artificial.

---

## 2. Finalidades y Alcance del Sistema

El sistema implementa un ciclo de vida cerrado de cuatro etapas operativas:

1. **Ingesta Universal y Estructuración Agéntica:**
   El usuario entrega una carpeta o repositorio arbitrario con código fuente, bases de datos, CSVs, logs o documentos explicativos. Un workflow automatizado gobernado por un *skill* riguroso procesa, limpia, valida y traduce todo ese volumen desordenado a un grafo causal unificado, sin requerir etiquetado manual previo.
2. **Visualización y Exploración Multidimensional:**
   Interfaz web interactiva dotada de filtros, capas de abstracción y paneles de detalle donde el usuario inspecciona:
   * La topología global del sistema (grafo causal).
   * La evidencia cuantitativa en cada nodo (distribución de datos, código fuente exacto, métricas de dispersión).
   * Los puntos de fricción e invariantes violadas expuestos visualmente como alertas críticas.
3. **Copiloto Agéntico con Comprensión de Contexto Nativa:**
   Un agente de IA conectado bidireccionalmente al grafo. El agente no razona "de memoria" ni alucina respuestas: utiliza el grafo como memoria de trabajo, consulta subgrafos específicos, ejecuta consultas deterministas y refleja sus conclusiones iluminando rutas y nodos directamente en la pantalla del usuario.
4. **Motor Autónomo de Resolución y Formulación de Problemas:**
   A partir de un objetivo definido por el usuario (o diagnosticado por el propio sistema), el motor formaliza la función de pérdida y ejecuta algoritmos de búsqueda y optimización matemática para entregar una receta concreta de intervención ("qué variables mover, en qué orden y con qué magnitud").

---

## 3. Fundamentación Lógica y Matemática

Para garantizar objetividad y trazabilidad absoluta, todo proyecto ingerido se compila en una tupla estructural formal:

$$\mathcal{M} = \langle U, X, V, \mathcal{F}, \mathcal{C}, J \rangle$$

### 3.1 Taxonomía Formal de Nodos
* **$U$ (Variables Exógenas / Entorno):** Factores externos fuera del control del operador. Se modelan como variables aleatorias con distribución $P(U)$ o series temporales estocásticas (ej. volatilidad de mercados, latencias de red, comportamiento de terceros).
* **$X$ (Variables de Control / Palancas de Decisión):** Parámetros, umbrales de configuración o decisiones de diseño sobre los que existe control directo ($x \in \Omega_X$).
* **$V$ (Variables Endógenas de Estado):** Métricas intermedias y resultados operacionales del sistema ($v \in \mathbb{R}^n \times \mathbb{Z}^m$).
* **$\mathcal{F}$ (Mecanismos de Propagación Causal):** El conjunto de funciones $v_i = f_i(\text{padres}(v_i), u_i, x_i)$ que dictan cómo una alteración se transmite a través de la red.
* **$\mathcal{C}$ (Invariantes y Restricciones de Frontera):** Condiciones que el sistema no puede violar bajo ninguna circunstancia:
  $$g(V) = 0 \quad \text{(Igualdades: conservación de masa, dinero o flujo)}$$
  $$h(V, X) \le 0 \quad \text{(Desigualdades: límites de riesgo, presupuestos, SLA)}$$
* **$J$ (Función de Valor / Objetivo):** La métrica escalar que se busca maximizar o minimizar.

### 3.2 Auditoría de Sentido y Diagnóstico de Coherencia
Para evaluar si una solución o código "tiene sentido", el motor aplica tres comprobaciones matemáticas:
1. **Comprobación de Invariantes:** Verifica si las ramas lógicas del sistema rompen leyes de conservación básicas (por ejemplo, transacciones en un ledger contable donde entradas y salidas no sumen cero).
2. **Jacobiano del Grafo y Análisis de Sensibilidad:** Calcula la derivada total o sensibilidad discreta de los nodos meta respecto a las palancas:
   $$\frac{\partial J}{\partial x_k} = \sum_{\text{rutas } p \in (x_k \to J)} \prod_{(i \to j) \in p} \frac{\partial v_j}{\partial v_i}$$
   * Revela **nodos muertos** ($\frac{\partial J}{\partial v} = 0$), descartando complejidad innecesaria en el código.
   * Identifica **nodos de alta fragilidad**, donde pequeñas fluctuaciones en $U$ desestabilizan el resultado.
3. **Detección de Mínimos Locales:** Evalúa si las decisiones aisladas en módulos específicos del código perjudican la optimización global del sistema.

---

## 4. Arquitectura de Resolución en Dos Capas (Opción C)

Para mantener una experiencia visual fluida sin comprometer la precisión matemática, la navegación sobre el grafo se divide en dos capas:
Entrada de Usuario / Objetivo J
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ CAPA 1: Heurística y Enrutamiento Rápido               │
  │ • Modelo: Grafo dirigido ponderado por sensibilidades. │
  │ • Motores: Algoritmos A*, Dijkstra, Flujo Máximo.      │
  │ • Función: Identificar las Top-K cadenas causales      │
  │   prometedoras entre X y el objetivo J en < 50 ms.     │
  └───────────────────────────┬────────────────────────────┘
                              │ Top-K Subgrafos Candidatos
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ CAPA 2: Motor Computacional Exacto (Validación Dura)   │
  │ • Modelo: Ecuaciones simbólicas, llamadas Python, SMT. │
  │ • Motores: Z3 Prover, SciPy Optimize, OR-Tools.        │
  │ • Función: Simular escenarios reales, verificar        │
  │   invariantes C y computar el valor exacto de ΔX.      │
  └────────────────────────────────────────────────────────┘


  1. **Capa Heurística:** Opera sobre aproximaciones lineales y pesos normalizados $w_{ij} \in [-1, 1]$. Permite al usuario interactuar con grafos de miles de nodos en tiempo real sin latencia computacional.
2. **Capa Simbólica / Exacta:** Se ejecuta exclusivamente sobre las rutas preseleccionadas en la Capa 1, resolviendo el problema inverso de optimización con restricciones duras:
   $$\min_{\Delta X} \quad \| V(X_0 + \Delta X, U) - V^* \|_2^2 + \lambda \|\Delta X\|_1 \quad \text{sujeto a } \mathcal{C}(V, X) \le 0$$

---

## 5. Esquema de Datos Canónico (`schema.json`)

El estado del sistema se serializa en un formato estándar, desacoplando el motor de análisis de la interfaz gráfica:

```json
{
  "$schema": "[http://json-schema.org/draft-07/schema#](http://json-schema.org/draft-07/schema#)",
  "system_id": "cge_instance_001",
  "version": "1.0.0",
  "nodes": [
    {
      "id": "node_unique_id",
      "label": "Nombre Descriptivo de la Variable",
      "category": "exogenous | control | state | target",
      "data_type": "continuous | discrete | boolean | categorical",
      "source_reference": {
        "file": "ruta/al/archivo.py",
        "entity": "nombre_de_funcion_o_clase",
        "line_range": [10, 45]
      },
      "current_value": null,
      "bounds": {
        "min": null,
        "max": null
      }
    }
  ],
  "edges": [
    {
      "source": "source_node_id",
      "target": "target_node_id",
      "type": "exact_code | discrete_rule | empirical_data",
      "sensitivity_weight": 0.85,
      "confidence": 1.0,
      "exact_mechanism": {
        "engine": "python_callable | sympy_expr | smt_clause",
        "definition": "target = source * 1.05 - fee"
      }
    }
  ],
  "constraints": [
    {
      "id": "c_balance_zero",
      "type": "hard",
      "expression": "sum(credits) - sum(debits) == 0",
      "description": "Invariante contable de suma cero en libro mayor"
    }
  ]
}

6. Pipeline de Ingesta y Protocolo del SkillPara garantizar determinismo y neutralidad, el procesamiento de cualquier directorio o repositorio sigue un protocolo secuencial estricto:Paso 1: Extracción Estática (Cero Alucinación):Inspección del código mediante analizadores sintácticos abstractos (ast nativo en Python).Mapeo de llamadas entre módulos, firmas de funciones, argumentos de entrada y variables devueltas.Identificación de esquemas de bases de datos relacionales (.sql, migraciones, modelos ORM).Paso 2: Deducción Causal y Etiquetado Semántico:Clasificación guiada de nodos según la tupla $\mathcal{M}$ ($U$, $X$, $V$).Detección de invariantes implícitas del dominio a partir de la documentación y los casos de prueba existentes.Estimación de sensibilidades iniciales para alimentar la Capa 1 del motor.Paso 3: Validación por Ejecución:Ejecución de las suites de pruebas unitarias y de integración (pytest, scripts de prueba).Captura de trazas de ejecución en tiempo real (sys.settrace) para asociar valores numéricos concretos a los nodos del mapa.Diagnóstico de inconsistencias lógicas entre la intención declarada en la documentación y el comportamiento del código ejecutado.7. Directrices de Construcción para el Agente (Reglas de Oro)Prioridad del Determinismo: Toda relación que pueda ser extraída mediante análisis estático de código, sintaxis AST o regresión numérica sobre datos reales debe resolverse por esas vías. El uso de modelos de lenguaje queda restringido a tareas semánticas de alto nivel y traducción contextual.Inmutabilidad del Estado sin Respaldo en el Grafo: La interfaz de usuario no almacena lógica propia; es un reflejo visual reactivo de schema.json. Cualquier modificación generada en la UI o sugerida por el asistente debe materializarse primero como una mutación validada en el grafo.Trazabilidad Estricta: Ningún nodo o arista puede crearse sin un puntero de origen verificable (file, line_range, column_name o expresión matemática documentada).Optimización con Restricciones Reales: Al formular soluciones, el sistema nunca propondrá rutas de intervención que violen una restricción clasificada como hard en $\mathcal{C}$.