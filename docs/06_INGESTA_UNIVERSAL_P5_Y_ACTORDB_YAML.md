# 06. INGESTA UNIVERSAL EN 2 CAPAS, PROTOCOLO EPISTÉMICO P5 Y CONTRATO `actordb.yaml`

> **Estado:** Implementado y verificado en código (`cge/ingest_adapters/generic_adapter.py`, `cge/mcp/oracle.py`, `cge/diff/engine.py`)  
> **Cierra:** Problemática **P5** y Decisiones Fundacionales **2 (Opción C)**, **3 (Opción A)** y **4 (Opción A Evolucionada)** de [`HOJA_DE_RUTA_CONCEPTUAL.md`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/docs/HOJA_DE_RUTA_CONCEPTUAL.md).

---

## 1. Motivación Arquitectónica

Un motor causal verificable no puede depender de escribir un adaptador artesanal por cada nuevo repositorio Python, ni tampoco puede adivinar mediante heurísticas opacas si un parámetro de función (`def evaluate_plan(seat_price=49.0, vat_rate=0.21)`) representa una **palanca de decisión modificable** ($X$, `control`) o una **condición exógena del entorno** ($U$, `exogenous`). Adivinar `vat_rate` como `control` fabricaría políticas ilegales o imposibles, violando el invariante fundacional de **Cero Alucinaciones**.

ActorDB resuelve esta tensión mediante una **Arquitectura de Ingesta Universal en 2 Capas**:

1. **Capa 1 — Zero-Config (`GenericPackageAdapter`)**: Extracción estática AST cross-módulo inmediata, sin requerir ningún archivo adicional en el paquete objetivo.
2. **Capa 2 — Co-evolución Declarativa (`actordb.yaml` / `actordb.yml` / `actordb.json`)**: Sidecar declarativo minimalista versionado en Git que fija roles semánticos, cotas operativas e invariantes normativas, validado determinísticamente contra el AST real.

---

## 2. Capa 1: Compilación Estática Zero-Config (`GenericPackageAdapter`)

Cuando `ActorDBOracle` (`cge/mcp/`) o `CausalDiffEngine` (`cge/diff/`) reciben un paquete Python multi-módulo no registrado en `ADAPTER_REGISTRY` (o una ruta con esquema `generic://<directorio>[:submodelo]`), `GenericPackageAdapter` ejecuta:

1. **Aislamiento de Plomería y Tests**: Excluye automáticamente `tests/`, `test_*.py`, `*_test.py`, `conftest.py` y `setup.py`.
2. **Resolución Cross-Módulo Estática**:
   - Evalúa literales y constantes globales exportadas e importadas entre submódulos (`from .tiers import BASE_PLATFORM_FEE as BPF`).
   - Registra funciones auxiliares cross-módulo (con soporte de alias `import ... as ...`) en `_helper_ast_map` y en `_SafeSimulationASTInterpreter`.
   - Extrae valores nominales desde defaults de firmas Python y campos de `@dataclass` / clases de configuración, e infiere tipos discretos/booleanos desde anotaciones AST (`int -> discrete`, `bool -> boolean`, `float -> continuous`).
3. **Extracción de Guardas y Precondiciones**:
   - Deduce cotas duras `[min, max]` desde `if <cond>: raise ValueError(...)` y `assert <cond>`, propagando también guardas transitivas cuando un parámetro de entrada se pasa directamente a un helper interno.
   - Traduce comparaciones relacionales multi-variable (`if infra_cost > budget_cap: raise ValueError(...)` o `assert gross_margin >= 0.65`) en restricciones duras (`Constraint(type="hard")`).
4. **Doble Nivel de Compilación Verificable**:
   - **Nivel `L0/L1` Simbólico (`seal = "PROVEN"`)**: Extrae asignaciones intermedias SSA como nodos `state` verificados e inyecta helpers cross-módulo en expresiones SymPy exactas (soportando retornos escalares y diccionarios literales de métricas `return {"gross_mrr": ..., "net_mrr": ...}`).
   - **Nivel `L2` Simulación AST Acotada (`seal = "SAMPLED_VERIFIED"`)**: Cuando una función emplea bucles imperativos o llamadas `math.*` no reducibles simbólicamente, construye un callable determinista mediante `_SafeSimulationASTInterpreter` con presupuesto estricto `max_loop_steps = 50_000` (cero uso de `exec`/`eval`), preservando análisis estático de parámetros no leídos y variables locales muertas (`T1-A`).
5. **Descubrimiento Automático de Submodelos**: Identifica las funciones públicas raíz (no consumidas exclusivamente como helpers internos) y las expone como submodelos direccionables (`<paquete>:<submodelo>`).

---

## 3. Resolución Formal de la Problemática P5 (Roles de Hojas sin Alucinación)

En **Capa 1 (Zero-Config)**, todo parámetro de entrada nace conservadoramente con categoría `exogenous`, registrándose en:

```json
{
  "roles_inferred": false,
  "unclassified_leaves": ["seat_price", "seats", "vat_rate"]
}
```

### Comportamiento por Primitiva MCP en Modo Zero-Config
- **`cge_trace`**: Operativo de inmediato (`PROVEN` / `SAMPLED_VERIFIED`). Calcula linaje completo, derivadas totales exactas respecto a cualquier `source_id` y audita nodos muertos `T1-A` / `T1-B`.
- **`cge_do`**: Operativo de inmediato. Ejecuta cirugía contrafactual de Pearl $\text{do}(I)$ sobre cualquier subconjunto de hojas o estados intermedios.
- **`cge_prove`**: Operativo de inmediato. Verifica o refuta invariantes sobre las cotas deducidas de guardas AST o `domain_bounds`.
- **`cge_solve`**:
  - Si se invoca **sin** declarar qué hojas son controlables (`modifiable_controls=None` y sin `actordb.yaml`), **rechaza honestamente** con `status = "UNSUPPORTED"` y `error_code = "AMBIGUOUS_LEAF_ROLES_P5"`, devolviendo `candidate_leaves` para que el agente o el usuario elijan explícitamente las palancas.
  - Si el agente o analista pasa `modifiable_controls=["seat_price"]` y opcionalmente `control_bounds={"seat_price": [15.0, 120.0]}`, el oráculo promueve **solo para esa consulta** dichas hojas verificadas a `control` (intersecando siempre las cotas solicitadas con cualquier guarda dura del AST, jamás relajándola) y emite una receta óptima certificada (`CERTIFIED_OPTIMAL` / `BEST_FOUND`) o certificado `INFEASIBLE_CERTIFIED` con `Minimal IIS`.

---

## 4. Capa 2: Gramática Minimalista del Contrato `actordb.yaml`

Para consolidar las reglas de negocio en CI/CD (**Decisión 2C: Co-evolución Híbrida**, **Decisión 3A: Minimalismo Radical**, **Decisión 4A: Archivos Declarativos Planos**), la raíz del paquete puede incluir un archivo `actordb.yaml` (o `.yml` / `.json`):

```yaml
version: "1.0"
system_id: "saas_unit_economics"
default_submodel: "evaluate_plan"

submodels:
  evaluate_plan:
    entrypoint: "evaluator.py:evaluate_plan"
    target: "net_mrr"
    sense: "max"
    controls:
      seat_price:
        bounds: [15.0, 120.0]
      discount_rate:
        bounds: [0.0, 0.30]
    exogenous:
      seats:
        bounds: [5, 500]
      vat_rate:
        bounds: [0.0, 0.25]
    constraints:
      c_margin_floor:
        type: "hard"
        expression: "gross_margin_pct >= 0.65"
        description: "Piso normativo de margen bruto SaaS"
```

### Validación Determinista AST $\leftrightarrow$ Contrato (`CONTRACT_AST_MISMATCH`)
Al cargar el paquete, `GenericPackageAdapter` contrasta cada identificador del sidecar contra el AST extraído:
1. `entrypoint` debe existir como función pública real (y coincidir con el archivo si se indicó `archivo.py:funcion`).
2. `target` debe existir en los actores del submodelo.
3. Todo identificador en `controls` y `exogenous` debe existir **y ser un parámetro hoja de entrada** (nunca un estado intermedio calculado, evitando contradicciones estructurales).
4. Ningún actor puede declararse simultáneamente en `controls` y `exogenous`.
5. Toda variable referenciada en `constraints` / `invariants` debe existir en el grafo AST del submodelo.

Cualquier violación aborta en cerrado con `error_code = "CONTRACT_AST_MISMATCH"` detallando `missing_identifiers` y `available_actors`. Asimismo, `cge.diff` audita automáticamente mutaciones conjuntas entre Python y `actordb.yaml` (detectando `HARD_CONSTRAINT_WEAKENED`, `HARD_CONSTRAINT_REMOVED`, `CONTROL_BOUNDS_WEAKENED`, `DEAD_NODE_ADDED` y `POLICY_OPTIMALITY_REGRESSION` a través de todos los submodelos compartidos).
