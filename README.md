# ActorDB — Causal Graph & Logic Engine (CGE)

> **Una Base de Datos de los Actores, Mecanismos e Invariantes Lógico-Causales de un Repositorio Python.**

---

## 🧭 Visión y Propósito

Mientras las bases de datos relacionales almacenan registros históricos (filas) y las bases vectoriales indexan fragmentos de texto por similitud semántica, **ActorDB & CGE** compila el núcleo de decisión y reglas de un proyecto de software en un **modelo estructural causal verificable** $\mathcal{M}_{\text{ActorDB}} = \langle U, X, V_{\text{num}}, V_{\text{logic}}, \mathcal{F}, \mathcal{C}, \mathcal{G} \rangle$ (variables exógenas $U$, palancas de control $X$, estados $V$, mecanismos $\mathcal{F}$, invariantes duras $\mathcal{C}$ y metas $\mathcal{G}$).

El objetivo es dotar tanto a ingenieros como a agentes de IA de un **oráculo causal sin alucinaciones**: capaz de trazar derivadas totales exactas y código muerto, simular intervenciones contrafactuales $\text{do}(I)$, resolver recetas óptimas certificadas o subsistemas infactibles mínimos (`Minimal IIS`), y auditar regresiones semánticas en cada Pull Request. Consulta [`MANIFIESTO.md`](MANIFIESTO.md) para la tesis completa y [`docs/HOJA_DE_RUTA_CONCEPTUAL.md`](docs/HOJA_DE_RUTA_CONCEPTUAL.md) para el estado de las decisiones fundacionales.

---

## ✅ Capacidades Construidas y Verificadas

1. **Oráculo MCP (`cge/mcp/`):** Servidor [Model Context Protocol](https://modelcontextprotocol.io/) (`stdio`, `sse`, `streamable-http`) con caché en memoria $O(1)$ invalidada por firma SHA-256 del árbol de archivos y 4 primitivas formales:
   - `cge_trace`: linaje dirigido, derivadas totales exactas / sensibilidades por ruta y detección de actores muertos (`T1-A` sin camino al objetivo y `T1-B` cancelación algebraica exacta).
   - `cge_do`: intervención contrafactual de Pearl $\text{do}(I)$ mediante cirugía de grafo.
   - `cge_solve`: optimización certificada (`LP`, `MILP`, `QP`, `NLP`, `MINLP`) con estado `CERTIFIED_OPTIMAL` / `INFEASIBLE_CERTIFIED` (`Minimal IIS`) y resistencia explícita al bypass de restricciones (`REJECTED_CONSTRAINT_BYPASS`).
   - `cge_prove`: verificación simbólica/intervalar (`PROVEN`), empírica acotada (`SAMPLED_VERIFIED`) o refutación con contraejemplo exacto (`FALSIFIED`).
2. **Ingesta Universal en 2 Capas y Compuerta Epistémica P5 (`cge/ingest_adapters/`):**
   - **Capa 1 — Zero-Config (`GenericPackageAdapter`):** extrae grafos causales cross-módulo directamente del AST Python (constantes, `@dataclass`, helpers, guardas `assert` / `raise ValueError`, compilación simbólica `L0/L1` y simulación AST segura `L2` con presupuesto de pasos y cero `exec`/`eval`). Por el principio **P5**, toda hoja sin contrato nace como `exogenous` (`AMBIGUOUS_LEAF_ROLES_P5` si se pide optimizar sin declarar `modifiable_controls`).
   - **Capa 2 — Contrato Declarativo (`actordb.yaml`):** sidecar minimalista versionado en Git que declara `controls`, `exogenous`, `bounds` y `constraints`, validado determinísticamente contra el AST (`CONTRACT_AST_MISMATCH`). Ver [`docs/06_INGESTA_UNIVERSAL_P5_Y_ACTORDB_YAML.md`](docs/06_INGESTA_UNIVERSAL_P5_Y_ACTORDB_YAML.md).
3. **Causal Diff para CI/CD (`cge/diff/` + `.github/workflows/causal-diff.yml`):** compara revisiones Git (`--base-ref` vs `--head-ref` / `WORKTREE`) sin mutar el árbol de trabajo, bloqueando Pull Requests que introduzcan regresiones de invariantes, debilitamiento de restricciones duras o nuevos actores muertos. Ver [`docs/04_INTEGRACION_CICD_CAUSAL_DIFF.md`](docs/04_INTEGRACION_CICD_CAUSAL_DIFF.md).
4. **Certificación de Fidelidad y Benchmarks Sellados (`cge/fidelity.py`, `benchmark/`):** verificación empírica Sobol ($\text{error relativo} \le 10^{-6}$), suite `OR-Bench` (20 problemas clásicos) y estudio `baseline_v0` congelado bajo manifiesto SHA-256 de 41 artefactos (`benchmark/baseline_bench/baseline_v0_manifest.json`).

---

## 📦 Instalación

Requiere **Python $\ge$ 3.11** (verificado en Python 3.12 en CI Ubuntu/Windows y Python 3.14 en Windows).

```powershell
# Instalación base del motor, servidor MCP y CLI Causal Diff
python -m pip install -e .

# Instalación completa para desarrollo, explorador UI, tests y linter
python -m pip install -e ".[dev]"
```

> **Nota sobre `mcp<2.0`:** `pyproject.toml` fija `mcp>=1.28,<2.0` porque `cge/mcp/server.py` utiliza `mcp.server.fastmcp.FastMCP`, API perteneciente a la serie `1.x` del SDK oficial de Python.

Tras instalar el paquete quedan disponibles en el entorno los ejecutables `actordb-mcp` y `cge-diff` (además de `python -m cge.mcp` y `python -m cge.diff`).

---

## 🚀 Quickstart (< 5 minutos)

### 1. Conectar el Oráculo MCP en tu IDE o Agente (Antigravity / Claude Code / Cursor)

Tras ejecutar `pip install -e .` en tu entorno Python, añade el servidor en tu `mcp_config.json`:

```json
{
  "mcpServers": {
    "actordb-cge": {
      "command": "actordb-mcp",
      "args": ["--transport", "stdio"]
    }
  }
}
```

*(Alternativa sin instalar scripts en `PATH`: `"command": "python"`, `"args": ["-m", "cge.mcp", "--transport", "stdio"]` con `"env": {"PYTHONPATH": "/ruta/al/repo"}`.)*

### 2. Probar Ingesta Gobernada vs. Zero-Config

El repositorio incluye dos paquetes SaaS multi-módulo listos para consultar desde MCP o Python:

- **[`examples/saas_governed`](examples/saas_governed)** *(Capa 2 — con `actordb.yaml`)*: roles `controls` / `exogenous` e invariantes declarados; `cge_solve(source_path="examples/saas_governed")` devuelve directamente la política óptima certificada (`PROVEN`).
- **[`examples/saas_unit_economics`](examples/saas_unit_economics)** *(Capa 1 — Zero-Config puro)*: sin archivos de configuración; `cge_trace` detecta de inmediato variables muertas y cancelaciones algebraicas (`partner_rebate_rate`), mientras que `cge_solve` aplica la compuerta P5 hasta indicar `modifiable_controls=["seat_price", "annual_discount"]`.

```powershell
python -c "from cge.mcp import ActorDBOracle; o = ActorDBOracle(allow_auto_discovery=False); print(o.solve(source_path='examples/saas_governed')['controls'])"
```

### 3. Ejecutar una Auditoría Causal Diferencial (`cge-diff`)

```powershell
# Comparar la rama o árbol de trabajo actual contra el commit anterior en un paquete gobernado
cge-diff --git-repo . --base-ref HEAD~1 --head-ref WORKTREE --subpath examples/saas_governed --format markdown

# Modo estricto para CI (devuelve exit code 1 si detecta regresiones críticas)
python -m cge.diff --git-repo . --base-ref main --head-ref WORKTREE --subpath golden_case --format json --check
```

### 4. Explorador Visual Local (`golden_case`)

```powershell
python run_ui.py golden_case
```

---

## 🧪 Verificación, Tests e Integridad

En Linux/macOS/WSL puedes usar `make`; en **Windows PowerShell** (donde `make` no suele estar instalado) usa directamente los comandos equivalentes:

| Objetivo | Linux / macOS (`make`) | Windows PowerShell | Duración orientativa |
|---|---|---|---|
| **Suite rápida (`test-fast`)** | `make test-fast` | `python -m pytest tests/unit tests/test_ingest_adapters.py golden_case/tests benchmark -m "not slow" -q` | ~70 s |
| **Tests lentos (stdio/HTTP/git)** | `python -m pytest tests/unit -m slow -q` | `python -m pytest tests/unit -m slow -q` | ~140 s |
| **Aceptación E2E + integridad** | `make test-e2e` | `python -m pytest tests/e2e_acceptance -q` | ~7 min |
| **Suite completa (`test-full`)** | `make test-full` | `python -m pytest -q` | ~11 min (363 tests) |
| **Lint acotado (`ruff`)** | `make lint` | `python -m ruff check cge/diff cge/ingest_adapters cge/mcp tests/unit tests/test_ingest_adapters.py examples run_ui.py` | < 2 s |
| **Candado `baseline_v0`** | `make verify-baseline` | `python benchmark/baseline_bench/generate_manifest.py --verify; python benchmark/baseline_bench/ground_truth.py --verify` | < 5 s |

---

## 📁 Estructura del Repositorio

```text
├── pyproject.toml              # Empaquetado actordb-cge, extras [ui]/[dev], scripts y config de ruff
├── Makefile                    # Atajos reproducibles para CI y entornos Unix
├── MANIFIESTO.md               # Tesis estratégica, formalización M_ActorDB y hoja de ruta resumida
├── README.md                   # Guía de instalación, conexión MCP, CLI y verificación
├── cge/                        # Núcleo matemático, compiladores, MCP y Causal Diff
│   ├── mcp/                    # Servidor MCP (stdio/SSE/HTTP), ActorDBOracle y prover formal/intervalar
│   ├── diff/                   # Motor de Causal Diff estructural/semántico e integración Git read-only
│   ├── ingest_adapters/        # GenericPackageAdapter (Zero-Config + actordb.yaml) y adaptadores de dominio
│   ├── solver/                 # Solvers exactos/numéricos (LP, MILP, QP, NLP, MINLP, Minimal IIS)
│   ├── ingest.py               # Extractor AST clásico (pila híbrida CSV + *_BOUNDS, ej. golden_case)
│   ├── universal_compiler.py   # Transpiladores multi-fuente experimentales (pandas, SQL, Markdown, CSV)
│   ├── fidelity.py             # Certificación empírica de fidelidad (muestreo Sobol)
│   ├── layer1_heuristic.py     # Enrutamiento causal rápido sobre semianillo Max-Product
│   ├── layer2_exact.py         # Interfaz de optimización exacta y precios sombra
│   ├── audit.py                # Diagnóstico de patologías estructurales (T1-T7)
│   ├── copilot.py              # Copiloto determinista sobre CausalIR
│   ├── empirical.py            # Estimación econométrica estructural desde tablas
│   ├── monte_carlo.py          # Simulación estocástica y optimización bajo incertidumbre (CVaR)
│   └── schema.py               # Esquema de grafo canónico (CausalGraph, Node, Edge, Constraint)
├── examples/                   # Paquetes multi-módulo de referencia
│   ├── saas_unit_economics/    # Ejemplo Capa 1 (Zero-Config puro + compuerta P5)
│   └── saas_governed/          # Ejemplo Capa 2 (gobernado por actordb.yaml)
├── docs/                       # Especificaciones conceptuales, operativas y estudios
│   ├── HOJA_DE_RUTA_CONCEPTUAL.md
│   ├── IR_SPEC.md
│   ├── 01_ALCANCE_Y_ONTOLOGIA_DOMINIO.md
│   ├── 02_FRONTERA_IMPERATIVA_Y_CONTRATOS.md
│   ├── 03_TIEMPO_ESTADO_Y_CICLOS.md
│   ├── 04_INTEGRACION_CICD_CAUSAL_DIFF.md
│   ├── 05_CATALOGO_ESCENARIOS_ORACULO_MCP.md
│   ├── 06_INGESTA_UNIVERSAL_P5_Y_ACTORDB_YAML.md
│   ├── 07_HITO_PRUEBA_DE_FUEGO_EXTERNA.md
│   └── baseline/               # Estudio de estado del arte, vacíos y marco de métricas (01..04)
├── .github/workflows/
│   ├── tests.yml               # Lint + integridad baseline + matriz test-fast (Ubuntu/Windows) + test-full
│   └── causal-diff.yml         # Auditoría diferencial automática en Pull Requests
├── golden_case/                # Caso operativo base (sellado en baseline_v0)
├── benchmark/                  # OR-Bench (20 problemas) y estudio sellado baseline_bench (41 SHA-256)
├── tests/                      # Suites unitarias (tests/unit/) y de aceptación E2E (tests/e2e_acceptance/)
└── ui/                         # Explorador visual FastAPI + frontend interactivo
```

---

## ⚠️ Límites Conocidos y Fronteras Actuales

1. **Naturaleza de `B0`–`B2` en `baseline_v0`:** En el estudio sellado `benchmark/baseline_bench/`, los contendientes `B0`–`B2` son arquetipos deterministas offline (reproducibles sin claves API), **no** agentes LLM ejecutados en vivo; además, `B3` empleó 5 adaptadores especializados de dominio. La validación empírica A/B con un agente LLM real sobre repositorios externos no vistos está diseñada como siguiente hito en [`docs/07_HITO_PRUEBA_DE_FUEGO_EXTERNA.md`](docs/07_HITO_PRUEBA_DE_FUEGO_EXTERNA.md).
2. **Doble pila de compilación actual:** `ActorDBOracle` (`cge/mcp/`) y `CausalDiffEngine` (`cge/diff/`) operan de forma nativa sobre `GenericPackageAdapter` + `actordb.yaml` (además de mantener retrocompatibilidad con `ASTIngestPipeline` para `golden_case`). En cambio, `ui/` y `cge/universal_compiler.py` siguen conectados únicamente a la pila clásica y aún no leen `actordb.yaml`.
3. **Fronteras de `GenericPackageAdapter`:** Soporta módulos Python numéricos/lógicos cross-módulo (`L0/L1` simbólico y `L2` bucles imperativos puros acotados), pero aún no modela mutación compleja de objetos en sitio (`O2`), pipelines `pandas`/`SQL` acoplados dentro de funciones ni actores tensoriales indexados (Problemática `P3`).
4. **Inmutabilidad del núcleo histórico:** Para preservar el candado de integridad del estudio `baseline_v0` (`TestCgeDirectoryIntegrity` y los 41 hashes SHA-256), los módulos raíz históricos de `cge/` permanecen congelados; la evolución activa vive en `cge/mcp/`, `cge/diff/` y `cge/ingest_adapters/`.
