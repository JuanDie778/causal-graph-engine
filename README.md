# ActorDB — Causal Graph & Logic Engine (CGE)

> **Hacia una Base de Datos de los Actores, Variables y Relaciones Lógicas de un Repositorio.**

---

## 🧭 Visión y Propósito

Mientras las bases de datos tradicionales organizan registros (filas) y las bases de datos vectoriales indexan fragmentos de texto por similitud, este proyecto explora una nueva categoría tecnológica: **estructurar, verificar y consultar a los actores lógicos y causales de un proyecto** (variables exógenas $U$, palancas de control $X$, estados $V$, mecanismos $\mathcal{F}$, invariantes $\mathcal{C}$ y objetivos $J$).

El objetivo es dotar tanto a las personas como a los agentes de Inteligencia Artificial de un **mapa lógico auditable, explícito y matemáticamente verificable** del sistema, superando las limitaciones del análisis descriptivo tradicional ("IA + Notebook").

Consulta el [**`MANIFIESTO.md`**](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/MANIFIESTO.md) para conocer la tesis completa, los cimientos matemáticos y las preguntas de diseño abiertas.

---

## 📁 Estructura Limpia del Repositorio

```text
├── MANIFIESTO.md               # Tesis estratégica, fundamentos y horizonte de diseño
├── README.md                   # Guía rápida y estructura del proyecto
├── cge/                        # Núcleo matemático y de compilación (materia prima reciclable)
│   ├── ingest.py               # Extracción AST de Python hacia representación causal (CausalIR)
│   ├── universal_compiler.py   # Transpiladores multi-fuente (pandas, SQL, manuales Markdown, CSV)
│   ├── fidelity.py             # Certificado de fidelidad empírica (código original vs IR)
│   ├── solver/                 # Solvers matemáticos (LP, MILP, QP, NLP, MINLP, IIS) y certificados
│   ├── layer1_heuristic.py     # Enrutamiento causal rápido sobre semianillo Max-Product
│   ├── layer2_exact.py         # Interfaz de optimización exacta y análisis de restricciones
│   ├── audit.py                # Motor de diagnóstico de coherencia e invariantes (T1-T7)
│   ├── copilot.py              # Copiloto determinista conectado al grafo
│   ├── empirical.py            # Estimación econométrica estructural desde datos tabulares
│   ├── monte_carlo.py          # Simulación estocástica y optimización bajo incertidumbre (CVaR)
│   └── schema.py               # Modelo de datos canónico
│
├── schema/                     # Esquema JSON formal (Draft-07)
│   └── canonical_schema.json
│
├── docs/                       # Especificaciones conceptuales y técnicas
│   ├── HOJA_DE_RUTA_CONCEPTUAL.md               # Seguimiento vivo de problemáticas (P1-P5) y decisiones
│   ├── 01_ALCANCE_Y_ONTOLOGIA_DOMINIO.md        # Decisión 1: Frontera causal y ontología de actores
│   ├── 02_FRONTERA_IMPERATIVA_Y_CONTRATOS.md    # Problemática P1: Opacidad O1-O5, niveles L0-L5 y @contract
│   ├── 03_TIEMPO_ESTADO_Y_CICLOS.md             # Problemática P2: Ciclos dinámicos vs lazos, G_T y políticas
│   └── IR_SPEC.md                               # Especificación de representación intermedia actual
│
├── ui/                         # Explorador visual interactivo y servidor FastAPI
│   ├── server.py
│   └── index.html
│
├── golden_case/                # Caso de estudio base (modelo operativo con patologías de prueba)
└── benchmark/                  # Suite ligera de verificación (fidelidad y OR-Bench)
    ├── or_bench/               # 20 problemas clásicos de Investigación de Operaciones
    ├── test_f10_fidelity.py
    └── test_n8_or_bench.py
```

---

## 🚀 Verificación y Ejecución Rápida

### 1. Ejecutar la Suite de Pruebas Ligera
Verifica el motor de fidelidad, el caso base y los 20 problemas de `OR-Bench`:
```powershell
python -m pytest golden_case/tests/ benchmark/ -v
```

### 2. Iniciar el Explorador Visual Actual
```powershell
python run_ui.py golden_case
```
