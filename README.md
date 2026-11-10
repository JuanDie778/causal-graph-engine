# Decision Intelligence & Causal Graph Engine (CGE)

> **Un "Google Maps multidimensional" para la toma de decisiones y optimización causal prescriptiva.**

---

## 🧭 Visión del Proyecto

El **Causal Graph Engine (CGE)** traduce código fuente, consultas SQL, pipelines tabulares (`pandas`), modelos de machine learning, reglas operativas y datos históricos en una **Tupla Causal Ejecutable unificada**:

$$\mathcal{M} = \langle U, X, V, \mathcal{F}, \mathcal{C}, J \rangle$$

* **$U$ (Variables Exógenas):** Entorno estocástico incontrolable (mercado, clima, latencias).
* **$X$ (Variables de Control):** Palancas sobre las que se puede intervenir directamente.
* **$V$ (Variables Endógenas de Estado):** Consecuencias intermedias calculadas por el sistema.
* **$\mathcal{F}$ (Mecanismos Causales):** Ecuaciones de propagación analítica (`SymPy`), reglas o subrogados de cajas negras.
* **$\mathcal{C}$ (Invariantes y Restricciones Duras):** Límites de capacidad, presupuestos, balances contables o normas.
* **$J$ (Objetivo / Función de Valor):** Métrica escalar a maximizar o minimizar.

A partir de esta estructura, CGE no solo predice o describe el sistema, sino que **resuelve de forma prescrita la intervención mínima necesaria ($\Delta X$)** para alcanzar cualquier estado deseado respetando todas las restricciones del mundo real.

---

## 📁 Estructura del Repositorio

```text
├── cge/                 # Motor central CGE
│   ├── fidelity.py      # Certificado de fidelidad: ejecución del código original vs IR
│   ├── ingest.py        # Extractor AST y compilación hacia CausalIR
│   ├── universal_compiler.py # Compilador multi-fuente (pandas, SQL, manuales normativos)
│   ├── layer1_heuristic.py   # Enrutador rápido Top-K por semianillo Max-Product
│   ├── layer2_exact.py       # Optimizador híbrido exacto MINLP / KKT
│   ├── solver/          # Estrategias formales de optimización (HiGHS, regímenes, certificados)
│   ├── audit.py         # Auditor formal de patologías (T1-T7) e inconsistencias
│   ├── copilot.py       # Copiloto neuro-simbólico con trazabilidad estricta
│   └── schema.py        # Validación del esquema canónico
│
├── schema/              # Definición formal de datos
│   └── canonical_schema.json # JSON Schema Draft-07 (v1.1)
│
├── ui/                  # Interfaz de usuario interactiva ("Google Maps")
│   ├── server.py        # API REST con FastAPI (evaluación, optimización, inspección)
│   └── static/          # Lienzo Cytoscape.js con layout causal por capas y controles
│
├── benchmark/           # Batería de pruebas y verificación DoD
│   ├── or_bench/        # Banco de problemas de Investigación Operativa con oráculos
│   ├── test_f1_anti_memorization.py # Escáner de cero fugas léxicas
│   ├── test_f6_api.py   # Pruebas de integración de la API
│   └── test_f6_e2e.py   # Pruebas End-to-End con Playwright
│
├── docs/                # Documentación de arquitectura y especificaciones
│   ├── IR_SPEC.md       # Especificación del Causal Intermediate Representation (IR)
│   ├── MARCO_EVALUACION.md # Protocolo de evaluación multi-pilar e IAI
│   ├── PLAN_MOTOR.md    # Hoja de ruta del motor
│   └── DoD_MVP.md       # Definición de Hecho del MVP
│
├── golden_case/         # Caso base de calibración con 7 patologías de prueba
├── math_lab/            # Laboratorio matemático de investigación y prototipado
└── eval_reports/        # Informes reproducibles de evaluación
```

---

## 🚀 Inicio Rápido

### 1. Requisitos
* Python 3.10+
* Dependencias principales: `scipy`, `sympy`, `pandas`, `fastapi`, `uvicorn`, `jsonschema`

### 2. Ejecutar la Interfaz Interactiva
Para iniciar el servidor local y abrir el visor en el navegador:
```powershell
python run_ui.py golden_case
```
Accede a la UI en `http://127.0.0.1:8765` y a la documentación OpenAPI en `http://127.0.0.1:8765/docs`.

### 3. Ejecutar las Pruebas y Evaluación
* **Benchmark completo del DoD:**
  ```powershell
  python run_benchmark.py --level all
  ```
* **Marco de Evaluación Avanzada (Gates & Pilares):**
  ```powershell
  python run_eval.py
  ```

---

## 🔒 Principios de Desarrollo y Cero Fugas
1. **Determinismo:** El cálculo causal y las recetas de optimización se resuelven exclusivamente mediante solvers matemáticos certificados.
2. **Cero fugas léxicas:** El motor no contiene nombres cableados de ningún caso de prueba específico.
3. **Fidelidad verificada:** Toda receta óptima se certifica ejecutando el código original del usuario.
