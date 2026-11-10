# 001 — Refundación Universal del Motor Causal (`cge/`) y Banco Anti-Memorización (F1)

- **Fecha**: 2026-11-08
- **Estado**: Aceptado y Verificado
- **Documentos Relacionados**: `docs/IR_SPEC.md`, `docs/PLAN_MOTOR.md`, `docs/MARCO_EVALUACION.md`, `docs/DoD_MVP.md`

---

## 1. Problema Detectado (Hipótesis de Partida)

En la auditoría inicial del código de `cge/` (`cge/ingest.py`, `cge/layer1_heuristic.py`, `cge/layer2_exact.py`, `cge/audit.py`, `cge/empirical.py`, `cge/monte_carlo.py`, `cge/copilot.py`) y `ui/export_ui.py`, se identificó que el motor dependía de identificadores, nombres de archivos y ecuaciones cableadas del caso de prueba ("Tostadora de Café"):
- `ingest.py` clasificaba variables mediante listas literales de nombres de archivos (`config.py`, `environment.py`, `demand.py`) y nombres de funciones (`calculate_demand`, `optimize_order`, `process_refund`).
- `layer2_exact.py` y `monte_carlo.py` contenían las fórmulas algebraicas de la cafetería escritas directamente en Python en lugar de leer y compilar las expresiones simbólicas del grafo causal (`CausalIR`).
- `audit.py` y `empirical.py` asumían nombres de columnas y variables específicos del caso base.

---

## 2. Decisiones Arquitectónicas Implementadas (D1–D6)

1. **Contrato Único de Representación Intermedia (`CausalIR` v1.1 en `schema/canonical_schema.json` y `cge/schema.py`)**:
   - Se enriqueció el esquema canónico manteniendo 100% de retrocompatibilidad con `schema.json` v1.0: cada nodo almacena opcionalmente `symbolic_expr` (expresión `sympy` exacta), `discontinuities` (umbrales `Piecewise`), `distribution` ($\mathcal{P}_U$), `characteristic_scale` y `clarke_kink_index`.
2. **Clasificación 100% Automática U / X / V / J / T1 sin Configuración Manual (`cge/ingest.py`)**:
   - **Exógenos ($U$)**: Detectados automáticamente por invocación a generadores de números aleatorios (`gauss`, `normal`, `uniform`, `lognormvariate`, etc.) o funciones muestreadoras `sample_*`.
   - **Controles ($X$)**: Detectados automáticamente desde constantes globales de módulo acompañadas de tuplas `*_BOUNDS` o inyectadas como argumentos raíz sin padres en el DAG.
   - **Objetivo ($J$)**: Seleccionado automáticamente como el nodo sumidero (`out_degree == 0`) con mayor número de ancestros causales y palancas de control conectadas.
   - **Estado ($V$) y Nodos Muertos ($T1$)**: Todo nodo intermedio con camino dirigido a $J$ es estado activo; aquellos sin camino dirigido (`T1-A`) o cuya derivada simbólica total se anula idénticamente (`T1-B`) se marcan automáticamente como nodos muertos.
3. **Compilación Simbólica AST $\to$ SymPy con Inlining de Helpers, Soporte OOP y Fallback de Caja Negra (`ASTFunctionToSymPy`)**:
   - Traduce asignaciones encadenadas, condicionales `if/else` (`sp.Piecewise`), acotaciones `min`/`max` (`sp.Min`/`sp.Max`), funciones matemáticas (`sqrt`, `log`, `log1p`, `exp`, `abs`), bucles `for` acotados y llamadas a funciones auxiliares privadas (inlining recursivo).
   - **Fallback D3.2**: Cualquier función fuera del subconjunto simbólico (ej. bucles `while` dinámicos) se encapsula automáticamente como `python_callable` (`confidence = 0.8`), calculando gradientes locales por diferencias finitas centrales de orden $O(h^2)$.
4. **Enrutamiento Tropical Max-Product y AD en Modo Inverso (`cge/layer1_heuristic.py`)**:
   - Reemplaza la suma de inversos por programación dinámica exacta en el semianillo Tropical Max-Product y diferenciación automática en modo inverso $O(|V|+|E|)$, priorizando aristas `exact_code`/`discrete_rule` sobre aristas `empirical_data` paralelas.
5. **Solucionador Universal Híbrido Simbólico-Numérico (`cge/layer2_exact.py`)**:
   - Compila dinámicamente cada expresión `symbolic_expr` del `CausalIR` mediante `sp.lambdify`, extrae automáticamente los regímenes de continuidad (`Piecewise`) y las ramas suaves $C^\infty$ de los cuellos de botella (`Min`/`Max`), y resuelve cada subproblema mediante SLSQP multi-arranque y proyección sobre la variedad kink.
   - Calcula precios sombra KKT ante metas infactibles y genera recetas de intervención ordenadas por precedencia causal.
6. **Auditoría Simbólica Formal (`cge/audit.py`) y Regresión Estructural Conjunta (`cge/empirical.py`)**:
   - Verificación simbólica de invariantes de conservación contable (`T2`) sobre todas las ramas de ejecución, detección automática de discrepancias porcentaje/tasa entre `README.md` y AST (`T3`), evaluación de subóptimos locales (`T4`), salto de discontinuidades (`T5`), índice de fragilidad de Clarke (`T6`) y regresión OLS estructural conjunta con selección de base por BIC y errores estándar HC3 (`T7`).

---

## 3. Verificación en el Banco Imposible de Memorizar (F1)

- **Puerta P8 (Cero Fugas de Dominio)**: 0 ocurrencias de identificadores del caso base o mundos procedurales en todo `cge/*.py` y `ui/export_ui.py`.
- **Suite M0 (N0–N5)**: 6/6 niveles superados al 100% (`run_benchmark.py --level all`).
- **Suite M1 y Relaciones Metamórficas (MR1–MR10)**: 100% de éxito en variantes con renombrado total (`renamed`), funciones auxiliares (`split_helpers`), fusión monolítica (`merged_monolith`) y clases OOP (`oop_service`).
- **Suites M2 y M3 (Mundos Procedurales Multi-Dominio)**: 100% de precisión estructural y gap de optimalidad $< 1\%$ frente al `GenericWorldOracle` en `saas_cloud`, `fleet_logistics` y `energy_microgrid`, con 0 falsas alarmas en mundos limpios.
