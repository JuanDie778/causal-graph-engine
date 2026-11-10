# Propuesta Matemática 06: Escalabilidad MPCC ($2^M$ Kinks), Grafos con Ciclos Temporales (MPC sobre DAG Expandido) y Econometría Causal bajo Endogeneidad (2SLS / IV & Back-Door)

## 1. Problema 1: Explosión Combinatoria $2^M$ de Operadores $\min/\max$ en Cadenas de Suministro

En un grafo causal con un único operador $\min(D, q)$, la partición en regímenes suaves exige resolver sólo $2$ subproblemas NLP ($D \le q$ y $q \le D$). Sin embargo, en una red empresarial general con $M$ cuellos de botella en serie o en paralelo:
$$s_k = \min\big(A_k(x),\, B_k(x)\big), \qquad k = 1, \dots, M,$$
el espacio de estados suaves se fragmenta en **$2^M$ ortantes combinatorios** indexados por la máscara binaria $b \in \{0, 1\}^M$. Para $M=6$ existen $64$ subproblemas NLP; para $M=20$ existen $1{,}048{,}576$ subproblemas, haciendo intratable la enumeración exhaustiva.

### 1.1. Formulación Matemática: Continuación Homotópica Log-Sum-Exp + Elevación de Epígrafe (MPCC)

Para evitar la enumeración exponencial $O(2^M)$ sin perder precisión $C^\infty$ sobre las variedades de kink activas ($A_k(x^*) = B_k(x^*)$), formulamos un algoritmo en **dos fases con complejidad $O(1)$ en el número de llamadas al solver NLP** ($3$ resoluciones independientemente de $2^M$):

1. **Fase 1 — Continuación Homotópica Suave Log-Sum-Exp ($\text{LSE}_\tau$):**
   Reemplazamos cada operador no diferenciable $\min(A_k, B_k)$ por su aproximación analítica $C^\infty$ estabilizada numéricamente con parámetro de temperatura inversa $\tau > 0$:
   $$\text{softmin}_\tau(A_k, B_k) = m_k - \frac{1}{\tau}\ln\Big(\exp\big(-\tau(A_k - m_k)\big) + \exp\big(-\tau(B_k - m_k)\big)\Big), \quad m_k = \min(A_k, B_k).$$
   - **Cota de error uniforme:** Para todo $(A_k, B_k) \in \mathbb{R}^2$ se cumple:
     $$\min(A_k, B_k) - \frac{\ln 2}{\tau} \;\le\; \text{softmin}_\tau(A_k, B_k) \;\le\; \min(A_k, B_k).$$
   - Resolvemos una secuencia corta de problemas sin restricciones de régimen ($\tau_1 = 0.5 \to \tau_2 = 5.0$) con `L-BFGS-B`, utilizando la solución de $\tau_r$ como *warm-start* para $\tau_{r+1}$. Esta trayectoria homotópica atraviesa suavemente las costuras no diferenciables y sitúa el iterado en la cuenca de atracción del equilibrio global.

2. **Fase 2 — Pulido Exacto mediante Elevación de Epígrafe / MPCC:**
   Introducimos el vector de variables auxiliares de epígrafe $s = (s_1, \dots, s_M) \in \mathbb{R}^M$ que representan el flujo efectivo de cada etapa y resolvemos un único NLP suave $C^\infty$ con $2M$ restricciones diferenciables:
   $$\max_{x,\, s} \; J(x, s) \quad \text{s.a.} \quad \begin{cases} A_k(x) - s_k \ge 0, & k = 1, \dots, M, \\ B_k(x) - s_k \ge 0, & k = 1, \dots, M. \end{cases}$$
   Siempre que $\frac{\partial J}{\partial s_k} > 0$ (ingreso marginal positivo por unidad de flujo), las condiciones KKT fuerzan automáticamente la complementariedad activa $s_k^* = \min(A_k(x^*), B_k(x^*))$ en el óptimo sin necesidad de variables binarias ni ramificación $2^M$.

---

## 2. Problema 2: Grafos Causales con Ciclos Temporales (Inventario Inter-Temporal y MPC)

El motor estático asume que el grafo causal $\mathcal{G} = (V, E)$ es un DAG acíclico. Sin embargo, los sistemas operativos reales contienen ciclos de retroalimentación de estado entre periodos consecutivos, tales como la ecuación de conservación de inventario con *carry-over*:
$$\begin{aligned}
\text{avail}_t &= I_{t-1} + q_t, \\
\text{sales}_t &= \min(D_t(p_t, m_t),\, \text{avail}_t), \\
I_t &= \max(0,\, \text{avail}_t - \text{sales}_t).
\end{aligned}$$
Si se proyecta estáticamente olvidando los subíndices temporales, aparece el ciclo dirigido $\text{inventory} \to \text{available} \to \text{sales} \to \text{inventory}$, violando la propiedad DAG (`nx.is_directed_acyclic_graph == False`).

### 2.1. Desenrollado en DAG Expandido en el Tiempo ($\mathcal{G}_T$) y Suboptimalidad de Políticas Miopes

Dado un horizonte de planificación finito $T \ge 2$, descomponemos el conjunto de aristas en aristas instantáneas $E_0$ ($\text{lag} = 0$) y aristas de transición inter-temporal $E_1$ ($\text{lag} = 1$, como $I_{t-1} \to \text{avail}_t$). El **Grafo Expandido en el Tiempo** $\mathcal{G}_T = (V_T, E_T)$ con:
$$V_T = \{I_0\} \cup \bigcup_{t=1}^T \{v_t : v \in V\}, \qquad E_T = \bigcup_{t=1}^T \big\{(u_t, v_t) : (u,v) \in E_0\big\} \cup \big\{(u_{t-1}, v_t) : (u,v) \in E_1\big\}$$
es **estrictamente un DAG acíclico** ($\mathcal{G}_T$ admite orden topológico exacto por bloques temporales $t=1, \dots, T$).

- **Fallo de la Optimización Miope ($T=1$ repetido):** Un optimizador miope que maximiza $\text{profit}_1(p_1, m_1, q_1)$ asigna valor marginal cero al inventario final $I_1$. Cuando la demanda base por periodo es moderada ($D_0 = 420$) y el descuento por volumen exige pedir $q \ge 800$ ($c = 1.00€$ frente a $1.35€$), el optimizador miope rechaza pedir $q_1 = 800$ (porque el excedente $I_1$ parece una pérdida pura en $t=1$) y realiza pedidos pequeños en ambos periodos pagando $1.35€/\text{ud}$.
- **Optimización Dinámica Multi-Periodo (MPC / Batch Ordering sobre $\mathcal{G}_T$):** Al optimizar conjuntamente $J_{1:T} = \sum_{t=1}^T \text{profit}_t$ sobre $\mathcal{G}_T$, el solver descubre la política de lote inter-temporal: pide $q_1^* = 800$ en $t=1$ (desbloqueando $c_1 = 1.00€/\text{ud}$), transfiere el inventario sobrante $I_1^* > 0$ hacia $t=2$ (pagando sólo $h = 0.15€/\text{ud} < 0.35€/\text{ud}$ de ahorro en compra) y reduce el pedido $q_2^*$ en $t=2$, superando estrictamente a la política miope.

---

## 3. Problema 3: Sesgo de Endogeneidad / Confusión en Inferencia Econométrica Causal (T7)

En la auditoría econométrica T7 (`exp04`), si los datos históricos provienen de un entorno observacional donde el gestor ajustaba el presupuesto de marketing $M_t$ en respuesta a un shock de demanda latente/confusor $W_t$ (por ejemplo, estacionalidad o fiebre de consumo que afecta tanto a la decisión de gasto publicitario $M_t \leftarrow W_t$ como a la demanda $W_t \to D_t$):
$$\begin{aligned}
\sqrt{M_t} &= \max\big(5,\, \sqrt{Z_t} + \gamma W_t + \nu_t\big), \\
D_t &= D_{0,t}\big(1 - \alpha(P_t - 3)\big) + \beta \sqrt{M_t} + \underbrace{\delta W_t + \varepsilon_t}_{u_t},
\end{aligned}$$
entonces $\text{Cov}(\sqrt{M_t}, u_t) = \gamma \delta \text{Var}(W_t) \ne 0$, violando la exogeneidad estricta de Gauss-Markov.

### 3.1. Diagnóstico de Hausman-Wu e Identificación Causal vía 2SLS (IV) y Criterio Back-Door

1. **Inconsistencia del OLS Ingenuo:**
   $$\text{plim}_{N \to \infty} \hat{\beta}_{\text{OLS}} = \beta + \frac{\text{Cov}(\widetilde{\sqrt{M}}, u)}{\text{Var}(\widetilde{\sqrt{M}})} \ne \beta.$$
   En nuestro experimento ($N=500, \alpha=0.15, \beta=8.0$), OLS estima $\hat{\beta}_{\text{OLS}} = 9.7115$ (un **sesgo de $+21.39\%$**).
2. **Test de Especificación de Hausman-Wu (Función de Control):**
   Dado un instrumento exógeno válido $Z_t$ (tal que $\text{Cov}(Z_t, W_t) = 0$ y $\text{Cov}(Z_t, \sqrt{M_t}) \ne 0$, ej. variación exógena en el coste de puja publicitaria), proyectamos $\sqrt{M}$ sobre el subespacio de instrumentos $Z = [D_0, D_0 P, \sqrt{Z_{\text{inst}}}]$ para obtener el residuo de primera etapa $\hat{v} = \sqrt{M} - P_Z \sqrt{M}$. Al incluir $\hat{v}$ en la regresión estructural aumentada:
   $$D = X\theta + \rho \hat{v} + \text{error},$$
   el estadístico $t$ sobre $H_0: \rho = 0$ alcanza $|t_{\text{Hausman}}| = 67.77 > 1.96$ ($p < 10^{-6}$), detectando automáticamente la endogeneidad y bloqueando el uso de OLS.
3. **Estimador de Variables Instrumentales en Dos Etapas (2SLS / IV):**
   $$\hat{\theta}_{\text{2SLS}} = (X^\top P_Z X)^{-1} X^\top P_Z Y, \qquad P_Z = Z(Z^\top Z)^{-1}Z^\top.$$
   Cuando $W_t$ es inobservable, 2SLS purga la componente correlacionada con $W_t$ y reduce el error relativo de $\hat{\beta}$ del **$21.39\%$ al $1.58\%$** ($\hat{\beta}_{\text{2SLS}} = 7.8739$).
4. **Estimador con Ajuste Back-Door de Pearl:**
   Cuando el nodo confusor $W_t$ está registrado en el grafo causal y es observable, bloquear el camino *back-door* $M_t \leftarrow W_t \to D_t$ condicionando explícitamente sobre $W_t$ en la matriz de diseño $X_{\text{BD}} = [D_0, D_0 P, \sqrt{M}, W]$ recupera $\hat{\alpha}_{\text{BD}} = 0.1501, \hat{\beta}_{\text{BD}} = 8.0165$ (**error relativo de $0.21\%$**).

---

## 4. Referencias de Implementación

- Prototipo ejecutable: [`math_lab/experiments/exp05_advanced_edge_cases_mpcc_mdp_iv.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/experiments/exp05_advanced_edge_cases_mpcc_mdp_iv.py)
- Clases principales:
  - [`MultiKinkSupplyChainBenchmark`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/experiments/exp05_advanced_edge_cases_mpcc_mdp_iv.py#L20-L185) y [`run_multi_kink_scalability_experiment`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/experiments/exp05_advanced_edge_cases_mpcc_mdp_iv.py#L188-L204)
  - [`TemporalCyclicSCMAndMPCSolver`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/experiments/exp05_advanced_edge_cases_mpcc_mdp_iv.py#L211-L365)
  - [`CausalEndogeneityIV2SLSEstimator`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/experiments/exp05_advanced_edge_cases_mpcc_mdp_iv.py#L372-L465)
