# Propuesta Matemática 03: Optimización Estocástica, Brecha de Jensen en Kinks No Suaves y Control de Riesgo ($\text{CVaR}_\alpha$)

**Autor:** Agente de Arquitectura Matemática (CGE Math Lab)  
**Estado:** Implementado y Verificado Experimentalmente en [`exp03_stochastic_cvar_optimizer.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/experiments/exp03_stochastic_cvar_optimizer.py)

---

## 1. El Problema Fundamental: La "Falacia de los Promedios" (Desigualdad de Jensen) en el Óptimo Determinista

En la implementación actual del repositorio ([`cge/layer2_exact.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/cge/layer2_exact.py) y [`cge/monte_carlo.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/cge/monte_carlo.py)):
1. `Layer2Solver` optimiza asumiendo que la variable exógena $D_0$ es determinista e igual a su media $\mu_{D_0} = \mathbb{E}[D_0] = 1000$.
2. El solucionador determinista coloca el sistema exactamente sobre la arista no diferenciable (el *kink* de `sales = min(demand, order_qty)`):
   $$q_{\text{det}}^* = 800, \quad p_{\text{det}}^* = 5.4370€, \quad m_{\text{det}}^* = 428.2140€ \implies d(\mu_{D_0}, p_{\text{det}}^*, m_{\text{det}}^*) = 800.0 = q_{\text{det}}^*$$
   reportando un beneficio nominal $J_{\text{det}}^* = 2990.88€$.
3. Luego, `monte_carlo.py` evalúa pasivamente esa misma solución ante $D_0 \sim \mathcal{N}(1000, 100^2)$ y observa que el beneficio medio cae a **$2852.08€$** (una degradación de **$-138.80€$**) y la probabilidad de rotura de stock es del **50.0%**, ¡pero el motor nunca utiliza esa información estocástica para re-optimizar $X$!

### Teorema 1 (Cuantificación Analítica Exacta de la Brecha de Jensen en el Kink)
Sea $D(p, m, D_0) = D_0 \phi(p) + \beta \sqrt{m}$ donde $\phi(p) = 1 - \alpha(p - 3)$ y $D_0 \sim \mathcal{N}(\mu_0, \sigma_0^2)$. Entonces $D \sim \mathcal{N}(\mu_D, \sigma_D^2)$ con:
$$\mu_D(p, m) = \mu_0 \phi(p) + \beta\sqrt{m}, \qquad \sigma_D(p) = \sigma_0 \phi(p)$$
Dado que la función de ventas $\text{sales}(D, q) = \min(D, q) = q - \max(0, q - D)$ es **cóncava** en $D$ y el coste de almacenamiento es $\text{holding}(D, q) = h \max(0, q - D)$, el beneficio para cualquier realización $D$ es:
$$J(p, m, q; D) = \big((1-\tau)p - c(q)\big) q - m - \big((1-\tau)p + h\big) \max(0, q - D)$$
Por la **Desigualdad de Jensen**, para toda distribución con $\sigma_D > 0$:
$$\mathbb{E}_{D_0}[J(p, m, q; D_0)] = J_{\text{det}}(p, m, q; \mu_0) - \big((1-\tau)p + h\big) \mathbb{E}\big[\max(0, D - \mu_D) \text{ o déficit}\big] < J_{\text{det}}(p, m, q; \mu_0)$$
Específicamente, definiendo el **margen de seguridad estandarizado (buffer $z$)** como:
$$z(p, m, q) = \frac{\mu_D(p, m) - q}{\sigma_D(p)}$$
la función de pérdida normal estándar $L(z) = \mathbb{E}_{Z \sim \mathcal{N}(0,1)}[\max(0, Z - z)] = \varphi(z) - z\big(1 - \Phi(z)\big)$ (donde $\varphi$ y $\Phi$ son la PDF y CDF de $\mathcal{N}(0,1)$) nos da la **fórmula cerrada exacta del beneficio esperado**:
$$\mathbb{E}[J(p, m, q)] = \big((1-\tau)p - c(q)\big) q - m - \big((1-\tau)p + h\big) \sigma_0 \phi(p) \big[\varphi(z) - z(1 - \Phi(z))\big]$$

#### Corolario (Por qué el óptimo determinista $z=0$ es subóptimo bajo incertidumbre)
En el óptimo determinista, $\mu_D = q \implies z = 0$. Como $\varphi(0) = \frac{1}{\sqrt{2\pi}} \approx 0.398942$ y $\phi(5.4370) = 0.63445$:
$$\Delta_{\text{Jensen}}(X_{\text{det}}^*) = \big(0.97(5.4370) + 0.20\big) \cdot 100 \cdot 0.63445 \cdot \frac{1}{\sqrt{2\pi}} = 5.4739 \times 63.445 \times 0.398942 = 138.54€$$
¡Esto explica analíticamente, con precisión de céntimos y sin necesidad de esperar mil simulaciones, por qué el beneficio cae de $2990.88€$ a $\approx 2852.34€$!

---

## 2. Optimización Estocástica Directa y Cuadratura Compuesta Dividida en el Kink

En lugar de optimizar para el promedio ($\max_X J(X, \mathbb{E}[U])$), el motor CGE debe permitir resolver dos problemas de optimización bajo incertidumbre sobre el DAG:

### 2.1. Integración Determinista Exacta vía Cuadratura de Gauss-Legendre Compuesta Dividida en el Kink
Un fenómeno numérico crucial en modelos causales con operadores $\min(D(U), q)$ o $\max(0, \cdot)$ es que el integrando $u \mapsto J(X, u)$ posee una **discontinuidad en la primera derivada (kink)** en el punto de conmutación $u_{\text{kink}}$ donde $D(u_{\text{kink}}) = q$.
* Una cuadratura polinómica global (como Gauss-Hermite estándar sobre $(-\infty, \infty)$) sufre degradación de convergencia de orden $O(1/N_q)$ al cruzar el kink sin partir el intervalo (produciendo un error de $\approx 1.15€$ incluso con $N_q = 99$).
* **Solución Espectral (Kink-Split Composite Gauss-Legendre):** Particionamos el soporte $[\mu_U - 6\sigma_U, \mu_U + 6\sigma_U]$ en dos subintervalos suaves alrededor de la frontera de conmutación ($[\mu_U - 6\sigma_U, \mu_U]$ y $[\mu_U, \mu_U + 6\sigma_U]$) y aplicamos cuadratura de Gauss-Legendre de $N_q = 32$ nodos ponderada por la densidad normal $\mathcal{N}(\mu_U, \sigma_U^2)$ en cada rama suave $C^\infty$:
  $$\mathbb{E}_U[J(X, U)] = \int_{\mu_U - 6\sigma_U}^{u_{\text{kink}}} J_{\text{dem}}(X, u)\,\phi_U(u)\,du + \int_{u_{\text{kink}}}^{\mu_U + 6\sigma_U} J_{\text{cap}}(X, u)\,\phi_U(u)\,du$$
  Como verificamos en [`exp03_stochastic_cvar_optimizer.py`](file:///c:/Users/ussaa/Documents/Sistema%20de%20solcuiones%20automatizado/math_lab/experiments/exp03_stochastic_cvar_optimizer.py), esta cuadratura compuesta coincide con la fórmula analítica cerrada hasta el cuarto decimal (**$2852.3391€$ vs $2852.3391€$, error $= 0.0000€$**) en menos de $1\text{ ms}$.

### 2.2. El Mecanismo del Óptimo Estocástico: Desplazamiento Fuera del Kink y Stock de Seguridad
¿Por qué la política determinista $(p_{\text{det}}^* = 5.4370€, m_{\text{det}}^* = 428.21€, q_{\text{det}}^* = 800)$ es subóptima bajo incertidumbre?
1. En el óptimo determinista, $\mu_D = 800 = q$, es decir, el sistema se sitúa **exactamente sobre la arista del kink** donde la Brecha de Jensen $\Delta_{\text{Jensen}}(X) = J_{\text{det}}(X) - \mathbb{E}[J(X, U)]$ alcanza su máximo ($138.54€$). Cuando $D_0 > 1000$ (el 50% de los días), toda la demanda superior a 800 se trunca y se pierde por falta de stock ($\min(D, 800) = 800$), mientras que cuando $D_0 < 1000$ se pierden ventas.
2. Cuando optimizamos directamente el valor esperado $\max_{p, m, q} \mathbb{E}_U[J(p, m, q; U)]$, el solucionador estocástico descubre la política:
   $$p_{\text{stoch}}^* = 5.7721€, \qquad m_{\text{stoch}}^* = 384.01€, \qquad q_{\text{stoch}}^* = 800$$
   En este punto:
   * La demanda nominal en la media es $\mu_D^* = 740.95 < 800$, lo que deja un **stock de seguridad (inventory buffer) de $q - \mu_D^* = +59.05$ unidades** por debajo del techo de capacidad $q = 800$ (manteniendo $q=800$ para conservar el descuento por volumen $c=1.00€$).
   * Aunque en un mundo determinista sin ruido esta política daría $34.7€$ menos que el óptimo determinista, bajo incertidumbre **reduce la penalización de Jensen de $138.54€$ a sólo $31.34€$** porque permite absorber los picos de demanda positiva ($D_0 \in [1000, 1100]$) vendiendo esas $59$ unidades de reserva a un precio unitario superior ($5.7721€$) y ahorrando $44.21€$ en marketing.
   * **Resultado neto:** El beneficio esperado real $\mathbb{E}[J]$ sube de **$2852.34€$ a $2925.41€$ ($+73.07€$/día de ganancia neta)** y la probabilidad de rotura de stock cae drásticamente del **$50.0\%$ al $15.3\%$**.

---

## 3. Optimización Aversa al Riesgo mediante $\text{CVaR}_\alpha$ (Fórmula Variacional de Rockafellar-Uryasev)

En sistemas financieros, logísticos o de SLA, maximizar únicamente la esperanza $\mathbb{E}[J]$ puede exponer al sistema a colas catastróficas. Definimos el **Valor en Riesgo Condicional ($\text{CVaR}_\alpha$)** de cola inferior al nivel $\alpha \in (0, 1)$ (ej. $\alpha = 0.05$, el promedio del 5% de los peores escenarios) mediante la **representación variacional convexa de Rockafellar y Uryasev (2000)**:
$$\text{CVaR}_\alpha\big(J(X, U)\big) = \max_{\eta \in \mathbb{R}} \left\{ \eta - \frac{1}{\alpha} \mathbb{E}_U\big[ \max(0, \eta - J(X, U)) \big] \right\}$$
El problema de optimización robusta / aversa al riesgo con parámetro de aversión $\rho \in [0, 1]$ se formula conjuntamente sobre $(X, \eta) \in \Omega_X \times \mathbb{R}$ como un único programa matemático:
$$\max_{X \in \Omega_X, \, \eta \in \mathbb{R}} \; (1 - \rho)\,\mathbb{E}_U[J(X, U)] + \rho \left( \eta - \frac{1}{\alpha} \sum_{s=1}^S \pi_s \max\big(0, \eta - J(X, u_s)\big) \right)$$
$$\text{sujeto a} \quad \mathcal{C}(X) \le 0$$
Esta formulación construye la **Frontera Eficiente Media-CVaR** completa sobre el grafo causal, ofreciendo al usuario una familia paramétrica de recetas desde *Neutral al Riesgo* ($\rho = 0$) hasta *Ultra-Conservadora / Protección de Cola* ($\rho \to 1$).
