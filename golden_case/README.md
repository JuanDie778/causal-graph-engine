# Tostadora de Café — Modelo Operativo y Financiero

Este repositorio contiene la lógica de negocio, configuración y datos de simulación para las operaciones de tostado y distribución de café de especialidad.

## Variables y Palancas de Control (X)
- `price` ($p$): Precio de venta por paquete en euros (€), restringido a $p \in [2.0, 6.0]$. Valor base sugerido: 3.50 €.
- `marketing_budget` ($m$): Inversión mensual en adquisición de clientes y publicidad en redes, $m \in [0.0, 2000.0]$ €.
- `order_qty` ($q$): Cantidad de paquetes de grano verde encargados al distribuidor de origen ($q \in \{0, \dots, 1500\}$, enteros).

## Variables Exógenas (U)
- `base_demand` ($D_0$): Demanda natural de mercado no controlada, con comportamiento estocástico $D_0 \sim \mathcal{N}(1000, 100^2)$.

## Mecanismos del Sistema (V)
1. **Demanda del mercado:**
   $$d = D_0 \cdot (1 - 0.15 \cdot (p - 3)) + 8 \cdot \sqrt{m}$$
2. **Ventas efectivas:** Limitadas por la capacidad disponible: $s = \min(d, q)$.
3. **Estructura de costes:**
   - Descuento por volumen: el coste unitario $c$ es de 1.20 € si $q < 800$, reduciéndose a 1.00 € si $q \ge 800$.
   - Coste de posesión y almacenamiento: $h = 0.20 \cdot (q - s)$ € por cada paquete no vendido.
4. **Comisiones y Pasarela:**
   - La comisión de pasarela y procesamiento es del 2.0% sobre los ingresos brutos generados.
5. **Métricas de marca:**
   - Se monitorea el `brand_awareness` a partir de la inversión de marketing.

## Restricciones Operativas (C)
- **Presupuesto máximo de aprovisionamiento:** $c \cdot q + m \le 2500$ €
- **Consistencia contable:** Invariante de suma cero en el balance del libro mayor ($\sum \text{créditos} = \sum \text{débitos}$).
- **Límite físico de inventario:** Las ventas no pueden exceder las existencias disponibles ($s \le q$).

## Métrica Objetivo (J)
El beneficio neto del negocio se calcula como:
$$J = R - f - c \cdot q - m - h$$
donde $R$ son los ingresos brutos y $f$ las comisiones aplicadas.
