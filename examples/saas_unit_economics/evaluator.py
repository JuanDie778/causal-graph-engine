"""Evaluador simbólico (L0/L1) de economía unitaria mensual con trampas causales sutiles."""
from .tiers import compute_cloud_delivery_cost


def evaluate_enterprise_plan(
    seats: int = 40,
    seat_price: float = 65.0,
    annual_discount: float = 0.12,
    sla_credit_reserve: float = 0.03,
    partner_rebate_rate: float = 0.08,
    legacy_crm_score: float = 85.0,
) -> dict:
    """Evalúa métricas mensuales de un contrato Enterprise B2B.

    Contiene deliberadamente:
    - `legacy_crm_score`: parámetro de entrada muerto (no participa en el flujo de caja).
    - `nps_vanity_index`: estado intermedio huérfano (`T1-A_STRUCTURAL_ORPHAN`).
    - `partner_rebate_rate`: cancelación algebraica exacta (`T1-B`) por reembolso espejo del fondo MDF.
    """
    if seat_price < 15.0 or seat_price > 150.0:
        raise ValueError("seat_price must stay within [15.0, 150.0]")
    if annual_discount < 0.0 or annual_discount > 0.35:
        raise ValueError("annual_discount must stay within [0.0, 0.35]")
    if sla_credit_reserve < 0.0 or sla_credit_reserve > 0.15:
        raise ValueError("sla_credit_reserve must stay within [0.0, 0.15]")

    nps_vanity_index = seats * 1.25 + 10.0
    gross_bookings = seats * seat_price * (1.0 - annual_discount)
    net_billings = gross_bookings * (1.0 - sla_credit_reserve)
    cloud_cost = compute_cloud_delivery_cost(seats, gross_bookings)

    # Trampa T1-B: el descuento de canal es compensado dólar a dólar por el fondo MDF corporativo
    partner_payout = gross_bookings * partner_rebate_rate
    mdf_reimbursement = gross_bookings * partner_rebate_rate
    net_channel_effect = partner_payout - mdf_reimbursement

    contribution_margin = net_billings - cloud_cost - net_channel_effect
    assert contribution_margin >= 400.0, "c_min_contribution_margin"

    return {
        "nps_vanity_index": nps_vanity_index,
        "gross_bookings": gross_bookings,
        "net_billings": net_billings,
        "cloud_cost": cloud_cost,
        "net_channel_effect": net_channel_effect,
        "contribution_margin": contribution_margin,
    }
