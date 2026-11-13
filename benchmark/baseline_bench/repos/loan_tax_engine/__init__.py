"""loan_tax_engine: Financial amortization and progressive tax schedule calculator."""
from .amortization import compute_french_schedule
from .tax_brackets import compute_progressive_tax, detect_tax_kinks

__all__ = [
    "compute_french_schedule",
    "compute_progressive_tax",
    "detect_tax_kinks",
]
