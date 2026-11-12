"""Paquete `cge.ingest_adapters`: Ingesta estática AST/Docstring y traducción a IR ejecutable CGE."""
from .base import (
    AdapterResponse,
    ExtractedClassSpec,
    ExtractedFunctionSpec,
    QueryContext,
    StaticRepoASTExtractor,
)
from .dispatch_adapter import DispatchCausalAdapter
from .engine import RepositoryCausalEngine
from .epidemiology_adapter import EpidemiologyCausalAdapter
from .loan_tax_adapter import LoanTaxCausalAdapter
from .quantecon_adapter import QuantEconCausalAdapter
from .supplychain_adapter import SupplyChainCausalAdapter

__all__ = [
    "AdapterResponse",
    "DispatchCausalAdapter",
    "EpidemiologyCausalAdapter",
    "ExtractedClassSpec",
    "ExtractedFunctionSpec",
    "LoanTaxCausalAdapter",
    "QuantEconCausalAdapter",
    "QueryContext",
    "RepositoryCausalEngine",
    "StaticRepoASTExtractor",
    "SupplyChainCausalAdapter",
]
