"""Paquete `cge.ingest_adapters`: Ingesta estática AST/Docstring y traducción a IR ejecutable CGE."""
from .base import (
    AdapterResponse,
    ExtractedClassSpec,
    ExtractedFunctionSpec,
    QueryContext,
    StaticRepoASTExtractor,
)
from .dispatch_adapter import DispatchCausalAdapter
from .engine import GENERIC_PACKAGE_REPO_ID, RepositoryCausalEngine
from .epidemiology_adapter import EpidemiologyCausalAdapter
from .generic_adapter import ActorDBSidecarSpec, ContractASTMismatchError, GenericPackageAdapter
from .loan_tax_adapter import LoanTaxCausalAdapter
from .quantecon_adapter import QuantEconCausalAdapter
from .supplychain_adapter import SupplyChainCausalAdapter

__all__ = [
    "ActorDBSidecarSpec",
    "AdapterResponse",
    "ContractASTMismatchError",
    "DispatchCausalAdapter",
    "EpidemiologyCausalAdapter",
    "ExtractedClassSpec",
    "ExtractedFunctionSpec",
    "GENERIC_PACKAGE_REPO_ID",
    "GenericPackageAdapter",
    "LoanTaxCausalAdapter",
    "QuantEconCausalAdapter",
    "QueryContext",
    "RepositoryCausalEngine",
    "StaticRepoASTExtractor",
    "SupplyChainCausalAdapter",
]
