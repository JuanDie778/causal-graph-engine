"""Servidor MCP Causal de ActorDB (`cge.mcp`): herramientas `cge_trace`, `cge_do`, `cge_solve`, `cge_prove`."""
from .oracle import ActorDBOracle
from .prover import prove_on_graph
from .server import (
    SERVER_INSTRUCTIONS,
    cge_do,
    cge_prove,
    cge_solve,
    cge_trace,
    create_mcp_server,
    default_oracle,
    mcp,
)

__all__ = [
    "SERVER_INSTRUCTIONS",
    "ActorDBOracle",
    "cge_do",
    "cge_prove",
    "cge_solve",
    "cge_trace",
    "create_mcp_server",
    "default_oracle",
    "mcp",
    "prove_on_graph",
]
