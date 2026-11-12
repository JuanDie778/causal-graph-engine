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
    "ActorDBOracle",
    "SERVER_INSTRUCTIONS",
    "create_mcp_server",
    "mcp",
    "default_oracle",
    "cge_trace",
    "cge_do",
    "cge_solve",
    "cge_prove",
    "prove_on_graph",
]
