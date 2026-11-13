"""Punto de entrada CLI para ejecutar el Servidor MCP Causal de ActorDB (`python -m cge.mcp`)."""
from __future__ import annotations

import argparse
from typing import List, Optional

from .server import create_mcp_server


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m cge.mcp",
        description="ActorDB & Causal Graph Engine (CGE) Standard MCP Server",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Default repository directory, Python file, or canonical schema.json to bind.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport protocol (default: stdio).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host for HTTP/SSE transports (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind port for HTTP/SSE transports (default: 8000).",
    )
    parser.add_argument(
        "--no-auto-discovery",
        action="store_true",
        help="Disable automatic workspace CausalGraph discovery when no explicit source is passed.",
    )
    args = parser.parse_args(argv)

    srv = create_mcp_server(
        default_source=args.source,
        allow_auto_discovery=not args.no_auto_discovery,
        host=args.host,
        port=args.port,
    )
    srv.run(transport=args.transport)


if __name__ == "__main__":
    main()
