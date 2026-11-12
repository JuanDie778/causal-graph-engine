"""Motor de Causal Diff de ActorDB (`cge.diff`)."""
from .engine import (
    CausalDiffEngine,
    CausalDiffError,
    CausalDiffReport,
    DiffVerdict,
    diff_graphs,
    diff_sources,
    load_causal_graph,
)
from .git_integration import diff_git_revisions

__all__ = [
    "CausalDiffEngine",
    "CausalDiffError",
    "CausalDiffReport",
    "DiffVerdict",
    "diff_graphs",
    "diff_sources",
    "diff_git_revisions",
    "load_causal_graph",
]
