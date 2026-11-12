"""CLI de Causal Diff para integración continua CI/CD (`python -m cge.diff`)."""
from __future__ import annotations

import argparse
import contextlib
import sys
from typing import List, Optional

from .engine import CausalDiffError, DiffVerdict, diff_sources
from .git_integration import diff_git_revisions


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cge.diff",
        description="ActorDB Causal Diff Engine for CI/CD Pull Request Verification",
    )
    parser.add_argument("base", nargs="?", help="Base causal graph, directory, or source file.")
    parser.add_argument("head", nargs="?", help="Head causal graph, directory, or source file.")
    parser.add_argument("--git-repo", default=None, help="Git repository path to diff revisions.")
    parser.add_argument("--base-ref", default="HEAD~1", help="Base Git ref (default: HEAD~1).")
    parser.add_argument("--head-ref", default="HEAD", help="Head Git ref or WORKTREE (default: HEAD).")
    parser.add_argument("--subpath", default=None, help="Optional subdirectory inside the Git repository.")
    parser.add_argument("--entrypoint", default=None, help="Optional target entrypoint function name.")
    parser.add_argument("--format", choices=["json", "markdown"], default="json", help="Output format.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Return exit code 1 if verdict == FAIL (for CI/CD gates).",
    )
    args = parser.parse_args(argv)

    try:
        with contextlib.redirect_stdout(sys.stderr):
            if args.git_repo is not None:
                report = diff_git_revisions(
                    repo_path=args.git_repo,
                    base_ref=args.base_ref,
                    head_ref=args.head_ref,
                    subpath=args.subpath,
                    entrypoint=args.entrypoint,
                )
            elif args.base is not None and args.head is not None:
                report = diff_sources(
                    base_source=args.base,
                    head_source=args.head,
                    entrypoint=args.entrypoint,
                )
            else:
                parser.error("Provide either positional `base` and `head` sources, or `--git-repo <path>`.")
                return 2
    except CausalDiffError as exc:
        sys.stderr.write(f"CausalDiffError: {exc}\n")
        return 2

    if args.format == "markdown":
        sys.stdout.write(report.to_markdown() + "\n")
    else:
        sys.stdout.write(report.to_json() + "\n")

    if args.check and report.verdict == DiffVerdict.FAIL.value:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
