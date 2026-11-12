"""Integración Git de solo lectura sin mutación del árbol de trabajo para `cge/diff/`."""
from __future__ import annotations

import io
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any, Dict, Optional, Tuple, Union

from .engine import CausalDiffEngine, CausalDiffError, CausalDiffReport


_IGNORED_WORKTREE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "node_modules", ".agents",
}


def _run_git(repo_dir: str, args: list[str], binary: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "--no-optional-locks"] + args,
        cwd=repo_dir,
        capture_output=True,
        text=not binary,
        encoding=None if binary else "utf-8",
    )


def _resolve_repo_and_subpath(
    repo_path: Union[str, Path],
    subpath: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Resuelve la raíz del repositorio Git y el subdirectorio efectivo relativo a dicha raíz."""
    raw_path = os.path.abspath(str(repo_path))
    if not os.path.exists(raw_path):
        raise CausalDiffError(f"Repository path '{repo_path}' does not exist.")

    cwd_dir = raw_path if os.path.isdir(raw_path) else os.path.dirname(raw_path)
    top_proc = _run_git(cwd_dir, ["rev-parse", "--show-toplevel"])
    if top_proc.returncode != 0:
        raise CausalDiffError(
            f"Path '{repo_path}' is not inside a valid Git repository: {(top_proc.stderr or '').strip()}"
        )

    repo_root = os.path.normpath(top_proc.stdout.strip())
    rel_from_root = os.path.relpath(raw_path, repo_root)
    implicit_sub = None if rel_from_root in (".", "") else rel_from_root.replace("\\", "/")

    if subpath:
        clean_sub = subpath.strip().strip("/\\").replace("\\", "/")
        if implicit_sub:
            effective_sub = f"{implicit_sub}/{clean_sub}"
        else:
            effective_sub = clean_sub
    else:
        effective_sub = implicit_sub

    return repo_root, effective_sub


def _materialize_revision(
    repo_root: str,
    ref: str,
    dest_dir: str,
    effective_subpath: Optional[str] = None,
) -> Tuple[str, str]:
    """Extrae una revisión Git (`<commit-ish>` o `'WORKTREE'`) en `dest_dir` sin tocar el working tree.

    Retorna `(extracted_target_path, resolved_commit_or_worktree_label)`.
    """
    ref_clean = (ref or "").strip()
    if not ref_clean:
        raise CausalDiffError("Git revision reference cannot be empty.")

    if ref_clean.upper() == "WORKTREE":
        src_target = (
            os.path.join(repo_root, *effective_subpath.split("/"))
            if effective_subpath else repo_root
        )
        if not os.path.exists(src_target):
            raise CausalDiffError(f"Subpath '{effective_subpath}' does not exist in WORKTREE.")

        folder_name = os.path.basename(src_target.rstrip("/\\")) or "worktree_model"
        out_target = os.path.join(dest_dir, folder_name)
        if os.path.isfile(src_target):
            os.makedirs(dest_dir, exist_ok=True)
            out_file = os.path.join(dest_dir, os.path.basename(src_target))
            shutil.copy2(src_target, out_file)
            return out_file, "WORKTREE"

        def _ignore_fn(_dir: str, names: list[str]) -> set[str]:
            return {n for n in names if n in _IGNORED_WORKTREE_DIRS}

        shutil.copytree(src_target, out_target, ignore=_ignore_fn)
        return out_target, "WORKTREE"

    # Verificar que la referencia Git resuelve a un commit válido
    rev_proc = _run_git(repo_root, ["rev-parse", "--verify", f"{ref_clean}^{{commit}}"])
    if rev_proc.returncode != 0:
        raise CausalDiffError(
            f"Invalid Git revision '{ref_clean}' in '{repo_root}': {(rev_proc.stderr or '').strip()}"
        )
    commit_sha = rev_proc.stdout.strip()

    archive_args = ["archive", "--format=tar", commit_sha]
    if effective_subpath:
        archive_args.extend(["--", effective_subpath])

    arc_proc = _run_git(repo_root, archive_args, binary=True)
    if arc_proc.returncode != 0:
        err_msg = arc_proc.stderr.decode("utf-8", errors="replace").strip() if arc_proc.stderr else "git archive failed"
        raise CausalDiffError(
            f"Failed to archive Git revision '{ref_clean}' ({commit_sha[:8]}): {err_msg}"
        )

    raw_bytes = arc_proc.stdout or b""
    with tarfile.open(fileobj=io.BytesIO(raw_bytes), mode="r:") as tf:
        members = tf.getmembers()
        if not members:
            raise CausalDiffError(
                f"Path '{effective_subpath or '.'}' does not exist or is empty at Git revision '{ref_clean}' ({commit_sha[:8]})."
            )
        tf.extractall(path=dest_dir, filter="data")

    if effective_subpath:
        extracted = os.path.join(dest_dir, *effective_subpath.split("/"))
        if not os.path.exists(extracted):
            raise CausalDiffError(
                f"Extracted archive for '{ref_clean}' did not contain '{effective_subpath}'."
            )
        return extracted, commit_sha

    return dest_dir, commit_sha


def diff_git_revisions(
    repo_path: Union[str, Path] = ".",
    base_ref: str = "HEAD~1",
    head_ref: str = "HEAD",
    subpath: Optional[str] = None,
    entrypoint: Optional[str] = None,
    optimality_tol: float = 1e-3,
    fail_on_dead_nodes: bool = True,
    engine: Optional[CausalDiffEngine] = None,
) -> CausalDiffReport:
    """Compara dos estados de un repositorio vía Git (`base_ref` vs `head_ref` o `'WORKTREE'`)."""
    repo_root, effective_subpath = _resolve_repo_and_subpath(repo_path, subpath=subpath)
    diff_engine = engine or CausalDiffEngine(
        optimality_tol=optimality_tol,
        fail_on_dead_nodes=fail_on_dead_nodes,
    )

    with tempfile.TemporaryDirectory(prefix="cge_git_diff_") as tmp_root:
        base_dest = os.path.join(tmp_root, "base_rev")
        head_dest = os.path.join(tmp_root, "head_rev")
        os.makedirs(base_dest, exist_ok=True)
        os.makedirs(head_dest, exist_ok=True)

        base_extracted, base_sha = _materialize_revision(
            repo_root=repo_root,
            ref=base_ref,
            dest_dir=base_dest,
            effective_subpath=effective_subpath,
        )
        head_extracted, head_sha = _materialize_revision(
            repo_root=repo_root,
            ref=head_ref,
            dest_dir=head_dest,
            effective_subpath=effective_subpath,
        )

        git_meta: Dict[str, Any] = {
            "repo_root": repo_root,
            "subpath": effective_subpath,
            "base_ref": base_ref,
            "base_commit": base_sha,
            "head_ref": head_ref,
            "head_commit": head_sha,
        }

        report = diff_engine.diff(
            base=base_extracted,
            head=head_extracted,
            entrypoint=entrypoint,
            git_metadata=git_meta,
        )
        return report
