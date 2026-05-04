"""Clone stage — git repository cloning and source updates."""

import logging
import subprocess
from pathlib import Path

from .._context import _BuildContext
from ..models import BuildProgress
from ..utils import (
    MSG_SOURCES_ALREADY_EXIST,
    _format_command,
    _redact_build_text,
    _tail_lines,
)

logger = logging.getLogger(__name__)


def run_clone(ctx: _BuildContext) -> BuildProgress:
    """Clone or update git repository source."""
    progress = BuildProgress(
        stage="clone",
        status="running",
        message="Cloning repository...",
        progress_percent=20,
    )

    logger.info("[clone] checking source_dir=%s", ctx.config.source_dir)
    logger.debug(
        "[clone] source_exists=%s is_git_repo=%s",
        source_exists(ctx.config.source_dir),
        is_git_repo(ctx.config.source_dir) if source_exists(ctx.config.source_dir) else False,
    )

    if source_exists(ctx.config.source_dir):
        result = _handle_existing_source(ctx, progress)
        if result is not None:
            return result

    logger.info(
        "[clone] source missing; cloning from %s",
        _redact_build_text(ctx.config.git_remote_url),
    )
    logger.debug(
        "[clone] branch=%s shallow=%s target=%s",
        ctx.config.git_branch,
        ctx.config.shallow_clone,
        ctx.config.source_dir,
    )
    source_existed_before_clone = source_exists(ctx.config.source_dir)
    return _execute_clone(ctx, progress, source_existed_before_clone=source_existed_before_clone)


def is_git_repo(source_dir: Path) -> bool:
    """Check if source directory contains a valid git repository."""
    git_dir = source_dir / ".git"
    return git_dir.exists() and git_dir.is_dir()


def source_exists(source_dir: Path) -> bool:
    """Check if source directory exists and is non-empty."""
    if not source_dir.exists():
        return False
    return any(source_dir.iterdir())


def _handle_existing_source(ctx: _BuildContext, progress: BuildProgress) -> BuildProgress | None:
    """Return a BuildProgress to skip clone, or None to proceed with cloning."""
    if is_git_repo(ctx.config.source_dir):
        if ctx.config.update_sources:
            logger.info("[clone] source exists and update_sources=True; updating existing clone")
            return _update_sources(ctx, progress)
        if ctx.config.git_commit:
            logger.info(
                "[clone] source exists; checking out configured git_commit=%s",
                ctx.config.git_commit,
            )
            return _checkout_commit(ctx, progress)
    logger.info("[clone] source exists; skipping clone")
    progress.status = "skipped"
    progress.message = MSG_SOURCES_ALREADY_EXIST
    progress.progress_percent = 30
    return progress


def _execute_clone(
    ctx: _BuildContext, progress: BuildProgress, source_existed_before_clone: bool = False
) -> BuildProgress:
    """Execute the git clone operation."""
    try:
        if ctx.dry_run:
            depth_flag = " --depth 1" if ctx.config.shallow_clone else ""
            progress.message = (
                f"Would run: git clone --branch {ctx.config.git_branch}{depth_flag} "
                f"{_redact_build_text(ctx.config.git_remote_url)} {ctx.config.source_dir}"
            )
            progress.status = "success"
            progress.progress_percent = 30
            logger.info("[clone] dry-run: %s", progress.message)
            return progress

        ctx.config.source_dir.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["git", "clone", "--branch", ctx.config.git_branch]
        if ctx.config.shallow_clone:
            cmd.extend(["--depth", "1"])
        cmd.extend([ctx.config.git_remote_url, str(ctx.config.source_dir)])

        logger.debug("[clone] running: %s", _format_command(cmd))
        clone_timeout = getattr(ctx.config, "clone_timeout", 120)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False, timeout=clone_timeout
            )
        except subprocess.TimeoutExpired as e:
            ctx.append_command_output(
                stage="clone",
                command=cmd,
                returncode=-1,
                stdout="",
                stderr=f"Git clone timed out after {clone_timeout}s: {e}",
            )
            raise

        ctx.append_command_output(
            stage="clone",
            command=cmd,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, cmd, output=result.stdout, stderr=result.stderr
            )

        progress.message = f"Cloned {_redact_build_text(ctx.config.git_remote_url)}"
        progress.status = "success"
        progress.progress_percent = 30
        logger.info("[clone] cloned successfully into %s", ctx.config.source_dir)

        if ctx.config.git_commit:
            progress = _checkout_commit(ctx, progress)
            if progress.status != "success":
                return progress

    except Exception as e:
        return _handle_clone_error(ctx, progress, e, source_existed_before_clone)

    return progress


def _handle_clone_error(
    ctx: _BuildContext,
    progress: BuildProgress,
    error: Exception,
    source_existed_before_clone: bool = False,
) -> BuildProgress:
    """Handle clone failure with offline-continue support."""
    if source_existed_before_clone or source_exists(ctx.config.source_dir):
        logger.warning("[clone] error but source available; continuing offline: %s", str(error))
        progress.status = "skipped"
        progress.message = MSG_SOURCES_ALREADY_EXIST
        progress.progress_percent = 30
    else:
        stderr = _redact_build_text(getattr(error, "stderr", None) or str(error))
        logger.error("[clone] clone failed: %s", stderr)
        progress.status = "failed"
        progress.message = f"Clone failed: {stderr}"
    return progress


def _checkout_commit(ctx: _BuildContext, progress: BuildProgress) -> BuildProgress:
    """Checkout a specific commit hash if configured."""
    if not ctx.config.git_commit:
        return progress

    commit = ctx.config.git_commit
    logger.info("[clone] checking out commit %s", commit)

    if ctx.dry_run:
        logger.info("[clone] dry-run: would checkout commit %s", commit)
        return progress

    try:
        checkout_cmd = ["git", "checkout", commit]
        logger.debug("[clone] running: %s", _format_command(checkout_cmd))
        result = subprocess.run(
            checkout_cmd,
            cwd=ctx.config.source_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        ctx.append_command_output(
            stage="clone (checkout)",
            command=checkout_cmd,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        if result.returncode != 0:
            redacted_stderr = _redact_build_text(_tail_lines(result.stderr))
            logger.warning(
                "[clone] commit checkout failed (rc=%s): %s",
                result.returncode,
                redacted_stderr,
            )
            progress.status = "skipped"
            progress.message = (
                f"Commit checkout failed; continuing with existing sources: {redacted_stderr}"
            )
            progress.progress_percent = 30
        else:
            logger.info("[clone] checked out commit %s", commit)
    except subprocess.SubprocessError as e:
        logger.warning("[clone] commit checkout error: %s", str(e))
        progress.status = "skipped"
        progress.message = f"Commit checkout failed: {str(e)}"
        progress.progress_percent = 30

    return progress


def _update_sources(ctx: _BuildContext, progress: BuildProgress) -> BuildProgress:
    """Fetch and fast-forward an existing clone to the configured branch.

    On network failure the stage falls back to ``skipped`` so the build
    continues with the local copy.
    """
    logger.info("[update-sources] fetching origin in %s", ctx.config.source_dir)

    if ctx.dry_run:
        progress.message = (
            f"Would run: git fetch origin && git checkout -B "
            f"{ctx.config.git_branch} origin/{ctx.config.git_branch}"
        )
        progress.status = "success"
        progress.progress_percent = 30
        logger.info("[update-sources] dry-run: %s", progress.message)
        return progress

    try:
        fetch_cmd = ["git", "fetch", "origin"]
        logger.debug("[update-sources] running: %s", _format_command(fetch_cmd))
        fetch_result = subprocess.run(
            fetch_cmd,
            cwd=ctx.config.source_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        ctx.append_command_output(
            stage="clone (fetch)",
            command=fetch_cmd,
            returncode=fetch_result.returncode,
            stdout=fetch_result.stdout,
            stderr=fetch_result.stderr,
        )
        if fetch_result.returncode != 0:
            err_msg = (fetch_result.stderr or fetch_result.stdout or "").lower()
            _NETWORK_KEYWORDS = (
                "network",
                "connect",
                "resolve",
                "timeout",
                "unreachable",
                "temporary failure",
            )
            if any(kw in err_msg for kw in _NETWORK_KEYWORDS):
                redacted_err_msg = _redact_build_text(err_msg)
                logger.warning("[update-sources] network error during fetch: %s", redacted_err_msg)
                progress.status = "skipped"
                progress.message = "Network unavailable; continuing with existing sources"
                progress.progress_percent = 30
                return progress
            redacted_err = _redact_build_text(_tail_lines(fetch_result.stderr))
            logger.error(
                "[update-sources] fetch failed (rc=%s): %s", fetch_result.returncode, redacted_err
            )
            progress.status = "failed"
            progress.message = f"Fetch failed: {redacted_err}"
            progress.progress_percent = 30
            return progress
        logger.info("[update-sources] fetch completed")
        logger.debug("[update-sources] fetch stdout: %s", fetch_result.stdout.strip() or "(empty)")

        checkout_cmd = [
            "git",
            "checkout",
            "-B",
            ctx.config.git_branch,
            f"origin/{ctx.config.git_branch}",
        ]
        logger.debug("[update-sources] running: %s", _format_command(checkout_cmd))
        result = subprocess.run(
            checkout_cmd,
            cwd=ctx.config.source_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        ctx.append_command_output(
            stage="clone (update)",
            command=checkout_cmd,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

        if result.returncode != 0:
            redacted_stderr = _redact_build_text(_tail_lines(result.stderr))
            logger.warning(
                "[update-sources] checkout failed (rc=%s): %s",
                result.returncode,
                redacted_stderr,
            )
            progress.status = "skipped"
            progress.message = (
                f"Source update failed; continuing with existing sources: {redacted_stderr}"
            )
            progress.progress_percent = 30
            return progress

        progress.message = f"Updated sources to origin/{ctx.config.git_branch}"
        progress.status = "success"
        progress.progress_percent = 30
        logger.info("[update-sources] checked out %s", progress.message)

        if ctx.config.git_commit:
            progress = _checkout_commit(ctx, progress)

    except subprocess.SubprocessError as e:
        logger.warning("[update-sources] network error during update: %s", str(e))
        progress.status = "skipped"
        progress.message = "Network unavailable; continuing with existing sources"
        progress.progress_percent = 30

    return progress
