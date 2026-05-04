"""Finalize stage — binary discovery, provenance recording."""

import logging
import subprocess
from pathlib import Path

from ...common.file_ops import atomic_write_json
from .._context import _BuildContext
from ..models import BuildArtifact, BuildProgress

logger = logging.getLogger(__name__)


def run_finalize(ctx: _BuildContext, build_progress: BuildProgress) -> BuildArtifact | None:
    """Collect metadata and write build provenance atomically."""
    if not build_progress.is_complete or build_progress.status != "success":
        logger.warning("[finalize] build stage incomplete or failed; skipping finalize")
        return None

    logger.info("[finalize] collecting build metadata")

    git_commit_sha = "unknown"
    if not ctx.dry_run:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ctx.config.source_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            git_commit_sha = result.stdout.strip()
            logger.info("[finalize] git commit=%s", git_commit_sha)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("[finalize] could not determine git commit SHA")

    binary_path = None
    binary_size_bytes = None
    build_dir_bin = ctx.config.build_dir / "bin"
    if build_dir_bin.exists():
        server_binary = build_dir_bin / "llama-server"
        if server_binary.exists():
            binary_path = server_binary
            binary_size_bytes = server_binary.stat().st_size
            logger.info("[finalize] found binary: %s (%s bytes)", binary_path, binary_size_bytes)
        else:
            logger.warning("[finalize] expected binary not found: %s", server_binary)
    else:
        logger.warning("[finalize] build bin/ directory not found: %s", build_dir_bin)

    build_log_path = ctx.write_build_log()
    if build_log_path:
        logger.info("[finalize] build log written to %s", build_log_path)
    else:
        logger.debug("[finalize] no build log to write")

    artifact = ctx.create_artifact(
        exit_code=0,
        binary_path=binary_path,
        binary_size_bytes=binary_size_bytes,
        build_log_path=build_log_path,
        failure_report_path=None,
        git_commit_sha=git_commit_sha,
    )

    logger.info(
        "[finalize] writing provenance to %s", ctx.config.output_dir / "build-artifact.json"
    )
    if write_provenance(artifact, ctx.config.output_dir):
        logger.info("[finalize] provenance written successfully")
        return artifact
    logger.error("[finalize] provenance write failed")
    return None


def write_provenance(artifact: BuildArtifact, output_dir: Path) -> bool:
    """Write build artifact provenance atomically (temp-file + rename)."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        artifact_data = artifact.to_dict()
        final_file = output_dir / "build-artifact.json"
        atomic_write_json(final_file, artifact_data, verify_permissions=True)

        logger.debug("[provenance] atomically wrote %s", final_file)
        return True

    except (OSError, ValueError, TypeError) as e:
        logger.warning("[provenance] failed to write: %s", e)
        return False
