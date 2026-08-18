"""Dependency-light capture of Git, container, and runtime environment metadata."""

from __future__ import annotations

import os
import platform
import re
import subprocess
from collections.abc import Iterable, Mapping
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Literal

from pydantic import field_validator

from trading_bot.config.base import FrozenConfigModel

_GIT_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{7,64}$")
_DEFAULT_PACKAGES = ("pydantic", "PyYAML", "torch", "triton", "transformer-engine")


class RuntimeMetadataError(RuntimeError):
    """Raised when required reproducibility metadata cannot be captured."""


class GitMetadata(FrozenConfigModel):
    """Source revision metadata for a run."""

    sha: str
    branch: str | None = None
    dirty: bool | None = None
    source: Literal["git", "environment", "explicit"] = "git"

    @field_validator("sha")
    @classmethod
    def normalize_sha(cls, value: str) -> str:
        if not _GIT_SHA_PATTERN.fullmatch(value):
            raise ValueError("Git SHA must contain 7-64 hexadecimal characters")
        return value.lower()


class ContainerMetadata(FrozenConfigModel):
    """Container image identity when the process is containerized."""

    image_digest: str | None = None
    image_tag: str | None = None
    runtime: str | None = None


class PackageVersion(FrozenConfigModel):
    """Resolved package version, or ``None`` when the package is unavailable."""

    name: str
    version: str | None


class EnvironmentMetadata(FrozenConfigModel):
    """Portable runtime facts useful for reproducing an execution."""

    python_version: str
    python_implementation: str
    platform: str
    system: str
    machine: str
    package_versions: tuple[PackageVersion, ...] = ()


def _run_git(repo_path: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeMetadataError(f"unable to capture Git metadata: {exc}") from exc
    return result.stdout.strip()


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise RuntimeMetadataError(f"invalid boolean environment value: {value!r}")


def capture_git_metadata(
    repo_path: str | Path = ".",
    environ: Mapping[str, str] | None = None,
) -> GitMetadata:
    """Capture Git SHA/branch/dirty state, honoring explicit container overrides."""
    environment = os.environ if environ is None else environ
    override_sha = environment.get("TRADING_BOT_GIT_SHA")
    if override_sha is not None:
        return GitMetadata(
            sha=override_sha,
            branch=environment.get("TRADING_BOT_GIT_BRANCH"),
            dirty=_parse_optional_bool(environment.get("TRADING_BOT_GIT_DIRTY")),
            source="environment",
        )

    path = Path(repo_path)
    sha = _run_git(path, "rev-parse", "HEAD")
    branch_value = _run_git(path, "branch", "--show-current")
    status = _run_git(path, "status", "--porcelain")
    return GitMetadata(
        sha=sha,
        branch=branch_value or None,
        dirty=bool(status),
        source="git",
    )


def capture_container_metadata(
    environ: Mapping[str, str] | None = None,
) -> ContainerMetadata:
    """Capture only allowlisted container identity variables, never arbitrary environment data."""
    environment = os.environ if environ is None else environ
    return ContainerMetadata(
        image_digest=environment.get("TRADING_BOT_IMAGE_DIGEST"),
        image_tag=environment.get("TRADING_BOT_IMAGE_TAG"),
        runtime=environment.get("TRADING_BOT_CONTAINER_RUNTIME"),
    )


def capture_environment_metadata(
    packages: Iterable[str] = _DEFAULT_PACKAGES,
) -> EnvironmentMetadata:
    """Capture Python/platform facts and package versions without importing heavy libraries."""
    versions: list[PackageVersion] = []
    for package in sorted(set(packages)):
        try:
            version = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            version = None
        versions.append(PackageVersion(name=package, version=version))

    return EnvironmentMetadata(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=platform.platform(),
        system=platform.system(),
        machine=platform.machine(),
        package_versions=tuple(versions),
    )
