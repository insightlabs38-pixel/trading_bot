"""Tests for Phase 1 common metadata and immutable run manifests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from trading_bot.config import AppConfig, load_config
from trading_bot.metadata import (
    CampaignId,
    CheckpointId,
    ContainerMetadata,
    DatasetVersion,
    EnvironmentMetadata,
    GitMetadata,
    ModelConfigId,
    PackageVersion,
    PredictionArtifactId,
    RunManifest,
    SplitVersion,
    TrialId,
    build_run_manifest,
    capture_container_metadata,
    capture_environment_metadata,
    capture_git_metadata,
    config_sha256,
    model_config_id,
)

EXAMPLE_CONFIG = Path(__file__).parents[2] / "configs" / "examples" / "minimal.yaml"


def test_identifier_types_validate_safe_values() -> None:
    assert str(DatasetVersion("dataset_dev_v001")) == "dataset_dev_v001"
    assert str(SplitVersion("split-2026.08-v1")) == "split-2026.08-v1"
    assert str(ModelConfigId("model_fixture")) == "model_fixture"
    assert str(TrialId("trial_0041_r1")) == "trial_0041_r1"
    assert str(CampaignId("campaign_2026_08")) == "campaign_2026_08"
    assert str(CheckpointId("checkpoint_0041")) == "checkpoint_0041"
    assert str(PredictionArtifactId("predictions_0041")) == "predictions_0041"


@pytest.mark.parametrize("value", ["", " leading", "trailing ", "has/slash", "has space"])
def test_identifiers_reject_unsafe_values(value: str) -> None:
    with pytest.raises(ValidationError):
        DatasetVersion(value)


def test_config_hash_is_stable_and_order_independent() -> None:
    config = load_config(EXAMPLE_CONFIG, environ={})
    payload = config.model_dump(mode="python")
    payload["model"]["parameters"] = {"layers": 2, "hidden_dim": 128}
    reordered = AppConfig.model_validate(payload)
    assert config_sha256(config) == config_sha256(reordered)
    expected = "a889110ed583183b4b34fea68745f06103e624ed0c88d2851049eea1b15ac34b"
    assert config_sha256(config) == expected


def test_config_hash_changes_for_material_configuration_change() -> None:
    config = load_config(EXAMPLE_CONFIG, environ={})
    payload = config.model_dump(mode="python")
    payload["training"]["seed"] = 43
    changed = AppConfig.model_validate(payload)
    assert config_sha256(config) != config_sha256(changed)


def test_model_config_id_is_content_derived_and_stable() -> None:
    config = load_config(EXAMPLE_CONFIG, environ={})
    payload = config.model_dump(mode="python")
    payload["model"]["parameters"] = {"layers": 2, "hidden_dim": 128}
    reordered = AppConfig.model_validate(payload)
    assert model_config_id(config.model) == model_config_id(reordered.model)
    assert str(model_config_id(config.model)).startswith("model_")


def test_git_capture_supports_environment_override() -> None:
    metadata = capture_git_metadata(
        environ={
            "TRADING_BOT_GIT_SHA": "ABCDEF1234567",
            "TRADING_BOT_GIT_BRANCH": "phase1-common-metadata",
            "TRADING_BOT_GIT_DIRTY": "false",
        }
    )
    assert metadata.sha == "abcdef1234567"
    assert metadata.branch == "phase1-common-metadata"
    assert metadata.dirty is False
    assert metadata.source == "environment"


def test_git_capture_reads_real_repository_state(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Metadata Test"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=tmp_path, check=True)

    clean = capture_git_metadata(tmp_path, environ={})
    assert clean.branch == "main"
    assert clean.dirty is False
    assert clean.source == "git"

    tracked.write_text("dirty\n", encoding="utf-8")
    dirty = capture_git_metadata(tmp_path, environ={})
    assert dirty.sha == clean.sha
    assert dirty.dirty is True


def test_container_capture_is_allowlisted() -> None:
    metadata = capture_container_metadata(
        {
            "TRADING_BOT_IMAGE_DIGEST": "sha256:abc",
            "TRADING_BOT_IMAGE_TAG": "trading-cpu:test",
            "TRADING_BOT_CONTAINER_RUNTIME": "docker",
            "SECRET_TOKEN": "must-not-be-captured",
        }
    )
    assert metadata == ContainerMetadata(
        image_digest="sha256:abc",
        image_tag="trading-cpu:test",
        runtime="docker",
    )
    assert "SECRET_TOKEN" not in metadata.model_dump_json()


def test_environment_capture_reports_requested_package_versions() -> None:
    metadata = capture_environment_metadata(packages=("pydantic", "definitely-not-installed"))
    assert metadata.python_version
    versions = {item.name: item.version for item in metadata.package_versions}
    assert versions["pydantic"]
    assert versions["definitely-not-installed"] is None


def test_run_manifest_is_frozen_redacted_and_complete() -> None:
    config = load_config(EXAMPLE_CONFIG, environ={})
    git = GitMetadata(sha="abcdef1234567", source="explicit")
    environment = EnvironmentMetadata(
        python_version="3.12.0",
        python_implementation="CPython",
        platform="test-platform",
        system="Linux",
        machine="x86_64",
        package_versions=(PackageVersion(name="pydantic", version="2.test"),),
    )
    manifest = build_run_manifest(
        config,
        split_version="split_dev_v001",
        trial_id="trial_dev_0001",
        git=git,
        container=ContainerMetadata(image_tag="trading-cpu:test"),
        environment=environment,
        created_at_utc=datetime(2026, 8, 18, 7, 0, tzinfo=UTC),
    )
    assert isinstance(manifest, RunManifest)
    assert str(manifest.dataset_version) == "dataset_dev_v001"
    assert str(manifest.split_version) == "split_dev_v001"
    assert str(manifest.campaign_id) == "dev_smoke"
    assert manifest.config_sha256 == config_sha256(config)
    assert json.loads(manifest.config_canonical_json)["notifications"]["webhook_url"] is None
    with pytest.raises(ValidationError):
        manifest.seed = 999  # type: ignore[misc]


def test_manifest_rejects_naive_timestamp() -> None:
    config = load_config(EXAMPLE_CONFIG, environ={})
    with pytest.raises(ValidationError, match="timezone-aware"):
        build_run_manifest(
            config,
            git=GitMetadata(sha="abcdef1234567", source="explicit"),
            container=ContainerMetadata(),
            environment=capture_environment_metadata(packages=()),
            created_at_utc=datetime(2026, 8, 18, 7, 0),
        )


def test_manifest_cli_generates_json_without_market_data_or_gpu(tmp_path: Path) -> None:
    output_path = tmp_path / "run_manifest.json"
    environment = os.environ.copy()
    source_root = str(Path(__file__).parents[2] / "src")
    environment["PYTHONPATH"] = source_root
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "trading_bot.metadata",
            str(EXAMPLE_CONFIG),
            "--split-version",
            "split_dev_v001",
            "--trial-id",
            "trial_dev_0001",
            "--git-sha",
            "abcdef1234567",
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["dataset_version"] == "dataset_dev_v001"
    assert payload["split_version"] == "split_dev_v001"
    assert payload["trial_id"] == "trial_dev_0001"
    assert payload["git"]["sha"] == "abcdef1234567"
