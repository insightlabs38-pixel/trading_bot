"""Command-line run-manifest generator for Phase 1 validation and later workflows."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from trading_bot.config import load_config
from trading_bot.metadata.manifest import build_run_manifest
from trading_bot.metadata.runtime import GitMetadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an immutable trading-bot run manifest.")
    parser.add_argument("config", type=Path, help="Validated YAML configuration path")
    parser.add_argument("--split-version")
    parser.add_argument("--trial-id")
    parser.add_argument("--parent-trial-id")
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--prediction-artifact-id")
    parser.add_argument(
        "--git-sha",
        help="Explicit Git SHA override for packaged/sandboxed validation environments",
    )
    parser.add_argument("--output", type=Path, help="Write JSON to a file instead of stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load a config, capture metadata, and emit a JSON run manifest."""
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    git = None if args.git_sha is None else GitMetadata(sha=args.git_sha, source="explicit")
    manifest = build_run_manifest(
        config,
        split_version=args.split_version,
        trial_id=args.trial_id,
        parent_trial_id=args.parent_trial_id,
        checkpoint_id=args.checkpoint_id,
        prediction_artifact_id=args.prediction_artifact_id,
        git=git,
        environ=os.environ,
    )
    output = manifest.model_dump_json(indent=2) + "\n"
    if args.output is None:
        print(output, end="")
    else:
        args.output.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
