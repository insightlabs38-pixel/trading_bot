"""Durability regressions for local atomic storage publication."""

from __future__ import annotations

from pathlib import Path

from trading_bot.storage import LocalStorageBackend
from trading_bot.storage import local as local_module
from trading_bot.storage.base import fsync_directory


def test_directory_fsync_accepts_real_directory(tmp_path: Path) -> None:
    fsync_directory(tmp_path)


def test_local_mutations_flush_parent_directory_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    flushed: list[Path] = []
    monkeypatch.setattr(
        local_module,
        "fsync_directory",
        lambda path: flushed.append(Path(path)),
    )

    backend = LocalStorageBackend(tmp_path / "objects")
    source = tmp_path / "source.bin"
    source.write_bytes(b"durable-payload")

    backend.upload(source, "data/source.bin")
    backend.copy("data/source.bin", "copies/source.bin")
    destination = tmp_path / "restored" / "source.bin"
    backend.download("copies/source.bin", destination)
    backend.delete("data/source.bin")

    assert flushed == [
        tmp_path / "objects" / "data",
        tmp_path / "objects" / "copies",
        tmp_path / "restored",
        tmp_path / "objects" / "data",
    ]
