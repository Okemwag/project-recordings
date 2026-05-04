"""Tests for Common Voice dataset ingestion."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pandas as pd
import pytest

from src.datasets.common_voice import (
    ArchiveExtractionError,
    build_common_voice_metadata,
    discover_common_voice_root,
    extract_archive,
    find_archives,
)


def _write_common_voice_fixture(root: Path) -> Path:
    cv_root = root / "cv-corpus-25.0-2026-03-09" / "sw"
    clips = cv_root / "clips"
    clips.mkdir(parents=True)
    for name in ("a.wav", "b.wav", "c.wav", "d.wav", "e.wav"):
        (clips / name).write_bytes(b"audio")

    pd.DataFrame(
        [
            {"client_id": "s1", "path": "a.wav", "sentence": "Habari", "accents": "coastal"},
            {"client_id": "s2", "path": "b.wav", "sentence": "Habari yako", "accents": "nairobi"},
            {"client_id": "s3", "path": "c.wav", "sentence": "Karibu", "accents": ""},
            {"client_id": "s4", "path": "d.wav", "sentence": "", "accents": "upcountry"},
            {"client_id": "s5", "path": "missing.wav", "sentence": "Hujambo", "accents": "coastal"},
        ]
    ).to_csv(cv_root / "validated.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {"clip": "a.wav", "duration[ms]": 1200},
            {"clip": "b.wav", "duration[ms]": 1500},
            {"clip": "c.wav", "duration[ms]": 900},
        ]
    ).to_csv(cv_root / "clip_durations.tsv", sep="\t", index=False)
    return cv_root


def test_find_archives_lists_supported_files(tmp_path):
    (tmp_path / "dataset.tar").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    assert find_archives(tmp_path) == [tmp_path / "dataset.tar"]


def test_extract_archive_and_discover_common_voice_root(tmp_path):
    source = tmp_path / "source"
    cv_root = _write_common_voice_fixture(source)
    archive = tmp_path / "cv.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(source / "cv-corpus-25.0-2026-03-09", arcname="cv-corpus-25.0-2026-03-09")

    dest = tmp_path / "raw"
    result = extract_archive(archive, dest)

    assert result.extracted_files >= 7
    assert result.common_voice_root == dest / cv_root.relative_to(source)
    assert discover_common_voice_root(dest) == result.common_voice_root


def test_extract_archive_rejects_corrupt_tar(tmp_path):
    archive = tmp_path / "broken.tar"
    archive.write_bytes(b"not a valid tar")

    with pytest.raises(ArchiveExtractionError, match="incomplete|corrupted"):
        extract_archive(archive, tmp_path / "raw")


def test_build_common_voice_metadata_from_validated_tsv(tmp_path):
    cv_root = _write_common_voice_fixture(tmp_path)
    output = tmp_path / "metadata.csv"

    result = build_common_voice_metadata(
        common_voice_root=cv_root,
        output_csv=output,
        data_dir=tmp_path,
        label_strategy="first_word",
    )

    frame = pd.read_csv(output)
    assert result.rows == 3
    assert result.skipped_rows == 2
    assert set(frame.columns) == {
        "file_path",
        "word_label",
        "accent_label",
        "speaker_id",
        "duration_sec",
        "split",
    }
    assert set(frame["word_label"]) == {"habari", "karibu"}
    assert set(frame["split"]) == {"train", "test"}
    assert frame.loc[frame["word_label"] == "karibu", "accent_label"].iloc[0] == "unknown"


def test_build_common_voice_metadata_can_filter_single_word_prompts(tmp_path):
    cv_root = _write_common_voice_fixture(tmp_path)

    result = build_common_voice_metadata(
        common_voice_root=cv_root,
        output_csv=tmp_path / "metadata.csv",
        data_dir=tmp_path,
        single_word_only=True,
    )

    frame = pd.read_csv(result.output_csv)
    assert set(frame["word_label"]) == {"habari", "karibu"}
