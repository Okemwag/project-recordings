"""Common Voice archive extraction and metadata generation."""

from __future__ import annotations

import csv
import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


COMMON_VOICE_TSVS = ("train.tsv", "dev.tsv", "test.tsv", "validated.tsv")
ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".zip")
METADATA_COLUMNS = [
    "file_path",
    "word_label",
    "accent_label",
    "speaker_id",
    "duration_sec",
    "split",
]


class ArchiveExtractionError(RuntimeError):
    """Raised when a dataset archive cannot be extracted safely."""


@dataclass(frozen=True)
class ExtractResult:
    archive_path: Path
    destination: Path
    extracted_files: int
    common_voice_root: Path | None


@dataclass(frozen=True)
class CommonVoiceMetadataResult:
    output_csv: Path
    rows: int
    labels: int
    accents: int
    splits: dict[str, int]
    skipped_rows: int


def find_archives(raw_dir: str | Path) -> list[Path]:
    """Return supported dataset archives found directly inside raw_dir."""
    root = Path(raw_dir)
    if not root.exists():
        return []
    archives = []
    for path in root.iterdir():
        if path.is_file() and _is_supported_archive(path):
            archives.append(path)
    return sorted(archives)


def extract_archive(
    archive_path: str | Path,
    destination: str | Path | None = None,
    overwrite: bool = False,
) -> ExtractResult:
    """Safely extract a .tar/.tar.gz/.tgz/.zip dataset archive."""
    archive = Path(archive_path)
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")
    if not _is_supported_archive(archive):
        raise ArchiveExtractionError(f"Unsupported archive type: {archive.name}")

    dest = Path(destination) if destination else archive.parent
    dest.mkdir(parents=True, exist_ok=True)

    try:
        if zipfile.is_zipfile(archive):
            count = _extract_zip(archive, dest, overwrite)
        else:
            count = _extract_tar(archive, dest, overwrite)
    except (tarfile.TarError, zipfile.BadZipFile, EOFError, OSError) as exc:
        raise ArchiveExtractionError(
            f"Could not extract {archive.name}. The archive may be incomplete "
            f"or corrupted: {exc}"
        ) from exc

    return ExtractResult(
        archive_path=archive,
        destination=dest,
        extracted_files=count,
        common_voice_root=discover_common_voice_root(dest),
    )


def discover_common_voice_root(raw_dir: str | Path) -> Path | None:
    """Find the nearest directory that looks like a Common Voice language root."""
    root = Path(raw_dir)
    if not root.exists():
        return None
    candidates = []
    for path in [root, *root.rglob("*")]:
        if not path.is_dir():
            continue
        if any((path / name).exists() for name in COMMON_VOICE_TSVS):
            candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: len(p.parts))[0]


def build_common_voice_metadata(
    common_voice_root: str | Path,
    output_csv: str | Path,
    data_dir: str | Path,
    label_strategy: str = "sentence",
    max_rows: int | None = None,
    single_word_only: bool = False,
    min_label_count: int = 1,
    max_labels: int | None = None,
) -> CommonVoiceMetadataResult:
    """
    Convert Common Voice TSV rows into this project's metadata.csv format.

    label_strategy:
        sentence    - classify the full normalized prompt
        first_word  - classify the first token of the prompt
    """
    cv_root = Path(common_voice_root)
    output = Path(output_csv)
    data_root = Path(data_dir)
    if not cv_root.exists():
        raise FileNotFoundError(f"Common Voice root not found: {cv_root}")
    if label_strategy not in {"sentence", "first_word"}:
        raise ValueError("label_strategy must be 'sentence' or 'first_word'.")

    durations = _load_clip_durations(cv_root)
    rows: list[dict[str, object]] = []
    skipped = 0

    tsv_paths = [cv_root / name for name in ("train.tsv", "dev.tsv", "test.tsv")]
    tsv_paths = [path for path in tsv_paths if path.exists()]
    if not tsv_paths and (cv_root / "validated.tsv").exists():
        tsv_paths = [cv_root / "validated.tsv"]
    if not tsv_paths:
        raise FileNotFoundError(f"No Common Voice TSV files found in {cv_root}")

    for tsv_path in tsv_paths:
        split_name = _split_from_tsv_name(tsv_path.name)
        frame = pd.read_csv(tsv_path, sep="\t", quoting=csv.QUOTE_NONE)
        for idx, row in frame.iterrows():
            item = _row_to_metadata(
                row=row,
                row_number=idx,
                split_name=split_name,
                cv_root=cv_root,
                data_root=data_root,
                durations=durations,
                label_strategy=label_strategy,
                single_word_only=single_word_only,
            )
            if item is None:
                skipped += 1
                continue
            rows.append(item)

    if not rows:
        raise ValueError("No usable Common Voice rows were found.")

    if len(tsv_paths) == 1 and tsv_paths[0].name == "validated.tsv":
        rows = _assign_deterministic_splits(rows)

    rows = _filter_labels(rows, min_label_count=min_label_count, max_labels=max_labels)
    if not rows:
        raise ValueError("No rows remain after label filtering.")
    if max_rows is not None:
        rows = _limit_rows(rows, max_rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=METADATA_COLUMNS).to_csv(output, index=False)

    split_counts = pd.Series([row["split"] for row in rows]).value_counts().to_dict()
    return CommonVoiceMetadataResult(
        output_csv=output,
        rows=len(rows),
        labels=len({str(row["word_label"]) for row in rows}),
        accents=len({str(row["accent_label"]) for row in rows}),
        splits={str(k): int(v) for k, v in split_counts.items()},
        skipped_rows=skipped,
    )


def _is_supported_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def _extract_tar(archive: Path, dest: Path, overwrite: bool) -> int:
    count = 0
    with tarfile.open(archive, mode="r:*") as tar:
        for member in tar:
            target = dest / member.name
            _ensure_safe_path(dest, target)
            if target.exists() and not overwrite:
                continue
            tar.extract(member, dest)
            if member.isfile():
                count += 1
    return count


def _extract_zip(archive: Path, dest: Path, overwrite: bool) -> int:
    count = 0
    with zipfile.ZipFile(archive) as zip_file:
        for info in zip_file.infolist():
            target = dest / info.filename
            _ensure_safe_path(dest, target)
            if target.exists() and not overwrite:
                continue
            zip_file.extract(info, dest)
            if not info.is_dir():
                count += 1
    return count


def _ensure_safe_path(root: Path, target: Path) -> None:
    root_resolved = root.resolve()
    target_resolved = target.resolve()
    if root_resolved != target_resolved and root_resolved not in target_resolved.parents:
        raise ArchiveExtractionError(f"Archive member escapes destination: {target}")


def _load_clip_durations(cv_root: Path) -> dict[str, float]:
    path = cv_root / "clip_durations.tsv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, sep="\t")
    if frame.empty:
        return {}

    clip_col = _first_existing_column(frame, ("clip", "path", "file_path"))
    duration_col = _first_existing_column(
        frame,
        ("duration[ms]", "duration_ms", "duration", "duration_sec"),
    )
    if clip_col is None or duration_col is None:
        return {}

    durations: dict[str, float] = {}
    for _, row in frame.iterrows():
        clip = str(row[clip_col])
        raw_duration = float(row[duration_col])
        duration = raw_duration / 1000.0 if raw_duration > 100 else raw_duration
        durations[Path(clip).name] = duration
    return durations


def _first_existing_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _split_from_tsv_name(name: str) -> str:
    stem = Path(name).stem.lower()
    if stem == "dev":
        return "test"
    if stem in {"train", "test"}:
        return stem
    return "validated"


def _row_to_metadata(
    row: pd.Series,
    row_number: int,
    split_name: str,
    cv_root: Path,
    data_root: Path,
    durations: dict[str, float],
    label_strategy: str,
    single_word_only: bool,
) -> dict[str, object] | None:
    clip_path = _clean_cell(row.get("path", ""))
    sentence = _normalize_label(_clean_cell(row.get("sentence", "")))
    if not clip_path or not sentence:
        return None

    tokens = sentence.split()
    if single_word_only and len(tokens) != 1:
        return None
    label = sentence if label_strategy == "sentence" else tokens[0]

    audio_path = cv_root / "clips" / clip_path
    if not audio_path.exists():
        return None

    accent = _normalize_accent(row)
    speaker_id = str(row.get("client_id", "")).strip() or f"speaker_{row_number}"
    duration = float(durations.get(Path(clip_path).name, 0.0))
    try:
        file_path = str(audio_path.relative_to(data_root))
    except ValueError:
        file_path = str(audio_path)

    return {
        "file_path": file_path,
        "word_label": label,
        "accent_label": accent,
        "speaker_id": speaker_id,
        "duration_sec": duration,
        "split": split_name,
    }


def _normalize_label(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^\w\s'-]", "", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value)
    return value


def _clean_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_accent(row: pd.Series) -> str:
    for column in ("accents", "accent", "variant", "locale"):
        value = str(row.get(column, "")).strip().lower()
        if value and value != "nan":
            return re.sub(r"[^a-z0-9_-]+", "_", value).strip("_") or "unknown"
    return "unknown"


def _assign_deterministic_splits(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    assigned = []
    for idx, row in enumerate(rows):
        item = dict(row)
        item["split"] = "test" if idx % 5 == 0 else "train"
        assigned.append(item)
    return assigned


def _filter_labels(
    rows: list[dict[str, object]],
    min_label_count: int = 1,
    max_labels: int | None = None,
) -> list[dict[str, object]]:
    counts = pd.Series([row["word_label"] for row in rows]).value_counts()
    if min_label_count > 1:
        counts = counts[counts >= min_label_count]
    if max_labels is not None:
        counts = counts.head(max_labels)
    allowed = set(counts.index.astype(str))
    return [row for row in rows if str(row["word_label"]) in allowed]


def _limit_rows(rows: list[dict[str, object]], max_rows: int) -> list[dict[str, object]]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows

    frame = pd.DataFrame(rows).reset_index(names="__row_id")
    pieces = []
    label_count = max(1, frame["word_label"].nunique())
    per_label = max(1, max_rows // label_count)

    for _, label_frame in frame.groupby("word_label", sort=False):
        train_rows = label_frame[label_frame["split"] == "train"]
        test_rows = label_frame[label_frame["split"] == "test"]
        desired_test = max(1, per_label // 5)
        desired_train = max(1, per_label - desired_test)
        pieces.append(train_rows.head(desired_train))
        pieces.append(test_rows.head(desired_test))

    limited = pd.concat(pieces, ignore_index=True)
    if len(limited) < max_rows:
        used = set(limited["__row_id"].tolist())
        remainder = frame[~frame["__row_id"].isin(used)].head(max_rows - len(limited))
        limited = pd.concat([limited, remainder], ignore_index=True)

    limited = limited.drop(columns=["__row_id"], errors="ignore")
    return limited.head(max_rows).to_dict(orient="records")
