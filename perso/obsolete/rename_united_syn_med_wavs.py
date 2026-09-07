"""Rename United-Syn-Med wav files from old sequential naming (000000.wav, 000001.wav, …)
back to their original filenames (derived from the MP3 stems in the CSV metadata).

Also updates the corresponding NeMo JSONL manifest so that ``audio_filepath``
entries point to the new filenames.

The old download script used ``f"{idx:06d}.wav"`` as the wav filename, where
``idx`` is the zero-based row index in the split's CSV.  The current version
preserves the original MP3 stem.  This script bridges the gap by downloading
the CSV from HuggingFace and using the row order to reconstruct the mapping
``{idx:06d}.wav`` → ``{original_mp3_stem}.wav``.

No audio data is re-downloaded or re-encoded.

Usage example::

    uv run rename_united_syn_med_wavs.py \\
        --json-path united_syn_med/train/train.json \\
        --split train \\
        --wav-dir united_syn_med/train/wavs \\
        --dry

Written with the help of Claude Code.
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface-hub",
#     "loguru",
#     "click",
# ]
# ///

import csv
import json
from pathlib import Path

import click
from huggingface_hub import hf_hub_download
from loguru import logger

# HuggingFace dataset identifier (gated — requires prior ``hf login``).
DATASET_ID: str = "united-we-care/United-Syn-Med"

# Valid split names for this dataset.
VALID_SPLITS: list[str] = ["train", "test", "validation"]


def _download_csv(*, split: str) -> Path:
    """Download the CSV metadata for a split and return its local path.

    Uses ``huggingface_hub`` so it benefits from the HF cache — repeated
    calls don't re-download.

    Parameters
    ----------
    split : str
        One of ``"train"``, ``"test"``, ``"validation"``.

    Returns
    -------
    Path
        Local path to the cached CSV file.
    """
    logger.info("Downloading/caching {}.csv from HuggingFace…", split)
    local_path = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        filename=f"data/{split}.csv",
    )
    return Path(local_path)


def _build_index_to_stem(*, csv_path: Path) -> dict[int, str]:
    """Read the CSV and return {row_index: original_mp3_stem}.

    The old download script iterated over the CSV rows in order and named
    each wav ``{idx:06d}.wav``.  The CSV's ``file_name`` column holds the
    original MP3 filename (e.g. ``some_audio.mp3``), whose stem is the
    correct new name.

    Parameters
    ----------
    csv_path : Path
        Path to the split's CSV file.

    Returns
    -------
    dict[int, str]
        Mapping from enumeration index to the original MP3 stem.
    """
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        mapping: dict[int, str] = {}
        for idx, row in enumerate(reader):
            original_stem = Path(row["file_name"].strip()).stem
            mapping[idx] = original_stem

    logger.info("Built mapping for {} samples from CSV.", len(mapping))
    return mapping


@click.command()
@click.option(
    "--json-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the NeMo JSONL manifest to fix.",
)
@click.option(
    "--split",
    type=click.Choice(VALID_SPLITS),
    required=True,
    help="Dataset split name (train, test, validation).",
)
@click.option(
    "--wav-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing the wav files to rename.",
)
@click.option(
    "--dry",
    is_flag=True,
    default=False,
    help="Dry run — only print what would happen, don't rename or rewrite.",
)
@click.option(
    "--ignore-missing",
    is_flag=True,
    default=False,
    help=(
        "If set, missing old-style wav files (already renamed in a previous run) "
        "are silently skipped but their manifest entries are still updated."
    ),
)
def main(
    *,
    json_path: Path,
    split: str,
    wav_dir: Path,
    dry: bool,
    ignore_missing: bool,
) -> None:
    """Rename sequentially-named United-Syn-Med wavs to their original filenames."""
    # Download (or use cached) CSV metadata from HuggingFace.
    csv_path = _download_csv(split=split)

    # Build the idx → original stem mapping from CSV row order.
    idx_to_stem = _build_index_to_stem(csv_path=csv_path)

    # --- Phase 1: plan renames ------------------------------------------------
    # Old naming convention: {idx:06d}.wav (6-digit zero-padded)
    renames: list[tuple[Path, Path]] = []  # (old_path, new_path)
    for idx, stem in idx_to_stem.items():
        old_path = wav_dir / f"{idx:06d}.wav"
        new_path = wav_dir / f"{stem}.wav"
        if old_path == new_path:
            continue
        if not old_path.exists():
            if ignore_missing:
                logger.debug("Old file not found (already renamed?), skipping rename: {}", old_path)
            else:
                logger.warning("Expected file not found, skipping: {}", old_path)
            continue
        if new_path.exists():
            logger.warning(
                "Target already exists, skipping: {} → {}", old_path, new_path
            )
            continue
        renames.append((old_path, new_path))

    logger.info("{} files to rename.", len(renames))

    # --- Phase 2: plan manifest rewrite ---------------------------------------
    # Build a lookup from old wav basename → new wav basename for the JSON fix.
    # Start from files that will actually be renamed on disk.
    basename_map: dict[str, str] = {
        old.name: new.name for old, new in renames
    }
    # When ignore_missing is set, also cover files whose old sequential name no
    # longer exists (already renamed in a previous run) so the manifest is still
    # brought up to date.
    if ignore_missing:
        for idx, stem in idx_to_stem.items():
            old_name = f"{idx:06d}.wav"
            new_name = f"{stem}.wav"
            if old_name not in basename_map:
                basename_map[old_name] = new_name

    # Read all manifest lines and patch audio_filepath entries.
    lines = json_path.read_text(encoding="utf-8").splitlines()
    new_lines: list[str] = []
    patched_count = 0
    for line in lines:
        if not line.strip():
            new_lines.append(line)
            continue
        entry = json.loads(line)
        old_basename = Path(entry["audio_filepath"]).name
        if old_basename in basename_map:
            # Replace only the filename portion, keeping the rest of the path.
            old_audio = Path(entry["audio_filepath"])
            entry["audio_filepath"] = str(
                old_audio.parent / basename_map[old_basename]
            )
            patched_count += 1
        new_lines.append(json.dumps(entry, ensure_ascii=False))

    logger.info("{} manifest entries to patch.", patched_count)

    # --- Phase 3: execute or print --------------------------------------------
    if dry:
        logger.info("DRY RUN — nothing will be modified.")
        for old, new in renames:
            print(f"  RENAME  {old}  →  {new}")
        if patched_count:
            print(f"  REWRITE {json_path} ({patched_count} lines patched)")
        return

    # Rename wav files.
    for old, new in renames:
        old.rename(new)
    logger.success("Renamed {} wav files.", len(renames))

    # Rewrite manifest.
    json_path.write_text(
        "\n".join(new_lines) + ("\n" if new_lines else ""),
        encoding="utf-8",
    )
    logger.success(
        "Rewrote manifest {} ({} entries patched).", json_path, patched_count
    )


if __name__ == "__main__":
    main()
