"""Rename FLEURS wav files from old sequential naming (00000.wav, 00001.wav, …)
back to their original filenames (as stored in the HuggingFace parquet metadata).

Also updates the corresponding NeMo JSONL manifest so that ``audio_filepath``
entries point to the new filenames.

The script downloads only the dataset metadata (audio decoding disabled) to
build the mapping from enumeration index → original filename stem.  No audio
data is re-downloaded or re-encoded.

Usage example::

    uv run rename_fleurs_wavs.py \\
        --json-path fleurs/fr/train.json \\
        --split train \\
        --wav-dir fleurs/fr/wavs_train \\
        --lang fr \\
        --dry

Written with the help of Claude Code.
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets>=2.18,<3.0",
#     "loguru",
#     "click",
# ]
# ///

import json
from pathlib import Path

import click
from datasets import Audio, load_dataset
from loguru import logger

# Same mapping as download_fleurs.py — short key → FLEURS folder name.
LANGUAGES: dict[str, str] = {
    "fr": "fr_fr",
    "en": "en_us",
}

# FLEURS calls it "validation" internally but the user-facing name is "val".
_SPLIT_ALIASES: dict[str, str] = {
    "val": "validation",
}


def _resolve_split(split: str) -> str:
    """Map user-friendly split names to the FLEURS dataset split name."""
    return _SPLIT_ALIASES.get(split, split)


def _build_index_to_stem(*, fleurs_code: str, split: str) -> dict[int, str]:
    """Download dataset metadata and return {enumeration_index: original_stem}.

    Audio decoding is disabled so only the parquet metadata is fetched —
    no heavy audio download happens here.
    """
    logger.info(
        "Downloading FLEURS metadata for '{}' split '{}'…", fleurs_code, split
    )
    ds = load_dataset(
        "google/fleurs",
        data_files={split: f"{fleurs_code}/{split}/0000.parquet"},
        split=split,
        revision="refs/convert/parquet",
    )
    # Disable audio decoding — we only need the path field.
    ds = ds.cast_column("audio", Audio(decode=False))

    mapping: dict[int, str] = {}
    for idx, sample in enumerate(ds):
        original_stem = Path(sample["audio"]["path"]).stem
        mapping[idx] = original_stem

    logger.info("Built mapping for {} samples.", len(mapping))
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
    type=click.Choice(["train", "val", "validation", "test"]),
    required=True,
    help="Dataset split name (train, val/validation, test).",
)
@click.option(
    "--wav-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing the wav files to rename.",
)
@click.option(
    "--lang",
    type=click.Choice(list(LANGUAGES.keys())),
    required=True,
    help="Language key (fr or en).",
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
    lang: str,
    dry: bool,
    ignore_missing: bool,
) -> None:
    """Rename sequentially-named FLEURS wavs to their original filenames."""
    fleurs_code = LANGUAGES[lang]
    resolved_split = _resolve_split(split)

    # Build the idx → original stem mapping from the HF dataset metadata.
    idx_to_stem = _build_index_to_stem(
        fleurs_code=fleurs_code, split=resolved_split
    )

    # --- Phase 1: plan renames ------------------------------------------------
    # Old naming convention: {idx:05d}.wav
    renames: list[tuple[Path, Path]] = []  # (old_path, new_path)
    for idx, stem in idx_to_stem.items():
        old_path = wav_dir / f"{idx:05d}.wav"
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
            old_name = f"{idx:05d}.wav"
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
