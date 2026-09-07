"""Download train, test and validation splits from United-Syn-Med
and convert them to NeMo-compatible JSONL manifest format.

United-Syn-Med is a medical ASR dataset hosted on HuggingFace at
``united-we-care/United-Syn-Med``.  It ships CSV metadata
(``file_name``, ``transcription``) plus tar.gz audio archives.

Processing is split into two phases so that lightweight JSON manifests
are available early (e.g. for inspection or row-count validation) before
the heavyweight audio download/conversion begins:

**Phase 1 — metadata (fast, small downloads):**
  For every requested split, download the CSV and write a *preliminary*
  JSONL manifest with ``duration: -1`` placeholders.  Audio filepaths
  already point to the expected WAV locations so downstream tooling can
  reason about paths before any audio exists.

**Phase 2 — audio (slow, large downloads):**
  For each split, download and extract the tar.gz, convert MP3 → 16 kHz
  mono WAV, delete source MP3s incrementally, then *rewrite* the manifest
  with real durations.

If ``<split>_extracted`` already exists the download + extraction are skipped,
so re-runs only redo the WAV conversion / manifest step.

Tar extraction uses ``pigz`` (parallel gzip) when available on ``$PATH``,
falling back to Python's ``tarfile`` otherwise.  Install it with
``sudo apt install pigz`` for a significant speed-up on large archives.

Audio conversion is parallelised with ``joblib`` using the threading backend
(avoids GIL issues for I/O-bound MP3 decode + WAV encode).  Use ``--n-jobs``
to control concurrency (default: 4).

Resampling uses ``soxr`` (same engine as ffmpeg/libsox) for high-quality,
alias-free conversion instead of linear interpolation.  A single rglob index
is built once per split so that file lookup is O(1) rather than O(N²).

Requires a prior ``huggingface-cli login`` for gated dataset access.
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "soundfile",
#     "soxr",
#     "loguru",
#     "click",
#     "tqdm",
#     "huggingface-hub",
#     "joblib",
# ]
# ///

import csv
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Optional

import click
import soundfile as sf
import soxr
from huggingface_hub import hf_hub_download
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

# The three splits shipped by United-Syn-Med.
SPLITS: list[str] = ["train", "test", "validation"]

# HuggingFace dataset identifier (gated — requires prior ``hf login``).
DATASET_ID: str = "united-we-care/United-Syn-Med"


def _download_csv(*, split: str, repo_dir: Path) -> Path:
    """Download the CSV metadata for one split (lightweight, ~KB).

    Parameters
    ----------
    split : str
        One of ``"train"``, ``"test"``, ``"validation"``.
    repo_dir : Path
        Root directory where the downloaded files are stored.

    Returns
    -------
    Path
        Path to the downloaded CSV file.
    """
    csv_path = repo_dir / "data" / f"{split}.csv"
    if not csv_path.exists():
        logger.info("Downloading {}.csv …", split)
        hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=f"data/{split}.csv",
            local_dir=str(repo_dir),
        )
    else:
        logger.info("{}.csv already present.", split)
    return csv_path


def _write_preliminary_manifest(
    *,
    split: str,
    csv_path: Path,
    output_dir: Path,
) -> Path:
    """Create a JSONL manifest from CSV metadata with placeholder durations.

    The manifest is written with ``duration: -1`` so that downstream tools
    can inspect file lists and transcriptions before any audio is downloaded.
    Audio filepaths point to the *expected* WAV locations so paths are
    stable across phases.

    Parameters
    ----------
    split : str
        One of ``"train"``, ``"test"``, ``"validation"``.
    csv_path : Path
        Path to the split's CSV file (``file_name``, ``transcription``).
    output_dir : Path
        Root output directory; manifest lands at ``<output_dir>/<split>/<name>.json``.

    Returns
    -------
    Path
        Path to the written manifest file.
    """
    split_dir = output_dir / split
    wav_dir = split_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    manifest_name = "val.json" if split == "validation" else f"{split}.json"
    manifest_path = split_dir / manifest_name

    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n_written = 0
    with manifest_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            wav_path = wav_dir / (Path(row["file_name"].strip()).stem + ".wav")
            entry = {
                "audio_filepath": str(wav_path.resolve()),
                "text": row["transcription"].strip(),
                # Placeholder — real duration is filled in phase 2 after
                # audio conversion.  -1 signals "not yet measured".
                "duration": -1,
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            n_written += 1

    logger.success(
        "Phase 1: wrote preliminary manifest ({} rows, duration=-1) → {}",
        n_written,
        manifest_path,
    )
    return manifest_path


def _prepare_audio(*, split: str, repo_dir: Path) -> None:
    """Download, extract and delete the tar.gz for one split.

    Downloads the audio archive for *split* individually so that only one
    archive lives on disk at a time.  After extraction the tar.gz is
    removed to reclaim space.

    If ``<split>_extracted`` already exists the whole step is skipped,
    making reruns cheap for all three splits (train / test / validation).

    Parameters
    ----------
    split : str
        One of ``"train"``, ``"test"``, ``"validation"``.
    repo_dir : Path
        Root directory where the downloaded files are stored.
    """
    raw_audio_dir = repo_dir / "data" / "audio" / f"{split}_extracted"

    if raw_audio_dir.exists():
        logger.info("'{}' already extracted at {}, skipping download.", split, raw_audio_dir)
        return

    # Download only this split's audio archive (not the whole repo).
    tar_path = repo_dir / "data" / "audio" / f"{split}.tar.gz"
    if not tar_path.exists():
        logger.info("Downloading {}.tar.gz …", split)
        hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=f"data/audio/{split}.tar.gz",
            local_dir=str(repo_dir),
        )

    # Extract then immediately delete the archive to free disk space.
    _extract_audio(tar_path=tar_path, dest_dir=raw_audio_dir)
    logger.info("Deleting {} to reclaim disk space.", tar_path.name)
    tar_path.unlink()
    logger.success("Deleted {}.", tar_path.name)


def _extract_audio(*, tar_path: Path, dest_dir: Path) -> None:
    """Extract a tar.gz audio archive into *dest_dir*.

    Uses ``pigz`` for parallel decompression when available on ``$PATH``
    (e.g. ``sudo apt install pigz``), falling back to Python's built-in
    ``tarfile`` otherwise.  For a 35 GB archive, ``pigz`` typically cuts
    extraction time by 3–5× on a modern multi-core machine.

    Parameters
    ----------
    tar_path : Path
        Path to the ``.tar.gz`` archive (e.g. ``data/audio/train.tar.gz``).
    dest_dir : Path
        Directory where audio files will be extracted.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting {} → {}", tar_path.name, dest_dir)

    if shutil.which("pigz"):
        # pigz -dc decompresses to stdout using multiple cores;
        # tar -x reads the raw tar stream from stdin and extracts to dest_dir.
        # Closing pigz_proc.stdout after handing it to tar allows pigz to
        # receive SIGPIPE if tar exits early (standard subprocess pipe idiom).
        logger.info("pigz found — using parallel decompression.")
        pigz_proc = subprocess.Popen(
            ["pigz", "-dc", str(tar_path)],
            stdout=subprocess.PIPE,
        )
        tar_proc = subprocess.Popen(
            ["tar", "-x", "-C", str(dest_dir)],
            stdin=pigz_proc.stdout,
        )
        pigz_proc.stdout.close()
        tar_proc.wait()
        pigz_proc.wait()
        if pigz_proc.returncode != 0 or tar_proc.returncode != 0:
            raise RuntimeError(
                f"Extraction failed (pigz={pigz_proc.returncode}, tar={tar_proc.returncode})"
            )
    else:
        logger.warning("pigz not found — falling back to single-threaded tarfile extraction.")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=dest_dir)

    logger.success("Extracted {} files from {}", len(list(dest_dir.rglob("*"))), tar_path.name)


def _mp3_to_wav(*, mp3_path: Path, wav_path: Path, target_sr: int = 16_000) -> float:
    """Convert an MP3 file to 16 kHz mono WAV and return its duration.

    Parameters
    ----------
    mp3_path : Path
        Source MP3 file.
    wav_path : Path
        Destination WAV file.
    target_sr : int
        Target sample rate in Hz.  NeMo expects 16 000.

    Returns
    -------
    float
        Duration of the audio in seconds.
    """
    data, sr = sf.read(str(mp3_path))

    # Convert stereo to mono by averaging channels if needed.
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample if the source sample rate differs from the target.
    # soxr uses the same high-quality resampler as ffmpeg/libsox,
    # which is both faster and alias-free compared to np.interp.
    if sr != target_sr:
        data = soxr.resample(data, in_rate=sr, out_rate=target_sr, quality="HQ")
        sr = target_sr

    sf.write(str(wav_path), data, sr)
    duration = len(data) / sr
    return duration


def _process_row(
    *,
    idx: int,
    row: dict,
    filename_map: dict[str, Path],
    wav_dir: Path,
) -> Optional[dict]:
    """Convert one CSV row's MP3 to WAV and return its manifest entry.

    Designed to be called from ``joblib.Parallel`` — all filesystem ops
    are independent per row so threading is safe.

    Parameters
    ----------
    idx : int
        Row index (unused for naming, kept for API compatibility with joblib).
    row : dict
        A single CSV row with ``file_name`` and ``transcription`` keys.
    filename_map : dict[str, Path]
        Pre-built mapping of bare filename → full path for all files in
        the extracted audio directory.  Avoids an O(N²) rglob-per-row.
    wav_dir : Path
        Directory where the converted WAV file will be written.

    Returns
    -------
    dict or None
        NeMo manifest entry on success, ``None`` if the file is missing
        or conversion fails (the caller counts skips).
    """
    file_name: str = row["file_name"].strip()
    transcription: str = row["transcription"].strip()

    # O(1) lookup instead of a full rglob traversal for every row.
    mp3_path = filename_map.get(file_name)
    if mp3_path is None:
        logger.warning("Audio file not found for '{}', skipping.", file_name)
        return None

    # Keep the original filename, only swapping the extension to .wav,
    # so the output is traceable back to the source MP3.
    wav_path = wav_dir / (Path(file_name).stem + ".wav")

    try:
        duration = _mp3_to_wav(mp3_path=mp3_path, wav_path=wav_path)
    except Exception as exc:
        logger.warning("Failed to convert '{}': {}", file_name, exc)
        return None

    # Delete the source MP3 right after a successful conversion so that
    # disk space is reclaimed incrementally rather than only at the end.
    try:
        mp3_path.unlink()
    except OSError as exc:
        # Non-fatal: log and continue — the WAV is already written.
        logger.warning("Could not delete '{}': {}", mp3_path, exc)

    return {
        "audio_filepath": str(wav_path.resolve()),
        "text": transcription,
        "duration": round(duration, 2),
    }


def _build_manifest(
    *,
    split: str,
    repo_dir: Path,
    output_dir: Path,
    n_jobs: int,
) -> None:
    """Build a NeMo manifest for one split.

    Reads the CSV metadata, converts each MP3 to WAV in parallel (threading)
    and writes the JSONL manifest that NeMo's ``AudioToCharDataset`` /
    ``AudioToBPEDataset`` expects.

    Parameters
    ----------
    split : str
        One of ``"train"``, ``"test"``, ``"validation"``.
    repo_dir : Path
        Root of the downloaded HuggingFace repo.
    output_dir : Path
        Where to write the converted WAV files and manifest.
    n_jobs : int
        Number of parallel worker threads for audio conversion.
    """
    csv_path = repo_dir / "data" / f"{split}.csv"

    if not csv_path.exists():
        logger.error("CSV not found: {}", csv_path)
        return

    # The tar.gz was already extracted (and deleted) by _prepare_split.
    raw_audio_dir = repo_dir / "data" / "audio" / f"{split}_extracted"
    if not raw_audio_dir.exists():
        logger.error("Extracted audio dir not found: {}", raw_audio_dir)
        return

    # Prepare output directories.
    split_dir = output_dir / split
    wav_dir = split_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    # Map the split name to the manifest file name that matches the
    # fleurs convention (val.json for validation, train.json, test.json).
    manifest_name = "val.json" if split == "validation" else f"{split}.json"
    manifest_path = split_dir / manifest_name

    # Read CSV into memory so we know the total for tqdm.
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Build a bare-filename → full-path map once for the whole split.
    # Without this, _process_row would call rglob() per file, which is
    # O(N²) over the directory tree and the dominant source of slowness.
    logger.info("Building filename index for '{}'…", split)
    filename_map: dict[str, Path] = {
        p.name: p for p in raw_audio_dir.rglob("*") if p.is_file()
    }
    logger.info("Indexed {} audio files.", len(filename_map))

    # --- Pre-scan: skip already converted files before entering the loop ---
    # WAV filenames mirror the source MP3 stem (e.g. ``foo.mp3`` → ``foo.wav``).
    # If the file exists we build its manifest entry directly from sf.info
    # (cheap header read), so reruns only process what is genuinely missing.
    already_done: dict[int, dict] = {}
    to_process: list[tuple[int, dict]] = []
    for idx, row in tqdm(enumerate(rows), total=len(rows), desc=f"[{split}] scanning existing WAVs", unit="file"):
        wav_path = wav_dir / (Path(row["file_name"].strip()).stem + ".wav")
        if wav_path.exists():
            info = sf.info(str(wav_path))
            already_done[idx] = {
                "audio_filepath": str(wav_path.resolve()),
                "text": row["transcription"].strip(),
                "duration": round(info.duration, 2),
            }
            # The WAV exists (converted in a prior run), but the MP3 may
            # still be on disk if that run was interrupted before deletion.
            # Clean it up now so reruns also free space.
            mp3_path = filename_map.get(row["file_name"].strip())
            if mp3_path is not None and mp3_path.exists():
                try:
                    mp3_path.unlink()
                except OSError as exc:
                    logger.warning("Could not delete leftover MP3 '{}': {}", mp3_path, exc)
        else:
            to_process.append((idx, row))

    logger.info(
        "Split '{}': {} already converted, {} to process, {} threads",
        split,
        len(already_done),
        len(to_process),
        n_jobs,
    )

    # Collect all entries keyed by row index so we can write the manifest
    # in the original CSV order regardless of completion order.
    all_entries: dict[int, Optional[dict]] = dict(already_done)

    if to_process:
        # Run MP3→WAV conversion in parallel using threads.
        # ``return_as="generator"`` lets tqdm track completions rather than
        # submissions, giving a more accurate progress bar.
        # Results are yielded in submission order (joblib guarantee), so we
        # can zip safely with ``to_process`` to recover each row's index.
        jobs = (
            delayed(_process_row)(idx=idx, row=row, filename_map=filename_map, wav_dir=wav_dir)
            for idx, row in to_process
        )
        results = tqdm(
            Parallel(n_jobs=n_jobs, prefer="threads", return_as="generator")(jobs),
            total=len(to_process),
            desc=f"[{split}] converting audio",
            unit="file",
        )
        for (idx, _row), entry in zip(to_process, results):
            all_entries[idx] = entry

    n_written = 0
    n_skipped = 0

    # Write manifest in original CSV row order.
    with manifest_path.open("w", encoding="utf-8") as fout:
        for idx in range(len(rows)):
            entry = all_entries.get(idx)
            if entry is None:
                n_skipped += 1
                continue
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            n_written += 1

    logger.success(
        "Split '{}': wrote {} samples, skipped {} → {}",
        split,
        n_written,
        n_skipped,
        manifest_path,
    )


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("united_syn_med"),
    show_default=True,
    help="Root directory for the converted NeMo-format data.",
)
@click.option(
    "--repo-dir",
    type=click.Path(path_type=Path),
    default=Path("/tmp/united-syn-med-repo"),
    show_default=True,
    help="Where to download / cache the raw HuggingFace repo.",
)
@click.option(
    "--skip-download",
    is_flag=True,
    default=False,
    help="Skip the download step (use if repo is already downloaded).",
)
@click.option(
    "--splits",
    type=str,
    default="train,test,validation",
    show_default=True,
    help="Comma-separated list of splits to process.",
)
@click.option(
    "--n-jobs",
    type=int,
    default=4,
    show_default=True,
    # Threading avoids GIL issues for I/O-bound work (MP3 decode + WAV write)
    # while keeping shared memory access simple (no pickle overhead).
    help="Number of parallel threads for audio conversion.",
)
def main(
    *,
    output_dir: Path,
    repo_dir: Path,
    skip_download: bool,
    splits: str,
    n_jobs: int,
) -> None:
    """Download United-Syn-Med and convert to NeMo manifest format.

    Produces the same JSONL + WAV structure as the FLEURS download
    script so that the data can be plugged directly into a NeMo
    training / validation config.
    """
    requested_splits = [s.strip() for s in splits.split(",")]
    for split in requested_splits:
        if split not in SPLITS:
            logger.error("Unknown split '{}', expected one of {}", split, SPLITS)
            continue

    # Keep only valid splits for the rest of the function.
    requested_splits = [s for s in requested_splits if s in SPLITS]

    # ── Phase 1: download CSVs and write preliminary JSON manifests ──
    # This is fast (~KB per CSV) and gives the user inspectable manifests
    # (with duration=-1 placeholders) before the slow audio phase starts.
    if not skip_download:
        for split in requested_splits:
            csv_path = _download_csv(split=split, repo_dir=repo_dir)
            _write_preliminary_manifest(
                split=split,
                csv_path=csv_path,
                output_dir=output_dir,
            )

    # ── Phase 2: download tar archives, convert audio, rewrite manifests ──
    for split in requested_splits:
        if not skip_download:
            _prepare_audio(split=split, repo_dir=repo_dir)

        # _build_manifest rewrites the manifest with real durations from
        # the converted WAV files, replacing the phase-1 placeholders.
        _build_manifest(
            split=split,
            repo_dir=repo_dir,
            output_dir=output_dir,
            n_jobs=n_jobs,
        )

    logger.info("All done.")


if __name__ == "__main__":
    main()
