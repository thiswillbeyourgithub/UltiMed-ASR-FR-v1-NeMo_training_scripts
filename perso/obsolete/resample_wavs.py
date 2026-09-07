#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["soundfile", "librosa", "loguru"]
# ///
"""Resample all .wav files in specified directories to a target sample rate.

Overwrites files in-place. Skips files already at the target rate.
Usage: uv run resample_wavs.py
"""

from pathlib import Path

import librosa
import soundfile as sf
from loguru import logger

# Target sample rate matching training_config.yaml
TARGET_SR: int = 16_000

# Directories containing wav files to resample,
# relative to this script's location (perso/)
DIRS: list[str] = [
    "nemo_dataset",
    "drug_sentence_dataset",
]


def resample_file(path: Path, *, target_sr: int = TARGET_SR) -> None:
    """Resample a single wav file in-place if its rate differs from target_sr."""
    info = sf.info(str(path))
    if info.samplerate == target_sr:
        logger.debug("Already {}Hz, skipping: {}", target_sr, path.name)
        return

    logger.info(
        "Resampling {} from {}Hz to {}Hz", path.name, info.samplerate, target_sr
    )
    # librosa.load resamples on the fly
    audio, _ = librosa.load(str(path), sr=target_sr, mono=False)
    sf.write(str(path), audio.T if audio.ndim > 1 else audio, target_sr)


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    for rel_dir in DIRS:
        wav_dir = script_dir / rel_dir
        if not wav_dir.is_dir():
            logger.warning("Directory not found, skipping: {}", wav_dir)
            continue

        wavs = sorted(wav_dir.rglob("*.wav"))
        logger.info("Found {} wav files in {}", len(wavs), wav_dir)

        for wav in wavs:
            resample_file(wav)

    logger.info("Done.")


if __name__ == "__main__":
    main()
