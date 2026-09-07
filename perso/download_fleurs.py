"""Download French and English splits (train, validation, test) from
Google FLEURS and convert them to NeMo-compatible JSONL manifest format.

Each line in the output manifest has the structure:
    {"audio_filepath": "<path>", "text": "<transcription>", "duration": <seconds>}

Audio files are saved as 16 kHz mono WAV alongside the manifest.

Processing is split into two phases so that lightweight JSON manifests
are available early (e.g. for inspection or row-count validation) before
audio decoding/writing begins:

**Phase 1 — metadata (fast):**
  Load the parquet with audio decoding disabled and write a *preliminary*
  JSONL manifest with ``duration: -1`` placeholders.  Audio filepaths
  already point to the expected WAV locations.

**Phase 2 — audio (slow):**
  Iterate the dataset again, decode raw audio bytes with ``soundfile``,
  write WAV files, then *rewrite* the manifest with real durations.

The parquet revision of the HuggingFace dataset is used because newer
versions of the ``datasets`` library no longer support the legacy
``fleurs.py`` dataset script.  Each language has its own parquet file
under ``<lang>/<split>/0000.parquet``, so we load only the languages
we need (no 102-language download).

Audio decoding is disabled at the ``datasets`` level to avoid a heavy
``librosa`` / ``torchcodec`` dependency; raw audio bytes from the
parquet are decoded with ``soundfile`` instead.
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets>=2.18,<3.0",
#     "soundfile",
#     "loguru",
#     "click",
# ]
# ///

import io
import json
from pathlib import Path

import click
import soundfile as sf
from datasets import Audio, load_dataset
from loguru import logger

# All FLEURS language codes (folder names on the ``refs/convert/parquet`` branch).
# Mapping: short key → fleurs parquet folder name.
LANGUAGES: dict[str, str] = {
    "af": "af_za", "am": "am_et", "ar": "ar_eg", "as": "as_in", "ast": "ast_es",
    "az": "az_az", "be": "be_by", "bg": "bg_bg", "bn": "bn_in", "bs": "bs_ba",
    "ca": "ca_es", "ceb": "ceb_ph", "ckb": "ckb_iq", "cmn": "cmn_hans_cn",
    "cs": "cs_cz", "cy": "cy_gb", "da": "da_dk", "de": "de_de", "el": "el_gr",
    "en": "en_us", "es": "es_419", "et": "et_ee", "fa": "fa_ir", "ff": "ff_sn",
    "fi": "fi_fi", "fil": "fil_ph", "fr": "fr_fr", "ga": "ga_ie", "gl": "gl_es",
    "gu": "gu_in", "ha": "ha_ng", "he": "he_il", "hi": "hi_in", "hr": "hr_hr",
    "hu": "hu_hu", "hy": "hy_am", "id": "id_id", "ig": "ig_ng", "is": "is_is",
    "it": "it_it", "ja": "ja_jp", "jv": "jv_id", "ka": "ka_ge", "kam": "kam_ke",
    "kea": "kea_cv", "kk": "kk_kz", "km": "km_kh", "kn": "kn_in", "ko": "ko_kr",
    "ku": "ckb_iq", "ky": "ky_kg", "lb": "lb_lu", "lg": "lg_ug", "ln": "ln_cd",
    "lo": "lo_la", "lt": "lt_lt", "luo": "luo_ke", "lv": "lv_lv", "mi": "mi_nz",
    "mk": "mk_mk", "ml": "ml_in", "mn": "mn_mn", "mr": "mr_in", "ms": "ms_my",
    "mt": "mt_mt", "my": "my_mm", "nb": "nb_no", "ne": "ne_np", "nl": "nl_nl",
    "nso": "nso_za", "ny": "ny_mw", "oc": "oc_fr", "om": "om_et", "or": "or_in",
    "pa": "pa_in", "pl": "pl_pl", "ps": "ps_af", "pt": "pt_br", "ro": "ro_ro",
    "ru": "ru_ru", "sd": "sd_in", "sk": "sk_sk", "sl": "sl_si", "sn": "sn_zw",
    "so": "so_so", "sr": "sr_rs", "sv": "sv_se", "sw": "sw_ke", "ta": "ta_in",
    "te": "te_in", "tg": "tg_tj", "th": "th_th", "tr": "tr_tr", "uk": "uk_ua",
    "umb": "umb_ao", "ur": "ur_pk", "uz": "uz_uz", "vi": "vi_vn", "wo": "wo_sn",
    "xh": "xh_za", "yo": "yo_ng", "yue": "yue_hant_hk", "zu": "zu_za",
}

# Dataset splits to download.  The manifest file for each split is named
# ``<split>.json`` (e.g. ``train.json``, ``validation.json``, ``test.json``).
SPLITS: list[str] = ["train", "validation", "test"]


def _load_split(
    *,
    fleurs_code: str,
    split: str,
):
    """Download and return one FLEURS split with audio decoding disabled.

    Parameters
    ----------
    fleurs_code : str
        FLEURS folder name on the parquet branch (e.g. ``"fr_fr"``).
    split : str
        Dataset split (``"train"``, ``"validation"``, or ``"test"``).

    Returns
    -------
    datasets.Dataset
        The loaded dataset with the ``audio`` column cast to raw bytes
        (no librosa/torchcodec decoding).
    """
    logger.info("Downloading FLEURS {} for '{}'…", split, fleurs_code)
    ds = load_dataset(
        "google/fleurs",
        data_files={split: f"{fleurs_code}/{split}/0000.parquet"},
        split=split,
        revision="refs/convert/parquet",
    )
    # Disable the built-in audio decoder (needs librosa or torchcodec).
    # We decode the raw bytes ourselves with soundfile in phase 2.
    ds = ds.cast_column("audio", Audio(decode=False))
    logger.info("Loaded {} samples for '{}' ({})", len(ds), fleurs_code, split)
    return ds


def _write_preliminary_manifest(
    *,
    ds,
    lang_key: str,
    split: str,
    output_dir: Path,
) -> Path:
    """Phase 1: write a JSONL manifest from parquet metadata only.

    Audio filepaths point to the *expected* WAV locations; durations are
    set to ``-1`` (placeholder) because audio bytes are not decoded yet.
    This gives the user an inspectable manifest before the slow audio
    phase begins.

    Parameters
    ----------
    ds : datasets.Dataset
        FLEURS dataset split with audio decoding disabled.
    lang_key : str
        Short language label (e.g. ``"fr"``).
    split : str
        Dataset split name.
    output_dir : Path
        Root output directory.

    Returns
    -------
    Path
        Path to the written manifest file.
    """
    lang_dir = output_dir / lang_key
    wav_dir = lang_dir / f"wavs_{split}"
    wav_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = lang_dir / f"{split}.json"
    n_written = 0

    with manifest_path.open("w", encoding="utf-8") as fout:
        for sample in ds:
            original_stem = Path(sample["audio"]["path"]).stem
            wav_path = wav_dir / f"{original_stem}.wav"
            entry = {
                "audio_filepath": str(wav_path),
                "text": sample["transcription"],
                # Placeholder — real duration is filled in phase 2 after
                # audio decoding.  -1 signals "not yet measured".
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


def _export_audio(
    *,
    ds,
    lang_key: str,
    split: str,
    output_dir: Path,
) -> None:
    """Phase 2: decode audio bytes, write WAVs, rewrite manifest with real durations.

    Parameters
    ----------
    ds : datasets.Dataset
        FLEURS dataset split with audio decoding disabled.
    lang_key : str
        Short language label (e.g. ``"fr"``).
    split : str
        Dataset split name.
    output_dir : Path
        Root output directory.
    """
    lang_dir = output_dir / lang_key
    wav_dir = lang_dir / f"wavs_{split}"
    wav_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = lang_dir / f"{split}.json"
    n_written = 0

    with manifest_path.open("w", encoding="utf-8") as fout:
        for sample in ds:
            raw_bytes: bytes = sample["audio"]["bytes"]

            # Decode the embedded audio (ogg/flac) into a numpy array.
            data, sr = sf.read(io.BytesIO(raw_bytes))
            duration = len(data) / sr

            # Preserve the original filename, only swapping the extension to .wav
            # (the source files are typically .ogg or .flac inside the parquet).
            original_stem = Path(sample["audio"]["path"]).stem
            wav_path = wav_dir / f"{original_stem}.wav"
            sf.write(str(wav_path), data, sr)

            # Use the normalised transcription provided by FLEURS.
            entry = {
                "audio_filepath": str(wav_path),
                "text": sample["transcription"],
                "duration": round(duration, 2),
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            n_written += 1

    logger.success(
        "Phase 2: wrote {} samples → {} (audio in {})",
        n_written,
        manifest_path,
        wav_dir,
    )


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("fleurs"),
    show_default=True,
    help="Root directory for the downloaded data.",
)
@click.option(
    "--lang",
    type=click.Choice(list(LANGUAGES.keys()), case_sensitive=False),
    default="fr",
    show_default=True,
    help="Language to download.",
)
@click.option(
    "--splits",
    default="train,val,test",
    show_default=True,
    help="Comma-separated list of splits to download (train, val, test).",
)
def main(*, output_dir: Path, lang: str, splits: str) -> None:
    """Download FLEURS splits for a given language into NeMo format."""
    split_map = {"train": "train", "val": "validation", "test": "test"}
    requested_splits = [s.strip() for s in splits.split(",")]
    for s in requested_splits:
        if s not in split_map:
            raise click.BadParameter(
                f"Unknown split '{s}'. Must be one of: {', '.join(split_map)}"
            )

    fleurs_code = LANGUAGES[lang]

    # ── Phase 1: download parquet metadata and write preliminary manifests ──
    # Each split's parquet is loaded once and reused in phase 2, avoiding a
    # redundant second download.
    datasets_by_split: dict[str, object] = {}
    for s in requested_splits:
        ds = _load_split(fleurs_code=fleurs_code, split=split_map[s])
        _write_preliminary_manifest(
            ds=ds,
            lang_key=lang,
            split=split_map[s],
            output_dir=output_dir,
        )
        datasets_by_split[s] = ds

    # ── Phase 2: decode audio, write WAVs, rewrite manifests with durations ──
    for s in requested_splits:
        _export_audio(
            ds=datasets_by_split[s],
            lang_key=lang,
            split=split_map[s],
            output_dir=output_dir,
        )

    logger.info("All done.")


if __name__ == "__main__":
    main()
