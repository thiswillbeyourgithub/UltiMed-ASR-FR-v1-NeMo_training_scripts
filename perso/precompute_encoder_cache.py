# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "click",
#     "loguru",
#     "tqdm",
#     "soundfile",
# ]
# ///
"""Precompute and cache encoder activations for all training samples.

Running the full encoder (608M params) every training step is wasteful when
it is frozen.  This script processes each audio file once through the
pretrained model's ``preprocessor → encoder`` pipeline and saves the
resulting ``(encoded, encoded_len)`` tensors to disk as fp16 ``.pt`` files.

Audio loading is parallelised via a ``torch.utils.data.DataLoader`` and
encoder inference is batched (``--batch-size``) to maximise GPU utilisation.

The output is:
  * One ``.pt`` file per sample in ``<cache_dir>/``.
  * A ``cache_manifest.json`` (NeMo-style JSONL) mapping each original
    audio path to its cached ``.pt`` path plus the transcript / duration.

When ``--variants-per-sample N`` is passed (requires ``--training-config``),
the script generates N cached versions per audio file.  Variant 0 is always
clean (eval mode, no augmentation).  Variants 1..N-1 are augmented draws
from the training augmentor + SpecAugment (``asr_model.train()`` is set,
enabling encoder dropout as additional stochasticity).  At training time
``CachedEncoderDataset`` randomly picks one variant per ``__getitem__``
call, giving the live encoder layers genuine acoustic diversity.  Smart
regen: increasing N only generates the missing new variants.

When ``--compute-stats`` is passed (requires ``--training-config``), the
script additionally measures per-dimension variability of the encoder
outputs under the training augmentations (audio augmentor + SpecAugment).
For a random subset of samples, multiple augmented forward passes are run
through the full pipeline, and the per-dimension std across augmentations
is averaged across samples.  The result is saved to
``<cache_dir>/cache_stats.json`` and later loaded by the training script
to scale latent Gaussian noise proportionally per dimension.

Usage
-----
::

    uv run perso/precompute_encoder_cache.py \\
        --model nvidia/parakeet-tdt-0.6b-v3 \\
        --manifests perso/nemo_dataset/train.json perso/drug_sentence_dataset/train.json \\
        --cache-dir perso/encoder_cache \\
        --device cuda \\
        --batch-size 16 \\
        --num-workers 4

Estimated storage: ~26k samples × 500 frames × 640 dims × 2 bytes ≈ 16 GB.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import torch
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def _load_audio(audio_path: str, sample_rate: int = 16000) -> torch.Tensor:
    """Load audio file and return a 1-D float32 tensor at *sample_rate*.

    Uses ``soundfile`` which handles WAV, FLAC, OGG, etc.  Resampling is
    *not* performed — the file must already be at the target sample rate
    (NeMo manifests guarantee this).
    """
    import soundfile as sf

    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    data, sr = sf.read(audio_path, dtype="float32")
    if sr != sample_rate:
        raise ValueError(
            f"Expected sample rate {sample_rate}, got {sr} for {audio_path}. "
            "Resample your audio first."
        )
    # soundfile returns (samples,) for mono; ensure 1-D
    if data.ndim > 1:
        data = data[:, 0]
    return torch.from_numpy(data)


def _stable_hash(text: str) -> str:
    """Deterministic short hash for file naming — avoids path collisions."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Cache validation helpers (content-aware, mtime+size shortcut)
# ---------------------------------------------------------------------------


def _audio_content_sha256(path: str) -> str:
    """Full SHA256 of an audio file's bytes.  Buffered, ~200 MB/s on SSD."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stat_file(path: str) -> tuple[int, float]:
    """Return ``(size, mtime)`` for a file."""
    st = os.stat(path)
    return st.st_size, st.st_mtime


def _global_config_fingerprint(model: str, cut_layer: int, sample_rate: int, num_variants: int = 1) -> str:
    """Hash of the cache-invalidating global config fields."""
    return hashlib.sha256(
        f"{model}|cut_layer={cut_layer}|sr={sample_rate}|nv={num_variants}".encode()
    ).hexdigest()


def _variant_pt_name(audio_hash: str, variant_idx: int) -> str:
    """Return the filename for a specific variant of a cached sample."""
    return f"{audio_hash}_v{variant_idx:02d}.pt"


def _stats_fingerprint(
    augmentor_cfg, spec_aug_cfg, sample_rate: int, cut_layer: int,
    stats_samples: int, stats_augmentations: int,
    outlier_top_k_frac: float, outlier_kurtosis_threshold: float,
) -> dict:
    """Fingerprint of inputs that determine cache_stats.json."""
    from omegaconf import OmegaConf

    def _yaml_hash(cfg):
        if cfg is None:
            return None
        return hashlib.sha256(OmegaConf.to_yaml(cfg).encode()).hexdigest()

    return {
        "stats_samples": stats_samples,
        "stats_augmentations": stats_augmentations,
        "augmentor_cfg_hash": _yaml_hash(augmentor_cfg),
        "spec_augment_cfg_hash": _yaml_hash(spec_aug_cfg),
        "sample_rate": sample_rate,
        "cut_layer": cut_layer,
        "outlier_top_k_frac": outlier_top_k_frac,
        "outlier_kurtosis_threshold": outlier_kurtosis_threshold,
    }


def _load_cache_index(path: Path) -> dict[str, dict]:
    """Read cache_index.json (JSONL).  Keyed by audio_filepath."""
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["audio_filepath"]] = rec
    return out


def _write_cache_index(path: Path, records: list[dict]) -> None:
    """Write cache_index.json as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _validate_audio_files_parallel(
    paths: list[str],
    prev_index: dict[str, dict],
    num_workers: int = 8,
) -> dict[str, dict]:
    """Stat each file; SHA256 only those whose ``(size, mtime)`` changed.

    Returns ``{path: {"size", "mtime", "audio_sha256", "unchanged"}}``.
    ``unchanged=True`` means the file's stat matched the recorded value and
    its previous SHA256 was reused (no content read).
    """
    def _one(p: str) -> tuple[str, dict]:
        size, mtime = _stat_file(p)
        prev = prev_index.get(p)
        if (
            prev is not None
            and prev.get("size") == size
            and prev.get("mtime") == mtime
            and "audio_sha256" in prev
        ):
            return p, {
                "size": size,
                "mtime": mtime,
                "audio_sha256": prev["audio_sha256"],
                "unchanged": True,
            }
        return p, {
            "size": size,
            "mtime": mtime,
            "audio_sha256": _audio_content_sha256(p),
            "unchanged": False,
        }

    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        for p, rec in tqdm(
            pool.map(_one, paths),
            total=len(paths),
            desc="Validating audio (stat+hash on change)",
            smoothing=0.01,
        ):
            out[p] = rec
    return out


# ---------------------------------------------------------------------------
# Dataset & collate for batched DataLoader
# ---------------------------------------------------------------------------


class AudioDataset(Dataset):
    """Thin dataset that loads audio files on the fly.

    Each item is an ``(index, audio_tensor)`` pair.  Failed loads return
    ``None`` which the custom collate function filters out so that a single
    corrupt file doesn't crash the whole batch.
    """

    def __init__(self, entries: list[dict], sample_rate: int = 16000) -> None:
        self.entries = entries
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor]:
        entry = self.entries[idx]
        audio = _load_audio(
            audio_path=entry["audio_filepath"],
            sample_rate=self.sample_rate,
        )
        return (idx, audio)


def _collate_audio(
    batch: list[tuple[int, torch.Tensor]],
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    """Collate variable-length audio tensors into a padded batch.

    Returns ``(indices, padded_audio, audio_lengths)`` where *padded_audio* has
    shape ``(B, T_max)`` and *audio_lengths* is ``(B,)``.
    """
    indices = [item[0] for item in batch]
    audios = [item[1] for item in batch]
    lengths = torch.tensor([a.shape[0] for a in audios], dtype=torch.long)

    # pad_sequence expects (T,) tensors; pads to max length with 0.0
    padded = pad_sequence(audios, batch_first=True, padding_value=0.0)

    return indices, padded, lengths


# ---------------------------------------------------------------------------
# Batch validation: ensure batched inference matches single-sample inference
# ---------------------------------------------------------------------------


def _validate_batched_inference(
    asr_model: torch.nn.Module,
    entries: list[dict],
    *,
    sample_rate: int,
    device: str,
    max_samples: int = 3,
) -> None:
    """Compare single-sample vs batched encoder outputs.

    Picks up to *max_samples* entries, runs each individually and then as a
    batch through the encoder, and checks that the results match within
    tolerances suitable for fp16 (``atol=1e-3`` warning, ``atol=1e-2``
    error).  This guards against padding or broadcasting bugs.
    """
    samples = entries[:max_samples]
    if not samples:
        return

    audios: list[torch.Tensor] = []
    for entry in samples:
        try:
            audios.append(
                _load_audio(
                    audio_path=entry["audio_filepath"],
                    sample_rate=sample_rate,
                )
            )
        except Exception:
            continue
    if not audios:
        logger.warning("Batch validation skipped — no loadable samples.")
        return

    # --- Single-sample reference outputs (also log tensor shape once) ---
    single_outputs: list[torch.Tensor] = []
    with torch.no_grad(), torch.amp.autocast(device_type=device.split(":")[0]):
        for audio in audios:
            sig = audio.unsqueeze(0).to(device)
            sig_len = torch.tensor([sig.shape[1]], device=device)
            enc, enc_len = asr_model.forward(
                input_signal=sig, input_signal_length=sig_len
            )
            if not single_outputs:
                logger.info(f"Encoder output shape: {enc.shape}, encoded_len: {enc_len}")
            # Trim to actual length and move to CPU
            actual_len = enc_len[0].item()
            single_outputs.append(enc[0, :, :actual_len].cpu())

    # --- Batched outputs ---
    lengths = torch.tensor([a.shape[0] for a in audios], dtype=torch.long)
    padded = pad_sequence(audios, batch_first=True, padding_value=0.0).to(device)
    lengths_dev = lengths.to(device)

    with torch.no_grad(), torch.amp.autocast(device_type=device.split(":")[0]):
        batch_enc, batch_enc_len = asr_model.forward(
            input_signal=padded, input_signal_length=lengths_dev
        )

    # --- Compare ---
    # Batched inference can produce slightly different encoded lengths than
    # single-sample inference because the preprocessor's conv layers see
    # different padding.  We compare only the overlapping prefix.
    for i, single in enumerate(single_outputs):
        actual_len = batch_enc_len[i].item()
        batched = batch_enc[i, :, :actual_len].cpu()

        single_f32 = single.float()
        batched_f32 = batched.float()

        # Compare up to the shorter of the two time lengths (dim 1 = T)
        min_len = min(single_f32.shape[1], batched_f32.shape[1])
        if single_f32.shape[1] != batched_f32.shape[1]:
            logger.warning(
                f"Batch validation sample {i}: encoded length differs "
                f"(single={single_f32.shape[1]}, batched={batched_f32.shape[1]}). "
                "This is expected — padding affects conv-based encoders. "
                "Comparing first {min_len} frames."
            )
        s = single_f32[:, :min_len]
        b = batched_f32[:, :min_len]

        if not torch.allclose(s, b, atol=1e-2):
            raise RuntimeError(
                f"Batch validation FAILED for sample {i}: large divergence "
                f"(max diff={torch.max(torch.abs(s - b)):.6f}). "
                "Batched inference may be unreliable — run with --skip-batch-val "
                "and --batch-size 1 as a workaround."
            )
        if not torch.allclose(s, b, atol=1e-3):
            logger.warning(
                f"Batch validation sample {i}: minor mismatch "
                f"(max diff={torch.max(torch.abs(s - b)):.6f}). "
                "This is usually harmless with fp16 autocast."
            )

    logger.info(
        f"Batch validation passed ({len(single_outputs)} samples, "
        "single vs batched outputs match within tolerance)."
    )


@click.command()
@click.option(
    "--model",
    required=True,
    help="Pretrained model name (e.g. 'nvidia/parakeet-tdt-0.6b-v3') or path to .nemo file.",
)
@click.option(
    "--manifests",
    required=True,
    multiple=True,
    help="One or more NeMo manifest JSONL files to process.",
)
@click.option(
    "--cache-dir",
    required=True,
    type=click.Path(),
    help="Directory to write cached .pt files and cache_manifest.json.",
)
@click.option(
    "--device",
    default="cuda",
    help="Torch device for inference (default: cuda).",
)
@click.option(
    "--sample-rate",
    default=16000,
    type=int,
    help="Expected audio sample rate.",
)
@click.option(
    "--precision",
    default="fp16",
    type=click.Choice(["fp16", "fp32"]),
    help="Storage precision for cached tensors (fp16 saves ~50%% disk).",
)
@click.option(
    "--batch-size",
    default=1,
    type=int,
    help="Number of audio samples to process per batch (default: 1).",
)
@click.option(
    "--num-workers",
    default=4,
    type=int,
    help="DataLoader worker processes for parallel audio loading (default: 4).",
)
@click.option(
    "--skip-batch-val",
    is_flag=True,
    default=False,
    help="Skip the startup batch-vs-single validation check.",
)
@click.option(
    "--training-config",
    default=None,
    type=click.Path(exists=True),
    help="Path to training YAML (e.g. perso/training_config.yaml). "
    "Required when --compute-stats is set, to read augmentor + spec_augment config.",
)
@click.option(
    "--compute-stats",
    is_flag=True,
    default=False,
    help="After caching, compute per-dimension encoder variability stats "
    "under audio augmentations and save to cache_stats.json.",
)
@click.option(
    "--stats-samples",
    default=1000,
    type=int,
    help="Number of random samples to use for stats computation (default: 1000).",
)
@click.option(
    "--stats-augmentations",
    default=10,
    type=int,
    help="Number of augmented forward passes per sample for stats (default: 10).",
)
@click.option(
    "--cut-layer",
    default=None,
    type=int,
    help="If set (>=0), cache activations after encoder.layers[cut_layer] "
    "instead of the full encoder output. Used when training with the last N "
    "encoder layers unfrozen. If unset, derived from --training-config "
    "(model.freeze.encoder_except_last_n).",
)
@click.option(
    "--check-only",
    is_flag=True,
    default=False,
    help="Do not load the model or run inference; only verify cache freshness "
    "(global fingerprint + per-sample stat/hash). Exit code 0 if the cache "
    "is fully up to date, 1 otherwise. Used by training_start.sh to skip "
    "the heavy precompute call when nothing has changed.",
)
@click.option(
    "--hash-workers",
    default=8,
    type=int,
    help="Threads for the audio stat+hash validation pass (default: 8). "
    "Stat-only is near-instant; hashing is I/O-bound.",
)
@click.option(
    "--yes",
    "-y",
    "assume_yes",
    is_flag=True,
    default=False,
    help="Assume 'yes' on global-mismatch wipe prompts (non-interactive).",
)
@click.option(
    "--variants-per-sample",
    default=None,
    type=int,
    help="Total cached variants per audio (default: 1, matching old behaviour). "
    "Variant 0 is always a clean pass (eval mode, no augmentation). "
    "Variants 1..N-1 are augmented draws through the training augmentor + "
    "SpecAugment (asr_model.train() is set, enabling encoder dropout too). "
    "Requires --training-config when N>1. "
    "Overrides model.cache_variants_per_sample in the YAML.",
)
def main(
    model: str,
    manifests: tuple[str, ...],
    cache_dir: str,
    device: str,
    sample_rate: int,
    precision: str,
    batch_size: int,
    num_workers: int,
    skip_batch_val: bool,
    training_config: str | None,
    compute_stats: bool,
    stats_samples: int,
    stats_augmentations: int,
    cut_layer: int | None,
    check_only: bool,
    hash_workers: int,
    assume_yes: bool,
    variants_per_sample: int | None,
) -> None:
    """Precompute encoder activations and write a cache manifest."""
    # ------------------------------------------------------------------
    # Patch torch.load for NeMo compatibility (PyTorch 2.6+ defaults to
    # weights_only=True which breaks NeMo checkpoint loading).
    # ------------------------------------------------------------------
    _original_torch_load = torch.load
    torch.load = lambda *a, **kw: _original_torch_load(
        *a, **{**kw, "weights_only": kw.get("weights_only", False)}
    )

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    meta_path = cache_path / "cache_meta.json"
    index_path = cache_path / "cache_index.json"

    # ------------------------------------------------------------------
    # Load training config (cheap; needed for cut_layer derivation and stats)
    # ------------------------------------------------------------------
    _train_cfg = None
    encoder_except_last_n = 0
    if training_config is not None:
        from omegaconf import OmegaConf as _OC
        _train_cfg = _OC.load(training_config)
        encoder_except_last_n = int(
            _train_cfg.get("model", {}).get("freeze", {}).get("encoder_except_last_n", 0)
            or 0
        )

    # ------------------------------------------------------------------
    # Resolve num_variants: CLI overrides YAML, YAML overrides default of 1.
    # ------------------------------------------------------------------
    if variants_per_sample is not None:
        num_variants = int(variants_per_sample)
    elif _train_cfg is not None:
        num_variants = int(_train_cfg.get("model", {}).get("cache_variants_per_sample", 1) or 1)
    else:
        num_variants = 1

    if num_variants < 1:
        raise click.UsageError("--variants-per-sample must be >= 1.")
    if num_variants > 1 and training_config is None:
        raise click.UsageError(
            "--variants-per-sample > 1 requires --training-config "
            "(needed to load the audio augmentor + spec_augment config)."
        )
    logger.info(f"Variants per sample: {num_variants} (variant 0=clean, 1..{num_variants-1}=augmented)"
                if num_variants > 1 else "Variants per sample: 1 (clean only)")

    # ------------------------------------------------------------------
    # Read existing cache_meta.json (if any) — drives global mismatch check
    # without needing to load the model.
    # ------------------------------------------------------------------
    existing_meta: dict | None = None
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            existing_meta = json.load(f)

    # ------------------------------------------------------------------
    # Read manifests
    # ------------------------------------------------------------------
    entries: list[dict] = []
    for manifest in manifests:
        logger.info(f"Reading manifest: {manifest}")
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    logger.info(f"Total manifest entries: {len(entries)}")

    unique_paths: list[str] = sorted({e["audio_filepath"] for e in entries})
    for p in unique_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"Audio file not found: {p}")

    # ------------------------------------------------------------------
    # Helper: derive expected cut_layer when n_total_layers is known
    # (either from existing meta or from a freshly-loaded model).
    # ------------------------------------------------------------------
    def _derive_cut_layer(n_total_layers_known: int) -> int:
        if cut_layer is not None:
            return cut_layer
        if encoder_except_last_n > 0 and n_total_layers_known > 0:
            return n_total_layers_known - encoder_except_last_n - 1
        return -1

    # ==================================================================
    # --check-only fast path (no model load, no GPU)
    # ==================================================================
    if check_only:
        if existing_meta is None:
            logger.warning("cache_meta.json missing — cache is not initialised.")
            sys.exit(1)
        n_total_layers_meta = int(existing_meta.get("n_total_layers", 0) or 0)
        expected_cut = _derive_cut_layer(n_total_layers_meta)
        expected_fp = _global_config_fingerprint(model, expected_cut, sample_rate, num_variants)
        if existing_meta.get("config_fingerprint") != expected_fp:
            logger.warning(
                f"Global fingerprint mismatch "
                f"(stored={str(existing_meta.get('config_fingerprint'))[:12]}, "
                f"current={expected_fp[:12]}; "
                f"model={model}, cut_layer={expected_cut}, sample_rate={sample_rate}, "
                f"num_variants={num_variants})."
            )
            sys.exit(1)
        prev_index = _load_cache_index(index_path)
        current = _validate_audio_files_parallel(unique_paths, prev_index, hash_workers)
        fresh = stale = missing = 0
        for path in unique_paths:
            h = _stable_hash(path)
            # All N variants must exist for a sample to be considered fresh.
            all_present = all(
                (cache_path / _variant_pt_name(h, nn)).exists() for nn in range(num_variants)
            )
            if not all_present:
                missing += 1
            elif prev_index.get(path, {}).get("audio_sha256") != current[path]["audio_sha256"]:
                stale += 1
            else:
                fresh += 1
        rehashed = sum(1 for r in current.values() if not r["unchanged"])
        if missing or stale:
            logger.warning(
                f"Cache stale: fresh={fresh}, stale={stale}, missing={missing}, "
                f"rehashed={rehashed}/{len(unique_paths)}."
            )
            sys.exit(1)
        if compute_stats:
            stats_meta_path = cache_path / "stats_meta.json"
            stats_path = cache_path / "cache_stats.json"
            if not (stats_meta_path.exists() and stats_path.exists()):
                logger.warning("Stats files missing — cache_stats.json or stats_meta.json absent.")
                sys.exit(1)
            if _train_cfg is None:
                logger.warning("--compute-stats with --check-only requires --training-config.")
                sys.exit(1)
            with open(stats_meta_path) as f:
                existing_stats_meta = json.load(f)
            augmentor_cfg = _train_cfg.get("model", {}).get("train_ds", {}).get("augmentor", None)
            spec_aug_cfg = _train_cfg.get("model", {}).get("spec_augment", None)
            la_cfg = _train_cfg.get("model", {}).get("latent_augment", {}) or {}
            new_stats_fp = _stats_fingerprint(
                augmentor_cfg, spec_aug_cfg, sample_rate, expected_cut,
                stats_samples, stats_augmentations,
                float(la_cfg.get("outlier_top_k_frac", 0.01)),
                float(la_cfg.get("outlier_kurtosis_threshold", 20.0)),
            )
            if existing_stats_meta != new_stats_fp:
                logger.warning("Stats fingerprint mismatch — cache_stats.json is stale.")
                sys.exit(1)
        logger.info(
            f"Cache is fresh: {fresh} entries, fingerprint={expected_fp[:12]} "
            f"(rehashed={rehashed} due to stat changes; others stat-skipped)."
        )
        sys.exit(0)

    # ==================================================================
    # Normal precompute path
    # ==================================================================

    # ------------------------------------------------------------------
    # Global mismatch check BEFORE loading the model (saves ~10 s if user
    # answers 'no' to a wipe prompt).
    # ------------------------------------------------------------------
    if existing_meta is not None:
        n_total_layers_meta = int(existing_meta.get("n_total_layers", 0) or 0)
        expected_cut = _derive_cut_layer(n_total_layers_meta)
        expected_fp = _global_config_fingerprint(model, expected_cut, sample_rate, num_variants)
        if existing_meta.get("config_fingerprint") != expected_fp:
            diffs = []
            if existing_meta.get("model") != model:
                diffs.append(f"  model: {existing_meta.get('model')} -> {model}")
            if existing_meta.get("cut_layer") != expected_cut:
                diffs.append(f"  cut_layer: {existing_meta.get('cut_layer')} -> {expected_cut}")
            if existing_meta.get("sample_rate") != sample_rate:
                diffs.append(f"  sample_rate: {existing_meta.get('sample_rate')} -> {sample_rate}")
            if existing_meta.get("num_variants", 1) != num_variants:
                diffs.append(f"  num_variants: {existing_meta.get('num_variants', 1)} -> {num_variants}")
            if not diffs:
                diffs = [f"  (fingerprint differs; no individual field changed — likely legacy meta)"]
            logger.warning(f"Global cache fingerprint mismatch at {cache_path}:")
            for d in diffs:
                logger.warning(d)
            if assume_yes:
                confirm = "y"
            else:
                confirm = click.prompt(
                    "Wipe existing cache and rebuild from scratch? [y/N]",
                    default="n", show_default=False,
                )
            if str(confirm).strip().lower() not in {"y", "yes"}:
                raise click.UsageError("Aborted by user (cache mismatch).")
            logger.warning(f"Wiping {cache_path} …")
            shutil.rmtree(cache_path)
            cache_path.mkdir(parents=True)
            existing_meta = None

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    from nemo.collections.asr.models import ASRModel

    logger.info(f"Loading model: {model}")
    if model.endswith(".nemo"):
        asr_model = ASRModel.restore_from(restore_path=model)
    else:
        asr_model = ASRModel.from_pretrained(model_name=model)

    asr_model = asr_model.to(device)
    asr_model.eval()
    asr_model.freeze()
    logger.info("Model loaded, frozen, and in eval mode.")

    # ------------------------------------------------------------------
    # Resolve cut_layer (now with authoritative n_total_layers from the model)
    # ------------------------------------------------------------------
    n_total_layers = len(asr_model.encoder.layers)
    encoder_dim = asr_model.encoder.d_model
    cut_layer = _derive_cut_layer(n_total_layers)

    if cut_layer < -1 or cut_layer >= n_total_layers:
        raise click.UsageError(
            f"--cut-layer={cut_layer} out of range (encoder has {n_total_layers} layers; "
            f"valid range: -1 .. {n_total_layers - 1})"
        )

    if cut_layer >= 0:
        logger.info(
            f"PARTIAL CACHE MODE: caching after encoder.layers[{cut_layer}] "
            f"(of {n_total_layers}). Last {n_total_layers - cut_layer - 1} "
            f"layers + out_proj will run live during training."
        )
    else:
        logger.info(f"FULL CACHE MODE: caching final encoder output (all {n_total_layers} layers).")

    # Cross-check existing meta against authoritative model layers (paranoia).
    if existing_meta is not None and int(existing_meta.get("n_total_layers", n_total_layers)) != n_total_layers:
        raise click.UsageError(
            f"Existing cache_meta.n_total_layers={existing_meta.get('n_total_layers')} "
            f"but loaded model has {n_total_layers}. The pre-load fingerprint check "
            f"should have caught this — please delete {cache_path} and retry."
        )

    # Import partial_encoder helper from this directory (perso/)
    import importlib.util as _ilu, sys as _sys
    _pe_spec = _ilu.spec_from_file_location(
        "perso_partial_encoder", Path(__file__).parent / "partial_encoder.py"
    )
    _pe = _ilu.module_from_spec(_pe_spec)
    _sys.modules["perso_partial_encoder"] = _pe
    _pe_spec.loader.exec_module(_pe)
    truncated_encoder = _pe.truncated_encoder

    # ------------------------------------------------------------------
    # Per-sample validation: stat each audio file, hash on stat-mismatch.
    # ------------------------------------------------------------------
    prev_index = _load_cache_index(index_path)
    current_records = _validate_audio_files_parallel(
        unique_paths, prev_index, hash_workers
    )

    # Bucket entries — a sample needs processing if variant 0 is missing/stale.
    # (Augmented variants 1..N-1 are generated separately in Path B below.)
    cache_manifest_lines: list[str] = []
    to_process: list[dict] = []
    fresh = stale = missing = 0
    use_fp16 = precision == "fp16"
    seen_to_process: set[str] = set()
    for entry in entries:
        audio_filepath = entry["audio_filepath"]
        h = _stable_hash(audio_filepath)
        v0_path = cache_path / _variant_pt_name(h, 0)
        cur_hash = current_records[audio_filepath]["audio_sha256"]
        prev_hash = prev_index.get(audio_filepath, {}).get("audio_sha256")
        if not v0_path.exists():
            missing += 1
            if audio_filepath not in seen_to_process:
                seen_to_process.add(audio_filepath)
                to_process.append(entry)
        elif prev_hash != cur_hash:
            stale += 1
            if audio_filepath not in seen_to_process:
                seen_to_process.add(audio_filepath)
                to_process.append(entry)
        else:
            fresh += 1
            cache_manifest_lines.append(
                json.dumps(
                    {
                        "cache_filepath": str(v0_path),
                        "audio_filepath": audio_filepath,
                        "text": entry.get("text", ""),
                        "duration": entry.get("duration", 0.0),
                    },
                    ensure_ascii=False,
                )
            )

    rehashed = sum(1 for r in current_records.values() if not r["unchanged"])
    logger.info(
        f"Per-sample status: fresh={fresh}, stale={stale}, missing={missing}, "
        f"rehashed={rehashed}/{len(unique_paths)} (others stat-skipped)."
    )

    # Sort by duration so batches contain similar-length samples → less padding waste
    to_process.sort(key=lambda e: e.get("duration", 0.0))
    logger.info(f"Unique entries to process: {len(to_process)} (sorted by duration)")

    # ------------------------------------------------------------------
    # Batch validation — sanity-check that batched encoder output matches
    # single-sample output before committing to the full run.
    # ------------------------------------------------------------------
    if not skip_batch_val and batch_size > 1 and to_process:
        logger.info("Running batch validation (disable with --skip-batch-val)…")
        with truncated_encoder(asr_model.encoder, cut_layer):
            _validate_batched_inference(
                asr_model,
                to_process,
                sample_rate=sample_rate,
                device=device,
            )

    # ------------------------------------------------------------------
    # Batched inference via DataLoader
    # ------------------------------------------------------------------
    dataset = AudioDataset(entries=to_process, sample_rate=sample_rate)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_audio,
        # pin_memory speeds up host→device transfer when num_workers > 0
        pin_memory=(num_workers > 0 and device.startswith("cuda")),
    )

    processed_paths: set[str] = set()
    for batch in tqdm(loader, desc="Caching encoder outputs (variant 0, clean)", smoothing=0.01):
        indices, padded_audio, audio_lengths = batch
        padded_audio = padded_audio.to(device)
        audio_lengths = audio_lengths.to(device)

        with torch.no_grad(), torch.amp.autocast(device_type=device.split(":")[0]):
            with truncated_encoder(asr_model.encoder, cut_layer):
                encoded, encoded_len = asr_model.forward(
                    input_signal=padded_audio, input_signal_length=audio_lengths
                )

        # Save each sample individually, trimmed to its actual encoded length
        for i, dataset_idx in enumerate(indices):
            entry = to_process[dataset_idx]
            audio_filepath = entry["audio_filepath"]
            h = _stable_hash(audio_filepath)
            v0_path = cache_path / _variant_pt_name(h, 0)

            actual_len = encoded_len[i].item()
            # Trim padding frames so the cached tensor is compact
            save_encoded = encoded[i, :, :actual_len].cpu()
            if use_fp16:
                save_encoded = save_encoded.half()
            save_len = encoded_len[i].cpu()

            torch.save(
                {
                    "encoded": save_encoded,
                    "encoded_len": save_len,
                    "cut_layer": cut_layer,
                },
                v0_path,
            )
            processed_paths.add(audio_filepath)

    # Append cache_manifest lines for every manifest entry (incl. duplicates)
    # whose audio was just (re)processed. Lines for already-fresh entries were
    # appended during the bucketing pass above.
    for entry in entries:
        audio_filepath = entry["audio_filepath"]
        if audio_filepath in processed_paths:
            v0_path = cache_path / _variant_pt_name(_stable_hash(audio_filepath), 0)
            cache_manifest_lines.append(
                json.dumps(
                    {
                        "cache_filepath": str(v0_path),
                        "audio_filepath": audio_filepath,
                        "text": entry.get("text", ""),
                        "duration": entry.get("duration", 0.0),
                    },
                    ensure_ascii=False,
                )
            )

    # ------------------------------------------------------------------
    # Path B: augmented variants 1..N-1 (per-sample loop, train mode)
    #
    # asr_model.train() enables SpecAugment (gated on self.training in NeMo)
    # and encoder dropout — both contribute additional stochasticity which is
    # intentional here. Variant 0 (clean, from Path A above) is never touched.
    #
    # For each unique audio file we generate only the variants that are still
    # missing (smart regen: increasing N adds new variants, doesn't redo existing).
    # ------------------------------------------------------------------
    if num_variants > 1:
        from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
        from omegaconf import OmegaConf as _OC2

        _train_cfg2 = _OC2.load(training_config)  # type: ignore[arg-type]
        augmentor, _ = _build_augmentations(_train_cfg2, asr_model)

        # Determine which unique paths still need augmented variants.
        paths_needing_aug: list[str] = []
        for path in unique_paths:
            h = _stable_hash(path)
            needed = [
                nn for nn in range(1, num_variants)
                if not (cache_path / _variant_pt_name(h, nn)).exists()
            ]
            if needed:
                paths_needing_aug.append(path)

        if paths_needing_aug:
            logger.info(
                f"Generating augmented variants 1..{num_variants - 1} "
                f"for {len(paths_needing_aug)}/{len(unique_paths)} unique audio files "
                f"({num_variants - 1} variants each)."
            )
            asr_model.train()
            try:
                for audio_filepath in tqdm(
                    paths_needing_aug,
                    desc=f"Augmented variants (1..{num_variants - 1})",
                    smoothing=0.01,
                ):
                    h = _stable_hash(audio_filepath)
                    for nn in range(1, num_variants):
                        vn_path = cache_path / _variant_pt_name(h, nn)
                        if vn_path.exists():
                            continue  # already present, skip

                        segment = AudioSegment.from_file(
                            audio_file=audio_filepath,
                            target_sr=sample_rate,
                        )
                        if augmentor is not None:
                            augmentor.perturb(segment)

                        audio_tensor = (
                            torch.tensor(segment.samples, dtype=torch.float32)
                            .unsqueeze(0)
                            .to(device)
                        )
                        audio_len = torch.tensor([audio_tensor.shape[1]], device=device)

                        with torch.no_grad(), torch.amp.autocast(
                            device_type=device.split(":")[0]
                        ):
                            with truncated_encoder(asr_model.encoder, cut_layer):
                                encoded, encoded_len = asr_model.forward(
                                    input_signal=audio_tensor,
                                    input_signal_length=audio_len,
                                )

                        actual_len = encoded_len[0].item()
                        save_encoded = encoded[0, :, :actual_len].cpu()
                        if use_fp16:
                            save_encoded = save_encoded.half()

                        torch.save(
                            {
                                "encoded": save_encoded,
                                "encoded_len": encoded_len[0].cpu(),
                                "cut_layer": cut_layer,
                            },
                            vn_path,
                        )
            finally:
                asr_model.eval()
                logger.info("Restored model to eval mode after augmented variant generation.")
        else:
            logger.info(
                f"All augmented variants 1..{num_variants - 1} already present — nothing to generate."
            )

    # ------------------------------------------------------------------
    # Write cache manifest + extended metadata + per-sample index
    # ------------------------------------------------------------------
    manifest_out = cache_path / "cache_manifest.json"
    with open(manifest_out, "w", encoding="utf-8") as f:
        f.write("\n".join(cache_manifest_lines) + "\n")

    config_fingerprint = _global_config_fingerprint(model, cut_layer, sample_rate, num_variants)
    meta_out = cache_path / "cache_meta.json"
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model,
                "cut_layer": cut_layer,
                "sample_rate": sample_rate,
                "n_total_layers": n_total_layers,
                "encoder_dim": encoder_dim,
                "num_variants": num_variants,
                "config_fingerprint": config_fingerprint,
            },
            f,
            indent=2,
        )
    logger.info(f"Wrote cache metadata: {meta_out} (fingerprint={config_fingerprint[:12]})")

    # Build per-sample index from current_records (covers all unique paths).
    # pt_names is a list covering all N variant filenames (some may not exist
    # yet if augmented variants are being added incrementally).
    index_records = []
    for path in unique_paths:
        rec = current_records[path]
        h = _stable_hash(path)
        index_records.append(
            {
                "audio_filepath": path,
                "size": rec["size"],
                "mtime": rec["mtime"],
                "audio_sha256": rec["audio_sha256"],
                "pt_names": [_variant_pt_name(h, nn) for nn in range(num_variants)],
            }
        )
    _write_cache_index(index_path, index_records)
    logger.info(f"Wrote cache index: {index_path} ({len(index_records)} entries)")

    logger.info(
        f"Done. {len(cache_manifest_lines)} cache_manifest entries at {cache_path} "
        f"(fresh={fresh}, stale={stale}, missing={missing}, recomputed={len(processed_paths)}). "
        f"Manifest: {manifest_out}"
    )

    # ------------------------------------------------------------------
    # Orphan report — .pt files whose audio_filepath isn't referenced by
    # any current manifest. No deletion (per project policy).
    # ------------------------------------------------------------------
    h_to_path = {_stable_hash(p): p for p in unique_paths}
    referenced_pt_names = {
        _variant_pt_name(h, nn) for h in h_to_path for nn in range(num_variants)
    }
    on_disk_pt_names = {p.name for p in cache_path.glob("*.pt")}
    orphan_pt_names = sorted(on_disk_pt_names - referenced_pt_names)
    if orphan_pt_names:
        # Try to map orphan .pt -> original audio path via hash prefix in filename.
        # Format is {hash}_v{NN}.pt; extract hash prefix (first 16 chars).
        sample_paths = []
        for n in orphan_pt_names[:5]:
            h_prefix = n[:16]
            sample_paths.append(h_to_path.get(h_prefix, n))
        logger.warning(
            f"Found {len(orphan_pt_names)} orphan .pt file(s) in {cache_path} "
            f"not referenced by any current manifest entry. "
            f"First 5: {sample_paths}. (No deletion performed.)"
        )

    # ------------------------------------------------------------------
    # Per-dimension latent augmentation stats
    # ------------------------------------------------------------------
    if compute_stats:
        if training_config is None:
            raise click.UsageError(
                "--compute-stats requires --training-config "
                "(path to training YAML with augmentor + spec_augment config)."
            )
        stats_meta_path = cache_path / "stats_meta.json"
        stats_path = cache_path / "cache_stats.json"
        augmentor_cfg = _train_cfg.get("model", {}).get("train_ds", {}).get("augmentor", None)
        spec_aug_cfg = _train_cfg.get("model", {}).get("spec_augment", None)
        la_cfg = _train_cfg.get("model", {}).get("latent_augment", {}) or {}
        outlier_top_k_frac = float(la_cfg.get("outlier_top_k_frac", 0.01))
        outlier_kurtosis_threshold = float(la_cfg.get("outlier_kurtosis_threshold", 20.0))
        new_stats_meta = _stats_fingerprint(
            augmentor_cfg, spec_aug_cfg, sample_rate, cut_layer,
            stats_samples, stats_augmentations,
            outlier_top_k_frac, outlier_kurtosis_threshold,
        )
        existing_stats_meta = None
        if stats_meta_path.exists():
            with open(stats_meta_path) as f:
                existing_stats_meta = json.load(f)
        if (
            stats_path.exists()
            and existing_stats_meta == new_stats_meta
        ):
            logger.info(
                f"Stats are up to date (fingerprint matches); skipping "
                f"_compute_augmentation_stats. Delete {stats_meta_path} to force."
            )
        else:
            # Wrap stats with the same truncation so per-dim std reflects the
            # actual cached representation (when partial caching is enabled).
            with truncated_encoder(asr_model.encoder, cut_layer):
                _compute_augmentation_stats(
                    asr_model=asr_model,
                    entries=entries,
                    training_config=training_config,
                    cache_path=cache_path,
                    device=device,
                    sample_rate=sample_rate,
                    stats_samples=stats_samples,
                    stats_augmentations=stats_augmentations,
                    outlier_top_k_frac=outlier_top_k_frac,
                    outlier_kurtosis_threshold=outlier_kurtosis_threshold,
                )
            with open(stats_meta_path, "w") as f:
                json.dump(new_stats_meta, f, indent=2)
            logger.info(f"Wrote stats metadata: {stats_meta_path}")


def _build_augmentations(train_cfg, asr_model):
    """Build audio augmentor and attach SpecAugment to asr_model from training config.

    Returns ``(augmentor, spec_augment)``. The spec_augment is also attached to
    ``asr_model.spec_augmentation`` so it activates when ``asr_model.train()`` is set.
    """
    from nemo.collections.asr.models import ASRModel
    from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations

    augmentor_cfg = train_cfg.model.get("train_ds", {}).get("augmentor", None)
    augmentor = process_augmentations(augmentor_cfg) if augmentor_cfg else None
    if augmentor is not None:
        logger.info(f"Audio augmentor loaded with {len(augmentor._pipeline)} augmentation(s).")
    else:
        logger.warning("No audio augmentor found in training config — only spec_augment will be used.")

    spec_aug_cfg = train_cfg.model.get("spec_augment", None)
    spec_augment = None
    if spec_aug_cfg is not None:
        spec_augment = ASRModel.from_config_dict(spec_aug_cfg)
        asr_model.spec_augmentation = spec_augment
        logger.info(f"SpectrogramAugmentation attached: {spec_augment}")
    else:
        logger.warning("No spec_augment found in training config.")

    return augmentor, spec_augment


def _compute_augmentation_stats(
    asr_model: torch.nn.Module,
    entries: list[dict],
    training_config: str,
    cache_path: Path,
    device: str,
    sample_rate: int,
    stats_samples: int,
    stats_augmentations: int,
    outlier_top_k_frac: float = 0.01,
    outlier_kurtosis_threshold: float = 20.0,
) -> None:
    """Measure per-dimension encoder variability under audio augmentations.

    For a random subset of *entries*, runs *stats_augmentations* augmented
    forward passes through the full preprocessor → spec_augment → encoder
    pipeline. For each sample, computes per-dimension std across the
    augmented outputs, then averages those stds across all samples.

    The result is saved to ``<cache_path>/cache_stats.json`` with the
    shape ``(D,)`` vector ``per_dim_std``, which the training script uses
    to scale latent Gaussian noise proportionally per encoder dimension.

    Parameters
    ----------
    asr_model : torch.nn.Module
        The pretrained ASR model (already loaded, on *device*).
    entries : list[dict]
        All manifest entries (used to pick random samples).
    training_config : str
        Path to the training YAML file containing ``model.train_ds.augmentor``
        and ``model.spec_augment`` configuration.
    cache_path : Path
        Output directory for ``cache_stats.json``.
    device : str
        Torch device string (e.g. "cuda").
    sample_rate : int
        Expected audio sample rate (must match model).
    stats_samples : int
        Number of random samples to use.
    stats_augmentations : int
        Number of augmented forward passes per sample.
    """
    import random

    from omegaconf import OmegaConf
    from nemo.collections.asr.parts.preprocessing.segment import AudioSegment

    logger.info("--- Computing per-dimension augmentation stats ---")

    # Load training config to get augmentor and spec_augment settings
    train_cfg = OmegaConf.load(training_config)

    # Build augmentor + spec_augment; spec_augment is attached to asr_model.
    augmentor, _ = _build_augmentations(train_cfg, asr_model)

    # Put model in train mode so spec_augment is active, but keep encoder
    # weights frozen (no gradient tracking needed).
    asr_model.train()

    # Pick random subset of samples
    n_samples = min(stats_samples, len(entries))
    selected = random.sample(entries, k=n_samples)
    logger.info(
        f"Running {stats_augmentations} augmented passes on {n_samples} samples "
        f"({n_samples * stats_augmentations} total forward passes)."
    )

    # Collect per-sample per-dim stds (for noise scaling) and accumulate
    # global running moments of activations across all augmented frames
    # (for outlier detection: mean, mean_abs, kurtosis).
    per_sample_stds: list[torch.Tensor] = []
    # Global running sums of x, x^2, x^3, x^4, |x|, plus count.
    # All shape (D,) — accumulated lazily once D is known.
    running = {
        "n": None, "s1": None, "s2": None, "s3": None, "s4": None, "sabs": None,
    }

    for i, entry in enumerate(tqdm(selected, desc="Stats: augmented passes", smoothing=0.01)):
        audio_path = entry["audio_filepath"]

        # Collect encoder outputs across augmented passes for this sample
        aug_outputs: list[torch.Tensor] = []

        for _ in range(stats_augmentations):
            # Load audio via AudioSegment (the format augmentors expect)
            segment = AudioSegment.from_file(
                audio_file=audio_path,
                target_sr=sample_rate,
            )

            # Apply audio-level augmentations (modifies segment in-place,
            # but AudioSegment.from_file creates a fresh copy each time)
            if augmentor is not None:
                augmentor.perturb(segment)

            # Convert to tensor and run through the full pipeline
            audio_tensor = torch.tensor(
                segment.samples, dtype=torch.float32
            ).unsqueeze(0).to(device)
            audio_len = torch.tensor(
                [audio_tensor.shape[1]], device=device
            )

            with torch.no_grad(), torch.amp.autocast(
                device_type=device.split(":")[0]
            ):
                encoded, encoded_len = asr_model.forward(
                    input_signal=audio_tensor,
                    input_signal_length=audio_len,
                )

            # Trim to valid length → shape (D, T_valid), then transpose
            # to (T_valid, D) for easier stacking later
            actual_len = encoded_len[0].item()
            trimmed = encoded[0, :, :actual_len].float().cpu().t()  # (T, D)
            aug_outputs.append(trimmed)

        # Stack all augmented outputs: find the shortest T across passes
        # (different augmentations can produce slightly different lengths
        # because gain/noise can affect the preprocessor's conv boundaries)
        min_t = min(out.shape[0] for out in aug_outputs)
        # (N_aug, T_min, D)
        stacked = torch.stack([out[:min_t, :] for out in aug_outputs], dim=0)

        # Per-dimension std across augmented passes, then mean over time
        # shape: std → (N_aug, T_min, D) → std over dim 0 → (T_min, D) → mean over T → (D,)
        dim_std = stacked.std(dim=0).mean(dim=0)  # (D,)
        per_sample_stds.append(dim_std)

        # Accumulate global running moments across all augmented frames for
        # this sample. flat shape: (N_aug * T_min, D).
        flat = stacked.reshape(-1, stacked.shape[-1])
        if running["n"] is None:
            D = flat.shape[1]
            running["n"] = torch.zeros((), dtype=torch.float64)
            running["s1"] = torch.zeros(D, dtype=torch.float64)
            running["s2"] = torch.zeros(D, dtype=torch.float64)
            running["s3"] = torch.zeros(D, dtype=torch.float64)
            running["s4"] = torch.zeros(D, dtype=torch.float64)
            running["sabs"] = torch.zeros(D, dtype=torch.float64)
        flat64 = flat.to(torch.float64)
        running["n"] += flat64.shape[0]
        running["s1"] += flat64.sum(dim=0)
        running["s2"] += (flat64 ** 2).sum(dim=0)
        running["s3"] += (flat64 ** 3).sum(dim=0)
        running["s4"] += (flat64 ** 4).sum(dim=0)
        running["sabs"] += flat64.abs().sum(dim=0)

    # Average per-sample stds → one (D,) vector
    mean_per_dim_std = torch.stack(per_sample_stds, dim=0).mean(dim=0)

    # Derive global per-dim moments from running sums.
    n = running["n"]
    mean = running["s1"] / n
    var = running["s2"] / n - mean ** 2
    var = var.clamp(min=1e-12)
    std_global = var.sqrt()
    mean_abs = running["sabs"] / n
    # Central 4th moment: E[x^4] - 4*μ*E[x^3] + 6*μ^2*E[x^2] - 3*μ^4
    m2 = running["s2"] / n
    m3 = running["s3"] / n
    m4 = running["s4"] / n
    mu = mean
    central_m4 = m4 - 4 * mu * m3 + 6 * mu ** 2 * m2 - 3 * mu ** 4
    kurtosis = central_m4 / (var ** 2)

    # Protected dims: union of top-k by mean_abs and dims with kurtosis above threshold.
    D = mean.shape[0]
    top_k = max(0, int(D * outlier_top_k_frac))
    mag_idx = mean_abs.topk(top_k).indices if top_k > 0 else torch.empty(0, dtype=torch.long)
    kurt_idx = (kurtosis > outlier_kurtosis_threshold).nonzero(as_tuple=False).squeeze(-1)
    protected = torch.zeros(D, dtype=torch.bool)
    if mag_idx.numel() > 0:
        protected[mag_idx] = True
    if kurt_idx.numel() > 0:
        protected[kurt_idx] = True

    # Save to cache_stats.json
    stats_out = cache_path / "cache_stats.json"
    stats_data = {
        "num_samples": n_samples,
        "num_augmentations": stats_augmentations,
        "per_dim_std": mean_per_dim_std.tolist(),
        "per_dim_mean": mean.float().tolist(),
        "per_dim_mean_abs": mean_abs.float().tolist(),
        "per_dim_std_global": std_global.float().tolist(),
        "per_dim_kurtosis": kurtosis.float().tolist(),
        "protected_dims": protected.tolist(),
        "outlier_top_k_frac": outlier_top_k_frac,
        "outlier_kurtosis_threshold": outlier_kurtosis_threshold,
    }
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)

    n_mag = int(mag_idx.numel())
    n_kurt = int(kurt_idx.numel())
    n_prot = int(protected.sum())
    logger.info(
        f"Saved per-dimension stats to {stats_out} "
        f"({mean_per_dim_std.shape[0]} dims, "
        f"mean std={mean_per_dim_std.mean():.6f}, "
        f"min={mean_per_dim_std.min():.6f}, "
        f"max={mean_per_dim_std.max():.6f})"
    )
    logger.info(
        f"Protected {n_prot}/{D} dims "
        f"(top-{top_k} by mean_abs={n_mag}, kurtosis>{outlier_kurtosis_threshold}={n_kurt}). "
        f"First 10 indices: {protected.nonzero(as_tuple=False).squeeze(-1)[:10].tolist()}"
    )


if __name__ == "__main__":
    main()
