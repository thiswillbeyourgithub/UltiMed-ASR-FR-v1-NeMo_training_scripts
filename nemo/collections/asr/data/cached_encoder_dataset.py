"""Dataset that loads precomputed encoder activations instead of audio.

When the encoder is frozen, re-running it every step wastes ~97% of
the forward-pass compute.  This dataset reads cached ``.pt`` files
(produced by ``precompute_encoder_cache.py``) and returns batches in
the same ``(encoded, encoded_len, transcript, transcript_len)`` format
that ``EncDecRNNTModel.training_step`` expects *after* the encoder
forward pass — so the rest of the training loop (decoder → joint → loss)
works unchanged.

Transcript tokenisation is done on-the-fly using the model's tokenizer
so the cache stays tokenizer-agnostic.

The dataset reads the **training manifests** (with any repetitions
defined in ``train_ds.manifest_filepath``) and resolves each sample to
its cached ``.pt`` file via the same ``_stable_hash(audio_filepath)``
used by ``precompute_encoder_cache.py``.  This means manifest-level
upsampling (listing the same manifest N times) is honoured correctly.

Variant support
---------------
``precompute_encoder_cache.py --variants-per-sample N`` generates up to
N cached versions per audio: variant 0 is always clean (eval mode, no
augmentation); variants 1..N-1 are augmented draws from the training
augmentor + SpecAugment.  At each ``__getitem__`` call the dataset
randomly picks one variant from the available list, giving the decoder
and live encoder layers genuine acoustic diversity without re-running the
full encoder.

Backward compatibility
----------------------
Old caches use ``{hash}.pt`` (no ``_vNN`` suffix).  If no ``_v*`` files
are found, the dataset falls back to the legacy single file so existing
caches remain usable without any migration.

Usage
-----
Instantiate via the training script when ``use_cached_encoder: true``.
See ``speech_to_text_finetune_cached.py`` for integration details.

Note: developed with Claude Code (https://claude.ai/claude-code).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)


def _stable_hash(text: str) -> str:
    """Deterministic short hash for file naming — must match precompute_encoder_cache.py."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _variant_pt_name(audio_hash: str, variant_idx: int) -> str:
    """Return the filename for a specific variant — must match precompute_encoder_cache.py."""
    return f"{audio_hash}_v{variant_idx:02d}.pt"


class CachedEncoderDataset(Dataset):
    """Reads cached encoder outputs for samples listed in training manifests.

    Parameters
    ----------
    manifest_filepaths : list[str]
        Training manifest JSONL files, potentially repeated for upsampling.
        Each line must have ``audio_filepath`` and ``text``.
    cache_dir : str or Path
        Directory containing ``.pt`` files produced by
        ``precompute_encoder_cache.py``.
    tokenizer
        NeMo tokenizer instance (``asr_model.tokenizer``).  Must
        expose a ``text_to_ids(text) -> list[int]`` method.
    max_duration : float or None
        If set, skip samples whose ``duration`` exceeds this (seconds).
    min_duration : float or None
        If set, skip samples whose ``duration`` is below this (seconds).
    """

    def __init__(
        self,
        manifest_filepaths: list[str],
        cache_dir: str | Path,
        tokenizer,
        *,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.cache_dir = Path(cache_dir)
        self.entries: list[dict] = []

        # Read cache metadata if present (produced by precompute_encoder_cache.py).
        # cut_layer == -1 → full-encoder cache (legacy / default).
        # cut_layer >= 0  → partial cache (training step must run remaining layers).
        meta_path = self.cache_dir / "cache_meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.cache_meta = json.load(f)
            logger.info(
                f"cache_meta: model={self.cache_meta.get('model')}, "
                f"cut_layer={self.cache_meta.get('cut_layer')}, "
                f"num_variants={self.cache_meta.get('num_variants', 1)}, "
                f"sample_rate={self.cache_meta.get('sample_rate')}, "
                f"fingerprint={str(self.cache_meta.get('config_fingerprint', ''))[:12]}"
            )
        else:
            logger.warning(
                f"No cache_meta.json at {meta_path}; assuming legacy cache "
                f"with cut_layer=-1 (full encoder output)."
            )
            self.cache_meta = {"cut_layer": -1}
        self.cut_layer: int = int(self.cache_meta.get("cut_layer", -1))

        # Build hash→variants index with a single directory scan.
        logger.info(f"Scanning cache directory {self.cache_dir} ...")
        cache_index: dict[str, list[str]] = {}
        for pt_path in tqdm(self.cache_dir.glob("*.pt"), desc="Indexing cache", unit="file", leave=True):
            name = pt_path.stem  # e.g. "abc123_v00" or legacy "abc123"
            if "_v" in name:
                h = name.rsplit("_v", 1)[0]
            else:
                h = name
            cache_index.setdefault(h, []).append(str(pt_path))
        for h in cache_index:
            cache_index[h].sort()
        logger.info(f"Cache index built: {len(cache_index)} unique hashes.")

        missing = set()
        for manifest_path in manifest_filepaths:
            with open(manifest_path, "r", encoding="utf-8") as f:
                lines = [l for l in f if l.strip()]
            with tqdm(lines, desc=f"Loading {Path(manifest_path).name}", unit="sample", leave=True) as pbar:
                for line in pbar:
                    entry = json.loads(line)
                    dur = entry.get("duration", 0.0)
                    if max_duration is not None and dur > max_duration:
                        continue
                    if min_duration is not None and dur < min_duration:
                        continue

                    audio_path = entry["audio_filepath"]
                    h = _stable_hash(audio_path)

                    variants = cache_index.get(h)
                    if variants:
                        entry["cache_filepaths"] = variants
                    else:
                        missing.add(audio_path)
                        continue

                    self.entries.append(entry)

        if missing:
            logger.warning(
                f"{len(missing)} unique audio files not found in cache "
                f"(skipped). First 5: {list(missing)[:5]}"
            )

        # Log variant-count distribution so it's easy to verify the cache.
        counts = Counter(len(e["cache_filepaths"]) for e in self.entries)
        logger.info(f"Variant counts per sample: { {k: v for k, v in sorted(counts.items())} }")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        """Return a single sample as a dict (collated later by ``collate_fn``)."""
        entry = self.entries[idx]
        # Randomly pick one variant; always variant 0 outside training mode
        # (LatentAugment already gates itself on self.training).
        cache_path = random.choice(entry["cache_filepaths"])
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)

        encoded = cached["encoded"].float()
        encoded_len = cached["encoded_len"]

        text = entry.get("text", "")
        token_ids = self.tokenizer.text_to_ids(text)
        transcript = torch.tensor(token_ids, dtype=torch.long)
        transcript_len = torch.tensor(len(token_ids), dtype=torch.long)

        return {
            "encoded": encoded,
            "encoded_len": encoded_len,
            "transcript": transcript,
            "transcript_len": transcript_len,
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad and stack samples into a batch."""
        encoded_list = [item["encoded"] for item in batch]
        encoded_lens = torch.stack([item["encoded_len"] for item in batch])
        transcript_list = [item["transcript"] for item in batch]
        transcript_lens = torch.stack([item["transcript_len"] for item in batch])

        encoded_list = [e.transpose(0, 1) for e in encoded_list]
        encoded_padded = pad_sequence(encoded_list, batch_first=True, padding_value=0.0)
        encoded_padded = encoded_padded.transpose(1, 2)
        transcript_padded = pad_sequence(transcript_list, batch_first=True, padding_value=0)

        return encoded_padded, encoded_lens, transcript_padded, transcript_lens


def build_cached_dataloader(
    manifest_filepaths: list[str],
    cache_dir: str | Path,
    tokenizer,
    *,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 2,
    max_duration: Optional[float] = None,
    min_duration: Optional[float] = None,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
) -> DataLoader:
    """Convenience factory that builds a ready-to-use DataLoader.

    Parameters mirror ``train_ds`` in ``training_config.yaml`` so the
    training script can forward them directly.
    """
    dataset = CachedEncoderDataset(
        manifest_filepaths=manifest_filepaths,
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        max_duration=max_duration,
        min_duration=min_duration,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=CachedEncoderDataset.collate_fn,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers and num_workers > 0,
    )
