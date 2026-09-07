# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fine-tune a speech-to-text model using cached encoder activations.

Instead of running the full pipeline (preprocessor → encoder → decoder →
joint) every training step, this script loads precomputed encoder outputs
from disk (produced by ``precompute_encoder_cache.py``).  A ``LatentAugment``
module replaces audio-level augmentations (SpecAugment, white noise, etc.)
with stochastic perturbations in the encoder feature space.

Validation and test still run the full pipeline so metrics remain comparable
to standard training.

Supports two anti-forgetting strategies (configure in YAML):

- **L2-SP** (``model.l2sp.enabled: true``): loss-level penalty toward
  pretrained weights (Xuhong et al., 2018).
- **AdamSPD** (``optim._target_: nemo.collections.asr.optim.adam_spd.AdamSPD``):
  Selective Projection Decay inside the optimizer step, which decays weights
  toward their pretrained values only when the gradient would push them
  further away. See ``nemo/collections/asr/optim/adam_spd.py``.

The encoder (608M params, ~97% of the model) is frozen and never executed
during training, cutting per-step compute to only the decoder + joint (18M).

Prerequisites:
  1. Run ``precompute_encoder_cache.py`` to generate the cache.
  2. Set ``encoder_cache_dir`` in the config to the cache directory.

Usage::

    python perso/speech_to_text_finetune_cached.py \\
        --config-path=../../perso --config-name=training_config

This script is a drop-in replacement for ``speech_to_text_finetune.py`` when
``use_cached_encoder: true`` and ``encoder_cache_dir`` are set in the config.
When ``use_cached_encoder`` is false (or absent), it behaves identically to
the original script.
"""
import contextlib
import math
import types
import time

# PyTorch 2.6+ defaults torch.load to weights_only=True, but NeMo
# checkpoints contain OmegaConf objects which are not in the default
# allowlist. Allowlisting them here avoids UnpicklingError on checkpoint
# save/resume while keeping weights_only=True everywhere else.
import torch
from omegaconf import OmegaConf

# PyTorch 2.6+ defaults torch.load to weights_only=True, but NeMo
# checkpoints contain OmegaConf objects, typing hints, and other
# non-standard globals. Rather than allowlisting them one by one,
# we override the default to weights_only=False. This is safe here
# because we only load our own trusted NeMo checkpoints.
_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: _original_torch_load(
    *args, **{**kwargs, "weights_only": kwargs.get("weights_only", False)}
)

import lightning.pytorch as pl

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.utils.asr_batching import get_duration_cost_batch_sampler
from nemo.collections.asr.parts.utils.config_audit import (
    assert_all_consumed,
    assert_effective_values,
    track_reads,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils
from nemo.utils.exp_manager import exp_manager
from nemo.utils.get_rank import is_global_rank_zero
from nemo.utils.trainer_utils import resolve_trainer_cfg


def get_base_model(trainer, cfg):
    """
    Returns the base model to be fine-tuned.
    Currently supports two types of initializations:
    1) `init_from_nemo_model`, and
    2) `init_from_pretrained_model`.
    Args:
        trainer: PyTorch Lightning Trainer
        cfg: config
    Returns:
        asr_model: ASRModel instance
    """
    asr_model = None
    nemo_model_path = cfg.get('init_from_nemo_model', None)
    pretrained_name = cfg.get('init_from_pretrained_model', None)
    if nemo_model_path is not None and pretrained_name is not None:
        raise ValueError("Only pass `init_from_nemo_model` or `init_from_pretrained_model` but not both")
    elif nemo_model_path is None and pretrained_name is None:
        raise ValueError(
            "Both `init_from_nemo_model` and `init_from_pretrained_model cannot be None, should pass atleast one of them"
        )
    elif nemo_model_path is not None:
        asr_model = ASRModel.restore_from(restore_path=nemo_model_path)
    elif pretrained_name is not None:
        # Due to potential first time download of the model on the cluster, we need to make sure that only one
        # rank downloads the model and the others wait for the download to finish.
        num_ranks = trainer.num_devices * trainer.num_devices

        if num_ranks > 1 and is_global_rank_zero():
            asr_model = ASRModel.from_pretrained(model_name=pretrained_name)
        else:
            # restore model from cached model dir
            asr_model = ASRModel.from_pretrained(model_name=pretrained_name)

    asr_model.set_trainer(trainer)
    return asr_model


def check_vocabulary(asr_model, cfg):
    """
    Checks if the decoder and vocabulary of the model needs to be updated.
    If either of them needs to be updated, it updates them and returns the updated model.
    else vocabulary will be reused from the pre-trained model.
    Args:
        asr_model: ASRModel instance
        cfg: config
    Returns:
        asr_model: ASRModel instance with updated decoder and vocabulary
    """
    if hasattr(cfg.model.tokenizer, 'update_tokenizer') and cfg.model.tokenizer.update_tokenizer:
        if hasattr(cfg.model.char_labels, 'update_labels') and cfg.model.char_labels.update_labels:
            raise ValueError(
                "Both `model.tokenizer.update_tokenizer` and `model.char_labels.update_labels` cannot be passed together"
            )
        else:
            asr_model = update_tokenizer(asr_model, cfg.model.tokenizer.dir, cfg.model.tokenizer.type)
    elif hasattr(cfg.model, 'char_labels') and cfg.model.char_labels.update_labels:
        asr_model.change_vocabulary(new_vocabulary=cfg.model.char_labels.labels)
        logging.warning("The vocabulary of the model has been updated with provided char labels.")
    else:
        logging.info("Reusing the vocabulary from the pre-trained model.")

    return asr_model


def update_tokenizer(asr_model, tokenizer_dir, tokenizer_type):
    """
    Updates the tokenizer of the model and also reinitializes the decoder if the vocabulary size
    of the new tokenizer differs from that of the loaded model.
    Args:
        asr_model: ASRModel instance
        tokenizer_dir: tokenizer directory
        tokenizer_type: tokenizer type
    Returns:
        asr_model: ASRModel instance with updated tokenizer and decoder
    """
    vocab_size = asr_model.tokenizer.vocab_size
    decoder = asr_model.decoder.state_dict()
    if hasattr(asr_model, 'joint'):
        joint_state = asr_model.joint.state_dict()
    else:
        joint_state = None

    if tokenizer_dir is None:
        raise ValueError("dir must be specified if update_tokenizer is True")
    logging.info("Using the tokenizer provided through config")
    asr_model.change_vocabulary(new_tokenizer_dir=tokenizer_dir, new_tokenizer_type=tokenizer_type)
    if asr_model.tokenizer.vocab_size != vocab_size:
        logging.warning(
            "The vocabulary size of the new tokenizer differs from that of the loaded model. As a result, finetuning will proceed with the new vocabulary, and the decoder will be reinitialized."
        )
    else:
        asr_model.decoder.load_state_dict(decoder)
        if joint_state is not None:
            asr_model.joint.load_state_dict(joint_state)

    return asr_model


def apply_cost_batch_sampler(asr_model, train_cfg):
    """Swap the fixed-size training batches for cost-based variable-size ones.

    Enabled by ``model.train_ds.cost_batching.enabled``. NeMo builds the train
    dataloader with a fixed ``batch_size``, which has to be sized for the
    longest clip the manifest can produce even though the RNNT/TDT joint costs
    roughly the square of the clip duration. This rebuilds the dataloader around
    :class:`DurationCostBatchSampler`, which packs each batch up to a memory
    budget instead, so short clips travel in large batches and the longest ones
    go alone.

    Done here rather than in ``_setup_dataloader_from_config`` so the NeMo
    dataset build, including the ``augmentor`` block, is left untouched: only
    the batching changes.

    Args:
        asr_model: model whose ``_train_dl`` was just built.
        train_cfg: the ``model.train_ds`` config block.
    """
    cost_cfg = train_cfg.get('cost_batching', None)
    if not cost_cfg or not cost_cfg.get('enabled', False):
        return

    dataloader = getattr(asr_model, '_train_dl', None)
    if dataloader is None:
        logging.warning("cost_batching is enabled but there is no training dataloader to rebuild.")
        return

    dataset = dataloader.dataset
    if isinstance(dataset, torch.utils.data.IterableDataset):
        raise ValueError(
            "cost_batching needs a map-style dataset (it draws batches by index), but the train "
            f"dataset is {type(dataset).__name__}. It is incompatible with tarred, concat and "
            "lhotse datasets."
        )

    sampler = get_duration_cost_batch_sampler(asr_model, dataset, train_cfg)
    num_workers = train_cfg.get('num_workers', 0)
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = train_cfg.get('persistent_workers', False)
        loader_kwargs['prefetch_factor'] = train_cfg.get('prefetch_factor', 2)

    asr_model._train_dl = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        collate_fn=dataloader.collate_fn,
        num_workers=num_workers,
        pin_memory=train_cfg.get('pin_memory', False),
        **loader_kwargs,
    )

    stats = sampler.stats()
    logging.info(
        f"Cost batching on: {stats['batches']} batches/epoch over {stats['samples']} clips "
        f"(was {math.ceil(stats['samples'] / max(train_cfg.get('batch_size', 1) or 1, 1))} at a fixed "
        f"batch_size of {train_cfg.get('batch_size')}). "
        f"Clips per batch {stats['clips_per_batch_min']}-{stats['clips_per_batch_max']} "
        f"(mean {stats['clips_per_batch_mean']:.1f}), audio seconds per batch "
        f"mean {stats['audio_seconds_per_batch_mean']:.0f} max {stats['audio_seconds_per_batch_max']:.0f}, "
        f"cost max {stats['cost_max']:.0f} of a {sampler.budget:.0f} budget."
    )


def setup_dataloaders(asr_model, cfg):
    """
    Sets up the training, validation and test dataloaders for the model.
    Args:
        asr_model: ASRModel instance
        cfg: config
    Returns:
        asr_model: ASRModel instance with updated dataloaders
    """
    cfg = model_utils.convert_model_config_to_dict_config(cfg)
    asr_model.setup_training_data(cfg.model.train_ds)
    apply_cost_batch_sampler(asr_model, cfg.model.train_ds)
    asr_model.setup_multiple_validation_data(cfg.model.validation_ds)
    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        asr_model.setup_multiple_test_data(cfg.model.test_ds)

    return asr_model


# ======================================================================
# Cached-encoder training support
# ======================================================================

def _compute_l2sp_penalty(
    model: "ASRModel",
    pretrained_params: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute the L2-SP penalty: sum of ||theta - theta_pretrained||^2.

    Only considers parameters that are both trainable and have a saved
    pretrained snapshot (i.e. decoder + joint weights).

    Parameters
    ----------
    model : ASRModel
        The model being fine-tuned.
    pretrained_params : dict[str, torch.Tensor]
        Frozen copy of the pretrained weights, keyed by parameter name.

    Returns
    -------
    torch.Tensor
        Scalar penalty (sum of squared differences).
    """
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if param.requires_grad and name in pretrained_params:
            penalty = penalty + ((param - pretrained_params[name]) ** 2).sum()
    return penalty


def _make_cached_training_step(latent_augment, cut_layer: int = -1, run_partial=None):
    """Create a replacement ``training_step`` that skips (or partially runs) the encoder.

    The returned function receives ``(encoded, encoded_len, transcript,
    transcript_len)`` from the cached dataloader instead of raw audio,
    applies latent augmentation, optionally runs ``encoder.layers[cut_layer+1:]``
    on the cached intermediate features, then runs decoder → joint → loss
    as usual.

    Parameters
    ----------
    latent_augment : LatentAugment
        Module that perturbs cached encoder features during training.
        Attached to the model so it follows ``.train()`` / ``.eval()``.
    cut_layer : int
        ``-1`` means the cache holds final encoder outputs (skip encoder
        entirely). ``>= 0`` means the cache holds activations after layer
        ``cut_layer``; ``layers[cut_layer+1:]`` + ``out_proj`` will run
        live each step.
    run_partial : callable or None
        ``run_encoder_from_layer`` (from perso/partial_encoder.py).
        Required when ``cut_layer >= 0``.
    """
    from nemo.core import adapter_mixins  # for AccessMixin

    # Try to import AccessMixin — the exact location varies across NeMo
    # versions, so we fall back gracefully.
    try:
        from nemo.core.classes.mixins import AccessMixin
    except ImportError:
        from nemo.core import AccessMixin

    def cached_training_step(self, batch, batch_nb):
        """training_step that uses cached encoder outputs instead of audio.

        Batch format: (encoded, encoded_len, transcript, transcript_len)
        — produced by CachedEncoderDataset.collate_fn.
        """
        # Reset access registry (same as original training_step)
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        encoded, encoded_len, transcript, transcript_len = batch

        # Apply latent augmentation (replaces SpecAugment + audio augmentors).
        # Acts on cached features regardless of where the cut sits — when
        # cut_layer >= 0 this perturbs intermediate activations before the
        # remaining live encoder layers (acts like SpecAugment in latent space).
        encoded = latent_augment(encoded, encoded_len)

        # If the cache stops at an intermediate layer, run the remaining
        # encoder layers + out_proj live on the cached features.
        if cut_layer >= 0:
            encoded, encoded_len = run_partial(
                self.encoder,
                cached_bdt=encoded,
                cached_len=encoded_len,
                start_layer=cut_layer + 1,
            )

        # --- From here on, identical to the original training_step ---
        # Decoder forward
        decoder, target_length, states = self.decoder(
            targets=transcript, target_length=transcript_len
        )

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        # Non-fused path (standard joint + loss)
        if not self.joint.fuse_loss_wer:
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            loss_value = self.loss(
                log_probs=joint,
                targets=transcript,
                input_lengths=encoded_len,
                target_lengths=target_length,
            )
            loss_value = self.add_auxiliary_losses(loss_value)

            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(
                    self.trainer.global_step, dtype=torch.float32
                ),
            }

            if (sample_id + 1) % log_every_n_steps == 0:
                self.wer.update(
                    predictions=encoded,
                    predictions_lengths=encoded_len,
                    targets=transcript,
                    targets_lengths=transcript_len,
                )
                _, scores, words = self.wer.compute()
                self.wer.reset()
                tensorboard_logs.update(
                    {'training_batch_wer': scores.float() / words}
                )
        else:
            # Fused joint-loss-WER path
            compute_wer = (sample_id + 1) % log_every_n_steps == 0
            loss_value, wer, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=transcript_len,
                compute_wer=compute_wer,
            )
            loss_value = self.add_auxiliary_losses(loss_value)

            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(
                    self.trainer.global_step, dtype=torch.float32
                ),
            }

            if compute_wer:
                tensorboard_logs.update({'training_batch_wer': wer})

        # Log to tensorboard
        self.log_dict({f"train/{k}": v for k, v in tensorboard_logs.items()}, prog_bar=True)
        return {'loss': loss_value}

    return cached_training_step


def setup_cached_training(asr_model, cfg):
    """Replace training dataloader and training_step for cached-encoder mode.

    This function:
    1. Builds a ``CachedEncoderDataset`` + DataLoader from the cache manifest.
    2. Creates a ``LatentAugment`` module and registers it on the model
       (so it moves to the correct device and follows train/eval mode).
    3. Monkey-patches ``training_step`` to skip the encoder forward pass.

    Validation and test dataloaders are left unchanged — they still run the
    full preprocessor → encoder → decoder → joint pipeline.

    Parameters
    ----------
    asr_model : ASRModel
        The model being fine-tuned.  Must already have its tokenizer set up.
    cfg : OmegaConf
        Full Hydra config.  Expected keys:

        - ``encoder_cache_dir``  — path to cache directory containing
          ``.pt`` files (produced by ``precompute_encoder_cache.py``).
        - ``model.latent_augment``  — sub-config for ``LatentAugment``.
        - ``model.train_ds``  — used for ``batch_size``, ``num_workers``, etc.
    """
    import importlib.util
    import sys
    from pathlib import Path

    # Import custom modules directly from file paths to avoid nemo package path issues
    _repo_root = Path(__file__).resolve().parents[2]

    def _import_from_file(module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod

    _ced = _import_from_file(
        "nemo.collections.asr.data.cached_encoder_dataset",
        _repo_root / "nemo" / "collections" / "asr" / "data" / "cached_encoder_dataset.py",
    )
    _la = _import_from_file(
        "nemo.collections.asr.modules.latent_augment",
        _repo_root / "nemo" / "collections" / "asr" / "modules" / "latent_augment.py",
    )
    _pe = _import_from_file(
        "perso.partial_encoder",
        _repo_root / "perso" / "partial_encoder.py",
    )
    build_cached_dataloader = _ced.build_cached_dataloader
    LatentAugment = _la.LatentAugment
    run_encoder_from_layer = _pe.run_encoder_from_layer

    cache_dir = Path(cfg.get("encoder_cache_dir"))
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Cache directory not found at {cache_dir}. "
            "Run precompute_encoder_cache.py first."
        )

    # --- Latent augmentation ---
    # If cache_stats.json exists (produced by precompute_encoder_cache.py
    # --compute-stats), load per-dimension noise scale so Gaussian noise is
    # proportional to each dimension's empirical variability under audio
    # augmentations.  Falls back to uniform gaussian_noise_scale otherwise.
    import json as _json

    la_cfg = cfg.model.get("latent_augment", {})
    la_enabled = la_cfg.get("enabled", False)

    if not la_enabled:
        class _NoOpAugment(torch.nn.Module):
            def forward(self, x, x_len):
                return x
            def __repr__(self):
                return "LatentAugment(enabled=False)"

        latent_augment = _NoOpAugment()
        asr_model.latent_augment = latent_augment
        logging.info("Latent augmentation disabled (enabled: false in config).")
    else:
        per_dim_noise_scale = None
        protected_dims = None
        stats_path = cache_dir / "cache_stats.json"
        stats = None

        if stats_path.exists():
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = _json.load(f)

        if stats is not None and la_cfg.get("use_per_dim_noise", True):
            per_dim_noise_scale = torch.tensor(
                stats["per_dim_std"], dtype=torch.float32
            )
            logging.info(
                f"Per-dimension latent noise active "
                f"({per_dim_noise_scale.shape[0]} dims, "
                f"multiplier={la_cfg.get('noise_scale_multiplier', 1.0)})"
            )
        elif stats is not None:
            logging.info(
                "cache_stats.json found but use_per_dim_noise is false — "
                "using uniform gaussian_noise_scale."
            )
        else:
            logging.info(
                "No cache_stats.json found — "
                "using uniform gaussian_noise_scale as fallback."
            )

        if stats is not None and "protected_dims" in stats:
            protected_dims = torch.tensor(stats["protected_dims"], dtype=torch.bool)
            logging.info(
                f"Outlier protection active: "
                f"{int(protected_dims.sum())}/{protected_dims.numel()} dims "
                f"exempt from noise + feature_dropout."
            )

        latent_augment = LatentAugment(
            gaussian_noise_scale=la_cfg.get("gaussian_noise_scale", 0.1),
            feature_dropout=la_cfg.get("feature_dropout", 0.1),
            time_mask_num=la_cfg.get("time_mask_num", 5),
            time_mask_width=la_cfg.get("time_mask_width", 25),
            per_dim_noise_scale=per_dim_noise_scale,
            noise_scale_multiplier=la_cfg.get("noise_scale_multiplier", 1.0),
            protected_dims=protected_dims,
        )
        # Register as a submodule so it follows .to(device) and .train()/.eval()
        asr_model.latent_augment = latent_augment
        logging.info(f"Latent augmentation: {latent_augment}")

    # --- Cached training dataloader ---
    # Read manifest_filepath from the training config (preserves repetitions
    # for upsampling), then resolve each sample to its cached .pt file.
    train_ds_cfg = cfg.model.get("train_ds", {})
    manifest_filepaths = train_ds_cfg.get("manifest_filepath", [])
    if isinstance(manifest_filepaths, str):
        manifest_filepaths = [manifest_filepaths]
    manifest_filepaths = list(manifest_filepaths)
    logging.info(
        f"Reading {len(manifest_filepaths)} training manifest(s) "
        f"(with repetitions) for cached training."
    )
    cached_dl = build_cached_dataloader(
        manifest_filepaths=manifest_filepaths,
        cache_dir=str(cache_dir),
        tokenizer=asr_model.tokenizer,
        batch_size=train_ds_cfg.get("batch_size", 1),
        shuffle=train_ds_cfg.get("shuffle", True),
        num_workers=train_ds_cfg.get("num_workers", 2),
        max_duration=train_ds_cfg.get("max_duration", None),
        min_duration=train_ds_cfg.get("min_duration", None),
        pin_memory=train_ds_cfg.get("pin_memory", True),
        prefetch_factor=train_ds_cfg.get("prefetch_factor", 4),
        persistent_workers=train_ds_cfg.get("persistent_workers", True),
    )
    # Override the training dataloader.  NeMo stores it in _train_dl and
    # Lightning calls model.train_dataloader() which returns it.
    asr_model._train_dl = cached_dl
    logging.info(
        f"Cached training dataloader: {len(cached_dl.dataset)} samples "
        f"from {len(manifest_filepaths)} manifest(s), cache dir: {cache_dir}"
    )

    # --- Verify the cache's cut_layer matches the configured freeze setting ---
    cache_cut_layer: int = int(getattr(cached_dl.dataset, "cut_layer", -1))
    freeze_cfg = cfg.model.get("freeze", {})
    except_last_n = int(freeze_cfg.get("encoder_except_last_n", 0) or 0)
    n_total_layers = len(asr_model.encoder.layers)
    expected_cut_layer = (
        n_total_layers - except_last_n - 1 if except_last_n > 0 else -1
    )
    if cache_cut_layer != expected_cut_layer:
        raise RuntimeError(
            f"Cache/config mismatch: cache at {cache_dir} was built with "
            f"cut_layer={cache_cut_layer} but config requests cut_layer="
            f"{expected_cut_layer} (encoder_except_last_n={except_last_n}, "
            f"n_layers={n_total_layers}). Regenerate the cache with the "
            f"matching --cut-layer or use a different cache directory."
        )
    if cache_cut_layer >= 0:
        logging.info(
            f"Partial-cache training: cache stops at layer {cache_cut_layer}, "
            f"layers[{cache_cut_layer + 1}:{n_total_layers}] + out_proj run live."
        )

    # --- Monkey-patch training_step ---
    new_step = _make_cached_training_step(
        latent_augment,
        cut_layer=cache_cut_layer,
        run_partial=run_encoder_from_layer,
    )
    asr_model.training_step = types.MethodType(new_step, asr_model)
    logging.info(
        "training_step patched "
        f"({'partial encoder' if cache_cut_layer >= 0 else 'encoder skipped'})."
    )


# ======================================================================


def _maybe_rebuild_encoder_cache(cfg) -> None:
    """Check encoder cache freshness and rebuild if encoder_cache.auto_rebuild is true.

    Reads all required parameters from the config:
      - encoder_cache.auto_rebuild      (bool, default false)
      - encoder_cache.compute_stats     (bool, default false)
      - encoder_cache_dir               (path)
      - init_from_pretrained_model / init_from_nemo_model
      - model.train_ds.manifest_filepath
    """
    import subprocess
    import sys
    from pathlib import Path

    ec_cfg = cfg.get("encoder_cache", {}) or {}
    auto_rebuild = bool(ec_cfg.get("auto_rebuild", False))
    compute_stats = bool(ec_cfg.get("compute_stats", False))

    if not auto_rebuild:
        return

    cache_dir = cfg.get("encoder_cache_dir")
    if not cache_dir:
        raise ValueError("encoder_cache.auto_rebuild is true but encoder_cache_dir is not set.")

    model = cfg.get("init_from_pretrained_model") or cfg.get("init_from_nemo_model")
    if not model:
        raise ValueError("encoder_cache.auto_rebuild requires init_from_pretrained_model or init_from_nemo_model.")

    train_ds_cfg = cfg.model.get("train_ds", {})
    manifest_filepaths = train_ds_cfg.get("manifest_filepath", [])
    if isinstance(manifest_filepaths, str):
        manifest_filepaths = [manifest_filepaths]
    manifest_filepaths = list(manifest_filepaths)
    if not manifest_filepaths:
        raise ValueError("encoder_cache.auto_rebuild requires model.train_ds.manifest_filepath to be set.")

    _repo_root = Path(__file__).resolve().parents[2]
    config_path = _repo_root / "perso" / "training_config.yaml"
    precompute_script = _repo_root / "perso" / "precompute_encoder_cache.py"

    base_args = [
        sys.executable, str(precompute_script),
        "--model", str(model),
        "--cache-dir", str(cache_dir),
        "--training-config", str(config_path),
        "--yes",
    ]
    for m in manifest_filepaths:
        base_args += ["--manifests", str(m)]
    if compute_stats:
        base_args.append("--compute-stats")

    logging.info("Checking encoder cache freshness...")
    check_result = subprocess.run(base_args + ["--check-only"], capture_output=True)
    if check_result.returncode == 0:
        logging.info("Encoder cache is up to date — skipping precompute.")
        return

    logging.info("Encoder cache is stale or missing — rebuilding...")
    result = subprocess.run(base_args)
    if result.returncode != 0:
        raise RuntimeError(
            f"precompute_encoder_cache.py failed with exit code {result.returncode}. "
            "Fix the cache issue before training."
        )
    logging.info("Encoder cache rebuild complete.")


class _FrozenEvalCallback(pl.Callback):
    """Pins frozen submodules to eval mode for the whole run.

    Freezing only sets ``requires_grad=False`` and ``.eval()`` once, but
    Lightning calls ``model.train()`` at the start of every training epoch
    and again when leaving validation, which re-enables dropout and, more
    importantly, BatchNorm running-stat updates on the frozen modules
    (Conformer conv modules use BatchNorm). In standard (non-cached) mode
    the frozen encoder layers still run forward on every step, so their BN
    stats would silently drift away from the pretrained values over a long
    run. This callback re-applies ``.eval()`` before each training batch.

    The re-apply is O(1) in steady state: it short-circuits on the first
    module's ``.training`` flag, so the full recursive ``.eval()`` only
    happens right after Lightning flipped the model back to train mode.
    """

    def __init__(self, modules):
        self.modules = [m for m in modules if m is not None]

    def _pin(self):
        for m in self.modules:
            m.eval()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.modules and self.modules[0].training:
            self._pin()

    def on_validation_start(self, trainer, pl_module):
        # Validation puts the whole model in eval anyway; this is only to
        # keep the flags consistent if a hook ordering ever changes.
        self._pin()


class _NormalisedWERCallback(pl.Callback):
    """Logs a punctuation- and case-insensitive WER next to the raw one.

    NeMo scores ``reference.split()`` against ``hypothesis.split()``, so
    formatting counts as recognition error and, because ``split()`` keeps
    punctuation glued to its word, one comma costs a whole word. On the 1.1.0
    run that was worth 2.10 WER points, 13% relative. See
    nemo/collections/asr/metrics/wer_normalised.py.

    Both numbers get logged. The raw one stays the checkpoint monitor so the
    curves remain comparable with earlier runs; this one says how much of the
    error is actually about words.

    Implementation note: the model's WER metric is update/compute/reset per
    batch by both RNNT validation paths, so the normalised counts cannot live
    in a torchmetrics state (reset would wipe them before anything read them).
    They accumulate on plain attributes which this callback drains after every
    batch, which is also what keeps per-dataloader attribution correct.
    """

    def __init__(self):
        self.validation_names = []
        self.totals = {}

    @staticmethod
    def _metric(pl_module):
        return getattr(pl_module, "wer", None)

    def on_validation_start(self, trainer, pl_module):
        metric = self._metric(pl_module)
        if metric is None:
            return
        metric.track_normalised = True
        metric.normalised_scores = 0
        metric.normalised_words = 0
        self.totals = {}
        # Taken from the model, not the config: NeMo derives these from
        # validation_ds.ds_item itself (nemo/utils/model_utils.py), and the raw
        # metric names come from the same list. Re-deriving them here would be
        # a second copy of that rule, free to drift from the one that decides
        # what the raw curves are called.
        self.validation_names = list(getattr(pl_module, "_validation_names", None) or [""])

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        metric = self._metric(pl_module)
        if metric is None:
            return
        scores, words = metric.normalised_scores, metric.normalised_words
        metric.normalised_scores = 0
        metric.normalised_words = 0
        total = self.totals.setdefault(dataloader_idx, [0, 0])
        total[0] += scores
        total[1] += words

    def on_validation_end(self, trainer, pl_module):
        metric = self._metric(pl_module)
        if metric is not None:
            metric.track_normalised = False
        if not self.totals:
            return

        logged = {}
        for dataloader_idx, (scores, words) in sorted(self.totals.items()):
            if words == 0 or dataloader_idx >= len(self.validation_names):
                continue
            name = f"{self.validation_names[dataloader_idx]}val_wer_norm"
            value = torch.tensor(scores / words, dtype=torch.float32)
            # callback_metrics, not self.log: this hook forbids self.log, and
            # the macro callback reads from here anyway.
            trainer.callback_metrics[name] = value
            logged[name] = value.item()

        if logged and trainer.logger is not None and not trainer.sanity_checking:
            trainer.logger.log_metrics(logged, step=trainer.global_step)


class _MacroMetricCallback(pl.Callback):
    """Averages already-logged validation metrics into macro metrics.

    Configured via the top-level ``macro_metrics`` config section, e.g.::

        macro_metrics:
          - name: "fleurs_fr_en_macro_val_wer"
            sources: ["fleurs_frval_wer", "fleurs_enval_wer"]

    Uses ``on_validation_end`` because the per-dataloader metrics are logged
    in the LightningModule's ``on_validation_epoch_end`` (NeMo's
    ``multi_validation_epoch_end``), which runs *after* callback
    ``on_validation_epoch_end`` hooks but *before* ``on_validation_end``.
    The result is written straight into ``trainer.callback_metrics``
    (``self.log`` is not allowed in this hook) so that ModelCheckpoint,
    which also triggers in ``on_validation_end``, can use it as ``monitor``.
    This callback must therefore sit *before* the checkpoint callback in
    ``trainer.callbacks``; main() inserts it at position 0 while exp_manager
    appends the checkpoint callback at the end.
    """

    def __init__(self, specs):
        self.specs = specs

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        for spec in self.specs:
            missing = [s for s in spec["sources"] if s not in metrics]
            if missing:
                # Expected during sanity checking (limited batches); a real
                # miss later usually means a validation set was renamed.
                logging.warning(
                    f"Macro metric '{spec['name']}': sources {missing} not in "
                    f"callback_metrics, skipping this round."
                )
                continue
            value = torch.stack(
                [torch.as_tensor(metrics[s], dtype=torch.float32) for s in spec["sources"]]
            ).mean()
            metrics[spec["name"]] = value
            if trainer.logger is not None and not trainer.sanity_checking:
                trainer.logger.log_metrics(
                    {spec["name"]: value.item()}, step=trainer.global_step
                )


@hydra_runner(config_path="conf/asr_finetune", config_name="speech_to_text_finetune")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    audit_cfg = cfg.get("config_audit", {}) or {}
    audit_enabled = audit_cfg.get("enabled", True)
    read_tracker = (
        track_reads(cfg) if audit_enabled else contextlib.nullcontext(set())
    )
    with read_tracker as seen_keys:
        asr_model, trainer = _build(cfg)

    if audit_enabled:
        assert_all_consumed(cfg, seen_keys, audit_cfg.get("allow_unused", []))
        assert_effective_values(cfg, asr_model, trainer)

    trainer.fit(asr_model)


def _build(cfg):
    """Everything from raw config to a fit-ready model, minus trainer.fit.

    Split out of main() so the config audit can wrap exactly the window in
    which config keys are legitimately read: every read that is going to
    happen has happened by the time this returns, and nothing has trained yet,
    so a failed audit costs a restart rather than a run.
    """
    # Seed before anything builds a generator: data shuffling, dropout,
    # augmentation, and the initial state of any newly initialised head.
    seed = cfg.model.get("seed", None)
    if seed is not None:
        pl.seed_everything(int(seed), workers=True)
        logging.info(f"Seeded everything with {seed} (workers=True).")
    else:
        logging.warning("model.seed is unset, this run is not reproducible.")

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    # Macro validation metrics (e.g. checkpoint monitor averaged over
    # several languages). Must be inserted at the *front* of the callback
    # list so it runs before the checkpoint callback appended by exp_manager.
    macro_specs = cfg.get("macro_metrics", None)
    if macro_specs:
        macro_specs = OmegaConf.to_container(macro_specs, resolve=True)
        if cfg.get("normalised_wer", True):
            # Every macro gets a _norm twin over the _norm sources, rather than
            # making the YAML restate the source lists and drift out of sync.
            macro_specs = macro_specs + [
                {"name": f"{s['name']}_norm", "sources": [f"{src}_norm" for src in s["sources"]]}
                for s in macro_specs
            ]
        trainer.callbacks.insert(0, _MacroMetricCallback(macro_specs))
        logging.info(
            "Macro metrics active: "
            + ", ".join(f"{s['name']} <- mean({', '.join(s['sources'])})" for s in macro_specs)
        )

    # Normalised WER, logged next to the raw one. Inserted after the macro
    # callback so it ends up *before* it, because the macros average whatever
    # is already in callback_metrics and the _norm sources have to be there
    # first. Order in trainer.callbacks is the order the hooks run.
    if cfg.get("normalised_wer", True):
        trainer.callbacks.insert(0, _NormalisedWERCallback())
        logging.info("Normalised WER active, logged as <name>val_wer_norm next to each raw <name>val_wer.")

    if hasattr(cfg, 'init_from_ptl_ckpt') and cfg.init_from_ptl_ckpt is not None:
        raise NotImplementedError(
            "Currently for simplicity of single script for all model types, we only support `init_from_nemo_model` and `init_from_pretrained_model`"
        )

    asr_model = get_base_model(trainer, cfg)

    # --- Decoding strategy ---
    # This has to be an explicit call. from_pretrained/restore_from bring the
    # checkpoint's own model config with them, which includes its `decoding`
    # block, so a `model.decoding` section in our YAML would be silently
    # overwritten and a top-level one read by nobody. Keep the block at the
    # top level and apply it here, which is also what makes validation WER
    # comparable across runs regardless of what the checkpoint shipped with.
    decoding_cfg = cfg.get("decoding", None)
    if decoding_cfg is not None and hasattr(asr_model, "change_decoding_strategy"):
        inherited = asr_model.cfg.get("decoding", {}).get("strategy", "?")
        asr_model.change_decoding_strategy(decoding_cfg)
        logging.info(
            f"Decoding strategy set to '{decoding_cfg.get('strategy')}' from the top-level "
            f"`decoding` block (the checkpoint shipped with '{inherited}')."
        )
    else:
        logging.info("No top-level `decoding` block, keeping the checkpoint's own strategy.")

    # Freeze components according to config (defaults: freeze nothing).
    # Special handling for the encoder: if `freeze.encoder_except_last_n > 0`
    # we freeze layers[:L-N] + pre_encode + pos_enc but leave the last N
    # conformer layers + out_proj trainable.
    freeze_cfg = cfg.model.get("freeze", {})
    except_last_n = int(freeze_cfg.get("encoder_except_last_n", 0) or 0)

    frozen_modules = []  # pinned to eval by _FrozenEvalCallback (BN drift fix)

    if freeze_cfg.get("encoder", False) and hasattr(asr_model, "encoder"):
        encoder = asr_model.encoder
        if except_last_n <= 0:
            logging.info("Freezing encoder (full)")
            encoder.freeze()
            frozen_modules.append(encoder)
        else:
            n_total = len(encoder.layers)
            if except_last_n >= n_total:
                raise ValueError(
                    f"encoder_except_last_n={except_last_n} >= n_layers={n_total}; "
                    f"either reduce it or set freeze.encoder=false."
                )
            n_frozen = n_total - except_last_n
            for p in encoder.pre_encode.parameters():
                p.requires_grad_(False)
            for p in encoder.pos_enc.parameters():
                p.requires_grad_(False)
            for layer in encoder.layers[:n_frozen]:
                for p in layer.parameters():
                    p.requires_grad_(False)
                layer.eval()
            encoder.pre_encode.eval()
            encoder.pos_enc.eval()
            frozen_modules.append(encoder.pre_encode)
            frozen_modules.append(encoder.pos_enc)
            frozen_modules.extend(encoder.layers[:n_frozen])
            logging.info(
                f"Encoder partial freeze: layers[0:{n_frozen}] frozen, "
                f"layers[{n_frozen}:{n_total}] + out_proj trainable."
            )
    else:
        logging.info("Leaving encoder trainable")

    for component in ("decoder", "joint"):
        if freeze_cfg.get(component, False) and hasattr(asr_model, component):
            logging.info(f"Freezing {component}")
            getattr(asr_model, component).freeze()
            frozen_modules.append(getattr(asr_model, component))
        else:
            logging.info(f"Leaving {component} trainable")

    if frozen_modules:
        # Lightning re-enables train mode (dropout + BatchNorm stat updates)
        # on ALL submodules at each train epoch start and after validation,
        # even where requires_grad is False. Frozen modules still run forward
        # in non-cached mode, so without this their BN stats drift over a
        # long run. The callback re-pins them to eval before every batch.
        trainer.callbacks.append(_FrozenEvalCallback(frozen_modules))
        logging.info(
            f"_FrozenEvalCallback active on {len(frozen_modules)} frozen module(s)."
        )

    # Verify what's trainable
    trainable_params = sum(p.numel() for p in asr_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in asr_model.parameters())
    logging.info(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

    # Check vocabulary type and update if needed
    asr_model = check_vocabulary(asr_model, cfg)

    # --- Joint fused loss/WER sub-batch size ---
    # The RNNT/TDT joint materialises a [B, T, U, V] lattice (V = 8198 for
    # parakeet-tdt-0.6b-v3). With fuse_loss_wer the joint and the loss run on
    # sub-batches of `fused_batch_size` clips at a time. The pretrained
    # checkpoint ships fused_batch_size=4, so with batch_size=4 the whole batch
    # is one sub-batch and every clip is padded to the longest one in it: cost
    # becomes B * max(T) * max(U) * V. Lowering it to 1 removes that padding
    # amplification, so peak memory follows sum_i(T_i * U_i) instead, which is
    # what makes a batch containing one long clip affordable.
    joint_cfg = cfg.model.get("joint", None)
    fused_bs = None if joint_cfg is None else joint_cfg.get("fused_batch_size", None)
    if fused_bs is not None:
        if not hasattr(asr_model, "joint"):
            raise ValueError("model.joint.fused_batch_size is set but this model has no joint module.")
        fused_bs = int(fused_bs)
        if not getattr(asr_model.joint, "fuse_loss_wer", False):
            logging.warning(
                "model.joint.fused_batch_size is set but the joint has fuse_loss_wer=False, "
                "so the setting has no effect."
            )
        asr_model.joint.set_fused_batch_size(fused_bs)
        if "joint" in asr_model.cfg:
            asr_model.cfg.joint.fused_batch_size = fused_bs
        logging.info(f"Joint fused_batch_size set to {fused_bs}")

    # --- Decide between cached-encoder and standard training ---
    use_cached = cfg.get("use_cached_encoder", False)

    if use_cached:
        logging.info("=== CACHED ENCODER MODE ===")
        logging.info("Training dataloader will read precomputed encoder outputs.")
        logging.info("Validation/test still use the full pipeline.")

        # --- Auto-rebuild encoder cache if requested ---
        _maybe_rebuild_encoder_cache(cfg)

        # Standard dataloaders for val/test (full pipeline)
        # We still call setup_dataloaders for val/test, then override train.
        asr_model = setup_dataloaders(asr_model, cfg)

        # Override training dataloader + training_step
        setup_cached_training(asr_model, cfg)

        # SpecAugment is not needed — latent augmentation replaces it.
        # Disable it so validation doesn't accidentally apply it either
        # (it's normally disabled in eval mode, but be explicit).
        asr_model.spec_augmentation = None
        logging.info("SpecAugment disabled (replaced by latent augmentation).")
    else:
        logging.info("=== STANDARD TRAINING MODE ===")
        # Setup Data (audio-based)
        asr_model = setup_dataloaders(asr_model, cfg)

        # Setup SpecAug
        if hasattr(cfg.model, 'spec_augment') and cfg.model.spec_augment is not None:
            asr_model.spec_augment = ASRModel.from_config_dict(cfg.model.spec_augment)

    # --- L2-SP regularization toward pretrained weights ---
    # Must be set up *after* freezing (so we only snapshot trainable params)
    # but *before* optimizer setup (so the penalty is active from step 1).
    l2sp_cfg = cfg.model.get("l2sp", {})
    l2sp_enabled = l2sp_cfg.get("enabled", False)
    l2sp_lambda = l2sp_cfg.get("lambda", 0.01)

    if l2sp_enabled:
        # Snapshot the pretrained weights for all trainable parameters.
        # These are cloned and detached so they stay constant throughout
        # training. Memory cost is negligible (~72 MB for 18M params in fp32).
        pretrained_params: dict[str, torch.Tensor] = {
            name: param.clone().detach()
            for name, param in asr_model.named_parameters()
            if param.requires_grad
        }
        logging.info(
            f"L2-SP enabled: lambda={l2sp_lambda}, "
            f"tracking {len(pretrained_params)} parameter tensors "
            f"({sum(p.numel() for p in pretrained_params.values()):,} values)"
        )

        # Wrap whatever training_step is currently installed (cached or
        # standard) to add the L2-SP penalty to the returned loss.
        _original_training_step = asr_model.training_step

        def _l2sp_training_step(self, batch, batch_nb):
            result = _original_training_step(batch, batch_nb)
            penalty = _compute_l2sp_penalty(self, pretrained_params)
            result["loss"] = result["loss"] + l2sp_lambda * penalty
            # Log the raw penalty so we can monitor drift from pretrained
            self.log("train/l2sp_penalty", penalty.detach(), prog_bar=False)
            return result

        asr_model.training_step = types.MethodType(_l2sp_training_step, asr_model)
        logging.info("training_step wrapped with L2-SP penalty.")
    else:
        logging.info("L2-SP disabled.")

    # Setup Optimizer
    asr_model.setup_optimization(cfg.model.optim)

    # NeMo overwrites sched.max_steps with trainer.max_steps whenever the
    # latter is set (modelPT.setup_optimization), so an explicit YAML value
    # would silently stop describing the anneal the moment the two differ.
    # They differ on purpose in exactly one design: an A/B arm that keeps the
    # reference run's cosine and truncates the run short, so the shared steps
    # follow the same LR trajectory (see config_audit.allow_truncated_anneal).
    # Restore the YAML's value on the live scheduler; the config audit then
    # verifies the live object. Safe post-construction because CosineAnnealing
    # reads max_steps per step and this config gives warmup as an absolute
    # step count (a warmup_ratio would already have been resolved against the
    # wrong horizon, so it is refused rather than silently mis-warmed).
    sched_cfg = cfg.model.optim.get("sched", None)
    explicit_max_steps = sched_cfg.get("max_steps", None) if sched_cfg is not None else None
    live_sched = (asr_model._scheduler or {}).get("scheduler", None)
    if explicit_max_steps and live_sched is not None and getattr(live_sched, "max_steps", None) not in (None, explicit_max_steps):
        if sched_cfg.get("warmup_ratio", None) is not None:
            raise ValueError(
                "sched.warmup_ratio was resolved against the trainer-derived horizon "
                f"({live_sched.max_steps}), so restoring max_steps={explicit_max_steps} would "
                "keep the wrong warmup. Use an absolute sched.warmup_steps instead."
            )
        logging.info(
            f"Restoring optim.sched.max_steps={explicit_max_steps} on the live scheduler "
            f"(NeMo had replaced it with trainer.max_steps={live_sched.max_steps})."
        )
        live_sched.max_steps = explicit_max_steps

    # --- Selective Projection Decay (SPD) ---
    # If the optimizer is AdamSPD, register the pretrained weight snapshot
    # so the optimizer can selectively decay toward pretrained values.
    # Must happen *after* optimizer creation and *after* freezing.
    from nemo.collections.asr.optim.adam_spd import AdamSPD, attach_pretrained_params
    if isinstance(asr_model._optimizer, AdamSPD):
        attach_pretrained_params(asr_model._optimizer, asr_model)
        n_params = sum(
            1 for p in asr_model.parameters() if p.requires_grad
        )
        n_values = sum(
            p.numel() for p in asr_model.parameters() if p.requires_grad
        )
        logging.info(
            f"AdamSPD: registered {n_params} pretrained param tensors "
            f"({n_values:,} values) for Selective Projection Decay "
            f"(weight_decay={asr_model._optimizer.defaults['weight_decay']})"
        )
    else:
        logging.info(
            f"Optimizer is {type(asr_model._optimizer).__name__}, "
            f"SPD not active."
        )

    return asr_model, trainer


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
