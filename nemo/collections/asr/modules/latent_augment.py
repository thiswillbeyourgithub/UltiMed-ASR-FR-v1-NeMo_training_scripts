"""Latent-space augmentation module for cached encoder activations.

When fine-tuning with precomputed encoder outputs, audio-level augmentations
(SpecAugment, white noise, gain, etc.) are no longer applicable because the
raw waveform is not available.  This module fills that gap by applying
stochastic perturbations directly to the encoder feature space:

  * Additive Gaussian noise  — analogous to white-noise injection on audio.
    Supports per-dimension noise scaling so dimensions with higher natural
    variability (measured empirically via ``precompute_encoder_cache.py
    --compute-stats``) receive proportionally more noise.
  * Feature dropout           — randomly zeros entire feature dimensions
                                across all time steps (like frequency masking).
  * Time masking              — zeros contiguous time segments, mirroring
                                SpecAugment's time-mask behaviour.

All perturbations are only active when ``self.training is True``; at
inference / validation time the module is a no-op.

Usage
-----
>>> aug = LatentAugment(gaussian_noise_scale=0.1, feature_dropout=0.1,
...                     time_mask_num=5, time_mask_width=25)
>>> encoded_out = aug(encoded, encoded_len)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LatentAugment(nn.Module):
    """Stochastic perturbation of cached encoder activations.

    Parameters
    ----------
    gaussian_noise_scale : float
        Std-dev of additive Gaussian noise (uniform fallback).  0 disables.
    feature_dropout : float
        Probability of zeroing an entire feature dimension for a given sample.
        Applied independently per (batch, feature_dim).  0 disables.
    time_mask_num : int
        Number of contiguous time masks to apply per sample.  0 disables.
    time_mask_width : int
        Maximum width (in encoder frames) of each time mask.
    per_dim_noise_scale : torch.Tensor, optional
        Per-dimension std-dev vector of shape ``(D,)`` measured empirically
        from augmented encoder passes (produced by
        ``precompute_encoder_cache.py --compute-stats``).  When provided,
        Gaussian noise is scaled per-dimension instead of uniformly —
        dimensions that vary more under real audio augmentations receive
        proportionally more noise.
    noise_scale_multiplier : float
        Global multiplier applied on top of ``per_dim_noise_scale``.
        Lets the user scale the empirical std up or down without
        recomputing stats.  Ignored when ``per_dim_noise_scale`` is None.
    protected_dims : torch.Tensor, optional
        Boolean mask of shape ``(D,)`` flagging "outlier" encoder dims that
        carry disproportionate signal (large persistent magnitude or heavy
        tails). When provided, these dims receive **zero** Gaussian noise
        and are never zeroed by feature_dropout. Computed by
        ``precompute_encoder_cache.py --compute-stats``.
    """

    def __init__(
        self,
        *,
        gaussian_noise_scale: float = 0.0,
        feature_dropout: float = 0.0,
        time_mask_num: int = 0,
        time_mask_width: int = 25,
        per_dim_noise_scale: Optional[torch.Tensor] = None,
        noise_scale_multiplier: float = 1.0,
        protected_dims: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.gaussian_noise_scale = gaussian_noise_scale
        self.feature_dropout = feature_dropout
        self.time_mask_num = time_mask_num
        self.time_mask_width = time_mask_width
        self.noise_scale_multiplier = noise_scale_multiplier

        # Register per-dimension noise scale as a buffer so it moves with
        # the model to the correct device and is saved/loaded with state_dict,
        # but is NOT treated as a trainable parameter.
        if per_dim_noise_scale is not None:
            self.register_buffer("per_dim_noise_scale", per_dim_noise_scale)
        else:
            self.per_dim_noise_scale = None

        if protected_dims is not None:
            self.register_buffer("protected_dims", protected_dims.bool(), persistent=False)
        else:
            self.protected_dims = None

    # ------------------------------------------------------------------
    def forward(
        self,
        encoded: torch.Tensor,
        encoded_len: torch.Tensor,
    ) -> torch.Tensor:
        """Apply latent augmentations in-place (returns the same tensor).

        Parameters
        ----------
        encoded : Tensor, shape ``(B, D, T)``
            Encoder activations in NeMo's channel-first layout (as produced
            by ``EncDecRNNTModel.forward`` and the cached encoder dataset).
        encoded_len : Tensor, shape ``(B,)``
            Valid lengths for each sample (frames, not seconds).

        Returns
        -------
        Tensor
            Augmented encoder activations, same shape as *encoded*.
        """
        if not self.training:
            return encoded

        # Per-dim noise takes priority over uniform gaussian_noise_scale
        if self.per_dim_noise_scale is not None:
            encoded = self._add_gaussian_noise(encoded)
        elif self.gaussian_noise_scale > 0:
            encoded = self._add_gaussian_noise(encoded)

        if self.feature_dropout > 0:
            encoded = self._apply_feature_dropout(encoded)

        if self.time_mask_num > 0 and self.time_mask_width > 0:
            encoded = self._apply_time_masking(encoded, encoded_len)

        return encoded

    # ------------------------------------------------------------------
    # Individual augmentation methods
    # ------------------------------------------------------------------

    def _add_gaussian_noise(self, encoded: torch.Tensor) -> torch.Tensor:
        """Add zero-mean Gaussian noise.

        When ``per_dim_noise_scale`` is set, each feature dimension gets noise
        proportional to its empirically measured std (from augmented encoder
        passes), scaled by ``noise_scale_multiplier``.  Otherwise falls back
        to a uniform ``gaussian_noise_scale`` across all dimensions.
        """
        if self.per_dim_noise_scale is not None:
            # per_dim_noise_scale shape: (D,) — reshape to (1, D, 1) to
            # broadcast over (B, D, T)
            noise = (
                torch.randn_like(encoded)
                * self.per_dim_noise_scale.view(1, -1, 1)
                * self.noise_scale_multiplier
            )
        else:
            noise = torch.randn_like(encoded) * self.gaussian_noise_scale
        if self.protected_dims is not None:
            noise = noise * (~self.protected_dims).to(noise.dtype).view(1, -1, 1)
        return encoded + noise

    def _apply_feature_dropout(self, encoded: torch.Tensor) -> torch.Tensor:
        """Zero entire feature dimensions with probability ``self.feature_dropout``.

        For a tensor of shape ``(B, D, T)``, a Bernoulli mask of shape
        ``(B, D, 1)`` is sampled so the same features are dropped across all
        time steps within a sample — analogous to frequency masking in
        SpecAugment.
        """
        B, D, _T = encoded.shape
        # mask shape (B, D, 1) → broadcast across time
        keep = torch.bernoulli(
            torch.full((B, D, 1), 1.0 - self.feature_dropout, device=encoded.device)
        )
        if self.protected_dims is not None:
            keep[:, self.protected_dims, :] = 1.0
        return encoded * keep

    def _apply_time_masking(
        self,
        encoded: torch.Tensor,
        encoded_len: torch.Tensor,
    ) -> torch.Tensor:
        """Zero contiguous time segments, SpecAugment-style.

        For each sample, ``self.time_mask_num`` masks are generated.  Each
        mask starts at a uniformly random position and has a random width
        up to ``self.time_mask_width`` (clamped to the valid length).
        """
        B, _D, T = encoded.shape
        for b in range(B):
            valid_len = int(encoded_len[b].item())
            if valid_len <= 1:
                continue
            for _ in range(self.time_mask_num):
                width = torch.randint(1, self.time_mask_width + 1, (1,)).item()
                width = min(width, valid_len)
                start = torch.randint(0, valid_len - width + 1, (1,)).item()
                encoded[b, :, start : start + width] = 0.0
        return encoded

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        per_dim = (
            f"per_dim_noise_scale=({self.per_dim_noise_scale.shape[0]} dims), "
            f"noise_scale_multiplier={self.noise_scale_multiplier}"
            if self.per_dim_noise_scale is not None
            else f"gaussian_noise_scale={self.gaussian_noise_scale}"
        )
        protected = (
            f", protected_dims={int(self.protected_dims.sum())}/{self.protected_dims.numel()}"
            if self.protected_dims is not None
            else ""
        )
        return (
            f"{self.__class__.__name__}("
            f"{per_dim}, "
            f"feature_dropout={self.feature_dropout}, "
            f"time_mask_num={self.time_mask_num}, "
            f"time_mask_width={self.time_mask_width}"
            f"{protected})"
        )
