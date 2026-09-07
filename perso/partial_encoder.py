"""Helpers for running a ConformerEncoder up to / from a specific layer.

Used by the cached-encoder training pipeline when ``encoder_except_last_n > 0``:
the cache stores activations after layer ``cut_layer`` (= ``L - except_last_n - 1``)
and the live training step runs only ``layers[cut_layer+1 :]`` plus ``out_proj``.

Two entry points
----------------

``truncated_encoder(encoder, cut_layer)``
    Context manager used at *cache time*.  Temporarily reduces
    ``encoder.layers`` to ``layers[: cut_layer + 1]`` and disables
    ``out_proj`` so a normal ``asr_model.forward(input_signal=...)`` call
    emits the intermediate activations.  ``cut_layer == -1`` is a no-op
    (full encoder).

``run_encoder_from_layer(encoder, cached_bdt, cached_len, start_layer)``
    Used at *training time*.  Takes cached activations of shape
    ``(B, D, T)`` (the format saved by the cache) and runs ``layers[start_layer:]``
    on them, returning the final encoder output ``(B, D_out, T)``.
"""

from __future__ import annotations

import random
from contextlib import contextmanager

import torch
import torch.nn as nn


@contextmanager
def truncated_encoder(encoder, cut_layer: int):
    """Temporarily truncate ``encoder`` so its forward stops after layer ``cut_layer``.

    Disables ``out_proj`` while the context is active so the cached tensor
    matches the input expected by ``run_encoder_from_layer`` at training time.
    """
    if cut_layer is None or cut_layer < 0:
        yield
        return

    n_total = len(encoder.layers)
    if cut_layer >= n_total:
        raise ValueError(
            f"cut_layer={cut_layer} is out of range for encoder with "
            f"{n_total} layers (max valid index is {n_total - 1})"
        )

    orig_layers = encoder.layers
    orig_out_proj = encoder.out_proj
    orig_drops = encoder.layer_drop_probs

    encoder.layers = nn.ModuleList(list(orig_layers[: cut_layer + 1]))
    encoder.out_proj = None
    encoder.layer_drop_probs = orig_drops[: cut_layer + 1]
    try:
        yield
    finally:
        encoder.layers = orig_layers
        encoder.out_proj = orig_out_proj
        encoder.layer_drop_probs = orig_drops


def _get_pos_emb(encoder, seq_len: int, device, dtype):
    """Extract the relative positional embedding tensor without modifying the signal.

    ``encoder.pos_enc.forward(x)`` would re-apply xscale + dropout to the
    cached tensor, which we already passed through pos_enc once at cache
    time.  This helper only reads the positional buffer ``pe``.
    """
    pos_enc = encoder.pos_enc
    pos_enc.extend_pe(seq_len, device, dtype)

    if encoder.self_attention_model == "rel_pos":
        center_pos = pos_enc.pe.size(1) // 2 + 1
        return pos_enc.pe[:, center_pos - seq_len : center_pos + seq_len - 1]
    elif encoder.self_attention_model == "abs_pos":
        return pos_enc.pe[:, :seq_len]
    else:
        raise NotImplementedError(
            f"Partial encoder resume does not support self_attention_model="
            f"{encoder.self_attention_model!r} (only rel_pos / abs_pos)."
        )


def run_encoder_from_layer(encoder, cached_bdt, cached_len, start_layer: int):
    """Run ``encoder.layers[start_layer:]`` on cached activations.

    Parameters
    ----------
    encoder : ConformerEncoder
    cached_bdt : torch.Tensor
        Shape ``(B, D, T)`` — output of ``layers[start_layer - 1]`` after the
        final transpose (i.e. exactly the format saved by the cache writer
        when ``truncated_encoder`` is in effect).
    cached_len : torch.Tensor
        Shape ``(B,)`` — valid frame counts for each sample.
    start_layer : int
        First layer to execute.  ``start_layer == cut_layer + 1``.

    Returns
    -------
    encoded : torch.Tensor of shape ``(B, D_out, T)``
    encoded_len : torch.Tensor of shape ``(B,)``, dtype int64
    """
    audio_signal = cached_bdt.transpose(1, 2).contiguous()  # (B, T, D)
    length = cached_len.to(audio_signal.device).to(torch.int64)
    seq_len = audio_signal.size(1)

    encoder.update_max_seq_length(seq_length=seq_len, device=audio_signal.device)

    pos_emb = _get_pos_emb(
        encoder, seq_len=seq_len, device=audio_signal.device, dtype=audio_signal.dtype
    )

    if encoder.training and len(encoder.att_context_size_all) > 1:
        cur_att_context_size = random.choices(
            encoder.att_context_size_all, weights=encoder.att_context_probs
        )[0]
    else:
        cur_att_context_size = encoder.att_context_size

    pad_mask, att_mask = encoder._create_masks(
        att_context_size=cur_att_context_size,
        padding_length=length,
        max_audio_length=seq_len,
        offset=None,
        device=audio_signal.device,
    )

    for drop_prob, layer in zip(
        encoder.layer_drop_probs[start_layer:], encoder.layers[start_layer:]
    ):
        original_signal = audio_signal
        audio_signal = layer(
            x=audio_signal,
            att_mask=att_mask,
            pos_emb=pos_emb,
            pad_mask=pad_mask,
            cache_last_channel=None,
            cache_last_time=None,
        )
        if encoder.training and drop_prob > 0.0:
            should_drop = torch.rand(1) < drop_prob
            if should_drop:
                audio_signal = audio_signal * 0.0 + original_signal
            else:
                audio_signal = (
                    (audio_signal - original_signal) / (1.0 - drop_prob)
                    + original_signal
                )

    if encoder.out_proj is not None:
        audio_signal = encoder.out_proj(audio_signal)

    audio_signal = audio_signal.transpose(1, 2)
    return audio_signal, length
