# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for the opt-in web-export flags (mask-free graph, runtime positional
encoding, padded-batch NaN tripwire) used by perso/export_onnx.py
--web-optimized. Written with Claude Code."""

import pytest
import torch

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.parts.submodules.multi_head_attention import RelPositionalEncoding


@pytest.fixture()
def encoder():
    enc = ConformerEncoder(
        feat_in=80,
        n_layers=1,
        d_model=64,
        n_heads=4,
        subsampling='dw_striding',
        subsampling_factor=8,
        conv_kernel_size=9,
        self_attention_model='rel_pos',
    )
    enc.eval()
    return enc


class TestRuntimePositionalEncoding:
    @pytest.mark.unit
    def test_matches_table_bitexact(self):
        pe_mod = RelPositionalEncoding(d_model=64, dropout_rate=0.0, max_len=256, dropout_rate_emb=0.0)
        pe_mod.eval()
        pe_mod.extend_pe(256, torch.device('cpu'), torch.float32)
        for t in (1, 7, 64, 255):
            x = torch.randn(1, t, 64)
            pe_mod.export_runtime_pe = False
            with torch.no_grad():
                _, emb_table = pe_mod(x)
            pe_mod.export_runtime_pe = True
            with torch.no_grad():
                _, emb_runtime = pe_mod(x)
            assert torch.equal(emb_table, emb_runtime), f"PE mismatch at T={t}"

    @pytest.mark.unit
    def test_beyond_table_length(self):
        pe_mod = RelPositionalEncoding(d_model=64, dropout_rate=0.0, max_len=64, dropout_rate_emb=0.0)
        pe_mod.eval()
        pe_mod.export_runtime_pe = True
        with torch.no_grad():
            _, emb = pe_mod(torch.randn(1, 500, 64))
        assert emb.shape == (1, 999, 64)
        assert torch.isfinite(emb).all()


class TestWebExportEncoderFlags:
    @pytest.mark.unit
    def test_skip_mask_matches_masked_on_full_length(self, encoder):
        torch.manual_seed(0)
        audio = torch.randn(1, 80, 64)
        length = torch.tensor([64], dtype=torch.int64)
        with torch.no_grad():
            out_masked, len_masked = encoder(audio_signal=audio, length=length)
        encoder.export_skip_mask = True
        encoder.export_pad_tripwire = True
        with torch.no_grad():
            out_free, len_free = encoder(audio_signal=audio, length=length)
        assert torch.equal(len_masked, len_free)
        assert torch.equal(out_masked, out_free)

    @pytest.mark.unit
    def test_create_masks_returns_none(self, encoder):
        encoder.export_skip_mask = True
        pad_mask, att_mask = encoder._create_masks(
            att_context_size=encoder.att_context_size,
            padding_length=torch.tensor([8]),
            max_audio_length=8,
            offset=None,
            device=torch.device('cpu'),
        )
        assert pad_mask is None and att_mask is None

    @pytest.mark.unit
    def test_tripwire_poisons_padded_batch(self, encoder):
        encoder.export_skip_mask = True
        encoder.export_pad_tripwire = True
        audio = torch.randn(2, 80, 64)
        length = torch.tensor([64, 32], dtype=torch.int64)
        with torch.no_grad():
            out, _ = encoder(audio_signal=audio, length=length)
        assert torch.isnan(out).all(), "padded batch must be NaN-poisoned"

    @pytest.mark.unit
    def test_tripwire_clean_on_equal_length_batch(self, encoder):
        encoder.export_skip_mask = True
        encoder.export_pad_tripwire = True
        audio = torch.randn(2, 80, 64)
        length = torch.tensor([64, 64], dtype=torch.int64)
        with torch.no_grad():
            out, _ = encoder(audio_signal=audio, length=length)
        assert torch.isfinite(out).all()

    @pytest.mark.unit
    def test_tripwire_clean_on_batch1_short_length(self, encoder):
        # Regression: real mel frontends report length < T for every clip in
        # some cases (onnx-asr emits one extra frame when the sample count is
        # a multiple of the hop). A batch-1 clip must NEVER be poisoned; the
        # tripwire only targets mixed-length batches.
        encoder.export_skip_mask = True
        encoder.export_pad_tripwire = True
        audio = torch.randn(1, 80, 65)
        length = torch.tensor([64], dtype=torch.int64)
        with torch.no_grad():
            out, _ = encoder(audio_signal=audio, length=length)
        assert torch.isfinite(out).all(), "batch-1 with length < T must stay finite"
