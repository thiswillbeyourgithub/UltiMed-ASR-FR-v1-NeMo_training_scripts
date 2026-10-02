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

"""Tests for the opt-in web-export flags (key-only additive padding mask,
runtime positional encoding) used by perso/export_onnx.py --web-optimized.
Written with Claude Code."""

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
    def test_key_pad_mask_bitexact_on_batch1(self, encoder):
        torch.manual_seed(0)
        audio = torch.randn(1, 80, 64)
        length = torch.tensor([64], dtype=torch.int64)
        with torch.no_grad():
            out_full, len_full = encoder(audio_signal=audio, length=length)
        encoder.export_key_pad_mask = True
        with torch.no_grad():
            out_light, len_light = encoder(audio_signal=audio, length=length)
        assert torch.equal(len_full, len_light)
        assert torch.equal(out_full, out_light)

    @pytest.mark.unit
    def test_key_pad_mask_matches_full_mask_on_short_batch1(self, encoder):
        # Real mel frontends can report length < T for a lone clip (onnx-asr
        # emits one extra frame when the sample count is a multiple of the
        # hop): the light mask must ignore that frame like the full mask does.
        torch.manual_seed(0)
        audio = torch.randn(1, 80, 72)
        length = torch.tensor([64], dtype=torch.int64)
        with torch.no_grad():
            out_full, len_full = encoder(audio_signal=audio, length=length)
        encoder.export_key_pad_mask = True
        with torch.no_grad():
            out_light, _ = encoder(audio_signal=audio, length=length)
        n = int(len_full[0])  # padded output frames differ (full mask zeroes them) and are trimmed anyway
        assert torch.allclose(out_full[..., :n], out_light[..., :n], atol=1e-5)

    @pytest.mark.unit
    def test_key_pad_mask_ragged_batch_matches_batch1(self, encoder):
        torch.manual_seed(0)
        encoder.export_key_pad_mask = True
        a, b = torch.randn(1, 80, 64), torch.randn(1, 80, 40)
        batch = torch.zeros(2, 80, 64)
        batch[0], batch[1, :, :40] = a[0], b[0]
        with torch.no_grad():
            out, lens = encoder(audio_signal=batch, length=torch.tensor([64, 40]))
            for k, x in enumerate((a, b)):
                ref, ref_len = encoder(audio_signal=x, length=torch.tensor([x.shape[-1]]))
                n = int(ref_len[0])
                assert int(lens[k]) == n
                assert torch.allclose(out[k, :, :n], ref[0, :, :n], atol=1e-4)

    @pytest.mark.unit
    def test_create_masks_returns_key_bias(self, encoder):
        encoder.export_key_pad_mask = True
        pad_mask, bias = encoder._create_masks(
            att_context_size=encoder.att_context_size,
            padding_length=torch.tensor([8, 5]),
            max_audio_length=8,
            offset=None,
            device=torch.device('cpu'),
        )
        assert pad_mask.tolist() == [[False] * 8, [False] * 5 + [True] * 3]
        assert bias.shape == (2, 1, 8) and bias.dtype == torch.float32
        assert bias[0].eq(0).all() and bias[1, 0, :5].eq(0).all() and bias[1, 0, 5:].eq(-10000.0).all()

    @pytest.mark.unit
    def test_key_pad_mask_rejects_sdpa(self, encoder):
        encoder.export_key_pad_mask = True
        encoder.use_pytorch_sdpa = True
        with pytest.raises(ValueError):
            encoder._create_masks(
                att_context_size=encoder.att_context_size,
                padding_length=torch.tensor([8]),
                max_audio_length=8,
                offset=None,
                device=torch.device('cpu'),
            )
