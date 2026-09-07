# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import os
import tempfile

import numpy as np
import pytest
import soundfile as sf
import torch

from nemo.collections.asr.data import audio_to_text
from nemo.collections.asr.parts.utils.asr_batching import (
    DurationCostBatchSampler,
    SemiSortBatchSampler,
    rnnt_batch_cost,
)
from nemo.collections.asr.parts.utils.manifest_utils import write_manifest


class TestASRSamplers:
    labels = [
        " ",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "'",
    ]

    @pytest.mark.unit
    def test_ssb_sampler(self):
        # Generate random signals
        data_min_duration = 0.1
        data_max_duration = 16.7

        random_seed = 42
        sample_rate = 16000

        _rng = np.random.default_rng(seed=random_seed)

        def generate_samples(num_examples: int) -> list:
            data_duration = np.round(_rng.uniform(low=data_min_duration, high=data_max_duration, size=num_examples), 3)
            data_duration_samples = np.floor(data_duration * sample_rate).astype(int)
            samples = []
            for data_duration_sample in data_duration_samples:
                samples.append(_rng.uniform(low=-0.5, high=0.5, size=(data_duration_sample)))
            return samples

        with tempfile.TemporaryDirectory() as test_dir:
            # Build metadata for manifest
            metadata = []

            # Test size of dataloader with and without ssb
            for num_samples in np.concatenate([np.array([1, 2]), _rng.integers(3, 10, 2), _rng.integers(10, 1000, 2)]):
                samples = generate_samples(num_samples)

                for n, sample in enumerate(samples):
                    meta = dict()
                    signal_filename = f'{n:04d}.wav'
                    # write audio files
                    sf.write(os.path.join(test_dir, signal_filename), sample, sample_rate)
                    # update metadata
                    meta['audio_filepath'] = os.path.join(test_dir, signal_filename)
                    meta['duration'] = len(sample) / sample_rate
                    meta['text'] = 'non empty'
                    metadata.append(meta)

                # Save manifest
                manifest_filepath = os.path.join(test_dir, 'manifest.json')
                write_manifest(manifest_filepath, metadata)

                # Make dataset
                dataset = audio_to_text.AudioToCharDataset(
                    manifest_filepath=manifest_filepath,
                    labels=self.labels,
                    sample_rate=sample_rate,
                    max_duration=data_max_duration,
                    min_duration=data_min_duration,
                )
                durations = [sample.duration for sample in dataset.manifest_processor.collection.data]

                # Compare two dataloader
                for batch_size in _rng.integers(1, n + 20, 5):
                    batch_size = int(batch_size)
                    drop_last = True if _rng.integers(0, 2) else False
                    sampler = SemiSortBatchSampler(
                        global_rank=0,
                        world_size=1,
                        durations=durations,
                        batch_size=batch_size,
                        batch_shuffle=True,
                        drop_last=drop_last,
                        randomization_factor=0.1,
                        seed=random_seed,
                    )
                    dataloader_with_ssb = torch.utils.data.DataLoader(
                        dataset=dataset,
                        batch_size=None,
                        sampler=sampler,
                        batch_sampler=None,
                        collate_fn=lambda x: audio_to_text._speech_collate_fn(x, pad_id=0),
                    )
                    dataloader = torch.utils.data.DataLoader(
                        dataset=dataset,
                        batch_size=batch_size,
                        collate_fn=lambda x: audio_to_text._speech_collate_fn(x, pad_id=0),
                        drop_last=drop_last,
                        shuffle=True,
                    )

                    assert abs(len(dataloader) - len(dataloader_with_ssb)) == 0, (
                        "Different num of batches with batch! Num of batches with ssb is "
                        f"{len(dataloader_with_ssb)} and without ssb is {len(dataloader)}!"
                    )

                    dataloader_with_ssb_exception, dataloader_exception = False, False

                    try:
                        list(dataloader_with_ssb)
                    except:
                        dataloader_with_ssb_exception = True

                    try:
                        list(dataloader)
                    except:
                        dataloader_exception = True

                    assert dataloader_with_ssb_exception == dataloader_exception

    # ---- DurationCostBatchSampler (adaptive batch size) ----------------------
    # Added for the UltiMed fine-tune with Claude Code. The sampler decides how
    # much fits on the GPU, so a silent regression here is an OOM ten hours into
    # a multi-day run: these lock down the cost model and every packing bound.

    @staticmethod
    def _durations(seed: int = 0, n: int = 2000) -> np.ndarray:
        """Duration spread close to the real corpus: median ~17 s, tail to 38 s."""
        rng = np.random.default_rng(seed)
        return np.clip(rng.gamma(shape=4.0, scale=4.5, size=n), 1.0, 38.0)

    def test_duration_cost_matches_calibration(self):
        # The budget was fitted against measured VRAM on an RTX 3090 Ti. If this
        # drifts, every budget in training_config.yaml is silently wrong.
        for durations, expected in (
            ([25.0] * 4, 13750.0),
            ([40.0], 16000.0),
            ([25.0] * 6, 18750.0),
            ([30.0] * 4, 19800.0),
        ):
            assert rnnt_batch_cost(durations) == pytest.approx(expected)
        assert rnnt_batch_cost([]) == 0.0

    def test_duration_cost_sampler_respects_every_bound(self):
        durations = self._durations()
        budget, max_bs, max_total = 15000.0, 16, 240.0
        sampler = DurationCostBatchSampler(
            durations=durations,
            budget=budget,
            max_batch_size=max_bs,
            max_total_duration=max_total,
            sort_pool_size=512,
        )
        for batch in sampler:
            assert len(batch) <= max_bs
            assert durations[batch].sum() <= max_total
            # a single clip has no smaller batch to fall back to, so it is
            # allowed through even if it busts the budget
            if len(batch) > 1:
                assert rnnt_batch_cost(durations[batch]) <= budget

    def test_duration_cost_sampler_covers_every_sample_once(self):
        durations = self._durations(n=500)
        sampler = DurationCostBatchSampler(durations=durations, budget=15000.0)
        for epoch in range(3):
            sampler.set_epoch(epoch)
            drawn = [i for batch in sampler for i in batch]
            assert sorted(drawn) == list(range(len(durations)))
            # Lightning sizes the progress bar and val_check_interval from
            # __len__ before iterating, so the two must agree.
            assert len(sampler) == len(list(iter(sampler)))

    def test_duration_cost_sampler_reshuffles_but_is_reproducible(self):
        durations = self._durations(n=500)
        first = DurationCostBatchSampler(durations=durations, budget=15000.0, seed=7)
        epoch0 = [list(b) for b in first]
        first.set_epoch(1)
        epoch1 = [list(b) for b in first]
        assert epoch0 != epoch1, "batches must change between epochs"

        same_seed = DurationCostBatchSampler(durations=durations, budget=15000.0, seed=7)
        assert [list(b) for b in same_seed] == epoch0, "same seed must replay the same epoch"

        no_shuffle = DurationCostBatchSampler(durations=durations, budget=15000.0, shuffle=False)
        before = [list(b) for b in no_shuffle]
        no_shuffle.set_epoch(1)
        assert [list(b) for b in no_shuffle] == before, "shuffle=False must be stable"

    def test_duration_cost_sampler_adapts_size_to_duration(self):
        # The whole point: long clips travel alone, short ones in full batches.
        long_only = DurationCostBatchSampler(durations=[38.0] * 20, budget=15000.0)
        assert {len(b) for b in long_only} == {1}

        short_only = DurationCostBatchSampler(
            durations=[5.0] * 100, budget=15000.0, max_batch_size=16, max_total_duration=240.0
        )
        assert max(len(b) for b in short_only) == 16

    def test_duration_cost_sampler_emits_oversized_clip_alone(self):
        # 60 s alone costs 36000, well over the budget. It must still come out,
        # on its own, rather than be dropped or padded into a bigger batch.
        durations = [60.0, 5.0, 5.0]
        sampler = DurationCostBatchSampler(durations=durations, budget=15000.0, shuffle=False)
        batches = [list(b) for b in sampler]
        assert [0] in batches
        assert sorted(i for b in batches for i in b) == [0, 1, 2]

    def test_duration_cost_sampler_shards_across_ranks(self):
        durations = self._durations(n=500)
        world_size = 3
        shards = [
            DurationCostBatchSampler(durations=durations, budget=15000.0, global_rank=rank, world_size=world_size)
            for rank in range(world_size)
        ]
        assert len({len(s) for s in shards}) == 1, "ranks must agree on the step count or DDP hangs"
        seen = [tuple(b) for s in shards for b in s]
        assert len(seen) == len(set(seen)), "no batch may be handed to two ranks"

    def test_duration_cost_sampler_rejects_bad_config(self):
        with pytest.raises(ValueError):
            DurationCostBatchSampler(durations=[], budget=15000.0)
        with pytest.raises(ValueError):
            DurationCostBatchSampler(durations=[10.0], budget=0)
        with pytest.raises(ValueError):
            DurationCostBatchSampler(durations=[10.0], budget=15000.0, max_batch_size=0)
