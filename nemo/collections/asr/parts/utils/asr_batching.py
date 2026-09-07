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

import math
from typing import Iterator, List, Optional, Union

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler

from nemo.collections.asr.data.audio_to_text import AudioToBPEDataset, AudioToCharDataset
from nemo.collections.asr.models.asr_model import ASRModel
from nemo.utils import logging


class SemiSortBatchSampler(DistributedSampler):
    def __init__(
        self,
        global_rank: int,
        world_size: int,
        durations: List[int],
        batch_size: int,
        batch_shuffle: bool = True,
        drop_last: bool = False,
        randomization_factor: Optional[float] = None,
        seed: int = 42,
    ) -> None:
        r"""
        Semi Sorted Batching, as proposed in _SSB ("Speed up training with variable
        length inputs by efficient batching strategies.", Zhenhao Ge et al. (2021).).

        The Semi Sorted Batch Sampler (SSB) samples the indices by their duration
        with the addition of pseudo noise that is sampled from the uniform
        distribution \mathbb{U}\left[ -delta * r, delta * r \right], where delta is
        defined as the difference between the maximum and minimum duration and r is
        the randomization factor that controls the strength of the noise (when r = 0,
        there will be a strong sorting). The heuristic value of the r according to
        the experiments from paper is 0.2.

        The torch calls the set_epoch method from the distributed data loader sampler
        at the end of each epoch to shuffle the samples according to the seed and
        epoch number. So the SSB is passed to the dataloader as a sampler with the
        dataloader's batch size options and the batch_sampler option set to None to
        disable automatical batching. In this case, the sampler has become an iterator
        that returns a list of batch indices.

        Args:
            global_rank: Rank among all GPUs.
            world_size: The number of GPUs used.
            durations: Sample durations parsed from `dataset.manifest_processor`.
            batch_size: Micro batch size or batch size per singe gpu.
            batch_shuffle: Batch sort before each epoch.
            drop_last: Drop the last batch if the number of samples is less than batch
                size. Defaults to False.
            randomization_factor: The strength of noise that will be added to the sample
                duration. If no value is passed, the value 0.2 will be used.
            seed: Seed for batch shuffleling. Defaults to 42.

        Raises:
            ValueError: Wrong randomization factor value.
            RuntimeError: Unexpected behavior.

        .. SSB_:
            https://www.isca-archive.org/interspeech_2021/ge21_interspeech.pdf
        """
        if randomization_factor is None:
            randomization_factor = 0.1
            logging.info("Randomization factor not found in config, default value 0.1 will be set.")
        else:
            logging.info(f"A randomization factor {randomization_factor} will be used.")

        if randomization_factor < 0.0:
            raise ValueError(f'Randomization factor must be non-negative but found {randomization_factor}.')

        self.rank: List = global_rank
        self.num_replicas: int = world_size

        self.durations: np.array = np.array(durations, dtype=np.float32)

        self.shuffle: bool = batch_shuffle
        self.micro_batch_size: int = batch_size
        self.drop_last: bool = drop_last
        self.epoch: int = 0
        self.seed: int = seed
        self.randomization_factor: float = randomization_factor

        self.local_num_batches: int = self._calculate_local_num_batches()

        logging.info(f"Semi Sorted Batch Sampler will be used")

    def _calculate_local_num_batches(self) -> int:
        init_num_samples = len(self.durations)

        # delete batches with a non-integer number of samples
        if self.drop_last:
            init_num_samples -= init_num_samples % self.micro_batch_size

        # calculate the number of batches according to the counted number of samples
        global_num_batches = math.ceil(init_num_samples / self.micro_batch_size)

        # add extra batches to make it divisible by world size (num replicas)
        num_batches_pad = (self.num_replicas - global_num_batches % self.num_replicas) % self.num_replicas
        global_num_batches += num_batches_pad

        # calculate the number of batches per rank
        local_num_batches = global_num_batches // self.num_replicas

        return local_num_batches

    def _make_batches(self) -> List[np.array]:
        max_duration: float = np.max(self.durations)
        min_duration: float = np.min(self.durations)
        bound: float = (max_duration - min_duration) * self.randomization_factor / 2

        # generate pseudo noise
        noise: np.array = np.random.uniform(low=-bound, high=bound, size=len(self.durations))

        # sort indices accroding to pseudo noise
        sorted_indices: np.array = np.argsort(self.durations + noise)

        # delete batches with a non-integer number of samples
        tail = 0
        if self.drop_last:
            tail: int = len(sorted_indices) % self.micro_batch_size
            exclude = np.random.choice(len(sorted_indices), tail, replace=False)
            sorted_indices = np.delete(sorted_indices, exclude)
            logging.warning(f"Drop last is set to True, so {len(exclude)} samples will be dropped.")

        global_num_batches: int = math.ceil(len(sorted_indices) / self.micro_batch_size)

        # if the global_num_batches is zero than return empty list
        if global_num_batches == 0:
            logging.warning(
                f"The number of all batches is {global_num_batches}, than dataloader will "
                "be empty. To avoid this try to decrease batch size or world size or set "
                "drop_last to False."
            )
            return []

        # add extra batches to make it divisible by world size (num replicas)
        pad_batches_num: int = (self.num_replicas - global_num_batches % self.num_replicas) % self.num_replicas
        if global_num_batches < self.num_replicas:
            logging.warning(
                f"The number of all batches is {global_num_batches}, which is less than the "
                f"world size of {self.num_replicas}. SSB Sampler will add {pad_batches_num} "
                "batches. To avoid this try to decrease batch size or world size."
            )

        if pad_batches_num != 0:
            # randomly select batch indeces to pad and concatenate them
            batch_indeces_pad: np.array = np.random.randint(
                low=0,
                high=len(sorted_indices),
                size=pad_batches_num * self.micro_batch_size,
            )
            sorted_indices: np.array = np.concatenate(
                (sorted_indices, sorted_indices[batch_indeces_pad]),
                axis=0,
            )

        # local indeces are selected by world size and local rank
        local_indices: np.array = sorted_indices[self.rank :: self.num_replicas]

        # split local batches
        size_mask = range(self.micro_batch_size, len(local_indices), self.micro_batch_size)
        local_batches = np.split(local_indices, size_mask, axis=0)

        if len(local_batches) != self.local_num_batches:
            raise RuntimeError(
                f'Number of calculated indices {len(local_batches)} is not equal to calculated '
                f'number of local batches {self.local_num_batches}.'
            )

        return local_batches

    def __iter__(self) -> Iterator[List[int]]:
        local_batches = self._make_batches()

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch + 1)
            indices = torch.randperm(self.local_num_batches, generator=g)
        else:
            indices = torch.arange(0, self.local_num_batches)

        for _, index in enumerate(indices):
            yield local_batches[index]

    def __len__(self) -> int:
        return self.local_num_batches


def get_semi_sorted_batch_sampler(
    model: ASRModel, dataset: Union[AudioToCharDataset, AudioToBPEDataset], config: dict
) -> SemiSortBatchSampler:
    """
    Instantiates a Semi Sorted (Batch) Sampler.

    Args:
        model: ASR Model.
        dataset: Dataset which allow iterate over all object and parse durations.
        config: Train, Vaidation or Test dataset config.

    Raises:
        ValueError: Wrong dataset type.

    Returns:
        SemiSortBatchSampler: Semi Sorted Batch Sampler class.
    """
    if not (isinstance(dataset, AudioToCharDataset) or isinstance(dataset, AudioToBPEDataset)):
        raise ValueError(
            "Only AudioToCharDataset or AudioToBPEDataset supported with semi sorted batching, "
            f"but found {type(dataset)}."
        )

    durations = [sample.duration for sample in dataset.manifest_processor.collection.data]

    sampler = SemiSortBatchSampler(
        global_rank=model.global_rank,
        world_size=model.world_size,
        durations=durations,
        batch_size=config['batch_size'],
        batch_shuffle=config.get('shuffle', True),
        drop_last=config.get('drop_last', False),
        randomization_factor=config.get('randomization_factor', None),
        seed=config.get('semi_sort_sampler_seed', 42),
    )

    return sampler


# =============================================================================
# Duration-cost batch sampler (adaptive batch size)
# =============================================================================
# Added for the UltiMed fine-tune with Claude Code.
#
# The RNNT/TDT joint materialises a [B, T, U, V] lattice, with T proportional to
# the clip duration and U proportional to the transcript length. On a corpus
# with a roughly constant speaking rate U tracks T, so the memory a clip costs
# grows with the SQUARE of its duration. A fixed batch size therefore has to be
# sized for the longest clip the manifest can produce, which wastes most of the
# GPU on the short ones and forces a low max_duration cap that throws away the
# long-form recordings entirely.
#
# This sampler instead packs each batch up to a memory budget, so a batch is one
# 40 s clip or eleven 17 s ones, whichever the budget allows.
#
# Cost model, fitted on an RTX 3090 Ti (24564 MiB) with parakeet-tdt-0.6b-v3,
# joint.fused_batch_size 1, last 4 encoder layers trainable:
#
#     cost = 4 * sum_i(d_i^2) + 6 * max_i(d_i^2)
#
# The second term is the transient of the single largest clip in the batch (the
# joint runs one clip at a time under fused_batch_size 1, so the peak is set by
# the biggest one, not by the sum).
#
# Measured, worst-case batches, cap:batch -> cost -> outcome
#     30:2 -> 12600 -> OK (21130 MiB)     25:4 -> 13750 -> OK (23353 MiB)
#     40:1 -> 16000 -> OK (23409 MiB)     30:3 -> 16200 -> OK (23983 MiB)
#     35:2 -> 17150 -> OK (23948 MiB)     25:6 -> 18750 -> OK (23940 MiB)
#     30:4 -> 19800 -> OOM                45:1 -> 20250 -> OOM
#     35:4 -> 26950 -> OOM                40:4 -> 35200 -> OOM
#
# Every passing configuration sits at or below 18750 and every failing one at or
# above 19800, so the model separates them cleanly. Re-fit it (and re-run
# perso/oom_margin_check.sh) if the GPU, the model, the number of trainable
# encoder layers or joint.fused_batch_size changes.

_COST_SUM_WEIGHT = 4.0
_COST_MAX_WEIGHT = 6.0


def rnnt_batch_cost(
    durations: Union[List[float], np.ndarray],
    sum_weight: float = _COST_SUM_WEIGHT,
    max_weight: float = _COST_MAX_WEIGHT,
) -> float:
    """Memory cost of one RNNT/TDT batch in the fitted (arbitrary) unit.

    Args:
        durations: clip durations in seconds, one per sample in the batch.
        sum_weight: weight of the per-clip term.
        max_weight: weight of the largest-clip transient.
    """
    if len(durations) == 0:
        return 0.0
    squares = np.square(np.asarray(durations, dtype=np.float64))
    return float(sum_weight * squares.sum() + max_weight * squares.max())


class DurationCostBatchSampler(torch.utils.data.Sampler):
    """Batch sampler that sizes every batch by memory cost instead of clip count.

    Yields lists of dataset indices, so it must be passed to the DataLoader as
    ``batch_sampler=`` (not ``sampler=``).

    Batches are built from a shuffled stream in windows of ``sort_pool_size``
    samples. Each window is sorted by duration before packing, because the
    encoder pads every clip in a batch up to the longest one: mixing a 3 s clip
    with a 39 s one would spend most of the batch on padding. Sorting inside a
    window rather than globally keeps batch composition different every epoch.

    Args:
        durations: duration in seconds of every sample in the dataset, in
            dataset index order.
        budget: maximum ``rnnt_batch_cost`` a batch may reach. A sample whose
            cost alone exceeds it is still emitted, alone, since no smaller
            batch exists: cap ``max_duration`` so that cannot happen.
        max_batch_size: hard cap on clips per batch, so a run of very short
            clips cannot produce an enormous batch.
        max_total_duration: optional cap on the summed audio seconds in a batch,
            bounding the encoder and preprocessor activations that scale with
            total audio rather than with the squared duration.
        sort_pool_size: window size used for the duration sort. 0 disables the
            sort and packs in shuffled order.
        shuffle: reshuffle every epoch. Requires ``set_epoch`` to be called,
            which Lightning does automatically.
        drop_last: drop the final batch of the epoch.
        seed: base seed. The epoch number is added to it.
        global_rank, world_size: batches are dealt round-robin across ranks.
    """

    def __init__(
        self,
        durations: Union[List[float], np.ndarray],
        budget: float,
        max_batch_size: int = 16,
        max_total_duration: Optional[float] = None,
        sort_pool_size: int = 512,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
        global_rank: int = 0,
        world_size: int = 1,
        sum_weight: float = _COST_SUM_WEIGHT,
        max_weight: float = _COST_MAX_WEIGHT,
    ):
        self.durations = np.asarray(durations, dtype=np.float64)
        if self.durations.ndim != 1 or self.durations.size == 0:
            raise ValueError("durations must be a non-empty 1-D sequence")
        if budget <= 0:
            raise ValueError(f"budget must be positive, got {budget}")
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be at least 1, got {max_batch_size}")

        self.budget = float(budget)
        self.max_batch_size = int(max_batch_size)
        self.max_total_duration = None if max_total_duration is None else float(max_total_duration)
        self.sort_pool_size = int(sort_pool_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.global_rank = int(global_rank)
        self.world_size = int(world_size)
        self.sum_weight = float(sum_weight)
        self.max_weight = float(max_weight)
        self.epoch = 0

        solo_cost = (self.sum_weight + self.max_weight) * np.square(self.durations)
        n_oversized = int((solo_cost > self.budget).sum())
        if n_oversized:
            logging.warning(
                f"DurationCostBatchSampler: {n_oversized} sample(s) cost more than the budget "
                f"({self.budget:.0f}) on their own, the longest being "
                f"{self.durations.max():.1f}s at cost {solo_cost.max():.0f}. They will be emitted "
                "alone and may still run out of memory. Lower max_duration or raise the budget."
            )

        self._batches = self._make_batches()

    def _pack(self, indices: np.ndarray) -> List[List[int]]:
        """Greedily fill batches from an ordered index sequence."""
        batches: List[List[int]] = []
        current: List[int] = []
        sum_sq = 0.0
        max_sq = 0.0
        total_dur = 0.0

        for idx in indices:
            duration = float(self.durations[idx])
            square = duration * duration
            new_sum_sq = sum_sq + square
            new_max_sq = square if square > max_sq else max_sq

            full = (
                self.sum_weight * new_sum_sq + self.max_weight * new_max_sq > self.budget
                or len(current) + 1 > self.max_batch_size
                or (self.max_total_duration is not None and total_dur + duration > self.max_total_duration)
            )
            if current and full:
                batches.append(current)
                current = []
                new_sum_sq, new_max_sq, total_dur = square, square, 0.0

            current.append(int(idx))
            sum_sq, max_sq = new_sum_sq, new_max_sq
            total_dur += duration

        if current:
            batches.append(current)
        return batches

    def _make_batches(self) -> List[List[int]]:
        n_samples = len(self.durations)
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(n_samples) if self.shuffle else np.arange(n_samples)

        pool = self.sort_pool_size if self.sort_pool_size > 0 else n_samples
        batches: List[List[int]] = []
        for start in range(0, n_samples, pool):
            window = order[start : start + pool]
            window = window[np.argsort(self.durations[window], kind="stable")]
            batches.extend(self._pack(window))

        if self.shuffle:
            # The sort above made consecutive batches similar in length. Shuffle
            # the batch order so the model does not see all the long clips of a
            # window back to back.
            batches = [batches[i] for i in rng.permutation(len(batches))]

        if self.world_size > 1:
            usable = (len(batches) // self.world_size) * self.world_size
            batches = batches[self.global_rank : usable : self.world_size]
        if self.drop_last and len(batches) > 1:
            batches = batches[:-1]
        return batches

    def set_epoch(self, epoch: int) -> None:
        """Rebuild the batches for a new epoch. Called by Lightning."""
        if int(epoch) != self.epoch:
            self.epoch = int(epoch)
            self._batches = self._make_batches()

    def stats(self) -> dict:
        """Summary of the current epoch's batches, for logging and calibration."""
        sizes = np.array([len(b) for b in self._batches], dtype=np.float64)
        seconds = np.array([self.durations[b].sum() for b in self._batches], dtype=np.float64)
        costs = np.array(
            [rnnt_batch_cost(self.durations[b], self.sum_weight, self.max_weight) for b in self._batches]
        )
        return {
            'batches': len(self._batches),
            'samples': int(sizes.sum()),
            'clips_per_batch_mean': float(sizes.mean()),
            'clips_per_batch_min': int(sizes.min()),
            'clips_per_batch_max': int(sizes.max()),
            'audio_seconds_per_batch_mean': float(seconds.mean()),
            'audio_seconds_per_batch_max': float(seconds.max()),
            'cost_mean': float(costs.mean()),
            'cost_max': float(costs.max()),
        }

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


def get_duration_cost_batch_sampler(
    model: ASRModel,
    dataset: Union[AudioToCharDataset, AudioToBPEDataset],
    config: dict,
) -> DurationCostBatchSampler:
    """Build a :class:`DurationCostBatchSampler` from a train_ds config block."""
    if not (isinstance(dataset, AudioToCharDataset) or isinstance(dataset, AudioToBPEDataset)):
        raise ValueError(
            "Only AudioToCharDataset or AudioToBPEDataset supported with duration cost batching, "
            f"but found {type(dataset)}."
        )

    cost_cfg = config.get('cost_batching', {}) or {}
    durations = [sample.duration for sample in dataset.manifest_processor.collection.data]

    return DurationCostBatchSampler(
        durations=durations,
        budget=cost_cfg.get('budget', 16500),
        max_batch_size=cost_cfg.get('max_batch_size', 16),
        max_total_duration=cost_cfg.get('max_total_duration', None),
        sort_pool_size=cost_cfg.get('sort_pool_size', 512),
        shuffle=config.get('shuffle', True),
        drop_last=config.get('drop_last', False),
        seed=cost_cfg.get('seed', 42),
        global_rank=model.global_rank,
        world_size=model.world_size,
    )
