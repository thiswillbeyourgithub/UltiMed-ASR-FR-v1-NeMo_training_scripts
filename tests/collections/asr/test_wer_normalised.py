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

"""Tests for the normalised WER logged alongside the raw one.

Two things here are easy to break without noticing, because both failure modes
still produce a plausible-looking number:

  * the normalisation deleting punctuation rather than replacing it with a
    space. In French that decides whether "l'ongle" is one token or two, so the
    wrong choice quietly shifts every WER and stops the figures being
    comparable with the UltiMed evaluation this was built to match.
  * the callback not draining the metric's counters after each batch, which
    silently attributes earlier dataloaders' errors to later ones. Every set
    still gets a number; they are just wrong, and monotonically worse down the
    list.

Written with Claude Code.
"""

import pytest
import torch
from torchmetrics import Metric

from examples.asr.speech_to_text_finetune_cached import _MacroMetricCallback, _NormalisedWERCallback
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.metrics.wer_normalised import normalise_text, normalised_edit_counts


class TestNormaliseText:
    def test_lowercases(self):
        assert normalise_text("Le Patient") == "le patient"

    def test_strips_punctuation(self):
        assert normalise_text("bonjour, docteur.") == "bonjour docteur"

    def test_collapses_whitespace_and_trims(self):
        assert normalise_text("  deux   mots  ") == "deux mots"

    def test_punctuation_is_deleted_not_spaced(self):
        # The whole point. Spacing out instead of deleting would make this
        # "l ongle", two tokens, and every elision in French would then be a
        # token-count difference between reference and hypothesis.
        assert normalise_text("l'ongle") == "longle"

    def test_hyphenated_and_unhyphenated_forms_converge(self):
        assert normalise_text("micro-traumatismes") == normalise_text("microtraumatismes")

    def test_returns_a_string_not_a_character_list(self):
        # jiwer returns a list for list input; a str promoted to a list here
        # would tokenise into characters downstream and quietly report CER.
        assert isinstance(normalise_text("un mot"), str)

    def test_digits_are_left_alone(self):
        # Deliberately not verbalised: "3" vs "trois" needs a French
        # verbaliser, and half-doing it would be worse than not touching it.
        assert normalise_text("3 bords") == "3 bords"


class TestNormalisedEditCounts:
    def test_identical_text_scores_zero(self):
        scores, words = normalised_edit_counts(["le chat"], ["le chat"])
        assert (scores, words) == (0, 2)

    def test_counts_are_over_the_reference_not_the_hypothesis(self):
        # Two reference words, but three hypothesis words: one substitution
        # (chat/chien) plus one insertion (noir), scored against a denominator
        # of 2. A WER above 100% is possible and is not a bug.
        scores, words = normalised_edit_counts(["le chat noir"], ["le chien"])
        assert words == 2
        assert scores == 2

    def test_punctuation_only_difference_is_free(self):
        # Raw, this is a whole substitution: "chat," != "chat".
        scores, words = normalised_edit_counts(["le chat, noir"], ["le chat noir"])
        assert (scores, words) == (0, 3)

    def test_case_only_difference_is_free(self):
        scores, _ = normalised_edit_counts(["Le Chat"], ["le chat"])
        assert scores == 0

    def test_cer_mode_counts_characters(self):
        scores, words = normalised_edit_counts(["chat"], ["chats"], use_cer=True)
        assert words == 5
        assert scores == 1

    def test_empty_input_is_not_an_error(self):
        assert normalised_edit_counts([], []) == (0, 0)


class _Hyp:
    def __init__(self, text):
        self.text = text


class _FakeDecoding:
    """Maps a single integer id back to a reference string."""

    def __init__(self, references):
        self.references = references

    def decode_ids_to_str(self, ids):
        return self.references[ids[0]]


def _wer_metric(references, hypotheses, use_cer=False):
    """A real WER, built without a real decoding object.

    WER.__init__ rejects anything that is not an Abstract*Decoding subclass, so
    building one properly would drag in a tokenizer and a full decoding config.
    Everything under test lives in update(), so this assembles the same state
    the constructor would and leaves update() itself untouched.
    """
    metric = WER.__new__(WER)
    Metric.__init__(metric)
    metric.decoding = _FakeDecoding(references)
    metric.use_cer = use_cer
    metric.log_prediction = False
    metric.fold_consecutive = True
    metric.batch_dim_index = 0
    metric.decode = lambda *args, **kwargs: [_Hyp(h) for h in hypotheses]
    metric.add_state("scores", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)
    metric.add_state("words", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)
    metric.track_normalised = False
    metric.normalised_scores = 0
    metric.normalised_words = 0
    return metric


def _update(metric, references):
    targets = torch.tensor([[i] for i in range(len(references))])
    lengths = torch.ones(len(references), dtype=torch.long)
    metric.update(
        predictions=torch.ones(len(references), 1),
        predictions_lengths=lengths,
        targets=targets,
        targets_lengths=lengths,
    )


class TestWERTracksNormalisedCounts:
    def test_off_by_default(self):
        metric = _wer_metric(["le chat"], ["le chat,"])
        _update(metric, ["le chat"])
        assert (metric.normalised_scores, metric.normalised_words) == (0, 0)

    def test_normalised_counts_ignore_formatting_the_raw_ones_charge_for(self):
        metric = _wer_metric(["le chat noir"], ["Le chat, noir"])
        metric.track_normalised = True
        _update(metric, ["le chat noir"])
        # Raw: "Le" and "chat," are both substitutions.
        assert metric.scores.item() == 2
        assert metric.words.item() == 3
        # Normalised: same words.
        assert (metric.normalised_scores, metric.normalised_words) == (0, 3)

    def test_normalised_counts_accumulate_across_updates(self):
        # Unlike scores/words, which each update() overwrites, these add up and
        # are drained by the callback instead.
        metric = _wer_metric(["le chat"], ["le chien"])
        metric.track_normalised = True
        _update(metric, ["le chat"])
        _update(metric, ["le chat"])
        assert (metric.normalised_scores, metric.normalised_words) == (2, 4)

    def test_tracking_uses_the_metrics_own_cer_setting(self):
        metric = _wer_metric(["chat"], ["chats"], use_cer=True)
        metric.track_normalised = True
        _update(metric, ["chat"])
        assert (metric.normalised_scores, metric.normalised_words) == (1, 4)


class _FakeMetric:
    def __init__(self):
        self.track_normalised = False
        self.normalised_scores = 0
        self.normalised_words = 0


class _FakeModule:
    def __init__(self, validation_names=None):
        self.wer = _FakeMetric()
        self._validation_names = validation_names


class _FakeLogger:
    def __init__(self):
        self.logged = []

    def log_metrics(self, metrics, step=None):
        self.logged.append((dict(metrics), step))


class _FakeTrainer:
    def __init__(self, sanity_checking=False, logger=True):
        self.callback_metrics = {}
        self.logger = _FakeLogger() if logger else None
        self.sanity_checking = sanity_checking
        self.global_step = 42


def _run_validation(callback, trainer, module, batches):
    """Drive the callback's hooks over (dataloader_idx, scores, words) batches."""
    callback.on_validation_start(trainer, module)
    for dataloader_idx, scores, words in batches:
        module.wer.normalised_scores += scores
        module.wer.normalised_words += words
        callback.on_validation_batch_end(trainer, module, None, None, 0, dataloader_idx)
    callback.on_validation_end(trainer, module)


class TestNormalisedWERCallback:
    def test_logs_one_metric_per_validation_set(self):
        trainer, module = _FakeTrainer(), _FakeModule(["oli", "fleurs_fr"])
        _run_validation(_NormalisedWERCallback(), trainer, module, [(0, 1, 10), (1, 3, 10)])
        assert trainer.callback_metrics["olival_wer_norm"].item() == pytest.approx(0.1)
        assert trainer.callback_metrics["fleurs_frval_wer_norm"].item() == pytest.approx(0.3)

    def test_counts_are_pooled_across_batches_of_one_set(self):
        # Micro-average, matching how NeMo pools the raw WER: 3 errors over 30
        # words, not the mean of the two per-batch rates.
        trainer, module = _FakeTrainer(), _FakeModule(["oli"])
        _run_validation(_NormalisedWERCallback(), trainer, module, [(0, 1, 10), (0, 2, 20)])
        assert trainer.callback_metrics["olival_wer_norm"].item() == pytest.approx(3 / 30)

    def test_a_batch_does_not_leak_into_the_next_dataloader(self):
        # The regression that motivated draining the counters per batch. Without
        # the reset, set two would be charged for set one's errors as well.
        trainer, module = _FakeTrainer(), _FakeModule(["oli", "fleurs_fr"])
        _run_validation(_NormalisedWERCallback(), trainer, module, [(0, 5, 10), (1, 0, 10)])
        assert trainer.callback_metrics["fleurs_frval_wer_norm"].item() == pytest.approx(0.0)

    def test_names_come_from_the_model_not_a_config_copy(self):
        trainer, module = _FakeTrainer(), _FakeModule(["renamed_"])
        _run_validation(_NormalisedWERCallback(), trainer, module, [(0, 1, 4)])
        assert "renamed_val_wer_norm" in trainer.callback_metrics

    def test_tracking_is_switched_off_after_validation(self):
        # Training batches call update() too; leaving this on would pay for
        # normalisation on every one of them for a number nobody reads.
        trainer, module = _FakeTrainer(), _FakeModule(["oli"])
        _run_validation(_NormalisedWERCallback(), trainer, module, [(0, 1, 4)])
        assert module.wer.track_normalised is False

    def test_an_empty_set_is_skipped_rather_than_dividing_by_zero(self):
        trainer, module = _FakeTrainer(), _FakeModule(["oli", "empty"])
        _run_validation(_NormalisedWERCallback(), trainer, module, [(0, 1, 10), (1, 0, 0)])
        assert "emptyval_wer_norm" not in trainer.callback_metrics
        assert "olival_wer_norm" in trainer.callback_metrics

    def test_an_unnamed_dataloader_is_skipped(self):
        trainer, module = _FakeTrainer(), _FakeModule(["oli"])
        _run_validation(_NormalisedWERCallback(), trainer, module, [(0, 1, 10), (1, 1, 10)])
        assert list(trainer.callback_metrics) == ["olival_wer_norm"]

    def test_sanity_check_results_are_not_written_to_the_logger(self):
        # Lightning suppresses logger writes during sanity checking; writing
        # anyway would put a point at step 0 that no raw curve has.
        trainer = _FakeTrainer(sanity_checking=True)
        _run_validation(_NormalisedWERCallback(), trainer, _FakeModule(["oli"]), [(0, 1, 10)])
        assert trainer.logger.logged == []
        # Still available to the checkpoint monitor and the macro callback.
        assert "olival_wer_norm" in trainer.callback_metrics

    def test_results_are_logged_at_the_current_step(self):
        trainer = _FakeTrainer()
        _run_validation(_NormalisedWERCallback(), trainer, _FakeModule(["oli"]), [(0, 1, 10)])
        assert trainer.logger.logged == [({"olival_wer_norm": pytest.approx(0.1)}, 42)]

    def test_a_second_validation_round_does_not_reuse_the_first(self):
        trainer, module = _FakeTrainer(), _FakeModule(["oli"])
        callback = _NormalisedWERCallback()
        _run_validation(callback, trainer, module, [(0, 5, 10)])
        _run_validation(callback, trainer, module, [(0, 1, 10)])
        assert trainer.callback_metrics["olival_wer_norm"].item() == pytest.approx(0.1)

    def test_a_module_without_a_wer_metric_is_ignored(self):
        trainer, module = _FakeTrainer(), _FakeModule(["oli"])
        module.wer = None
        callback = _NormalisedWERCallback()
        _run_validation(callback, trainer, module, [])
        assert trainer.callback_metrics == {}

    def test_macro_twins_average_the_normalised_metrics(self):
        # Both callbacks write in on_validation_end, so trainer.callbacks order
        # decides which sees the other's output. _build inserts the macro
        # callback at 0 and then this one at 0, leaving [norm, macro, ...,
        # checkpoint]; a swap would leave the macros averaging metrics that are
        # not there yet, which only shows up as a warning in the log.
        trainer, module = _FakeTrainer(), _FakeModule(["oli", "fleurs_fr"])
        norm = _NormalisedWERCallback()
        macro = _MacroMetricCallback(
            [{"name": "combined_norm", "sources": ["olival_wer_norm", "fleurs_frval_wer_norm"]}]
        )
        _run_validation(norm, trainer, module, [(0, 1, 10), (1, 3, 10)])
        macro.on_validation_end(trainer, module)
        assert trainer.callback_metrics["combined_norm"].item() == pytest.approx(0.2)

    def test_a_model_with_no_validation_names_still_logs(self):
        # Single validation set: NeMo logs a bare "val_wer", so the normalised
        # twin must be "val_wer_norm" rather than crashing on a missing name.
        trainer, module = _FakeTrainer(), _FakeModule(None)
        _run_validation(_NormalisedWERCallback(), trainer, module, [(0, 1, 10)])
        assert "val_wer_norm" in trainer.callback_metrics
