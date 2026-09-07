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

"""Tests for the config audit.

The audit is itself a piece of safety equipment, so the interesting tests are
the ones covering the two ways it silently stops working: crediting a read that
came from a different config tree, and crediting a subtree that was only
traversed. Both bugs make it pass everything, which is the failure mode nobody
notices.

Written with Claude Code.
"""

import types

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.parts.utils.config_audit import (
    ConfigAuditError,
    assert_all_consumed,
    assert_effective_values,
    track_reads,
)


class TestTrackReads:
    def test_records_attr_item_and_get_access(self):
        cfg = OmegaConf.create({"a": 1, "b": 2, "c": 3, "d": 4})
        with track_reads(cfg) as seen:
            cfg.a
            cfg["b"]
            cfg.get("c")
        assert {"a", "b", "c"} <= seen
        assert "d" not in seen

    def test_records_nested_paths_in_dotted_form(self):
        cfg = OmegaConf.create({"model": {"optim": {"lr": 1e-4}}})
        with track_reads(cfg) as seen:
            cfg.model.optim.lr
        assert seen == {"model", "model.optim", "model.optim.lr"}

    def test_ignores_reads_from_a_different_config_tree(self):
        # The regression test that matters most. A restored checkpoint carries
        # its own config whose keys are named exactly like ours, and
        # _get_full_key reports a path relative to whatever tree the node is
        # in. Without the root check, reading the checkpoint's
        # decoding.strategy marks ours as consumed and the audit goes blind.
        ours = OmegaConf.create({"decoding": {"strategy": "greedy_batch"}})
        theirs = OmegaConf.create({"decoding": {"strategy": "beam"}})
        with track_reads(ours) as seen:
            theirs.decoding.strategy
        assert seen == set(), f"reads from a foreign config leaked in: {seen}"

    def test_restores_omegaconf_even_when_the_body_raises(self):
        cfg = OmegaConf.create({"a": 1})
        before = (DictConfig.__getattr__, DictConfig.__getitem__, DictConfig.get)
        with pytest.raises(ValueError):
            with track_reads(cfg):
                raise ValueError("boom")
        after = (DictConfig.__getattr__, DictConfig.__getitem__, DictConfig.get)
        assert before == after, "OmegaConf left patched, the leak would follow into training"


class TestAssertAllConsumed:
    def test_passes_when_every_leaf_was_read(self):
        cfg = OmegaConf.create({"a": 1, "b": {"c": 2}})
        with track_reads(cfg) as seen:
            cfg.a
            cfg.b.c
        assert_all_consumed(cfg, seen)

    def test_flags_a_leaf_nothing_read(self):
        cfg = OmegaConf.create({"a": 1, "typoo": 2})
        with track_reads(cfg) as seen:
            cfg.a
        with pytest.raises(ConfigAuditError, match="typoo"):
            assert_all_consumed(cfg, seen)

    def test_traversing_a_subtree_does_not_excuse_its_other_keys(self):
        # `cfg.model.train_ds` records a read of `model` on the way past. If
        # that counted as consuming the subtree, every other key under `model`
        # would be excused and the audit would be worthless. This is exactly
        # what the first version of the check did.
        cfg = OmegaConf.create({"model": {"seed": 42, "never_read": 1}})
        with track_reads(cfg) as seen:
            cfg.model.seed
        assert "model" in seen, "precondition: traversal does record the parent"
        with pytest.raises(ConfigAuditError, match="model.never_read"):
            assert_all_consumed(cfg, seen)

    def test_consuming_a_subtree_whole_excuses_its_keys(self):
        # The mirror case: a block handed to a callback in one piece, e.g.
        # OmegaConf.to_container(cfg.macro_metrics). Nothing below it is ever
        # subscripted, so nothing below it can be required.
        cfg = OmegaConf.create({"macro_metrics": {"name": "x", "sources": ["a"]}})
        with track_reads(cfg) as seen:
            OmegaConf.to_container(cfg.get("macro_metrics"), resolve=True)
        assert_all_consumed(cfg, seen)

    def test_allow_unused_exact_path(self):
        cfg = OmegaConf.create({"a": 1, "off": 2})
        with track_reads(cfg) as seen:
            cfg.a
        assert_all_consumed(cfg, seen, allow_unused=["off"])

    def test_allow_unused_glob_covers_a_subtree(self):
        cfg = OmegaConf.create({"a": 1, "encoder_cache": {"x": 1, "y": {"z": 2}}})
        with track_reads(cfg) as seen:
            cfg.a
        assert_all_consumed(cfg, seen, allow_unused=["encoder_cache.*"])

    def test_allow_unused_glob_does_not_leak_to_a_sibling_prefix(self):
        cfg = OmegaConf.create({"a": 1, "encoder_cache_dir": "/x"})
        with track_reads(cfg) as seen:
            cfg.a
        with pytest.raises(ConfigAuditError, match="encoder_cache_dir"):
            assert_all_consumed(cfg, seen, allow_unused=["encoder_cache.*"])

    def test_delegated_subtrees_are_not_required(self):
        cfg = OmegaConf.create({"trainer": {"max_epochs": 12, "precision": "bf16-mixed"}})
        with track_reads(cfg) as seen:
            pass
        assert_all_consumed(cfg, seen)

    def test_the_audits_own_block_is_self_exempt(self):
        # config_audit.enabled has to be read before tracking can start, so it
        # can never appear in `seen`. If it were not exempt, every run would
        # fail the audit.
        cfg = OmegaConf.create({"a": 1, "config_audit": {"enabled": True, "allow_unused": []}})
        with track_reads(cfg) as seen:
            cfg.a
        assert_all_consumed(cfg, seen)


class _FakeGreedy:
    def __init__(self, max_symbols):
        self.max_symbols = max_symbols


class _FakeDecoding:
    def __init__(self, cfg):
        self.cfg = cfg
        self.decoding = _FakeGreedy(cfg.get("greedy", {}).get("max_symbols"))


class _FakeModel(torch.nn.Module):
    """Just enough surface for assert_effective_values to inspect."""

    def __init__(self, decoding_cfg=None, weight_decay=0.01, lr=2e-5):
        super().__init__()
        self.encoder = torch.nn.Module()
        if decoding_cfg is not None:
            self.decoding = _FakeDecoding(decoding_cfg)
        self._param = torch.nn.Parameter(torch.zeros(2))
        self._optimizer = torch.optim.AdamW([self._param], lr=lr, weight_decay=weight_decay)
        self._optimizer.param_groups[0]["initial_lr"] = lr
        self._train_dl = None


class _FakeTrainDataLoader:
    """Only len() matters here; the rest is what the dataloader checks read."""

    def __init__(self, n_batches):
        self.n_batches = n_batches
        self.num_workers = None
        self.persistent_workers = None
        self.prefetch_factor = None

    def __len__(self):
        return self.n_batches


class _FakeModelCheckpoint:
    def __init__(self, dirpath):
        self.dirpath = dirpath


class _FakeOtherCallback:
    """An exp_manager callback that happens to carry an unrelated dirpath."""

    def __init__(self, dirpath):
        self.dirpath = dirpath


class _FakeTrainer:
    def __init__(self, dirpath=None, log_dir=None, decoy_dirpath=None):
        self.callbacks = []
        if decoy_dirpath:
            self.callbacks.append(_FakeOtherCallback(decoy_dirpath))
        if dirpath:
            self.callbacks.append(_FakeModelCheckpoint(dirpath))
        self.log_dir = log_dir


def _minimal_cfg(**overrides):
    base = {
        "model": {
            "optim": {"lr": 2e-5, "weight_decay": 0.01},
            "train_ds": {},
            "validation_ds": {"ds_item": [{"name": "fleurs_fr"}]},
        },
        "trainer": {},
        "exp_manager": {"checkpoint_callback_params": {"monitor": "x"}},
    }
    cfg = OmegaConf.create(base)
    for path, value in overrides.items():
        OmegaConf.update(cfg, path, value, force_add=True)
    return cfg


class TestAssertEffectiveValues:
    def test_passes_when_the_live_objects_match(self):
        cfg = _minimal_cfg()
        assert_effective_values(cfg, _FakeModel(), trainer=None)

    def test_catches_a_weight_decay_the_optimizer_did_not_take(self):
        cfg = _minimal_cfg()
        model = _FakeModel(weight_decay=0.0)
        with pytest.raises(ConfigAuditError, match="weight_decay"):
            assert_effective_values(cfg, model, trainer=None)

    def test_lr_is_checked_against_initial_lr_not_the_warmed_up_value(self):
        # A scheduler has already moved param_groups[0]["lr"] to its warmup
        # start (base_lr / warmup_steps) by the time the audit runs. Comparing
        # against that would fail every single run with warmup enabled.
        cfg = _minimal_cfg()
        model = _FakeModel(lr=2e-5)
        model._optimizer.param_groups[0]["lr"] = 2e-5 / 800
        assert_effective_values(cfg, model, trainer=None)

    def test_catches_a_decoding_strategy_the_model_ignored(self):
        cfg = _minimal_cfg(decoding={"strategy": "greedy_batch"})
        model = _FakeModel(decoding_cfg=OmegaConf.create({"strategy": "beam"}))
        with pytest.raises(ConfigAuditError, match="decoding.strategy"):
            assert_effective_values(cfg, model, trainer=None)

    def test_checks_max_symbols_on_the_decoder_object_not_its_config(self):
        # The greedy decoder enforces max_symbols; its config is only what it
        # was built from. Checking the config would pass even if the decoder
        # were built with something else.
        cfg = _minimal_cfg(decoding={"strategy": "greedy_batch", "greedy": {"max_symbols": 10}})
        live = OmegaConf.create({"strategy": "greedy_batch", "greedy": {"max_symbols": 10}})
        model = _FakeModel(decoding_cfg=live)
        model.decoding.decoding.max_symbols = 3
        with pytest.raises(ConfigAuditError, match="max_symbols"):
            assert_effective_values(cfg, model, trainer=None)

    def test_catches_a_macro_source_that_matches_no_validation_set(self):
        # _MacroMetricCallback skips the whole macro when a source is missing,
        # which would leave ModelCheckpoint monitoring a metric never written.
        cfg = _minimal_cfg(
            macro_metrics=[{"name": "combined", "sources": ["fleurs_frval_wer", "ghostval_wer"]}]
        )
        with pytest.raises(ConfigAuditError, match="ghostval_wer"):
            assert_effective_values(cfg, _FakeModel(), trainer=None)

    def test_accepts_macro_sources_that_all_resolve(self):
        cfg = _minimal_cfg(macro_metrics=[{"name": "combined", "sources": ["fleurs_frval_wer"]}])
        assert_effective_values(cfg, _FakeModel(), trainer=None)

    def test_catches_a_partial_freeze_that_did_not_take(self):
        cfg = _minimal_cfg(**{"model.freeze": {"encoder": True, "encoder_except_last_n": 2}})
        model = _FakeModel()
        model.encoder.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(4)])
        for p in model.encoder.parameters():  # everything frozen, nothing left trainable
            p.requires_grad_(False)
        with pytest.raises(ConfigAuditError, match="encoder_except_last_n"):
            assert_effective_values(cfg, model, trainer=None)

    def test_catches_a_run_directory_that_lost_the_run_name(self):
        # exp_manager.name is ${name}, resolved by OmegaConf through _get_node,
        # so layer 1 never sees the read. This check is the only thing standing
        # between a broken interpolation and a run that quietly lands under
        # <exp_dir>/default/ where the next resume will not find it.
        cfg = _minimal_cfg(**{"exp_manager.name": "parakeet-tdt-ultimed-french-medical"})
        trainer = _FakeTrainer(dirpath="/media/disk/nemo_experiments/default/1.1.0/checkpoints")
        with pytest.raises(ConfigAuditError, match="does not contain it"):
            assert_effective_values(cfg, _FakeModel(), trainer)

    def test_accepts_a_run_directory_that_carries_the_run_name(self):
        cfg = _minimal_cfg(**{"exp_manager.name": "parakeet-tdt-ultimed-french-medical"})
        trainer = _FakeTrainer(
            dirpath="/media/disk/nemo_experiments/parakeet-tdt-ultimed-french-medical/1.1.0/checkpoints"
        )
        assert_effective_values(cfg, _FakeModel(), trainer)

    def test_run_directory_check_prefers_the_checkpoint_dirpath(self):
        # log_dir is the run directory; dirpath is where checkpoints actually
        # go and is what resume_if_exists searches, so it wins.
        cfg = _minimal_cfg(**{"exp_manager.name": "runname"})
        trainer = _FakeTrainer(dirpath="/exp/default/1.1.0/checkpoints", log_dir="/exp/runname/1.1.0")
        with pytest.raises(ConfigAuditError, match="exp_manager.name"):
            assert_effective_values(cfg, _FakeModel(), trainer)

    def test_run_directory_ignores_a_non_checkpoint_callbacks_dirpath(self):
        # exp_manager attaches several callbacks. Taking whichever carries a
        # dirpath first would check some unrelated directory and pass while the
        # checkpoints go somewhere else entirely.
        cfg = _minimal_cfg(**{"exp_manager.name": "runname"})
        trainer = _FakeTrainer(
            decoy_dirpath="/somewhere/runname/else",
            dirpath="/exp/default/1.1.0/checkpoints",
        )
        with pytest.raises(ConfigAuditError, match="exp_manager.name"):
            assert_effective_values(cfg, _FakeModel(), trainer)

    def test_run_directory_check_is_skipped_when_the_name_is_unset(self):
        cfg = _minimal_cfg()
        trainer = _FakeTrainer(dirpath="/exp/default/1.1.0/checkpoints")
        assert_effective_values(cfg, _FakeModel(), trainer)

    @staticmethod
    def _sched_cfg(sched_max_steps, n_batches=8000, accum=8, **trainer):
        # 8000 batches / accum 8 = 1000 optimizer steps per epoch.
        cfg = _minimal_cfg(**{"model.optim.sched": {"max_steps": sched_max_steps}})
        cfg.trainer.accumulate_grad_batches = accum
        for key, value in trainer.items():
            cfg.trainer[key] = value
        model = _FakeModel()
        model._train_dl = _FakeTrainDataLoader(n_batches)
        return cfg, model

    def test_accepts_a_cosine_that_spans_the_epoch_budget(self):
        cfg, model = self._sched_cfg(3000, max_epochs=3)
        assert_effective_values(cfg, model, trainer=None)

    def test_catches_a_cosine_that_does_not_span_the_epoch_budget(self):
        cfg, model = self._sched_cfg(1000, max_epochs=3)
        with pytest.raises(ConfigAuditError, match="max_epochs"):
            assert_effective_values(cfg, model, trainer=None)

    def test_a_step_capped_run_is_measured_against_max_steps(self):
        # Lightning stops at whichever limit lands first, so a 1.5-epoch cosine
        # under a 1.5-epoch step cap is coherent even though max_epochs is 3.
        cfg, model = self._sched_cfg(1500, max_epochs=3, max_steps=1500)
        assert_effective_values(cfg, model, trainer=None)

    def test_catches_a_cosine_that_outruns_the_step_cap(self):
        cfg, model = self._sched_cfg(3000, max_epochs=3, max_steps=1500)
        with pytest.raises(ConfigAuditError, match="max_steps"):
            assert_effective_values(cfg, model, trainer=None)

    def test_the_epoch_budget_still_binds_when_it_is_the_smaller_limit(self):
        # max_steps set high enough to never be reached: max_epochs decides.
        cfg, model = self._sched_cfg(9000, max_epochs=3, max_steps=99999)
        with pytest.raises(ConfigAuditError, match="max_epochs"):
            assert_effective_values(cfg, model, trainer=None)

    def test_catches_a_live_scheduler_annealing_over_a_different_horizon(self):
        # NeMo replaces sched.max_steps with trainer.max_steps in
        # modelPT.setup_optimization; if the script's restore does not take,
        # the config's cosine is a fiction. Config coherence is fine here
        # (sched 3000 = 3 epochs = max_epochs), only the live object is wrong.
        cfg, model = self._sched_cfg(3000, max_epochs=3)
        model._scheduler = {"scheduler": types.SimpleNamespace(max_steps=1500)}
        with pytest.raises(ConfigAuditError, match="live scheduler"):
            assert_effective_values(cfg, model, trainer=None)

    def test_accepts_a_live_scheduler_matching_the_config(self):
        cfg, model = self._sched_cfg(3000, max_epochs=3)
        model._scheduler = {"scheduler": types.SimpleNamespace(max_steps=3000)}
        assert_effective_values(cfg, model, trainer=None)

    def test_the_truncated_arm_needs_the_live_scheduler_to_keep_the_cosine(self):
        # The full A/B contract: acknowledged truncation AND the live
        # scheduler still annealing over the reference horizon. If the
        # restore failed, the acknowledgement must not paper over it.
        cfg, model = self._sched_cfg(3000, max_epochs=3, max_steps=1500)
        cfg.config_audit = {"allow_truncated_anneal": True}
        model._scheduler = {"scheduler": types.SimpleNamespace(max_steps=1500)}
        with pytest.raises(ConfigAuditError, match="live scheduler"):
            assert_effective_values(cfg, model, trainer=None)
        model._scheduler = {"scheduler": types.SimpleNamespace(max_steps=3000)}
        assert_effective_values(cfg, model, trainer=None)

    def test_an_acknowledged_truncated_anneal_passes(self):
        # The A/B design: keep the reference run's cosine (3000) so the LR
        # trajectory matches point for point, truncate the run at 1500.
        cfg, model = self._sched_cfg(3000, max_epochs=3, max_steps=1500)
        cfg.config_audit = {"allow_truncated_anneal": True}
        assert_effective_values(cfg, model, trainer=None)

    def test_acknowledgement_does_not_excuse_a_cosine_shorter_than_the_run(self):
        # The other direction parks the tail at min_lr, which no comparison
        # design calls for, so the flag must not silence it.
        cfg, model = self._sched_cfg(1000, max_epochs=3)
        cfg.config_audit = {"allow_truncated_anneal": True}
        with pytest.raises(ConfigAuditError, match="max_epochs"):
            assert_effective_values(cfg, model, trainer=None)

    def test_an_unacknowledged_truncation_still_fails_and_names_the_hatch(self):
        cfg, model = self._sched_cfg(3000, max_epochs=3, max_steps=1500)
        with pytest.raises(ConfigAuditError, match="allow_truncated_anneal"):
            assert_effective_values(cfg, model, trainer=None)

    def test_lightnings_unset_max_steps_sentinel_is_ignored(self):
        # -1 means "no step limit"; treating it as a real cap would make every
        # cosine look like it overruns by its whole length.
        cfg, model = self._sched_cfg(3000, max_epochs=3, max_steps=-1)
        assert_effective_values(cfg, model, trainer=None)

    def test_accepts_a_partial_freeze_that_took(self):
        cfg = _minimal_cfg(**{"model.freeze": {"encoder": True, "encoder_except_last_n": 2}})
        model = _FakeModel()
        model.encoder.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(4)])
        for layer in model.encoder.layers[:2]:
            for p in layer.parameters():
                p.requires_grad_(False)
        assert_effective_values(cfg, model, trainer=None)
