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

"""Fail fast when a config key does not do what the YAML says it does.

Written after six separate settings in perso/training_config.yaml turned out
to be inert, four of them discovered only by reading the code that was
supposed to consume them:

  * ``optim.weight_decay``          read correctly, but AdamSPD's step() never
                                    applied it (an inverted sign, see
                                    tests/collections/asr/test_adam_spd.py)
  * top-level ``decoding``          read by nothing; the effective strategy was
                                    silently inherited from the .nemo
  * ``cache_variants_per_sample``   at the top level, but its consumer reads
                                    ``model.cache_variants_per_sample``, so it
                                    silently fell back to 1 instead of 6
  * ``model.seed``                  no consumer anywhere; the run was unseeded
  * ``bucketing_strategy``          tarred-datasets only, inert otherwise
  * ``persistent_workers``          accepted by our loader, dropped by NeMo's

Three distinct failure modes hide in that list, and they need different
guards:

  1. NEVER READ.  A key sits at the wrong nesting level, or is a typo, or
     nothing consumes it. :func:`track_reads` plus :func:`assert_all_consumed`
     catch these.
  2. READ THEN DROPPED.  The key is read and handed onward, and the thing
     downstream ignores it. Read-tracking sees a read and stays quiet, so
     :func:`assert_effective_values` re-reads the value back off the live
     object instead.
  3. WIRED BUT INERT.  The value reaches its destination and the maths there
     ignores it. No config-level check can see this; only a behavioural test
     can. That is what test_adam_spd.py is for.

Both checks here run before ``trainer.fit()``, so a false positive costs a
restart rather than a multi-day run.

Written with Claude Code.
"""

from contextlib import contextmanager
from typing import Any, List, Optional, Sequence, Set, Tuple

from omegaconf import DictConfig

from nemo.utils import logging

__all__ = [
    "track_reads",
    "assert_all_consumed",
    "assert_effective_values",
    "ConfigAuditError",
]


class ConfigAuditError(RuntimeError):
    """Raised when the config does not mean what it says."""


# Subtrees handed to a consumer wholesale (NeMo, Lightning, hydra.instantiate)
# rather than key by key. Requiring a recorded read of every leaf underneath
# these would be all false positives, because the consumer may iterate or
# serialise the block instead of subscripting it. Anything OUTSIDE them is
# custom to this fork and must be read explicitly by name.
#
# This is the deliberate blind spot of layer 1, and exactly why layer 2 exists:
# assert_effective_values reaches into these subtrees and checks the settings
# that matter actually landed.
DELEGATED_SUBTREES: Tuple[str, ...] = (
    "trainer",
    "exp_manager",
    "model.train_ds",
    "model.validation_ds",
    "model.test_ds",
    "model.optim",
    "model.spec_augment",
    "model.tokenizer",
    "model.joint",
    "model.decoder",
    "model.encoder",
    "model.preprocessor",
)

# The audit's own settings. They have to be read before tracking can start, so
# they can never appear in `seen`, and flagging them would make every run fail.
SELF_KEY = "config_audit"


def _leaf_paths(cfg: Any, prefix: str = "") -> List[str]:
    """Every scalar leaf path in dotted form, lists treated as leaves."""
    out: List[str] = []
    if isinstance(cfg, DictConfig):
        for key in cfg.keys():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            try:
                child = cfg._get_node(key)
            except Exception:  # pragma: no cover - defensive
                out.append(child_prefix)
                continue
            if isinstance(child, DictConfig):
                out.extend(_leaf_paths(child, child_prefix))
            else:
                # ListConfig counts as a leaf: a list of dataset entries is
                # consumed as a unit, not path by path.
                out.append(child_prefix)
    return out


def _is_delegated(path: str) -> bool:
    return any(path == root or path.startswith(root + ".") for root in DELEGATED_SUBTREES)


def _matches(path: str, pattern: str) -> bool:
    """Glob-lite: trailing ``.*`` matches a whole subtree, otherwise exact."""
    if pattern.endswith(".*"):
        root = pattern[:-2]
        return path == root or path.startswith(root + ".")
    return path == pattern


@contextmanager
def track_reads(root: DictConfig):
    """Record every path read from ``root`` while the block runs.

    Patches the three DictConfig entry points a normal ``cfg.x`` / ``cfg["x"]``
    / ``cfg.get("x")`` read goes through. Restored on exit even if the body
    raises, since leaving OmegaConf patched would follow the process into
    training.

    The ``root`` filter is not optional. ``_get_full_key`` reports a path
    relative to whatever tree the node belongs to, and the model restored by
    ``from_pretrained`` carries its own config with keys named exactly like
    ours. Without the identity check, a read of the *checkpoint's*
    ``decoding.strategy`` would mark our top-level ``decoding.strategy`` as
    consumed, which is precisely the bug this module was written to catch.
    """
    seen: Set[str] = set()

    original_getattr = DictConfig.__getattr__
    original_getitem = DictConfig.__getitem__
    original_get = DictConfig.get

    def _record(node: DictConfig, key: Any) -> None:
        try:
            if node._get_root() is not root:
                return
            full = node._get_full_key(str(key))
        except Exception:  # pragma: no cover - defensive
            return
        if full:
            seen.add(str(full))

    def patched_getattr(self, key):
        _record(self, key)
        return original_getattr(self, key)

    def patched_getitem(self, key):
        _record(self, key)
        return original_getitem(self, key)

    def patched_get(self, key, default_value=None):
        _record(self, key)
        return original_get(self, key, default_value)

    DictConfig.__getattr__ = patched_getattr
    DictConfig.__getitem__ = patched_getitem
    DictConfig.get = patched_get
    try:
        yield seen
    finally:
        DictConfig.__getattr__ = original_getattr
        DictConfig.__getitem__ = original_getitem
        DictConfig.get = original_get


def assert_all_consumed(
    cfg: DictConfig,
    seen: Set[str],
    allow_unused: Optional[Sequence[str]] = None,
) -> None:
    """Raise if any non-delegated config leaf was never read.

    Parameters
    ----------
    cfg
        The root config the run was configured from.
    seen
        Paths recorded by :func:`track_reads`.
    allow_unused
        Paths (or ``subtree.*`` globs) that are legitimately unread, typically
        because they belong to a feature switched off for this run. Each entry
        is a deliberate, reviewable exemption rather than a blanket silence.
    """
    allow_unused = list(allow_unused or [])

    # Reading a subtree means one of two very different things, and crediting
    # them alike would gut this check. `cfg.model.train_ds` records a read of
    # `model` on the way past, which must NOT excuse every other key under
    # `model`. But `cfg.macro_metrics`, handed whole to a callback, really does
    # consume everything beneath it.
    #
    # The two are distinguishable after the fact: a subtree that was merely
    # traversed has recorded reads below it, a subtree consumed whole does not.
    wholesale = {p for p in seen if not any(q.startswith(p + ".") for q in seen)}

    unread: List[str] = []
    for path in _leaf_paths(cfg):
        if _is_delegated(path) or path in seen:
            continue
        if path == SELF_KEY or path.startswith(SELF_KEY + "."):
            continue
        if any(path.startswith(parent + ".") for parent in wholesale):
            continue
        if any(_matches(path, pattern) for pattern in allow_unused):
            continue
        unread.append(path)

    if unread:
        lines = "\n".join(f"    {p}" for p in sorted(unread))
        raise ConfigAuditError(
            f"{len(unread)} config key(s) were never read by anything:\n{lines}\n\n"
            "  A key nothing reads does nothing, however carefully it is set.\n"
            "  Usually this means the key sits at the wrong nesting level, is\n"
            "  misspelled, or its consumer was removed.\n\n"
            "  If a key is unread on purpose (typically a feature switched off for\n"
            "  this run), add its path to the top-level `config_audit.allow_unused`\n"
            "  list in the YAML, with a comment saying why. A trailing '.*' exempts\n"
            "  a whole subtree."
        )

    logging.info(f"Config audit: all {len(_leaf_paths(cfg))} leaf keys accounted for.")


def _run_directory(trainer) -> Optional[str]:
    """Where this run will actually write, preferring the checkpoint dirpath.

    The checkpoint callback's dirpath is the one that matters: it is what
    resume_if_exists searches on the next launch. exp_manager attaches several
    other callbacks, so match the checkpoint one by type rather than taking
    whichever happens to carry a dirpath first.
    """
    with_dirpath = [
        cb for cb in getattr(trainer, "callbacks", None) or []
        if getattr(cb, "dirpath", None)
    ]
    for callback in with_dirpath:
        if "Checkpoint" in type(callback).__name__:
            return str(callback.dirpath)
    if with_dirpath:
        return str(with_dirpath[0].dirpath)
    log_dir = getattr(trainer, "log_dir", None)
    return str(log_dir) if log_dir else None


def _check(failures: List[str], label: str, configured: Any, effective: Any) -> None:
    if configured is None:
        return
    if configured != effective:
        failures.append(f"    {label}: config says {configured!r}, live object has {effective!r}")


def assert_effective_values(cfg: DictConfig, asr_model, trainer) -> None:
    """Re-read the settings that matter back off the objects that were built.

    Layer 1 cannot see a key that is read and then discarded downstream, and it
    deliberately does not descend into DELEGATED_SUBTREES at all. This closes
    both gaps for the handful of settings whose silent loss would actually cost
    a run.
    """
    failures: List[str] = []

    # --- decoding -----------------------------------------------------------
    # The model is restored by ASRModel.from_pretrained(), which brings the
    # checkpoint's own decoding config with it, so this is the check that the
    # explicit change_decoding_strategy call actually took.
    decoding_cfg = cfg.get("decoding", None)
    if decoding_cfg is not None and hasattr(asr_model, "decoding"):
        live = getattr(asr_model.decoding, "cfg", None)
        if live is not None:
            _check(failures, "decoding.strategy", decoding_cfg.get("strategy"), live.get("strategy"))
            _check(failures, "decoding.model_type", decoding_cfg.get("model_type"), live.get("model_type"))
            configured_durations = decoding_cfg.get("durations", None)
            if configured_durations is not None:
                _check(failures, "decoding.durations",
                       list(configured_durations), list(live.get("durations") or []))
            configured_symbols = decoding_cfg.get("greedy", {}).get("max_symbols", None)
            if configured_symbols is not None:
                # The greedy decoder object is what actually enforces this, not
                # the config it was built from.
                _check(failures, "decoding.greedy.max_symbols", configured_symbols,
                       getattr(asr_model.decoding.decoding, "max_symbols", None))

    # --- optimizer ----------------------------------------------------------
    optim_cfg = cfg.model.get("optim", None)
    optimizer = getattr(asr_model, "_optimizer", None)
    if optim_cfg is not None and optimizer is not None:
        group = optimizer.param_groups[0]
        # Not group["lr"]: the LR scheduler has already been constructed and has
        # set the group to its warmup starting point (base_lr / warmup_steps).
        # PyTorch stashes the value the optimizer was actually built with in
        # `initial_lr`, which is the one that has to match the config.
        _check(failures, "optim.lr", optim_cfg.get("lr"), group.get("initial_lr", group.get("lr")))
        _check(failures, "optim.weight_decay", optim_cfg.get("weight_decay"), group.get("weight_decay"))

    # --- partial freeze -----------------------------------------------------
    freeze_cfg = cfg.model.get("freeze", None)
    if freeze_cfg is not None and freeze_cfg.get("encoder", False):
        keep = int(freeze_cfg.get("encoder_except_last_n", 0) or 0)
        layers = getattr(asr_model.encoder, "layers", None)
        if layers is not None:
            trainable = [i for i, layer in enumerate(layers)
                         if any(p.requires_grad for p in layer.parameters())]
            expected = list(range(len(layers) - keep, len(layers))) if keep else []
            if trainable != expected:
                failures.append(
                    f"    freeze.encoder_except_last_n={keep}: expected trainable encoder "
                    f"layers {expected}, found {trainable}"
                )

    # --- cost batching ------------------------------------------------------
    cost_cfg = cfg.model.train_ds.get("cost_batching", None)
    train_dl = getattr(asr_model, "_train_dl", None)
    if cost_cfg is not None and cost_cfg.get("enabled", False) and train_dl is not None:
        sampler = getattr(train_dl, "batch_sampler", None)
        if sampler is None or type(sampler).__name__ != "DurationCostBatchSampler":
            failures.append(
                "    train_ds.cost_batching.enabled is true but the train dataloader's "
                f"batch_sampler is {type(sampler).__name__ if sampler else None}"
            )
        else:
            _check(failures, "cost_batching.budget", cost_cfg.get("budget"), sampler.budget)
            _check(failures, "cost_batching.max_batch_size",
                   cost_cfg.get("max_batch_size"), sampler.max_batch_size)

    # --- dataloader knobs NeMo is known to drop -----------------------------
    if train_dl is not None:
        train_ds = cfg.model.train_ds
        _check(failures, "train_ds.num_workers", train_ds.get("num_workers"), train_dl.num_workers)
        if train_ds.get("num_workers", 0):
            _check(failures, "train_ds.persistent_workers",
                   train_ds.get("persistent_workers"), train_dl.persistent_workers)
            _check(failures, "train_ds.prefetch_factor",
                   train_ds.get("prefetch_factor"), train_dl.prefetch_factor)

    # --- the monitored macro must be computable -----------------------------
    val_names = {ds.get("name") for ds in cfg.model.validation_ds.get("ds_item", [])}
    monitor = cfg.exp_manager.checkpoint_callback_params.get("monitor", None)
    for spec in cfg.get("macro_metrics", []) or []:
        for source in spec.get("sources", []):
            # metric names are "<val set name>val_wer"
            if not any(source == f"{name}val_wer" for name in val_names if name):
                failures.append(
                    f"    macro_metrics '{spec.get('name')}' source '{source}' matches no "
                    f"validation set; _MacroMetricCallback would skip the whole macro"
                    + (f", leaving monitor '{monitor}' undefined" if spec.get('name') == monitor else "")
                )

    # --- the run directory carries the run name -----------------------------
    # exp_manager.name is an interpolation of the top-level `name`, and
    # OmegaConf resolves interpolations through _get_node, which layer 1 cannot
    # see. So this is the check that makes `name` a real setting rather than
    # decoration: if it ever stops reaching exp_manager, runs silently land
    # under <exp_dir>/default/ again, and the next launch's resume looks in the
    # wrong place and restarts from scratch without complaining.
    exp_name = cfg.exp_manager.get("name", None)
    if exp_name and trainer is not None:
        run_dir = _run_directory(trainer)
        if run_dir and exp_name not in run_dir:
            failures.append(
                f"    exp_manager.name is {exp_name!r} but this run writes to {run_dir!r}, "
                f"which does not contain it"
            )

    # --- schedule coherence -------------------------------------------------
    sched = cfg.model.optim.get("sched", {}) or {}
    sched_max_steps = sched.get("max_steps", None)
    max_epochs = cfg.trainer.get("max_epochs", None)
    # -1 is Lightning's "unset" for max_steps, and both limits may be given at
    # once: training then stops at whichever is reached first, so that one is
    # the length the cosine actually has to anneal over.
    trainer_max_steps = cfg.trainer.get("max_steps", None)
    accum = cfg.trainer.get("accumulate_grad_batches", 1) or 1
    if sched_max_steps and train_dl is not None:
        try:
            steps_per_epoch = len(train_dl) / accum
        except TypeError:  # pragma: no cover - iterable dataset has no len()
            steps_per_epoch = None

        limits = {}
        if trainer_max_steps and trainer_max_steps > 0:
            limits["trainer.max_steps"] = float(trainer_max_steps)
        if max_epochs and steps_per_epoch:
            limits["trainer.max_epochs"] = max_epochs * steps_per_epoch

        if limits and steps_per_epoch:
            binding, run_steps = min(limits.items(), key=lambda kv: kv[1])
            if abs(sched_max_steps - run_steps) > 0.5 * steps_per_epoch:
                # A/B escape hatch: an ablation arm that must REPLICATE a longer
                # run's LR trajectory deliberately keeps that run's cosine and
                # truncates with trainer.max_steps, so its mid-run validation
                # points compare like for like. Only the truncation direction is
                # excusable: a cosine SHORTER than the run parks the tail at
                # min_lr, which no comparison design calls for.
                acknowledged = bool(
                    (cfg.get(SELF_KEY, {}) or {}).get("allow_truncated_anneal", False)
                )
                if acknowledged and sched_max_steps > run_steps:
                    logging.info(
                        f"Config audit: cosine of {sched_max_steps} steps deliberately truncated "
                        f"at {run_steps:.0f} ({binding}); acknowledged via "
                        f"{SELF_KEY}.allow_truncated_anneal for LR-trajectory replication."
                    )
                else:
                    failures.append(
                        f"    optim.sched.max_steps={sched_max_steps} is "
                        f"{sched_max_steps / steps_per_epoch:.2f} epochs of cosine but the run stops "
                        f"after {run_steps:.0f} steps ({run_steps / steps_per_epoch:.2f} epochs), set by "
                        f"{binding}. These set the anneal SHAPE and the run length; a mismatch either "
                        f"truncates the anneal or parks the tail at min_lr. A deliberate truncation "
                        f"that replicates a longer run's LR trajectory (A/B arm) can be acknowledged "
                        f"with {SELF_KEY}.allow_truncated_anneal: true."
                    )

    # The live scheduler, not just the config: NeMo replaces sched.max_steps
    # with trainer.max_steps whenever the latter is set (modelPT), so the
    # config's number can stop describing the actual anneal without any key
    # going unread. The training script restores the explicit YAML value; this
    # verifies the restoration happened (and catches any future path that
    # rebuilds the scheduler without it).
    live_container = getattr(asr_model, "_scheduler", None)
    live_scheduler = live_container.get("scheduler") if isinstance(live_container, dict) else None
    live_max = getattr(live_scheduler, "max_steps", None)
    if sched_max_steps and live_max is not None and live_max != sched_max_steps:
        failures.append(
            f"    optim.sched.max_steps={sched_max_steps} but the live scheduler anneals over "
            f"{live_max} steps. NeMo overrode it (trainer.max_steps takes precedence in "
            f"modelPT.setup_optimization) and the training script's restore did not take."
        )

    if failures:
        raise ConfigAuditError(
            "Config values did not survive into the objects that were built:\n"
            + "\n".join(failures)
            + "\n\n  These keys were read, so they look fine to the unread-key check, but\n"
              "  something downstream overrode or discarded them."
        )

    logging.info("Config audit: effective values match the config.")
