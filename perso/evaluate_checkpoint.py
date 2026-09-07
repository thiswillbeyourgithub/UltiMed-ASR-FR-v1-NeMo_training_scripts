#!/usr/bin/env python
"""Score a model on the training config's validation sets, without training.

Two jobs, one script:

  * the pre-fine-tune baseline. Nothing measured it before 1.1.0 launched,
    because Lightning does not validate before the first training step and the
    sanity check does not log, so every WER curve so far has been unanchored:
    improvement and degradation were both guesses against an unknown zero.
  * comparing saved checkpoints afterwards, on identical footing.

Comparability is the whole point, so this reuses the training config rather
than restating it: the same validation manifests, the same decoding strategy,
the same precision, and the same macro definitions via the training script's
own callback. Numbers it prints drop straight onto the tensorboard curves.

Usage:

    # the pretrained model the fine-tune starts from
    python perso/evaluate_checkpoint.py

    # a saved run
    python perso/evaluate_checkpoint.py --checkpoint <path>.ckpt --label 1.1.0-step7500
    python perso/evaluate_checkpoint.py --checkpoint <path>.nemo

Results append to perso/eval_results.jsonl and print as a table. Written with
Claude Code.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.models import ASRModel
from nemo.utils import logging

# Reuse the training script's macro callback rather than reimplementing the
# averaging, so "combined_macro_val_wer" here means exactly what it means in
# the training curves. Importing it also applies that module's torch.load
# weights_only shim, which .ckpt loading below needs.
from examples.asr.speech_to_text_finetune_cached import _MacroMetricCallback


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="perso/training_config.yaml",
                   help="training config to take validation sets, decoding and macros from")
    p.add_argument("--checkpoint", default=None,
                   help=".ckpt or .nemo to score. Omitted = the pretrained model the fine-tune starts from")
    p.add_argument("--label", default=None,
                   help="name for this row in the results file (default: derived from --checkpoint)")
    p.add_argument("--out", default="perso/eval_results.jsonl",
                   help="results file, appended to")
    p.add_argument("--devices", default=1, type=int)
    return p.parse_args()


def load_model(cfg, checkpoint):
    """Return the model to score, and a label describing what it is."""
    pretrained = cfg.get("init_from_pretrained_model", None)

    if checkpoint is None:
        if not pretrained:
            raise SystemExit("No --checkpoint given and the config has no init_from_pretrained_model.")
        logging.info(f"Scoring the pretrained baseline: {pretrained}")
        return ASRModel.from_pretrained(model_name=pretrained), f"pretrained:{pretrained}"

    path = Path(checkpoint)
    if not path.exists():
        raise SystemExit(f"No such checkpoint: {path}")

    if path.suffix == ".nemo":
        logging.info(f"Restoring {path}")
        return ASRModel.restore_from(restore_path=str(path)), path.name

    # A Lightning .ckpt holds weights but not the architecture, so build the
    # model from the same pretrained name the run started from and load into it.
    logging.info(f"Building {pretrained} and loading weights from {path}")
    model = ASRModel.from_pretrained(model_name=pretrained)
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Report rather than silently accept: a partial load would produce numbers
    # that look plausible and mean nothing.
    if missing or unexpected:
        raise SystemExit(
            f"Checkpoint does not match the model.\n"
            f"  {len(missing)} missing key(s): {list(missing)[:5]}\n"
            f"  {len(unexpected)} unexpected key(s): {list(unexpected)[:5]}\n"
            f"Scoring this would produce meaningless numbers."
        )
    return model, path.name


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    model, label = load_model(cfg, args.checkpoint)
    label = args.label or label

    # Same decoding as training validation. Without this the model would use
    # whatever strategy its own config carries, and the WERs would not be
    # comparable with the curves.
    decoding_cfg = cfg.get("decoding", None)
    if decoding_cfg is not None:
        model.change_decoding_strategy(decoding_cfg)
        logging.info(f"Decoding strategy: {decoding_cfg.get('strategy')}")

    trainer = pl.Trainer(
        devices=args.devices,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=cfg.trainer.get("precision", "bf16-mixed"),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    macro_specs = cfg.get("macro_metrics", None)
    if macro_specs:
        trainer.callbacks.insert(0, _MacroMetricCallback(OmegaConf.to_container(macro_specs, resolve=True)))
    model.set_trainer(trainer)

    # Validation sets only. Deliberately not setup_dataloaders(), which would
    # also scan the 487k-clip training manifest for nothing.
    model.setup_multiple_validation_data(cfg.model.validation_ds)

    trainer.validate(model)

    # The macro callback writes into callback_metrics rather than logging, so
    # read the results from there.
    results = {
        k: float(v)
        for k, v in trainer.callback_metrics.items()
        if k.endswith("val_wer")
    }
    if not results:
        raise SystemExit("No *val_wer metrics were produced. Did validation actually run?")

    row = {"label": label, "checkpoint": args.checkpoint, "config": args.config, "wer": results}
    with open(args.out, "a") as fh:
        fh.write(json.dumps(row) + "\n")

    width = max(len(k) for k in results)
    print(f"\n{'=' * (width + 12)}")
    print(f"{label}")
    print(f"{'=' * (width + 12)}")
    for k in sorted(results, key=lambda k: (not k.startswith("combined"), k)):
        print(f"  {k:{width}s}  {results[k] * 100:6.2f}%")
    print(f"\nAppended to {args.out}")


if __name__ == "__main__":
    main()
