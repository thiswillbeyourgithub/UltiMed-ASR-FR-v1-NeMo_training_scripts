#!/usr/bin/env python
"""Turn a training .ckpt (or a perso/average_checkpoints.py output) into a .nemo.

The model is rebuilt exactly the way perso/evaluate_checkpoint.py scores it (the
base .nemo named by the training config, then a strict state_dict load), so the
.nemo that gets exported to ONNX is the very model the benchmark picked.

    ./.venv/bin/python perso/ckpt_to_nemo.py <in>.ckpt <out>.nemo [--config <run config>.yaml]

Written with Claude Code.
"""
import argparse
from pathlib import Path

from omegaconf import OmegaConf

from evaluate_checkpoint import load_model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("out")
    p.add_argument("--config", default="perso/training_config.yaml",
                   help="training config naming the base model the checkpoint was fine-tuned from")
    args = p.parse_args()
    model, _ = load_model(OmegaConf.load(args.config), args.checkpoint)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save_to(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
