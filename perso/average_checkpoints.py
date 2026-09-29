#!/usr/bin/env python
"""Average the weights of several Lightning .ckpt files of one run into a new .ckpt.

Only the model ``state_dict`` is averaged and saved (the optimizer state is dropped),
which is all ``perso/evaluate_checkpoint.py --checkpoint`` reads. Float tensors get a
(weighted) mean; integer buffers (BatchNorm ``num_batches_tracked``) are taken from the
last checkpoint. The sum is accumulated in float64 one checkpoint at a time, so RAM
stays at about two model copies whatever the number of inputs.

NeMo's own ``scripts/checkpoint_averaging/average_model_checkpoints.py`` hardcodes a CTC
model class behind a hydra config, hence this small TDT-agnostic version.

    ./.venv/bin/python perso/average_checkpoints.py out.ckpt a.ckpt b.ckpt c.ckpt
    ./.venv/bin/python perso/average_checkpoints.py out.ckpt a.ckpt b.ckpt --weights 1 2

This file was written by Claude Code.
"""
import argparse

import torch


def average_state_dicts(paths, weights=None):
    """Weighted mean of the ``state_dict`` of each checkpoint in ``paths``."""
    weights = weights or [1.0] * len(paths)
    if len(weights) != len(paths):
        raise SystemExit(f"{len(weights)} weights for {len(paths)} checkpoints")
    total = float(sum(weights))
    acc, keys = {}, None
    for path, w in zip(paths, weights):
        sd = torch.load(path, map_location="cpu", weights_only=False, mmap=True)["state_dict"]
        if keys is None:
            keys = list(sd)
        elif set(sd) != set(keys):
            raise SystemExit(f"{path} does not have the same parameters as {paths[0]}")
        for k, v in sd.items():
            if v.is_floating_point():
                acc[k] = acc.get(k, 0) + v.double() * (w / total)
            else:
                acc[k] = v.clone()
        del sd
    ref = torch.load(paths[-1], map_location="cpu", weights_only=False, mmap=True)["state_dict"]
    return {k: acc[k].to(ref[k].dtype) for k in keys}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--weights", nargs="+", type=float)
    args = p.parse_args()
    avg = average_state_dicts(args.checkpoints, args.weights)
    torch.save({"state_dict": avg, "averaged_from": args.checkpoints, "weights": args.weights}, args.out)
    print(f"wrote {args.out} ({len(avg)} tensors from {len(args.checkpoints)} checkpoints)")


if __name__ == "__main__":
    main()
