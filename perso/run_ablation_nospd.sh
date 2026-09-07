#!/bin/bash
# SPD ablation arm: what does AdamSPD actually buy over plain AdamW?
#
# No run has ever answered this. All three real runs (1.1.0, 1.2.0, 1.3.0)
# used working AdamSPD (the pre-2026-08-22 sign bug that silently disabled it
# was fixed before 1.1.0 launched), so there is no baseline without it.
#
# Design: identical to 1.3.0 in everything except the optimizer class, then
# truncated at 5,000 opt-steps. Same seed, so the same data order; same
# optim.sched.max_steps=10000, so the SAME LR trajectory point for point over
# the shared 5,000 steps. The only changed variable is AdamSPD -> AdamW
# (lr, betas and weight_decay are identical; the semantics of weight_decay
# shift from "decay toward the pretrained anchor when moving away from it" to
# AdamW's plain decay toward zero, which is the standard-practice arm this
# comparison is about). With the anchor gone, the training script's
# isinstance(AdamSPD) gate skips anchor registration on its own.
#
# Because of the shared trajectory, every validation round compares directly
# against 1.3.0's logged curve. The reference values to beat or match, from
# 1.3.0 at step 5000:
#
#   combined_macro_val_wer        7.63      (norm: 5.38)
#   in-domain 4-set               6.22 avg  (dictionary 5.25, parhaf 5.85,
#                                            drugs 7.21, oli_drug 6.57)
#   fleurs_fr / fleurs_en         8.95 / 11.95   (baseline 8.55 / 10.38)
#
# How to read the outcome:
#   - no-SPD in-domain clearly better, fleurs similar  -> SPD is costing
#     adaptation; drop it or scope it to the encoder only.
#   - no-SPD fleurs clearly worse, in-domain similar   -> SPD is doing real
#     anti-forgetting work; keep it, and scoping it to the encoder becomes
#     the interesting refinement.
#   - both similar -> SPD is inert at wd=0.01; simplify it away.
#
# The truncated cosine is acknowledged to the config audit via
# config_audit.allow_truncated_anneal (see the comment in training_config.yaml);
# it would rightly fail the schedule-coherence check otherwise.
#
# Cost: ~8 h (40,000 batches at ~1.46 b/s plus 5 validation rounds).
# Extra args pass through to training_start.sh (e.g. --skip-docker).
#
# Written with Claude Code.

cd "$(dirname "$0")/.." || exit 1

exec ./training_start.sh \
  model.optim._target_=torch.optim.AdamW \
  trainer.max_steps=5000 \
  exp_manager.version=1.3.0-nospd \
  config_audit.allow_truncated_anneal=true \
  "$@"
