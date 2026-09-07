#!/usr/bin/env bash
# Worst-case VRAM check for training_config.yaml. Made with Claude Code.
#
# Builds a manifest of clips carrying the longest transcripts available at their
# length, then trains on it, so every batch is the worst batch training could
# ever draw. If this passes with headroom, a real run cannot OOM on an unlucky
# batch.
#
# With cost batching on (model.train_ds.cost_batching.enabled), the worst case
# is not one batch shape but a frontier: the sampler fills every batch up to the
# same cost budget, so a batch is one 40 s clip or sixteen 15 s ones and both sit
# at the budget. The fixture therefore holds clips at the duration that exactly
# saturates the budget for each batch size in 1, 2, 4, 8, 16, and the sampler
# regroups them into exactly those batches. Passing means the whole frontier
# passes, not just one corner of it.
#
# With cost batching off it falls back to the old behaviour: 200 clips all at
# max_duration, at the fixed batch_size.
#
# Re-run this after changing max_duration, batch_size, cost_batching.budget or
# joint.fused_batch_size.
#
# Usage:  NEMO_EXP_DIR=/tmp/oom_check ./perso/oom_margin_check.sh [cap] [batch]
# Defaults come from the config itself.
set -uo pipefail
cd "$(dirname "$0")/.."

: "${NEMO_EXP_DIR:?set it to a scratch dir, e.g. NEMO_EXP_DIR=/tmp/oom_check}"

PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python

CAP="${1:-$($PY -c 'from omegaconf import OmegaConf; print(int(OmegaConf.load("perso/training_config.yaml").model.train_ds.max_duration))')}"
BS="${2:-$($PY -c 'from omegaconf import OmegaConf; print(OmegaConf.load("perso/training_config.yaml").model.train_ds.batch_size)')}"

SRC="./perso/ultimed_data_ignore-backups/NeMO_files/train.jsonl"
WORST="$(mktemp -t worst_XXXX.jsonl)"
LOG="$(mktemp -t oom_check_XXXX.log)"
VLOG="$(mktemp -t oom_check_XXXX.vram)"
trap 'rm -f "$WORST" "$VLOG"' EXIT

"$PY" - "$SRC" "$WORST" "$CAP" <<'EOF'
import json, os, sys
from omegaconf import OmegaConf

src, out, cap = sys.argv[1], sys.argv[2], float(sys.argv[3])
cost_cfg = OmegaConf.load("perso/training_config.yaml").model.train_ds.get("cost_batching", None)
cost_on = bool(cost_cfg) and bool(cost_cfg.get("enabled", False))

if cost_on:
    # Duration at which a batch of n clips exactly reaches the budget, for the
    # batch sizes the sampler can actually produce. Each is a different corner
    # of the same frontier: same cost, very different shape.
    budget = float(cost_cfg.get("budget", 16500))
    n_max = int(cost_cfg.get("max_batch_size", 16))
    sizes = [n for n in (1, 2, 4, 8, 16) if n <= n_max]
    targets = sorted({round(min(cap, (budget / (4 * n + 6)) ** 0.5), 1) for n in sizes}, reverse=True)
    per_target = max(200 // len(targets), 1)
else:
    targets, per_target = [cap], 200

# audio_filepath is relative to the manifest's own directory, and this fixture
# is written to a temp dir, so the paths have to be made absolute.
base = os.path.dirname(os.path.abspath(src))
rows = []
with open(src) as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

selected, report = [], []
for target in targets:
    window = 0.75 if cost_on else 1.5
    at_target = [d for d in rows if target - window <= float(d.get("duration") or 0) <= target]
    # longest transcript at that duration is the worst case for U in the lattice
    at_target.sort(key=lambda d: -len(d.get("text") or ""))
    chosen = at_target[:per_target]
    selected.extend(chosen)
    report.append(f"{len(chosen)}x{target}s")

with open(out, "w") as f:
    for d in selected:
        p = d["audio_filepath"]
        if not os.path.isabs(p):
            d["audio_filepath"] = os.path.normpath(os.path.join(base, p))
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

mode = "cost batching" if cost_on else f"fixed batch_size, cap {cap}s"
print(f"[oom-check] fixture ({mode}): {len(selected)} clips, {' '.join(report)}")
if len(selected) < sum(min(per_target, 1) for _ in targets):
    print("[oom-check] WARNING: manifest has no clips at some target durations")
EOF

export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export PYTHONUNBUFFERED=1

TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > "$VLOG" &
SAMPLER=$!

"$PY" examples/asr/speech_to_text_finetune_cached.py \
  --config-dir="$PWD/perso" --config-name=training_config \
  "model.train_ds.manifest_filepath=[\"$WORST\"]" \
  "model.train_ds.max_duration=${CAP}.0" \
  "model.train_ds.batch_size=${BS}" \
  trainer.max_epochs=1 \
  trainer.val_check_interval=1.0 \
  +trainer.limit_val_batches=0.0 \
  trainer.num_sanity_val_steps=0 \
  exp_manager.create_checkpoint_callback=false \
  exp_manager.resume_if_exists=false \
  exp_manager.version=oom_margin_check \
  > "$LOG" 2>&1
CODE=$?

kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null
PEAK=$(sort -n "$VLOG" | tail -1)

if [ "$CODE" -eq 0 ]; then
  grep -m1 -o "Cost batching on:.*" "$LOG" | sed 's/^/[oom-check] /'
  echo "[oom-check] PASS  peak ${PEAK}/${TOTAL} MiB  (margin $((TOTAL-PEAK)) MiB)"
  echo "[oom-check] under ~1000 MiB of margin is uncomfortably thin for a multi-day run"
elif grep -q "OutOfMemoryError" "$LOG"; then
  echo "[oom-check] FAIL: OOM. Lower max_duration first (cost scales with duration squared), then batch_size."
  grep -m1 "Tried to allocate" "$LOG"
else
  echo "[oom-check] FAIL (not an OOM), see $LOG"
  tail -5 "$LOG"
fi
rm -f "$LOG"
exit "$CODE"
