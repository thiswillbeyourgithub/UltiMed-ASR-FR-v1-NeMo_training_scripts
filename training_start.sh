#!/bin/bash
# Training launcher for the UltiMed fine-tune.
#
# Launches training and relaunches it automatically if it crashes (OOM, driver
# hiccup, dataloader worker death):
# exp_manager.resume_if_exists picks the run back up from the last checkpoint
# inside $NEMO_EXP_DIR. Stops on its own when training finishes cleanly
# (trainer.max_time or max_epochs reached) or when you Ctrl-C.
#
# Repeated fast failures (5 in a row under 10 min uptime) abort the loop: that
# pattern means a config or driver problem, not a transient crash.
#
# NOTE: trainer.max_time counts wall time of the CURRENT process only, so every
# relaunch resets the 10-day timer. If the run crashes often the calendar end
# date slips; lower max_time in the config on relaunch if you need a hard
# deadline.
#
# The checkpoint location comes from the .env file next to this script: copy
# .env.example to .env and set NEMO_EXP_DIR there. Exporting NEMO_EXP_DIR
# before calling still overrides it for a single run.
#
# The relaunch loop, logging and env setup were added with Claude Code.

USE_PDB=false
HYDRA_OVERRIDES=()

usage() {
  echo "Usage: $0 [options] [hydra overrides]"
  echo ""
  echo "Options:"
  echo "  --debug        Run training under pdb (single run, no relaunch loop)"
  echo "  -h, --help     Show this help message"
  echo ""
  echo "Any argument containing '=' (or starting with '+' or '~') is passed to"
  echo "the training script as a hydra override, e.g.:"
  echo "  $0 trainer.max_steps=5000 exp_manager.version=1.3.0-nospd"
  echo "Overrides survive the relaunch loop: every relaunch reuses them."
  echo ""
  echo "Checkpoints go to \$NEMO_EXP_DIR, read from the .env file next to this"
  echo "script (cp .env.example .env and set the path). Exporting NEMO_EXP_DIR"
  echo "yourself overrides the .env value for a single run."
  echo "Cache rebuild is controlled by encoder_cache.auto_rebuild in perso/training_config.yaml."
}

for arg in "$@"; do
  case "$arg" in
    --debug) USE_PDB=true ;;
    -h|--help) usage; exit 0 ;;
    # Hydra override syntax: key=value, +key=value (add), ~key (delete).
    # Anything else is still a typo worth refusing.
    *=*|+*|~*) HYDRA_OVERRIDES+=("$arg") ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

cd "$(dirname "$0")" || exit 1

# --- Where checkpoints go -----------------------------------------------------
# NEMO_EXP_DIR is resolved by perso/resolve_exp_dir.sh, shared with
# tensorboard_start.sh so both agree on where the runs live: it reads the
# gitignored .env next to this script, lets an already-exported value win, and
# refuses to run if the drive looks unmounted. The helper never creates the
# directory, that stays here.
# shellcheck source=perso/resolve_exp_dir.sh
. ./perso/resolve_exp_dir.sh

mkdir -p "$NEMO_EXP_DIR" || { echo "ERROR: cannot create $NEMO_EXP_DIR" >&2; exit 1; }

# --- Live log --------------------------------------------------------------
# Everything is teed to a timestamped file as it happens, so the run can be
# followed (or read by Claude Code) while it is still going. latest.log always
# points at the newest run.
LOG_DIR="perso/training_logs_ignore-backups"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date '+%Y-%m-%d_%H%M%S').log"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_DIR/latest.log"

log() { echo "$@" | tee -a "$LOG_FILE"; }

PYTHON_CMD="./.venv/bin/python"
[ -x "$PYTHON_CMD" ] || PYTHON_CMD="python"
if $USE_PDB; then
  PYTHON_CMD="$PYTHON_CMD -m pdb"
fi

# Env notes:
# - expandable_segments: lets the CUDA allocator grow/shrink segments instead of
#   keeping fixed pools, which avoids fragmentation OOM when long and short
#   samples (very different activation sizes) are mixed in the same run.
# - garbage_collection_threshold: above 80% usage the allocator releases
#   cached-but-unused blocks instead of sitting on them. nvidia-smi reports
#   ~23.6 of 24.5 GB on the worst-case batch, but that is mostly the allocator
#   holding freed blocks: stealing VRAM from a second process puts the true
#   peak at ~21.3 GiB, so real headroom is ~2.2 GB (see README.md).
# - NUMBA_CUDA_USE_NVIDIA_BINDING: required by the numba TDT loss kernels.
# - PYTHONUNBUFFERED: so the log file gets lines as they happen.
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export PYTHONUNBUFFERED=1

if ! nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG_FILE"; then
  log "ERROR: nvidia-smi failed, the GPU driver is not responding. Fix that before training."
  exit 1
fi
log "[wrapper] checkpoints -> $NEMO_EXP_DIR ($(df -h --output=avail "$NEMO_EXP_DIR" | tail -1 | tr -d ' ') free)"
log "[wrapper] live log     -> $LOG_FILE  (also $LOG_DIR/latest.log)"
if [ ${#HYDRA_OVERRIDES[@]} -gt 0 ]; then
  # In the log so the run is self-describing: the yaml alone does not tell
  # you what an override run actually trained with.
  log "[wrapper] hydra overrides: ${HYDRA_OVERRIDES[*]}"
fi

STOP=false
trap 'STOP=true; log "[wrapper] stop requested, finishing up"' INT TERM

FAST_FAIL_SECS=600
MAX_FAST_FAILS=5
fast_fails=0
attempt=0
code=0

while ! $STOP; do
  attempt=$((attempt + 1))
  start=$(date +%s)
  log "[wrapper] $(date '+%F %T') launch #$attempt"

  $PYTHON_CMD ./examples/asr/speech_to_text_finetune_cached.py \
    --config-dir="$(pwd)/perso" --config-name=training_config \
    "${HYDRA_OVERRIDES[@]}" 2>&1 | tee -a "$LOG_FILE"
  code=${PIPESTATUS[0]}
  runtime=$(( $(date +%s) - start ))

  if $USE_PDB; then
    log "[wrapper] --debug: single run only, not relaunching"
    break
  fi
  if [ "$code" -eq 0 ]; then
    log "[wrapper] training finished cleanly after launch #$attempt"
    break
  fi
  $STOP && break

  if [ "$runtime" -lt "$FAST_FAIL_SECS" ]; then
    fast_fails=$((fast_fails + 1))
  else
    fast_fails=0
  fi
  if [ "$fast_fails" -ge "$MAX_FAST_FAILS" ]; then
    log "[wrapper] $fast_fails consecutive failures under ${FAST_FAIL_SECS}s (last exit $code): aborting, this is not a transient crash."
    break
  fi

  log "[wrapper] exit $code after ${runtime}s; relaunching in 30s (resume_if_exists picks up the last checkpoint)"
  sleep 30
done

log "Done"

exit "$code"
