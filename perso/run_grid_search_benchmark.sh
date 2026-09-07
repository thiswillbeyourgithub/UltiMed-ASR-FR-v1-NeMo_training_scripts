#!/usr/bin/env bash
# Typical run of the Parakeet web grid-search WER benchmark over NeMo manifest(s).
# Edit MODEL_DIR to point at your finetuned model exported to ONNX
# (encoder-model.int8.onnx, decoder_joint-model.int8.onnx, vocab.txt).
#
# Pass --manifest more than once (see the second, commented line below) to score
# the SAME grid over several datasets at once: the accuracy table then breaks
# every cell down per dataset plus an "overall" row. That is how you check a
# phrase boost tuned for one domain does not degrade WER on unrelated data.
# Built with Claude Code.
set -euo pipefail

# Resolve paths relative to this script so it runs from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# The benchmark itself lives inside the parakeet_web repo (next to the
# transcribe.mjs pipeline it reuses); this wrapper just invokes it.
# Local checkout of the parakeet_web front end, which owns the ONNX runtime
# benchmark harness this script drives. Override with the env var to point
# elsewhere: PARAKEET_WEB=/path/to/parakeet_web ./perso/run_grid_search_benchmark.sh
PARAKEET_WEB="${PARAKEET_WEB:-$HOME/parakeet_web}"
BENCHMARK="$PARAKEET_WEB/scripts/grid_search_benchmark.mjs"

# In-domain (medical) set the phrase boost targets.
MANIFEST="$SCRIPT_DIR/oli_spoken_dataset/nemo_manifest_ordered_uncapitaliazedDCI.json"
# Unrelated general-French set (FLEURS fr validation, the same split training
# monitored as fleurs_frval_wer) to catch the boost degrading off-domain WER.
MANIFEST_FLEURS_FR="$SCRIPT_DIR/downloaded_datasets_ignore-backups/fleurs/fr/validation.altered.json"
# TODO: point this at your finetuned ONNX export dir.
MODEL_DIR="$PARAKEET_WEB/fallback_models/Olicorne/parakeet-tdt-0.6b-v3-smoothquant-onnx"
PHRASE_BOOST="$PARAKEET_WEB/phrase_boosting/french_medical.txt"

# fix the finding libcublast.so.12 lib issue
export LD_LIBRARY_PATH=$(printf '%s:' /usr/local/lib/python3.10/dist-packages/nvidia/*/lib):$LD_LIBRARY_PATH

# CPU beam sweep to find the knee on the node (CPU) backend. The MAES sweep settled
# the decode knobs (prefix-alpha 0 is a free ~15% decode saving at equal WER/CER;
# num-steps 1 and gamma 1.5 both lose accuracy for nothing), so we bake prefix-alpha 0
# and leave num-steps/gamma at their NeMo defaults, then sweep beam 1..10 to read the
# accuracy/decode-time knee. Unlike GPU (where extra beams ride one batched call so the
# knee sat at beam 5), CPU has no batch parallelism, so per-beam cost grows ~linearly
# and the knee is expected lower. NOTE: the encoder is pre-computed and cached across
# cells, so proc_t/dur_t / dec_t/aud measure DECODE ONLY; the (GPU-favouring) encoder
# cost is not in these numbers. --cuda stays commented so --ort=node runs on CPU.
node "$BENCHMARK" \
  --manifest "oli_drug=$MANIFEST" \
  --manifest "fleurs_fr=$MANIFEST_FLEURS_FR" \
  --audio-root "$NEMO_ROOT" \
  --model-dir "$MODEL_DIR" \
  --beam-width 1,2,3,4,5,6,7,8,9,10 \
  --maes-prefix-alpha 0 \
  --phrase-boost "$PHRASE_BOOST" \
  --boost-strength 1 --no-baseline \
  --quant int8 \
  --decoder-quants int8 \
  --ort=node \
  --jsonl "$SCRIPT_DIR/benchmark_results.jsonl" \
  --md "$SCRIPT_DIR/benchmark_results.md"
  # --cuda                               # uncomment to run the sweep on GPU instead
  # --maes-num-steps 1,2 \               # settled: 2 (1 loses accuracy, no speedup)
  # --maes-expansion-gamma 1.5,2.3 \     # settled: 2.3 (1.5 loses accuracy, no speedup)
  # --maes-expansion-beta 1,2 \          # mostly CPU fan-out; minor on GPU
  # --quant int8,fp32 \
  # --boost-minp 0.1,0.15,0.2 \
  # --depth-scaling 0.5,0.25 \
  # --resume
