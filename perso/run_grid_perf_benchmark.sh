#!/usr/bin/env bash
# run_grid_perf_benchmark.sh
#
# CPU decode-cost + accuracy grid for parakeet_web issue #338, across stitched
# audio-length buckets. Reuses the production decoder via
# scripts/grid_search_benchmark.mjs, so the encoder output is cached per
# utterance and shared across every decode cell: proc_t/dur_t and dec_t/aud
# measure DECODE ONLY. Reports hyp_med/hyp_max/steps (how wide the beam expands
# per step, the CPU-cost driver) and a per-cell 5-min load average (load5).
#
# Three phases (each = grid_search invocation(s) with own output + marker):
#   A  accuracy: beam-width x boost-strength (incl. no-boost baseline), default MAES.
#   B  MAES value sweep at a fixed beam/strength (boost, no baseline), on a length subset.
#   C  REAL continuous speeches (fp32-referenced), beam x {no-boost, boost@1}: the
#      apples-to-apples check of the stitched no-boost beam degradation on native audio.
#
# >>> SOURCE this file, do NOT execute it <<<
#   source run_grid_perf_benchmark.sh    # 1st time in a shell: defines
#                                         # pause_run / resume_run, then returns
#   source run_grid_perf_benchmark.sh    # 2nd time in the SAME shell: launches
#                                         # the run (RAM-capped via systemd)
#
# Pause/resume from ANOTHER shell when the flat gets too hot: source once there
# to get the functions, then `pause_run` (systemd cgroup freeze) / `resume_run`.
# They target the named scope, so they work across shells.
#
# Resume after a crash: source-to-launch again. Each (phase,length) writes its
# own perf_<phase>_<L>.jsonl/.md + a .done_<phase>_<L> marker; completed
# (phase,length) pairs are skipped and a half-finished one resumes via
# grid_search's --resume, so a crash never loses prior work.
#
# Built with Claude Code.

# ---------------------------------------------------------------- config (edit me)
# Local checkout of the parakeet_web front end, which owns the ONNX runtime
# benchmark harness this script drives. Override with the env var to point
# elsewhere: PARAKEET_WEB=/path/to/parakeet_web ./perso/run_grid_perf_benchmark.sh
PARAKEET_WEB="${PARAKEET_WEB:-$HOME/parakeet_web}"
STITCH_ROOT="$HOME/Downloads/parakeet finetuning/NeMo/perso/grid_stitch_datasets"
MODEL_DIR="$PARAKEET_WEB/fallback_models/Olicorne/parakeet-tdt-0.6b-v3-smoothquant-onnx"
PHRASE_BOOST="$PARAKEET_WEB/phrase_boosting/french_medical.pwc"  # precompiled trie (fast load)
BENCHMARK="$PARAKEET_WEB/scripts/grid_search_benchmark.mjs"
OUT="$STITCH_ROOT/_results"

UNIT="grid-perf-bench"          # systemd --user scope name (pause/resume target)
MEM_HIGH="58G"                  # soft throttle
MEM_MAX="60G"                   # hard ceiling (cgroup OOM-kills above this)

DATASETS="fleurs_fr fleurs_en diy_drugs"
LIMIT=30                        # utterances per (dataset,length) cell

# --- Phase A: accuracy sweep -- beam-width x boost-strength, default MAES -----------
# Cells/length = beams(5) x [no-boost + strengths(3)] = 20. x5 lengths = 100 cells.
# Answers: does higher strength restore the in-domain (diy_drugs) win, and how does
# boost interact with beam width / audio length.
# prefix-alpha is deliberately 0 (NOT NeMo's default of 1): the phase-B sweep found
# pa=0 and pa=1 accuracy-equal while pa=1 costs ~15-21% more decode time (prefix-search
# overhead), so 0 is a free speedup. Matches the shipped default in parakeet.js /
# transcribe.mjs / App.jsx (all 0, same rationale).
A_LENGTHS="10s 20s 30s 60s 120s"
A_BEAMS="1,2,3,4,5"
A_STRENGTHS="1,2,3"
A_BASELINE="yes"                 # yes = include the no-boost baseline row
A_MAES="--maes-prefix-alpha 0"

# --- Phase B: MAES value sweep -- fixed beam/strength, length subset ----------------
# Round 1 settled two knobs: prefix-alpha (0 = free speedup at equal accuracy, kept)
# and gamma (inert across 1.5-3.0). Round 2 fixes those (pa 0, gamma 2.3 = NeMo default)
# and sweeps the two still-open knobs: num-steps {2,3,4} (confirm 2 is the knee; num-steps 1
# was dropped, round 1 showed it loses ~0.5 CER for no speed) and expansion-beta {1,2,3}
# (candidate-pool width = top-(beam+beta), the likeliest real cost/accuracy dial).
# Cells/length = num-steps(3) x beta(3) = 9, at beam 5, boost@1, no baseline. x2 lengths = 18 cells.
B_LENGTHS="30s 120s"
B_BEAMS="5"
B_STRENGTHS="1"
B_BASELINE="no"
B_MAES="--maes-num-steps 2,3,4 --maes-expansion-beta 1,2,3 --maes-expansion-gamma 2.3 --maes-prefix-alpha 0"

# --- Phase C: REAL continuous speech (political speeches, fp32-referenced) ----------
# One invocation over speeches/manifest.json (8 clips, 6x120s + 2x60s), beam x
# {no-boost, boost@1}, default MAES. Native audio, so it tells whether the stitched
# 120s no-boost beam degradation is real or a stitching artifact.
C_BEAMS="1,2,3,4,5"
C_STRENGTHS="1"                  # boost@1 + no-boost baseline
C_LIMIT=0                        # 0 = all utterances in the manifest

# --- Phase D: MAES presets vs AUDIO QUALITY (SNR sweep) -----------------------------
# The clean-audio sweeps found the MAES fine-params inert - but MAES only widens the
# beam when the model is UNCERTAIN, so that may not hold on bad audio. This degrades the
# 30s clips with pink noise at graded SNR (via augment_snr.py) and A/Bs two presets at
# beam 5, NO boost:
#   default = NeMo/benchmark   (num-steps 2, beta 2, gamma 2.3)
#   app     = current App.jsx  (num-steps 3, beta 4, gamma 4.0)
# hyp_med/hyp_max/steps + dec_t/aud + load5 are captured per cell automatically (beam 5).
D_LENGTH="30s"
D_SNRS="clean snr10 snr5 snr0"       # 'clean' = the original 30s clips (SNR = inf)
D_DATASETS="fleurs_fr fleurs_en diy_drugs"
D_DEFAULT_MAES="--maes-num-steps 2 --maes-expansion-beta 2 --maes-expansion-gamma 2.3 --maes-prefix-alpha 0"
D_APP_MAES="--maes-num-steps 3 --maes-expansion-beta 4 --maes-expansion-gamma 4.0 --maes-prefix-alpha 0"

# --- Phase E: BEAM WIDTH vs AUDIO QUALITY (does a wide beam help when noisy?) --------
# Phase A showed beam width is ~inert for accuracy on CLEAN audio (real speeches wander
# in a ~0.5 CER band across widths 1-5) - but we never swept width on BAD audio. This
# sweeps beam 1-5 over the SAME SNR levels as phase D, with the no-boost baseline AND
# boost@1 in one invocation per level (encoder cached across all cells), default MAES
# (2/2/2.3, pa 0). If beam 5 beats beam 2-3 as SNR drops, an adaptive
# narrow-clean/wide-noisy beam is motivated; if not, width is inert everywhere and only
# serves boosting. beam_med/beam_max/batch_med/batch_max/steps captured per cell.
E_LENGTH="30s"
E_SNRS="clean snr10 snr5 snr0"
E_DATASETS="fleurs_fr fleurs_en diy_drugs"
E_BEAMS="1,2,3,4,5"
E_STRENGTHS="1"                  # boost@1 + no-boost baseline (both, one invocation/level)
E_MAES="--maes-num-steps 2 --maes-expansion-beta 2 --maes-expansion-gamma 2.3 --maes-prefix-alpha 0"
# Rough total: phase A ~2 h + phase B ~1 h + phase C ~15 min + phase D ~40 min + phase E ~50 min. Trim to cut it.
# --------------------------------------------------------------------------------

# pause_run / resume_run act on the named systemd scope (work from any shell).
if ! declare -F pause_run >/dev/null 2>&1; then
  pause_run()  { systemctl --user freeze "${UNIT:-grid-perf-bench}.scope" && echo "[paused]  ${UNIT:-grid-perf-bench}.scope frozen"; }
  resume_run() { systemctl --user thaw   "${UNIT:-grid-perf-bench}.scope" && echo "[resumed] ${UNIT:-grid-perf-bench}.scope thawed"; }
  echo "Defined pause_run / resume_run (target: ${UNIT}.scope)."
  echo "Source this file again in THIS shell to launch the run."
  return 0 2>/dev/null || exit 0
fi

# Run one phase = one grid_search invocation per length in that phase.
_run_phase() {
  local phase="$1" lengths="$2" beams="$3" strengths="$4" baseline="$5" maes="$6"
  local L D m rc marker baseflag
  local -a manifests
  for L in $lengths; do
    marker="$OUT/.done_${phase}_$L"
    if [[ -f "$marker" ]]; then echo "[skip] phase $phase / $L (.done marker)"; continue; fi
    manifests=()
    for D in $DATASETS; do
      m="$STITCH_ROOT/$D/$L/manifest.json"
      [[ -f "$m" ]] && manifests+=( --manifest "${D}_${L}=$m" )
    done
    if [[ ${#manifests[@]} -eq 0 ]]; then echo "[warn] no manifests for $L; skipping"; continue; fi
    baseflag=""; [[ "$baseline" == "no" ]] && baseflag="--no-baseline"
    echo "[run] phase=$phase length=$L beams=$beams strengths=$strengths ${baseflag:+(no baseline) }maes=[$maes]"
    node "$BENCHMARK" \
      "${manifests[@]}" \
      --audio-root "$STITCH_ROOT" \
      --model-dir "$MODEL_DIR" \
      --quant int8 --decoder-quants int8 --ort=node \
      --beam-width "$beams" \
      --phrase-boost "$PHRASE_BOOST" --boost-strength "$strengths" $baseflag \
      $maes \
      --limit "$LIMIT" --resume \
      --jsonl "$OUT/perf_${phase}_$L.jsonl" \
      --md "$OUT/perf_${phase}_$L.md"
    rc=$?
    if [[ $rc -eq 0 ]]; then touch "$marker"; echo "[ok] phase $phase / $L"; else
      echo "[FAIL] phase $phase / $L (exit $rc). Re-source to resume; --resume skips done cells." >&2
    fi
  done
}

# Phase C: real continuous speeches (fp32-referenced), one invocation, no length loop.
_run_speeches() {
  local marker="$OUT/.done_C_speeches" m="$STITCH_ROOT/speeches/manifest.json" rc
  if [[ -f "$marker" ]]; then echo "[skip] phase C / speeches (.done marker)"; return; fi
  if [[ ! -f "$m" ]]; then echo "[warn] no speeches manifest ($m); run make_reference_manifest.py first. Skipping phase C." >&2; return; fi
  echo "[run] phase=C speeches (real audio) beams=$C_BEAMS strength=$C_STRENGTHS (+baseline)"
  node "$BENCHMARK" \
    --manifest "speeches=$m" \
    --audio-root "$STITCH_ROOT" \
    --model-dir "$MODEL_DIR" \
    --quant int8 --decoder-quants int8 --ort=node \
    --beam-width "$C_BEAMS" \
    --phrase-boost "$PHRASE_BOOST" --boost-strength "$C_STRENGTHS" \
    --maes-prefix-alpha 0 \
    --limit "$C_LIMIT" --resume \
    --jsonl "$OUT/perf_C_speeches.jsonl" --md "$OUT/perf_C_speeches.md"
  rc=$?
  if [[ $rc -eq 0 ]]; then touch "$marker"; echo "[ok] phase C / speeches"; else
    echo "[FAIL] phase C / speeches (exit $rc). Re-source to resume; --resume skips done cells." >&2
  fi
}

# Phase D: MAES preset (default vs app) x SNR level, beam 5, NO boost. One invocation
# per (level,preset). 'clean' uses the original 30s clips; snrN use the augmented sets.
_run_degraded() {
  local level ds m rc marker preset maes
  local -a manifests
  for level in $D_SNRS; do
    manifests=()
    for ds in $D_DATASETS; do
      if [[ "$level" == "clean" ]]; then m="$STITCH_ROOT/$ds/$D_LENGTH/manifest.json"
      else m="$STITCH_ROOT/degraded/$level/$ds/manifest.json"; fi
      [[ -f "$m" ]] && manifests+=( --manifest "${ds}_${level}=$m" )
    done
    if [[ ${#manifests[@]} -eq 0 ]]; then echo "[warn] no manifests for D/$level (run augment_snr.py); skipping" >&2; continue; fi
    for preset in default app; do
      marker="$OUT/.done_D_${level}_${preset}"
      if [[ -f "$marker" ]]; then echo "[skip] phase D / $level / $preset (.done marker)"; continue; fi
      if [[ "$preset" == "default" ]]; then maes="$D_DEFAULT_MAES"; else maes="$D_APP_MAES"; fi
      echo "[run] phase=D level=$level preset=$preset (beam 5, no boost) maes=[$maes]"
      node "$BENCHMARK" \
        "${manifests[@]}" \
        --audio-root "$STITCH_ROOT" \
        --model-dir "$MODEL_DIR" \
        --quant int8 --decoder-quants int8 --ort=node \
        --beam-width 5 \
        $maes \
        --limit "$LIMIT" --resume \
        --jsonl "$OUT/perf_D_${level}_${preset}.jsonl" \
        --md "$OUT/perf_D_${level}_${preset}.md"
      rc=$?
      if [[ $rc -eq 0 ]]; then touch "$marker"; echo "[ok] phase D / $level / $preset"; else
        echo "[FAIL] phase D / $level / $preset (exit $rc). Re-source to resume." >&2
      fi
    done
  done
}

# Phase E: beam width 1-5 x SNR level, no-boost baseline + boost@1, default MAES. One
# invocation per level (encoder cached across every beam/boost cell). Tests whether a
# wide beam earns its keep as audio quality drops. 'clean' uses the original 30s clips.
_run_beam_snr() {
  local level ds m rc marker
  local -a manifests
  for level in $E_SNRS; do
    marker="$OUT/.done_E_${level}"
    if [[ -f "$marker" ]]; then echo "[skip] phase E / $level (.done marker)"; continue; fi
    manifests=()
    for ds in $E_DATASETS; do
      if [[ "$level" == "clean" ]]; then m="$STITCH_ROOT/$ds/$E_LENGTH/manifest.json"
      else m="$STITCH_ROOT/degraded/$level/$ds/manifest.json"; fi
      [[ -f "$m" ]] && manifests+=( --manifest "${ds}_${level}=$m" )
    done
    if [[ ${#manifests[@]} -eq 0 ]]; then echo "[warn] no manifests for E/$level (run augment_snr.py); skipping" >&2; continue; fi
    echo "[run] phase=E level=$level beams=$E_BEAMS strength=$E_STRENGTHS (+baseline) maes=[$E_MAES]"
    node "$BENCHMARK" \
      "${manifests[@]}" \
      --audio-root "$STITCH_ROOT" \
      --model-dir "$MODEL_DIR" \
      --quant int8 --decoder-quants int8 --ort=node \
      --beam-width "$E_BEAMS" \
      --phrase-boost "$PHRASE_BOOST" --boost-strength "$E_STRENGTHS" \
      $E_MAES \
      --limit "$LIMIT" --resume \
      --jsonl "$OUT/perf_E_${level}.jsonl" \
      --md "$OUT/perf_E_${level}.md"
    rc=$?
    if [[ $rc -eq 0 ]]; then touch "$marker"; echo "[ok] phase E / $level"; else
      echo "[FAIL] phase E / $level (exit $rc). Re-source to resume." >&2
    fi
  done
}

_grid_perf_run_all() {
  set -uo pipefail
  mkdir -p "$OUT"
  _run_phase A "$A_LENGTHS" "$A_BEAMS" "$A_STRENGTHS" "$A_BASELINE" "$A_MAES"
  # Phase B round 2 runs under label "B2" so it writes fresh perf_B2_* and KEEPS the
  # round-1 perf_B_* results (different sweep). Just source-and-go: A/C are skipped by
  # their .done markers, B2 has none yet so it runs.
  _run_phase B2 "$B_LENGTHS" "$B_BEAMS" "$B_STRENGTHS" "$B_BASELINE" "$B_MAES"
  _run_speeches
  _run_degraded
  _run_beam_snr
  echo "[grid-perf] all phases done. Results: $OUT/perf_A_<L>.md, perf_B2_<L>.md, perf_C_speeches.md, perf_D_<snr>_<preset>.md, perf_E_<snr>.md (+ .jsonl); round-1 perf_B_<L>.md kept"
}

# Clear any stale scope, refuse to double-launch.
systemctl --user reset-failed "${UNIT}.scope" 2>/dev/null || true
if systemctl --user is-active "${UNIT}.scope" >/dev/null 2>&1; then
  echo "[abort] ${UNIT}.scope is still active (a run is in progress). resume_run, or: systemctl --user stop ${UNIT}.scope" >&2
  return 1 2>/dev/null || exit 1
fi

echo "[launch] phase A: beams=$A_BEAMS x strengths=$A_STRENGTHS (+baseline) over [$A_LENGTHS]"
echo "[launch] phase B2: MAES num-steps/beta sweep at beam=$B_BEAMS over [$B_LENGTHS] (fresh perf_B2_*, keeps round-1 perf_B_*)"
echo "[launch] phase C: speeches (real audio) beams=$C_BEAMS x {none,boost@$C_STRENGTHS}"
echo "[launch] phase D: MAES presets (default vs app) x SNR [$D_SNRS] at beam 5, no boost"
echo "[launch] phase E: beam width [$E_BEAMS] x SNR [$E_SNRS] (+baseline, boost@$E_STRENGTHS), default MAES"
echo "[launch] RAM-capped in ${UNIT}.scope (MemoryMax=$MEM_MAX). Pause from another shell: pause_run"
_payload="$(declare -p DATASETS LIMIT OUT STITCH_ROOT BENCHMARK MODEL_DIR PHRASE_BOOST A_LENGTHS A_BEAMS A_STRENGTHS A_BASELINE A_MAES B_LENGTHS B_BEAMS B_STRENGTHS B_BASELINE B_MAES C_BEAMS C_STRENGTHS C_LIMIT D_LENGTH D_SNRS D_DATASETS D_DEFAULT_MAES D_APP_MAES E_LENGTH E_SNRS E_DATASETS E_BEAMS E_STRENGTHS E_MAES); $(declare -f _run_phase _run_speeches _run_degraded _run_beam_snr _grid_perf_run_all); _grid_perf_run_all"
systemd-run --user --scope --unit="$UNIT" \
  -p MemoryHigh="$MEM_HIGH" -p MemoryMax="$MEM_MAX" \
  bash -c "$_payload"
