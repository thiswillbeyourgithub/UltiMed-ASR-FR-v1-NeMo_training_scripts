# UltiMed-ASR-FR-v1: NeMo training scripts

**This repository is a fork of [NVIDIA-NeMo/NeMo](https://github.com/NVIDIA-NeMo/NeMo).**
It is not a general-purpose NeMo distribution and it is not maintained as one. It exists to
document, reproducibly, how [Olicorne/parakeet-tdt-0.6b-v3-UltiMed-onnx](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-UltiMed-onnx)
was fine-tuned: the patches NeMo needed, the training configuration, and the reasoning behind
both. If you want NeMo itself, go upstream.

Everything here sits on top of upstream commit
[`6442018984`](https://github.com/NVIDIA-NeMo/NeMo/commit/644201898480ec8c8d0a637f0c773825509ac4dc)
(`[TTS][MagpieTTS] Longform TTS using MagpieTTS (#15210)`, 2025-12-23). That is the exact tree
the released model was trained on. The branch has deliberately **not** been rebased onto a newer
NeMo: the patches below were never tested against anything else, and the point of this repo is to
record what actually ran.

## Related repositories

| What | Where |
|---|---|
| The fine-tuned model (ONNX, fp32 / fp16 / int8 / w4a8) | [huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-UltiMed-onnx](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-UltiMed-onnx) |
| The training dataset | [huggingface.co/datasets/Olicorne/UltiMed-ASR-FR-v1](https://huggingface.co/datasets/Olicorne/UltiMed-ASR-FR-v1) |
| Scripts that built that dataset | [github.com/thiswillbeyourgithub/UltiMed-ASR-FR-v1-scripts](https://github.com/thiswillbeyourgithub/UltiMed-ASR-FR-v1-scripts) |
| The TTS container that synthesised its audio | [github.com/thiswillbeyourgithub/UltiMed-ASR-FR-v1-Voxtral](https://github.com/thiswillbeyourgithub/UltiMed-ASR-FR-v1-Voxtral) |
| The un-finetuned baseline, same ONNX optimisations | [huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-optimized-onnx](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-optimized-onnx) |
| Upstream base model | [huggingface.co/nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |

**The detailed engineering notes are in [`perso/README.md`](perso/README.md)**: VRAM measurements,
the OOM frontier, why the optimizer was silently doing nothing, what the config audit found, and
the version-by-version record of each training run. That file is the interesting one. This one is
the map.

## What each patch does

The history is squashed into one commit per coherent change, oldest first. Individual commits are
not each independently runnable (the config that ties everything together arrives last); only the
branch tip is a working tree. Each commit message explains the why at length.

1. **`asr: batch by duration cost instead of clip count`**
   The RNNT/TDT joint builds a dense `[B, T, U, V]` lattice, so a clip's memory cost grows with the
   *square* of its duration and a fixed batch size has to be sized for the worst clip in the
   manifest. `DurationCostBatchSampler` packs each batch to a memory budget instead. Fitted against
   ten measured OOM points on a 24 GB card. Net effect on the real corpus: 1.56x throughput *and*
   96.6% of corpus hours kept instead of 86.8%.

2. **`optim: add AdamSPD (Selective Projection Decay)`**
   AdamW decays toward zero, which is the wrong direction when the goal is to not forget the
   pretrained multilingual model. SPD decays toward the pretrained weights, selectively. Includes
   the sign fix for a bug where two errors cancelled and *no decay was ever applied*, so every
   `weight_decay` value produced bit-identical runs.

3. **`metrics: normalised WER and per-source macro averaging`**
   Score after normalising casing, punctuation and number/unit spelling, and weight each validation
   source equally, so a 30-minute real-speech set counts as much as a multi-hour synthetic one.

4. **`finetune: cached-encoder training path and latent augmentation`**
   The training entry point (`examples/asr/speech_to_text_finetune_cached.py`, used for both modes),
   plus an optional path that freezes the lower encoder layers and precomputes their output. Because
   a frozen prefix means waveform augmentors no longer reach the trainable part, augmentation is
   applied in latent space instead.

5. **`config: audit that every setting is actually read, and crash if not`**
   Six settings in the config turned out to do nothing, and all six looked correct in the file. Two
   layers: reject any key nothing read, then re-read the settings that matter back off the objects
   that were actually built. Runs before the first training step, because these are expensive to
   discover on day five of a six-day run.

6. **`export: opt-in web-export encoder flags and ONNX exporter`**
   Three off-by-default encoder flags (mask-free graph, runtime relative-position encoding, a
   padded-batch NaN tripwire) that make the exported encoder usable under onnxruntime-web, plus the
   exporter and the in-domain int8 calibration set.

7. **`data: corpus preparation, rehearsal sets and the manifests`**
   Everything that builds the manifests: the LLM-generated in-domain sentence sets, the FLEURS and
   Common Voice rehearsal material used to measure and limit forgetting, and the UltiMed validation
   slices.

8. **`train: launcher, training config and notes`** (this commit)
   `training_start.sh`, `tensorboard_start.sh`, the full annotated `perso/training_config.yaml`, and
   the engineering notes.

## Reproducing

```bash
git clone https://github.com/thiswillbeyourgithub/UltiMed-ASR-FR-v1-NeMo_training_scripts
uv venv --python=3.10.12
uv pip install nemo_toolkit[asr] onnx==1.18.0 numba-cuda==0.15.1 cuda-core cuda-bindings \
  cuda-pathfinder cuda-python==13.1.1 tensorboard>=2.19.0 setuptools==81.0.0 \
  --force-reinstall loguru deepspeed

cp .env.example .env      # set NEMO_EXP_DIR to a path on a drive with ~22 GB free per run
./training_start.sh
```

`uv.lock` pins the exact resolved environment, and `perso/obsolete/working_pip_list` is the
`uv pip list` of the venv that actually produced the released model. Keep the `onnx<1.19` cap if
you re-lock: 1.19 hard-imports `ml_dtypes.float4_e2m1fn`, which the `numpy<2` pin makes
unsatisfiable, and NeMo then fails to import at all.

Trained on a single RTX 3090 Ti (24 GB). The VRAM budget in the config is fitted to that card;
see `perso/README.md` before changing `max_duration` or `cost_batching.budget`, which are coupled.

## What is deliberately not in this repo

- **`perso/oli_spoken_dataset/`**, 205 clips of my own voice reading sentences containing drug names aloud. It was the
  only real (non-synthetic) in-domain signal in the whole setup and one of the six equally weighted
  checkpoint monitors, so it is still referenced in `training_config.yaml` to keep that an accurate
  record. It will not resolve on a fresh clone. See the note at those entries for how to run
  without it.
- **Audio files.** The `.wav` files behind `perso/nemo_dataset/` and `perso/drug_sentence_dataset/`
  are gitignored; only the manifests are here. Regenerate them with the TTS container linked above.
- **Measured benchmark results.** The scripts are here, the numbers are not.
- **The docker-stopping block** that the real `training_start.sh` used to free GPU memory before a
  run. It hardcoded paths on my machine.
- **Large dataset trees**, reached in the real setup through gitignored symlinks
  (`perso/*_ignore-backups`). Paths in the config are relative to those symlinks, so nothing
  machine-specific is committed.

## Credits and licence

Upstream NeMo is Apache-2.0 and remains so; see [LICENSE](LICENSE). The patches and the `perso/`
scripts in this fork are offered under the same licence.

Note that the PARROT subset of the training corpus is CC BY-NC-SA and was used
**evaluation-only**, never for training.

Much of the work in this fork was done with AI coding agents, principally
[Claude Code](https://claude.com/claude-code) and, for a handful of commits, [aider](https://aider.chat).
The `Co-authored-by` trailers in the original unsquashed history recorded which model made which
change; that detail is lost in the squash, so it is stated here instead. The measurements,
diagnoses and design decisions were reviewed and run by me.

---

*Upstream NeMo's own README is available in the
[upstream repository](https://github.com/NVIDIA-NeMo/NeMo/blob/main/README.md).*
