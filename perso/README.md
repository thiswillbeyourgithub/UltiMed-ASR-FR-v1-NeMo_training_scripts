Fine-tuning notes for [Olicorne/parakeet-tdt-0.6b-v3-UltiMed-onnx](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-UltiMed-onnx).
Related repositories: the dataset [Olicorne/UltiMed-ASR-FR-v1](https://huggingface.co/datasets/Olicorne/UltiMed-ASR-FR-v1),
the build recipe [UltiMed-ASR-FR-v1-scripts](https://github.com/thiswillbeyourgithub/UltiMed-ASR-FR-v1-scripts),
and the TTS container that synthesized its audio, [UltiMed-ASR-FR-v1-Voxtral](https://github.com/thiswillbeyourgithub/UltiMed-ASR-FR-v1-Voxtral).

---

created my own dataset with gpt-4o-mini-tts because it was the only one to be able to often get the word pronunciations right. Including units etc. So created my text dataset, because counted on augmentators to fix it.
I randomized the voice too


git clone https://github.com/NVIDIA-NeMo/NeMo

# last commit: 64420189

uname -a
Linux [hostname] 6.8.0-90-generic #91~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC [date] 2 x86_64 x86_64 x86_64 GNU/Linux

uv venv --python=3.10.12

# had to pip install this:
uv pip install nemo_toolkit[asr] onnx==1.18.0 numba-cuda==0.15.1 cuda-core cuda-bindings cuda-pathfinder cuda-python==13.1.1

    uv pip list output is in perso/obsolete/working_pip_list


# the slr download script:
https://huggingface.co/camenduru/NeMo/blob/main/scripts/dataset_processing/get_openslr_rir_data.py


couldn't figure out how to use tarred datasets

we can freeze the encoder, the decoder, or even both to only train the join network.

# full install script
uv pip install nemo_toolkit[asr] onnx==1.18.0 numba-cuda==0.15.1 cuda-core cuda-bindings cuda-pathfinder cuda-python==13.1.1 tensorboard>=2.19.0 setuptools==81.0.0 --force-reinstall loguru deepspeed

(added deepspeed)

# launch training

Use `./training_start.sh` (repo root). It relaunches training automatically
after a crash (resume picks up the last checkpoint), stops on clean finish or
Ctrl-C, and aborts after 5 consecutive sub-10-min failures:

`./training_start.sh`            (add `--debug` if needed)

(The version I actually ran also stopped my local GPU-hungry docker containers
before launching and offered to restart them at the end. That block hardcoded
paths on my machine, so it was removed for the public repo rather than
genericised.)

It reads `NEMO_EXP_DIR` from a `.env` file at the repo root, so run `cp .env.example .env` once and set the path to a directory on your external drive. The launcher sources that file and exports the variable. An already-exported `NEMO_EXP_DIR` still wins over the `.env` value, so `NEMO_EXP_DIR=/somewhere ./training_start.sh` overrides it for a single run. `.env` is gitignored because the path contains the username; only the placeholder `.env.example` is tracked.

The launcher refuses to start if the parent directory of `NEMO_EXP_DIR` does not exist. That is the unmounted-drive case: without the check, `mkdir -p` would silently rebuild the tree on the system disk and fill it with multi-GB checkpoints.

It also tees everything to
`perso/training_logs_ignore-backups/train_<timestamp>.log` as it happens, with
`latest.log` symlinked to the newest run, so you can follow (or have Claude
Code read) a run while it is still going.

Manual launch (the entry point is `speech_to_text_finetune_cached.py` for BOTH
modes; with `use_cached_encoder: false` in the config it trains the standard way
but keeps partial unfreeze, AdamSPD, macro metrics and frozen-module eval
pinning):

`NUMBA_CUDA_USE_NVIDIA_BINDING=1 HYDRA_FULL_ERROR=1 python examples/asr/speech_to_text_finetune_cached.py --config-dir=$PWD/perso --config-name=training_config`

Checkpoints, .nemo exports and tensorboard events go to `$NEMO_EXP_DIR/<exp_manager.name>/<exp_manager.version>/`, currently `parakeet-tdt-ultimed-french-medical/1.1.0/` (see `exp_manager.exp_dir` in `training_config.yaml`). A manual launch does not read `.env` on its own, and the config falls back to `./nemo_experiments` when the variable is unset, so either export it yourself or source the file first with `set -a; . ./.env; set +a`. Pointing it at another disk is what keeps the multi-GB checkpoints off the system drive.

Note that `resume_if_exists` searches that same directory, so a fresh disk means
previous runs are not picked up for resuming unless you move them there.

# VRAM: why batches are sized by duration cost, not by clip count

Measured on the RTX 3090 Ti (24564 MiB) for parakeet-tdt-0.6b-v3, 2026-08-21 (with Claude Code).

The problem: the RNNT/TDT joint builds a dense `[B, T, U, V]` lattice with V = 8198, where T = duration * 12.5 and U = tokens + 1. This corpus runs about 10 tokens per second, so U tracks T and **the memory a clip costs grows with the SQUARE of its duration**. A fixed batch size therefore has to be sized for the longest clip the manifest can produce, which wastes most of the GPU on the short clips.

Two things were wrong before and are now fixed in `training_config.yaml`:

1. The pretrained checkpoint ships `joint.fused_batch_size: 4`. With `batch_size: 4` the whole batch became a single fused sub-batch, so every clip was padded to the longest one in the batch and the cost was `B * max(T) * max(U) * V`. A batch of four 45 s clips asked for a single 20.87 GiB allocation and died. It is now set to 1, so cost follows `sum_i(T_i * U_i)` with no padding amplification.
2. `max_duration` was 45 s, which does not fit at any batch size.

The fix now in place is `model.train_ds.cost_batching`, implemented as `DurationCostBatchSampler` in `nemo/collections/asr/parts/utils/asr_batching.py` and wired into the dataloader by `apply_cost_batch_sampler` in `examples/asr/speech_to_text_finetune_cached.py`. It packs each batch up to a memory budget instead of a fixed clip count, so one batch is a single 38 s clip and the next is sixteen 14 s ones. The cost model is `cost = 4 * sum(d^2) + 6 * max(d^2)` with a budget of 15000, the second term being the transient of the single largest clip (under `fused_batch_size: 1` the joint runs one clip at a time, so the peak is set by the biggest clip, not by the sum). Batches are packed from windows of 512 shuffled clips, sorted by duration inside the window, because the encoder pads every clip up to the longest in its batch: mixing a 3 s clip with a 39 s one would spend most of the batch on padding.

The cost model was fitted on these ten fixed-batch measurements (200 clips all sitting at the cap with the longest transcripts available at that length, so every batch is the worst batch training can draw), peak VRAM out of 24564 MiB:

| max_duration | batch | cost | peak MiB | result |
|---|---|---|---|---|
| 30 | 2 | 12600 | 21130 | OK |
| 25 | 4 | 13750 | 23353 | OK |
| 40 | 1 | 16000 | 23409 | OK |
| 30 | 3 | 16200 | 23983 | OK |
| 35 | 2 | 17150 | 23948 | OK |
| 25 | 6 | 18750 | 23940 | OK |
| 30 | 4 | 19800 | - | OOM |
| 45 | 1 | 20250 | - | OOM |
| 35 | 4 | 26950 | - | OOM |
| 40 | 4 | 35200 | - | OOM |

Every passing configuration scores at or below 18750 and every failing one at or above 19800, so the model separates them cleanly. That clean split is why the cost model is trusted enough to drive batching.

Important and non-obvious: **nvidia-smi overstates the requirement here**. The worst-case fixture reports a peak of about 23.6 of 24.5 GB, but that is the PyTorch caching allocator holding on to freed blocks, not need. Three pieces of evidence: budgets of 14000 and 15000 report the same peak (23664 and 23652 MiB), which would be impossible if the reading tracked real demand; re-running the worst case while a separate process held 1.0 GB of the card still PASSED; and the OOM messages when a second process held 2.0 GB and 2.5 GB both put the true peak at about 21.3 GiB of the 23.54 GiB torch sees. So the real headroom on the worst possible batch mix is about 2.2 GB, not the ~900 MiB nvidia-smi suggests. On the real manifests the peak is 22490 MiB.

Validation deserved its own check rather than an argument, because `compute_eval_loss` is not set in the config and NeMo defaults it to true: validation builds the same joint lattice training does, just without gradients. Nine of the fifteen validation sets score over the 15000 training budget on the cost model (worst: fleurs_it at 37344, six clips near 40 s), and `num_sanity_val_steps` cannot catch a problem there because validation sets have `shuffle: false`, so the sanity pass sees the *first* two batches rather than the longest. A validation OOM would also land about 3.5 h in, before the first checkpoint exists, and would be too slow to trip the launcher's fast-fail guard, so it would relaunch forever.

Measured instead, 2026-08-22 (with Claude Code): a fixture holding the **two longest batches of every one of the 15 validation sets**, run through the real training script after one training batch so the optimizer state is resident. Result: exit 0, **peak 21363 of 24564 MiB, no OOM**, all 15 dataloaders completed. So validation is the cheaper phase despite the longer clips, and the uncapped 44.8 s validation clips are fine. Re-run that check if validation batch sizes or manifests change.

The measured payoff, both columns on the real manifests:

| | fixed batch 4, cap 25 s | cost batching, cap 38 s |
|---|---|---|
| batches per epoch | 114,459 | 63,452 (1 to 16 clips, mean 7.7) |
| audio per batch | 69 s | 138 s |
| throughput | 142.2 audio-s per wall-s | 221.4 audio-s per wall-s |
| corpus hours kept | 86.8% | 96.6% |
| PARHAF hours kept | 18.6% (51 h) | 69.0% (190 h) |

So it is both faster AND trains on more data, because the old fixed batch size was sized for a worst case that almost never occurred.

`accumulate_grad_batches` went from 16 to 8 so an optimizer step stays the same size: 8 x 7.7 = 62 clips and 1104 audio-s per step, against 16 x 4 = 64 clips and 1100 audio-s before. `sched.max_steps` is now 155000 and `val_check_interval` 20000 to match the new batch count.

Known and accepted side effect: the RNNT loss reduces with mean_batch, and gradient accumulation weights every micro-batch equally, so a batch of one long clip carries the same gradient weight as a batch of sixteen short ones. Long clips are therefore upweighted roughly 8x per clip. That is wanted here (long-form is the weak point this fine-tune targets) but it is a real reweighting of the corpus, not a neutral change.

How to change it safely: `max_duration` and `cost_batching.budget` are coupled, because a clip alone costs `10 * d^2`, so the largest clip the budget can hold is `sqrt(budget / 10)`, which is 38.7 s at budget 15000. Change the two together. The sampler warns at startup if any clip exceeds that, and emits it alone anyway, which may OOM.

To re-check the margin:

`NEMO_EXP_DIR=/tmp/oom_check ./perso/oom_margin_check.sh`

The fixture now holds clips at the duration that exactly saturates the budget for each batch size in 1, 2, 4, 8, 16, so one run covers the whole frontier (a batch is one 38 s clip or sixteen 14 s ones and both sit at the budget) rather than a single batch shape. Re-run it after changing `max_duration`, `cost_batching.budget` or `joint.fused_batch_size`.

Future option not taken: NeMo can batch dynamically via lhotse (`model.train_ds.use_lhotse: true` with `batch_duration` and `quadratic_duration`), which is the off-the-shelf way to size each batch by cost instead of by clip count. It was not taken because the lhotse data path never reads the `augmentor` block this config relies on (white noise, gain, RIR, which matter a lot for a single-voice TTS corpus) and it changes epoch and step semantics. The custom sampler was written instead precisely because it keeps the classic NeMo dataset path and the augmentor intact.

# Forgetting protection: the optimizer, the monitor, and whether the config is real

Worth reading before touching the optimizer, the validation list or the schedule (investigated 2026-08-22 and 2026-08-23 with Claude Code). The common thread is that all of it was configured correctly and some of it was not happening.

**AdamSPD was applying zero decay.** The Selective Projection Decay criterion read `condition = -sum(grad * (p - pre))` and decayed when `condition < 0`, which selects the case where the step moves *toward* the pretrained weights. That is also the case where the ratio term is negative and gets hardtanh-clamped to zero, so the two mistakes cancelled and no decay was ever applied: `weight_decay` of 0.0, 0.01 and even 1.0 produced bit-identical trajectories. The run was plain Adam with the configured `weight_decay: 0.01` inert. Fixed by dropping the negation, and pinned by `tests/collections/asr/test_adam_spd.py`, two of whose tests fail on the old sign. If you ever see every `weight_decay` giving identical results again, that is the symptom.

Worth keeping in mind if you consider swapping the optimizer: AdamW would be a downgrade here, because it decays toward *zero* rather than toward the pretrained weights, which is the wrong direction when the whole goal is to not forget. That is what L2-SP and SPD exist to fix.

**The checkpoint monitor now includes `oli_drug_sentence`.** (Not published:
that set is 205 clips of my own voice, so it ships with neither this repo nor
the released dataset. Its entries are still in `training_config.yaml` because
they record how the released checkpoint was actually selected; see the note
there for how to run the config without it.) Every training hour is synthetic, 98.7% of it a single Voxtral `fr_female` voice, and FLEURS is real speech but general rather than medical. So `oli` (205 clips of real recorded human speech saying drug names) is the only thing in the whole setup that measures the actual deployment condition. Before it was added, checkpoints were selected on TTS performance alone. It is one of six equally weighted sources rather than the monitor by itself, because at 30 minutes it is also the noisiest, and `save_top_k: 5` leaves a band of finalists to compare on it directly afterwards.

**Run length is set by the cosine, not by a time budget.** `optim.sched.max_steps` and `trainer.max_epochs` must move together (7,932 optimizer steps per epoch): `max_steps` sets the *shape* of the anneal, so a schedule longer than the run leaves the best checkpoint taken mid-anneal at a high LR, never having had its final decay. A completed cosine at N epochs beats one truncated at N-of-20. Currently 12 epochs and 95,000 steps, about 5.7 days.

**The config is audited before every run.** Six settings in `training_config.yaml` turned out to do nothing, and every one of them looked correct in the file, so `config_audit` now crashes the run before the first training step rather than letting it discover this on day five. Two layers, because the failure modes differ: layer 1 records every key read during setup and rejects any leaf nothing touched (wrong nesting level, typo, dead consumer), while layer 2 re-reads the settings that matter back off the objects that were actually built (decoder, optimizer, batch sampler, dataloaders), which is the only way to catch a value that is read and then discarded downstream. A third mode, wired but inert, is invisible to both and is what the AdamSPD tests above are for.

When it fires, the message names the keys. Either the key is genuinely broken and needs fixing, or it is dead on purpose (a feature switched off for this run) and belongs in `config_audit.allow_unused` with a comment saying why. Do not disable the audit to get past it. Turning on the cached-encoder path means removing `encoder_cache.*`, `encoder_cache_dir` and `model.latent_augment.*` from that list, since they stop being inert.

The audit's first find was that the top-level `name:` did nothing, so runs landed in `<exp_dir>/default/<version>/` and anything else ever trained on that disk would have piled into the same directory. `exp_manager.name: ${name}` fixes it, done on 2026-08-23 while `nemo_experiments/` was still empty and nothing had to move. Do not change it once a run has started: `resume_if_exists` looks in `<exp_dir>/<name>/<version>/checkpoints`, so a renamed directory means resume finds nothing and restarts from scratch without saying so.

`name` stays in `allow_unused` even so, because OmegaConf resolves `${name}` through `_get_node` and layer 1 only sees the three normal read paths. Layer 2 covers it instead, by checking the run directory the trainer was actually given contains the name. Worth remembering as a general limit: any key consumed only through an interpolation is invisible to layer 1 and needs a layer 2 check to be genuinely pinned.

**On the four `perso/` entries that look like duplicates:** `cached_encoder_dataset.py`, `latent_augment.py`, `speech_to_text_finetune.py` and `speech_to_text_finetune_cached.py` (the last being the training entry point) are symlinks into the live tree, not copies, so there is only ever one real file. They started life as genuine byte-identical copies and were converted once that was noticed. A fifth, `perso/adam_spd.py`, was deleted outright when the sign fix landed, because following the `_target_` its own docstring advertised would have silently selected the broken optimizer.

NOTE (2026-08-20): onnx must stay `<1.19` in this venv: onnx 1.19 hard-imports
`ml_dtypes.float4_e2m1fn` (needs ml_dtypes>=0.5) but the numpy<2 pin holds
ml_dtypes at 0.4.1, so importing nemo dies at startup. `requirements/requirements.txt`
and `uv.lock` now cap it; if you re-lock, keep the cap.


I manually cleanup the drug sentence dataset to uncapitalize drug DCI that were in full caps like DEXAMETHASONE


To use tensorboard: `./tensorboard_start.sh` from the repo root. It reads the run directory from `.env` the same way `training_start.sh` does, so it always points at where the checkpoints and event files actually went, even once the runs live on another disk. `-p PORT` (default 6006) changes the port, and any other argument is passed straight through to tensorboard. It refuses to start if the run directory does not exist yet, and warns (without stopping) when it exists but holds no event file yet.

Without the script the raw command is `uvx --python 3.10.12 --with setuptools==81.0.0 --from tensorboard==2.20.0 tensorboard --logdir nemo_experiments/ --bind_all` (point `--logdir` at `$NEMO_EXP_DIR` if you moved the runs to another disk). The `setuptools==81.0.0` pin is needed either way: tensorboard 2.20 still imports `pkg_resources`.


Reminder to use nvtop during training to troubleshoot things


# Runtime benchmarks

`perso/run_grid_search_benchmark.sh` and `perso/run_grid_perf_benchmark.sh` drive
the ONNX runtime benchmark harness, which lives in the `parakeet_web` front end
rather than here. Both read `PARAKEET_WEB` and default to `$HOME/parakeet_web`,
so point it at your checkout:

`PARAKEET_WEB=/path/to/parakeet_web ./perso/run_grid_perf_benchmark.sh`

The measured results are not published, only the scripts.
