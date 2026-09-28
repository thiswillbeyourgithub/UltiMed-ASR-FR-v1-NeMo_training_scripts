"""Transcribe every clip of the given NeMo manifests with a checkpoint, write hypotheses only.

Output: one ``{"audio": <absolute path>, "hyp": <text>}`` per line. Labels are NOT
stored: the consumer (UltiMed-ASR-FR-v1-scripts ``06_hotfixes/04_flag_asr_defects.py``,
which flags TTS preambles and skipped phrases for the release's drop step) joins them
from the manifests itself, so the labels may be renormalized between the two runs.

Resumable: clips already in <out.jsonl> are skipped, and hypotheses are flushed every
CHUNK clips. Batch size comes from the BS env var (default 16: a 24 GB card shared with
another job ran out of memory at 64). The run is usually CPU-bound (audio decoding),
not GPU-bound. First used 2026-09-28 with run 1.6.0 step 8000 on all of UltiMed v1.3.

Run from the NeMo repo root:
    BS=32 ./.venv/bin/python perso/transcribe_manifests.py <ckpt> <out.jsonl> <manifest> [<manifest> ...]

This file was written by Claude Code.
"""
import json
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, ".")
sys.path.insert(0, "perso")
from evaluate_checkpoint import load_model  # noqa: E402

ckpt, out_path, manifests = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
done = set()
if out_path.exists():
    done = {json.loads(l)["audio"] for l in open(out_path)}
todo = []
for m in manifests:
    base = Path(m).resolve().parent
    for line in open(m):
        r = json.loads(line)
        a = os.path.normpath(base / r["audio_filepath"]) if not os.path.isabs(r["audio_filepath"]) else r["audio_filepath"]
        if a not in done:
            done.add(a)
            todo.append(a)
print(f"{len(todo)} clips to transcribe", flush=True)
if not todo:
    sys.exit(0)

cfg = OmegaConf.load("perso/training_config.yaml")
model, _ = load_model(cfg, ckpt)
model.change_decoding_strategy(cfg.decoding)
model = model.eval().cuda()
CHUNK = 2000  # flush to disk every CHUNK clips so a crash loses little
with open(out_path, "a") as out, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for i in range(0, len(todo), CHUNK):
        part = todo[i:i + CHUNK]
        hyps = model.transcribe(part, batch_size=int(os.environ.get("BS", 16)), verbose=False)
        for a, h in zip(part, hyps):
            out.write(json.dumps({"audio": a, "hyp": h.text if hasattr(h, "text") else h}, ensure_ascii=False) + "\n")
        out.flush()
        print(f"{i + len(part)}/{len(todo)}", flush=True)
