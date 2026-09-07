"""Build the medical calibration-audio folder for the ONNX SmoothQuant export.

The int8 encoder export (quantize-int8-smoothquant.py in the ONNX model repo)
calibrates activation ranges on whatever --audio / --fleurs-dir provide. The
reference (non-finetuned) build calibrated on balanced FLEURS plus 8 public
speeches; for the UltiMed fine-tune the deployment domain is French medical
dictation, so this script adds that domain to the pool, and it does so from
TRAIN-split clips only, so every eval set (UltiMed val/test, FLEURS
validation, the 8 speeches) stays strictly held out of calibration.

Two kinds of files are produced, because calibration windows are per-file:
  - short/: individual train clips resampled to 16 kHz mono s16 WAV,
    balanced across the UltiMed categories (dictionary/PARHAF/drugs/acronyms)
    plus drug_sentence, for short-window domain coverage;
  - long "sessions": concatenations of consecutive-sampled train clips with
    0.4 s silence gaps, ~4 min each, mimicking a real dictation session.
    These give the quantizer long-window coverage (the failure mode
    SmoothQuant exists to fix) WITHOUT touching the held-out speeches the
    long-audio benchmark uses.

Everything is deterministic (--seed) and idempotent (existing outputs kept).

Written with Claude Code.
"""

import argparse
import json
import random
import subprocess
from pathlib import Path

ULTIMED_MANIFEST = Path("./perso/ultimed_data_ignore-backups/NeMO_files/train.jsonl")
DRUG_MANIFEST = Path("./perso/drug_sentence_dataset/train.json")


def log(msg: str) -> None:
    print(msg, flush=True)


def load_rows(manifest: Path):
    rows = []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # UltiMed paths are relative to the manifest dir; drug_sentence
            # paths are relative to the repo root (the training cwd).
            path = (manifest.parent / r["audio_filepath"]).resolve()
            if not path.exists():
                path = Path(r["audio_filepath"]).resolve()
            rows.append((path, float(r.get("duration", 0.0))))
    return rows


def category(path: Path) -> str:
    for part in ("dictionary", "PARHAF", "drugs", "acronyms"):
        if part in path.parts:
            return part
    return "other"


def ffmpeg(args):
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.strip()[:300]}")


def build_short(rows_by_cat, out_dir: Path, per_cat: int, rng: random.Random):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for cat, rows in sorted(rows_by_cat.items()):
        picks = rng.sample(rows, min(per_cat, len(rows)))
        for path, _ in picks:
            out = out_dir / f"med_short_{cat}_{path.stem}.wav"
            if not out.exists():
                ffmpeg(["-i", str(path), "-ac", "1", "-ar", "16000",
                        "-sample_fmt", "s16", str(out)])
            n += 1
    log(f"short windows: {n} clips in {out_dir}")


def build_sessions(rows, out_dir: Path, n_sessions: int, target_s: float,
                   rng: random.Random):
    out_dir.mkdir(parents=True, exist_ok=True)
    silence = out_dir / "_silence_400ms.wav"
    if not silence.exists():
        ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                "-t", "0.4", "-sample_fmt", "s16", str(silence)])
    pool = rows[:]
    rng.shuffle(pool)
    it = iter(pool)
    for s in range(n_sessions):
        out = out_dir / f"med_session_{s}.wav"
        if out.exists():
            log(f"{out.name} exists, kept")
            continue
        picked, acc = [], 0.0
        for path, dur in it:
            picked.append(path)
            acc += dur + 0.4
            if acc >= target_s:
                break
        inputs, parts = [], []
        for i, p in enumerate(picked):
            inputs += ["-i", str(p)]
            parts.append(f"[{2*i}:a]")
            inputs += ["-i", str(silence)]
            parts.append(f"[{2*i+1}:a]")
        graph = "".join(parts) + f"concat=n={len(parts)}:v=0:a=1[out]"
        ffmpeg([*inputs, "-filter_complex", graph, "-map", "[out]",
                "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(out)])
        log(f"{out.name}: {len(picked)} clips, ~{acc:.0f}s")
    silence.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="./perso/calibration_audio_medical")
    ap.add_argument("--per-category", type=int, default=8)
    ap.add_argument("--sessions", type=int, default=6)
    ap.add_argument("--session-sec", type=float, default=240.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ultimed = load_rows(ULTIMED_MANIFEST)
    rows_by_cat = {}
    for path, dur in ultimed:
        if 2.0 <= dur <= 38.0:
            rows_by_cat.setdefault(category(path), []).append((path, dur))
    rows_by_cat["drug_sentence"] = [
        (p, d) for p, d in load_rows(DRUG_MANIFEST) if 2.0 <= d <= 38.0
    ]
    log("pool sizes: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(rows_by_cat.items())))

    out = Path(args.out)
    build_short(rows_by_cat, out, args.per_category, rng)
    build_sessions([r for rows in rows_by_cat.values() for r in rows],
                   out, args.sessions, args.session_sec, rng)
    total = sorted(out.glob("*.wav"))
    log(f"DONE: {len(total)} wav files in {out}")


if __name__ == "__main__":
    main()
