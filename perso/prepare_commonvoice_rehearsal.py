"""Build Common Voice 22.0 rehearsal manifests for the 1.4.x runs.

Common Voice (MCV v7.0) is part of parakeet-tdt-0.6b-v3's pretraining data,
which is exactly what rehearsal wants: replaying pretraining-like data while
fine-tuning on UltiMed fights catastrophic forgetting (fleurs stays untouched
as the external canary). Sentences in the CV tsvs are the original cased and
punctuated text, matching the model's output convention, so unlike
MLS/LibriSpeech they do not teach formatting-dropping.

Input: the fsicoli/common_voice_22_0 snapshot (original release layout:
per-split audio tar shards + tsv transcripts) as downloaded to the big drive.
The repo reaches it through the gitignored symlink
perso/downloaded_datasets_ignore-backups/commonvoice so no user-specific
absolute path lives in committed files.

Pipeline (idempotent, safe to re-run after a crash: existing FLACs are kept):
  1. read train.tsv (text) and clip_durations.tsv (duration) per language,
     filter to [1, 38] s and non-empty text, collapse unicode whitespace
     (French CV uses nbsp before ':;!?' which the model never emits);
  2. scan the tar shards once to learn which clips are actually on disk
     (en only has shards 0-7 of 29, so train.tsv alone would mostly name
     absent files); the shard index is cached next to the data so re-runs
     skip the scan;
  3. seeded shuffle of the on-disk pool, keep clips until the per-language
     hour target is reached (train_ds balances by audio-seconds, so hours
     are the knob: ~200 h fr + ~100 h en on top of UltiMed's 2398 h lands
     at ~11% rehearsal, 2:1 fr:en to match the deployment language without
     abandoning en), then a second pass extracts just those clips;
  4. ffmpeg mp3 -> 16 kHz mono 16-bit FLAC (the UltiMed precedent; libsndfile
     decodes FLAC natively and cheaply in the dataloader workers), niced to
     not starve a training run's dataloaders;
  5. write cv_<lang>_train.jsonl at the CV root with audio paths RELATIVE TO
     THE MANIFEST (NeMo's get_full_path resolves those), duration read back
     from each produced FLAC header: the cost batcher sizes batches from
     manifest durations, so they must reflect the real file, and a FLAC whose
     header cannot be read is dropped here rather than crashing training.

Languages and per-language hour targets come from --langs as CV locale codes
(e.g. "de:20,it:20,sv-SE:99,fi:99"); a target above the on-disk pool takes the
whole pool, which is how the small locales are used in full. 1.4.0 used the
default fr:200,en:100; 1.5.0 added de/it/sv-SE/fi on top.

Written with Claude Code.
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

MIN_S, MAX_S = 1.0, 38.0


def log(msg: str) -> None:
    print(msg, flush=True)


def load_candidates(cv_root: Path, lang: str):
    """path -> (duration_s, cleaned sentence) for usable train-split rows."""
    tdir = cv_root / "transcript" / lang
    sentences = {}
    dropped_text = 0
    with open(tdir / "train.tsv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            text = " ".join((row.get("sentence") or "").split())
            if not text or any(ord(c) < 32 for c in text):
                dropped_text += 1
                continue
            sentences[row["path"]] = text
    cands = {}
    dropped_dur = 0
    with open(tdir / "clip_durations.tsv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        assert header[0] == "clip", f"unexpected clip_durations header: {header}"
        for name, dur_ms in reader:
            text = sentences.get(name)
            if text is None:
                continue
            dur = int(dur_ms) / 1000.0
            if MIN_S <= dur <= MAX_S:
                cands[name] = (dur, text)
            else:
                dropped_dur += 1
    log(
        f"[{lang}] train.tsv rows kept: {len(cands)} "
        f"(dropped {dropped_text} empty/control text, {dropped_dur} outside {MIN_S}-{MAX_S}s), "
        f"pool {sum(d for d, _ in cands.values()) / 3600:.1f} h"
    )
    return cands


def shard_tars(cv_root: Path, lang: str):
    tars = sorted((cv_root / "audio" / lang / "train").glob("*.tar"))
    assert tars, f"no tar shards under audio/{lang}/train"
    return tars


def scan_members(cv_root: Path, lang: str) -> set:
    """Basenames of clips present in the shards on disk, cached across runs."""
    tars = shard_tars(cv_root, lang)
    cache = cv_root / f"shard_index_{lang}_{len(tars)}tars.txt"
    if cache.exists():
        members = set(cache.read_text(encoding="utf-8").split())
        log(f"[{lang}] shard index cache: {len(members)} clips in {len(tars)} tars")
        return members
    members = set()
    for tp in tars:
        with tarfile.open(tp) as tf:
            while True:
                m = tf.next()
                if m is None:
                    break
                if m.isfile() and m.name.endswith(".mp3"):
                    members.add(os.path.basename(m.name))
        log(f"[{lang}] indexed {tp.name} ({len(members)} clips so far)")
    tmp = cache.with_suffix(".part")
    tmp.write_text("\n".join(sorted(members)), encoding="utf-8")
    tmp.rename(cache)
    return members


def extract_selected(cv_root: Path, lang: str, wanted: set, mp3_dir: Path, flac_dir: Path):
    """One sequential pass per shard; extract wanted clips missing their FLAC."""
    tars = shard_tars(cv_root, lang)
    seen = set()
    extracted = 0
    for tp in tars:
        with tarfile.open(tp) as tf:
            while True:
                m = tf.next()
                if m is None:
                    break
                if not m.isfile():
                    continue
                base = os.path.basename(m.name)
                if base not in wanted:
                    continue
                seen.add(base)
                stem = os.path.splitext(base)[0]
                if (flac_dir / f"{stem}.flac").exists() or (mp3_dir / base).exists():
                    continue
                src = tf.extractfile(m)
                tmp = mp3_dir / (base + ".part")
                with open(tmp, "wb") as out:
                    while chunk := src.read(1 << 20):
                        out.write(chunk)
                tmp.rename(mp3_dir / base)
                extracted += 1
        log(f"[{lang}] scanned {tp.name}: {len(seen)}/{len(wanted)} wanted clips seen so far")
    missing = len(wanted) - len(seen)
    if missing:
        log(f"[{lang}] WARNING: {missing} selected clips absent from the shards on disk (kept out of the manifest)")
    return seen, extracted


def convert_all(mp3_dir: Path, flac_dir: Path, lang: str, jobs: int):
    """mp3 -> 16 kHz mono s16 FLAC, niced so a live training run keeps its CPU."""
    todo = [
        p for p in mp3_dir.glob("*.mp3")
        if not (flac_dir / f"{p.stem}.flac").exists()
    ]
    failed = []
    done = 0

    def one(mp3: Path):
        out = flac_dir / f"{mp3.stem}.flac"
        tmp = flac_dir / f"{mp3.stem}.part.flac"
        r = subprocess.run(
            ["nice", "-n", "19", "ionice", "-c3", "ffmpeg", "-nostdin", "-v", "error",
             "-y", "-i", str(mp3), "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(tmp)],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            tmp.rename(out)
            return None
        tmp.unlink(missing_ok=True)
        return f"{mp3.name}: {r.stderr.strip()[:150]}"

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for i, err in enumerate(ex.map(one, todo), 1):
            if err:
                failed.append(err)
            done += 1
            if i % 5000 == 0:
                log(f"[{lang}] converted {i}/{len(todo)}")
    log(f"[{lang}] conversion: {done - len(failed)} ok, {len(failed)} failed of {len(todo)} pending")
    for err in failed[:10]:
        log(f"[{lang}]   ffmpeg fail: {err}")
    return failed


def build_manifest(cv_root: Path, lang: str, selected, flac_dir: Path):
    """Manifest rows only for clips whose FLAC exists and parses."""
    manifest = cv_root / f"cv_{lang}_train.jsonl"
    rows = 0
    hours = 0.0
    dropped = 0
    with open(manifest, "w", encoding="utf-8") as out:
        for name, (_, text) in selected.items():
            stem = os.path.splitext(name)[0]
            flac = flac_dir / f"{stem}.flac"
            if not flac.exists():
                dropped += 1
                continue
            try:
                dur = sf.info(str(flac)).duration
            except Exception:
                dropped += 1
                continue
            if not (MIN_S <= dur <= MAX_S):
                dropped += 1
                continue
            out.write(json.dumps(
                {"audio_filepath": f"clips_16k/{lang}/{stem}.flac",
                 "duration": round(dur, 3), "text": text},
                ensure_ascii=False) + "\n")
            rows += 1
            hours += dur
    log(f"[{lang}] manifest {manifest.name}: {rows} clips / {hours / 3600:.1f} h ({dropped} selected clips dropped)")
    return rows, hours


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cv-root", default="./perso/downloaded_datasets_ignore-backups/commonvoice")
    ap.add_argument(
        "--langs", default="fr:200,en:100",
        help="comma-separated CV locale:target_hours pairs (locale as in the CV "
             "layout, e.g. sv-SE). A target larger than the on-disk pool takes "
             "the whole pool.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=4, help="parallel ffmpeg processes (keep low while a training run is live)")
    ap.add_argument("--keep-mp3", action="store_true", help="keep the extracted mp3s instead of deleting them at the end")
    args = ap.parse_args()

    cv_root = Path(args.cv_root).resolve()
    targets = {}
    for pair in args.langs.split(","):
        lang, _, hours = pair.strip().partition(":")
        targets[lang] = float(hours)
    summary = {}
    for lang, hours_target in targets.items():
        members = scan_members(cv_root, lang)
        cands = load_candidates(cv_root, lang)
        n_all = len(cands)
        cands = {k: v for k, v in cands.items() if k in members}
        log(f"[{lang}] on-disk pool: {len(cands)} of {n_all} usable rows, "
            f"{sum(d for d, _ in cands.values()) / 3600:.1f} h")
        order = sorted(cands)
        random.Random(args.seed).shuffle(order)
        selected = {}
        acc = 0.0
        for name in order:
            if acc >= hours_target * 3600:
                break
            selected[name] = cands[name]
            acc += cands[name][0]
        log(f"[{lang}] selected {len(selected)} clips / {acc / 3600:.1f} h (target {hours_target} h, seed {args.seed})")

        mp3_dir = cv_root / "extracted_mp3" / lang
        flac_dir = cv_root / "clips_16k" / lang
        mp3_dir.mkdir(parents=True, exist_ok=True)
        flac_dir.mkdir(parents=True, exist_ok=True)
        seen, extracted = extract_selected(cv_root, lang, set(selected), mp3_dir, flac_dir)
        log(f"[{lang}] extracted {extracted} new mp3s")
        convert_all(mp3_dir, flac_dir, lang, args.jobs)
        summary[lang] = build_manifest(cv_root, lang, selected, flac_dir)

        if not args.keep_mp3:
            for p in mp3_dir.glob("*"):
                p.unlink()
            log(f"[{lang}] cleaned extracted mp3s")

    total_h = sum(h for _, h in summary.values())
    log(f"ALL DONE: {' + '.join(f'{lang} {r} clips/{h / 3600:.1f}h' for lang, (r, h) in summary.items())} = {total_h / 3600:.1f} h rehearsal")


if __name__ == "__main__":
    main()
