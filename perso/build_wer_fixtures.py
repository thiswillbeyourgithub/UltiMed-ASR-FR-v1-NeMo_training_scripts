"""Build FLEURS-layout WER fixtures for the medical eval sets.

parakeet_web's scripts/wer-quants.py --manifest mode resolves every clip as
<manifest dir>/wavs_validation/<basename(audio_filepath)>, i.e. the FLEURS
layout. The UltiMed / drug_sentence eval manifests instead point into
per-category subfolders, so this script materializes each eval set as
  <out>/<label>/validation.json      (audio_filepath = flat unique basename)
  <out>/<label>/wavs_validation/     (symlinks to the real audio files)
which makes the medical WER benchmark a plain multi-manifest wer-quants run.

Idempotent; symlinks and manifests are rewritten each run. Written with
Claude Code.
"""

import argparse
import json
from pathlib import Path

# label -> eval manifest (the same downsampled val sets the training monitor
# used for the in-domain part of combined_macro_val_wer, plus acronyms)
SETS = {
    "med_dictionary": "./perso/ultimed_data_ignore-backups/NeMO_files/dictionary/val.down-600.jsonl",
    "med_parhaf": "./perso/ultimed_data_ignore-backups/NeMO_files/PARHAF/val.down-300.jsonl",
    "med_drugs": "./perso/ultimed_data_ignore-backups/NeMO_files/drugs/val.down-300.jsonl",
    "med_acronyms": "./perso/ultimed_data_ignore-backups/NeMO_files/acronyms/val.down-150.jsonl",
    "med_drug_sentence": "./perso/drug_sentence_dataset/val.json",
    # PARROT is eval-only: unlike the five val.down-* sets above it was never
    # used to monitor training, so it is the one genuinely held-out medical set.
    "med_parrot": "./perso/ultimed_data_ignore-backups/NeMO_files/PARROT/test.down-200.jsonl",
}


def resolve(manifest: Path, audio_filepath: str) -> Path:
    # UltiMed paths are relative to the manifest dir; drug_sentence paths are
    # relative to the repo root (same rule as build_calibration_medical.py).
    path = (manifest.parent / audio_filepath).resolve()
    if not path.exists():
        path = Path(audio_filepath).resolve()
    return path


def build(label: str, manifest: Path, out_root: Path) -> None:
    out_dir = out_root / label
    wav_dir = out_dir / "wavs_validation"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows, missing = [], 0
    with open(manifest, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            src = resolve(manifest, r["audio_filepath"])
            if not src.exists():
                missing += 1
                continue
            name = f"{i:05d}_{src.name}"
            link = wav_dir / name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(src)
            rows.append({"audio_filepath": name, "text": r["text"],
                         "duration": r.get("duration")})
    with open(out_dir / "validation.json", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{label}: {len(rows)} clips ({missing} missing) -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="./perso/ultimed_data_ignore-backups/wer_fixtures")
    args = ap.parse_args()
    out_root = Path(args.out)
    for label, mf in SETS.items():
        build(label, Path(mf), out_root)
    print("DONE")


if __name__ == "__main__":
    main()
