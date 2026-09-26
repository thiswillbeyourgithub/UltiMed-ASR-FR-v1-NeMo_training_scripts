#!/usr/bin/env python
"""Build small deterministic validation slices from the UltiMed NeMo manifests.

The full UltiMed val split is 57.7k clips (~310 h), far too much audio to
decode at every validation round. This script samples a fixed, seeded subset
per source so validation stays cheap while still tracking every source.

Every UltiMed slice samples one category out of the RELEASE-WIDE split
(top-level val.jsonl / test.jsonl, the split the trainer's train.jsonl comes
from). They must NOT come from the per-source <source>/val.jsonl: those are an
independent per-source partition written by 01_build_nemo_manifest.py before
02_combine_nemo_manifests.py re-split the whole corpus, so a third to three
quarters of a per-source val manifest sits in the global train.jsonl (measured
2026-09-25: dictionary 33%, PARHAF 61%, acronyms 67%, drugs 72%). Sampling val slices
from there made validation score training clips (every count still added up,
since both are complete partitions, which is why it went unnoticed). The
trainer's data_leak_check now refuses such a config.

Slices are written next to their source manifest because audio paths inside
are relative to the manifest's directory.

Clips longer than --max-duration are excluded from the validation slices:
validation computes the RNNT loss lattice, whose memory scales with (batch
padded length), so one 60 s clip in a batch would spike VRAM. Training applies
the same 45 s cap anyway.

The test slices serve the final ONNX benchmark. They keep every duration, since
that benchmark decodes one clip at a time and must represent the split as
released: 10,000 clips in total, every acronym and drug clip plus fixed PARHAF
and dictionary samples.

PARROT is eval-only (CC BY-NC-SA), absent from the release-wide splits, and
never trained on, so its out-of-domain monitor slice comes from PARROT/test.jsonl.

One TRAINING subset is written here too, because it is the same "one category
of a release-wide split" filter: train.drugs.jsonl holds EVERY drugs row of
train.jsonl (no sampling, no duration cap: the trainer applies its own
max_duration). The training config lists it after train.jsonl extra times to
upsample the drug names, which are only ~2.3% of the training audio otherwise
(NeMo reads a manifest listed N times N times per epoch). Its rows are a subset
of train.jsonl, so data_leak_check treats them like any other training row.

Made with Claude Code.

Usage:
    python perso/build_ultimed_val_slices.py \
        [--nemo-files-dir ./perso/ultimed_data_ignore-backups/NeMO_files]
"""

import argparse
import json
import random
from pathlib import Path

# (source manifest, category or None for all rows, output, sample size or None
#  for every eligible row, duration-capped). Paths are relative to
#  --nemo-files-dir.
SLICES = [
    ("val.jsonl", "dictionary", "val.dictionary.down-600.jsonl", 600, True),
    ("val.jsonl", "parhaf", "val.parhaf.down-300.jsonl", 300, True),
    ("val.jsonl", "drugs", "val.drugs.down-300.jsonl", 300, True),
    ("val.jsonl", "acronyms", "val.acronyms.down-150.jsonl", 150, True),
    ("PARROT/test.jsonl", None, "PARROT/test.down-200.jsonl", 200, True),
    ("test.jsonl", "dictionary", "test.dictionary.down-5290.jsonl", 5290, False),
    ("test.jsonl", "parhaf", "test.parhaf.down-2500.jsonl", 2500, False),
    ("test.jsonl", "drugs", "test.drugs.down-2059.jsonl", 2059, False),
    ("test.jsonl", "acronyms", "test.acronyms.down-151.jsonl", 151, False),
    ("train.jsonl", "drugs", "train.drugs.jsonl", None, False),
]

SEED = 42


def write_slice(rows: list[str], n: "int | None", out: Path) -> None:
    rng = random.Random(SEED)
    picked = rows if n is None or len(rows) <= n else rng.sample(rows, n)
    out.write_text("\n".join(picked) + "\n")
    hours = sum(json.loads(r)["duration"] for r in picked) / 3600
    print(f"{out}: {len(picked)} clips ({hours:.2f} h) from {len(rows)} eligible")


def read_rows(src: Path, keep) -> list[str]:
    rows = []
    with src.open() as f:
        for line in f:
            line = line.strip()
            if line and keep(json.loads(line)):
                rows.append(line)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nemo-files-dir",
        default="./perso/ultimed_data_ignore-backups/NeMO_files",
        help="UltiMed NeMO_files directory (release-wide splits + PARROT/)",
    )
    parser.add_argument("--max-duration", type=float, default=45.0)
    parser.add_argument("--min-duration", type=float, default=1.0)
    args = parser.parse_args()

    base = Path(args.nemo_files_dir)
    for src_name, category, out_name, n, capped in SLICES:

        def keep(r: dict, category=category, capped=capped) -> bool:
            if category is not None and r["category"] != category:
                return False
            return not capped or args.min_duration <= r["duration"] <= args.max_duration

        write_slice(read_rows(base / src_name, keep), n, base / out_name)

if __name__ == "__main__":
    main()
