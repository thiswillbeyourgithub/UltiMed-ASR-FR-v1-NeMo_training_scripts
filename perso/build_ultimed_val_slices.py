#!/usr/bin/env python
"""Build small deterministic validation slices from the UltiMed NeMo manifests.

The full UltiMed val split is 57.7k clips (~310 h), far too much audio to
decode at every validation round. This script samples a fixed, seeded subset
per source so validation stays cheap while still tracking every source.

Slices are written NEXT TO their source manifest (val.down-<N>.jsonl) because
audio paths inside are relative to the manifest's directory.

Clips longer than --max-duration are excluded: validation computes the RNNT
loss lattice, whose memory scales with (batch padded length), so one 60 s
clip in a batch would spike VRAM. Training applies the same 45 s cap anyway.

Made with Claude Code.

Usage:
    python perso/build_ultimed_val_slices.py \
        [--nemo-files-dir ./perso/ultimed_data_ignore-backups/NeMO_files]
"""

import argparse
import json
import random
from pathlib import Path

# (subdir, source manifest, output name, sample size)
SLICES = [
    ("dictionary", "val.jsonl", "val.down-600.jsonl", 600),
    ("PARHAF", "val.jsonl", "val.down-300.jsonl", 300),
    ("drugs", "val.jsonl", "val.down-300.jsonl", 300),
    ("acronyms", "val.jsonl", "val.down-150.jsonl", 150),
    # PARROT is eval-only (CC BY-NC-SA): never trained on, sliced from test
    # for use as an out-of-domain radiology monitor during validation.
    ("PARROT", "test.jsonl", "test.down-200.jsonl", 200),
]

SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nemo-files-dir",
        default="./perso/ultimed_data_ignore-backups/NeMO_files",
        help="Directory holding the per-source UltiMed NeMo manifest subdirs",
    )
    parser.add_argument("--max-duration", type=float, default=45.0)
    parser.add_argument("--min-duration", type=float, default=1.0)
    args = parser.parse_args()

    base = Path(args.nemo_files_dir)
    for subdir, src_name, out_name, n in SLICES:
        src = base / subdir / src_name
        rows = []
        with src.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if args.min_duration <= row["duration"] <= args.max_duration:
                    rows.append(line)
        rng = random.Random(SEED)
        picked = rows if len(rows) <= n else rng.sample(rows, n)
        out = base / subdir / out_name
        out.write_text("\n".join(picked) + "\n")
        hours = sum(json.loads(r)["duration"] for r in picked) / 3600
        print(f"{out}: {len(picked)} clips ({hours:.2f} h) from {len(rows)} eligible")


if __name__ == "__main__":
    main()
