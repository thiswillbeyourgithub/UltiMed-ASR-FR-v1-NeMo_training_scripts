# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Refuse to train when a validation or test sample is also a training sample.

Written after the UltiMed validation slices (``NeMO_files/<source>/val.down-N``)
turned out to come from an older per-source split while training used the
release-wide ``NeMO_files/train.jsonl``: 205/600 dictionary, 174/300 PARHAF,
232/300 drugs and 103/150 acronyms validation clips were training clips, so
checkpoint selection and every number measured on them was partly on seen
data. Nothing flagged it because every manifest looked fine on its own.

What is compared
----------------
Every manifest the config lists under ``model.train_ds``, ``model.validation_ds``
and ``model.test_ds`` (plain ``manifest_filepath`` or ``ds_item`` lists) is read,
and each row is reduced to two fingerprints:

* **text**: SHA-256 of the normalised transcript (lowercased, punctuation
  stripped, whitespace collapsed), so ``"Le patient."`` and ``"le patient"``
  collide. Normalising matters because the leak that prompted this module
  would have hidden behind a casing change in a re-export.
* **audio**: SHA-256 of the audio file BYTES (plus ``offset``/``duration`` when
  a row slices a longer file, since two slices of one file are different
  samples). A renamed or copied file still collides, which a path comparison
  would miss.

Then ``train & (val | test)`` must be empty for both fingerprints, otherwise
:class:`DataLeakError` is raised before any model is built. ``val & test`` is
only logged: the same manual-eval manifest is sometimes listed in both on
purpose (e.g. a small real-speech set), and that does not leak into weights.

Why hashing stays cheap
-----------------------
The UltiMed train split alone is ~350 GB of FLAC, so hashing every training
file on each launch would add an hour. Identical files necessarily agree on
every cheap property, so the audio comparison narrows down in three stages,
each reading more bytes but on fewer files:

1. byte size (one ``os.stat``): a training file whose size matches no eval
   file is ruled out without being opened.
2. quick fingerprint: SHA-256 of the size plus the first and last 4 KiB. For
   FLAC the head holds the STREAMINFO block, which itself carries an MD5 of
   the decoded audio, so this is already close to conclusive. ~8 KiB per file,
   i.e. ~4 GB of reads for 486k training clips instead of ~350 GB.
3. full SHA-256, only for files whose quick fingerprint is shared between two
   roles. Only this decides a collision, so a quick-fingerprint false positive
   can never raise, and a real duplicate can never slip through stage 1 or 2.

Size alone is not a usable filter on its own: with ~70k eval clips most
training FLAC sizes match one of them, which is what made the first version
of this check hash nearly the whole training set.

Rows whose audio file does not exist (a manifest not mounted on this machine,
or a tarred set whose ``audio_filepath`` is a member name) still get the text
check; their audio is counted as unchecked and reported, not silently passed.

Written with Claude Code.
"""

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.utils import logging

# Only a handful of example keys are printed per overlap, the full count is in
# the message anyway and a 100k-line exception helps nobody.
_MAX_EXAMPLES = 5
_HASH_CHUNK = 1 << 20
_QUICK_BYTES = 4096

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


class DataLeakError(RuntimeError):
    """Raised when a training sample also appears in a validation or test manifest."""


def normalize_text(text: str) -> str:
    """Normalise a transcript so formatting differences do not hide a duplicate.

    Parameters
    ----------
    text : str
        Raw manifest ``text`` field.

    Returns
    -------
    str
        NFKC-folded, lowercased, punctuation removed, whitespace collapsed.
        ``"  Le Patient, 3 mg. "`` -> ``"le patient 3 mg"``.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class _Row:
    """One manifest row, reduced to what the comparison needs."""

    manifest: str
    audio_path: Optional[str]  # resolved absolute path, None if the file is missing
    segment: Tuple[Optional[float], Optional[float]]  # (offset, duration) if sliced
    text_hash: Optional[str]
    size: Optional[int] = None


@dataclass
class _Split:
    """All rows of one role (train / val / test) across its manifests."""

    name: str
    rows: List[_Row] = field(default_factory=list)
    n_missing_audio: int = 0


def _flatten_manifests(value) -> List[str]:
    """Turn a NeMo ``manifest_filepath`` value into a flat list of paths.

    NeMo accepts a string (possibly comma-separated), a list, or a list of
    lists (bucketed tarred sets), so all three are flattened here.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    out: List[str] = []
    for item in value:
        out.extend(_flatten_manifests(item))
    return out


def collect_manifests(ds_cfg) -> List[str]:
    """Every manifest path listed in one ``train_ds`` / ``validation_ds`` / ``test_ds`` block.

    Parameters
    ----------
    ds_cfg : DictConfig or dict or None
        The dataset block. Both the plain ``manifest_filepath`` key and the
        fork's ``ds_item`` list (one dict per named set) are read.

    Returns
    -------
    list of str
        Manifest paths, de-duplicated, in config order.
    """
    if ds_cfg is None:
        return []
    paths = _flatten_manifests(ds_cfg.get("manifest_filepath", None))
    for item in ds_cfg.get("ds_item", None) or []:
        paths.extend(_flatten_manifests(item.get("manifest_filepath", None)))
    return list(dict.fromkeys(paths))


def _read_split(name: str, manifests: Iterable[str]) -> _Split:
    split = _Split(name=name)
    for manifest in manifests:
        if not os.path.isfile(manifest):
            raise DataLeakError(f"data_leak_check: {name} manifest not found: {manifest}")
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text", row.get("normalized_text"))
                audio = row.get("audio_filepath")
                resolved = None
                if isinstance(audio, str):
                    # Same resolution rule the NeMo dataloaders apply: relative
                    # paths are tried against the manifest's own directory.
                    resolved = get_full_path(audio_file=audio, manifest_file=manifest, force_cache=False)
                    resolved = os.path.abspath(resolved) if os.path.isfile(resolved) else None
                if resolved is None:
                    split.n_missing_audio += 1
                # A row slicing a longer file is keyed by its slice. offset 0 or
                # absent means "from the start", keyed like a whole-file row, so
                # a leading slice conservatively collides with the full file.
                offset = row.get("offset") or None
                split.rows.append(
                    _Row(
                        manifest=manifest,
                        audio_path=resolved,
                        segment=(offset, row.get("duration") if offset else None),
                        text_hash=_sha256_text(text) if isinstance(text, str) and text.strip() else None,
                    )
                )
    return split


def _quick_fingerprint(path: str, size: int) -> str:
    """Stage-2 fingerprint: size + first and last ``_QUICK_BYTES`` bytes."""
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(_QUICK_BYTES))
        if size > 2 * _QUICK_BYTES:
            f.seek(-_QUICK_BYTES, os.SEEK_END)
            h.update(f.read(_QUICK_BYTES))
    return h.hexdigest()


def _audio_keys(splits: Dict[str, "_Split"]) -> Dict[str, Dict[tuple, _Row]]:
    """Map each role to ``{audio fingerprint: representative row}``.

    Implements the three-stage narrowing described in the module docstring.
    The returned keys are ``("full", sha256, segment)`` for files that needed
    a full hash, and ``("quick", fingerprint, segment)`` for files whose quick
    fingerprint is unique to one role (those cannot match anything in another
    role, so their key only has to be distinct).
    """
    for split in splits.values():
        for row in split.rows:
            if row.audio_path is not None:
                row.size = os.stat(row.audio_path).st_size
    eval_sizes = {r.size for name in ("val", "test") for r in splits[name].rows if r.size is not None}

    # Stage 1 + 2: quick fingerprint of every eval file and of the training
    # files that survive the size filter. Cached per path, since one file can
    # be listed in several manifests.
    quick: Dict[str, str] = {}
    candidates: Dict[str, List[_Row]] = {}
    for name, split in splits.items():
        rows = []
        for row in split.rows:
            if row.audio_path is None or (name == "train" and row.size not in eval_sizes):
                continue
            if row.audio_path not in quick:
                quick[row.audio_path] = _quick_fingerprint(row.audio_path, row.size)
            rows.append(row)
        candidates[name] = rows

    # Stage 3: full hash only where a quick fingerprint appears in 2+ roles.
    roles_per_fp: Dict[str, Set[str]] = {}
    for name, rows in candidates.items():
        for row in rows:
            roles_per_fp.setdefault(quick[row.audio_path], set()).add(name)
    full: Dict[str, str] = {}
    keys: Dict[str, Dict[tuple, _Row]] = {}
    for name, rows in candidates.items():
        table: Dict[tuple, _Row] = {}
        for row in rows:
            fp = quick[row.audio_path]
            if len(roles_per_fp[fp]) > 1:
                if row.audio_path not in full:
                    full[row.audio_path] = _sha256_file(row.audio_path)
                key = ("full", full[row.audio_path], row.segment)
            else:
                key = ("quick", fp, row.segment)
            table.setdefault(key, row)
        keys[name] = table
    logging.info(
        f"data_leak_check: {len(quick)} audio files quick-fingerprinted, {len(full)} fully hashed."
    )
    return keys


def _describe(label: str, overlap: Set, a: Dict, b: Dict) -> str:
    examples = []
    for key in list(overlap)[:_MAX_EXAMPLES]:
        ra, rb = a[key], b[key]
        examples.append(f"      {ra.audio_path or ra.manifest}  <->  {rb.audio_path or rb.manifest}")
    return f"  {label}: {len(overlap)} shared\n" + "\n".join(examples)


def check_data_leaks(train_manifests: List[str], val_manifests: List[str], test_manifests: List[str]) -> dict:
    """Raise :class:`DataLeakError` if any eval sample is also a training sample.

    Parameters
    ----------
    train_manifests, val_manifests, test_manifests : list of str
        Manifest paths per role, e.g. from :func:`collect_manifests`.

    Returns
    -------
    dict
        Overlap counts, e.g. ``{"train_val_text": 0, "train_val_audio": 0,
        "train_test_text": 0, ..., "val_test_audio": 3}``, for logging.

    Raises
    ------
    DataLeakError
        When ``train`` shares a text or audio fingerprint with ``val`` or
        ``test``. The message lists counts and a few example pairs.
    """
    splits = {
        name: _read_split(name, manifests)
        for name, manifests in (("train", train_manifests), ("val", val_manifests), ("test", test_manifests))
    }
    for split in splits.values():
        if split.n_missing_audio:
            logging.warning(
                f"data_leak_check: {split.n_missing_audio}/{len(split.rows)} {split.name} rows have no "
                "readable audio file, only their text is checked."
            )

    texts = {
        name: {r.text_hash: r for r in split.rows if r.text_hash is not None} for name, split in splits.items()
    }
    audio = _audio_keys(splits)

    counts: dict = {}
    fatal: List[str] = []
    soft: List[str] = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        for kind, table in (("text", texts), ("audio", audio)):
            overlap = table[a].keys() & table[b].keys()
            counts[f"{a}_{b}_{kind}"] = len(overlap)
            if overlap:
                (fatal if a == "train" else soft).append(_describe(f"{a} & {b} ({kind})", overlap, table[a], table[b]))

    if soft:
        logging.warning("data_leak_check: val and test share samples (not fatal):\n" + "\n".join(soft))
    if fatal:
        raise DataLeakError(
            "data_leak_check: training samples also appear in validation/test manifests, so eval metrics "
            "and checkpoint selection would be measured on seen data. Fix the manifests, or set "
            "data_leak_check.enabled=false if this run knowingly accepts it.\n" + "\n".join(fatal)
        )
    logging.info(
        f"data_leak_check: no train/eval overlap ({len(splits['train'].rows)} train rows, "
        f"{len(splits['val'].rows)} val, {len(splits['test'].rows)} test)."
    )
    return counts
