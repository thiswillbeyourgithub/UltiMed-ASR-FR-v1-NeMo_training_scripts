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

"""Tests for the train/eval data leak check.

The cases that matter are the leaks a naive check would miss: the same audio
under another file name, the same transcript with different casing and
punctuation, and a training file that is only hashed because its size matches
an eval file (the size pre-filter must never hide a real duplicate).

Written with Claude Code.
"""

import json

import pytest
from omegaconf import OmegaConf

from nemo.collections.asr.parts.utils.data_leak_check import (
    DataLeakError,
    check_data_leaks,
    collect_manifests,
    normalize_text,
)


def _write_manifest(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


@pytest.fixture
def corpus(tmp_path):
    """Three distinct fake audio files plus a byte-identical copy of the first one."""
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "a.flac").write_bytes(b"A" * 100)
    (audio / "b.flac").write_bytes(b"B" * 200)
    (audio / "c.flac").write_bytes(b"C" * 100)  # same size as a.flac, different bytes
    (audio / "a_copy.flac").write_bytes(b"A" * 100)
    return tmp_path


def test_normalize_text():
    assert normalize_text("  Le Patient, 3 mg. ") == "le patient 3 mg"
    assert normalize_text("L'aorte") == normalize_text("l aorte")


def test_clean_split_passes(corpus):
    train = _write_manifest(corpus / "train.json", [{"audio_filepath": "audio/a.flac", "text": "un"}])
    val = _write_manifest(corpus / "val.json", [{"audio_filepath": "audio/b.flac", "text": "deux"}])
    # c.flac has a.flac's size, so the train file gets hashed, and must still not collide.
    test = _write_manifest(corpus / "test.json", [{"audio_filepath": "audio/c.flac", "text": "trois"}])
    counts = check_data_leaks([train], [val], [test])
    assert all(v == 0 for v in counts.values()), counts


def test_renamed_audio_copy_is_a_leak(corpus):
    train = _write_manifest(corpus / "train.json", [{"audio_filepath": "audio/a.flac", "text": "un"}])
    val = _write_manifest(corpus / "val.json", [{"audio_filepath": "audio/a_copy.flac", "text": "autre"}])
    with pytest.raises(DataLeakError, match=r"train & val \(audio\): 1 shared"):
        check_data_leaks([train], [val], [])


def test_reformatted_text_is_a_leak(corpus):
    train = _write_manifest(corpus / "train.json", [{"audio_filepath": "audio/a.flac", "text": "Le patient."}])
    test = _write_manifest(corpus / "test.json", [{"audio_filepath": "audio/b.flac", "text": "le PATIENT"}])
    with pytest.raises(DataLeakError, match=r"train & test \(text\): 1 shared"):
        check_data_leaks([train], [], [test])


def test_val_test_overlap_is_not_fatal(corpus):
    train = _write_manifest(corpus / "train.json", [{"audio_filepath": "audio/a.flac", "text": "un"}])
    shared = _write_manifest(corpus / "shared.json", [{"audio_filepath": "audio/b.flac", "text": "deux"}])
    counts = check_data_leaks([train], [shared], [shared])
    assert counts["val_test_audio"] == 1 and counts["val_test_text"] == 1
    assert counts["train_val_audio"] == 0


def test_missing_audio_still_checks_text(corpus):
    train = _write_manifest(corpus / "train.json", [{"audio_filepath": "audio/nope.flac", "text": "un"}])
    val = _write_manifest(corpus / "val.json", [{"audio_filepath": "audio/gone.flac", "text": "Un !"}])
    with pytest.raises(DataLeakError, match=r"\(text\)"):
        check_data_leaks([train], [val], [])


def test_different_slices_of_one_file_do_not_collide(corpus):
    train = _write_manifest(
        corpus / "train.json", [{"audio_filepath": "audio/b.flac", "offset": 1.0, "duration": 2.0, "text": "un"}]
    )
    val = _write_manifest(
        corpus / "val.json", [{"audio_filepath": "audio/b.flac", "offset": 5.0, "duration": 2.0, "text": "deux"}]
    )
    assert check_data_leaks([train], [val], [])["train_val_audio"] == 0


def test_collect_manifests_reads_plain_list_and_ds_item():
    cfg = OmegaConf.create(
        {
            "manifest_filepath": ["a.json", "b.json,c.json"],
            "ds_item": [{"name": "x", "manifest_filepath": "d.json"}, {"name": "y", "manifest_filepath": "a.json"}],
        }
    )
    assert collect_manifests(cfg) == ["a.json", "b.json", "c.json", "d.json"]
    assert collect_manifests(None) == []


def test_quick_fingerprint_collision_is_settled_by_full_hash(corpus):
    """Same size, same head and tail, different middle: stage 2 matches, stage 3 must clear it."""
    head, tail = b"H" * 4096, b"T" * 4096
    (corpus / "audio" / "x.flac").write_bytes(head + b"1" * 1000 + tail)
    (corpus / "audio" / "y.flac").write_bytes(head + b"2" * 1000 + tail)
    train = _write_manifest(corpus / "train.json", [{"audio_filepath": "audio/x.flac", "text": "un"}])
    val = _write_manifest(corpus / "val.json", [{"audio_filepath": "audio/y.flac", "text": "deux"}])
    assert check_data_leaks([train], [val], [])["train_val_audio"] == 0
