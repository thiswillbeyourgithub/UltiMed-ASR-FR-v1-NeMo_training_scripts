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

"""Text normalisation for a formatting-insensitive WER, alongside the raw one.

NeMo's WER compares ``reference.split()`` against ``hypothesis.split()``. That
counts formatting as recognition error, and ``split()`` keeps punctuation
attached to its word, so a single comma costs a whole word:

    reference   "... sécrétion pancréatique exocrine, ainsi que ..."
    hypothesis  "... sécrétion pancréatique exocrine ainsi que ..."
                                            ^ one substitution

Measured over 128,649 reference/hypothesis pairs from the 1.1.0 run, that is
worth 2.10 WER points, 13% relative: 15.89% raw against 13.79% normalised.
Capitalisation alone accounts for 0.64 of it, punctuation for 1.43.

Neither number is the "right" one. Raw WER is what a user reads on screen,
punctuation included; normalised WER is what the model knows about the words.
They answer different questions, so both get logged, and the raw one stays the
checkpoint monitor so curves remain comparable with earlier runs.

This deliberately does NOT expand numbers ("3" vs "trois"), which is a real
source of error in these transcripts but needs a French verbaliser to do
correctly. Half-normalising numbers would be worse than not touching them.

Written with Claude Code.
"""

from typing import List

from jiwer import Compose, RemoveMultipleSpaces, RemovePunctuation, Strip, ToLowerCase

__all__ = ["normalise_text", "normalised_edit_counts", "NORMALISE_TRANSFORM"]

# The exact pipeline from the original perso/diy_metrics.py (commit 40ff3b2852,
# deleted in d7b2573498), kept identical on purpose: these numbers have to be
# comparable with the UltiMed WER/CER measured outside this repo, and a
# normalisation that is merely similar would be worse than none, because the
# difference would be invisible.
#
# RemovePunctuation DELETES rather than spacing out, which matters in French:
# "l'ongle" becomes "longle" (one token), not "l ongle" (two), and
# "micro-traumatismes" matches "microtraumatismes". Hyphenation and elision
# differences therefore cancel instead of costing a word.
NORMALISE_TRANSFORM: Compose = Compose(
    [
        RemovePunctuation(),
        ToLowerCase(),
        RemoveMultipleSpaces(),
        Strip(),
    ]
)


def normalise_text(text: str) -> str:
    """Drop punctuation, lowercase, collapse whitespace, trim."""
    result = NORMALISE_TRANSFORM(text)
    # jiwer returns a list when handed a list; guard against a str input being
    # promoted, which would silently produce character-level tokens downstream.
    if isinstance(result, list):
        result = " ".join(result)
    return result


def normalised_edit_counts(hypotheses: List[str], references: List[str], use_cer: bool = False):
    """Return (edit distance, reference token count) over normalised text.

    Same word-level Levenshtein as the raw metric, only the strings differ, so
    the two numbers stay directly comparable.
    """
    # Imported here rather than at module scope to match nemo/collections/asr/metrics/wer.py,
    # which also treats editdistance as a soft dependency of the metric path.
    import editdistance

    scores = 0
    words = 0
    for hypothesis, reference in zip(hypotheses, references):
        hyp_norm = normalise_text(hypothesis)
        ref_norm = normalise_text(reference)
        if use_cer:
            hyp_tokens, ref_tokens = list(hyp_norm), list(ref_norm)
        else:
            hyp_tokens, ref_tokens = hyp_norm.split(), ref_norm.split()
        words += len(ref_tokens)
        scores += editdistance.eval(hyp_tokens, ref_tokens)
    return scores, words
