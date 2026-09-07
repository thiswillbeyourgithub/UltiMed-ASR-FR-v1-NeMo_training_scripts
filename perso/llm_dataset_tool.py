#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "click",
#     "litellm",
#     "tiktoken",
#     "loguru",
#     "tqdm",
#     "joblib",
#     "tenacity",
# ]
# ///
"""LLM-powered NeMo dataset filter/alter tool.

Reads a NeMo-formatted JSONL file and processes each sample through an LLM.
Two modes:
  - filter: LLM decides "keep" or "removed" per sample; outputs go to
    separate files (output-path, output-path.removed, output-path.neither).
  - alter:  LLM rewrites the "text" field (e.g. restore capitalisation/punctuation).

Uses litellm for model routing and tiktoken for token usage estimation.
The LLM response must use <thinking>...</thinking> (ignored) then <answer>...</answer>.

Written with the help of Claude Code.
"""

from __future__ import annotations

import json
import re
import sys
import threading
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path

import click
import joblib
import litellm
import tiktoken
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    retry_if_exception_type,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "openrouter/anthropic/claude-sonnet-4-6"
DEFAULT_N_JOBS = 4
# tiktoken doesn't have a Claude tokeniser; cl100k_base is a reasonable proxy
# for estimating token counts (GPT-4 tokeniser, similar subword granularity).
TIKTOKEN_ENCODING = "cl100k_base"

# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_FILTER = """\
You are a dataset quality filter for speech-recognition training data.
You will receive a single data sample (JSON with at least a "text" field).

Your task: {instructions}

Reply using EXACTLY this XML format (no other text outside the tags):

<thinking>
(your reasoning here – this will be ignored)
</thinking>
<answer>keep</answer>

or

<answer>removed</answer>

Only use "keep" or "removed". Nothing else in <answer>.
"""

SYSTEM_PROMPT_ALTER = """\
You are a dataset text normaliser for speech-recognition training data.
You will receive a single data sample (JSON with at least a "text" field).

Your task: {instructions}

Reply using EXACTLY this XML format (no other text outside the tags):

<thinking>
(your reasoning here – this will be ignored)
</thinking>
<answer>
(the corrected/modified text – plain text, no quotes, no JSON)
</answer>
"""


def _build_system_prompt(*, action: str, instructions: str) -> str:
    """Build the cached system prompt from action type and user instructions."""
    template = SYSTEM_PROMPT_FILTER if action == "filter" else SYSTEM_PROMPT_ALTER
    return template.format(instructions=instructions)


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _parse_answer(response_text: str) -> str | None:
    """Extract the content of the first <answer>…</answer> tag.

    Returns None if no valid tag is found.
    """
    match = _ANSWER_RE.search(response_text)
    if match is None:
        return None
    return match.group(1).strip()


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


class TokenCounter:
    """Accumulates estimated input/output token counts via tiktoken.

    Thread-safe: internal lock protects the running totals so multiple
    joblib threads can call add_input / add_output concurrently.
    """

    def __init__(self, encoding_name: str = TIKTOKEN_ENCODING) -> None:
        self._enc = tiktoken.get_encoding(encoding_name)
        self._lock = threading.Lock()
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    def count(self, text: str) -> int:
        """Return the number of tokens in *text*."""
        return len(self._enc.encode(text))

    def add_input(self, text: str) -> None:
        n = self.count(text)
        with self._lock:
            self.input_tokens += n

    def add_output(self, text: str) -> None:
        n = self.count(text)
        with self._lock:
            self.output_tokens += n

    def summary(self) -> str:
        total = self.input_tokens + self.output_tokens
        return (
            f"Token estimate (tiktoken {TIKTOKEN_ENCODING}): "
            f"input={self.input_tokens:,}  output={self.output_tokens:,}  "
            f"total={total:,}"
        )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


# Max retries and backoff bounds for transient LLM API errors (rate limits,
# server errors, timeouts). Exponential wait: 2^attempt seconds, capped at 60s.
_LLM_MAX_RETRIES = 6
_LLM_BACKOFF_MIN_SECONDS = 2
_LLM_BACKOFF_MAX_SECONDS = 60


@retry(
    stop=stop_after_attempt(_LLM_MAX_RETRIES),
    wait=wait_exponential(
        min=_LLM_BACKOFF_MIN_SECONDS,
        max=_LLM_BACKOFF_MAX_SECONDS,
    ),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, "WARNING"),  # type: ignore[arg-type]
    reraise=True,
)
def _call_llm(
    *,
    system_prompt: str,
    user_message: str,
    model: str,
    base_url: str | None,
    counter: TokenCounter,
) -> str:
    """Send a single request to the LLM and return raw response text.

    Retries up to ``_LLM_MAX_RETRIES`` times with exponential backoff
    (2^attempt seconds, capped at 60 s) on any exception — covers rate
    limits, transient server errors, and timeouts from litellm.

    Also updates *counter* with estimated token usage.
    """
    counter.add_input(system_prompt + user_message)

    kwargs: dict = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
    )
    if base_url is not None:
        kwargs["api_base"] = base_url

    response = litellm.completion(**kwargs)
    text: str = response.choices[0].message.content  # type: ignore[union-attr]

    counter.add_output(text)
    return text


# ---------------------------------------------------------------------------
# Processing logic
# ---------------------------------------------------------------------------


def _suffixed_path(input_path: Path, suffix: str) -> Path:
    """Insert *suffix* before the file extension.

    Example: _suffixed_path(Path("data/train.json"), "kept") -> Path("data/train.kept.json")
    """
    return input_path.with_suffix(f".{suffix}{input_path.suffix}")


def _rotate_if_exists(path: Path) -> None:
    """If *path* already exists, rename it with a timestamp to avoid overwriting.

    Example: train.kept.json -> train.kept.20260223_154512.json
    """
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_suffix(f".{ts}{path.suffix}")
        logger.info(f"Output already exists, renaming: {path} -> {backup}")
        path.rename(backup)


def _collect_processed_ids(paths: list[Path]) -> set[str]:
    """Read all JSONL files in *paths* and return a set of audio_filepath values.

    Used by --resume to figure out which samples were already processed in a
    previous (interrupted) run. audio_filepath is the unique key per sample in
    NeMo manifests.
    """
    seen: set[str] = set()
    for p in paths:
        if not p.exists():
            continue
        for raw_line in p.read_text(encoding="utf-8").splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            afp = obj.get("audio_filepath")
            if afp is not None:
                seen.add(afp)
    return seen


def _find_resumable_files(input_path: Path, suffixes: list[str]) -> list[Path]:
    """For each suffix, find the most recent matching file (canonical or timestamped).

    Looks for the canonical path (e.g. train.kept.json) first, then falls back
    to the most recent timestamped backup (e.g. train.kept.20260223_154512.json)
    sorted by modification time.

    Returns one path per suffix (the best candidate), or nothing for that suffix
    if no file exists at all.
    """
    results: list[Path] = []
    for suffix in suffixes:
        canonical = _suffixed_path(input_path, suffix)
        if canonical.exists():
            results.append(canonical)
            continue
        # Look for timestamped backups: train.kept.YYYYMMDD_HHMMSS.json
        # The glob pattern matches the timestamp inserted by _rotate_if_exists
        ext = input_path.suffix  # e.g. ".json"
        stem_with_suffix = input_path.with_suffix(f".{suffix}")  # train.kept
        pattern = f"{stem_with_suffix.name}.*{ext}"
        candidates = sorted(
            stem_with_suffix.parent.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            results.append(candidates[0])
    return results


def _process_filter(
    *,
    lines: list[str],
    system_prompt: str,
    model: str,
    base_url: str | None,
    input_path: Path,
    counter: TokenCounter,
    resume: bool,
    n_jobs: int,
) -> None:
    """Filter mode: route each sample to keep / removed / neither file."""
    kept_path = _suffixed_path(input_path, "kept")
    removed_path = _suffixed_path(input_path, "removed")
    neither_path = _suffixed_path(input_path, "neither")

    # In resume mode: collect audio_filepath values already written, then
    # append to the existing files. Otherwise rotate and start fresh.
    already_done: set[str] = set()
    if resume:
        resume_sources = _find_resumable_files(
            input_path, ["kept", "removed", "neither"]
        )
        already_done = _collect_processed_ids(resume_sources)
        logger.info(f"Resuming: {len(already_done)} samples already processed")
        # If the canonical files don't exist yet (only timestamped backups),
        # copy the content from the backup so we can append to the canonical.
        for suffix, path in [
            ("kept", kept_path),
            ("removed", removed_path),
            ("neither", neither_path),
        ]:
            if not path.exists():
                candidates = _find_resumable_files(input_path, [suffix])
                if candidates and candidates[0] != path:
                    path.write_text(
                        candidates[0].read_text(encoding="utf-8"),
                        encoding="utf-8",
                    )
                    logger.info(f"Copied {candidates[0]} -> {path} for resume")
    else:
        for p in (kept_path, removed_path, neither_path):
            _rotate_if_exists(p)

    # "a" to append in resume mode (file may already have lines), "w" otherwise
    file_mode = "a" if resume else "w"
    stats = {"keep": 0, "removed": 0, "neither": 0, "skipped": 0}

    # Pre-filter lines: parse JSON, skip blanks/invalid/already-done
    work_items: list[tuple[int, str, dict]] = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(f"Skipping non-JSON line {i + 1}")
            continue
        afp = sample.get("audio_filepath")
        if afp and afp in already_done:
            stats["skipped"] += 1
            continue
        work_items.append((i, line, sample))

    # Lock protects file handles and stats dict across threads
    write_lock = threading.Lock()

    def _process_one_filter(
        idx: int,
        raw_line: str,
        sample: dict,
        f_keep: TextIOWrapper,
        f_removed: TextIOWrapper,
        f_neither: TextIOWrapper,
        pbar: tqdm,
    ) -> None:
        """Process a single sample: call LLM, write result, update stats."""
        raw = _call_llm(
            system_prompt=system_prompt,
            user_message=json.dumps(sample, ensure_ascii=False),
            model=model,
            base_url=base_url,
            counter=counter,
        )
        answer = _parse_answer(raw)

        with write_lock:
            if answer == "keep":
                logger.info(f"[{idx + 1}/{len(lines)}] KEPT: {sample.get('text', '')!r}")
                f_keep.write(raw_line + "\n")
                f_keep.flush()
                stats["keep"] += 1
            elif answer == "removed":
                logger.info(f"[{idx + 1}/{len(lines)}] REMOVED: {sample.get('text', '')!r}")
                f_removed.write(raw_line + "\n")
                f_removed.flush()
                stats["removed"] += 1
            else:
                logger.warning(
                    f"Line {idx + 1}: unexpected answer {answer!r}, routing to .neither"
                )
                f_neither.write(raw_line + "\n")
                f_neither.flush()
                stats["neither"] += 1
            pbar.update(1)

    with (
        open(kept_path, file_mode, encoding="utf-8") as f_keep,
        open(removed_path, file_mode, encoding="utf-8") as f_removed,
        open(neither_path, file_mode, encoding="utf-8") as f_neither,
    ):
        with tqdm(total=len(work_items), desc="Filtering", unit="sample", smoothing=0.05) as pbar:
            joblib.Parallel(n_jobs=n_jobs, backend="threading")(
                joblib.delayed(_process_one_filter)(
                    idx, raw_line, sample, f_keep, f_removed, f_neither, pbar
                )
                for idx, raw_line, sample in work_items
            )

    logger.info(f"Filter stats: {stats}")


def _process_alter(
    *,
    lines: list[str],
    system_prompt: str,
    model: str,
    base_url: str | None,
    input_path: Path,
    counter: TokenCounter,
    resume: bool,
    n_jobs: int,
) -> None:
    """Alter mode: rewrite the 'text' field of each sample via the LLM."""
    output_path = _suffixed_path(input_path, "altered")

    # In resume mode: collect audio_filepath values already written, then
    # append to the existing file. Otherwise rotate and start fresh.
    already_done: set[str] = set()
    if resume:
        resume_sources = _find_resumable_files(input_path, ["altered"])
        already_done = _collect_processed_ids(resume_sources)
        logger.info(f"Resuming: {len(already_done)} samples already processed")
        if not output_path.exists() and resume_sources:
            src = resume_sources[0]
            if src != output_path:
                output_path.write_text(
                    src.read_text(encoding="utf-8"), encoding="utf-8"
                )
                logger.info(f"Copied {src} -> {output_path} for resume")
    else:
        _rotate_if_exists(output_path)

    file_mode = "a" if resume else "w"

    # Pre-filter lines: parse JSON, skip blanks/invalid/already-done
    work_items: list[tuple[int, dict]] = []
    skipped = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(f"Skipping non-JSON line {i + 1}")
            continue
        afp = sample.get("audio_filepath")
        if afp and afp in already_done:
            skipped += 1
            continue
        work_items.append((i, sample))

    # Lock protects file handle across threads
    write_lock = threading.Lock()

    def _process_one_alter(
        idx: int,
        sample: dict,
        f_out: TextIOWrapper,
        pbar: tqdm,
    ) -> None:
        """Process a single sample: call LLM, write result."""
        original_text = sample.get("text", "")

        raw = _call_llm(
            system_prompt=system_prompt,
            user_message=json.dumps(sample, ensure_ascii=False),
            model=model,
            base_url=base_url,
            counter=counter,
        )
        answer = _parse_answer(raw)

        if answer is None:
            logger.warning(
                f"Line {idx + 1}: no <answer> tag found, keeping original text"
            )
        else:
            sample["text"] = answer

        # Single log line with before/after to stay readable under multithreading
        logger.info(
            f"[{idx + 1}/{len(lines)}] {original_text!r} -> {sample['text']!r}"
        )

        with write_lock:
            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            f_out.flush()
            pbar.update(1)

    with open(output_path, file_mode, encoding="utf-8") as f_out:
        with tqdm(total=len(work_items), desc="Altering", unit="sample", smoothing=0.05) as pbar:
            joblib.Parallel(n_jobs=n_jobs, backend="threading")(
                joblib.delayed(_process_one_alter)(idx, sample, f_out, pbar)
                for idx, sample in work_items
            )

    if skipped:
        logger.info(f"Skipped {skipped} already-processed samples")
    logger.info(f"Wrote {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--action",
    type=click.Choice(["filter", "alter"], case_sensitive=False),
    required=True,
    help="'filter' to keep/remove samples, 'alter' to rewrite the text field.",
)
@click.option(
    "--instructions",
    required=True,
    help="Natural-language instructions for the LLM (inserted into the system prompt).",
)
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    show_default=True,
    help="litellm model identifier.",
)
@click.option(
    "--base-url",
    default=None,
    help="Optional base URL override for the LLM API (passed as api_base to litellm).",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume a previous interrupted run. Finds the most recent output files "
    "and skips samples whose audio_filepath was already processed.",
)
@click.option(
    "--n-jobs",
    default=DEFAULT_N_JOBS,
    show_default=True,
    help="Number of parallel threads for LLM calls (joblib threading backend).",
)
@click.option(
    "--no-confirm",
    is_flag=True,
    default=False,
    help="Skip the token estimate confirmation prompt (useful for automated runs).",
)
def main(
    input_path: Path,
    action: str,
    instructions: str,
    model: str,
    base_url: str | None,
    resume: bool,
    n_jobs: int,
    no_confirm: bool,
) -> None:
    """Process a NeMo JSONL dataset through an LLM for filtering or alteration.

    INPUT_PATH is the source .json/.jsonl manifest.
    Output files are derived from INPUT_PATH by inserting a suffix before the
    extension: e.g. train.json -> train.kept.json, train.removed.json,
    train.neither.json (filter) or train.altered.json (alter).

    Examples
    --------
    Filter out commercial drug names::

        uv run llm_dataset_tool.py data/train.json \\
            --action filter \\
            --instructions "Only keep the sample if the drug mentioned is NOT a commercial/brand name."

    Restore capitalisation and punctuation::

        uv run llm_dataset_tool.py data/val.json \\
            --action alter \\
            --instructions "Restore proper capitalisation and missing punctuation."
    """
    logger.info(f"Action: {action}")
    logger.info(f"Model:  {model}")
    logger.info(f"Input:  {input_path}")

    # Read all lines up-front so we can show progress as i/N
    lines = input_path.read_text(encoding="utf-8").splitlines()
    logger.info(f"Loaded {len(lines)} lines from {input_path}")

    system_prompt = _build_system_prompt(action=action, instructions=instructions)
    counter = TokenCounter()

    # --- Pre-flight: find already-processed samples if resuming ---
    already_done: set[str] = set()
    if resume:
        suffixes = ["kept", "removed", "neither"] if action == "filter" else ["altered"]
        resume_sources = _find_resumable_files(input_path, suffixes)
        already_done = _collect_processed_ids(resume_sources)
        logger.info(
            f"Resume: found {len(already_done)} already-processed samples "
            f"across {[str(p) for p in resume_sources]}"
        )

    # --- Pre-flight token estimate ---
    # Estimate total input tokens before making any LLM calls so the user can
    # decide whether to proceed (useful to gauge cost).
    system_tokens = counter.count(system_prompt)
    sample_tokens = 0
    n_samples = 0
    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue
        try:
            obj = json.loads(line_s)
        except json.JSONDecodeError:
            continue
        # Skip already-processed samples from the estimate
        afp = obj.get("audio_filepath")
        if afp and afp in already_done:
            continue
        n_samples += 1
        sample_tokens += counter.count(line_s)

    # Each request sends the system prompt + one sample as user message
    estimated_input = (system_tokens * n_samples) + sample_tokens
    logger.info(
        f"Estimated input tokens: ~{estimated_input:,} "
        f"({n_samples} samples × ~{system_tokens} system tokens + sample text)"
    )
    if not no_confirm and not click.confirm("Proceed?", default=True):
        logger.info("Aborted by user.")
        raise SystemExit(0)

    if action == "filter":
        _process_filter(
            lines=lines,
            system_prompt=system_prompt,
            model=model,
            base_url=base_url,
            input_path=input_path,
            counter=counter,
            resume=resume,
            n_jobs=n_jobs,
        )
    else:
        _process_alter(
            lines=lines,
            system_prompt=system_prompt,
            model=model,
            base_url=base_url,
            input_path=input_path,
            counter=counter,
            resume=resume,
            n_jobs=n_jobs,
        )

    logger.info(counter.summary())


if __name__ == "__main__":
    main()
