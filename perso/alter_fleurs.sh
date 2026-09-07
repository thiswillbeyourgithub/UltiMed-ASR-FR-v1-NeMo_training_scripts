#!/usr/bin/env bash
set -euo pipefail

BASEDIR="downloaded_datasets_ignore-backups/fleurs"
SPLITS=(train validation test)
SKIP_LANGS=(fr en)

INSTRUCTIONS='This text was normalized for ASR training. Please restore it to natural written form:
- Add back punctuation (periods, commas, question marks, exclamation marks, apostrophes).
- Restore accents and diacritics appropriate for this language.
- Capitalize the first word of each sentence.
- If the text does not end abruptly, add a period at the end.
- Convert spelled-out numbers back to their natural written form. Usually keep numbers as words, but use digits where conventional (e.g. sport scores, years, addresses).
- Do NOT add punctuation you cannot be confident about from the text alone (no colons, semicolons, quotation marks, etc.).
- Do NOT change, add, or remove any words. Only modify casing, punctuation, accents, and number formatting.
- Output ONLY the corrected text, nothing else.'

for lang_dir in "$BASEDIR"/*/; do
    lang=$(basename "$lang_dir")

    # Skip already-done languages
    skip=false
    for s in "${SKIP_LANGS[@]}"; do
        [[ "$lang" == "$s" ]] && skip=true && break
    done
    $skip && continue

    for split in "${SPLITS[@]}"; do
        json_file="${lang_dir}${split}.json"
        [[ -f "$json_file" ]] || continue

        # # Skip if already altered
        # if [[ -f "${lang_dir}${split}.altered.json" ]]; then
        #     echo "SKIP $json_file (already altered)"
        #     continue
        # fi

        echo "=== Altering $json_file ==="
        uv run llm_dataset_tool.py \
            --action "alter" \
            --instructions "$INSTRUCTIONS" \
            --model openrouter/openai/gpt-oss-120b \
            --resume \
            --no-confirm \
            --n-jobs=8 \
            "$json_file"
    done
done

echo "Done!"
