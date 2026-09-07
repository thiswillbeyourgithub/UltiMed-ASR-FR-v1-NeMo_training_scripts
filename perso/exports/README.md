# ONNX exports

Working exports of NeMo checkpoints, nested as `exports/<checkpoint>/<date>-<variant>/`. Only this README is tracked: the export folders are large and stay local.

| folder | source checkpoint | contents |
|---|---|---|
| `ultimed/2026-08-31-plain/` | UltiMed fine-tune (`parakeet-tdt-0.6b-v3-ultimed.nemo`) | plain NeMo export: `encoder-model.onnx` graph plus per-tensor external data files, `decoder_joint-model.onnx`, `vocab.txt` |
| `ultimed/2026-09-01-web2/` | same checkpoint | the web-optimized re-export (encoder web-export flags: mask-free graph, runtime rel-pos positional encoding, padded-batch NaN tripwire): `encoder-model.web2.onnx` (+ `.data`, 4682 nodes) and its graph-optimized form `encoder-model.web2.opt.onnx` (+ `.data`, 2247 nodes). The `.opt` graph is the one that was sharded into the UltiMed model folder's `fp32/encoder-model.onnx` |

The stock `nvidia/parakeet-tdt-0.6b-v3` checkpoint **was** exported through this path on 2026-09-08, plain and `--web-optimized` (4955 and 4682 nodes; the web one folds to 2110). Those two export folders live in the model repo's own gitignored scratch area, `parakeet-tdt-0.6b-v3-optimized-onnx/local/nemo-export/stock/`, rather than here, because only `perso/exports/ultimed/` is gitignored in this repo and adding a second ignore entry was not wanted. The web-optimized export is what the `parakeet-tdt-0.6b-v3-optimized-onnx` repo root now ships; its earlier istupakov-derived generation was moved down into that repo's `istupakov_smoothquant/` folder and is still published.

Note for anyone re-running `perso/export_onnx.py`: the `.venv` here advertises an editable install of `nemo_toolkit` whose finder mapping is empty, so `import nemo` resolves only when the fork root happens to be `sys.path[0]`. That is true for `python -c` but not for `python perso/export_onnx.py`, which fails outright. Run it with `PYTHONPATH=<fork root>`. The script's PEP 723 header nominally allows `uv run`, but that would install **stock** NeMo from PyPI, where the three `--web-optimized` flags are unrecognized attributes that get silently ignored, yielding a plain export under a web-optimized name.

Shipping copies live in `parakeet_web/fallback_models/Olicorne/<repo>/`, grouped by precision folder (`fp32/`, `fp16/`, `int8/`, `w4a8/`, `int8-lite/`) with the canonical basenames unchanged.

Nesting introduced on 2026-09-03 with Claude Code; the stock export note added 2026-09-08, likewise with Claude Code.
