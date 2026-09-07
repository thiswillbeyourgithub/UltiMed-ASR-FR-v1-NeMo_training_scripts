# ONNX exports

Working exports of NeMo checkpoints, nested as `exports/<checkpoint>/<date>-<variant>/`. Only this README is tracked: the export folders are large and stay local.

| folder | source checkpoint | contents |
|---|---|---|
| `ultimed/2026-08-31-plain/` | UltiMed fine-tune (`parakeet-tdt-0.6b-v3-ultimed.nemo`) | plain NeMo export: `encoder-model.onnx` graph plus per-tensor external data files, `decoder_joint-model.onnx`, `vocab.txt` |
| `ultimed/2026-09-01-web2/` | same checkpoint | the web-optimized re-export (encoder web-export flags: mask-free graph, runtime rel-pos positional encoding, padded-batch NaN tripwire): `encoder-model.web2.onnx` (+ `.data`, 4682 nodes) and its graph-optimized form `encoder-model.web2.opt.onnx` (+ `.data`, 2247 nodes). The `.opt` graph is the one that was sharded into the UltiMed model folder's `fp32/encoder-model.onnx` |

The stock `nvidia/parakeet-tdt-0.6b-v3` checkpoint has not been exported through this path. The published stock repo (`parakeet-tdt-0.6b-v3-optimized-onnx`) descends from istupakov's export.

Shipping copies live in `parakeet_web/fallback_models/Olicorne/<repo>/`, grouped by precision folder (`fp32/`, `fp16/`, `int8/`, `w4a8/`, `int8-lite/`) with the canonical basenames unchanged.

Nesting introduced on 2026-09-03 with Claude Code.
