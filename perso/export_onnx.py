"""Export a NeMo ASR checkpoint to ONNX format.

Converts a saved NeMo ASR checkpoint (e.g. from finetuning parakeet-tdt)
to ONNX, with optional INT8 quantization. Also exports the tokenizer
vocabulary alongside the model.
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "click",
#     "loguru",
#     "nemo_toolkit[asr]",
#     "onnxruntime",
#     "onnx",
# ]
# ///

from pathlib import Path

import click
from loguru import logger


@click.command()
@click.option(
    "--model-file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to a saved NeMo ASR checkpoint (.nemo file).",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Directory where the ONNX model and vocab will be written.",
)
@click.option(
    "--quantization",
    type=click.Choice(["int8"]),
    default=None,
    help="Optional post-export quantization (only 'int8' supported).",
)
@click.option(
    "--web-optimized",
    is_flag=True,
    default=False,
    help=(
        "Export an encoder graph specialized for batch-1 / equal-length-batch "
        "runtimes (the parakeet_web contract): drops the attention/padding "
        "masks (72 Where ops + the O(T^2) mask construction), computes the "
        "relative positional encoding in-graph (removes the 41MB baked table "
        "AND the ~400s pos_emb_max_len cap on input length), and adds a NaN "
        "tripwire so a PADDED (unequal-length) batch returns NaN instead of "
        "silently degraded output. WARNING: the exported encoder is INVALID "
        "for padded batches; benchmark/eval harnesses must run batch 1 or "
        "equal-length batches only."
    ),
)
def main(
    model_file: Path,
    output_dir: Path,
    quantization: str | None,
    web_optimized: bool,
) -> None:
    """Export a NeMo ASR checkpoint to ONNX format."""
    # Lazy imports so --help stays fast and CLI validation happens first.
    import nemo.collections.asr as nemo_asr

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    logger.info("Loading checkpoint from {}", model_file)
    model = nemo_asr.models.ASRModel.restore_from(restore_path=str(model_file))

    if web_optimized:
        # Opt-in flags consumed by ConformerEncoder._create_masks /
        # forward_internal and RelPositionalEncoding.forward during export.
        model.encoder.export_skip_mask = True
        model.encoder.export_pad_tripwire = True
        model.encoder.pos_enc.export_runtime_pe = True
        logger.info(
            "web-optimized export: mask-free graph + runtime positional "
            "encoding + padded-batch NaN tripwire (batch-1/equal-length only)"
        )

    logger.info("Exporting ONNX to {}", onnx_path)
    model.export(str(onnx_path))

    # --- Vocabulary export ---------------------------------------------------
    vocab_path = output_dir / "vocab.txt"
    # The blank token (<blk>) is appended after the regular vocab, matching
    # the CTC/TDT convention where blank_id == len(vocab).
    tokens = [*model.tokenizer.vocab, "<blk>"]
    with vocab_path.open("wt") as f:
        for idx, token in enumerate(tokens):
            f.write(f"{token} {idx}\n")
    logger.info("Wrote {} tokens to {}", len(tokens), vocab_path)

    # --- Optional INT8 quantization ------------------------------------------
    if quantization == "int8":
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantized_path = output_dir / "model_int8.onnx"
        logger.info("Quantizing to INT8 → {}", quantized_path)
        quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(quantized_path),
            weight_type=QuantType.QInt8,
        )
        logger.info("INT8 model saved to {}", quantized_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
