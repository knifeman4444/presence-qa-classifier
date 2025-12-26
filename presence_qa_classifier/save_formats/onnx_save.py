import logging
from pathlib import Path

from omegaconf import DictConfig
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


def export_to_onnx(cfg: DictConfig):
    """
    Convert the saved model to ONNX format.

    Args:
        cfg: Hydra configuration object
    """
    output_dir = Path(cfg.models.output_dir) / cfg.project_name

    if not output_dir.exists():
        logger.error(f"Model directory does not exist: {output_dir}. Please train the model first.")
        raise FileNotFoundError(f"Model directory not found: {output_dir}")

    logger.info(f"Exporting model from {output_dir} to ONNX...")
    onnx_output_dir = output_dir / "onnx"

    logger.info(f"Loading model from {output_dir}...")
    model = ORTModelForCausalLM.from_pretrained(output_dir, export=True)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)

    logger.info(f"Saving ONNX model to {onnx_output_dir}...")
    model.save_pretrained(onnx_output_dir)
    tokenizer.save_pretrained(onnx_output_dir)
    logger.info("ONNX export complete.")
