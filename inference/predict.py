"""Prediction script for sparse signal reconstruction.

Executable CLI script that loads a trained model, reads compressed
measurements and a sensing matrix from disk, runs a forward pass,
and saves the reconstructed signal.

Usage::

    python -m inference.predict \
        --checkpoint checkpoints/model_best.pt \
        --config configs/default.yaml \
        --input measurements.npy \
        --sensing_matrix Phi.npy \
        --output reconstruction.npy \
        --device cuda

    # With ground-truth comparison:
    python -m inference.predict \
        --checkpoint checkpoints/model_best.pt \
        --config configs/default.yaml \
        --input measurements.npy \
        --sensing_matrix Phi.npy \
        --output reconstruction.npy \
        --ground_truth x_true.npy
"""

import argparse
import importlib
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the prediction script.

    Args:
        argv: Optional list of argument strings (defaults to ``sys.argv``).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run reconstruction inference on compressed measurements.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint file (.pt or .pth).",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input measurements file (.npy or .pt).",
    )
    parser.add_argument(
        "--sensing_matrix",
        type=str,
        required=True,
        help="Path to the sensing matrix file (.npy or .pt).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (e.g., 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reconstruction.npy",
        help="Output path for the reconstructed signal (.npy).",
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        default=None,
        help="Optional path to ground-truth signal for NMSE comparison.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference (for multi-sample inputs).",
    )
    return parser.parse_args(argv)


def load_tensor(file_path: str, device: torch.device) -> torch.Tensor:
    """Load a tensor from a ``.npy`` or ``.pt`` file.

    Args:
        file_path: Path to the data file.
        device: Target device for the loaded tensor.

    Returns:
        Loaded tensor on the specified device.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(str(path))
        tensor = torch.from_numpy(data).float().to(device)
    elif suffix in (".pt", ".pth"):
        tensor = torch.load(str(path), map_location=device, weights_only=False)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"Expected a torch.Tensor in {file_path}, got {type(tensor)}"
            )
        tensor = tensor.float()
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'. Use .npy or .pt files."
        )

    logger.info("Loaded tensor from %s — shape: %s", file_path, tuple(tensor.shape))
    return tensor


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as fh:
        config: Dict[str, Any] = yaml.safe_load(fh)

    logger.info("Loaded configuration from %s", config_path)
    return config


def load_model(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """Load a trained model from a checkpoint.

    Dynamically imports the model class from the module specified in
    the config, instantiates it with the provided parameters, and
    loads the saved state dictionary.

    Args:
        checkpoint_path: Path to the checkpoint file.
        config: Experiment configuration dictionary.
        device: Target device for the model.

    Returns:
        Model in evaluation mode with loaded weights.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Dynamic model construction
    model_cfg = config.get("model", {})
    module_name = model_cfg.get("module", "models.model_architecture")
    class_name = model_cfg.get("class", "UDiff")

    module = importlib.import_module(module_name)
    model_cls = getattr(module, class_name)
    model_params = {
        k: v for k, v in model_cfg.items() if k not in ("module", "class")
    }
    model: torch.nn.Module = model_cls(**model_params)

    # Load weights
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "unknown")
        logger.info("Loaded model weights from epoch %s", epoch)
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        logger.info("Loaded model weights (state_dict key).")
    else:
        model.load_state_dict(checkpoint)
        logger.info("Loaded raw state dict.")

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_prediction(
    model: torch.nn.Module,
    y: torch.Tensor,
    Phi: torch.Tensor,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Run model inference on input measurements.

    Processes measurements in mini-batches to handle large inputs
    without exceeding GPU memory.

    Args:
        model: Trained reconstruction model in eval mode.
        y: Compressed measurements of shape ``(N, M)`` or ``(M,)``.
        Phi: Sensing matrix of shape ``(M, N_signal)``.
        batch_size: Number of samples per forward pass.
        device: Inference device.

    Returns:
        Reconstructed signals of shape ``(N, N_signal)`` or
        ``(N_signal,)`` for single-sample input.
    """
    single_sample = y.dim() == 1
    if single_sample:
        y = y.unsqueeze(0)

    y = y.to(device)
    Phi = Phi.to(device)
    n_samples = y.shape[0]
    reconstructions: List[torch.Tensor] = []

    logger.info("Running prediction on %d sample(s)...", n_samples)
    start_time = time.time()

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        y_batch = y[start_idx:end_idx]
        x_hat_batch = model(y_batch, Phi)
        reconstructions.append(x_hat_batch.cpu())

    x_hat = torch.cat(reconstructions, dim=0)
    elapsed = time.time() - start_time
    logger.info(
        "Prediction completed in %.4fs (%.4fs/sample)",
        elapsed,
        elapsed / n_samples,
    )

    if single_sample:
        x_hat = x_hat.squeeze(0)

    return x_hat


def compute_nmse(x_hat: torch.Tensor, x_true: torch.Tensor) -> float:
    """Compute NMSE in dB between reconstructed and true signals.

    Args:
        x_hat: Reconstructed signal.
        x_true: Ground-truth signal.

    Returns:
        NMSE in dB as a float.
    """
    x_hat_flat = x_hat.reshape(-1) if x_hat.dim() <= 1 else x_hat.reshape(x_hat.shape[0], -1)
    x_true_flat = x_true.reshape(-1) if x_true.dim() <= 1 else x_true.reshape(x_true.shape[0], -1)

    if x_hat_flat.dim() == 1:
        error = torch.sum((x_hat_flat - x_true_flat) ** 2)
        signal = torch.sum(x_true_flat ** 2)
    else:
        error = torch.sum((x_hat_flat - x_true_flat) ** 2, dim=1).mean()
        signal = torch.sum(x_true_flat ** 2, dim=1).mean()

    nmse_linear = error / (signal + 1e-12)
    nmse_db = 10.0 * torch.log10(nmse_linear + 1e-12)
    return nmse_db.item()


def save_reconstruction(x_hat: torch.Tensor, output_path: str) -> None:
    """Save reconstructed signal to a ``.npy`` file.

    Args:
        x_hat: Reconstructed signal tensor.
        output_path: Destination file path.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    np.save(str(output), x_hat.numpy())
    logger.info("Reconstruction saved to %s — shape: %s", output_path, tuple(x_hat.shape))


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point for the prediction script.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).
    """
    args = parse_args(argv)
    device = torch.device(args.device)

    logger.info("=" * 60)
    logger.info("  Reconstruction Prediction")
    logger.info("=" * 60)
    logger.info("Checkpoint     : %s", args.checkpoint)
    logger.info("Config         : %s", args.config)
    logger.info("Input          : %s", args.input)
    logger.info("Sensing matrix : %s", args.sensing_matrix)
    logger.info("Device         : %s", device)
    logger.info("Output         : %s", args.output)
    logger.info("Ground truth   : %s", args.ground_truth or "(not provided)")
    logger.info("=" * 60)

    # Load config and model
    config = load_config(args.config)
    model = load_model(args.checkpoint, config, device)

    # Load inputs
    y = load_tensor(args.input, device)
    Phi = load_tensor(args.sensing_matrix, device)

    # Run prediction
    x_hat = run_prediction(
        model, y, Phi, batch_size=args.batch_size, device=device
    )

    # Save reconstruction
    save_reconstruction(x_hat, args.output)

    # Compute and report NMSE if ground truth is available
    if args.ground_truth is not None:
        x_true = load_tensor(args.ground_truth, device=torch.device("cpu"))
        x_hat_cpu = x_hat.cpu() if x_hat.device != torch.device("cpu") else x_hat

        nmse_val = compute_nmse(x_hat_cpu, x_true)
        logger.info("-" * 40)
        logger.info("  NMSE (dB): %.4f", nmse_val)
        logger.info("-" * 40)

    logger.info("Prediction complete.")


if __name__ == "__main__":
    main()
