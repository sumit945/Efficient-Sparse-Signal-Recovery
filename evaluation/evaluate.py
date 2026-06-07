"""Evaluation pipeline for sparse signal reconstruction models.

Executable CLI script that loads a trained model checkpoint, runs
inference on a test dataset, computes all reconstruction metrics,
and produces a summary report saved as JSON.

Usage::

    python -m evaluation.evaluate \
        --checkpoint checkpoints/model_best.pt \
        --config configs/default.yaml \
        --data_path data/test \
        --output_dir results/eval \
        --device cuda
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from evaluation.metrics import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the evaluation script.

    Args:
        argv: Optional list of argument strings (defaults to ``sys.argv``).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained reconstruction model on test data.",
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
        "--data_path",
        type=str,
        default=None,
        help=(
            "Path to test data directory or file. If not provided, "
            "synthetic test data will be generated using config parameters."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (e.g., 'cuda', 'cpu', 'cuda:1').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/eval",
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--save_reconstructions",
        action="store_true",
        help="If set, save reconstructed signals/images to output_dir.",
    )
    parser.add_argument(
        "--signal_type",
        type=str,
        default="1d",
        choices=["1d", "2d"],
        help="Signal type for metric computation.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of synthetic test samples if data_path is not provided.",
    )
    return parser.parse_args(argv)


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

    This function dynamically imports the model class specified in the
    configuration, instantiates it, and loads the saved state dict.

    Args:
        checkpoint_path: Path to the ``.pt`` / ``.pth`` checkpoint.
        config: Experiment configuration dictionary containing model
            architecture parameters under the ``'model'`` key.
        device: Target device for the loaded model.

    Returns:
        Model in evaluation mode with loaded weights.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        KeyError: If required keys are missing from the config.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Dynamic model import based on config
    model_cfg = config.get("model", {})
    model_module_name = model_cfg.get("module", "models.model_architecture")
    model_class_name = model_cfg.get("class", "UDiff")

    import importlib

    module = importlib.import_module(model_module_name)
    model_cls = getattr(module, model_class_name)
    model_params = {
        k: v for k, v in model_cfg.items() if k not in ("module", "class")
    }
    model: torch.nn.Module = model_cls(**model_params)

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            "Loaded model weights from epoch %d",
            checkpoint.get("epoch", -1),
        )
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        logger.info("Loaded model weights (state_dict key).")
    else:
        model.load_state_dict(checkpoint)
        logger.info("Loaded raw state dict.")

    model = model.to(device)
    model.eval()
    return model


def load_test_data(
    data_path: Optional[str],
    config: Dict[str, Any],
    num_samples: int = 1000,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """Load or generate test data for evaluation.

    If ``data_path`` is provided, loads precomputed test data from
    ``.npz`` or ``.pt`` files. Otherwise, generates synthetic sparse
    signals and compressed measurements using the sensing matrix
    specified in the configuration.

    Args:
        data_path: Optional path to test data file or directory.
        config: Experiment configuration dictionary.
        num_samples: Number of synthetic test samples to generate.
        device: Target device for the tensors.

    Returns:
        Dictionary with keys ``'y'`` (measurements), ``'Phi'``
        (sensing matrix), and ``'x_true'`` (ground-truth signals).
    """
    if data_path is not None:
        path = Path(data_path)
        if path.suffix == ".npz":
            data = np.load(str(path))
            result = {
                k: torch.from_numpy(v).float().to(device)
                for k, v in data.items()
            }
            logger.info(
                "Loaded test data from %s (%d samples)",
                data_path,
                result.get("x_true", result.get("y", torch.empty(0))).shape[0],
            )
            return result

        if path.suffix in (".pt", ".pth"):
            result = torch.load(str(path), map_location=device, weights_only=False)
            if isinstance(result, dict):
                logger.info("Loaded test data from %s", data_path)
                return result

        # Directory with individual files
        if path.is_dir():
            files = {}
            for name in ("y", "Phi", "x_true", "measurements", "sensing_matrix", "ground_truth"):
                for ext in (".npy", ".pt"):
                    fpath = path / f"{name}{ext}"
                    if fpath.exists():
                        if ext == ".npy":
                            files[name] = torch.from_numpy(
                                np.load(str(fpath))
                            ).float().to(device)
                        else:
                            files[name] = torch.load(
                                str(fpath), map_location=device, weights_only=False
                            )

            # Normalize key names
            key_map = {
                "measurements": "y",
                "sensing_matrix": "Phi",
                "ground_truth": "x_true",
            }
            normalized: Dict[str, torch.Tensor] = {}
            for k, v in files.items():
                normalized[key_map.get(k, k)] = v

            logger.info("Loaded test data from directory %s", data_path)
            return normalized

    # Generate synthetic test data
    logger.info("Generating %d synthetic test samples...", num_samples)
    data_cfg = config.get("data", {})
    n = data_cfg.get("signal_dim", 256)
    m = data_cfg.get("measurement_dim", 64)
    sparsity = data_cfg.get("sparsity", 10)

    # Random Gaussian sensing matrix
    Phi = torch.randn(m, n, device=device) / (m ** 0.5)

    # Generate sparse signals
    x_true = torch.zeros(num_samples, n, device=device)
    for i in range(num_samples):
        support = torch.randperm(n, device=device)[:sparsity]
        x_true[i, support] = torch.randn(sparsity, device=device)

    # Compressed measurements
    y = torch.mm(x_true, Phi.t())

    return {"y": y, "Phi": Phi, "x_true": x_true}


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    data: Dict[str, torch.Tensor],
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Run model inference on the test dataset.

    Processes the test set in mini-batches and concatenates results.

    Args:
        model: Trained reconstruction model in eval mode.
        data: Dictionary with ``'y'`` and ``'Phi'`` tensors.
        batch_size: Number of samples per batch.
        device: Inference device.

    Returns:
        Reconstructed signals tensor of shape ``(N, *)``.
    """
    y = data["y"].to(device)
    Phi = data["Phi"].to(device)
    n_samples = y.shape[0]
    reconstructions: List[torch.Tensor] = []

    logger.info("Running inference on %d samples...", n_samples)
    start_time = time.time()

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        y_batch = y[start_idx:end_idx]
        x_hat_batch = model(y_batch, Phi)
        reconstructions.append(x_hat_batch.cpu())

        if (start_idx // batch_size + 1) % 10 == 0:
            elapsed = time.time() - start_time
            progress = end_idx / n_samples * 100
            logger.info(
                "  Progress: %d/%d (%.1f%%) - %.1fs elapsed",
                end_idx,
                n_samples,
                progress,
                elapsed,
            )

    elapsed = time.time() - start_time
    logger.info("Inference completed in %.2fs (%.4fs/sample)", elapsed, elapsed / n_samples)

    return torch.cat(reconstructions, dim=0)


def format_results_table(metrics: Dict[str, float]) -> str:
    """Format metrics as a human-readable ASCII table.

    Args:
        metrics: Dictionary mapping metric names to scalar values.

    Returns:
        Formatted table string.
    """
    header = f"{'Metric':<25} {'Value':>15}"
    separator = "-" * 42
    rows = [header, separator]

    for name, value in sorted(metrics.items()):
        if isinstance(value, float):
            rows.append(f"{name:<25} {value:>15.6f}")
        else:
            rows.append(f"{name:<25} {str(value):>15}")

    rows.append(separator)
    return "\n".join(rows)


def save_results(
    metrics: Dict[str, float],
    output_dir: Path,
    reconstructions: Optional[torch.Tensor] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Save evaluation results to disk.

    Writes metrics as JSON and optionally saves reconstructed signals
    as a NumPy array.

    Args:
        metrics: Computed metrics dictionary.
        output_dir: Directory for output files.
        reconstructions: Optional reconstructed signals tensor.
        config: Optional experiment config for provenance tracking.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    metrics_path = output_dir / "metrics.json"
    results_payload: Dict[str, Any] = {
        "metrics": metrics,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if config is not None:
        results_payload["config"] = config

    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(results_payload, fh, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # Save reconstructions
    if reconstructions is not None:
        recon_path = output_dir / "reconstructions.npy"
        np.save(str(recon_path), reconstructions.numpy())
        logger.info("Reconstructions saved to %s", recon_path)


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point for the evaluation pipeline.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).
    """
    args = parse_args(argv)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("  Reconstruction Model Evaluation")
    logger.info("=" * 60)
    logger.info("Checkpoint : %s", args.checkpoint)
    logger.info("Config     : %s", args.config)
    logger.info("Data path  : %s", args.data_path or "(synthetic)")
    logger.info("Device     : %s", device)
    logger.info("Output dir : %s", output_dir)
    logger.info("Signal type: %s", args.signal_type)
    logger.info("=" * 60)

    # Load configuration
    config = load_config(args.config)

    # Load model
    model = load_model(args.checkpoint, config, device)

    # Load or generate test data
    data = load_test_data(
        args.data_path, config, num_samples=args.num_samples, device=device
    )

    # Run inference
    x_hat = run_inference(model, data, batch_size=args.batch_size, device=device)
    x_true = data["x_true"].cpu()

    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_all_metrics(
        x_hat, x_true, signal_type=args.signal_type
    )

    # Display results
    table = format_results_table(metrics)
    logger.info("\n%s", table)

    # Save results
    save_results(
        metrics=metrics,
        output_dir=output_dir,
        reconstructions=x_hat if args.save_reconstructions else None,
        config=config,
    )

    logger.info("Evaluation complete. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
