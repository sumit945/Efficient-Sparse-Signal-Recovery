#!/usr/bin/env python
"""CLI entry-point for training a deep unfolding sparse recovery network.

Usage examples
--------------
Train with a YAML config::

    python -m training.train --config configs/default.yaml

Train with synthetic data and custom seed::

    python -m training.train --seed 123 --device cuda:0

Resume from a checkpoint::

    python -m training.train --config configs/default.yaml --resume checkpoints/best_model.pth

The script generates 1-D Bernoulli-Gaussian sparse signals when no
external dataset path is provided in the config.  The sensing matrix
is a random Gaussian matrix, and additive white Gaussian noise is
controlled via a configurable SNR parameter.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split

# ---------------------------------------------------------------------------
# Lazy / local imports so the module can be imported without the full
# model package being installed (e.g., for unit tests).
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===================================================================
# Synthetic dataset
# ===================================================================


class SyntheticSparseDataset(Dataset):
    """Synthetic Bernoulli-Gaussian compressed sensing dataset.

    Each sample consists of:

    * ``x`` — sparse signal of length *N*.  Each entry is non-zero with
      probability *sparsity_ratio*, and the non-zero values are drawn
      from :math:`\\mathcal{N}(0, 1)`.
    * ``y`` — noisy measurement vector :math:`y = \\Phi x + n` of
      length *M*, where :math:`\\Phi` is a random Gaussian sensing
      matrix and *n* is AWGN at the specified SNR.
    * ``phi`` — the *M × N* sensing matrix used for this sample.

    Args:
        num_samples: Number of signal–measurement pairs to generate.
        signal_length: Length *N* of the sparse signal.
        compression_ratio: Ratio *M / N* (default 0.5 → 50% measurements).
        sparsity_ratio: Probability of each entry being non-zero.
        snr_db: Signal-to-noise ratio in dB for measurement noise.
            ``None`` produces noiseless measurements.
        shared_sensing_matrix: If ``True``, all samples share one
            sensing matrix (more memory-efficient).
        seed: Random seed for data generation.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        signal_length: int = 256,
        compression_ratio: float = 0.5,
        sparsity_ratio: float = 0.1,
        snr_db: Optional[float] = 40.0,
        shared_sensing_matrix: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()

        if not 0 < compression_ratio <= 1.0:
            raise ValueError(
                f"compression_ratio must be in (0, 1], got {compression_ratio}"
            )
        if not 0 < sparsity_ratio <= 1.0:
            raise ValueError(
                f"sparsity_ratio must be in (0, 1], got {sparsity_ratio}"
            )
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        self.num_samples = num_samples
        self.signal_length = signal_length
        self.num_measurements = int(compression_ratio * signal_length)
        self.sparsity_ratio = sparsity_ratio
        self.snr_db = snr_db

        rng = np.random.RandomState(seed)

        # ----- Sensing matrix(es) -----
        if shared_sensing_matrix:
            phi_np = rng.randn(self.num_measurements, signal_length).astype(
                np.float32
            )
            phi_np /= np.sqrt(self.num_measurements)  # normalize columns
            self._phi_shared: Optional[torch.Tensor] = torch.from_numpy(phi_np)
        else:
            self._phi_shared = None

        # ----- Generate all signals and measurements -----
        self.signals: List[torch.Tensor] = []
        self.measurements: List[torch.Tensor] = []
        self.phi_list: List[torch.Tensor] = []

        for _ in range(num_samples):
            x, y, phi = self._generate_sample(rng)
            self.signals.append(x)
            self.measurements.append(y)
            self.phi_list.append(phi)

        logger.info(
            "Generated %d samples — N=%d, M=%d, sparsity=%.2f, SNR=%s dB",
            num_samples,
            signal_length,
            self.num_measurements,
            sparsity_ratio,
            snr_db if snr_db is not None else "inf",
        )

    # ------------------------------------------------------------------

    def _generate_sample(
        self, rng: np.random.RandomState
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a single (x, y, Φ) triplet.

        Args:
            rng: Numpy random state for generation.

        Returns:
            Tuple of ``(x, y, phi)`` as float32 tensors.
        """
        n = self.signal_length

        # Sparse signal: Bernoulli support × Gaussian amplitudes.
        support = rng.rand(n) < self.sparsity_ratio
        amplitudes = rng.randn(n).astype(np.float32)
        x_np = (amplitudes * support).astype(np.float32)

        # Sensing matrix.
        if self._phi_shared is not None:
            phi_t = self._phi_shared
            phi_np = phi_t.numpy()
        else:
            phi_np = rng.randn(self.num_measurements, n).astype(np.float32)
            phi_np /= np.sqrt(self.num_measurements)
            phi_t = torch.from_numpy(phi_np)

        # Measurements y = Φx + noise.
        y_np = phi_np @ x_np

        if self.snr_db is not None:
            signal_power = np.mean(y_np ** 2) + 1e-10
            noise_power = signal_power / (10.0 ** (self.snr_db / 10.0))
            noise = rng.randn(self.num_measurements).astype(np.float32)
            noise *= np.sqrt(noise_power)
            y_np = y_np + noise

        x_t = torch.from_numpy(x_np)
        y_t = torch.from_numpy(y_np)
        return x_t, y_t, phi_t

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a sample dictionary.

        Returns:
            Dictionary with keys ``x``, ``y``, and ``phi``.
        """
        return {
            "x": self.signals[idx],
            "y": self.measurements[idx],
            "phi": self.phi_list[idx],
        }


# ===================================================================
# Multi-scale (deep-supervision) NMSE loss
# ===================================================================


class MultiScaleNMSELoss(nn.Module):
    """Weighted NMSE loss with deep supervision across unfolding stages.

    Given *K* intermediate reconstructions
    :math:`\\{\\hat{x}^{(k)}\\}_{k=1}^{K}`, the total loss is

    .. math::
        \\mathcal{L} = \\sum_{k=1}^{K} w_k \\,
        \\frac{\\|\\hat{x}^{(k)} - x\\|_2^2}{\\|x\\|_2^2}

    Weights can be supplied explicitly or default to a linearly
    increasing schedule (later stages weighted more heavily).

    If the model returns a single tensor (no deep supervision), this
    loss degrades to a standard NMSE loss.

    Args:
        num_stages: Number of unfolding stages *K*.
        stage_weights: Optional explicit weight per stage.
    """

    def __init__(
        self,
        num_stages: int = 8,
        stage_weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        if stage_weights is not None:
            if len(stage_weights) != num_stages:
                raise ValueError(
                    f"Length of stage_weights ({len(stage_weights)}) "
                    f"does not match num_stages ({num_stages})."
                )
            weights = torch.tensor(stage_weights, dtype=torch.float32)
        else:
            # Linearly increasing weights: [1, 2, ..., K] normalized.
            raw = torch.arange(1, num_stages + 1, dtype=torch.float32)
            weights = raw / raw.sum()

        self.register_buffer("weights", weights)
        self.num_stages = num_stages

    def forward(
        self,
        predictions: Any,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the multi-scale NMSE loss.

        Args:
            predictions: Either a ``list[Tensor]`` of per-stage outputs
                or a single ``Tensor`` (final-stage only).
            target: Ground-truth signal tensor.

        Returns:
            Scalar loss tensor.
        """
        if isinstance(predictions, (list, tuple)):
            # If predictions is a 2-tuple of (x_final, intermediates_list), unpack it
            if len(predictions) == 2 and isinstance(predictions[1], (list, tuple)):
                predictions = predictions[1]
            total_loss = torch.tensor(
                0.0, device=target.device, dtype=target.dtype
            )
            num_preds = len(predictions)
            for k, x_hat in enumerate(predictions):
                nmse_k = self._nmse(x_hat, target)
                # Map prediction index to weight index for variable outputs.
                weight_idx = k if num_preds == self.num_stages else (
                    int(k * self.num_stages / num_preds)
                )
                weight_idx = min(weight_idx, self.num_stages - 1)
                total_loss = total_loss + self.weights[weight_idx] * nmse_k
            return total_loss
        else:
            return self._nmse(predictions, target)

    @staticmethod
    def _nmse(x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        """Batch-averaged linear-scale NMSE (not in dB, for gradient flow).

        Args:
            x_pred: Predicted signal tensor.
            x_true: Ground-truth signal tensor.

        Returns:
            Scalar NMSE value.
        """
        batch = x_true.shape[0]
        pred_flat = x_pred.reshape(batch, -1)
        true_flat = x_true.reshape(batch, -1)
        mse = torch.sum((pred_flat - true_flat) ** 2, dim=-1)
        power = torch.sum(true_flat ** 2, dim=-1) + 1e-10
        return (mse / power).mean()


# ===================================================================
# Configuration helpers
# ===================================================================


def _load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to a ``.yaml`` or ``.yml`` file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    import yaml  # deferred import to avoid hard dep in lightweight envs

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        config: Dict[str, Any] = yaml.safe_load(fh)
    logger.info("Loaded config from %s", path)
    return config


def _default_config() -> Dict[str, Any]:
    """Return sensible default hyper-parameters matching the paper.

    Returns:
        Default configuration dictionary.
    """
    return {
        # Model.
        "num_stages": 8,
        "signal_length": 256,
        "compression_ratio": 0.5,
        # Data.
        "num_train_samples": 50000,
        "num_val_samples": 5000,
        "num_test_samples": 5000,
        "sparsity_ratio": 0.1,
        "snr_db": 40.0,
        "shared_sensing_matrix": True,
        # Training.
        "num_epochs": 500,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
        "use_amp": False,
        "seed": 42,
        # Scheduler.
        "scheduler_T_max": 500,
        "scheduler_eta_min": 1e-6,
        # Logging.
        "log_dir": "runs",
        "checkpoint_dir": "checkpoints",
        "log_interval": 50,
        # DataLoader.
        "num_workers": 4,
        "pin_memory": True,
    }


# ===================================================================
# Model builder (lazy import wrapper)
# ===================================================================


def _build_model(config: Dict[str, Any]) -> nn.Module:
    """Instantiate the deep unfolding network.

    Attempts to import the project-local model definition.  Falls back
    to a minimal placeholder if the model module is unavailable (useful
    for running data-only smoke tests).

    Args:
        config: Full training configuration.

    Returns:
        A ``torch.nn.Module`` instance.
    """
    try:
        from models.model_architecture import UDiff

        model = UDiff(
            num_stages=config.get("num_stages", 8),
            in_channels=config.get("in_channels", 1),
            signal_dim=config.get("signal_dim", 1),
            channels=config.get("denoiser_channels", [64, 128, 128]),
            consistency_mode=config.get("sensing_mode", "general"),
        )
        logger.info("Built UDiff model with %d stages.", config.get("num_stages", 8))
    except ImportError:
        logger.warning(
            "Could not import models.model_architecture — using a placeholder "
            "linear model. Ensure the models package is on the Python path "
            "for real training."
        )
        signal_len = config.get("signal_length", 256)
        num_meas = int(config.get("compression_ratio", 0.5) * signal_len)

        class _PlaceholderModel(nn.Module):
            """Minimal stand-in for smoke testing the training loop."""

            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(num_meas, signal_len)

            def forward(
                self,
                measurements: torch.Tensor,
                sensing_matrix: torch.Tensor,
            ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
                x_hat = self.linear(measurements)
                return x_hat, [x_hat]

        model = _PlaceholderModel()

    return model


# ===================================================================
# Dataset / DataLoader factory
# ===================================================================


def _build_dataloaders(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train / val / test ``DataLoader`` instances.

    If a ``dataset_path`` key is present in *config*, data is loaded
    from the given directory (expected to contain pre-saved tensors).
    Otherwise, synthetic Bernoulli-Gaussian data is generated on the fly.

    Args:
        config: Training configuration dictionary.

    Returns:
        A 3-tuple of ``(train_loader, val_loader, test_loader)``.
    """
    dataset_path = config.get("dataset_path")

    if dataset_path is not None and Path(dataset_path).exists():
        logger.info("Loading external dataset from %s", dataset_path)
        data = torch.load(dataset_path, weights_only=False)
        full_dataset = data if isinstance(data, Dataset) else _dict_to_dataset(data)
        n_total = len(full_dataset)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        n_test = n_total - n_train - n_val
        train_ds, val_ds, test_ds = random_split(
            full_dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(config.get("seed", 42)),
        )
    else:
        logger.info("Generating synthetic Bernoulli-Gaussian dataset.")
        seed = config.get("seed", 42)
        shared_kwargs = dict(
            signal_length=config.get("signal_length", 256),
            compression_ratio=config.get("compression_ratio", 0.5),
            sparsity_ratio=config.get("sparsity_ratio", 0.1),
            snr_db=config.get("snr_db", 40.0),
            shared_sensing_matrix=config.get("shared_sensing_matrix", True),
        )
        train_ds = SyntheticSparseDataset(
            num_samples=config.get("num_train_samples", 50000),
            seed=seed,
            **shared_kwargs,
        )
        val_ds = SyntheticSparseDataset(
            num_samples=config.get("num_val_samples", 5000),
            seed=seed + 1,
            **shared_kwargs,
        )
        test_ds = SyntheticSparseDataset(
            num_samples=config.get("num_test_samples", 5000),
            seed=seed + 2,
            **shared_kwargs,
        )

    batch_size = config.get("batch_size", 16)
    num_workers = config.get("num_workers", 4)
    pin = config.get("pin_memory", True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    logger.info(
        "DataLoaders — train=%d, val=%d, test=%d  (batch=%d, workers=%d)",
        len(train_ds),
        len(val_ds),
        len(test_ds),
        batch_size,
        num_workers,
    )
    return train_loader, val_loader, test_loader


def _dict_to_dataset(
    data: Dict[str, torch.Tensor],
) -> Dataset:
    """Wrap a dictionary of tensors into a ``Dataset``.

    Args:
        data: Dictionary with at least keys ``x`` and ``y``, each of
            shape ``(N, ...)``.

    Returns:
        A simple map-style ``Dataset``.
    """

    class _TensorDictDataset(Dataset):
        def __init__(self, d: Dict[str, torch.Tensor]) -> None:
            self._data = d
            self._keys = list(d.keys())
            self._len = d[self._keys[0]].shape[0]

        def __len__(self) -> int:
            return self._len

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            return {k: v[idx] for k, v in self._data.items()}

    return _TensorDictDataset(data)


# ===================================================================
# CLI argument parser
# ===================================================================


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list. ``None`` reads from ``sys.argv``.

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description="Train a deep unfolding network for compressed sensing recovery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (e.g. 'cuda', 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs in the config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the batch size in the config.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override the learning rate in the config.",
    )
    return parser.parse_args(argv)


# ===================================================================
# Main
# ===================================================================


def main(argv: Optional[List[str]] = None) -> None:
    """Entry-point: parse arguments, build components, and train.

    Args:
        argv: Optional argument list for programmatic invocation.
    """
    args = parse_args(argv)

    # ---- Configuration ----
    config = _default_config()
    if args.config is not None:
        yaml_cfg = _load_yaml_config(args.config)
        # Flatten nested dictionary from YAML and map keys to train.py's expected flat keys
        def flatten(d):
            flat = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    flat.update(flatten(v))
                else:
                    flat[k] = v
            return flat
        flat_cfg = flatten(yaml_cfg)
        
        # Map specific keys to match train.py's internal keys
        if "signal_type" in flat_cfg:
            if flat_cfg["signal_type"] == "1d":
                flat_cfg["signal_dim"] = 1
            elif flat_cfg["signal_type"] == "2d":
                flat_cfg["signal_dim"] = 2
        if "signal_channels" in flat_cfg:
            flat_cfg["in_channels"] = flat_cfg["signal_channels"]
        if "num_train" in flat_cfg:
            flat_cfg["num_train_samples"] = flat_cfg["num_train"]
        if "num_val" in flat_cfg:
            flat_cfg["num_val_samples"] = flat_cfg["num_val"]
        if "num_test" in flat_cfg:
            flat_cfg["num_test_samples"] = flat_cfg["num_test"]
        if "gradient_clip" in flat_cfg:
            flat_cfg["grad_clip_norm"] = flat_cfg["gradient_clip"]
        if "mixed_precision" in flat_cfg:
            flat_cfg["use_amp"] = flat_cfg["mixed_precision"]
        if "save_dir" in flat_cfg:
            flat_cfg["checkpoint_dir"] = flat_cfg["save_dir"]
            
        config.update(flat_cfg)

    # CLI overrides take precedence.
    config["seed"] = args.seed
    if args.epochs is not None:
        config["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["learning_rate"] = args.lr

    device = torch.device(args.device)
    logger.info("Training device: %s", device)

    # ---- Data ----
    train_loader, val_loader, _test_loader = _build_dataloaders(config)

    # ---- Model ----
    model = _build_model(config)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model has %s trainable parameters.", f"{num_params:,}")

    # ---- Optimizer & scheduler ----
    optimizer = Adam(
        model.parameters(),
        lr=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 0.0),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.get("scheduler_T_max", config.get("num_epochs", 500)),
        eta_min=config.get("scheduler_eta_min", 1e-6),
    )

    # ---- Criterion ----
    criterion = MultiScaleNMSELoss(
        num_stages=config.get("num_stages", 8),
    )

    # ---- Trainer ----
    from training.trainer import Trainer  # local import to avoid circular

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        config=config,
        logger=logger,
    )

    # ---- Resume ----
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    # ---- Train ----
    trainer.train()

    logger.info("Done.")


if __name__ == "__main__":
    main()
