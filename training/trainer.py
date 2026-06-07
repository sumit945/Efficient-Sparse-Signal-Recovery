"""Trainer for deep unfolding compressed sensing recovery networks.

Implements a complete training loop with:
- Mixed-precision training via ``torch.cuda.amp``
- Gradient clipping (configurable max norm)
- Per-epoch validation with NMSE and PSNR metrics
- Best-model checkpointing based on validation NMSE
- TensorBoard logging for losses and metrics
- Deterministic seeding for reproducibility
- Deep supervision support (multi-stage loss aggregation)
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all backends.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Random seed set to %d for reproducibility.", seed)


def compute_nmse(x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    """Compute Normalized Mean Squared Error (NMSE) in dB.

    .. math::
        \\text{NMSE}(\\hat{x}, x) = 10 \\log_{10}
        \\frac{\\|\\hat{x} - x\\|_2^2}{\\|x\\|_2^2}

    Args:
        x_pred: Predicted signal tensor of shape ``(B, ...)``.
        x_true: Ground-truth signal tensor of shape ``(B, ...)``.

    Returns:
        Scalar tensor containing the batch-averaged NMSE in dB.
    """
    batch_size = x_true.shape[0]
    x_pred_flat = x_pred.reshape(batch_size, -1)
    x_true_flat = x_true.reshape(batch_size, -1)

    mse = torch.sum((x_pred_flat - x_true_flat) ** 2, dim=-1)
    signal_power = torch.sum(x_true_flat ** 2, dim=-1)
    nmse_linear = mse / (signal_power + 1e-10)
    nmse_db = 10.0 * torch.log10(nmse_linear + 1e-10)
    return nmse_db.mean()


def compute_psnr(x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    """Compute Peak Signal-to-Noise Ratio (PSNR).

    Uses the maximum value of the ground truth as the reference peak.

    Args:
        x_pred: Predicted signal tensor of shape ``(B, ...)``.
        x_true: Ground-truth signal tensor of shape ``(B, ...)``.

    Returns:
        Scalar tensor containing the batch-averaged PSNR in dB.
    """
    batch_size = x_true.shape[0]
    x_pred_flat = x_pred.reshape(batch_size, -1)
    x_true_flat = x_true.reshape(batch_size, -1)

    mse = torch.mean((x_pred_flat - x_true_flat) ** 2, dim=-1)
    peak = torch.amax(torch.abs(x_true_flat), dim=-1) ** 2
    psnr = 10.0 * torch.log10(peak / (mse + 1e-10))
    return psnr.mean()


class Trainer:
    """Handles training, validation, checkpointing, and logging.

    The trainer expects the model to return either a single output tensor
    (final-stage reconstruction) or a list of tensors (one per unfolding
    stage) when deep supervision is enabled.  The criterion should accept
    the model output and ground-truth signal accordingly.

    Args:
        model: Deep unfolding network instance.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        optimizer: Optimizer instance (e.g., Adam).
        scheduler: Learning-rate scheduler instance.
        criterion: Loss function supporting deep supervision.
        device: Target ``torch.device``.
        config: Dictionary of training hyper-parameters. Expected keys:

            - ``num_epochs`` (int): Total training epochs.
            - ``grad_clip_norm`` (float | None): Max gradient norm.
            - ``use_amp`` (bool): Enable mixed-precision training.
            - ``seed`` (int): Random seed.
            - ``log_dir`` (str): TensorBoard log directory.
            - ``checkpoint_dir`` (str): Directory for saving checkpoints.
            - ``log_interval`` (int): Batches between log prints.

        logger: Optional Python logger; falls back to module logger.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        criterion: nn.Module,
        device: torch.device,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config
        self.log = logger or globals()["logger"]

        # Training hyper-parameters with safe defaults.
        self.num_epochs: int = config.get("num_epochs", 500)
        self.grad_clip_norm: Optional[float] = config.get("grad_clip_norm", 1.0)
        self.use_amp: bool = config.get("use_amp", False)
        self.log_interval: int = config.get("log_interval", 50)

        # Directories.
        self.log_dir = Path(config.get("log_dir", "runs"))
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer.
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Mixed-precision scaler.
        self.scaler = torch.amp.GradScaler(
            device=device.type, enabled=self.use_amp
        )

        # Tracking state.
        self.best_val_nmse: float = float("inf")
        self.current_epoch: int = 0
        self.global_step: int = 0

        # Reproducibility.
        seed: int = config.get("seed", 42)
        _set_seed(seed)

        self.log.info(
            "Trainer initialized — device=%s, amp=%s, grad_clip=%.2f",
            device,
            self.use_amp,
            self.grad_clip_norm if self.grad_clip_norm else 0.0,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_one_epoch(self) -> float:
        """Run one training epoch.

        Returns:
            Average training loss over the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            y = batch["y"].to(self.device, non_blocking=True)
            x_true = batch["x"].to(self.device, non_blocking=True)

            # Optional: sensing matrix may be per-sample or shared.
            phi = batch.get("phi")
            if phi is not None:
                phi = phi.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=self.device.type, enabled=self.use_amp
            ):
                output = self.model(y, phi) if phi is not None else self.model(y)
                loss = self.criterion(output, x_true)

            self.scaler.scale(loss).backward()

            if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1
            self.global_step += 1

            # Logging.
            if (batch_idx + 1) % self.log_interval == 0:
                self.log.info(
                    "Epoch [%d] Batch [%d/%d]  loss=%.6f",
                    self.current_epoch + 1,
                    batch_idx + 1,
                    len(self.train_loader),
                    batch_loss,
                )
            self.writer.add_scalar("train/batch_loss", batch_loss, self.global_step)

        avg_loss = total_loss / max(num_batches, 1)
        self.writer.add_scalar("train/epoch_loss", avg_loss, self.current_epoch)
        return avg_loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Evaluate the model on the validation set.

        Returns:
            Dictionary with keys ``nmse_db`` and ``psnr_db`` containing
            the average metrics over the validation set.
        """
        self.model.eval()
        total_nmse = 0.0
        total_psnr = 0.0
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            y = batch["y"].to(self.device, non_blocking=True)
            x_true = batch["x"].to(self.device, non_blocking=True)
            phi = batch.get("phi")
            if phi is not None:
                phi = phi.to(self.device, non_blocking=True)

            with torch.amp.autocast(
                device_type=self.device.type, enabled=self.use_amp
            ):
                output = self.model(y, phi) if phi is not None else self.model(y)
                loss = self.criterion(output, x_true)

            # For deep-supervision outputs, evaluate the final stage only.
            x_pred = output[-1] if isinstance(output, (list, tuple)) else output

            total_nmse += compute_nmse(x_pred, x_true).item()
            total_psnr += compute_psnr(x_pred, x_true).item()
            total_loss += loss.item()
            num_batches += 1

        avg_nmse = total_nmse / max(num_batches, 1)
        avg_psnr = total_psnr / max(num_batches, 1)
        avg_loss = total_loss / max(num_batches, 1)

        self.writer.add_scalar("val/nmse_db", avg_nmse, self.current_epoch)
        self.writer.add_scalar("val/psnr_db", avg_psnr, self.current_epoch)
        self.writer.add_scalar("val/loss", avg_loss, self.current_epoch)

        return {"nmse_db": avg_nmse, "psnr_db": avg_psnr, "loss": avg_loss}

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self, num_epochs: Optional[int] = None) -> None:
        """Execute the full training loop.

        Runs ``num_epochs`` epochs, calling :meth:`train_one_epoch` and
        :meth:`validate` each epoch.  Saves the best checkpoint whenever
        the validation NMSE improves.

        Args:
            num_epochs: Override the epoch count from ``config`` if given.
        """
        epochs = num_epochs if num_epochs is not None else self.num_epochs
        self.log.info("Starting training for %d epochs.", epochs)

        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            train_loss = self.train_one_epoch()

            val_metrics = self.validate()

            if self.scheduler is not None:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("train/lr", lr, epoch)

            elapsed = time.time() - epoch_start
            self.log.info(
                "Epoch [%d/%d]  train_loss=%.6f  val_nmse=%.2f dB  "
                "val_psnr=%.2f dB  lr=%.2e  time=%.1fs",
                epoch + 1,
                self.current_epoch + epochs - epoch,
                train_loss,
                val_metrics["nmse_db"],
                val_metrics["psnr_db"],
                lr,
                elapsed,
            )

            # Save best model.
            if val_metrics["nmse_db"] < self.best_val_nmse:
                self.best_val_nmse = val_metrics["nmse_db"]
                best_path = self.checkpoint_dir / "best_model.pth"
                self.save_checkpoint(str(best_path))
                self.log.info(
                    "New best model saved (val NMSE=%.4f dB).", self.best_val_nmse
                )

            # Periodic checkpoint every 50 epochs.
            if (epoch + 1) % 50 == 0:
                periodic_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
                self.save_checkpoint(str(periodic_path))

        # Final checkpoint.
        final_path = self.checkpoint_dir / "final_model.pth"
        self.save_checkpoint(str(final_path))
        self.writer.close()
        self.log.info("Training complete. Best val NMSE: %.4f dB", self.best_val_nmse)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Persist training state to disk.

        Saves model weights, optimizer state, scheduler state, scaler
        state, and bookkeeping counters so that training can be resumed
        exactly from the saved point.

        Args:
            path: File path for the checkpoint.
        """
        checkpoint: Dict[str, Any] = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_nmse": self.best_val_nmse,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": self.config,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        parent = Path(path).parent
        parent.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, path)
        self.log.debug("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str) -> None:
        """Restore training state from a checkpoint file.

        Args:
            path: Path to a previously saved checkpoint.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        if not Path(path).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint.get("epoch", 0) + 1
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_nmse = checkpoint.get("best_val_nmse", float("inf"))

        self.log.info(
            "Resumed from checkpoint %s (epoch %d, step %d).",
            path,
            self.current_epoch,
            self.global_step,
        )
