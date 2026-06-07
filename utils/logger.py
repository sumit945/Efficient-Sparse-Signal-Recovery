"""Experiment logging utilities with TensorBoard and console output.

Provides a unified logging interface that writes scalar metrics, images,
hyper-parameters, and free-form text to both TensorBoard and the console.
A convenience factory (``get_logger``) is also included for obtaining
standard Python loggers with pre-configured handlers.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[assignment,misc]


class ExperimentLogger:
    """Unified experiment logger backed by TensorBoard and console output.

    All scalar values, images, and hyper-parameters are forwarded to a
    ``SummaryWriter`` (when available) and simultaneously echoed to the
    console with human-readable timestamps.

    Args:
        log_dir: Root directory for log files and TensorBoard event files.
        experiment_name: Descriptive experiment identifier used as a
            sub-directory under *log_dir*.
        use_tensorboard: Whether to create a ``SummaryWriter``.  Set to
            ``False`` to disable TensorBoard logging entirely.
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_tensorboard: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir) / experiment_name
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard and (SummaryWriter is not None)

        # Ensure the log directory exists.
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set up TensorBoard writer.
        self._writer: Optional[SummaryWriter] = None
        if self.use_tensorboard:
            self._writer = SummaryWriter(log_dir=str(self.log_dir))

        # Set up a Python logger for console + file output.
        self._logger = logging.getLogger(f"experiment.{experiment_name}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        # Avoid duplicate handlers on repeated instantiation.
        if not self._logger.handlers:
            formatter = logging.Formatter(
                fmt="[%(asctime)s] [%(name)s] %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            # Console handler.
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

            # File handler.
            file_handler = logging.FileHandler(
                self.log_dir / "experiment.log", encoding="utf-8"
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

        self._logger.info(
            "ExperimentLogger initialised — experiment='%s', log_dir='%s'",
            experiment_name,
            self.log_dir,
        )

    # ------------------------------------------------------------------
    # Scalar logging
    # ------------------------------------------------------------------

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar value.

        Args:
            tag: Metric name (e.g. ``'loss/train'``).
            value: Scalar value to record.
            step: Global training step or epoch number.
        """
        if self._writer is not None:
            self._writer.add_scalar(tag, value, global_step=step)
        self._logger.info("[Step %d] %s = %.6g", step, tag, value)

    def log_scalars(self, tag_value_dict: Dict[str, float], step: int) -> None:
        """Log multiple scalar values at the same step.

        Args:
            tag_value_dict: Mapping from metric names to scalar values.
            step: Global training step or epoch number.
        """
        for tag, value in tag_value_dict.items():
            self.log_scalar(tag, value, step)

    # ------------------------------------------------------------------
    # Image logging
    # ------------------------------------------------------------------

    def log_image(
        self,
        tag: str,
        image: Union[Tensor, "np.ndarray"],  # noqa: F821
        step: int,
    ) -> None:
        """Log an image tensor (useful for 2-D reconstruction visualisation).

        Args:
            tag: Identifier for the image series.
            image: Image data.  Accepted shapes are ``(C, H, W)`` or
                ``(H, W)``.  Values are expected in ``[0, 1]``.
            step: Global training step or epoch number.
        """
        if isinstance(image, Tensor):
            if image.dim() == 2:
                image = image.unsqueeze(0)  # (H, W) → (1, H, W)
        if self._writer is not None:
            self._writer.add_image(tag, image, global_step=step)
        self._logger.info("[Step %d] Image logged: %s", step, tag)

    # ------------------------------------------------------------------
    # Hyper-parameter logging
    # ------------------------------------------------------------------

    def log_hyperparams(self, hparams_dict: Dict[str, Any]) -> None:
        """Record experiment hyper-parameters.

        The hyper-parameters are written to TensorBoard's *HPARAMS* tab and
        also dumped to the console for convenience.

        Args:
            hparams_dict: Flat dictionary of hyper-parameter names to values.
        """
        if self._writer is not None:
            self._writer.add_hparams(
                hparams_dict,
                metric_dict={},
                run_name="hparams",
            )
        self._logger.info("Hyperparameters: %s", hparams_dict)

    # ------------------------------------------------------------------
    # Text logging
    # ------------------------------------------------------------------

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log a free-form text snippet.

        Args:
            tag: Identifier for the text series.
            text: Text content to record.
            step: Global training step or epoch number.
        """
        if self._writer is not None:
            self._writer.add_text(tag, text, global_step=step)
        self._logger.info("[Step %d] %s: %s", step, tag, text)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close all logging back-ends."""
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None
        self._logger.info("ExperimentLogger closed.")


# ----------------------------------------------------------------------
# Convenience factory for standard Python loggers
# ----------------------------------------------------------------------


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Return a configured Python logger with console (and optional file) handlers.

    If a logger with the given *name* already has handlers attached, it is
    returned as-is to prevent duplicate output.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Minimum severity level for messages.
        log_file: Optional path to a log file.  When provided, a
            ``FileHandler`` is attached in addition to the console handler.

    Returns:
        A ready-to-use ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(name)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
