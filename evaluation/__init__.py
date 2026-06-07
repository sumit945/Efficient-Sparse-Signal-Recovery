"""Evaluation module for sparse signal reconstruction.

Provides metrics computation and evaluation pipelines for assessing
reconstruction quality of compressed sensing models.
"""

from evaluation.metrics import (
    nmse,
    nmse_db,
    psnr,
    ssim_2d,
    support_recovery_fscore,
    compute_all_metrics,
)

__all__ = [
    "nmse",
    "nmse_db",
    "psnr",
    "ssim_2d",
    "support_recovery_fscore",
    "compute_all_metrics",
]
