"""Reconstruction quality metrics for compressed sensing evaluation.

Implements standard metrics for evaluating sparse signal reconstruction:
NMSE, PSNR, SSIM (2D), and support recovery F-score. All functions
operate on batched PyTorch tensors.
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F


def nmse(
    x_hat: torch.Tensor,
    x_true: torch.Tensor,
    reduction: str = "mean",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute Normalized Mean Squared Error.

    NMSE = ||x_hat - x_true||^2 / ||x_true||^2

    Args:
        x_hat: Reconstructed signal of shape ``(B, *)``.
        x_true: Ground-truth signal of shape ``(B, *)``.
        reduction: ``'mean'`` returns both per-sample and batch mean,
            ``'none'`` returns only per-sample values.

    Returns:
        If ``reduction='mean'``: tuple of (per_sample, mean) tensors.
        If ``reduction='none'``: per-sample NMSE of shape ``(B,)``.

    Raises:
        ValueError: If input shapes do not match.
    """
    if x_hat.shape != x_true.shape:
        raise ValueError(
            f"Shape mismatch: x_hat {x_hat.shape} vs x_true {x_true.shape}"
        )

    batch_size = x_hat.shape[0]
    x_hat_flat = x_hat.reshape(batch_size, -1)
    x_true_flat = x_true.reshape(batch_size, -1)

    error_norm_sq = torch.sum((x_hat_flat - x_true_flat) ** 2, dim=1)
    signal_norm_sq = torch.sum(x_true_flat ** 2, dim=1)

    # Avoid division by zero with a small epsilon
    per_sample = error_norm_sq / (signal_norm_sq + 1e-12)

    if reduction == "none":
        return per_sample

    return per_sample, per_sample.mean()


def nmse_db(
    x_hat: torch.Tensor,
    x_true: torch.Tensor,
    reduction: str = "mean",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute NMSE in decibels.

    NMSE_dB = 10 * log10(NMSE)

    Args:
        x_hat: Reconstructed signal of shape ``(B, *)``.
        x_true: Ground-truth signal of shape ``(B, *)``.
        reduction: ``'mean'`` returns both per-sample and batch mean,
            ``'none'`` returns only per-sample values.

    Returns:
        If ``reduction='mean'``: tuple of (per_sample_db, mean_db) tensors.
        If ``reduction='none'``: per-sample NMSE in dB of shape ``(B,)``.
    """
    per_sample_linear = nmse(x_hat, x_true, reduction="none")
    per_sample_db = 10.0 * torch.log10(per_sample_linear + 1e-12)

    if reduction == "none":
        return per_sample_db

    return per_sample_db, per_sample_db.mean()


def psnr(
    x_hat: torch.Tensor,
    x_true: torch.Tensor,
    data_range: Optional[float] = None,
    reduction: str = "mean",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute Peak Signal-to-Noise Ratio.

    PSNR = 10 * log10(data_range^2 / MSE)

    Args:
        x_hat: Reconstructed signal of shape ``(B, *)``.
        x_true: Ground-truth signal of shape ``(B, *)``.
        data_range: Dynamic range of the signal. If ``None``, inferred
            per sample as ``max(x_true) - min(x_true)``.
        reduction: ``'mean'`` returns both per-sample and batch mean,
            ``'none'`` returns only per-sample values.

    Returns:
        If ``reduction='mean'``: tuple of (per_sample, mean) tensors.
        If ``reduction='none'``: per-sample PSNR in dB of shape ``(B,)``.

    Raises:
        ValueError: If input shapes do not match.
    """
    if x_hat.shape != x_true.shape:
        raise ValueError(
            f"Shape mismatch: x_hat {x_hat.shape} vs x_true {x_true.shape}"
        )

    batch_size = x_hat.shape[0]
    x_hat_flat = x_hat.reshape(batch_size, -1)
    x_true_flat = x_true.reshape(batch_size, -1)

    mse_vals = torch.mean((x_hat_flat - x_true_flat) ** 2, dim=1)

    if data_range is not None:
        peak = torch.tensor(data_range, device=x_hat.device, dtype=x_hat.dtype)
    else:
        peak = x_true_flat.max(dim=1).values - x_true_flat.min(dim=1).values

    per_sample = 10.0 * torch.log10((peak ** 2) / (mse_vals + 1e-12))

    if reduction == "none":
        return per_sample

    return per_sample, per_sample.mean()


def _gaussian_kernel_1d(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create a 1-D Gaussian kernel.

    Args:
        size: Kernel size (must be odd).
        sigma: Standard deviation of the Gaussian.
        device: Torch device for the kernel tensor.

    Returns:
        Normalized 1-D Gaussian kernel of shape ``(size,)``.
    """
    coords = torch.arange(size, dtype=torch.float32, device=device)
    coords -= size // 2
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def _gaussian_kernel_2d(
    size: int, sigma: float, channels: int, device: torch.device
) -> torch.Tensor:
    """Create a 2-D Gaussian kernel for multi-channel convolution.

    Args:
        size: Kernel size (must be odd).
        sigma: Standard deviation of the Gaussian.
        channels: Number of input channels.
        device: Torch device for the kernel tensor.

    Returns:
        Gaussian kernel of shape ``(channels, 1, size, size)``.
    """
    kernel_1d = _gaussian_kernel_1d(size, sigma, device)
    kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
    kernel_2d = kernel_2d.expand(channels, 1, size, size).contiguous()
    return kernel_2d


def ssim_2d(
    x_hat: torch.Tensor,
    x_true: torch.Tensor,
    data_range: Optional[float] = None,
    window_size: int = 11,
    sigma: float = 1.5,
    reduction: str = "mean",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute Structural Similarity Index for 2-D images.

    Uses a Gaussian sliding window to compute local statistics and
    derives SSIM from luminance, contrast, and structural comparisons.

    Args:
        x_hat: Reconstructed images of shape ``(B, C, H, W)``.
        x_true: Ground-truth images of shape ``(B, C, H, W)``.
        data_range: Dynamic range of the images. If ``None``, inferred
            per batch as ``max(x_true) - min(x_true)``.
        window_size: Size of the Gaussian window (must be odd).
        sigma: Standard deviation of the Gaussian window.
        reduction: ``'mean'`` returns both per-sample and batch mean,
            ``'none'`` returns only per-sample values.

    Returns:
        If ``reduction='mean'``: tuple of (per_sample, mean) tensors.
        If ``reduction='none'``: per-sample SSIM of shape ``(B,)``.

    Raises:
        ValueError: If input shapes do not match or are not 4-D.
    """
    if x_hat.shape != x_true.shape:
        raise ValueError(
            f"Shape mismatch: x_hat {x_hat.shape} vs x_true {x_true.shape}"
        )
    if x_hat.dim() != 4:
        raise ValueError(
            f"Expected 4-D input (B, C, H, W), got {x_hat.dim()}-D"
        )

    channels = x_hat.shape[1]
    device = x_hat.device

    # Stability constants
    if data_range is not None:
        L = data_range
    else:
        L = float(
            (x_true.max() - x_true.min()).item()
        )
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    kernel = _gaussian_kernel_2d(window_size, sigma, channels, device)
    padding = window_size // 2

    mu_x = F.conv2d(x_hat, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(x_true, kernel, padding=padding, groups=channels)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = (
        F.conv2d(x_hat ** 2, kernel, padding=padding, groups=channels)
        - mu_x_sq
    )
    sigma_y_sq = (
        F.conv2d(x_true ** 2, kernel, padding=padding, groups=channels)
        - mu_y_sq
    )
    sigma_xy = (
        F.conv2d(x_hat * x_true, kernel, padding=padding, groups=channels)
        - mu_xy
    )

    numerator = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

    ssim_map = numerator / (denominator + 1e-12)

    # Per-sample mean SSIM over spatial and channel dimensions
    per_sample = ssim_map.mean(dim=[1, 2, 3])

    if reduction == "none":
        return per_sample

    return per_sample, per_sample.mean()


def support_recovery_fscore(
    x_hat: torch.Tensor,
    x_true: torch.Tensor,
    threshold: float = 1e-3,
) -> Dict[str, torch.Tensor]:
    """Compute F-score for support recovery of sparse signals.

    The support of a signal is the set of indices where the signal is
    non-zero. This function thresholds both reconstructed and true
    signals, then computes precision, recall, and F1-score of the
    recovered support.

    Args:
        x_hat: Reconstructed signal of shape ``(B, *)``.
        x_true: Ground-truth signal of shape ``(B, *)``.
        threshold: Absolute-value threshold below which entries are
            considered zero.

    Returns:
        Dictionary with keys ``'precision'``, ``'recall'``, ``'f1'``,
        each containing per-sample values of shape ``(B,)`` and a
        scalar ``'mean_f1'``.

    Raises:
        ValueError: If input shapes do not match.
    """
    if x_hat.shape != x_true.shape:
        raise ValueError(
            f"Shape mismatch: x_hat {x_hat.shape} vs x_true {x_true.shape}"
        )

    batch_size = x_hat.shape[0]
    x_hat_flat = x_hat.reshape(batch_size, -1)
    x_true_flat = x_true.reshape(batch_size, -1)

    # Binary support masks
    pred_support = (x_hat_flat.abs() > threshold).float()
    true_support = (x_true_flat.abs() > threshold).float()

    # True positives, predicted positives, actual positives
    tp = (pred_support * true_support).sum(dim=1)
    pred_positive = pred_support.sum(dim=1)
    true_positive = true_support.sum(dim=1)

    precision = tp / (pred_positive + 1e-12)
    recall = tp / (true_positive + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)

    # Zero out metrics when denominators are truly zero
    precision = torch.where(pred_positive > 0, precision, torch.zeros_like(precision))
    recall = torch.where(true_positive > 0, recall, torch.zeros_like(recall))
    f1 = torch.where(
        (pred_positive > 0) | (true_positive > 0), f1, torch.ones_like(f1)
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_precision": precision.mean(),
        "mean_recall": recall.mean(),
        "mean_f1": f1.mean(),
    }


def compute_all_metrics(
    x_hat: torch.Tensor,
    x_true: torch.Tensor,
    signal_type: str = "1d",
    data_range: Optional[float] = None,
    support_threshold: float = 1e-3,
) -> Dict[str, float]:
    """Compute all applicable reconstruction metrics.

    For 1-D signals: NMSE, NMSE (dB), PSNR, and support F-score.
    For 2-D images: additionally computes SSIM.

    Args:
        x_hat: Reconstructed signal of shape ``(B, *)`` or ``(B, C, H, W)``.
        x_true: Ground-truth signal of shape ``(B, *)`` or ``(B, C, H, W)``.
        signal_type: ``'1d'`` for sparse vectors, ``'2d'`` for images.
        data_range: Dynamic range for PSNR/SSIM. If ``None``, auto-inferred.
        support_threshold: Threshold for support recovery computation.

    Returns:
        Dictionary mapping metric names to their scalar values.

    Raises:
        ValueError: If ``signal_type`` is not ``'1d'`` or ``'2d'``.
    """
    if signal_type not in ("1d", "2d"):
        raise ValueError(
            f"signal_type must be '1d' or '2d', got '{signal_type}'"
        )

    results: Dict[str, float] = {}

    # NMSE
    _, nmse_mean = nmse(x_hat, x_true, reduction="mean")
    results["nmse"] = nmse_mean.item()

    # NMSE in dB
    _, nmse_db_mean = nmse_db(x_hat, x_true, reduction="mean")
    results["nmse_db"] = nmse_db_mean.item()

    # PSNR
    _, psnr_mean = psnr(x_hat, x_true, data_range=data_range, reduction="mean")
    results["psnr"] = psnr_mean.item()

    # Support recovery F-score
    support_metrics = support_recovery_fscore(
        x_hat, x_true, threshold=support_threshold
    )
    results["support_precision"] = support_metrics["mean_precision"].item()
    results["support_recall"] = support_metrics["mean_recall"].item()
    results["support_f1"] = support_metrics["mean_f1"].item()

    # SSIM for 2D images
    if signal_type == "2d":
        if x_hat.dim() != 4:
            raise ValueError(
                f"For signal_type='2d', expected 4-D input (B, C, H, W), "
                f"got {x_hat.dim()}-D"
            )
        _, ssim_mean = ssim_2d(
            x_hat, x_true, data_range=data_range, reduction="mean"
        )
        results["ssim"] = ssim_mean.item()

    return results
