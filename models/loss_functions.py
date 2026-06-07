"""
Loss functions for the UDiff framework.

This module provides loss functions tailored to the deep-unfolding training
paradigm:

- **NMSELoss**: Normalized Mean Squared Error for evaluating reconstruction
  quality in a scale-invariant manner.
- **MultiScaleNMSELoss**: Weighted combination of NMSE losses computed at
  each unrolled stage, enabling deep supervision where later stages are
  given higher importance by default.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn


class NMSELoss(nn.Module):
    """Normalized Mean Squared Error (NMSE) loss.

    Computes the ratio of reconstruction error energy to ground-truth
    signal energy, averaged over the batch:

        NMSE = (1/B) * sum_{i=1}^{B} ||x_hat_i - x_i||^2 / ||x_i||^2

    This normalization makes the loss scale-invariant, which is desirable
    when signals have varying magnitudes.

    Args:
        eps: Small constant for numerical stability in the denominator
            (default: ``1e-8``).
        reduction: Batch reduction mode: ``'mean'`` (default) or ``'none'``
            to return per-sample losses.
    """

    def __init__(
        self, eps: float = 1e-8, reduction: str = "mean"
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "none"):
            raise ValueError(
                f"Invalid reduction '{reduction}'. Expected 'mean' or 'none'."
            )
        self.eps = eps
        self.reduction = reduction

    def forward(
        self, x_hat: torch.Tensor, x_true: torch.Tensor
    ) -> torch.Tensor:
        """Compute NMSE loss.

        Args:
            x_hat: Reconstructed signal of shape ``(B, *)`` where ``*``
                denotes arbitrary spatial dimensions.
            x_true: Ground-truth signal of the same shape as ``x_hat``.

        Returns:
            NMSE loss. Scalar if ``reduction='mean'``, or tensor of shape
            ``(B,)`` if ``reduction='none'``.
        """
        # Flatten spatial dimensions: (B, *) -> (B, D).
        B = x_hat.shape[0]
        x_hat_flat = x_hat.reshape(B, -1)
        x_true_flat = x_true.reshape(B, -1)

        # Per-sample squared error and signal energy.
        error_energy = torch.sum((x_hat_flat - x_true_flat) ** 2, dim=-1)  # (B,)
        signal_energy = torch.sum(x_true_flat ** 2, dim=-1)  # (B,)

        # Normalized MSE per sample.
        nmse = error_energy / (signal_energy + self.eps)  # (B,)

        if self.reduction == "mean":
            return nmse.mean()
        return nmse


class MultiScaleNMSELoss(nn.Module):
    """Multi-scale NMSE loss for deep supervision of unrolled stages.

    Computes a weighted sum of NMSE losses evaluated at each intermediate
    reconstruction from the K unrolled stages:

        L = sum_{k=1}^{K} w_k * NMSE(x_k, x_true)

    This encourages all stages to produce progressively better estimates,
    not just the final stage. By default, weights increase linearly so
    that later stages (which should produce better reconstructions) are
    weighted more heavily.

    Args:
        num_stages: Number of unrolled stages K.
        weights: Optional list of K non-negative weights. If ``None``,
            linearly increasing weights ``[1, 2, ..., K]`` are used,
            normalized to sum to 1.
        eps: Small constant for numerical stability in NMSE computation
            (default: ``1e-8``).
    """

    def __init__(
        self,
        num_stages: int = 8,
        weights: Optional[List[float]] = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.num_stages = num_stages
        self.nmse = NMSELoss(eps=eps, reduction="mean")

        # Compute and normalize stage weights.
        if weights is not None:
            if len(weights) != num_stages:
                raise ValueError(
                    f"Expected {num_stages} weights, got {len(weights)}."
                )
            w = torch.tensor(weights, dtype=torch.float32)
        else:
            # Linearly increasing: [1, 2, ..., K].
            w = torch.arange(1, num_stages + 1, dtype=torch.float32)

        # Normalize to sum to 1 for stable gradient magnitudes.
        w = w / w.sum()

        # Register as buffer (not a learnable parameter, but moves with device).
        self.register_buffer("weights", w)

    def forward(
        self,
        intermediates: Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]],
        x_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted multi-scale NMSE loss.

        Args:
            intermediates: List of K intermediate reconstructions from
                each unrolled stage, or tuple of (x_final, intermediates).
            x_true: Ground-truth signal of shape ``(B, *)``.

        Returns:
            Scalar weighted NMSE loss.

        Raises:
            ValueError: If the number of intermediates does not match
                ``num_stages``.
        """
        # If intermediates is the tuple (x_final, intermediates_list) from model forward
        if isinstance(intermediates, tuple) and len(intermediates) == 2 and isinstance(intermediates[1], list):
            intermediates = intermediates[1]

        if len(intermediates) != self.num_stages:
            raise ValueError(
                f"Expected {self.num_stages} intermediate reconstructions, "
                f"got {len(intermediates)}."
            )

        total_loss = torch.tensor(0.0, device=x_true.device, dtype=x_true.dtype)
        for k, x_k in enumerate(intermediates):
            stage_loss = self.nmse(x_k, x_true)
            total_loss = total_loss + self.weights[k] * stage_loss

        return total_loss
