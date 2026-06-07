"""
Shared denoiser backbone for the UDiff framework.

This module implements the shared denoiser D_theta used across all unrolled
stages. The denoiser is a ResNet-style architecture conditioned on the noise
level sigma_k via sinusoidal positional encoding. It predicts the clean signal
directly (not the noise residual).

The architecture supports both 1D signals (using Conv1d) and 2D signals
(using Conv2d), selectable via the ``signal_dim`` parameter.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal positional encoding for scalar noise levels.

    Maps the scalar noise level sigma_k to a high-dimensional embedding
    vector using sinusoidal functions at logarithmically spaced frequencies,
    followed by a two-layer MLP that projects to the desired embedding
    dimension.

    The encoding follows the formulation:

        PE(sigma, 2i)   = sin(sigma / 10000^{2i/d})
        PE(sigma, 2i+1) = cos(sigma / 10000^{2i/d})

    Args:
        embed_dim: Dimension of the sinusoidal encoding (before MLP).
        mlp_dim: Output dimension of the MLP projection.
    """

    def __init__(self, embed_dim: int = 64, mlp_dim: int = 128) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Two-layer MLP to project sinusoidal features to conditioning vector.
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute noise-level embedding.

        Args:
            sigma: Noise level scalar(s) of shape ``(B,)`` or ``(1,)``.

        Returns:
            Embedding tensor of shape ``(B, mlp_dim)``.
        """
        device = sigma.device
        half_dim = self.embed_dim // 2

        # Logarithmically spaced frequencies.
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=device, dtype=torch.float32)
            / half_dim
        )  # (half_dim,)

        # Outer product: (B, 1) * (1, half_dim) -> (B, half_dim)
        args = sigma.unsqueeze(-1) * freq.unsqueeze(0)

        # Interleaved sin/cos encoding -> (B, embed_dim)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        # Project through MLP -> (B, mlp_dim)
        return self.mlp(embedding)


class ResidualBlock(nn.Module):
    """Residual block with noise-level conditioning.

    Architecture:
        Conv -> GroupNorm -> (+ noise embedding) -> SiLU -> Conv -> skip

    The noise embedding is added element-wise after the first GroupNorm,
    enabling the block to adapt its behaviour based on the current
    denoising stage's noise level sigma_k.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        noise_embed_dim: Dimension of the noise-level embedding vector.
        signal_dim: Spatial dimensionality of the signal (1 or 2).
        num_groups: Number of groups for GroupNorm (default: 8).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        noise_embed_dim: int = 128,
        signal_dim: int = 1,
        num_groups: int = 8,
    ) -> None:
        super().__init__()

        # Select convolution type based on signal dimensionality.
        Conv = nn.Conv1d if signal_dim == 1 else nn.Conv2d

        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(
            num_groups=min(num_groups, out_channels), num_channels=out_channels
        )
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(
            num_groups=min(num_groups, out_channels), num_channels=out_channels
        )

        # Linear projection to map noise embedding to channel dimension
        # for additive conditioning.
        self.noise_proj = nn.Linear(noise_embed_dim, out_channels)

        # Skip connection: 1x1 conv if channel dimensions change.
        if in_channels != out_channels:
            self.skip = Conv(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(
        self, x: torch.Tensor, noise_embed: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with noise conditioning.

        Args:
            x: Input feature map of shape ``(B, C_in, *)`` where ``*`` is
                the spatial dimension(s).
            noise_embed: Noise-level embedding of shape ``(B, noise_embed_dim)``.

        Returns:
            Output feature map of shape ``(B, C_out, *)``.
        """
        residual = self.skip(x)

        # First conv + norm.
        h = self.conv1(x)
        h = self.norm1(h)

        # Add noise embedding (broadcast across spatial dimensions).
        # noise_proj output: (B, C_out) -> reshape to (B, C_out, 1, ...).
        noise_cond = self.noise_proj(noise_embed)
        # Reshape for broadcasting: (B, C_out) -> (B, C_out, 1) or (B, C_out, 1, 1).
        for _ in range(h.dim() - 2):
            noise_cond = noise_cond.unsqueeze(-1)
        h = h + noise_cond

        h = F.silu(h)

        # Second conv + norm.
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + residual


class Denoiser(nn.Module):
    """Shared ResNet denoiser D_theta for UDiff.

    This denoiser is shared across all K unrolled stages and is conditioned
    on the per-stage noise level sigma_k. It predicts the clean signal
    directly (i.e., outputs x_hat, not the noise residual).

    Architecture:
        1. Initial convolution to lift input to feature space.
        2. Three residual blocks with channel progression [64, 128, 128],
           each conditioned on the noise-level embedding.
        3. Final convolution to project back to the input signal space.

    Args:
        in_channels: Number of channels in the input signal (e.g., 1 for
            single-channel 1D signals, 1 or 3 for images).
        channels: List of channel counts for each residual block.
            Default: ``[64, 128, 128]``.
        signal_dim: Spatial dimensionality of the signal (1 or 2).
            Use 1 for 1D signals and 2 for 2D images.
        noise_embed_dim: Dimension of the sinusoidal encoding (before MLP).
        noise_mlp_dim: Output dimension of the noise embedding MLP.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: Optional[List[int]] = None,
        signal_dim: int = 1,
        noise_embed_dim: int = 64,
        noise_mlp_dim: int = 128,
    ) -> None:
        super().__init__()

        if channels is None:
            channels = [64, 128, 128]

        self.signal_dim = signal_dim
        Conv = nn.Conv1d if signal_dim == 1 else nn.Conv2d

        # Noise-level embedding network.
        self.noise_embedding = SinusoidalPositionEmbedding(
            embed_dim=noise_embed_dim, mlp_dim=noise_mlp_dim
        )

        # Initial convolution: input channels -> first block channels.
        self.input_conv = Conv(
            in_channels, channels[0], kernel_size=3, padding=1
        )

        # Residual blocks with noise conditioning.
        self.res_blocks = nn.ModuleList()
        ch_in = channels[0]
        for ch_out in channels:
            self.res_blocks.append(
                ResidualBlock(
                    in_channels=ch_in,
                    out_channels=ch_out,
                    noise_embed_dim=noise_mlp_dim,
                    signal_dim=signal_dim,
                )
            )
            ch_in = ch_out

        # Final convolution: last block channels -> input signal channels.
        self.output_conv = Conv(
            channels[-1], in_channels, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Denoise the input signal conditioned on noise level.

        Args:
            x: Noisy input signal of shape ``(B, C, *)`` where ``*``
                is the spatial dimension(s).
            sigma: Noise level(s) of shape ``(B,)`` or ``(1,)``.
                Will be broadcast if a single scalar is provided.

        Returns:
            Denoised (clean) signal estimate of the same shape as ``x``.
        """
        # Ensure sigma has batch dimension.
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)
        if sigma.shape[0] == 1 and x.shape[0] > 1:
            sigma = sigma.expand(x.shape[0])

        # Compute noise-level embedding: (B,) -> (B, noise_mlp_dim).
        noise_embed = self.noise_embedding(sigma)

        # Lift to feature space.
        h = self.input_conv(x)

        # Pass through residual blocks with noise conditioning.
        for block in self.res_blocks:
            h = block(h, noise_embed)

        # Project back to signal space.
        return self.output_conv(h)
