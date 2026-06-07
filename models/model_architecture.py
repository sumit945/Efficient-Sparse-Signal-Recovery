"""
UDiff model architecture: Deep Unfolded Diffusion Recovery.

This module implements the full UDiff framework for compressed sensing (CS)
sparse signal recovery. UDiff unrolls K iterative recovery stages, each
consisting of:

    1. **Denoising**: Apply the shared denoiser D_theta conditioned on
       noise level sigma_k to produce a denoised estimate.
    2. **Measurement consistency**: Project the denoised estimate onto the
       measurement-consistent subspace using P_{lambda_k}.
    3. **Relaxation**: Blend the projected estimate with the previous
       iterate via the relaxation factor eta_k.

The update rule for stage k is:

    x_{k+1} = eta_k * P_{lambda_k}(D_theta(x_k, sigma_k))
              + (1 - eta_k) * x_k

Learnable parameters per stage:
    - sigma_k > 0  (noise level, parameterized via softplus)
    - lambda_k > 0 (consistency weight, parameterized via softplus)
    - eta_k in [0, 1] (relaxation factor, parameterized via sigmoid)
"""

from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import Denoiser


class MeasurementConsistency(nn.Module):
    """Measurement-consistency projection P_{lambda_k}.

    Projects the current signal estimate towards the feasible set defined
    by the linear measurement model y = Phi @ x + noise, with a tunable
    trade-off parameter lambda_k.

    Two modes are supported:

    **Orthonormal mode** (``mode='orthonormal'``):
        When Phi has orthonormal rows (Phi @ Phi^T = I), the projection
        simplifies to:

            P(x) = x + (1 / (1 + lambda)) * Phi^T (y - Phi @ x)

    **General mode** (``mode='general'``):
        For arbitrary sensing matrices, the Woodbury/Sherman-Morrison-
        Woodbury (SMW) identity is used for an exact closed-form solution:

            beta = Phi^T @ y + lambda * x
            P(x) = (1 / lambda) * [beta - Phi^T @ (Phi @ Phi^T + lambda * I)^{-1} @ Phi @ beta]

    Args:
        mode: Projection mode, either ``'orthonormal'`` or ``'general'``.
    """

    def __init__(self, mode: Literal["orthonormal", "general"] = "general") -> None:
        super().__init__()
        if mode not in ("orthonormal", "general"):
            raise ValueError(f"Invalid mode '{mode}'. Expected 'orthonormal' or 'general'.")
        self.mode = mode

    def forward(
        self,
        x: torch.Tensor,
        sensing_matrix: torch.Tensor,
        measurements: torch.Tensor,
        lambda_k: torch.Tensor,
    ) -> torch.Tensor:
        """Apply measurement-consistency projection.

        Args:
            x: Current signal estimate of shape ``(B, N)`` where N is
                the signal dimension.
            sensing_matrix: Sensing/measurement matrix Phi of shape
                ``(M, N)`` or ``(B, M, N)``.
            measurements: Compressed measurements y of shape ``(B, M)``.
            lambda_k: Consistency weight (positive scalar).

        Returns:
            Projected signal estimate of shape ``(B, N)``.
        """
        # Ensure Phi is batched: (B, M, N).
        if sensing_matrix.dim() == 2:
            Phi = sensing_matrix.unsqueeze(0).expand(x.shape[0], -1, -1)
        else:
            Phi = sensing_matrix

        Phi_T = Phi.transpose(-2, -1)  # (B, N, M)

        if self.mode == "orthonormal":
            return self._orthonormal_projection(x, Phi, Phi_T, measurements, lambda_k)
        else:
            return self._general_projection(x, Phi, Phi_T, measurements, lambda_k)

    def _orthonormal_projection(
        self,
        x: torch.Tensor,
        Phi: torch.Tensor,
        Phi_T: torch.Tensor,
        y: torch.Tensor,
        lambda_k: torch.Tensor,
    ) -> torch.Tensor:
        """Orthonormal-row projection (Phi @ Phi^T = I).

        P(x) = x + (1 / (1 + lambda)) * Phi^T @ (y - Phi @ x)

        Args:
            x: Signal estimate, shape ``(B, N)``.
            Phi: Sensing matrix, shape ``(B, M, N)``.
            Phi_T: Transposed sensing matrix, shape ``(B, N, M)``.
            y: Measurements, shape ``(B, M)``.
            lambda_k: Consistency weight (positive scalar).

        Returns:
            Projected estimate, shape ``(B, N)``.
        """
        # Residual in measurement space: y - Phi @ x -> (B, M).
        residual = y - torch.bmm(Phi, x.unsqueeze(-1)).squeeze(-1)

        # Back-project residual: Phi^T @ residual -> (B, N).
        correction = torch.bmm(Phi_T, residual.unsqueeze(-1)).squeeze(-1)

        return x + (1.0 / (1.0 + lambda_k)) * correction

    def _general_projection(
        self,
        x: torch.Tensor,
        Phi: torch.Tensor,
        Phi_T: torch.Tensor,
        y: torch.Tensor,
        lambda_k: torch.Tensor,
    ) -> torch.Tensor:
        """General projection via Woodbury identity.

        beta = Phi^T @ y + lambda * x
        P(x) = (1 / lambda) * [beta - Phi^T @ (Phi @ Phi^T + lambda * I)^{-1} @ Phi @ beta]

        Args:
            x: Signal estimate, shape ``(B, N)``.
            Phi: Sensing matrix, shape ``(B, M, N)``.
            Phi_T: Transposed sensing matrix, shape ``(B, N, M)``.
            y: Measurements, shape ``(B, M)``.
            lambda_k: Consistency weight (positive scalar).

        Returns:
            Projected estimate, shape ``(B, N)``.
        """
        B, M, N = Phi.shape

        # beta = Phi^T @ y + lambda * x  ->  (B, N)
        Phi_T_y = torch.bmm(Phi_T, y.unsqueeze(-1)).squeeze(-1)  # (B, N)
        beta = Phi_T_y + lambda_k * x

        # Gram matrix with regularization: G = Phi @ Phi^T + lambda * I  ->  (B, M, M)
        Gram = torch.bmm(Phi, Phi_T)  # (B, M, M)
        eye = torch.eye(M, device=Phi.device, dtype=Phi.dtype).unsqueeze(0)
        G = Gram + lambda_k * eye  # (B, M, M)

        # Phi @ beta -> (B, M)
        Phi_beta = torch.bmm(Phi, beta.unsqueeze(-1)).squeeze(-1)

        # Solve G @ z = Phi @ beta for z using Cholesky (G is positive definite).
        # z = G^{-1} @ Phi @ beta  ->  (B, M)
        z = torch.linalg.solve(G, Phi_beta.unsqueeze(-1)).squeeze(-1)

        # Phi^T @ z -> (B, N)
        Phi_T_z = torch.bmm(Phi_T, z.unsqueeze(-1)).squeeze(-1)

        # P(x) = (1 / lambda) * (beta - Phi^T @ z)
        return (1.0 / lambda_k) * (beta - Phi_T_z)


class UDiffStage(nn.Module):
    """Single unrolled stage of the UDiff framework.

    Each stage performs three sequential operations:

        1. Denoise:  z_k = D_theta(x_k, sigma_k)
        2. Project:  p_k = P_{lambda_k}(z_k)
        3. Relax:    x_{k+1} = eta_k * p_k + (1 - eta_k) * x_k

    This module does not own any parameters itself; the denoiser is shared
    and the per-stage hyperparameters (sigma_k, lambda_k, eta_k) are
    passed in from the parent UDiff model.

    Args:
        consistency_mode: Projection mode for ``MeasurementConsistency``.
        signal_dim: Dimensionality of the signal (1 for 1D, 2 for 2D).
    """

    def __init__(
        self,
        consistency_mode: Literal["orthonormal", "general"] = "general",
        signal_dim: int = 1,
    ) -> None:
        super().__init__()
        self.projection = MeasurementConsistency(mode=consistency_mode)
        self.signal_dim = signal_dim

    def forward(
        self,
        x: torch.Tensor,
        denoiser: Denoiser,
        sensing_matrix: torch.Tensor,
        measurements: torch.Tensor,
        sigma_k: torch.Tensor,
        lambda_k: torch.Tensor,
        eta_k: torch.Tensor,
    ) -> torch.Tensor:
        """Execute one unrolled stage.

        Args:
            x: Current signal estimate. For 1D: ``(B, N)``; for 2D:
                ``(B, C, H, W)``.
            denoiser: Shared denoiser network D_theta.
            sensing_matrix: Sensing matrix Phi, shape ``(M, N)`` or
                ``(B, M, N)``.
            measurements: Measurements y, shape ``(B, M)``.
            sigma_k: Noise level for this stage (positive scalar).
            lambda_k: Consistency weight for this stage (positive scalar).
            eta_k: Relaxation factor for this stage (scalar in [0, 1]).

        Returns:
            Updated signal estimate of the same shape as ``x``.
        """
        # --- Step 1: Denoise ---
        # Reshape for the denoiser if signal is 1D: (B, N) -> (B, 1, N).
        if self.signal_dim == 1 and x.dim() == 2:
            x_input = x.unsqueeze(1)  # (B, 1, N)
        else:
            x_input = x

        z_k = denoiser(x_input, sigma_k.expand(x.shape[0]))

        # Flatten back for projection if 1D: (B, 1, N) -> (B, N).
        if self.signal_dim == 1 and z_k.dim() == 3 and z_k.shape[1] == 1:
            z_k = z_k.squeeze(1)

        # --- Step 2: Measurement consistency projection ---
        # For 2D signals, flatten for projection then reshape back.
        original_shape = x.shape
        if self.signal_dim == 2:
            B = x.shape[0]
            z_k_flat = z_k.reshape(B, -1)  # (B, C*H*W)
            x_flat = x.reshape(B, -1)       # (B, C*H*W)
        else:
            z_k_flat = z_k
            x_flat = x

        p_k = self.projection(z_k_flat, sensing_matrix, measurements, lambda_k)

        # Reshape back for 2D signals.
        if self.signal_dim == 2:
            p_k = p_k.reshape(original_shape)
            x_flat = x  # use original shape for relaxation

        # --- Step 3: Relaxation ---
        if self.signal_dim == 2:
            x_next = eta_k * p_k + (1.0 - eta_k) * x
        else:
            x_next = eta_k * p_k + (1.0 - eta_k) * x_flat

        return x_next


class UDiff(nn.Module):
    """UDiff: Deep Unfolded Diffusion Recovery for Sparse Signal Recovery.

    This model unrolls K iterative recovery stages, each consisting of
    denoising with a shared diffusion-style denoiser, measurement-consistency
    projection, and relaxation. The per-stage parameters (sigma_k, lambda_k,
    eta_k) are learned end-to-end while the denoiser weights are shared
    across all stages.

    Learnable parameters per stage (stored as unconstrained raw values):
        - ``raw_sigma``:  sigma_k = softplus(raw_sigma_k) > 0
        - ``raw_lambda``: lambda_k = softplus(raw_lambda_k) > 0
        - ``raw_eta``:    eta_k = sigmoid(raw_eta_k) in [0, 1]

    Initialization:
        - sigma_k: Linearly decreasing from 1.0 to 0.1 across stages
          (more noise early, less noise later).
        - lambda_k: Initialized to 1.0 for all stages.
        - eta_k: Initialized to 0.5 for all stages.

    Args:
        num_stages: Number of unrolled stages K (default: 8).
        in_channels: Number of input signal channels (default: 1).
        signal_dim: Signal dimensionality, 1 for 1D or 2 for 2D (default: 1).
        channels: Channel list for the denoiser ResBlocks
            (default: ``[64, 128, 128]``).
        consistency_mode: Mode for measurement-consistency projection,
            ``'orthonormal'`` or ``'general'`` (default: ``'general'``).
        init_sigma_range: Tuple ``(start, end)`` for linearly-spaced initial
            sigma values (default: ``(1.0, 0.1)``).
        init_lambda: Initial value for all lambda_k (default: 1.0).
        init_eta: Initial value for all eta_k (default: 0.5).
    """

    def __init__(
        self,
        num_stages: int = 8,
        in_channels: int = 1,
        signal_dim: int = 1,
        channels: Optional[List[int]] = None,
        consistency_mode: Literal["orthonormal", "general"] = "general",
        init_sigma_range: Tuple[float, float] = (1.0, 0.1),
        init_lambda: float = 1.0,
        init_eta: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__()

        # Resolve aliases from configuration file names
        if "denoiser_channels" in kwargs:
            channels = kwargs.pop("denoiser_channels")
        if "sensing_mode" in kwargs:
            sm = kwargs.pop("sensing_mode")
            if sm in ("orthonormal", "general"):
                consistency_mode = sm
            elif sm == "mri":
                consistency_mode = "general"  # MRI uses general SMW projection
        if "signal_channels" in kwargs:
            in_channels = kwargs.pop("signal_channels")
        if "signal_type" in kwargs:
            st = kwargs.pop("signal_type")
            if st == "1d":
                signal_dim = 1
            elif st == "2d":
                signal_dim = 2

        self.num_stages = num_stages
        self.signal_dim = signal_dim

        # Shared denoiser across all stages.
        self.denoiser = Denoiser(
            in_channels=in_channels,
            channels=channels,
            signal_dim=signal_dim,
        )

        # Unrolled stages (each contains its own projection module).
        self.stages = nn.ModuleList(
            [
                UDiffStage(
                    consistency_mode=consistency_mode,
                    signal_dim=signal_dim,
                )
                for _ in range(num_stages)
            ]
        )

        # --- Learnable per-stage parameters (unconstrained) ---
        # sigma_k: linearly decreasing initialization.
        init_sigmas = torch.linspace(
            init_sigma_range[0], init_sigma_range[1], num_stages
        )
        # Invert softplus to get raw values: raw = log(exp(x) - 1).
        raw_sigmas = torch.log(torch.exp(init_sigmas) - 1.0)
        self.raw_sigma = nn.Parameter(raw_sigmas)

        # lambda_k: uniform initialization.
        init_lambdas = torch.full((num_stages,), init_lambda)
        raw_lambdas = torch.log(torch.exp(init_lambdas) - 1.0)
        self.raw_lambda = nn.Parameter(raw_lambdas)

        # eta_k: uniform initialization.
        # Invert sigmoid to get raw values: raw = log(x / (1 - x)).
        init_etas = torch.full((num_stages,), init_eta)
        raw_etas = torch.log(init_etas / (1.0 - init_etas))
        self.raw_eta = nn.Parameter(raw_etas)

    def get_constrained_params(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert raw (unconstrained) parameters to constrained values.

        Returns:
            Tuple of:
                - sigmas: Noise levels, shape ``(K,)``, all > 0.
                - lambdas: Consistency weights, shape ``(K,)``, all > 0.
                - etas: Relaxation factors, shape ``(K,)``, all in [0, 1].
        """
        sigmas = F.softplus(self.raw_sigma)
        lambdas = F.softplus(self.raw_lambda)
        etas = torch.sigmoid(self.raw_eta)
        return sigmas, lambdas, etas

    def forward(
        self,
        measurements: torch.Tensor,
        sensing_matrix: torch.Tensor,
        signal_shape: Optional[Tuple[int, ...]] = None,
        return_intermediates: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Run the full UDiff recovery pipeline.

        The initial estimate is computed as x_0 = Phi^T @ y (matched filter).

        Args:
            measurements: Compressed measurements y of shape ``(B, M)``.
            sensing_matrix: Sensing matrix Phi of shape ``(M, N)`` or
                ``(B, M, N)``.
            signal_shape: (Optional) Shape to reshape the signal for 2D
                processing, e.g., ``(C, H, W)``. Required when
                ``signal_dim=2``.
            return_intermediates: (Optional) If True, returns both final
                reconstruction and stage intermediates list. Defaults to
                self.training.

        Returns:
            If return_intermediates is True:
                Tuple of (x_final, intermediates).
            If return_intermediates is False:
                Final reconstructed signal.
        """
        if return_intermediates is None:
            return_intermediates = self.training

        # Compute initial estimate: x_0 = Phi^T @ y.
        if sensing_matrix.dim() == 2:
            # x_0 = Phi^T @ y  ->  (B, M) @ (M, N) = (B, N)
            x = measurements @ sensing_matrix  # (B, N)
        else:
            Phi_T = sensing_matrix.transpose(-2, -1)  # (B, N, M)
            x = torch.bmm(Phi_T, measurements.unsqueeze(-1)).squeeze(-1)  # (B, N)

        # Reshape for 2D signal processing if needed.
        if self.signal_dim == 2 and signal_shape is not None:
            x = x.reshape(x.shape[0], *signal_shape)

        # Get constrained per-stage parameters.
        sigmas, lambdas, etas = self.get_constrained_params()

        # Iterate through unrolled stages.
        intermediates: List[torch.Tensor] = []
        for k, stage in enumerate(self.stages):
            x = stage(
                x=x,
                denoiser=self.denoiser,
                sensing_matrix=sensing_matrix,
                measurements=measurements,
                sigma_k=sigmas[k].unsqueeze(0),    # (1,)
                lambda_k=lambdas[k],                # scalar
                eta_k=etas[k],                      # scalar
            )
            intermediates.append(x)

        if return_intermediates:
            return x, intermediates
        return x

    def __repr__(self) -> str:
        """Pretty-print model summary with constrained parameter values."""
        sigmas, lambdas, etas = self.get_constrained_params()
        lines = [f"UDiff(num_stages={self.num_stages}, signal_dim={self.signal_dim})"]
        lines.append(f"  Denoiser: {self.denoiser.__class__.__name__}")
        lines.append("  Per-stage parameters:")
        for k in range(self.num_stages):
            lines.append(
                f"    Stage {k}: sigma={sigmas[k].item():.4f}, "
                f"lambda={lambdas[k].item():.4f}, "
                f"eta={etas[k].item():.4f}"
            )
        return "\n".join(lines)
