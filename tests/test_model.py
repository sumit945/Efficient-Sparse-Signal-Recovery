"""Comprehensive test suite for UDiff model components.

Tests cover the denoiser backbone, measurement consistency projection,
full UDiff forward/backward passes, loss functions, and evaluation metrics.
"""

import math

import pytest
import torch

from evaluation.metrics import (
    compute_all_metrics,
    nmse,
    nmse_db,
    psnr,
    support_recovery_fscore,
)
from models.backbone import Denoiser
from models.loss_functions import MultiScaleNMSELoss, NMSELoss
from models.model_architecture import MeasurementConsistency, UDiff
from utils.preprocessing import (
    create_measurements,
    generate_sensing_matrix,
    generate_sparse_signal,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_config():
    """Return a lightweight configuration dictionary for fast tests."""
    return {
        "N": 32,
        "M": 16,
        "K": 4,
        "in_channels": 1,
        "channels_list": [8, 16, 16],
        "embedding_dim": 16,
        "batch_size": 4,
        "sparsity_ratio": 0.1,
        "snr_db": 20.0,
    }


@pytest.fixture()
def device():
    """Select the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture()
def synthetic_1d_data(small_config, device):
    """Generate synthetic 1-D sparse data, sensing matrix, and measurements."""
    cfg = small_config
    x = generate_sparse_signal(cfg["N"], cfg["sparsity_ratio"], cfg["batch_size"])
    x = x.to(device)
    Phi = generate_sensing_matrix(cfg["M"], cfg["N"], matrix_type="gaussian", seed=42)
    Phi = Phi.to(device)
    y = create_measurements(x, Phi, snr_db=cfg["snr_db"])
    y = y.to(device)
    return x, Phi, y


@pytest.fixture()
def denoiser_1d(small_config, device):
    """Instantiate a small 1-D denoiser."""
    cfg = small_config
    model = Denoiser(
        signal_dim=1,
        in_channels=cfg["in_channels"],
        channels=cfg["channels_list"],
        noise_embed_dim=cfg["embedding_dim"],
    ).to(device)
    return model


@pytest.fixture()
def denoiser_2d(device):
    """Instantiate a small 2-D denoiser."""
    model = Denoiser(
        signal_dim=2,
        in_channels=1,
        channels=[8, 16, 16],
        noise_embed_dim=16,
    ).to(device)
    return model


@pytest.fixture()
def udiff_1d(small_config, device):
    """Instantiate a small 1-D UDiff model."""
    cfg = small_config
    model = UDiff(
        signal_dim=1,
        num_stages=cfg["K"],
        in_channels=cfg["in_channels"],
        channels=cfg["channels_list"],
        consistency_mode="general",
    ).to(device)
    return model


# ===================================================================
# TestDenoiser
# ===================================================================

class TestDenoiser:
    """Tests for the ResNet-based denoiser backbone."""

    def test_1d_forward_shape(self, denoiser_1d, small_config, device):
        """1-D denoiser output must match the input spatial shape."""
        B = small_config["batch_size"]
        N = small_config["N"]
        x = torch.randn(B, 1, N, device=device)
        sigma = torch.tensor([0.1], device=device).expand(B)
        out = denoiser_1d(x, sigma)
        assert out.shape == x.shape, (
            f"Expected output shape {x.shape}, got {out.shape}"
        )

    def test_2d_forward_shape(self, denoiser_2d, device):
        """2-D denoiser output must match the input spatial shape."""
        B, C, H, W = 2, 1, 8, 8
        x = torch.randn(B, C, H, W, device=device)
        sigma = torch.tensor([0.5], device=device).expand(B)
        out = denoiser_2d(x, sigma)
        assert out.shape == x.shape, (
            f"Expected output shape {x.shape}, got {out.shape}"
        )

    def test_noise_conditioning(self, denoiser_1d, small_config, device):
        """Different noise levels (sigma) must produce different outputs."""
        B = small_config["batch_size"]
        N = small_config["N"]
        x = torch.randn(B, 1, N, device=device)

        sigma_low = torch.tensor([0.01], device=device).expand(B)
        sigma_high = torch.tensor([1.0], device=device).expand(B)

        with torch.no_grad():
            out_low = denoiser_1d(x, sigma_low)
            out_high = denoiser_1d(x, sigma_high)

        assert not torch.allclose(out_low, out_high, atol=1e-6), (
            "Denoiser should produce distinct outputs for different sigma values."
        )


# ===================================================================
# TestMeasurementConsistency
# ===================================================================

class TestMeasurementConsistency:
    """Tests for the measurement consistency projection layer."""

    def test_general_mode(self, small_config, device):
        """General (Gaussian) mode: output shape must equal input signal shape."""
        cfg = small_config
        mc = MeasurementConsistency(mode="general").to(device)
        Phi = generate_sensing_matrix(
            cfg["M"], cfg["N"], matrix_type="gaussian", seed=0,
        ).to(device)
        x = torch.randn(cfg["batch_size"], cfg["N"], device=device)
        y = Phi @ x.unsqueeze(-1)
        y = y.squeeze(-1)
        lambda_k = torch.tensor(0.5, device=device)
        out = mc(x, Phi, y, lambda_k)
        assert out.shape == x.shape, (
            f"Expected {x.shape}, got {out.shape}"
        )

    def test_orthonormal_mode(self, small_config, device):
        """Orthonormal mode with partial-identity Phi."""
        cfg = small_config
        mc = MeasurementConsistency(mode="orthonormal").to(device)
        # Partial identity: select first M rows of I_N
        Phi = torch.eye(cfg["N"], device=device)[: cfg["M"]]
        x = torch.randn(cfg["batch_size"], cfg["N"], device=device)
        y = (Phi @ x.unsqueeze(-1)).squeeze(-1)
        lambda_k = torch.tensor(0.5, device=device)
        out = mc(x, Phi, y, lambda_k)
        assert out.shape == x.shape

    def test_consistency_reduces_measurement_error(self, small_config, device):
        """Projection should reduce ||y - Phi x||."""
        cfg = small_config
        mc = MeasurementConsistency(mode="general").to(device)
        Phi = generate_sensing_matrix(
            cfg["M"], cfg["N"], matrix_type="gaussian", seed=7,
        ).to(device)
        # Random (non-consistent) signal
        x = torch.randn(cfg["batch_size"], cfg["N"], device=device)
        y = (Phi @ x.unsqueeze(-1)).squeeze(-1) + 0.1 * torch.randn(
            cfg["batch_size"], cfg["M"], device=device,
        )
        lambda_k = torch.tensor(0.9, device=device)
        x_proj = mc(x, Phi, y, lambda_k)

        residual_before = torch.norm(
            y - (Phi @ x.unsqueeze(-1)).squeeze(-1), dim=-1,
        )
        residual_after = torch.norm(
            y - (Phi @ x_proj.unsqueeze(-1)).squeeze(-1), dim=-1,
        )
        # At least on average the residual should decrease
        assert residual_after.mean() <= residual_before.mean(), (
            "Measurement consistency projection should reduce residual on average."
        )


# ===================================================================
# TestUDiff
# ===================================================================

class TestUDiff:
    """End-to-end tests for the UDiff unfolding model."""

    def test_forward_1d(self, udiff_1d, synthetic_1d_data):
        """Forward pass should return a tensor with the correct signal shape."""
        x_true, Phi, y = synthetic_1d_data
        x_rec, _ = udiff_1d(y, Phi)
        assert x_rec.shape == x_true.shape, (
            f"Reconstructed shape {x_rec.shape} != true shape {x_true.shape}"
        )

    def test_forward_returns_intermediates(self, udiff_1d, synthetic_1d_data, small_config):
        """Intermediates list should have exactly K elements (one per stage)."""
        _, Phi, y = synthetic_1d_data
        _, intermediates = udiff_1d(y, Phi)
        expected_k = small_config["K"]
        assert len(intermediates) == expected_k, (
            f"Expected {expected_k} intermediates, got {len(intermediates)}"
        )

    def test_backward_gradients(self, udiff_1d, synthetic_1d_data):
        """All parameters should receive gradients after a backward pass."""
        x_true, Phi, y = synthetic_1d_data
        x_rec, _ = udiff_1d(y, Phi)

        loss = torch.nn.functional.mse_loss(x_rec, x_true)
        loss.backward()

        params_without_grad = [
            name
            for name, p in udiff_1d.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert len(params_without_grad) == 0, (
            f"Parameters without gradients: {params_without_grad}"
        )

    def test_parameter_constraints(self, udiff_1d, synthetic_1d_data):
        """After a forward pass the learned schedules must satisfy constraints.

        sigma_k > 0, lambda_k > 0, 0 < eta_k < 1.
        """
        _, Phi, y = synthetic_1d_data
        with torch.no_grad():
            udiff_1d(y, Phi)

        sigmas, lambdas, etas = udiff_1d.get_constrained_params()
        assert (sigmas > 0).all(), f"sigmas contain non-positive values: {sigmas}"
        assert (lambdas > 0).all(), f"lambdas contain non-positive values: {lambdas}"
        assert (etas > 0).all() and (etas < 1).all(), f"etas violate 0 < eta < 1: {etas}"


# ===================================================================
# TestLossFunctions
# ===================================================================

class TestLossFunctions:
    """Tests for training loss functions."""

    def test_nmse_perfect_reconstruction(self, device):
        """NMSE of identical tensors should be (approximately) zero."""
        criterion = NMSELoss()
        x = torch.randn(4, 32, device=device)
        loss = criterion(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_nmse_positive(self, device):
        """NMSE must be strictly positive for non-identical inputs."""
        criterion = NMSELoss()
        x = torch.randn(4, 32, device=device)
        x_hat = x + 0.1 * torch.randn_like(x)
        loss = criterion(x_hat, x)
        assert loss.item() > 0.0

    def test_multiscale_nmse(self, small_config, device):
        """MultiScaleNMSELoss should accept a list of intermediates."""
        cfg = small_config
        criterion = MultiScaleNMSELoss(
            num_stages=cfg["K"],
            weights=[1.0 / cfg["K"]] * cfg["K"],
        )
        x_true = torch.randn(cfg["batch_size"], cfg["N"], device=device)
        intermediates = [
            x_true + 0.1 * (cfg["K"] - k) * torch.randn_like(x_true)
            for k in range(cfg["K"])
        ]
        loss = criterion(intermediates, x_true)
        assert loss.item() > 0.0
        assert torch.isfinite(loss), "Multi-scale NMSE loss must be finite."


# ===================================================================
# TestMetrics
# ===================================================================

class TestMetrics:
    """Tests for evaluation metrics."""

    def test_nmse_db(self):
        """NMSE in dB for a known error level."""
        x = torch.ones(1, 100)
        # 10 % relative error in energy  -> NMSE = 0.01, dB = -20
        x_hat = x + 0.1 * torch.ones_like(x)
        _, val_mean = nmse_db(x_hat, x)
        val = val_mean.item()
        expected_nmse_linear = (0.1 ** 2 * 100) / (1.0 ** 2 * 100)
        expected_db = 10.0 * math.log10(expected_nmse_linear)
        assert abs(val - expected_db) < 0.5, (
            f"Expected ~{expected_db:.1f} dB, got {val:.1f} dB"
        )

    def test_psnr(self):
        """PSNR for a known noise level."""
        x = torch.linspace(0.0, 1.0, 64).unsqueeze(0)
        x_hat = x.clone()
        _, val_mean = psnr(x_hat, x)
        val = val_mean.item()
        # Perfect reconstruction -> very high PSNR
        assert val > 100.0, f"Perfect reconstruction PSNR should be very high, got {val}"

    def test_fscore_perfect(self):
        """Perfect support recovery should yield F1 = 1.0."""
        x = torch.zeros(1, 64)
        x[0, :5] = torch.randn(5)
        f1_dict = support_recovery_fscore(x, x)
        f1 = f1_dict["mean_f1"].item()
        assert f1 == pytest.approx(1.0, abs=1e-6)

    def test_fscore_partial(self):
        """Partial support recovery should yield F1 < 1.0."""
        x_true = torch.zeros(1, 64)
        x_true[0, :5] = 1.0
        x_hat = torch.zeros(1, 64)
        x_hat[0, 2:7] = 1.0  # overlaps indices 2-4 only
        f1_dict = support_recovery_fscore(x_hat, x_true)
        f1 = f1_dict["mean_f1"].item()
        assert 0.0 < f1 < 1.0, f"Expected partial F1, got {f1}"

    def test_compute_all_metrics(self):
        """compute_all_metrics should return a dict with expected keys."""
        x = torch.randn(2, 32)
        x_hat = x + 0.05 * torch.randn_like(x)
        metrics = compute_all_metrics(x_hat, x)
        for key in ("nmse", "nmse_db", "psnr", "support_f1"):
            assert key in metrics, f"Missing metric key: {key}"
