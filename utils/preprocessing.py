"""Data preprocessing and generation utilities for sparse signal recovery.

This module provides functions for generating sensing matrices, sparse signals,
and compressed measurements, as well as PyTorch dataset and dataloader
construction for training and evaluation.
"""

import math
import random
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


def set_seed(seed: int) -> None:
    """Set random seeds across all libraries for reproducibility.

    Configures deterministic behaviour for ``random``, ``numpy``, and
    ``torch`` (including CUDA back-ends) so that experiments can be
    exactly reproduced.

    Args:
        seed: Integer seed value applied to every random number generator.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_sensing_matrix(
    M: int,
    N: int,
    matrix_type: Literal["gaussian", "bernoulli", "partial_fourier"] = "gaussian",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Generate a compressed-sensing measurement matrix.

    Constructs an ``M × N`` sensing matrix of the requested type.  When a
    *seed* is given the global RNG state is saved, the seed is applied, and
    the original state is restored afterwards so that calling code is not
    affected.

    Args:
        M: Number of measurements (rows).
        N: Signal dimension (columns).  Must satisfy ``M <= N``.
        matrix_type: Distribution family for the matrix entries.

            * ``'gaussian'`` — i.i.d. :math:`\\mathcal{N}(0, 1/M)` entries.
            * ``'bernoulli'`` — i.i.d. :math:`\\pm 1/\\sqrt{M}` entries.
            * ``'partial_fourier'`` — ``M`` randomly selected rows of the
              ``N × N`` normalised DFT matrix.

        seed: Optional seed for reproducibility.

    Returns:
        Sensing matrix of shape ``(M, N)`` with dtype ``torch.float32``.

    Raises:
        ValueError: If ``matrix_type`` is not one of the supported types.
    """
    if seed is not None:
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        np_rng_state = np.random.get_state()
        np.random.seed(seed)

    try:
        if matrix_type == "gaussian":
            Phi = torch.randn(M, N) / math.sqrt(M)

        elif matrix_type == "bernoulli":
            Phi = (
                2.0 * torch.bernoulli(0.5 * torch.ones(M, N)) - 1.0
            ) / math.sqrt(M)

        elif matrix_type == "partial_fourier":
            # Build the full N×N normalised DFT matrix and select M rows.
            dft_full = np.fft.fft(np.eye(N)) / math.sqrt(N)
            row_indices = np.random.choice(N, size=M, replace=False)
            row_indices.sort()
            Phi = torch.tensor(
                dft_full[row_indices].real, dtype=torch.float32
            )

        else:
            raise ValueError(
                f"Unsupported matrix type '{matrix_type}'. "
                "Choose from 'gaussian', 'bernoulli', or 'partial_fourier'."
            )
    finally:
        if seed is not None:
            torch.random.set_rng_state(rng_state)
            np.random.set_state(np_rng_state)

    return Phi


def generate_sparse_signal(
    N: int,
    sparsity_ratio: float,
    num_samples: int,
    distribution: Literal["gaussian", "uniform", "laplacian"] = "gaussian",
) -> torch.Tensor:
    """Generate a batch of sparse signals.

    Each signal has ``N`` entries, of which roughly
    ``floor(sparsity_ratio * N)`` are non-zero.  The non-zero values are
    drawn from the specified *distribution* and placed at uniformly random
    positions.

    Args:
        N: Length of each signal.
        sparsity_ratio: Fraction of non-zero entries (0, 1].
        num_samples: Number of signals to generate.
        distribution: Distribution of non-zero entries.

            * ``'gaussian'`` — :math:`\\mathcal{N}(0, 1)`.
            * ``'uniform'`` — :math:`\\mathrm{Uniform}(-1, 1)`.
            * ``'laplacian'`` — :math:`\\mathrm{Laplace}(0, 1)`.

    Returns:
        Signal tensor of shape ``(num_samples, N)``.

    Raises:
        ValueError: If *distribution* is not recognised.
    """
    K = max(1, int(sparsity_ratio * N))
    signals = torch.zeros(num_samples, N)

    for i in range(num_samples):
        support = torch.randperm(N)[:K]

        if distribution == "gaussian":
            values = torch.randn(K)
        elif distribution == "uniform":
            values = 2.0 * torch.rand(K) - 1.0
        elif distribution == "laplacian":
            values = torch.distributions.Laplace(0.0, 1.0).sample((K,))
        else:
            raise ValueError(
                f"Unsupported distribution '{distribution}'. "
                "Choose from 'gaussian', 'uniform', or 'laplacian'."
            )

        signals[i, support] = values

    return signals


def create_measurements(
    x: torch.Tensor,
    Phi: torch.Tensor,
    snr_db: Optional[float] = None,
) -> torch.Tensor:
    """Compute compressed measurements ``y = Phi @ x + noise``.

    If *snr_db* is provided, additive white Gaussian noise is scaled so
    that the signal-to-noise ratio of each measurement vector matches the
    requested level.

    Args:
        x: Signal tensor of shape ``(batch, N)`` or ``(N,)``.
        Phi: Sensing matrix of shape ``(M, N)``.
        snr_db: Desired signal-to-noise ratio in decibels.  When ``None``,
            no noise is added.

    Returns:
        Measurement tensor of shape ``(batch, M)`` or ``(M,)``.
    """
    squeeze = False
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze = True

    # y_clean shape: (batch, M)
    y_clean = torch.mm(x, Phi.t())

    if snr_db is not None:
        signal_power = torch.mean(y_clean ** 2, dim=-1, keepdim=True)
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise = torch.randn_like(y_clean) * torch.sqrt(noise_power)
        y = y_clean + noise
    else:
        y = y_clean

    if squeeze:
        y = y.squeeze(0)

    return y


class SparseSignalDataset(Dataset):
    """PyTorch dataset holding ``(y, Phi, x_true)`` measurement tuples.

    This dataset stores precomputed measurements, the sensing matrix, and
    ground-truth signals so that dataloaders can iterate over individual
    samples during training and evaluation.

    Args:
        y: Measurement vectors, shape ``(num_samples, M)``.
        Phi: Sensing matrix, shape ``(M, N)``.  The same matrix is returned
            for every sample.
        x_true: Ground-truth signals, shape ``(num_samples, N)``.
    """

    def __init__(
        self,
        y: torch.Tensor,
        Phi: torch.Tensor,
        x_true: torch.Tensor,
    ) -> None:
        super().__init__()
        if y.shape[0] != x_true.shape[0]:
            raise ValueError(
                f"Number of measurements ({y.shape[0]}) and signals "
                f"({x_true.shape[0]}) must match."
            )
        self.y = y
        self.Phi = Phi
        self.x_true = x_true

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the ``(y, Phi, x_true)`` tuple for sample *idx*."""
        return self.y[idx], self.Phi, self.x_true[idx]


def create_data_loaders(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / validation / test dataloaders from a config dict.

    The *config* dictionary is expected to contain a ``"data"`` key whose
    value is a mapping with the fields documented in the project's default
    ``config.yaml``.  Sensing matrix, signals, and measurements are
    generated on the fly according to those parameters.

    Args:
        config: Nested configuration dictionary.  Required keys under
            ``config["data"]``:

            * ``signal_length`` (int)
            * ``sparsity_ratio`` (float)
            * ``compression_ratio`` (float)
            * ``sensing_matrix`` (str)
            * ``snr_db`` (float | None)
            * ``num_train`` (int)
            * ``num_val`` (int)
            * ``num_test`` (int)

    Returns:
        A 3-tuple ``(train_loader, val_loader, test_loader)``.
    """
    data_cfg = config["data"]
    train_cfg = config.get("training", {})

    N: int = data_cfg["signal_length"]
    sparsity_ratio: float = data_cfg["sparsity_ratio"]
    compression_ratio: float = data_cfg["compression_ratio"]
    matrix_type: str = data_cfg["sensing_matrix"]
    snr_db: Optional[float] = data_cfg.get("snr_db")
    num_train: int = data_cfg["num_train"]
    num_val: int = data_cfg["num_val"]
    num_test: int = data_cfg["num_test"]
    batch_size: int = train_cfg.get("batch_size", 16)
    seed: Optional[int] = config.get("experiment", {}).get("seed")

    M = max(1, int(compression_ratio * N))

    # Generate the shared sensing matrix.
    Phi = generate_sensing_matrix(M, N, matrix_type=matrix_type, seed=seed)

    # Generate sparse signals for each split.
    total_samples = num_train + num_val + num_test
    x_all = generate_sparse_signal(
        N, sparsity_ratio, total_samples, distribution="gaussian"
    )

    # Create measurements.
    y_all = create_measurements(x_all, Phi, snr_db=snr_db)

    # Split into train / val / test.
    x_train = x_all[:num_train]
    y_train = y_all[:num_train]

    x_val = x_all[num_train : num_train + num_val]
    y_val = y_all[num_train : num_train + num_val]

    x_test = x_all[num_train + num_val :]
    y_test = y_all[num_train + num_val :]

    train_dataset = SparseSignalDataset(y_train, Phi, x_train)
    val_dataset = SparseSignalDataset(y_val, Phi, x_val)
    test_dataset = SparseSignalDataset(y_test, Phi, x_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
