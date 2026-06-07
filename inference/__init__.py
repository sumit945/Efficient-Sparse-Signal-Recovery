"""Inference module for sparse signal reconstruction.

Provides prediction utilities for running trained models on new
compressed measurements and producing reconstructed signals.
"""

from inference.predict import run_prediction

__all__ = [
    "run_prediction",
]
