"""Training module for deep unfolding sparse recovery.

Provides training utilities, data generation, and CLI entrypoints
for training unfolded iterative networks on compressed sensing tasks.
"""

from training.trainer import Trainer

__all__ = ["Trainer"]
