"""
Quantum State Discrimination Package
Modern implementation of Minimum-Error Quantum State Discrimination using Semidefinite Programming.
"""

from .core import QuantumStateDiscriminator, DiscriminationResult
from . import states
from . import visualization

__all__ = [
    "QuantumStateDiscriminator",
    "DiscriminationResult",
    "states",
    "visualization"
]
