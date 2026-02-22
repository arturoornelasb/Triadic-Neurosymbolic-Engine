"""
Triadic Neurosymbolic Engine — Core Package

Deterministic algebraic framework for neurosymbolic validation,
semantic projection, and AI model auditing.
"""

from neurosym.encoder import (
    BaseEncoder,
    ContinuousEncoder,
    OpenAIEncoder,
    CohereEncoder,
    DiscreteMapper,
    create_encoder,
)
from neurosym.triadic import DiscreteValidator
from neurosym.storage import PrimeIndexDB

__all__ = [
    "BaseEncoder",
    "ContinuousEncoder",
    "OpenAIEncoder",
    "CohereEncoder",
    "DiscreteMapper",
    "DiscreteValidator",
    "PrimeIndexDB",
    "create_encoder",
]

__version__ = "0.1.0"
