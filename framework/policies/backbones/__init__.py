"""
Policy backbone implementations.

Backbones extract features from raw observations that can be used by
different types of output heads (discrete, continuous, etc.).
"""

from .base import BackboneABC
from .mlp import MLPBackbone