"""
Neural networks for reinforcement learning.

Networks define function approximators for both policies (state to action mappings)
and value functions (state to value mappings). This module provides base classes
and implementations for different network architectures.
"""

from .base import NetworkABC
from .composed import ComposedNetwork
from .backbones import BackboneABC, MLPBackbone
from .heads import HeadABC, DiscretePolicyHead