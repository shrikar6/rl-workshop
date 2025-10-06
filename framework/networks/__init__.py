"""
Neural networks for reinforcement learning.

Networks define function approximators for policies (state to action mappings).
This module provides base classes and implementations for different network architectures.
"""

from .base import NetworkABC, BackboneABC, HeadABC
from .backbones import MLPBackbone
from .policy import PolicyNetworkABC, PolicyHeadABC, ComposedPolicyNetwork
from .policy.heads import DiscretePolicyHead