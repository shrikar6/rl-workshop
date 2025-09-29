"""
Neural networks for reinforcement learning.

Networks define function approximators for policies (state to action mappings).
This module provides base classes and implementations for different network architectures.
"""

from .base import NetworkABC, HeadABC
from .backbones import BackboneABC, MLPBackbone
from .policy import PolicyNetworkABC, ComposedPolicyNetwork
from .policy.heads import PolicyHeadABC, DiscretePolicyHead