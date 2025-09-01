"""
Neural networks for reinforcement learning.

Networks define function approximators for both policies (state to action mappings)
and value functions (state to value mappings). This module provides base classes
and implementations for different network architectures.
"""

from .base import NetworkABC, HeadABC
from .backbones import BackboneABC, MLPBackbone
from .policy import PolicyNetworkABC, ComposedPolicyNetwork
from .policy.heads import PolicyHeadABC, DiscretePolicyHead
from .value import ValueNetworkABC, ComposedValueNetwork
from .value.heads import ValueHeadABC