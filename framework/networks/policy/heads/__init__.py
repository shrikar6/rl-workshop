"""
Policy head implementations.

Heads convert backbone features into policy outputs (actions, log probabilities)
for policy-based RL algorithms.
"""

from ..base import PolicyHeadABC
from .discrete import DiscretePolicyHead