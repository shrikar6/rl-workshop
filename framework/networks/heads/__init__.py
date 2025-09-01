"""
Policy head implementations.

Heads convert backbone features into actions for different action spaces
(discrete, continuous, etc.).
"""

from .base import HeadABC
from .discrete_policy import DiscretePolicyHead