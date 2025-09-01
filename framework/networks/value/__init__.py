"""
Value network implementations.

Components for value-based RL algorithms that map states to Q-values.
"""

from . import heads
from .base import ValueNetworkABC
from .composed import ComposedValueNetwork