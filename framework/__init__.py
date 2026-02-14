"""
Minimalist RL framework for research.

Provides core components for reinforcement learning experiments with
focus on simplicity, modularity, and ease of experimentation.
"""

from .trainer import Trainer
from .tracking import Tracker
from .agents.reinforce import REINFORCEAgent
from .environments.cartpole import CartPoleEnv
from .environments.acrobot import AcrobotEnv
from .environments.lunarlander import LunarLanderEnv
from .networks.policy.composed import ComposedPolicyNetwork
from .networks.backbones.mlp import MLPBackbone
from .networks.policy.heads.discrete import DiscretePolicyHead
from .networks.value.composed import ComposedValueNetwork
from .networks.value.heads.scalar import ScalarValueHead
