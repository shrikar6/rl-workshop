"""
Minimalist RL framework for research.

Provides core components for reinforcement learning experiments with
focus on simplicity, modularity, and ease of experimentation.
"""

# Module imports
from . import agents
from . import environments
from . import networks

# Core classes
from .trainer import Trainer
from .tracking import Tracker

# Commonly used components
from .agents import REINFORCEAgent
from .environments import CartPoleEnv, AcrobotEnv, LunarLanderEnv
from .networks import ComposedPolicyNetwork
from .networks.backbones import MLPBackbone
from .networks.policy.heads import DiscretePolicyHead