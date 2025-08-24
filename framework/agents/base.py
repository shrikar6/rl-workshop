from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict
import gymnasium as gym
from jax import Array
from ..policies import PolicyABC


class AgentABC(ABC):
    """
    Abstract base class for all reinforcement learning agents.
    
    An agent combines a policy (for action selection) with a learning algorithm
    (for updating the policy from experience). This separation enables mixing
    different algorithms with different policy architectures.
    
    This interface follows functional programming principles where all methods
    are pure functions that return new state rather than mutating existing state.
    Each algorithm defines its own state structure as needed.
    
    Subclasses should implement:
        __init__(self, policy: PolicyABC, observation_space: gym.Space, action_space: gym.Space, **kwargs)
    to initialize with policy and environment space specifications, and create initial state.
    """
    
    @abstractmethod
    def select_action(self, state: Any, observation: Array, key: Array) -> Tuple[Array, Any]:
        """
        Choose an action given an observation and return new state.
        
        Args:
            state: Current agent state
            observation: Current state observation
            key: JAX random key for stochastic policies
            
        Returns:
            Tuple of (action, new_agent_state)
        """
        pass
    
    @abstractmethod
    def update(self, state: Any, obs: Array, action: Array, reward: float, next_obs: Array, done: bool, key: Array) -> Tuple[Any, Dict[str, float]]:
        """
        Update agent from experience and return new state and metrics.
        
        Args:
            state: Current agent state
            obs: Current observation
            action: Action that was taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode ended
            key: JAX random key for stochastic updates
            
        Returns:
            Tuple of (new agent state, metrics dict)
        """
        pass