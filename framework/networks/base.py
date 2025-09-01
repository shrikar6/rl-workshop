from abc import ABC, abstractmethod
from typing import Any
import gymnasium as gym
from jax import Array


class NetworkABC(ABC):
    """
    Abstract base class for all neural networks.
    
    A network is a pure function that can be either a policy (maps states to actions)
    or a value function (maps states to values). This functional design enables easy
    JIT compilation, vectorization, and clear separation between computation (network)
    and state management (agent).
    """
    
    @abstractmethod
    def sample_action(self, params: Any, observation: Array, key: Array) -> Array:
        """
        Sample action given parameters, observation, and random key.
        
        Args:
            params: Network parameters (neural network weights, etc.)
            observation: Current state observation
            key: JAX random key for stochastic policies
            
        Returns:
            Action sampled from the policy distribution
        """
        pass
    
    @abstractmethod
    def init_params(self, key: Array, observation_space: gym.Space, action_space: gym.Space) -> Any:
        """
        Initialize network parameters.
        
        Args:
            key: JAX random key for parameter initialization
            observation_space: Gymnasium space describing observations
            action_space: Gymnasium space describing actions
            
        Returns:
            Initial network parameters
        """
        pass
    
    @abstractmethod
    def get_log_prob(self, params: Any, observation: Array, action: Array) -> Array:
        """
        Compute log probability of action given observation.
        
        Required for policy gradient methods (REINFORCE, PPO, A2C, etc.).
        For deterministic policies or policies that don't support probability
        computation, this can raise NotImplementedError.
        
        Args:
            params: Policy parameters
            observation: Current state observation
            action: Action taken (as array for consistency)
            
        Returns:
            Log probability of the action
        """
        pass