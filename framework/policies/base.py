from abc import ABC, abstractmethod
from typing import Any
import gymnasium as gym
from jax import Array


class PolicyABC(ABC):
    """
    Abstract base class for all policies.
    
    A policy is a pure function that maps (parameters, observation, random_key) -> action.
    This functional design enables easy JIT compilation, vectorization, and clear
    separation between computation (policy) and state management (agent).
    """
    
    @abstractmethod
    def sample_action(self, params: Any, observation: Array, key: Array) -> Array:
        """
        Sample action given parameters, observation, and random key.
        
        Args:
            params: Policy parameters (neural network weights, etc.)
            observation: Current state observation
            key: JAX random key for stochastic policies
            
        Returns:
            Action sampled from the policy distribution
        """
        pass
    
    @abstractmethod
    def init_params(self, key: Array, observation_space: gym.Space, action_space: gym.Space) -> Any:
        """
        Initialize policy parameters.
        
        Args:
            key: JAX random key for parameter initialization
            observation_space: Gymnasium space describing observations
            action_space: Gymnasium space describing actions
            
        Returns:
            Initial policy parameters
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