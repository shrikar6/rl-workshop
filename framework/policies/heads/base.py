from abc import ABC, abstractmethod
from typing import Any
import gymnasium as gym
from jax import Array


class HeadABC(ABC):
    """
    Abstract base class for policy heads.
    
    A head converts feature representations into actions. Different heads
    handle different action spaces (discrete vs continuous) while working
    with the same backbone feature extractors.
    
    Architecture decisions (input dimension) are made at construction time.
    Environment binding (action space) happens at parameter initialization time.
    """
    
    def __init__(self, input_dim: int):
        """
        Initialize head architecture.
        
        Args:
            input_dim: Dimensionality of input features (must match backbone output_dim)
        """
        self.input_dim = input_dim
    
    @staticmethod
    @abstractmethod
    def sample_action(params: Any, features: Array, key: Array) -> Array:
        """
        Sample action from features.
        
        Args:
            params: Head parameters (neural network weights, etc.)
            features: Feature representation from backbone
            key: JAX random key for stochastic action sampling
            
        Returns:
            Action sampled from the policy distribution
        """
        pass
    
    @abstractmethod
    def init_params(self, key: Array, action_space: gym.Space) -> Any:
        """
        Initialize head parameters for a specific action space.
        
        Args:
            key: JAX random key for parameter initialization
            action_space: Gymnasium space describing actions
            
        Returns:
            Initial head parameters
        """
        pass
    
    @staticmethod
    @abstractmethod
    def get_log_prob(params: Any, features: Array, action: Array) -> Array:
        """
        Compute log probability of action given features.
        
        Args:
            params: Head parameters
            features: Feature representation from backbone
            action: Action taken (as array for consistency)
            
        Returns:
            Log probability of the action
        """
        pass