from abc import ABC, abstractmethod
from typing import Any
import gymnasium as gym
from jax import Array


class HeadABC(ABC):
    """
    Abstract base class for all network heads.
    
    A head converts feature representations into outputs (actions, Q-values, etc.).
    Different head types handle different algorithm requirements while working
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


class NetworkABC(ABC):
    """
    Abstract base class for all networks.
    
    A network combines backbones and heads to create complete function approximators.
    Different network types handle different algorithm requirements (policy vs value).
    """
    
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