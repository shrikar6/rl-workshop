from abc import ABC, abstractmethod
from typing import Any
import gymnasium as gym
from jax import Array


class BackboneABC(ABC):
    """
    Abstract base class for policy backbones.
    
    A backbone extracts features from raw observations. This allows the same
    feature extraction logic to be reused with different output heads for
    different action spaces (discrete, continuous, etc.).
    
    Architecture decisions (hidden layer sizes, activation functions) are made
    at construction time. Environment binding (observation space) happens at parameter
    initialization time.
    """
    
    def __init__(self, output_dim: int):
        """
        Initialize backbone architecture.
        
        Args:
            output_dim: Dimensionality of feature output
        """
        self.output_dim = output_dim
    
    @abstractmethod
    def forward(self, params: Any, observation: Array) -> Array:
        """
        Extract features from observations.

        Args:
            params: Backbone parameters (neural network weights, etc.)
            observation: Raw observation from environment

        Returns:
            Feature representation of the observation
        """
        pass
    
    @abstractmethod
    def init_params(self, key: Array, observation_space: gym.Space) -> Any:
        """
        Initialize backbone parameters for a specific observation space.
        
        Args:
            key: JAX random key for parameter initialization
            observation_space: Gymnasium space describing observations
            
        Returns:
            Initial backbone parameters
        """
        pass