import jax
import jax.numpy as jnp
import gymnasium as gym
from typing import Any, Sequence
from jax import Array
from .base import BackboneABC
from ...utils import get_input_dim


class MLPBackbone(BackboneABC):
    """
    Multi-layer perceptron backbone for feature extraction.
    
    Creates a feedforward neural network that transforms raw observations
    into feature representations. The feature representation can then be
    used by different heads for different action types.
    
    Example:
        backbone = MLPBackbone(
            hidden_dims=[64, 32], 
            output_dim=16,
            activation=jax.nn.relu
        )
    """
    
    def __init__(self, hidden_dims: Sequence[int], output_dim: int, activation=jax.nn.relu):
        """
        Initialize MLP backbone architecture.
        
        Args:
            hidden_dims: Sizes of hidden layers
            output_dim: Dimensionality of feature output (must match head input_dim)
            activation: Activation function to use between layers
        """
        super().__init__(output_dim)
        self.hidden_dims = tuple(hidden_dims)
        self.activation = activation
    
    def __call__(self, params: Any, observation: Array) -> Array:
        """
        Extract features from observations using the MLP.
        
        Args:
            params: MLP parameters (list of (weight, bias) tuples)
            observation: Raw observation from environment
            
        Returns:
            Feature representation of the observation
        """
        x = observation
        for w, b in params[:-1]:
            x = self.activation(jnp.dot(x, w) + b)
        
        w_final, b_final = params[-1]
        features = jnp.dot(x, w_final) + b_final
        
        return features
    
    def init_params(self, key: Array, observation_space: gym.Space) -> Any:
        """
        Initialize MLP parameters for a specific observation space.
        
        Args:
            key: JAX random key for parameter initialization
            observation_space: Gymnasium space describing observations
            
        Returns:
            List of (weight, bias) tuples for each layer
        """
        input_dim = get_input_dim(observation_space)
        layer_dims = [input_dim] + list(self.hidden_dims) + [self.output_dim]
        
        params = []
        keys = jax.random.split(key, len(layer_dims) - 1)

        for in_dim, out_dim, layer_key in zip(layer_dims[:-1], layer_dims[1:], keys):
            # Xavier/Glorot initialization
            scale = jnp.sqrt(2.0 / (in_dim + out_dim))
            w_key, _ = jax.random.split(layer_key)
            
            w = jax.random.normal(w_key, (in_dim, out_dim)) * scale
            b = jnp.zeros(out_dim)
            
            params.append((w, b))
        
        return params