import jax
import jax.numpy as jnp
import gymnasium as gym
from typing import Any, Sequence, Callable, Optional, List, Tuple
from functools import partial
from jax import Array
from .base import BackboneABC
from ...utils import get_input_dim

# Type alias for MLP parameter structure
MLPParams = List[Tuple[Array, Array]]  # List of (weight, bias) tuples


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
            activation=jax.nn.relu  # or jax.nn.tanh, jax.nn.elu, etc.
        )
    """
    
    def forward(self, params: MLPParams, observation: Array) -> Array:
        """
        MLP forward pass with configurable activation.

        Args:
            params: MLP parameters (list of (weight, bias) tuples)
            observation: Raw observation from environment

        Returns:
            Feature representation of the observation
        """
        return self._forward_jit(params, observation, self.activation)
    
    @staticmethod
    @partial(jax.jit, static_argnums=(2,))  # Mark activation as static for JIT
    def _forward_jit(params: MLPParams, observation: Array, activation: Callable[[Array], Array]) -> Array:
        """
        JIT-compiled MLP forward pass implementation.

        Args:
            params: MLP parameters (list of (weight, bias) tuples)
            observation: Raw observation from environment
            activation: Activation function to apply

        Returns:
            Feature representation of the observation
        """
        x = observation
        for w, b in params[:-1]:
            x = activation(jnp.dot(x, w) + b)

        w_final, b_final = params[-1]
        return jnp.dot(x, w_final) + b_final
    
    def __init__(
        self, 
        hidden_dims: Sequence[int], 
        output_dim: int,
        activation: Optional[Callable[[Array], Array]] = None
    ):
        """
        Initialize MLP backbone architecture.
        
        Args:
            hidden_dims: Sizes of hidden layers
            output_dim: Dimensionality of feature output (must match head input_dim)
            activation: Activation function (default: jax.nn.relu)
        """
        super().__init__(output_dim)
        self.hidden_dims = tuple(hidden_dims)
        self.activation = activation if activation is not None else jax.nn.relu
    
    def init_params(self, key: Array, observation_space: gym.Space) -> MLPParams:
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