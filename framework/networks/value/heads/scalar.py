import jax
import jax.numpy as jnp
from typing import Tuple
from jax import Array
from ..base import ValueHeadABC


class ScalarValueHead(ValueHeadABC):
    """
    Scalar value head for value networks.

    Converts feature representations into a single scalar value estimate
    via a linear projection: V = features . w + b.

    Example:
        head = ScalarValueHead(input_dim=32)

        # Usage in ComposedValueNetwork:
        value_net = ComposedValueNetwork(
            backbone=MLPBackbone(hidden_dims=[64, 32], output_dim=32),
            head=ScalarValueHead(input_dim=32)
        )
    """

    def __init__(self, input_dim: int):
        """
        Initialize scalar value head.

        Args:
            input_dim: Dimensionality of input features (must match backbone output_dim)
        """
        super().__init__(input_dim)

    def forward(self, params: Tuple[Array, Array], features: Array) -> Array:
        """
        Compute scalar value estimate from features.

        Args:
            params: Head parameters (weight vector and bias scalar)
            features: Feature representation from backbone

        Returns:
            Scalar value estimate, shape ()
        """
        w, b = params
        return jnp.dot(features, w) + b

    def init_params(self, key: Array) -> Tuple[Array, Array]:
        """
        Initialize head parameters.

        Args:
            key: JAX random key for parameter initialization

        Returns:
            Tuple of (weight_vector, bias_scalar)
        """
        # Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))
        # fan_out = 1 for scalar output
        scale = jnp.sqrt(2.0 / (self.input_dim + 1))
        w = jax.random.normal(key, (self.input_dim,)) * scale
        b = jnp.zeros(())

        return (w, b)
