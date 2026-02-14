import jax
import gymnasium as gym
from typing import Any, Tuple
from jax import Array
from .base import ValueNetworkABC, ValueHeadABC
from ..base import BackboneABC


class ComposedValueNetwork(ValueNetworkABC):
    """
    A value network that composes a backbone and value head.

    The backbone extracts features from observations, and the head converts
    those features into a value estimate.

    Examples:
        value_net = ComposedValueNetwork(
            backbone=MLPBackbone(hidden_dims=[64, 32], output_dim=32),
            head=ScalarValueHead(input_dim=32)
        )
    """

    def __init__(self, backbone: BackboneABC, head: ValueHeadABC):
        """
        Initialize composed value network.

        Args:
            backbone: Feature extraction component
            head: Value output component

        Note:
            backbone.output_dim must match head.input_dim
        """
        if backbone.output_dim != head.input_dim:
            raise ValueError(
                f"Backbone output dimension ({backbone.output_dim}) must match "
                f"head input dimension ({head.input_dim})"
            )

        self.backbone = backbone
        self.head = head

    def forward(self, params: Tuple[Any, Any], observation: Array) -> Array:
        """
        Compute value estimate using composed backbone and head.

        Args:
            params: Tuple of (backbone_params, head_params)
            observation: Current state observation

        Returns:
            Scalar value estimate
        """
        backbone_params, head_params = params
        features = self.backbone.forward(backbone_params, observation)
        return self.head.forward(head_params, features)

    def init_params(self, key: Array, observation_space: gym.Space) -> Tuple[Any, Any]:
        """
        Initialize network parameters.

        Args:
            key: JAX random key for parameter initialization
            observation_space: Gymnasium space describing observations

        Returns:
            Tuple of (backbone_params, head_params)
        """
        backbone_key, head_key = jax.random.split(key, 2)

        backbone_params = self.backbone.init_params(backbone_key, observation_space)
        head_params = self.head.init_params(head_key)

        return (backbone_params, head_params)
