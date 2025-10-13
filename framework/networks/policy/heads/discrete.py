import jax
import jax.numpy as jnp
import gymnasium as gym
from typing import Any, Tuple
from jax import Array
from ..base import PolicyHeadABC


class DiscretePolicyHead(PolicyHeadABC):
    """
    Discrete action head for policy networks.
    
    Converts feature representations into discrete actions by:
    1. Computing logits for each possible action
    2. Converting logits to probabilities (softmax)
    3. Sampling an action from the probability distribution
    
    Supports gym.spaces.Discrete action spaces only.
    
    Example:
        # For CartPole (2 actions: left=0, right=1)
        head = DiscretePolicyHead(input_dim=32)
        
        # Usage in ComposedNetwork:
        policy = ComposedNetwork(
            backbone=MLPBackbone(hidden_dims=[64, 32], output_dim=32),
            head=DiscretePolicyHead(input_dim=32)
        )
    """
    
    def __init__(self, input_dim: int):
        """
        Initialize discrete action head.
        
        Args:
            input_dim: Dimensionality of input features (must match backbone output_dim)
        """
        super().__init__(input_dim)
    
    def forward(self, params: Tuple[Array, Array], features: Array) -> Array:
        """
        Compute action logits from features.

        Args:
            params: Head parameters (weight matrix and bias vector)
            features: Feature representation from backbone

        Returns:
            Logits for all actions
        """
        w, b = params
        return jnp.dot(features, w) + b

    def sample_action(self, params: Tuple[Array, Array], features: Array, key: Array) -> Array:
        """
        Sample discrete action from policy distribution.

        Args:
            params: Head parameters (weight matrix and bias vector)
            features: Feature representation from backbone
            key: JAX random key for action sampling

        Returns:
            Sampled discrete action as Array([action_index])
        """
        logits = self.forward(params, features)
        action = jax.random.categorical(key, logits)
        return action.reshape(-1)

    def get_log_prob(self, params: Tuple[Array, Array], features: Array, action: Array) -> float:
        """
        Compute log probability of action under policy.

        Useful for policy gradient methods that need to compute log π(a|s).

        Args:
            params: Head parameters (weight matrix and bias vector)
            features: Feature representation from backbone
            action: Action as array (consistent with sample_action output)

        Returns:
            Log probability of the specified action
        """
        logits = self.forward(params, features)
        log_probs = jax.nn.log_softmax(logits)
        action_idx = action[0].astype(int)
        return log_probs[action_idx]
    
    def init_params(self, key: Array, action_space: gym.Space) -> Tuple[Array, Array]:
        """
        Initialize head parameters for discrete actions.

        Args:
            key: JAX random key for parameter initialization
            action_space: Gymnasium space describing actions (must be Discrete)

        Returns:
            Tuple of (weight_matrix, bias_vector)

        Raises:
            ValueError: If action_space is not gym.spaces.Discrete
        """
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError(
                f"DiscretePolicyHead only supports gym.spaces.Discrete action spaces, "
                f"got {type(action_space)}. For MultiDiscrete or other complex "
                f"discrete spaces, please use a specialized head."
            )
        
        num_actions = action_space.n
        
        # Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))
        scale = jnp.sqrt(2.0 / (self.input_dim + num_actions))
        w_key, b_key = jax.random.split(key)
        
        w = jax.random.normal(w_key, (self.input_dim, num_actions)) * scale
        b = jnp.zeros(num_actions)
        
        return (w, b)