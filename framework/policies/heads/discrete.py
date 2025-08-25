import jax
import jax.numpy as jnp
import gymnasium as gym
from typing import Any
from jax import Array
from .base import HeadABC


class DiscreteHead(HeadABC):
    """
    Discrete action head for policy networks.
    
    Converts feature representations into discrete actions by:
    1. Computing logits for each possible action
    2. Converting logits to probabilities (softmax)
    3. Sampling an action from the probability distribution
    
    Supports gym.spaces.Discrete action spaces only.
    
    Example:
        # For CartPole (2 actions: left=0, right=1)
        head = DiscreteHead(input_dim=32)
        
        # Usage in ComposedPolicy:
        policy = ComposedPolicy(
            backbone=MLPBackbone(hidden_dims=[64, 32], output_dim=32),
            head=DiscreteHead(input_dim=32)
        )
    """
    
    def __init__(self, input_dim: int):
        """
        Initialize discrete action head.
        
        Args:
            input_dim: Dimensionality of input features (must match backbone output_dim)
        """
        super().__init__(input_dim)
    
    @staticmethod
    @jax.jit
    def sample_action(params: Any, features: Array, key: Array) -> Array:
        """
        JIT-compiled discrete action sampling.
        
        Args:
            params: Head parameters (weight matrix and bias vector)
            features: Feature representation from backbone
            key: JAX random key for action sampling
            
        Returns:
            Sampled discrete action as Array([action_index])
        """
        w, b = params
        logits = jnp.dot(features, w) + b
        action = jax.random.categorical(key, logits)
        return jnp.array([action])
    
    
    @staticmethod
    @jax.jit
    def get_log_prob(params: Any, features: Array, action: Array) -> float:
        """
        JIT-compiled log probability computation for specific action.
        
        Useful for policy gradient methods that need to compute log π(a|s).
        
        Args:
            params: Head parameters
            features: Feature representation from backbone
            action: Action as array (consistent with sample_action output)
            
        Returns:
            Log probability of the specified action
        """
        w, b = params
        logits = jnp.dot(features, w) + b
        log_probs = jax.nn.log_softmax(logits)
        action_idx = action[0].astype(int)
        return log_probs[action_idx]
    
    def init_params(self, key: Array, action_space: gym.Space) -> Any:
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
                f"DiscreteHead only supports gym.spaces.Discrete action spaces, "
                f"got {type(action_space)}. For MultiDiscrete or other complex "
                f"discrete spaces, please use a specialized head."
            )
        
        num_actions = action_space.n
        
        # Xavier initialization
        scale = jnp.sqrt(2.0 / (self.input_dim + num_actions))
        w_key, b_key = jax.random.split(key)
        
        w = jax.random.normal(w_key, (self.input_dim, num_actions)) * scale
        b = jnp.zeros(num_actions)
        
        return (w, b)