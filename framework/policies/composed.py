import jax
import gymnasium as gym
from typing import Any
from jax import Array
from .base import PolicyABC
from .backbones import BackboneABC
from .heads import HeadABC


class ComposedPolicy(PolicyABC):
    """
    A policy implementation that composes a backbone and head.
    
    This enables maximum reusability by separating feature extraction (backbone)
    from action generation (head). Any backbone can be combined with any head.
    
    Examples:
        # MLP backbone + discrete actions
        policy = ComposedPolicy(
            backbone=MLPBackbone(hidden_dims=[64, 32], output_dim=32),
            head=DiscreteHead(input_dim=32)
        )
        
        # CNN backbone + continuous actions  
        policy = ComposedPolicy(
            backbone=CNNBackbone(filters=[32, 64], output_dim=128),
            head=ContinuousHead(input_dim=128)
        )
    """
    
    def __init__(self, backbone: BackboneABC, head: HeadABC):
        """
        Initialize composed policy.
        
        Args:
            backbone: Feature extraction component
            head: Action generation component
            
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
    
    def __call__(self, params: Any, observation: Array, key: Array) -> Array:
        """
        Compute action given parameters, observation, and random key.
        
        Args:
            params: Tuple of (backbone_params, head_params)
            observation: Current state observation
            key: JAX random key for stochastic policies
            
        Returns:
            Action to take in the environment
        """
        backbone_params, head_params = params
        
        features = self.backbone(backbone_params, observation)
        action = self.head(head_params, features, key)
        
        return action
    
    def init_params(self, key: Array, observation_space: gym.Space, action_space: gym.Space) -> Any:
        """
        Initialize policy parameters.
        
        Args:
            key: JAX random key for parameter initialization
            observation_space: Gymnasium space describing observations
            action_space: Gymnasium space describing actions
            
        Returns:
            Tuple of (backbone_params, head_params)
        """
        backbone_key, head_key = jax.random.split(key, 2)
        
        backbone_params = self.backbone.init_params(backbone_key, observation_space)
        head_params = self.head.init_params(head_key, action_space)
        
        return (backbone_params, head_params)
    
    def get_log_prob(self, params: Any, observation: Array, action: Array) -> Array:
        """
        Compute log probability of taking action given observation.
        
        Args:
            params: Tuple of (backbone_params, head_params)
            observation: State observation
            action: Action taken
            
        Returns:
            Log probability of the action
        """
        backbone_params, head_params = params
        
        features = self.backbone(backbone_params, observation)
        return self.head.get_log_prob(head_params, features, action)