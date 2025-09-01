import jax
import gymnasium as gym
from typing import Any
from jax import Array
from .base import ValueNetworkABC
from ..backbones import BackboneABC
from .heads import ValueHeadABC


class ComposedValueNetwork(ValueNetworkABC):
    """
    A value network that composes a backbone and value head.
    
    This enables maximum reusability by separating feature extraction (backbone)
    from Q-value output generation (head). Any backbone can be combined with any 
    value head for different action spaces and algorithms.
    
    Examples:
        # For discrete Q-network (DQN, Double DQN, etc.)
        q_network = ComposedValueNetwork(
            backbone=MLPBackbone(hidden_dims=[64, 32], output_dim=32),
            head=DiscreteValueHead(input_dim=32)
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
    
    def forward(self, params, observation):
        """
        Compute Q-values using composed backbone and head.
        
        Args:
            params: Tuple of (backbone_params, head_params)
            observation: Current state observation
            
        Returns:
            Q-values for all actions
        """
        backbone_params, head_params = params
        features = self.backbone.forward(backbone_params, observation)
        q_values = self.head.forward(head_params, features)
        return q_values
    
    def get_value(self, params, observation, action):
        """
        Get Q-value for specific action using composed backbone and head.
        
        Args:
            params: Tuple of (backbone_params, head_params)
            observation: State observation
            action: Action as array
            
        Returns:
            Q-value for the specified action
        """
        backbone_params, head_params = params
        features = self.backbone.forward(backbone_params, observation)
        return self.head.get_value(head_params, features, action)
    
    def select_greedy_action(self, params, observation):
        """
        Select greedy action using composed backbone and head.
        
        Args:
            params: Tuple of (backbone_params, head_params)
            observation: Current state observation
            
        Returns:
            Action with highest Q-value
        """
        backbone_params, head_params = params
        features = self.backbone.forward(backbone_params, observation)
        return self.head.select_greedy_action(head_params, features)
    
    def init_params(self, key: Array, observation_space: gym.Space, action_space: gym.Space) -> Any:
        """
        Initialize network parameters.
        
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