import jax
import gymnasium as gym
from typing import Any
from jax import Array
from .base import NetworkABC
from .backbones import BackboneABC
from .heads import HeadABC
from .backbones.mlp import MLPBackbone
from .heads.discrete_policy import DiscretePolicyHead


class ComposedNetwork(NetworkABC):
    """
    A network implementation that composes a backbone and head.
    
    This enables maximum reusability by separating feature extraction (backbone)
    from output generation (head). Any backbone can be combined with any head.
    Can be used for policies (with policy heads) or value functions (with Q-heads).
    
    Examples:
        # For policy (REINFORCE, etc.)
        policy = ComposedNetwork(
            backbone=MLPBackbone(hidden_dims=[64, 32], output_dim=32),
            head=DiscretePolicyHead(input_dim=32)
        )
        
        # For Q-network (DQN, etc.)  
        q_network = ComposedNetwork(
            backbone=MLPBackbone(hidden_dims=[64, 32], output_dim=32),
            head=DiscreteQHead(input_dim=32)
        )
    """
    
    def sample_action(self, params, observation, key):
        """
        Sample action using composed backbone and head.
        
        Args:
            params: Tuple of (backbone_params, head_params)
            observation: Current state observation
            key: JAX random key for stochastic policies
            
        Returns:
            Action sampled from the policy distribution
        """
        backbone_params, head_params = params
        features = self.backbone.forward(backbone_params, observation)
        action = self.head.sample_action(head_params, features, key)
        return action
    
    def get_log_prob(self, params, observation, action):
        """
        Compute log probability using composed backbone and head.
        
        Args:
            params: Tuple of (backbone_params, head_params)
            observation: State observation
            action: Action taken
            
        Returns:
            Log probability of the action
        """
        backbone_params, head_params = params
        features = self.backbone.forward(backbone_params, observation)
        return self.head.get_log_prob(head_params, features, action)
    
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
    
