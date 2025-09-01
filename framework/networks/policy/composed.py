import jax
import gymnasium as gym
from typing import Any
from jax import Array
from .base import PolicyNetworkABC
from ..backbones import BackboneABC
from .heads import PolicyHeadABC


class ComposedPolicyNetwork(PolicyNetworkABC):
    """
    A policy network that composes a backbone and policy head.
    
    This enables maximum reusability by separating feature extraction (backbone)
    from policy output generation (head). Any backbone can be combined with any 
    policy head for different action spaces and algorithms.
    
    Examples:
        # For discrete policy (REINFORCE, PPO, etc.)
        policy = ComposedPolicyNetwork(
            backbone=MLPBackbone(hidden_dims=[64, 32], output_dim=32),
            head=DiscretePolicyHead(input_dim=32)
        )
    """
    
    def __init__(self, backbone: BackboneABC, head: PolicyHeadABC):
        """
        Initialize composed policy network.
        
        Args:
            backbone: Feature extraction component
            head: Policy output component
            
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
        Raw policy outputs using composed backbone and head.
        
        Args:
            params: Tuple of (backbone_params, head_params)
            observation: Current state observation
            
        Returns:
            Raw policy outputs (logits for discrete, means for continuous)
        """
        backbone_params, head_params = params
        features = self.backbone.forward(backbone_params, observation)
        return self.head.forward(head_params, features)
    
    def sample_action(self, params, observation, key):
        """
        Sample action using composed backbone and head.
        
        Args:
            params: Tuple of (backbone_params, head_params)
            observation: Current state observation
            key: JAX random key for stochastic action sampling
            
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