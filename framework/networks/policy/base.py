from abc import abstractmethod
from typing import Any
from jax import Array
from ..base import NetworkABC


class PolicyNetworkABC(NetworkABC):
    """
    Abstract base class for policy networks.
    
    Policy networks map states to action distributions for policy-based
    RL algorithms (REINFORCE, PPO, A2C, etc.).
    """
    
    @abstractmethod
    def forward(self, params: Any, observation: Array) -> Array:
        """
        Raw policy outputs (logits, means, etc.).
        
        Args:
            params: Network parameters
            observation: Current state observation
            
        Returns:
            Raw policy parameters (logits for discrete, means for continuous)
        """
        pass
    
    @abstractmethod
    def sample_action(self, params: Any, observation: Array, key: Array) -> Array:
        """
        Sample action from policy distribution.
        
        Args:
            params: Network parameters
            observation: Current state observation
            key: JAX random key for stochastic action sampling
            
        Returns:
            Action sampled from the policy distribution
        """
        pass
    
    @abstractmethod
    def get_log_prob(self, params: Any, observation: Array, action: Array) -> float:
        """
        Compute log probability of action for policy gradients.
        
        Args:
            params: Network parameters
            observation: Current state observation
            action: Action taken (as array for consistency)
            
        Returns:
            Log probability of the action
        """
        pass