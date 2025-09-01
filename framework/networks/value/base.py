from abc import abstractmethod
from typing import Any
from jax import Array
from ..base import NetworkABC


class ValueNetworkABC(NetworkABC):
    """
    Abstract base class for value networks.
    
    Value networks map states to Q-values for value-based
    RL algorithms (DQN, Double DQN, etc.).
    """
    
    @abstractmethod
    def forward(self, params: Any, observation: Array) -> Array:
        """
        Compute Q-values for all actions.
        
        Args:
            params: Network parameters
            observation: Current state observation
            
        Returns:
            Q-values for all actions
        """
        pass
    
    @abstractmethod
    def get_value(self, params: Any, observation: Array, action: Array) -> float:
        """
        Get Q-value for specific action.
        
        Args:
            params: Network parameters
            observation: Current state observation
            action: Action as array
            
        Returns:
            Q-value for the specified action
        """
        pass
    
    @abstractmethod
    def select_greedy_action(self, params: Any, observation: Array) -> Array:
        """
        Select action with highest Q-value.
        
        Args:
            params: Network parameters
            observation: Current state observation
            
        Returns:
            Action with highest Q-value
        """
        pass