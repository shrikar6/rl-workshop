from abc import abstractmethod
from typing import Any
from jax import Array
from ...base import HeadABC


class ValueHeadABC(HeadABC):
    """
    Abstract base class for value network heads.
    
    Value heads convert features into Q-values for value-based
    RL algorithms (DQN, Double DQN, etc.).
    """
    
    @staticmethod
    @abstractmethod
    def forward(params: Any, features: Array) -> Array:
        """
        Compute Q-values for all actions.
        
        Args:
            params: Head parameters
            features: Feature representation from backbone
            
        Returns:
            Q-values for all actions
        """
        pass
    
    @staticmethod
    @abstractmethod
    def get_value(params: Any, features: Array, action: Array) -> float:
        """
        Get Q-value for specific action.
        
        Args:
            params: Head parameters
            features: Feature representation from backbone
            action: Action as array
            
        Returns:
            Q-value for the specified action
        """
        pass
    
    @staticmethod
    @abstractmethod
    def select_greedy_action(params: Any, features: Array) -> Array:
        """
        Select action with highest Q-value.
        
        Args:
            params: Head parameters
            features: Feature representation from backbone
            
        Returns:
            Action with highest Q-value
        """
        pass