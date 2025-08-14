from abc import ABC, abstractmethod
from typing import Tuple
import gymnasium as gym
from jax import Array


class EnvironmentABC(ABC):
    """
    Abstract base class for all reinforcement learning environments.
    
    Defines a minimal interface focused on the core responsibilities:
    simulate world, provide observations/rewards, manage episodes, define problem space.
    """
    
    @abstractmethod
    def reset(self) -> Array:
        """
        Reset the environment to initial state and return the initial observation.
        
        Returns:
            Initial state observation
        """
        pass
    
    @abstractmethod
    def step(self, action: Array) -> Tuple[Array, float, bool]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take in the environment
            
        Returns:
            Tuple of (observation, reward, done):
            - observation: Next state observation  
            - reward: Reward received for this step
            - done: Whether episode has ended
        """
        pass
    
    @property
    @abstractmethod
    def observation_space(self) -> gym.Space:
        """Get the observation space."""
        pass
    
    @property
    @abstractmethod
    def action_space(self) -> gym.Space:
        """Get the action space."""
        pass