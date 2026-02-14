from abc import abstractmethod
from typing import Any
import gymnasium as gym
from jax import Array
from ..base import NetworkABC, HeadABC


class ValueHeadABC(HeadABC):
    """
    Abstract base class for value network heads.

    Value heads convert features into value estimates for critic-based
    RL algorithms (A2C, PPO, etc.).
    """

    @abstractmethod
    def init_params(self, key: Array) -> Any:
        """
        Initialize head parameters.

        Args:
            key: JAX random key for parameter initialization

        Returns:
            Initial head parameters
        """
        pass

    @abstractmethod
    def forward(self, params: Any, features: Array) -> Array:
        """
        Compute value estimate from features.

        Args:
            params: Head parameters
            features: Feature representation from backbone

        Returns:
            Value estimate
        """
        pass


class ValueNetworkABC(NetworkABC):
    """
    Abstract base class for value networks.

    Value networks map states to value estimates for critic-based
    RL algorithms (A2C, PPO, etc.).
    """

    @abstractmethod
    def init_params(self, key: Array, observation_space: gym.Space) -> Any:
        """
        Initialize network parameters.

        Args:
            key: JAX random key for parameter initialization
            observation_space: Gymnasium space describing observations

        Returns:
            Initial network parameters
        """
        pass

    @abstractmethod
    def forward(self, params: Any, observation: Array) -> Array:
        """
        Compute value estimate for an observation.

        Args:
            params: Network parameters
            observation: Current state observation

        Returns:
            Value estimate
        """
        pass
