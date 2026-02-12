from abc import ABC, abstractmethod
from typing import Any
import gymnasium as gym
from jax import Array


class BackboneABC(ABC):
    """
    Abstract base class for network backbones.

    A backbone extracts features from raw observations that can be used by
    different types of output heads.

    Architecture decisions (hidden layer sizes, activation functions) are made
    at construction time. Environment binding (observation space) happens at parameter
    initialization time.
    """

    def __init__(self, output_dim: int):
        """
        Initialize backbone architecture.

        Args:
            output_dim: Dimensionality of feature output
        """
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, params: Any, observation: Array) -> Array:
        """
        Extract features from observations.

        Args:
            params: Backbone parameters (neural network weights, etc.)
            observation: Raw observation from environment

        Returns:
            Feature representation of the observation
        """
        pass

    @abstractmethod
    def init_params(self, key: Array, observation_space: gym.Space) -> Any:
        """
        Initialize backbone parameters for a specific observation space.

        Args:
            key: JAX random key for parameter initialization
            observation_space: Gymnasium space describing observations

        Returns:
            Initial backbone parameters
        """
        pass


class HeadABC(ABC):
    """
    Abstract base class for all network heads.

    A head converts feature representations into task-specific outputs.
    Different head types handle different algorithm requirements while working
    with the same backbone feature extractors.

    Architecture decisions (input dimension) are made at construction time.
    Subclasses define their own init_params signatures based on what
    information they need (e.g., policy heads need action_space,
    value heads do not).
    """

    def __init__(self, input_dim: int):
        """
        Initialize head architecture.

        Args:
            input_dim: Dimensionality of input features (must match backbone output_dim)
        """
        self.input_dim = input_dim


class NetworkABC(ABC):
    """
    Abstract base class for all networks.

    A network combines backbones and heads to create complete function approximators.
    Different network types handle different algorithm requirements (policy vs value).

    Subclasses define their own init_params signatures based on what
    information they need (e.g., policy networks need both observation_space
    and action_space, value networks only need observation_space).
    """