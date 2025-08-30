"""
Abstract base class for replay buffers.

Defines the interface that all buffer implementations must follow,
maintaining functional programming principles with immutable state.
"""

from abc import ABC, abstractmethod
from typing import NamedTuple, Any, Tuple
from jax import Array


class Transition(NamedTuple):
    """
    A single transition in the environment.
    
    Immutable representation of one step of interaction.
    """
    observation: Array
    action: Array
    reward: float
    next_observation: Array
    done: bool


class BufferABC(ABC):
    """
    Abstract base class for replay buffers.
    
    Follows functional programming principles - all methods return new state
    rather than mutating existing state.
    """
    
    @abstractmethod
    def add(self, state: Any, transition: Transition) -> Any:
        """
        Add a transition to the buffer and return new buffer state.
        
        Args:
            state: Current buffer state
            transition: Transition to add
            
        Returns:
            New buffer state with transition added
        """
        pass
    
    @abstractmethod
    def sample(self, state: Any, batch_size: int, key: Array) -> Tuple[Any, Tuple[Array, ...]]:
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            state: Current buffer state
            batch_size: Number of transitions to sample
            key: JAX random key for sampling
            
        Returns:
            Tuple of (new_buffer_state, batch) where batch is a tuple of
            (observations, actions, rewards, next_observations, dones)
        """
        pass
    
    @abstractmethod
    def can_sample(self, state: Any, batch_size: int) -> bool:
        """
        Check if buffer has enough transitions to sample a batch.
        
        Args:
            state: Current buffer state
            batch_size: Desired batch size
            
        Returns:
            True if buffer has at least batch_size transitions
        """
        pass