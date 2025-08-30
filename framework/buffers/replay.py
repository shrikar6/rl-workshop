"""
Circular replay buffer implementation using JAX arrays.

Stores transitions in fixed-size pre-allocated arrays for memory efficiency
and maintains functional programming principles.
"""

import jax
import jax.numpy as jnp
import gymnasium as gym
from typing import NamedTuple, Tuple
from jax import Array
from .base import BufferABC, Transition
from ..utils import get_input_dim, get_action_dim


class ReplayBufferState(NamedTuple):
    """
    Immutable state for circular replay buffer.
    
    Uses fixed-size JAX arrays to store transitions efficiently.
    """
    observations: Array     # Shape: (capacity, obs_dim) 
    actions: Array         # Shape: (capacity, action_dim)
    rewards: Array         # Shape: (capacity,)
    next_observations: Array  # Shape: (capacity, obs_dim)
    dones: Array           # Shape: (capacity,)
    size: int              # Current number of stored transitions
    position: int          # Next write position (circular)


class ReplayBuffer(BufferABC):
    """
    Circular replay buffer using pre-allocated JAX arrays.
    
    Maintains constant memory usage by overwriting old transitions
    when the buffer is full. Enables efficient random sampling
    for off-policy learning algorithms like DQN.
    """
    
    def __init__(
        self, 
        capacity: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        seed: int = 0
    ):
        """
        Initialize replay buffer with pre-allocated arrays.
        
        Args:
            capacity: Maximum number of transitions to store
            observation_space: Environment observation space
            action_space: Environment action space
            seed: Random seed for sampling
        """
        self.capacity = capacity
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Determine array dimensions
        obs_dim = get_input_dim(observation_space)
        action_dim = get_action_dim(action_space)
        
        # Create initial state with empty arrays
        self.state = ReplayBufferState(
            observations=jnp.zeros((capacity, obs_dim)),
            actions=jnp.zeros((capacity, action_dim)),
            rewards=jnp.zeros(capacity),
            next_observations=jnp.zeros((capacity, obs_dim)),
            dones=jnp.zeros(capacity, dtype=bool),
            size=0,
            position=0
        )
    
    def add(self, state: ReplayBufferState, transition: Transition) -> ReplayBufferState:
        """
        Add a transition to the buffer at current position.
        
        When buffer is full, overwrites the oldest transition (circular).
        
        Args:
            state: Current buffer state
            transition: Transition to add
            
        Returns:
            New buffer state with transition added
        """
        return self._add_jit(state, transition, self.capacity)
    
    @staticmethod
    @jax.jit
    def _add_jit(state: ReplayBufferState, transition: Transition, capacity: int) -> ReplayBufferState:
        """
        JIT-compiled implementation of add operation.
        
        Args:
            state: Current buffer state
            transition: Transition to add
            capacity: Buffer capacity (static)
            
        Returns:
            New buffer state with transition added
        """
        pos = state.position
        
        # Store transition components at current position
        new_observations = state.observations.at[pos].set(transition.observation)
        new_actions = state.actions.at[pos].set(transition.action)
        new_rewards = state.rewards.at[pos].set(transition.reward)
        new_next_observations = state.next_observations.at[pos].set(transition.next_observation)
        new_dones = state.dones.at[pos].set(transition.done)
        
        # Update size and position
        new_size = jnp.minimum(state.size + 1, capacity)
        new_position = (pos + 1) % capacity
        
        return ReplayBufferState(
            observations=new_observations,
            actions=new_actions,
            rewards=new_rewards,
            next_observations=new_next_observations,
            dones=new_dones,
            size=new_size,
            position=new_position
        )
    
    def sample(
        self, 
        state: ReplayBufferState, 
        batch_size: int, 
        key: Array
    ) -> Tuple[ReplayBufferState, Tuple[Array, Array, Array, Array, Array]]:
        """
        Sample a random batch of transitions from the buffer.
        
        Note: Not JIT-compiled due to dynamic state.size in jax.random.choice.
        
        Args:
            state: Current buffer state
            batch_size: Number of transitions to sample
            key: JAX random key for sampling
            
        Returns:
            Tuple of (unchanged_state, batch) where batch contains
            (observations, actions, rewards, next_observations, dones)
        """
        # Sample random indices from stored transitions
        indices = jax.random.choice(key, state.size, shape=(batch_size,), replace=False)
        
        # Extract batch using advanced indexing
        batch_observations = state.observations[indices]
        batch_actions = state.actions[indices]  
        batch_rewards = state.rewards[indices]
        batch_next_observations = state.next_observations[indices]
        batch_dones = state.dones[indices]
        
        batch = (batch_observations, batch_actions, batch_rewards, 
                 batch_next_observations, batch_dones)
        
        return state, batch  # State unchanged by sampling
    
    def can_sample(self, state: ReplayBufferState, batch_size: int) -> bool:
        """
        Check if buffer contains enough transitions for a batch.
        
        Args:
            state: Current buffer state
            batch_size: Desired batch size
            
        Returns:
            True if buffer has at least batch_size transitions
        """
        return state.size >= batch_size