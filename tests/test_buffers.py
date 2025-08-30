"""
Tests for replay buffer implementations.
"""

import pytest
import jax
import jax.numpy as jnp
import gymnasium as gym
from framework.buffers import ReplayBuffer, Transition


class TestReplayBuffer:
    """Tests for ReplayBuffer implementation."""
    
    def test_buffer_initialization(self):
        """Test replay buffer initialization."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        action_space = gym.spaces.Discrete(2)
        
        buffer = ReplayBuffer(
            capacity=100,
            observation_space=obs_space,
            action_space=action_space
        )
        
        assert buffer.capacity == 100
        assert buffer.state.size == 0
        assert buffer.state.position == 0
        assert buffer.state.observations.shape == (100, 4)
        assert buffer.state.actions.shape == (100, 2)  # Discrete actions stored as one-hot
        assert buffer.state.rewards.shape == (100,)
        assert buffer.state.next_observations.shape == (100, 4)
        assert buffer.state.dones.shape == (100,)
    
    def test_add_single_transition(self):
        """Test adding a single transition to buffer."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        action_space = gym.spaces.Discrete(2)
        buffer = ReplayBuffer(100, obs_space, action_space)
        
        # Create a transition
        obs = jnp.array([1.0, 2.0, 3.0, 4.0])
        action = jnp.array([0, 1])  # One-hot encoded
        reward = 1.0
        next_obs = jnp.array([1.1, 2.1, 3.1, 4.1])
        done = False
        
        transition = Transition(obs, action, reward, next_obs, done)
        
        # Add transition
        new_state = buffer.add(buffer.state, transition)
        
        # Check state updates
        assert new_state.size == 1
        assert new_state.position == 1
        
        # Check stored data
        assert jnp.allclose(new_state.observations[0], obs)
        assert jnp.allclose(new_state.actions[0], action)
        assert new_state.rewards[0] == reward
        assert jnp.allclose(new_state.next_observations[0], next_obs)
        assert new_state.dones[0] == done
    
    def test_add_multiple_transitions(self):
        """Test adding multiple transitions."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        action_space = gym.spaces.Discrete(2)
        buffer = ReplayBuffer(10, obs_space, action_space)
        
        state = buffer.state
        
        # Add 5 transitions
        for i in range(5):
            transition = Transition(
                observation=jnp.array([i, i+1], dtype=float),
                action=jnp.array([1, 0]),  # One-hot
                reward=float(i),
                next_observation=jnp.array([i+1, i+2], dtype=float),
                done=i == 4
            )
            state = buffer.add(state, transition)
        
        # Check final state
        assert state.size == 5
        assert state.position == 5
        
        # Check stored data
        assert state.rewards[0] == 0.0
        assert state.rewards[4] == 4.0
        assert state.dones[4] == True
        assert jnp.allclose(state.observations[2], jnp.array([2.0, 3.0]))
    
    def test_circular_buffer_behavior(self):
        """Test that buffer overwrites old data when full."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        action_space = gym.spaces.Discrete(2)
        buffer = ReplayBuffer(3, obs_space, action_space)  # Small capacity
        
        state = buffer.state
        
        # Add 5 transitions (more than capacity)
        for i in range(5):
            transition = Transition(
                observation=jnp.array([float(i)]),
                action=jnp.array([1, 0]),
                reward=float(i),
                next_observation=jnp.array([float(i+1)]),
                done=False
            )
            state = buffer.add(state, transition)
        
        # Size should be capped at capacity
        assert state.size == 3
        assert state.position == 2  # (5 % 3) = 2
        
        # Should contain the last 3 transitions (2, 3, 4)
        # But they're stored in positions [2, 0, 1] due to circular nature
        assert state.rewards[2] == 2.0  # First stored (i=2)
        assert state.rewards[0] == 3.0  # Second stored (i=3) 
        assert state.rewards[1] == 4.0  # Third stored (i=4)
    
    def test_can_sample(self):
        """Test can_sample method."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        action_space = gym.spaces.Discrete(2)
        buffer = ReplayBuffer(10, obs_space, action_space)
        
        # Empty buffer
        assert not buffer.can_sample(buffer.state, 1)
        assert not buffer.can_sample(buffer.state, 5)
        
        # Add some transitions
        state = buffer.state
        for i in range(3):
            transition = Transition(
                observation=jnp.zeros(2),
                action=jnp.array([1, 0]),
                reward=0.0,
                next_observation=jnp.zeros(2),
                done=False
            )
            state = buffer.add(state, transition)
        
        # Now we have 3 transitions
        assert buffer.can_sample(state, 1)
        assert buffer.can_sample(state, 3)
        assert not buffer.can_sample(state, 4)
        assert not buffer.can_sample(state, 10)
    
    def test_sample_batch(self):
        """Test sampling a batch of transitions."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        action_space = gym.spaces.Discrete(2)
        buffer = ReplayBuffer(10, obs_space, action_space)
        
        # Add some transitions with distinct values
        state = buffer.state
        for i in range(5):
            transition = Transition(
                observation=jnp.array([i, i+10], dtype=float),
                action=jnp.array([1, 0]),
                reward=float(i * 10),
                next_observation=jnp.array([i+1, i+11], dtype=float),
                done=i == 4
            )
            state = buffer.add(state, transition)
        
        # Sample a batch
        key = jax.random.PRNGKey(42)
        new_state, batch = buffer.sample(state, batch_size=3, key=key)
        
        # State should be unchanged
        assert new_state is state
        
        # Check batch structure
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = batch
        
        assert obs_batch.shape == (3, 2)
        assert action_batch.shape == (3, 2)  
        assert reward_batch.shape == (3,)
        assert next_obs_batch.shape == (3, 2)
        assert done_batch.shape == (3,)
        
        # Check that we got valid data (rewards should be multiples of 10)
        assert jnp.all(reward_batch % 10 == 0)
        assert jnp.all((reward_batch >= 0) & (reward_batch <= 40))
    
    def test_sample_without_replacement(self):
        """Test that sampling without replacement gives unique transitions."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        action_space = gym.spaces.Discrete(2)
        buffer = ReplayBuffer(10, obs_space, action_space)
        
        # Add transitions with unique rewards
        state = buffer.state
        for i in range(5):
            transition = Transition(
                observation=jnp.zeros(1),
                action=jnp.array([1, 0]),
                reward=float(i),  # Unique rewards: 0, 1, 2, 3, 4
                next_observation=jnp.zeros(1),
                done=False
            )
            state = buffer.add(state, transition)
        
        # Sample all 5 transitions
        key = jax.random.PRNGKey(123)
        _, batch = buffer.sample(state, batch_size=5, key=key)
        _, _, reward_batch, _, _ = batch
        
        # All rewards should be unique
        unique_rewards = jnp.unique(reward_batch)
        assert len(unique_rewards) == 5
        assert jnp.array_equal(jnp.sort(unique_rewards), jnp.array([0, 1, 2, 3, 4]))
    
    def test_sample_error_when_insufficient_data(self):
        """Test that sampling fails when not enough data."""
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        action_space = gym.spaces.Discrete(2)
        buffer = ReplayBuffer(10, obs_space, action_space)
        
        # Add only 2 transitions
        state = buffer.state
        for i in range(2):
            transition = Transition(
                observation=jnp.zeros(1),
                action=jnp.array([1, 0]),
                reward=0.0,
                next_observation=jnp.zeros(1),
                done=False
            )
            state = buffer.add(state, transition)
        
        # Should not be able to sample 5 transitions without replacement
        key = jax.random.PRNGKey(0)
        with pytest.raises(ValueError):
            buffer.sample(state, batch_size=5, key=key)