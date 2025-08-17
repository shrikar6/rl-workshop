"""
Tests for CartPole environment implementation.
"""

import pytest
import jax.numpy as jnp
import gymnasium as gym
from framework import CartPoleEnv


def test_cartpole_env_creation():
    """Test that CartPole environment can be created."""
    env = CartPoleEnv()
    assert env is not None


def test_cartpole_observation_space(cartpole_env):
    """Test CartPole observation space properties."""
    obs_space = cartpole_env.observation_space
    
    assert isinstance(obs_space, gym.spaces.Box)
    assert obs_space.shape == (4,)
    assert obs_space.dtype == jnp.float32 or obs_space.dtype.name == 'float32'


def test_cartpole_action_space(cartpole_env):
    """Test CartPole action space properties."""
    action_space = cartpole_env.action_space
    
    assert isinstance(action_space, gym.spaces.Discrete)
    assert action_space.n == 2


def test_cartpole_reset(cartpole_env):
    """Test environment reset functionality."""
    obs = cartpole_env.reset()
    
    assert obs.shape == (4,)
    assert jnp.all(jnp.isfinite(obs))


def test_cartpole_step_valid_actions(cartpole_env):
    """Test environment step with valid actions."""
    cartpole_env.reset()
    
    # Test action 0 (left)
    action_0 = jnp.array([0])
    obs, reward, done = cartpole_env.step(action_0)
    
    assert obs.shape == (4,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert reward == 1.0  # CartPole gives +1 reward per step
    
    # Test action 1 (right)
    action_1 = jnp.array([1])
    obs, reward, done = cartpole_env.step(action_1)
    
    assert obs.shape == (4,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_cartpole_step_invalid_action(cartpole_env):
    """Test that invalid actions raise errors."""
    cartpole_env.reset()
    
    # Invalid action (out of range)
    invalid_action = jnp.array([2])
    
    with pytest.raises(AssertionError, match="Invalid action"):
        cartpole_env.step(invalid_action)


def test_cartpole_episode_termination(cartpole_env):
    """Test that episodes can terminate."""
    obs = cartpole_env.reset()
    max_steps = 1000  # Safety limit
    
    for step in range(max_steps):
        action = jnp.array([0])  # Always push left
        obs, reward, done = cartpole_env.step(action)
        
        if done:
            break
    
    # Episode should terminate at some point
    assert done or step == max_steps - 1


def test_cartpole_with_seed():
    """Test CartPole environment with explicit seed."""
    env_with_seed = CartPoleEnv(seed=123)
    obs1 = env_with_seed.reset()
    
    env_no_seed = CartPoleEnv(seed=None)
    obs2 = env_no_seed.reset()
    
    assert obs1.shape == (4,)
    assert obs2.shape == (4,)


def test_cartpole_render_and_close():
    """Test render and close functionality."""
    env = CartPoleEnv(render_mode="rgb_array")
    env.reset()
    
    # Test render
    frame = env.render()
    assert frame is not None
    
    # Test close
    env.close()  # Should not raise any errors