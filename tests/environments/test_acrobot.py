"""
Tests for Acrobot environment implementation.
"""

import pytest
import jax.numpy as jnp
import gymnasium as gym
from framework import AcrobotEnv


@pytest.fixture
def acrobot_env():
    """Provides an Acrobot environment for testing."""
    return AcrobotEnv()


def test_acrobot_env_creation():
    """Test that Acrobot environment can be created."""
    env = AcrobotEnv()
    assert env is not None


def test_acrobot_observation_space(acrobot_env):
    """Test Acrobot observation space properties."""
    obs_space = acrobot_env.observation_space
    
    assert isinstance(obs_space, gym.spaces.Box)
    assert obs_space.shape == (6,)  # [cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta1_dot, theta2_dot]
    assert obs_space.dtype == jnp.float32 or obs_space.dtype.name == 'float32'


def test_acrobot_action_space(acrobot_env):
    """Test Acrobot action space properties."""
    action_space = acrobot_env.action_space
    
    assert isinstance(action_space, gym.spaces.Discrete)
    assert action_space.n == 3  # Three actions: -1, 0, +1 torque


def test_acrobot_reset(acrobot_env):
    """Test environment reset functionality."""
    obs = acrobot_env.reset()
    
    assert obs.shape == (6,)
    assert jnp.all(jnp.isfinite(obs))
    
    # Check that cos/sin values are in valid range [-1, 1]
    cos_sin_values = obs[:4]  # First 4 elements are cos/sin
    assert jnp.all(cos_sin_values >= -1.0)
    assert jnp.all(cos_sin_values <= 1.0)


def test_acrobot_step_valid_actions(acrobot_env):
    """Test environment step with valid actions."""
    acrobot_env.reset()
    
    # Test all valid actions
    for action_value in [0, 1, 2]:
        action = jnp.array([action_value])
        obs, reward, done = acrobot_env.step(action)
        
        assert obs.shape == (6,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert reward == -1.0  # Acrobot gives -1 reward per step until goal


def test_acrobot_step_invalid_action(acrobot_env):
    """Test that invalid actions raise errors."""
    acrobot_env.reset()
    
    # Invalid actions (out of range)
    invalid_actions = [jnp.array([3]), jnp.array([-1]), jnp.array([4])]
    
    for invalid_action in invalid_actions:
        with pytest.raises(AssertionError, match="Invalid action"):
            acrobot_env.step(invalid_action)


def test_acrobot_episode_termination(acrobot_env):
    """Test that episodes can terminate."""
    obs = acrobot_env.reset()
    max_steps = 600  # Acrobot has 500 step limit + safety margin
    
    total_reward = 0
    for step in range(max_steps):
        action = jnp.array([1])  # Apply 0 torque
        obs, reward, done = acrobot_env.step(action)
        total_reward += reward
        
        if done:
            break
    
    # Episode should terminate at some point (either success or max steps)
    assert done or step == max_steps - 1
    
    # Check that we accumulated negative rewards (expected for Acrobot)
    assert total_reward <= 0


def test_acrobot_with_seed():
    """Test Acrobot environment with explicit seed."""
    env_with_seed = AcrobotEnv(seed=123)
    obs1 = env_with_seed.reset()
    
    env_no_seed = AcrobotEnv(seed=None)
    obs2 = env_no_seed.reset()
    
    assert obs1.shape == (6,)
    assert obs2.shape == (6,)


def test_acrobot_render_and_close():
    """Test render and close functionality."""
    env = AcrobotEnv(render_mode="rgb_array")
    env.reset()
    
    # Test render
    frame = env.render()
    assert frame is not None
    
    # Test close
    env.close()  # Should not raise any errors


def test_acrobot_observation_bounds(acrobot_env):
    """Test that observations stay within expected bounds."""
    acrobot_env.reset()
    
    # Run a few steps and check observation bounds
    for _ in range(10):
        action = jnp.array([acrobot_env.action_space.sample()])
        obs, _, done = acrobot_env.step(action)
        
        # cos/sin values should be in [-1, 1]
        cos_sin_values = obs[:4]
        assert jnp.all(cos_sin_values >= -1.0)
        assert jnp.all(cos_sin_values <= 1.0)
        
        # Angular velocities should be finite
        angular_velocities = obs[4:6]
        assert jnp.all(jnp.isfinite(angular_velocities))
        
        if done:
            break


def test_acrobot_reward_structure(acrobot_env):
    """Test Acrobot reward structure."""
    acrobot_env.reset()
    
    # Run a few steps and verify reward structure
    rewards = []
    for _ in range(5):
        action = jnp.array([1])  # Apply 0 torque
        _, reward, done = acrobot_env.step(action)
        rewards.append(reward)
        
        if done:
            break
    
    # All rewards should be -1.0 until goal is reached
    for reward in rewards[:-1]:  # All but last (which might be different if goal reached)
        assert reward == -1.0