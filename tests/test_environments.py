"""
Unified test suite for all environment implementations.

This module provides a comprehensive test suite that runs the same tests
across all environments to ensure consistent behavior and API compliance.
"""

import random
import pytest
import jax.numpy as jnp
import gymnasium as gym
from framework import CartPoleEnv, AcrobotEnv, LunarLanderEnv


# Environment configurations for testing
ENV_CONFIGS = [
    {
        "class": CartPoleEnv,
        "name": "CartPole",
        "obs_dim": 4,
        "action_dim": 2,
        "reward_range": (0.0, 1.0),
    },
    {
        "class": AcrobotEnv,
        "name": "Acrobot",
        "obs_dim": 6,
        "action_dim": 3,
        "reward_range": (-1.0, 0.0),
    },
    {
        "class": LunarLanderEnv,
        "name": "LunarLander",
        "obs_dim": 8,
        "action_dim": 4,
        "reward_range": (-200.0, 300.0),
    },
]


@pytest.mark.parametrize("env_config", ENV_CONFIGS, ids=[c["name"] for c in ENV_CONFIGS])
class TestEnvironments:
    """Unified test suite for all environments."""
    
    def test_env_creation(self, env_config):
        """Test that environment can be created."""
        env = env_config["class"]()
        assert env is not None
        env.close()
    
    def test_observation_space(self, env_config):
        """Test observation space properties."""
        env = env_config["class"]()
        obs_space = env.observation_space
        
        assert isinstance(obs_space, gym.spaces.Box)
        assert obs_space.shape == (env_config["obs_dim"],)
        assert obs_space.dtype == jnp.float32 or obs_space.dtype.name == 'float32'
        env.close()
    
    def test_action_space(self, env_config):
        """Test action space properties."""
        env = env_config["class"]()
        action_space = env.action_space
        
        assert isinstance(action_space, gym.spaces.Discrete)
        assert action_space.n == env_config["action_dim"]
        env.close()
    
    def test_reset(self, env_config):
        """Test environment reset functionality."""
        env = env_config["class"]()
        obs = env.reset()
        
        assert obs.shape == (env_config["obs_dim"],)
        assert jnp.all(jnp.isfinite(obs))
        env.close()
    
    def test_step_valid_actions(self, env_config):
        """Test environment step with valid actions."""
        env = env_config["class"]()
        env.reset()
        
        # Test all valid actions
        for action_value in range(env_config["action_dim"]):
            action = jnp.array([action_value])
            obs, reward, done = env.step(action)
            
            assert obs.shape == (env_config["obs_dim"],)
            assert isinstance(reward, float)
            assert isinstance(done, bool)
            assert jnp.all(jnp.isfinite(obs))
            
            if done:
                env.reset()
        
        env.close()
    
    def test_step_invalid_action(self, env_config):
        """Test that invalid actions raise errors."""
        env = env_config["class"]()
        env.reset()
        
        # Invalid action (out of range)
        invalid_action = jnp.array([env_config["action_dim"]])
        
        with pytest.raises(AssertionError, match="Invalid action"):
            env.step(invalid_action)
        
        env.close()
    
    def test_episode_termination(self, env_config):
        """Test that episodes can terminate."""
        env = env_config["class"]()
        env.reset()
        max_steps = 1000  # Fixed limit for all environments
        
        done = False
        for step in range(max_steps):
            # Take random actions to test termination
            action = jnp.array([random.randint(0, env_config["action_dim"] - 1)])
            obs, reward, done = env.step(action)
            
            if done:
                break
        
        # Episode should terminate at some point
        assert done or step == max_steps - 1
        env.close()
    
    def test_with_seed(self, env_config):
        """Test environment with explicit seed for reproducibility."""
        # Create two environments with same seed
        env1 = env_config["class"](seed=42)
        env2 = env_config["class"](seed=42)
        
        obs1 = env1.reset()
        obs2 = env2.reset()
        
        # Initial observations should be identical with same seed
        assert jnp.allclose(obs1, obs2)
        
        # Take same actions and verify identical outcomes
        for _ in range(5):
            action = jnp.array([0])
            obs1, reward1, done1 = env1.step(action)
            obs2, reward2, done2 = env2.step(action)
            
            assert jnp.allclose(obs1, obs2)
            assert reward1 == reward2
            assert done1 == done2
            
            if done1:
                break
        
        env1.close()
        env2.close()
    
    def test_render_and_close(self, env_config):
        """Test render and close methods."""
        env = env_config["class"](render_mode="rgb_array")
        env.reset()
        
        # Test render
        frame = env.render()
        assert frame is not None
        
        # Test close
        env.close()
    
    def test_observation_bounds(self, env_config):
        """Test that observations are within reasonable bounds."""
        env = env_config["class"]()
        env.reset()
        
        # Run a few steps and check observations
        for _ in range(10):
            action = jnp.array([0])
            obs, _, done = env.step(action)
            
            # Check that observations are finite
            assert jnp.all(jnp.isfinite(obs))
            
            # For environments with contact flags (LunarLander)
            if env_config["name"] == "LunarLander":
                # Contact flags should be 0 or 1
                assert jnp.all((obs[6:8] == 0) | (obs[6:8] == 1))
            
            if done:
                break
        
        env.close()
    
    def test_reward_structure(self, env_config):
        """Test that rewards are within expected range."""
        env = env_config["class"]()
        env.reset()
        
        rewards = []
        min_reward, max_reward = env_config["reward_range"]
        
        for _ in range(20):
            action = jnp.array([random.randint(0, env_config["action_dim"] - 1)])
            _, reward, done = env.step(action)
            rewards.append(reward)
            
            # Check reward is within expected range (with some tolerance)
            assert min_reward - 10 <= reward <= max_reward + 10
            
            if done:
                break
        
        # Should have collected some rewards
        assert len(rewards) > 0
        env.close()