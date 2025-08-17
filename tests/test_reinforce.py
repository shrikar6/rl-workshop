"""
Tests for REINFORCE agent implementation.

Focus on testing the mechanics and correctness of the implementation,
not learning performance (which is tested separately in experiments).
"""

import jax
import jax.numpy as jnp
import pytest
from framework.agents.reinforce import REINFORCEAgent
from framework.policies.composed import ComposedPolicy
from framework.policies.backbones.mlp import MLPBackbone
from framework.policies.heads.discrete import DiscreteHead
from framework.environments.cartpole import CartPoleEnv


class TestREINFORCEAgent:
    """Test suite for REINFORCE agent mechanics."""
    
    @pytest.fixture
    def agent(self):
        """Create a REINFORCE agent for testing."""
        env = CartPoleEnv()
        policy = ComposedPolicy(
            backbone=MLPBackbone(hidden_dims=[32], output_dim=16),
            head=DiscreteHead(input_dim=16)
        )
        
        return REINFORCEAgent(
            policy=policy,
            observation_space=env.observation_space,
            action_space=env.action_space,
            learning_rate=1e-3,
            gamma=0.99
        )
    
    def test_initialization(self, agent):
        """Test that agent initializes correctly."""
        assert agent.policy is not None
        assert agent.state.policy_params is not None
        assert agent.state.opt_state is not None
        assert agent.gamma == 0.99
        assert agent.learning_rate == 1e-3
        
        assert len(agent.state.episode_observations) == 0
        assert len(agent.state.episode_actions) == 0
        assert len(agent.state.episode_rewards) == 0
    
    def test_return_computation(self):
        """Test discounted return calculation."""
        env = CartPoleEnv()
        policy = ComposedPolicy(
            backbone=MLPBackbone(hidden_dims=[32], output_dim=16),
            head=DiscreteHead(input_dim=16)
        )
        
        agent = REINFORCEAgent(
            policy=policy,
            observation_space=env.observation_space,
            action_space=env.action_space,
            gamma=0.9  # Use 0.9 for easy manual verification
        )
        
        test_state = agent.state._replace(episode_rewards=[1.0, 1.0, 1.0])
        returns = agent._compute_returns(test_state)
        
        # Manual calculation:
        # G_2 = 1.0
        # G_1 = 1.0 + 0.9 * 1.0 = 1.9
        # G_0 = 1.0 + 0.9 * 1.9 = 2.71
        expected = jnp.array([2.71, 1.9, 1.0])
        assert jnp.allclose(returns, expected, atol=1e-6)
        
        test_state = agent.state._replace(episode_rewards=[1.0, 2.0, 3.0])
        returns = agent._compute_returns(test_state)
        
        # G_2 = 3.0
        # G_1 = 2.0 + 0.9 * 3.0 = 4.7
        # G_0 = 1.0 + 0.9 * 4.7 = 5.23
        expected = jnp.array([5.23, 4.7, 3.0])
        assert jnp.allclose(returns, expected, atol=1e-6)
    
    def test_episode_buffer_management(self, agent):
        """Test that episode buffers are managed correctly."""
        env = CartPoleEnv()
        obs = env.reset()
        key = jax.random.PRNGKey(0)
        
        key, action_key = jax.random.split(key)
        action, new_state = agent.select_action(agent.state, obs, action_key)
        
        assert len(new_state.episode_observations) == 1
        assert len(new_state.episode_actions) == 1
        assert len(new_state.episode_rewards) == 0
        
        new_state = agent.update(new_state, obs, action, 1.0, obs, done=False, key=key)
        
        assert len(new_state.episode_rewards) == 1
        assert len(new_state.episode_observations) == 1
        
        final_state = agent.update(new_state, obs, action, 1.0, obs, done=True, key=key)
        
        assert len(final_state.episode_observations) == 0
        assert len(final_state.episode_actions) == 0
        assert len(final_state.episode_rewards) == 0
    
    def test_update_only_at_episode_end(self, agent):
        """Test that policy update only happens when episode ends."""
        env = CartPoleEnv()
        obs = env.reset()
        key = jax.random.PRNGKey(0)
        
        initial_params = jax.tree.map(lambda x: x.copy(), agent.state.policy_params)
        
        key, action_key = jax.random.split(key)
        action, new_state = agent.select_action(agent.state, obs, action_key)
        new_state = agent.update(new_state, obs, action, 1.0, obs, done=False, key=key)
        
        params_unchanged = jax.tree.map(
            lambda x, y: jnp.allclose(x, y),
            initial_params, new_state.policy_params
        )
        assert jax.tree_util.tree_all(params_unchanged)
        
        final_state = agent.update(new_state, obs, action, 1.0, obs, done=True, key=key)
        
        assert len(final_state.episode_observations) == 0
    
    def test_constant_returns_normalization(self, agent):
        """Test normalization when all returns are the same (std = 0)."""
        test_state = agent.state._replace(
            episode_rewards=[1.0, 1.0, 1.0],
            episode_observations=[jnp.array([0.1, 0.2, 0.3, 0.4]) for _ in range(3)],
            episode_actions=[jnp.array([0]) for _ in range(3)],
            baseline=0.0
        )
        
        updated_params, updated_opt_state, updated_baseline = agent._update_policy(test_state)
        
        assert updated_params is not None
        assert updated_opt_state is not None