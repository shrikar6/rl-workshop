"""
Tests for REINFORCE agent implementation.

Focus on testing the mechanics and correctness of the implementation,
not learning performance (which is tested separately in experiments).
"""

import jax
import jax.numpy as jnp
import pytest
from framework.agents.reinforce import REINFORCEAgent
from framework.networks.policy.composed import ComposedPolicyNetwork
from framework.networks.backbones.mlp import MLPBackbone
from framework.networks.policy.heads.discrete import DiscretePolicyHead
from framework.environments.cartpole import CartPoleEnv


class TestREINFORCEAgent:
    """Test suite for REINFORCE agent mechanics."""
    
    @pytest.fixture
    def agent(self):
        """Create a REINFORCE agent for testing."""
        env = CartPoleEnv()
        policy = ComposedPolicyNetwork(
            backbone=MLPBackbone(hidden_dims=[32], output_dim=16),
            head=DiscretePolicyHead(input_dim=16)
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
        assert agent.state.baseline == 0.0
        assert agent.baseline_alpha == 0.01
    
    def test_baseline_and_advantages_computation(self):
        """Test the JIT-compiled baseline and advantages computation."""
        env = CartPoleEnv()
        policy = ComposedPolicyNetwork(
            backbone=MLPBackbone(hidden_dims=[32], output_dim=16),
            head=DiscretePolicyHead(input_dim=16)
        )

        agent = REINFORCEAgent(
            policy=policy,
            observation_space=env.observation_space,
            action_space=env.action_space,
            gamma=0.9,
            baseline_alpha=0.1
        )

        # Test simple single-step case
        rewards = jnp.array([2.0])
        old_baseline = 1.0
        
        updated_baseline, advantages = agent._compute_baseline_and_advantages_jit(
            rewards, 0.9, old_baseline, 0.1
        )
        
        # Expected: return = 2.0, new_baseline = 0.9*1.0 + 0.1*2.0 = 1.1
        assert jnp.isclose(updated_baseline, 1.1, atol=1e-6)
        assert advantages.shape == (1,)
        
        # Test multi-step case  
        rewards = jnp.array([1.0, 2.0])
        old_baseline = 0.0
        
        updated_baseline, advantages = agent._compute_baseline_and_advantages_jit(
            rewards, 0.9, old_baseline, 0.1
        )
        
        # Expected returns: [1.0 + 0.9*2.0, 2.0] = [2.8, 2.0]
        # Expected baseline update: 0.9*0.0 + 0.1*2.8 = 0.28
        assert jnp.isclose(updated_baseline, 0.28, atol=1e-6)
        assert advantages.shape == (2,)
        
        # Verify types and finite values
        assert jnp.isfinite(updated_baseline)
        assert jnp.all(jnp.isfinite(advantages))
    
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
        
        new_state, _ = agent.update(new_state, obs, action, 1.0, obs, done=False, key=key)
        
        assert len(new_state.episode_rewards) == 1
        assert len(new_state.episode_observations) == 1
        
        final_state, _ = agent.update(new_state, obs, action, 1.0, obs, done=True, key=key)
        
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
        new_state, _ = agent.update(new_state, obs, action, 1.0, obs, done=False, key=key)
        
        params_unchanged = jax.tree.map(
            lambda x, y: jnp.allclose(x, y),
            initial_params, new_state.policy_params
        )
        assert jax.tree_util.tree_all(params_unchanged)
        
        final_state, _ = agent.update(new_state, obs, action, 1.0, obs, done=True, key=key)
        
        assert len(final_state.episode_observations) == 0
    
    def test_constant_returns_normalization(self, agent):
        """Test normalization when all returns are the same (std = 0)."""
        test_state = agent.state._replace(
            episode_rewards=[1.0, 1.0, 1.0],
            episode_observations=[jnp.array([0.1, 0.2, 0.3, 0.4]) for _ in range(3)],
            episode_actions=[jnp.array([0]) for _ in range(3)],
            baseline=0.0
        )
        
        updated_params, updated_opt_state, updated_baseline, _ = agent._update_policy(test_state)
        
        assert updated_params is not None
        assert updated_opt_state is not None
    
    def test_baseline_initialization(self, agent):
        """Test baseline starts at zero."""
        assert agent.state.baseline == 0.0
        assert agent.baseline_alpha == 0.01
    
    def test_baseline_update(self, agent):
        """Test baseline exponential moving average update."""
        # Create test state with known rewards
        test_state = agent.state._replace(
            episode_rewards=[1.0, 2.0, 3.0],
            episode_observations=[jnp.array([0.1, 0.2, 0.3, 0.4]) for _ in range(3)],
            episode_actions=[jnp.array([0]) for _ in range(3)],
            baseline=5.0
        )
        
        # Update policy (includes baseline update)
        _, _, updated_baseline, _ = agent._update_policy(test_state)
        
        # Calculate expected baseline update
        # returns[0] with rewards [1,2,3] and gamma=0.99
        expected_episode_return = 1.0 + 0.99 * 2.0 + 0.99**2 * 3.0
        expected_baseline = (1 - 0.01) * 5.0 + 0.01 * expected_episode_return
        
        assert jnp.isclose(updated_baseline, expected_baseline, atol=1e-6)
    
    def test_policy_parameters_change_after_update(self, agent):
        """Test that policy parameters actually change after an update."""
        # Set up a complete episode
        test_state = agent.state._replace(
            episode_rewards=[2.0, 1.0],
            episode_observations=[jnp.array([0.1, 0.2, 0.3, 0.4]) for _ in range(2)],
            episode_actions=[jnp.array([0]) for _ in range(2)],
            baseline=1.0
        )
        
        # Run policy update
        new_params, _, new_baseline, metrics = agent._update_policy(test_state)
        
        # Verify policy parameters changed (learning occurred)
        assert test_state.policy_params is not new_params
        
        # Verify baseline was updated
        assert new_baseline != 1.0
        
        # Verify metrics were computed
        expected_metrics = {"policy_loss", "baseline", "mean_advantage", "grad_norm"}
        assert all(metric in metrics for metric in expected_metrics)
    
    def test_baseline_with_different_alpha(self):
        """Test baseline update with different alpha values."""
        env = CartPoleEnv()
        policy = ComposedPolicyNetwork(
            backbone=MLPBackbone(hidden_dims=[32], output_dim=16),
            head=DiscretePolicyHead(input_dim=16)
        )
        
        # Create agent with higher alpha for faster updates
        agent = REINFORCEAgent(
            policy=policy,
            observation_space=env.observation_space,
            action_space=env.action_space,
            baseline_alpha=0.5,
            seed=42
        )
        
        test_state = agent.state._replace(
            episode_rewards=[10.0],
            episode_observations=[jnp.array([0.1, 0.2, 0.3, 0.4])],
            episode_actions=[jnp.array([0])],
            baseline=0.0
        )
        
        _, _, updated_baseline, _ = agent._update_policy(test_state)
        
        # With alpha=0.5 and episode_return=10.0, baseline should be 5.0
        expected_baseline = 0.5 * 10.0
        assert jnp.isclose(updated_baseline, expected_baseline, atol=1e-6)