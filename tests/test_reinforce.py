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
            max_episode_length=env.max_episode_length,
            learning_rate=1e-3,
            gamma=0.99
        )

    @pytest.fixture
    def state(self, agent):
        """Create initial agent state for testing."""
        key = jax.random.PRNGKey(0)
        return agent.init_state(key)
    
    def test_initialization(self, agent, state):
        """Test that agent initializes correctly."""
        assert agent.policy is not None
        assert state.policy_params is not None
        assert state.opt_state is not None
        assert agent.gamma == 0.99
        assert agent.learning_rate == 1e-3

        # Check episode buffers are pre-allocated with correct shapes
        assert state.episode_length == 0
        assert state.episode_observations.shape == (agent.max_episode_length, 4)  # CartPole obs dim
        assert state.episode_actions.shape == (agent.max_episode_length, 1)
        assert state.episode_rewards.shape == (agent.max_episode_length,)
        assert state.baseline == 0.0
        assert agent.baseline_alpha == 0.01
    
    def test_baseline_and_advantages_computation(self):
        """Test baseline and advantages computation with pre-allocated buffers."""
        env = CartPoleEnv()
        policy = ComposedPolicyNetwork(
            backbone=MLPBackbone(hidden_dims=[32], output_dim=16),
            head=DiscretePolicyHead(input_dim=16)
        )

        agent = REINFORCEAgent(
            policy=policy,
            observation_space=env.observation_space,
            action_space=env.action_space,
            max_episode_length=env.max_episode_length,
            gamma=0.9,
            baseline_alpha=0.1
        )

        max_len = env.max_episode_length

        # Test simple single-step case
        episode_length = 1
        rewards = jnp.zeros(max_len).at[0].set(2.0)
        mask = jnp.arange(max_len) < episode_length
        old_baseline = 1.0

        updated_baseline, advantages = agent.compute_baseline_and_advantages(
            rewards, 0.9, old_baseline, 0.1, mask, episode_length
        )

        # Expected: return = 2.0, new_baseline = 0.9*1.0 + 0.1*2.0 = 1.1
        assert jnp.isclose(updated_baseline, 1.1, atol=1e-6)
        # Advantage = return - old_baseline = 2.0 - 1.0 = 1.0
        assert jnp.isclose(advantages[0], 1.0, atol=1e-6)
        # Padding positions should be masked to 0
        assert jnp.all(advantages[1:] == 0.0)

        # Test multi-step case
        episode_length = 2
        rewards = jnp.zeros(max_len).at[0].set(1.0).at[1].set(2.0)
        mask = jnp.arange(max_len) < episode_length
        old_baseline = 0.0

        updated_baseline, advantages = agent.compute_baseline_and_advantages(
            rewards, 0.9, old_baseline, 0.1, mask, episode_length
        )

        # Expected returns: [1.0 + 0.9*2.0, 2.0] = [2.8, 2.0]
        # Expected baseline update: 0.9*0.0 + 0.1*2.8 = 0.28
        assert jnp.isclose(updated_baseline, 0.28, atol=1e-6)
        # Advantages = returns - old_baseline(0.0) = returns
        assert jnp.isclose(advantages[0], 2.8, atol=1e-6)
        assert jnp.isclose(advantages[1], 2.0, atol=1e-6)
        # Padding should be 0
        assert jnp.all(advantages[2:] == 0.0)

        # Verify finite values
        assert jnp.isfinite(updated_baseline)
        assert jnp.all(jnp.isfinite(advantages))
    
    def test_episode_buffer_management(self, agent, state):
        """Test that episode buffers are managed correctly."""
        env = CartPoleEnv()
        obs = env.reset()
        key = jax.random.PRNGKey(0)

        key, action_key = jax.random.split(key)
        action, new_state = agent.select_action(state, obs, action_key)

        # After select_action: counter incremented and data actually stored
        assert new_state.episode_length == 1
        assert jnp.allclose(new_state.episode_observations[0], obs)
        assert jnp.array_equal(new_state.episode_actions[0], action)

        new_state, _ = agent.update(new_state, obs, action, 1.0, obs, done=False, key=key)

        # After update (not done): reward stored, counter unchanged
        assert new_state.episode_length == 1
        assert new_state.episode_rewards[0] == 1.0

        final_state, _ = agent.update(new_state, obs, action, 1.0, obs, done=True, key=key)

        # After done, buffers should be reset (episode_length back to 0)
        assert final_state.episode_length == 0
    
    def test_update_only_at_episode_end(self, agent, state):
        """Test that policy update only happens when episode ends."""
        env = CartPoleEnv()
        obs = env.reset()
        key = jax.random.PRNGKey(0)

        initial_params = jax.tree.map(lambda x: x.copy(), state.policy_params)

        key, action_key = jax.random.split(key)
        action, new_state = agent.select_action(state, obs, action_key)
        new_state, _ = agent.update(new_state, obs, action, 1.0, obs, done=False, key=key)

        # Params should be unchanged after non-done update
        params_unchanged = jax.tree.map(
            lambda x, y: jnp.allclose(x, y),
            initial_params, new_state.policy_params
        )
        assert jax.tree_util.tree_all(params_unchanged)

        final_state, _ = agent.update(new_state, obs, action, 1.0, obs, done=True, key=key)

        # After episode ends: buffers reset AND params actually updated
        assert final_state.episode_length == 0
        params_same = jax.tree.map(
            lambda x, y: jnp.allclose(x, y),
            initial_params, final_state.policy_params
        )
        assert not jax.tree_util.tree_all(params_same)
    
    def test_constant_returns_normalization(self):
        """Test normalization with near-zero advantage std doesn't produce NaN/Inf."""
        env = CartPoleEnv()
        policy = ComposedPolicyNetwork(
            backbone=MLPBackbone(hidden_dims=[32], output_dim=16),
            head=DiscretePolicyHead(input_dim=16)
        )

        # Must enable normalization to exercise the normalization code path
        agent = REINFORCEAgent(
            policy=policy,
            observation_space=env.observation_space,
            action_space=env.action_space,
            max_episode_length=env.max_episode_length,
            learning_rate=1e-3,
            gamma=0.99,
            normalize_advantages=True
        )

        key = jax.random.PRNGKey(0)
        state = agent.init_state(key)

        # Single step with baseline equal to the return → advantage = 0, std = 0
        # This is the dangerous case: normalization divides by std, which is 0
        episode_length = 1
        test_state = state._replace(
            episode_rewards=state.episode_rewards.at[0].set(5.0),
            episode_observations=state.episode_observations.at[0].set(
                jnp.array([0.1, 0.2, 0.3, 0.4])
            ),
            episode_actions=state.episode_actions.at[0].set(jnp.array([0])),
            episode_length=episode_length,
            baseline=5.0  # Equal to the single-step return
        )

        updated_params, _, updated_baseline, metrics = agent._update_policy(test_state)

        # Core assertion: no NaN/Inf anywhere in the results
        assert jnp.isfinite(metrics["policy_loss"])
        assert jnp.isfinite(metrics["grad_norm"])
        assert jnp.isfinite(updated_baseline)
        params_finite = jax.tree.map(lambda x: jnp.all(jnp.isfinite(x)), updated_params)
        assert jax.tree_util.tree_all(params_finite)
    
    def test_baseline_initialization(self, agent, state):
        """Test baseline starts at zero."""
        assert state.baseline == 0.0
        assert agent.baseline_alpha == 0.01
    
    def test_baseline_update(self, agent, state):
        """Test baseline exponential moving average update."""
        # Create test state with known rewards
        episode_length = 3
        test_state = state._replace(
            episode_rewards=state.episode_rewards.at[:episode_length].set(jnp.array([1.0, 2.0, 3.0])),
            episode_observations=state.episode_observations.at[:episode_length].set(
                jnp.array([[0.1, 0.2, 0.3, 0.4] for _ in range(episode_length)])
            ),
            episode_actions=state.episode_actions.at[:episode_length].set(
                jnp.array([[0] for _ in range(episode_length)])
            ),
            episode_length=episode_length,
            baseline=5.0
        )

        # Update policy (includes baseline update)
        _, _, updated_baseline, _ = agent._update_policy(test_state)

        # Calculate expected baseline update
        # returns[0] with rewards [1,2,3] and gamma=0.99
        expected_episode_return = 1.0 + 0.99 * 2.0 + 0.99**2 * 3.0
        expected_baseline = (1 - 0.01) * 5.0 + 0.01 * expected_episode_return

        assert jnp.isclose(updated_baseline, expected_baseline, atol=1e-6)
    
    def test_policy_parameters_change_after_update(self, agent, state):
        """Test that policy parameters actually change after an update."""
        # Set up a complete episode
        episode_length = 2
        test_state = state._replace(
            episode_rewards=state.episode_rewards.at[:episode_length].set(jnp.array([2.0, 1.0])),
            episode_observations=state.episode_observations.at[:episode_length].set(
                jnp.array([[0.1, 0.2, 0.3, 0.4] for _ in range(episode_length)])
            ),
            episode_actions=state.episode_actions.at[:episode_length].set(
                jnp.array([[0] for _ in range(episode_length)])
            ),
            episode_length=episode_length,
            baseline=1.0
        )

        # Run policy update
        new_params, _, new_baseline, metrics = agent._update_policy(test_state)

        # Verify policy parameters actually changed in value (not just identity)
        params_same = jax.tree.map(
            lambda x, y: jnp.allclose(x, y),
            test_state.policy_params, new_params
        )
        assert not jax.tree_util.tree_all(params_same)

        # Verify baseline was updated
        assert new_baseline != 1.0

        # Verify metrics are present and sensible
        expected_metrics = {"policy_loss", "baseline", "mean_advantage", "grad_norm"}
        assert all(metric in metrics for metric in expected_metrics)
        assert jnp.isfinite(metrics["policy_loss"])
        assert jnp.isfinite(metrics["grad_norm"])
        assert metrics["grad_norm"] > 0
    
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
            max_episode_length=env.max_episode_length,
            baseline_alpha=0.5
        )

        key = jax.random.PRNGKey(42)
        state = agent.init_state(key)

        episode_length = 1
        test_state = state._replace(
            episode_rewards=state.episode_rewards.at[:episode_length].set(jnp.array([10.0])),
            episode_observations=state.episode_observations.at[:episode_length].set(
                jnp.array([[0.1, 0.2, 0.3, 0.4]])
            ),
            episode_actions=state.episode_actions.at[:episode_length].set(jnp.array([[0]])),
            episode_length=episode_length,
            baseline=0.0
        )

        _, _, updated_baseline, _ = agent._update_policy(test_state)

        # With alpha=0.5 and episode_return=10.0, baseline should be 5.0
        expected_baseline = 0.5 * 10.0
        assert jnp.isclose(updated_baseline, expected_baseline, atol=1e-6)