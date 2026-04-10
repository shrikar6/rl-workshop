"""
Tests for A2C agent implementation.

Focus on testing the mechanics and correctness of the implementation,
not learning performance (which is tested separately in experiments).
"""

import jax
import jax.numpy as jnp
import pytest
from framework.agents.a2c import A2CAgent
from framework.networks.policy.composed import ComposedPolicyNetwork
from framework.networks.value.composed import ComposedValueNetwork
from framework.networks.backbones.mlp import MLPBackbone
from framework.networks.policy.heads.discrete import DiscretePolicyHead
from framework.networks.value.heads.scalar import ScalarValueHead
from framework.environments.cartpole import CartPoleEnv


def _build_policy():
    return ComposedPolicyNetwork(
        backbone=MLPBackbone(hidden_dims=[32], output_dim=16),
        head=DiscretePolicyHead(input_dim=16),
    )


def _build_value_network():
    return ComposedValueNetwork(
        backbone=MLPBackbone(hidden_dims=[32], output_dim=16),
        head=ScalarValueHead(input_dim=16),
    )


class TestA2CAgent:
    """Test suite for A2C agent mechanics."""

    @pytest.fixture
    def agent(self):
        """Create an A2C agent for testing."""
        env = CartPoleEnv()
        return A2CAgent(
            policy=_build_policy(),
            value_network=_build_value_network(),
            observation_space=env.observation_space,
            action_space=env.action_space,
            max_episode_length=env.max_episode_length,
            policy_lr=3e-4,
            value_lr=1e-3,
            gamma=0.99,
            gae_lambda=0.95,
        )

    @pytest.fixture
    def state(self, agent):
        """Create initial agent state for testing."""
        return agent.init_state(jax.random.PRNGKey(0))

    def test_initialization(self, agent, state):
        """Test that agent initializes correctly."""
        assert agent.policy is not None
        assert agent.value_network is not None
        assert state.policy_params is not None
        assert state.value_params is not None
        assert state.policy_opt_state is not None
        assert state.value_opt_state is not None
        assert agent.gamma == 0.99
        assert agent.gae_lambda == 0.95

        # Check episode buffers are pre-allocated with correct shapes
        assert state.episode_length == 0
        assert state.episode_observations.shape == (agent.max_episode_length, 4)
        assert state.episode_actions.shape == (agent.max_episode_length, 1)
        assert state.episode_rewards.shape == (agent.max_episode_length,)

    def test_hyperparameter_validation(self):
        """Test that invalid hyperparameters raise ValueError."""
        env = CartPoleEnv()
        common = dict(
            policy=_build_policy(),
            value_network=_build_value_network(),
            observation_space=env.observation_space,
            action_space=env.action_space,
            max_episode_length=env.max_episode_length,
        )

        with pytest.raises(ValueError, match="gamma"):
            A2CAgent(**common, gamma=1.5)
        with pytest.raises(ValueError, match="gae_lambda"):
            A2CAgent(**common, gae_lambda=-0.1)
        with pytest.raises(ValueError, match="policy_lr"):
            A2CAgent(**common, policy_lr=0.0)
        with pytest.raises(ValueError, match="value_lr"):
            A2CAgent(**common, value_lr=-1e-3)
        with pytest.raises(ValueError, match="max_episode_length"):
            A2CAgent(
                policy=_build_policy(),
                value_network=_build_value_network(),
                observation_space=env.observation_space,
                action_space=env.action_space,
                max_episode_length=0,
            )

    def test_gae_lambda_zero_is_pure_td(self, agent):
        """At lambda=0, GAE should equal single-step TD residuals."""
        max_len = agent.max_episode_length
        episode_length = 4

        rewards = jnp.zeros(max_len).at[:episode_length].set(
            jnp.array([1.0, 2.0, 3.0, 4.0])
        )
        values = jnp.zeros(max_len).at[:episode_length].set(
            jnp.array([0.5, 1.5, 2.5, 3.5])
        )
        mask = jnp.arange(max_len) < episode_length

        advantages, _ = agent.compute_gae(
            rewards, values, gamma=0.9, gae_lambda=0.0,
            mask=mask, episode_length=episode_length,
        )

        # At lambda=0: A_t = delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
        # delta_0 = 1 + 0.9*1.5 - 0.5 = 1.85
        # delta_1 = 2 + 0.9*2.5 - 1.5 = 2.75
        # delta_2 = 3 + 0.9*3.5 - 2.5 = 3.65
        # delta_3 (last) = 4 + 0.9*0 - 3.5 = 0.5  (bootstrap from 0 at terminal)
        expected = jnp.array([1.85, 2.75, 3.65, 0.5])
        assert jnp.allclose(advantages[:episode_length], expected, atol=1e-5)
        assert jnp.all(advantages[episode_length:] == 0.0)

    def test_gae_lambda_one_is_monte_carlo(self, agent):
        """At lambda=1, GAE should equal G_t - V(s_t) (Monte Carlo advantage)."""
        max_len = agent.max_episode_length
        episode_length = 3

        rewards = jnp.zeros(max_len).at[:episode_length].set(
            jnp.array([1.0, 2.0, 3.0])
        )
        values = jnp.zeros(max_len).at[:episode_length].set(
            jnp.array([0.5, 1.0, 1.5])
        )
        mask = jnp.arange(max_len) < episode_length

        advantages, returns = agent.compute_gae(
            rewards, values, gamma=0.9, gae_lambda=1.0,
            mask=mask, episode_length=episode_length,
        )

        # Monte Carlo returns: G_2=3, G_1=2+0.9*3=4.7, G_0=1+0.9*4.7=5.23
        # Advantages: A_t = G_t - V(s_t)
        expected_returns = jnp.array([5.23, 4.7, 3.0])
        expected_advantages = jnp.array([4.73, 3.7, 1.5])

        assert jnp.allclose(returns[:episode_length], expected_returns, atol=1e-5)
        assert jnp.allclose(advantages[:episode_length], expected_advantages, atol=1e-5)

    def test_gae_intermediate_lambda(self, agent):
        """Test GAE with intermediate lambda against hand-computed values."""
        max_len = agent.max_episode_length
        episode_length = 3

        # Simple setup: constant rewards, zero values, gamma=1, lambda=0.5
        rewards = jnp.zeros(max_len).at[:episode_length].set(jnp.array([1.0, 1.0, 1.0]))
        values = jnp.zeros(max_len)
        mask = jnp.arange(max_len) < episode_length

        advantages, _ = agent.compute_gae(
            rewards, values, gamma=1.0, gae_lambda=0.5,
            mask=mask, episode_length=episode_length,
        )

        # With values=0, gamma=1: delta_t = r_t = 1 for all valid t
        # Reverse accumulation: A_2=1, A_1=1+0.5*1=1.5, A_0=1+0.5*1.5=1.75
        expected = jnp.array([1.75, 1.5, 1.0])
        assert jnp.allclose(advantages[:episode_length], expected, atol=1e-5)

    def test_gae_returns_consistency(self, agent):
        """Returns should equal advantages + values at valid positions."""
        max_len = agent.max_episode_length
        episode_length = 3

        rewards = jnp.zeros(max_len).at[:episode_length].set(jnp.array([1.0, 2.0, 3.0]))
        values = jnp.zeros(max_len).at[:episode_length].set(jnp.array([0.5, 1.0, 1.5]))
        mask = jnp.arange(max_len) < episode_length

        advantages, returns = agent.compute_gae(
            rewards, values, gamma=0.9, gae_lambda=0.7,
            mask=mask, episode_length=episode_length,
        )

        # returns = advantages + values (definitional)
        assert jnp.allclose(
            returns[:episode_length],
            advantages[:episode_length] + values[:episode_length],
            atol=1e-6,
        )

    def test_gae_padding_positions_are_zero(self, agent):
        """Padding positions in advantages should be exactly 0."""
        max_len = agent.max_episode_length
        episode_length = 3

        rewards = jnp.zeros(max_len).at[:episode_length].set(jnp.array([1.0, 2.0, 3.0]))
        values = jnp.zeros(max_len).at[:episode_length].set(jnp.array([0.5, 1.0, 1.5]))
        mask = jnp.arange(max_len) < episode_length

        advantages, _ = agent.compute_gae(
            rewards, values, gamma=0.9, gae_lambda=0.95,
            mask=mask, episode_length=episode_length,
        )

        assert jnp.all(advantages[episode_length:] == 0.0)

    def test_episode_buffer_management(self, agent, state):
        """Test that episode buffers are managed correctly across transitions."""
        env = CartPoleEnv()
        obs = env.reset()
        key = jax.random.PRNGKey(0)

        key, ak = jax.random.split(key)
        action, s1 = agent.select_action(state, obs, ak)

        # After select_action: counter incremented and data stored
        assert s1.episode_length == 1
        assert jnp.allclose(s1.episode_observations[0], obs)
        assert jnp.array_equal(s1.episode_actions[0], action)

        # Update with done=False: reward stored, counter unchanged
        s2, _ = agent.update(s1, obs, action, 1.0, obs, done=False, key=key)
        assert s2.episode_length == 1
        assert s2.episode_rewards[0] == 1.0

        # Update with done=True: buffers reset (episode_length back to 0)
        s3, _ = agent.update(s2, obs, action, 1.0, obs, done=True, key=key)
        assert s3.episode_length == 0

    def test_update_only_at_episode_end(self, agent, state):
        """Test that network updates only happen when episode ends."""
        env = CartPoleEnv()
        obs = env.reset()
        key = jax.random.PRNGKey(0)

        initial_policy = jax.tree.map(lambda x: x.copy(), state.policy_params)
        initial_value = jax.tree.map(lambda x: x.copy(), state.value_params)

        # Take one step, update mid-episode
        key, ak = jax.random.split(key)
        action, s1 = agent.select_action(state, obs, ak)
        s2, _ = agent.update(s1, obs, action, 1.0, obs, done=False, key=key)

        # Both networks should be unchanged mid-episode
        policy_unchanged = jax.tree.map(
            lambda x, y: jnp.allclose(x, y), initial_policy, s2.policy_params
        )
        value_unchanged = jax.tree.map(
            lambda x, y: jnp.allclose(x, y), initial_value, s2.value_params
        )
        assert jax.tree_util.tree_all(policy_unchanged)
        assert jax.tree_util.tree_all(value_unchanged)

        # End episode: both networks should update
        s3, _ = agent.update(s2, obs, action, 1.0, obs, done=True, key=key)
        assert s3.episode_length == 0

        policy_same_as_initial = jax.tree.map(
            lambda x, y: jnp.allclose(x, y), initial_policy, s3.policy_params
        )
        value_same_as_initial = jax.tree.map(
            lambda x, y: jnp.allclose(x, y), initial_value, s3.value_params
        )
        assert not jax.tree_util.tree_all(policy_same_as_initial)
        assert not jax.tree_util.tree_all(value_same_as_initial)

    def test_both_networks_update(self, agent, state):
        """Test that both policy and value networks update during _update_networks."""
        episode_length = 3
        rewards = jnp.array([1.0, 2.0, 3.0])
        obs_array = jnp.tile(jnp.array([0.1, 0.2, 0.3, 0.4]), (episode_length, 1))
        actions = jnp.zeros((episode_length, 1))

        test_state = state._replace(
            episode_rewards=state.episode_rewards.at[:episode_length].set(rewards),
            episode_observations=state.episode_observations.at[:episode_length].set(obs_array),
            episode_actions=state.episode_actions.at[:episode_length].set(actions),
            episode_length=episode_length,
        )

        new_policy, new_value, _, _, _ = agent._update_networks(test_state)

        # Policy params should have changed
        policy_same = jax.tree.map(
            lambda x, y: jnp.allclose(x, y), test_state.policy_params, new_policy
        )
        assert not jax.tree_util.tree_all(policy_same)

        # Value params should have changed
        value_same = jax.tree.map(
            lambda x, y: jnp.allclose(x, y), test_state.value_params, new_value
        )
        assert not jax.tree_util.tree_all(value_same)

    def test_metrics_structure(self, agent, state):
        """Test that metrics dict has expected keys and all values are finite."""
        episode_length = 2
        rewards = jnp.array([1.0, 2.0])
        obs_array = jnp.tile(jnp.array([0.1, 0.2, 0.3, 0.4]), (episode_length, 1))
        actions = jnp.zeros((episode_length, 1))

        test_state = state._replace(
            episode_rewards=state.episode_rewards.at[:episode_length].set(rewards),
            episode_observations=state.episode_observations.at[:episode_length].set(obs_array),
            episode_actions=state.episode_actions.at[:episode_length].set(actions),
            episode_length=episode_length,
        )

        _, _, _, _, metrics = agent._update_networks(test_state)

        expected_keys = {
            "policy_loss", "value_loss", "mean_advantage",
            "policy_grad_norm", "value_grad_norm", "mean_value",
        }
        assert set(metrics.keys()) == expected_keys
        for k, v in metrics.items():
            assert jnp.isfinite(v), f"Metric {k} is not finite"

    def test_normalization_padding_invariance(self):
        """Advantage normalization should be invariant to buffer padding.

        Regression test: previously, the squared_diff computation did not mask
        padding positions, so (0 - mean_adv)^2 at padding would pollute variance
        and produce different normalizations for different max_episode_length values.
        With the fix, loss should be identical regardless of buffer size.
        """
        env = CartPoleEnv()

        def build_and_compute_loss(max_len):
            # Fresh components with same seed for deterministic init
            agent = A2CAgent(
                policy=_build_policy(),
                value_network=_build_value_network(),
                observation_space=env.observation_space,
                action_space=env.action_space,
                max_episode_length=max_len,
                policy_lr=3e-4,
                value_lr=1e-3,
                gamma=0.99,
                gae_lambda=0.95,
                normalize_advantages=True,
            )
            state = agent.init_state(jax.random.PRNGKey(0))

            # Identical episode data across both buffer sizes
            episode_length = 3
            rewards = jnp.array([3.0, 2.0, 1.0])
            obs_array = jnp.tile(jnp.array([0.1, 0.2, 0.3, 0.4]), (episode_length, 1))
            actions = jnp.zeros((episode_length, 1))

            test_state = state._replace(
                episode_rewards=state.episode_rewards.at[:episode_length].set(rewards),
                episode_observations=state.episode_observations.at[:episode_length].set(obs_array),
                episode_actions=state.episode_actions.at[:episode_length].set(actions),
                episode_length=episode_length,
            )
            _, _, _, _, metrics = agent._update_networks(test_state)
            return metrics["policy_loss"]

        loss_small = build_and_compute_loss(max_len=5)
        loss_large = build_and_compute_loss(max_len=50)

        assert jnp.isclose(loss_small, loss_large, atol=1e-5), (
            f"Policy loss should be invariant to buffer padding, "
            f"got {loss_small} (max_len=5) vs {loss_large} (max_len=50)"
        )
