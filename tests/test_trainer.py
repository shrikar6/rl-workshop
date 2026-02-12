"""
Tests for Trainer class.
"""

import tempfile
import jax
import jax.numpy as jnp
from typing import NamedTuple
from jax import Array
from framework import Trainer, Tracker
from framework.environments.base import EnvironmentABC
from framework.agents.base import AgentABC


class MockEnvironment(EnvironmentABC):
    """Mock environment for testing."""

    def __init__(self, episode_length: int = 5, reward_per_step: float = 1.0):
        self._episode_length = episode_length
        self.reward_per_step = reward_per_step
        self.step_count = 0
        self.total_steps = 0
        self.episode_count = 0

    def reset(self):
        self.step_count = 0
        self.episode_count += 1
        return jnp.array([0.0, 0.0, 0.0, 0.0])

    def step(self, action):
        self.step_count += 1
        self.total_steps += 1
        obs = jnp.array([0.1, 0.2, 0.3, 0.4])
        reward = self.reward_per_step
        done = self.step_count >= self._episode_length
        return obs, reward, done

    @property
    def observation_space(self):
        import gymnasium as gym
        return gym.spaces.Box(-1, 1, shape=(4,))

    @property
    def action_space(self):
        import gymnasium as gym
        return gym.spaces.Discrete(2)

    @property
    def max_episode_length(self):
        return self._episode_length

    def render(self):
        return None

    def close(self):
        pass


class MockState(NamedTuple):
    """JIT-compatible mock agent state."""
    dummy: Array


class MockAgent(AgentABC):
    """JIT-compatible mock agent for testing the Trainer's training loop."""

    def __init__(self, fixed_action: int = 0):
        self.fixed_action = fixed_action

    def init_state(self, key):
        return MockState(dummy=jnp.array(0.0))

    def select_action(self, state, observation, key):
        return jnp.array([self.fixed_action]), state

    def update(self, state, obs, action, reward, next_obs, done, key):
        return state, {}


class TestTrainer:
    """Tests for Trainer class."""

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        env = MockEnvironment()
        agent = MockAgent()
        trainer = Trainer(env, agent)

        assert trainer.env == env
        assert trainer.agent == agent
        assert trainer.tracker is None

    def test_trainer_initialization_with_tracker(self):
        """Test trainer initialization with tracker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = MockEnvironment()
            agent = MockAgent()
            tracker = Tracker(results_dir=tmpdir)
            trainer = Trainer(env, agent, tracker=tracker)

            assert trainer.tracker == tracker

    def test_train_episode(self):
        """Test single episode training."""
        env = MockEnvironment(episode_length=3, reward_per_step=1.0)
        agent = MockAgent(fixed_action=1)
        trainer = Trainer(env, agent)

        key = jax.random.PRNGKey(0)
        state = agent.init_state(key)
        trainer_key = jax.random.PRNGKey(42)

        state, trainer_key, episode_metrics = trainer.train_episode(
            state, trainer_key
        )

        # Check episode reward
        assert episode_metrics["return"] == 3.0  # 3 steps * 1.0 reward per step

        # Check correct number of environment steps
        assert env.step_count == 3

    def test_train_multiple_episodes(self):
        """Test training multiple episodes."""
        env = MockEnvironment(episode_length=2, reward_per_step=2.0)
        agent = MockAgent()
        trainer = Trainer(env, agent)

        key = jax.random.PRNGKey(0)
        state = agent.init_state(key)
        trainer_key = jax.random.PRNGKey(42)

        trainer.train(state, trainer_key, num_episodes=3)

        # Should have 3 episodes * 2 steps each = 6 total steps
        assert env.total_steps == 6
        assert env.episode_count == 3

    def test_train_with_tracker(self, capsys):
        """Test training with tracker integration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = MockEnvironment(episode_length=2, reward_per_step=1.5)
            agent = MockAgent()
            tracker = Tracker(log_interval=1, results_dir=tmpdir)
            trainer = Trainer(env, agent, tracker=tracker)

            key = jax.random.PRNGKey(0)
            state = agent.init_state(key)
            trainer_key = jax.random.PRNGKey(42)

            trainer.train(state, trainer_key, num_episodes=2)

            # Check tracker received episode data
            returns = tracker.get_metric("return")
            assert len(returns) == 2
            assert returns[0] == 3.0  # 2 steps * 1.5 reward
            assert returns[1] == 3.0

            # Check logging output
            captured = capsys.readouterr()
            assert "Episode    1" in captured.out
            assert "Episode    2" in captured.out

    def test_train_without_tracker_no_logging(self, capsys):
        """Test that training without tracker produces no log output."""
        env = MockEnvironment(episode_length=1)
        agent = MockAgent()
        trainer = Trainer(env, agent)

        key = jax.random.PRNGKey(0)
        state = agent.init_state(key)
        trainer_key = jax.random.PRNGKey(42)

        trainer.train(state, trainer_key, num_episodes=2)

        # Should produce no output
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same behavior."""
        def create_trainer_and_run():
            env = MockEnvironment(episode_length=2)
            agent = MockAgent()
            trainer = Trainer(env, agent)
            key = jax.random.PRNGKey(0)
            state = agent.init_state(key)
            trainer_key = jax.random.PRNGKey(999)
            state, trainer_key, episode_metrics = trainer.train_episode(
                state, trainer_key
            )
            return episode_metrics, trainer_key

        metrics1, key1 = create_trainer_and_run()
        metrics2, key2 = create_trainer_and_run()

        assert metrics1["return"] == metrics2["return"]
        assert jnp.array_equal(key1, key2)

    def test_different_seeds_different_keys(self):
        """Test that different seeds produce different key evolution."""
        def run_episode_with_seed(seed):
            env = MockEnvironment()
            agent = MockAgent()
            trainer = Trainer(env, agent)
            key = jax.random.PRNGKey(0)
            state = agent.init_state(key)
            trainer_key = jax.random.PRNGKey(seed)
            _, new_trainer_key, _ = trainer.train_episode(state, trainer_key)
            return new_trainer_key

        key1 = run_episode_with_seed(1)
        key2 = run_episode_with_seed(2)

        assert not jnp.array_equal(key1, key2)

    def test_key_management(self):
        """Test proper JAX key splitting and management."""
        env = MockEnvironment(episode_length=2)
        agent = MockAgent()
        trainer = Trainer(env, agent)

        key = jax.random.PRNGKey(0)
        state = agent.init_state(key)

        initial_key = jax.random.PRNGKey(42)
        state, new_trainer_key, episode_metrics = trainer.train_episode(
            state, initial_key
        )

        # Trainer key should evolve after episode
        assert not jnp.array_equal(new_trainer_key, initial_key)
