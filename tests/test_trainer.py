"""
Tests for Trainer class.
"""

import tempfile
import jax
import jax.numpy as jnp
from unittest.mock import Mock
from framework import Trainer, Tracker
from framework.environments.base import EnvironmentABC
from framework.agents.base import AgentABC


class MockEnvironment(EnvironmentABC):
    """Mock environment for testing."""

    def __init__(self, episode_length: int = 5, reward_per_step: float = 1.0):
        self.episode_length = episode_length
        self.reward_per_step = reward_per_step
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return jnp.array([0.0, 0.0, 0.0, 0.0])  # CartPole-like observation

    def step(self, action):
        self.step_count += 1
        obs = jnp.array([0.1, 0.2, 0.3, 0.4])  # Mock next observation
        reward = self.reward_per_step
        done = self.step_count >= self.episode_length
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
        """Mock max episode length."""
        return self.episode_length

    def render(self):
        """Mock render method."""
        return None

    def close(self):
        """Mock close method."""
        pass


class MockAgent(AgentABC):
    """Mock agent for testing."""

    def __init__(self, fixed_action: int = 0):
        self.fixed_action = fixed_action
        self.select_action_calls = []
        self.update_calls = []

    def init_state(self, key):
        """Create initial mock state."""
        return {"mock_params": "test"}

    def select_action(self, state, observation, key):
        self.select_action_calls.append((state, observation, key))
        return jnp.array([self.fixed_action]), state

    def update(self, state, obs, action, reward, next_obs, done, key):
        self.update_calls.append((state, obs, action, reward, next_obs, done, key))
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

        # Check agent interactions
        assert len(agent.select_action_calls) == 3
        assert len(agent.update_calls) == 3

        # Check final update had done=True
        final_update = agent.update_calls[-1]
        assert final_update[5] is True
        
    def test_train_multiple_episodes(self):
        """Test training multiple episodes."""
        env = MockEnvironment(episode_length=2, reward_per_step=2.0)
        agent = MockAgent()
        trainer = Trainer(env, agent)

        key = jax.random.PRNGKey(0)
        state = agent.init_state(key)
        trainer_key = jax.random.PRNGKey(42)

        trainer.train(state, trainer_key, num_episodes=3)

        # Should have 3 episodes * 2 steps each = 6 total interactions
        assert len(agent.select_action_calls) == 6
        assert len(agent.update_calls) == 6
        
    def test_train_with_tracker(self, capsys):
        """Test training with tracker integration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = MockEnvironment(episode_length=2, reward_per_step=1.5)
            agent = MockAgent()
            tracker = Tracker(log_interval=1, results_dir=tmpdir)  # Log every episode
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
            return episode_metrics

        metrics1 = create_trainer_and_run()
        metrics2 = create_trainer_and_run()

        assert metrics1["return"] == metrics2["return"]
        
    def test_different_seeds_different_keys(self):
        """Test that different seeds produce different key sequences."""
        env = MockEnvironment()
        agent1 = MockAgent()
        agent2 = MockAgent()

        trainer1 = Trainer(env, agent1)
        trainer2 = Trainer(env, agent2)

        key1 = jax.random.PRNGKey(0)
        state1 = agent1.init_state(key1)
        key2 = jax.random.PRNGKey(0)
        state2 = agent2.init_state(key2)

        trainer_key1 = jax.random.PRNGKey(1)
        trainer_key2 = jax.random.PRNGKey(2)

        trainer1.train_episode(state1, trainer_key1)
        trainer2.train_episode(state2, trainer_key2)

        keys1 = [call[2] for call in agent1.select_action_calls]
        keys2 = [call[2] for call in agent2.select_action_calls]

        assert any(not jnp.array_equal(k1, k2) for k1, k2 in zip(keys1, keys2))
        
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

        assert not jnp.array_equal(new_trainer_key, initial_key)

        action_keys = [call[2] for call in agent.select_action_calls]
        update_keys = [call[6] for call in agent.update_calls]

        all_keys = action_keys + update_keys
        for i, key1 in enumerate(all_keys):
            for j, key2 in enumerate(all_keys):
                if i != j:
                    assert not jnp.array_equal(key1, key2)