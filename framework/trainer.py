import jax
from typing import Optional, Dict, Any, Tuple
from jax import Array
from .environments import EnvironmentABC
from .agents import AgentABC
from .tracking import Tracker


class Trainer:
    """
    Basic training loop for reinforcement learning.

    Handles the core episode loop: reset environment, run episode,
    update agent from experience. Random keys are passed functionally
    to ensure reproducible training. Optionally tracks training progress.
    """
    
    def __init__(
        self,
        environment: EnvironmentABC,
        agent: AgentABC,
        tracker: Optional[Tracker] = None
    ):
        """
        Initialize trainer with environment and agent.

        Args:
            environment: Environment to train in
            agent: Agent to train
            tracker: Optional tracker for logging training progress
        """
        self.env = environment
        self.agent = agent
        self.tracker = tracker

        # Create JIT-compiled versions of agent methods for performance
        # Compiled once here, reused throughout training
        self.select_action_jit = jax.jit(agent.select_action)
        self.update_jit = jax.jit(agent.update)
    
    def train_episode(self, state: Any, trainer_key: Array, record_video: bool = False) -> Tuple[Any, Array, Dict[str, float]]:
        """
        Run one complete episode and return new states and metrics.

        Args:
            state: Current agent state
            trainer_key: Current JAX random key for trainer
            record_video: Whether to record video for this episode

        Returns:
            Tuple of (new_state, new_trainer_key, episode_metrics)
        """
        obs = self.env.reset()
        total_reward = 0.0
        done = False
        episode_metrics = {}
        current_key = trainer_key

        while not done:
            # Record frame if video recording is enabled
            if record_video and self.tracker is not None:
                frame = self.env.render()
                if frame is not None:
                    self.tracker.add_video_frame(frame)

            # Split key into 3: next iteration, action selection, update
            keys = jax.random.split(current_key, 3)
            current_key = keys[0]

            action, state = self.select_action_jit(state, obs, keys[1])
            next_obs, reward, done = self.env.step(action)

            state, step_metrics = self.update_jit(state, obs, action, reward, next_obs, done, keys[2])

            # Collect metrics from agent updates (only when episode ends)
            if step_metrics:
                episode_metrics.update(step_metrics)

            obs = next_obs
            total_reward += reward

        # Always include episode return in metrics
        episode_metrics["return"] = total_reward

        return state, current_key, episode_metrics
    
    def train(self, state: Any, key: Array, num_episodes: int) -> Tuple[Any, Array]:
        """
        Train for multiple episodes and return final states.

        Args:
            state: Agent state to start training from
            key: JAX random key for reproducible training
            num_episodes: Number of episodes to train for

        Returns:
            Tuple of (final_state, final_trainer_key)
        """
        trainer_key = key

        for episode in range(1, num_episodes + 1):
            # Check if we should record video for this episode
            record_video = False
            if self.tracker is not None and self.tracker.should_record_video(episode):
                record_video = True

            # Run functional episode training
            state, trainer_key, episode_metrics = self.train_episode(
                state, trainer_key, record_video=record_video
            )

            if self.tracker is not None:
                self.tracker.log_metrics(episode, episode_metrics)

                # Save video if we recorded one
                if record_video:
                    self.tracker.save_video(episode)

        return state, trainer_key