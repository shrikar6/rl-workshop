import jax
from typing import Optional, Dict, Any
from .environments import EnvironmentABC
from .agents import AgentABC
from .tracking import Tracker


class Trainer:
    """
    Basic training loop for reinforcement learning.
    
    Handles the core episode loop: reset environment, run episode,
    update agent from experience. Manages JAX random keys for reproducible training.
    Optionally tracks training progress with a Tracker.
    """
    
    def __init__(
        self, 
        environment: EnvironmentABC, 
        agent: AgentABC, 
        seed: int = 42,
        tracker: Optional[Tracker] = None
    ):
        """
        Initialize trainer with environment and agent.
        
        Args:
            environment: Environment to train in
            agent: Agent to train
            seed: Random seed for reproducible training
            tracker: Optional tracker for logging training progress
        """
        self.env = environment
        self.agent = agent
        self.key = jax.random.PRNGKey(seed)
        self.tracker = tracker
    
    def train_episode(self, state: Any, trainer_key, record_video: bool = False):
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

            action, state = self.agent.select_action(state, obs, keys[1])
            next_obs, reward, done = self.env.step(action)

            state, step_metrics = self.agent.update(state, obs, action, reward, next_obs, done, keys[2])

            # Collect metrics from agent updates (only when episode ends)
            if step_metrics:
                episode_metrics.update(step_metrics)

            obs = next_obs
            total_reward += reward

        # Always include episode return in metrics
        episode_metrics["return"] = total_reward

        return state, current_key, episode_metrics
    
    def train(self, state: Any, num_episodes: int):
        """
        Train for multiple episodes and return final states.

        Args:
            state: Agent state to start training from
            num_episodes: Number of episodes to train for

        Returns:
            Tuple of (final_state, final_trainer_key)
        """
        trainer_key = self.key

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