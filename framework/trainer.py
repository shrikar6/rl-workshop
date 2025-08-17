import jax
from typing import Optional
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
    
    def train_episode(self, record_video: bool = False) -> float:
        """
        Run one complete episode and return total reward.
        
        Args:
            record_video: Whether to record video for this episode
            
        Returns:
            Total reward accumulated during the episode
        """
        obs = self.env.reset()
        total_reward = 0.0
        done = False
        
        # Start with current agent state
        agent_state = self.agent.state
        
        while not done:
            # Record frame if video recording is enabled
            if record_video and self.tracker is not None:
                frame = self.env.render()
                if frame is not None:
                    self.tracker.add_video_frame(frame)
            
            # Split keys for action selection and policy updates
            keys = jax.random.split(self.key, 3)
            self.key = keys[0]
            
            action, agent_state = self.agent.select_action(agent_state, obs, keys[1])
            next_obs, reward, done = self.env.step(action)
            
            agent_state = self.agent.update(agent_state, obs, action, reward, next_obs, done, keys[2])
            
            obs = next_obs
            total_reward += reward
        
        # Update agent's state after episode
        self.agent.state = agent_state
        
        return total_reward
    
    def train(self, num_episodes: int) -> None:
        """
        Train for multiple episodes.
        
        Args:
            num_episodes: Number of episodes to train for
        """
        for episode in range(1, num_episodes + 1):
            # Check if we should record video for this episode
            record_video = False
            if self.tracker is not None and self.tracker.should_record_video(episode):
                record_video = True
            
            episode_reward = self.train_episode(record_video=record_video)
            
            if self.tracker is not None:
                self.tracker.add_episode(episode, episode_reward)
                
                # Save video if we recorded one
                if record_video:
                    self.tracker.save_video(episode)