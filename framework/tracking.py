"""
Unified tracking for RL experiments.

Handles metrics collection, logging, and visualization in a single class.
"""

import shutil
import jax.numpy as jnp
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
from typing import List, Optional


class Tracker:
    """
    Tracks training progress with logging and visualization.
    """
    
    def __init__(self, log_interval: int = 10, window: int = 100, results_dir: str = "results", 
                 video_interval: Optional[int] = None, experiment_name: Optional[str] = None):
        """
        Initialize tracker.
        
        Args:
            log_interval: Episodes between progress logs
            window: Number of recent episodes to use for statistics and moving average
            results_dir: Base directory for results
            video_interval: Episodes between video recordings (None to disable)
            experiment_name: Name of the experiment for organizing outputs
        """
        self.episode_returns: List[float] = []
        self.log_interval = log_interval
        self.window = window
        self.video_interval = video_interval
        self.experiment_name = experiment_name
        self.current_video_frames: List = []
        
        # Set up directory structure: results/{experiment_name}/ and results/{experiment_name}/videos/
        base_dir = Path(results_dir)
        if experiment_name:
            self.results_dir = base_dir / experiment_name
            self.results_dir.mkdir(parents=True, exist_ok=True)
            
            if video_interval is not None:
                self.videos_dir = self.results_dir / "videos"
                # Clear existing videos if directory exists
                if self.videos_dir.exists():
                    shutil.rmtree(self.videos_dir)
                self.videos_dir.mkdir(exist_ok=True)
        else:
            self.results_dir = base_dir
            self.results_dir.mkdir(exist_ok=True)
    
    def add_episode(self, episode: int, episode_return: float):
        """
        Record episode return and log progress if at interval.
        
        Args:
            episode: Current episode number (1-indexed)
            episode_return: Return for the current episode
        """
        self.episode_returns.append(float(episode_return))
        
        if episode % self.log_interval == 0:
            recent = self.episode_returns[-self.window:] if len(self.episode_returns) > self.window else self.episode_returns
            recent_array = jnp.array(recent)
            
            mean_return = float(jnp.mean(recent_array))
            min_return = float(jnp.min(recent_array))
            max_return = float(jnp.max(recent_array))
            
            print(f"Episode {episode:4d}: "
                  f"Avg, min, max return (last {len(recent)}): {mean_return:.2f} "
                  f"[{min_return:.2f}, {max_return:.2f}]")
    
    def should_record_video(self, episode: int) -> bool:
        """Check if we should record video for this episode."""
        return (self.video_interval is not None and 
                episode > 0 and 
                episode % self.video_interval == 0)
    
    def add_video_frame(self, frame):
        """Add a frame to the current video being recorded."""
        self.current_video_frames.append(frame)
    
    def save_video(self, episode: int):
        """Save collected frames as a video file."""
        if not self.current_video_frames:
            return
        
        video_path = self.videos_dir / f"episode_{episode}.mp4"
        imageio.mimsave(video_path, self.current_video_frames, fps=30)
        print(f"Video saved: {video_path}")
        self.current_video_frames = []
    
    def plot(self):
        """
        Create and save returns plot.
        """
        returns = jnp.array(self.episode_returns)
        episodes = jnp.arange(1, len(returns) + 1)
        
        # Compute moving average
        if self.window > len(returns):
            raise ValueError(f"Window size ({self.window}) cannot be larger than data length ({len(returns)})")
        
        kernel = jnp.ones(self.window) / self.window
        padded = jnp.concatenate([jnp.full(self.window-1, returns[0]), returns])
        moving_avg = jnp.convolve(padded, kernel, mode='valid')
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, returns, alpha=0.3, label='Episode returns', color='blue')
        plt.plot(episodes, moving_avg, label=f'Moving average ({self.window})', color='red', linewidth=2)
        
        plt.xlabel('Episode')
        plt.ylabel('Return')
        title = f'{self.experiment_name} - Training Progress' if self.experiment_name else 'Training Progress'
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        filename = "returns.png"
        plot_path = self.results_dir / filename
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to: {plot_path}")
    
    def log_final(self, success_threshold: Optional[float] = None):
        """
        Log final training results.
        
        Args:
            success_threshold: Optional threshold for success
        """
        final_returns = self.episode_returns[-self.window:] if len(self.episode_returns) > self.window else self.episode_returns
        mean_return = float(jnp.mean(jnp.array(final_returns)))
        
        print(f"\nTraining completed!")
        print(f"Total episodes: {len(self.episode_returns)}")
        print(f"Final average return (last {len(final_returns)} episodes): {mean_return:.2f}")
        
        if success_threshold is not None:
            print(f"Success threshold: {success_threshold}")
            if mean_return >= success_threshold:
                print("Environment solved!")
            else:
                print("Environment not yet solved.")