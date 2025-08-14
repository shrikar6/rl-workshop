"""
Unified tracking for RL experiments.

Handles metrics collection, logging, and visualization in a single class.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional


class Tracker:
    """
    Tracks training progress with logging and visualization.
    """
    
    def __init__(self, log_interval: int = 10, window: int = 100, results_dir: str = "results"):
        """
        Initialize tracker.
        
        Args:
            log_interval: Episodes between progress logs
            window: Number of recent episodes to use for statistics and moving average
            results_dir: Directory to save plots
        """
        self.episode_returns: List[float] = []
        self.log_interval = log_interval
        self.window = window
        self.results_dir = Path(results_dir)
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
    
    def plot(self, experiment_name: str):
        """
        Create and save returns plot.
        
        Args:
            experiment_name: Name for the plot file
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
        plt.title(f'{experiment_name} - Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = self.results_dir / f"{experiment_name}_returns.png"
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