"""
Unified tracking for RL experiments.

Handles metrics collection, logging, and visualization in a single class.
"""

import shutil
import jax.numpy as jnp
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime


class Tracker:
    """
    Tracks training progress with logging and visualization for arbitrary metrics.
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
        self.metrics: Dict[str, List[float]] = {}
        self.log_interval = log_interval
        self.window = window
        self.video_interval = video_interval
        self.experiment_name = experiment_name
        self.current_video_frames: List = []
        self.episode_count = 0
        
        # Set up directory structure: results/{experiment_name}/
        base_dir = Path(results_dir)
        if experiment_name:
            self.results_dir = base_dir / experiment_name
            self.results_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            self.plots_dir = self.results_dir / "plots"
            self.logs_dir = self.results_dir / "logs"
            self.plots_dir.mkdir(exist_ok=True)
            self.logs_dir.mkdir(exist_ok=True)
            
            if video_interval is not None:
                self.videos_dir = self.results_dir / "videos"
                # Clear existing videos if directory exists
                if self.videos_dir.exists():
                    shutil.rmtree(self.videos_dir)
                self.videos_dir.mkdir(exist_ok=True)
        else:
            self.results_dir = base_dir
            self.results_dir.mkdir(exist_ok=True)
            self.plots_dir = self.results_dir / "plots"
            self.logs_dir = self.results_dir / "logs"
            self.plots_dir.mkdir(exist_ok=True)
            self.logs_dir.mkdir(exist_ok=True)
            
        # Set up file logging
        self._setup_file_logging()
    
    def log_metrics(self, episode: int, metrics: Dict[str, float]):
        """
        Record metrics for the current episode and log progress if at interval.
        
        Args:
            episode: Current episode number (1-indexed)
            metrics: Dictionary of metric name to value pairs
        """
        # Store metrics
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(float(value))
        
        self.episode_count = episode
        
        # Log at intervals
        if episode % self.log_interval == 0:
            log_str = f"Episode {episode:4d}: "
            log_parts = []
            
            for name, values in self.metrics.items():
                recent = values[-self.window:] if len(values) > self.window else values
                recent_array = jnp.array(recent)
                mean_val = float(jnp.mean(recent_array))
                std_val = float(jnp.std(recent_array))
                log_parts.append(f"{name}: mean={mean_val:.4f}, std={std_val:.4f}")
            
            log_message = log_str + " | ".join(log_parts)
            print(log_message)
            # Also log to file
            if hasattr(self, 'file_logger'):
                self.file_logger.info(log_message)
    
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
    
    def plot(self, metrics: Optional[List[str]] = None):
        """
        Create and save plots for specified metrics.
        
        Args:
            metrics: List of metric names to plot. If None, plots all tracked metrics.
        """
        if not self.metrics:
            print("No metrics to plot")
            return
        
        # Determine which metrics to plot
        metrics_to_plot = metrics if metrics is not None else list(self.metrics.keys())
        metrics_to_plot = [m for m in metrics_to_plot if m in self.metrics]
        
        if not metrics_to_plot:
            print("No valid metrics to plot")
            return
        
        # Create individual plots for each metric
        for metric_name in metrics_to_plot:
            fig, (ax_mean, ax_var) = plt.subplots(2, 1, figsize=(10, 8))
            
            values = jnp.array(self.metrics[metric_name])
            episodes = jnp.arange(1, len(values) + 1)
            
            # Plot mean subplot
            ax_mean.plot(episodes, values, alpha=0.3, label=f'{metric_name} (raw)', color='blue')
            
            # Compute and plot moving average and variance if we have enough data
            if len(values) >= self.window:
                # Compute moving mean
                kernel = jnp.ones(self.window) / self.window
                padded = jnp.concatenate([jnp.full(self.window-1, values[0]), values])
                moving_avg = jnp.convolve(padded, kernel, mode='valid')
                
                # Compute moving standard deviation
                moving_std = jnp.zeros(len(values))
                for j in range(len(values)):
                    window_start = max(0, j - self.window + 1)
                    window_values = values[window_start:j+1]
                    moving_std = moving_std.at[j].set(float(jnp.std(window_values)))
                
                # Plot moving average
                ax_mean.plot(episodes, moving_avg, label=f'{metric_name} (mean, window={self.window})', 
                           color='red', linewidth=2)
                
                # Plot moving standard deviation
                ax_var.plot(episodes, moving_std, label=f'{metric_name} (std dev, window={self.window})',
                          color='green', linewidth=2)
            
            # Configure mean subplot
            ax_mean.set_xlabel('Episode')
            ax_mean.set_ylabel(f'{metric_name.replace("_", " ").title()} (Mean)')
            ax_mean.set_title(f'{metric_name.replace("_", " ").title()} - Mean over Training')
            ax_mean.legend()
            ax_mean.grid(True, alpha=0.3)
            
            # Configure standard deviation subplot
            ax_var.set_xlabel('Episode')
            ax_var.set_ylabel(f'{metric_name.replace("_", " ").title()} (Std Dev)')
            ax_var.set_title(f'{metric_name.replace("_", " ").title()} - Standard Deviation over Training')
            ax_var.legend()
            ax_var.grid(True, alpha=0.3)
            
            # Overall title
            if self.experiment_name:
                fig.suptitle(f'{self.experiment_name} - {metric_name.replace("_", " ").title()}', fontsize=14, y=1.02)
            
            plt.tight_layout()
            
            # Save plot with metric name in filename in plots subdirectory
            filename = f"{metric_name}.png"
            plot_path = self.plots_dir / filename
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Plot saved to: {plot_path}")
    
    def log_final(self, metric: str = "return", success_threshold: Optional[float] = None, 
                  window: Optional[int] = None):
        """
        Log final training results for a specific metric.
        
        Args:
            metric: Name of the metric to report on
            success_threshold: Optional threshold for success
            window: Window size for final statistics (defaults to tracker's window)
        """
        if metric not in self.metrics:
            print(f"Metric '{metric}' not found in tracked metrics")
            return
        
        values = self.metrics[metric]
        final_window = window if window is not None else self.window
        final_values = values[-final_window:] if len(values) > final_window else values
        final_array = jnp.array(final_values)
        mean_value = float(jnp.mean(final_array))
        std_value = float(jnp.std(final_array))
        
        final_message = f"\nTraining completed!"
        episode_message = f"Total episodes: {self.episode_count}"
        result_message = f"Final {metric} (last {len(final_values)} episodes): mean={mean_value:.2f}, std={std_value:.2f}"
        
        print(final_message)
        print(episode_message)
        print(result_message)
        
        # Also log to file
        if hasattr(self, 'file_logger'):
            self.file_logger.info(final_message.strip())
            self.file_logger.info(episode_message)
            self.file_logger.info(result_message)
        
        if success_threshold is not None:
            print(f"Success threshold: {success_threshold}")
            if mean_value >= success_threshold:
                print("Environment solved!")
            else:
                print("Environment not yet solved.")
    
    def get_metric(self, name: str) -> Optional[List[float]]:
        """
        Get all values for a specific metric.
        
        Args:
            name: Name of the metric
            
        Returns:
            List of metric values or None if metric doesn't exist
        """
        return self.metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, List[float]]:
        """
        Get all tracked metrics.
        
        Returns:
            Dictionary of all metrics and their values
        """
        return self.metrics.copy()
    
    def _setup_file_logging(self):
        """Set up file logging for training progress."""
        # Create logger
        self.file_logger = logging.getLogger(f'tracker_{id(self)}')
        self.file_logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if self.file_logger.handlers:
            self.file_logger.handlers.clear()
        
        # Create file handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f'training_{timestamp}.log'
        log_path = self.logs_dir / log_filename
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.file_logger.addHandler(file_handler)
        
        # Log experiment start
        self.file_logger.info(f'Training experiment started: {self.experiment_name or "unnamed"}')
        print(f"Logging to: {log_path}")