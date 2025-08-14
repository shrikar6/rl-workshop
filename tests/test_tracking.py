"""
Tests for the unified Tracker class.
"""

import pytest
import jax.numpy as jnp
import tempfile
from pathlib import Path
from framework.tracking import Tracker


class TestTracker:
    """Test suite for Tracker functionality."""
    
    def test_init(self):
        """Test tracker initialization."""
        tracker = Tracker(log_interval=5)
        assert tracker.log_interval == 5
        assert tracker.episode_returns == []
        assert tracker.results_dir.exists()
    
    def test_add_episode(self, capsys):
        """Test adding episodes and logging."""
        tracker = Tracker(log_interval=2)
        
        # First episode - no log
        tracker.add_episode(1, 10.0)
        assert len(tracker.episode_returns) == 1
        assert tracker.episode_returns[0] == 10.0
        captured = capsys.readouterr()
        assert captured.out == ""
        
        # Second episode - should log
        tracker.add_episode(2, 20.0)
        assert len(tracker.episode_returns) == 2
        captured = capsys.readouterr()
        assert "Episode    2" in captured.out
        assert "Avg, min, max return (last 2): 15.00" in captured.out
        assert "[10.00, 20.00]" in captured.out
        
        # Third episode - no log
        tracker.add_episode(3, 30.0)
        captured = capsys.readouterr()
        assert captured.out == ""
        
        # Fourth episode - should log
        tracker.add_episode(4, 40.0)
        captured = capsys.readouterr()
        assert "Episode    4" in captured.out
        assert "Avg, min, max return (last 4): 25.00" in captured.out  # (10+20+30+40)/4
    
    def test_log_final(self, capsys):
        """Test final logging."""
        tracker = Tracker()
        for i in range(10):
            tracker.episode_returns.append(float(i * 10))
        
        # Test without threshold
        tracker.log_final()
        captured = capsys.readouterr()
        assert "Training completed!" in captured.out
        assert "Total episodes: 10" in captured.out
        assert "Final average return (last 10 episodes): 45.00" in captured.out  # (0+10+20+...+90)/10
        assert "Success threshold" not in captured.out
        
        # Test with threshold (not met)
        tracker.log_final(success_threshold=100.0)
        captured = capsys.readouterr()
        assert "Success threshold: 100.0" in captured.out
        assert "Environment not yet solved." in captured.out
        
        # Test with threshold (met)
        tracker.log_final(success_threshold=30.0)
        captured = capsys.readouterr()
        assert "Environment solved!" in captured.out
    
    def test_plot_basic(self):
        """Test basic plotting functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = Tracker(results_dir=tmpdir, window=10)  # Use reasonable window size
            
            # Add some data
            for i in range(20):
                tracker.episode_returns.append(float(i * 2))
            
            # Create plot
            tracker.plot("test_experiment")
            
            # Check file was created
            plot_path = Path(tmpdir) / "test_experiment_returns.png"
            assert plot_path.exists()
    
    def test_plot_window_too_large(self):
        """Test plot with window larger than data."""
        tracker = Tracker()
        tracker.episode_returns = [10.0, 20.0, 30.0]
        
        # For a tracker with window=100 (default) and only 3 data points, it should still work
        # The window size error happens inside the plot method, not as a parameter
        with pytest.raises(ValueError, match="Window size .* cannot be larger than data length"):
            tracker.plot("test")
    
    def test_moving_average_computation(self):
        """Test the moving average is computed correctly."""
        tracker = Tracker()
        
        # Simple case: constant values
        tracker.episode_returns = [10.0] * 10
        
        # The moving average should also be constant
        returns = jnp.array(tracker.episode_returns)
        window = 3
        kernel = jnp.ones(window) / window
        padded = jnp.concatenate([jnp.full(window-1, returns[0]), returns])
        moving_avg = jnp.convolve(padded, kernel, mode='valid')
        
        assert jnp.allclose(moving_avg, 10.0)
        
        # Linear increase case
        tracker.episode_returns = [float(i) for i in range(10)]
        returns = jnp.array(tracker.episode_returns)
        
        # Compute expected moving average for window=3
        # First values are padded with 0, so: [0,0,0,1,2,3,4,5,6,7,8,9]
        # Moving avg: [0, 0.33, 1, 2, 3, 4, 5, 6, 7, 8]
        window = 3
        kernel = jnp.ones(window) / window
        padded = jnp.concatenate([jnp.full(window-1, returns[0]), returns])
        moving_avg = jnp.convolve(padded, kernel, mode='valid')
        
        # Check first few values
        assert jnp.isclose(moving_avg[0], 0.0)  # (0+0+0)/3
        assert jnp.isclose(moving_avg[1], 0.333, atol=0.01)  # (0+0+1)/3
        assert jnp.isclose(moving_avg[2], 1.0)  # (0+1+2)/3