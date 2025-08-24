"""
Tests for the unified Tracker class.
"""

import tempfile
from pathlib import Path
from framework.tracking import Tracker


class TestTracker:
    """Test suite for Tracker functionality."""
    
    def test_init(self):
        """Test tracker initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = Tracker(log_interval=5, results_dir=tmpdir)
            assert tracker.log_interval == 5
            assert tracker.metrics == {}
            assert tracker.results_dir.exists()
    
    def test_log_metrics(self, capsys):
        """Test logging metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = Tracker(log_interval=2, window=2, results_dir=tmpdir)
            
            # Clear initial logging message
            capsys.readouterr()
            
            # First episode - no log
            tracker.log_metrics(1, {"return": 10.0, "loss": 0.5})
            assert len(tracker.metrics) == 2
            assert tracker.metrics["return"] == [10.0]
            assert tracker.metrics["loss"] == [0.5]
            captured = capsys.readouterr()
            assert captured.out == ""
            
            # Second episode - should log
            tracker.log_metrics(2, {"return": 20.0, "loss": 0.3})
            assert len(tracker.metrics["return"]) == 2
            captured = capsys.readouterr()
            assert "Episode    2" in captured.out
            assert "return: mean=15.00" in captured.out  # (10+20)/2
            assert "std=5.00" in captured.out  # std dev of [10, 20]
            assert "loss: mean=0.40" in captured.out  # (0.5+0.3)/2
            
            # Third episode - no log
            tracker.log_metrics(3, {"return": 30.0, "loss": 0.2})
            captured = capsys.readouterr()
            assert captured.out == ""
            
            # Fourth episode - should log (window of 2)
            tracker.log_metrics(4, {"return": 40.0, "loss": 0.1})
            captured = capsys.readouterr()
            assert "Episode    4" in captured.out
            assert "return: mean=35.00" in captured.out  # (30+40)/2 (window=2)
            assert "std=5.00" in captured.out  # std dev of [30, 40]
            assert "loss: mean=0.15" in captured.out  # (0.2+0.1)/2
    
    def test_log_final(self, capsys):
        """Test final logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = Tracker(results_dir=tmpdir)
            for i in range(10):
                tracker.log_metrics(i+1, {"return": float(i * 10), "loss": float(i * 0.1)})
            
            # Test without threshold
            tracker.log_final(metric="return")
            captured = capsys.readouterr()
            assert "Training completed!" in captured.out
            assert "Total episodes: 10" in captured.out
            assert "Final return (last 10 episodes): mean=45.00" in captured.out  # (0+10+20+...+90)/10
            assert "std=" in captured.out  # Should include std dev
            assert "Success threshold" not in captured.out
            
            # Test with threshold (not met)
            tracker.log_final(metric="return", success_threshold=100.0)
            captured = capsys.readouterr()
            assert "Success threshold: 100.0" in captured.out
            assert "Environment not yet solved." in captured.out
            
            # Test with threshold (met)
            tracker.log_final(metric="return", success_threshold=30.0)
            captured = capsys.readouterr()
            assert "Environment solved!" in captured.out
            
            # Test with custom window
            tracker.log_final(metric="return", success_threshold=50.0, window=5)
            captured = capsys.readouterr()
            assert "Final return (last 5 episodes): mean=70.00" in captured.out  # (50+60+70+80+90)/5
            assert "std=" in captured.out  # Should include std dev
            assert "Environment solved!" in captured.out
            
            # Test with non-existent metric
            tracker.log_final(metric="nonexistent")
            captured = capsys.readouterr()
            assert "Metric 'nonexistent' not found" in captured.out
    
    def test_plot_basic(self):
        """Test basic plotting functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = Tracker(
                results_dir=tmpdir, 
                window=10,
                experiment_name="test_experiment"
            )
            
            # Add some data
            for i in range(20):
                tracker.log_metrics(i+1, {"return": float(i * 2), "loss": float(20 - i)})
            
            # Create plot
            tracker.plot()
            
            # Check individual metric files were created in plots subdirectory
            return_plot_path = Path(tmpdir) / "test_experiment" / "plots" / "return.png"
            loss_plot_path = Path(tmpdir) / "test_experiment" / "plots" / "loss.png"
            assert return_plot_path.exists()
            assert loss_plot_path.exists()
    
    def test_plot_specific_metrics(self):
        """Test plotting specific metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = Tracker(results_dir=tmpdir, window=5)
            
            # Add multiple metrics
            for i in range(10):
                tracker.log_metrics(i+1, {
                    "return": float(i * 10),
                    "loss": float(1.0 / (i + 1)),
                    "entropy": float(i * 0.5)
                })
            
            # Plot only specific metrics
            tracker.plot(metrics=["return", "entropy"])
            
            # Check individual files were created for specified metrics in plots subdirectory
            return_plot_path = Path(tmpdir) / "plots" / "return.png"
            entropy_plot_path = Path(tmpdir) / "plots" / "entropy.png"
            loss_plot_path = Path(tmpdir) / "plots" / "loss.png"
            assert return_plot_path.exists()
            assert entropy_plot_path.exists()
            assert not loss_plot_path.exists()  # Should not plot loss since not specified
    
    def test_plot_window_too_large(self):
        """Test plot with window larger than data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = Tracker(results_dir=tmpdir, window=100)
            tracker.log_metrics(1, {"return": 10.0})
            tracker.log_metrics(2, {"return": 20.0})
            tracker.log_metrics(3, {"return": 30.0})
            
            # Should not raise error with new implementation
            # Moving average just won't be plotted if insufficient data
            tracker.plot()
    
    def test_get_metric(self):
        """Test getting individual metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = Tracker(results_dir=tmpdir)
            tracker.log_metrics(1, {"return": 10.0, "loss": 0.5})
            tracker.log_metrics(2, {"return": 20.0, "loss": 0.3})
            
            returns = tracker.get_metric("return")
            assert returns == [10.0, 20.0]
            
            losses = tracker.get_metric("loss")
            assert losses == [0.5, 0.3]
            
            nonexistent = tracker.get_metric("nonexistent")
            assert nonexistent is None
    
    def test_get_all_metrics(self):
        """Test getting all metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = Tracker(results_dir=tmpdir)
            tracker.log_metrics(1, {"return": 10.0, "loss": 0.5})
            tracker.log_metrics(2, {"return": 20.0, "loss": 0.3})
            
            all_metrics = tracker.get_all_metrics()
            assert "return" in all_metrics
            assert "loss" in all_metrics
            assert all_metrics["return"] == [10.0, 20.0]
            assert all_metrics["loss"] == [0.5, 0.3]
    
    def test_episode_count(self):
        """Test that episode count is tracked correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = Tracker(results_dir=tmpdir)
            
            tracker.log_metrics(5, {"return": 50.0})
            assert tracker.episode_count == 5
            
            tracker.log_metrics(10, {"return": 100.0})
            assert tracker.episode_count == 10