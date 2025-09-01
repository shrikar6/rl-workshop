"""Test video recording functionality."""

import tempfile
from pathlib import Path
import numpy as np

from framework import Tracker, Trainer, CartPoleEnv, REINFORCEAgent, ComposedPolicyNetwork, MLPBackbone, DiscretePolicyHead


def test_tracker_video_directory_creation():
    """Test that Tracker creates correct directory structure for videos."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create tracker with video recording enabled
        tracker = Tracker(
            log_interval=10,
            window=10,
            results_dir=tmpdir,
            video_interval=50,
            experiment_name="test_exp"
        )
        
        # Check directory structure
        exp_dir = Path(tmpdir) / "test_exp"
        videos_dir = exp_dir / "videos"
        
        assert exp_dir.exists()
        assert videos_dir.exists()


def test_tracker_should_record_video():
    """Test that should_record_video returns True at correct intervals."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = Tracker(video_interval=50, results_dir=tmpdir)
        
        # Should not record on episode 0
        assert not tracker.should_record_video(0)
        
        # Should not record on non-interval episodes
        assert not tracker.should_record_video(25)
        assert not tracker.should_record_video(49)
        
        # Should record on interval episodes
        assert tracker.should_record_video(50)
        assert tracker.should_record_video(100)
        assert tracker.should_record_video(150)


def test_tracker_no_video_when_disabled():
    """Test that video recording is disabled when video_interval is None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = Tracker(video_interval=None, results_dir=tmpdir)
        
        assert not tracker.should_record_video(50)
        assert not tracker.should_record_video(100)


def test_video_frame_collection():
    """Test that frames are collected and cleared properly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = Tracker(video_interval=50, results_dir=tmpdir)
        
        # Add some dummy frames
        frame1 = np.zeros((64, 64, 3), dtype=np.uint8)
        frame2 = np.ones((64, 64, 3), dtype=np.uint8) * 255
        
        tracker.add_video_frame(frame1)
        tracker.add_video_frame(frame2)
        
        assert len(tracker.current_video_frames) == 2
        assert np.array_equal(tracker.current_video_frames[0], frame1)
        assert np.array_equal(tracker.current_video_frames[1], frame2)


def test_integration_with_trainer():
    """Test that Trainer properly integrates with video recording."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set up environment and agent
        env = CartPoleEnv(seed=42)
        
        backbone = MLPBackbone(hidden_dims=[32], output_dim=16)
        head = DiscretePolicyHead(input_dim=16)
        policy = ComposedPolicyNetwork(backbone, head)
        
        agent = REINFORCEAgent(
            policy=policy,
            observation_space=env.observation_space,
            action_space=env.action_space,
            learning_rate=1e-3,
            gamma=0.99,
            seed=42
        )
        
        # Create tracker with video recording
        tracker = Tracker(
            log_interval=5,
            window=5,
            results_dir=tmpdir,
            video_interval=5,  # Record every 5 episodes for testing
            experiment_name="test_trainer"
        )
        
        # Create trainer
        trainer = Trainer(environment=env, agent=agent, seed=42, tracker=tracker)
        
        # Train for a few episodes
        trainer.train(num_episodes=10)
        
        # Check that video files were created at right episodes
        videos_dir = Path(tmpdir) / "test_trainer" / "videos"
        
        # Should have videos for episodes 5 and 10
        assert (videos_dir / "episode_5.mp4").exists()
        assert (videos_dir / "episode_10.mp4").exists()
        
        # Should not have videos for other episodes
        assert not (videos_dir / "episode_1.mp4").exists()
        assert not (videos_dir / "episode_3.mp4").exists()
        
        env.close()


def test_clear_old_videos():
    """Test that old videos are cleared when creating a new tracker."""
    with tempfile.TemporaryDirectory() as tmpdir:
        videos_dir = Path(tmpdir) / "test_exp" / "videos"
        videos_dir.mkdir(parents=True)
        
        # Create an old video file
        old_video = videos_dir / "old_video.mp4"
        old_video.write_text("dummy content")
        
        assert old_video.exists()
        
        # Create new tracker - should clear old videos
        tracker = Tracker(
            results_dir=tmpdir,
            video_interval=50,
            experiment_name="test_exp"
        )
        
        # Old video should be gone
        assert not old_video.exists()
        assert videos_dir.exists()  # Directory should still exist