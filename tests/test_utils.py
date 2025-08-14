"""
Tests for utility functions.
"""

import pytest
import gymnasium as gym
from framework.utils import get_input_dim, get_action_dim


class TestGetInputDim:
    """Tests for get_input_dim utility function."""
    
    def test_box_space_simple(self):
        """Test input dimension extraction from simple Box spaces."""
        space = gym.spaces.Box(-1, 1, shape=(4,))
        assert get_input_dim(space) == 4
    
    def test_box_space_multidimensional(self):
        """Test input dimension extraction from multidimensional Box spaces."""
        # Image-like space
        space = gym.spaces.Box(0, 255, shape=(84, 84, 3))
        assert get_input_dim(space) == 84 * 84 * 3
        
        # 2D space
        space = gym.spaces.Box(-1, 1, shape=(10, 5))
        assert get_input_dim(space) == 50
    
    def test_discrete_space(self):
        """Test input dimension extraction from Discrete spaces."""
        space = gym.spaces.Discrete(10)
        assert get_input_dim(space) == 10
        
        space = gym.spaces.Discrete(100)
        assert get_input_dim(space) == 100
    
    def test_unsupported_space_type(self):
        """Test that unsupported space types raise appropriate errors."""
        space = gym.spaces.MultiDiscrete([3, 4])
        
        with pytest.raises(ValueError, match="Unsupported observation space type"):
            get_input_dim(space)


class TestGetActionDim:
    """Tests for get_action_dim utility function."""
    
    def test_discrete_action_space(self):
        """Test action dimension extraction from Discrete spaces."""
        space = gym.spaces.Discrete(2)
        assert get_action_dim(space) == 2
        
        space = gym.spaces.Discrete(5)
        assert get_action_dim(space) == 5
    
    def test_box_action_space_simple(self):
        """Test action dimension extraction from simple Box spaces."""
        space = gym.spaces.Box(-1, 1, shape=(4,))
        assert get_action_dim(space) == 4
    
    def test_box_action_space_multidimensional(self):
        """Test action dimension extraction from multidimensional Box spaces."""
        space = gym.spaces.Box(-1, 1, shape=(2, 3))
        assert get_action_dim(space) == 6
    
    def test_unsupported_action_space_type(self):
        """Test that unsupported action space types raise appropriate errors."""
        space = gym.spaces.MultiDiscrete([3, 4])
        
        with pytest.raises(ValueError, match="Unsupported action space type"):
            get_action_dim(space)