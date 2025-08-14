"""
Utility functions for the RL framework.

Common utilities that are used across different framework components.
"""

import jax.numpy as jnp
import gymnasium as gym


def get_input_dim(observation_space: gym.Space) -> int:
    """
    Extract the input dimension from a Gymnasium observation space.
    
    This utility handles the common cases of converting different observation
    space types into a single input dimension for neural networks.
    
    For discrete observation spaces, we use one-hot encoding, so the input
    dimension equals the number of possible discrete values.
    
    Args:
        observation_space: Gymnasium space describing observations
        
    Returns:
        Input dimension as an integer
        
    Raises:
        ValueError: If the observation space type is not supported
        
    Examples:
        >>> import gymnasium as gym
        >>> space = gym.spaces.Box(-1, 1, shape=(4,))
        >>> get_input_dim(space)
        4
        
        >>> space = gym.spaces.Discrete(10)  # One-hot encoded
        >>> get_input_dim(space)
        10
        
        >>> space = gym.spaces.Box(0, 255, shape=(84, 84, 3))
        >>> get_input_dim(space)
        21168
    """
    if isinstance(observation_space, gym.spaces.Box):
        # For Box spaces, flatten all dimensions
        return int(jnp.prod(jnp.array(observation_space.shape)))
    elif isinstance(observation_space, gym.spaces.Discrete):
        # For discrete spaces, use one-hot encoding (input_dim = number of possible values)
        return observation_space.n
    else:
        raise ValueError(
            f"Unsupported observation space type: {type(observation_space)}. "
            f"Currently supported: Box, Discrete. "
            f"Please extend get_input_dim() to handle {type(observation_space).__name__} spaces."
        )


def get_action_dim(action_space: gym.Space) -> int:
    """
    Extract action dimension information from a Gymnasium action space.
    
    Args:
        action_space: Gymnasium space describing actions
        
    Returns:
        Number of discrete actions or continuous action dimensions
        
    Raises:
        ValueError: If the action space type is not supported
        
    Examples:
        >>> import gymnasium as gym  
        >>> space = gym.spaces.Discrete(2)
        >>> get_action_dim(space)
        2
        
        >>> space = gym.spaces.Box(-1, 1, shape=(4,))
        >>> get_action_dim(space)
        4
    """
    if isinstance(action_space, gym.spaces.Discrete):
        # For discrete actions, return number of possible actions
        return action_space.n
    elif isinstance(action_space, gym.spaces.Box):
        # For continuous actions, return the action dimension
        return int(jnp.prod(jnp.array(action_space.shape)))
    else:
        raise ValueError(
            f"Unsupported action space type: {type(action_space)}. "
            f"Currently supported: Discrete, Box. "
            f"Please extend get_action_dim() to handle {type(action_space).__name__} spaces."
        )