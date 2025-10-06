import gymnasium as gym
import jax.numpy as jnp
from typing import Tuple, Optional
from jax import Array
from .base import EnvironmentABC


class CartPoleEnv(EnvironmentABC):
    """
    CartPole balance environment wrapper around Gymnasium's CartPole-v1.
    
    Classic CartPole balance task: start near upright, keep the pole balanced.
    - Observation: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    - Action: 0 (push left) or 1 (push right) 
    - Reward: +1 for each step the pole stays upright
    - Episode ends when pole falls too far or cart goes out of bounds
    """
    
    def __init__(self, seed: Optional[int] = None, render_mode: Optional[str] = "rgb_array"):
        """
        Initialize CartPole balance environment using Gymnasium.
        
        Args:
            seed: Optional random seed for environment reproducibility
            render_mode: Render mode for the environment ("human", "rgb_array", or None)
        """
        self.env = gym.make('CartPole-v1', render_mode=render_mode)
        if seed is not None:
            self.env.reset(seed=seed)
    
    def reset(self) -> Array:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation as JAX array
        """
        obs, _ = self.env.reset()
        return jnp.array(obs)
    
    def step(self, action: Array) -> Tuple[Array, float, bool]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action array containing discrete action (0 or 1)
            
        Returns:
            Tuple of (observation, reward, done)
        """
        # Convert JAX array to discrete action
        discrete_action = int(action[0])
        obs, reward, terminated, truncated, _ = self.env.step(discrete_action)
        return jnp.array(obs), float(reward), bool(terminated or truncated)
    
    @property
    def observation_space(self) -> gym.Space:
        """Observation space: [x, x_dot, theta, theta_dot]"""
        return self.env.observation_space
    
    @property
    def action_space(self) -> gym.Space:
        """Action space: discrete actions {0, 1}"""
        return self.env.action_space
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment and clean up resources."""
        self.env.close()