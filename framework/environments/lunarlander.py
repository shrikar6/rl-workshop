import gymnasium as gym
import jax.numpy as jnp
from typing import Tuple, Optional
from jax import Array
from .base import EnvironmentABC


class LunarLanderEnv(EnvironmentABC):
    """
    Lunar Lander environment wrapper around Gymnasium's LunarLander-v3.
    
    The lander starts at the top of the screen with random initial force applied
    to its center of mass. The goal is to land on the landing pad using as little fuel as possible.
    - Observation: [x, y, x_vel, y_vel, angle, angular_vel, left_leg_contact, right_leg_contact]
    - Action: 0 (do nothing), 1 (fire left engine), 2 (fire main engine), 3 (fire right engine)
    - Reward: Shaped reward based on distance to landing pad, fuel consumption, and successful landing
    - Episode ends when lander crashes, lands, or goes out of bounds
    """
    
    def __init__(self, seed: Optional[int] = None, render_mode: Optional[str] = "rgb_array"):
        """
        Initialize Lunar Lander environment using Gymnasium.
        
        Args:
            seed: Optional random seed for environment reproducibility
            render_mode: Render mode for the environment ("human", "rgb_array", or None)
        """
        self.env = gym.make('LunarLander-v3', render_mode=render_mode)
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
            action: Action array containing discrete action (0, 1, 2, or 3)
            
        Returns:
            Tuple of (observation, reward, done)
        """
        # Convert JAX array to discrete action
        discrete_action = int(action[0])
        obs, reward, terminated, truncated, _ = self.env.step(discrete_action)
        return jnp.array(obs), float(reward), bool(terminated or truncated)
    
    @property
    def observation_space(self) -> gym.Space:
        """Observation space: [x, y, x_vel, y_vel, angle, angular_vel, left_leg_contact, right_leg_contact]"""
        return self.env.observation_space
    
    @property
    def action_space(self) -> gym.Space:
        """Action space: discrete actions {0, 1, 2, 3}"""
        return self.env.action_space
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment and clean up resources."""
        self.env.close()