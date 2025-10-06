import gymnasium as gym
import jax.numpy as jnp
from typing import Tuple, Optional
from jax import Array
from .base import EnvironmentABC


class AcrobotEnv(EnvironmentABC):
    """
    Acrobot swing-up environment wrapper around Gymnasium's Acrobot-v1.
    
    Acrobot is an underactuated pendulum with two links. The goal is to swing
    the end-effector to a height at least the length of one link above the base.
    - Observation: [cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta1_dot, theta2_dot]
    - Action: 0 (apply -1 torque), 1 (apply 0 torque), 2 (apply +1 torque)
    - Reward: -1 for each step until the goal is reached
    - Episode ends when goal is reached or after 500 steps
    """
    
    def __init__(self, seed: Optional[int] = None, render_mode: Optional[str] = "rgb_array"):
        """
        Initialize Acrobot swing-up environment using Gymnasium.
        
        Args:
            seed: Optional random seed for environment reproducibility
            render_mode: Render mode for the environment ("human", "rgb_array", or None)
        """
        self.env = gym.make('Acrobot-v1', render_mode=render_mode)
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
            action: Action array containing discrete action (0, 1, or 2)
            
        Returns:
            Tuple of (observation, reward, done)
        """
        # Convert JAX array to discrete action
        discrete_action = int(action[0])
        obs, reward, terminated, truncated, _ = self.env.step(discrete_action)
        return jnp.array(obs), float(reward), bool(terminated or truncated)
    
    @property
    def observation_space(self) -> gym.Space:
        """Observation space: [cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta1_dot, theta2_dot]"""
        return self.env.observation_space
    
    @property
    def action_space(self) -> gym.Space:
        """Action space: discrete actions {0, 1, 2}"""
        return self.env.action_space
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment and clean up resources."""
        self.env.close()