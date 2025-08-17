"""
REINFORCE: Monte Carlo Policy Gradient

The simplest policy gradient algorithm. Uses complete episode returns
to estimate gradients, making it high-variance but unbiased.

Reference: Williams (1992) "Simple Statistical Gradient-Following Algorithms
for Connectionist Reinforcement Learning"
"""

import jax
import jax.numpy as jnp
import optax
import gymnasium as gym
from typing import List, NamedTuple, Any, Tuple
from jax import Array
from ..policies import PolicyABC
from .base import AgentABC


class REINFORCEState(NamedTuple):
    """
    Immutable state for REINFORCE agent.
    
    All state is immutable - methods return new state objects rather than
    mutating existing state. This enables functional programming patterns.
    """
    policy_params: Any
    opt_state: Any
    episode_observations: List[Array]
    episode_actions: List[Array]
    episode_rewards: List[float]
    baseline: float  # Moving average of episode returns for variance reduction


class REINFORCEAgent(AgentABC):
    """
    REINFORCE agent implementation.
    
    This agent collects complete episodes and updates the policy using
    Monte Carlo returns to estimate policy gradients.
    
    Follows functional programming principles - all methods are pure functions
    that return new state rather than mutating existing state.
    """
    
    def __init__(
        self, 
        policy: PolicyABC, 
        observation_space: gym.Space,
        action_space: gym.Space,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        baseline_alpha: float = 0.01,
        seed: int = 0
    ):
        """
        Initialize REINFORCE agent and create initial state.
        
        Args:
            policy: Policy network to optimize
            observation_space: Environment observation space
            action_space: Environment action space  
            learning_rate: Step size for gradient updates
            gamma: Discount factor for future rewards
            baseline_alpha: Exponential moving average coefficient for baseline update
            seed: Random seed for parameter initialization
        """
        self.policy = policy
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.baseline_alpha = baseline_alpha
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Initialize optimizer
        self.optimizer = optax.adam(learning_rate)
        
        # Create initial state
        key = jax.random.PRNGKey(seed)
        key, init_key = jax.random.split(key)
        policy_params = policy.init_params(init_key, observation_space, action_space)
        opt_state = self.optimizer.init(policy_params)
        
        self.state = REINFORCEState(
            policy_params=policy_params,
            opt_state=opt_state,
            episode_observations=[],
            episode_actions=[],
            episode_rewards=[],
            baseline=0.0
        )
    
    def select_action(self, state: REINFORCEState, observation: Array, key: Array) -> Tuple[Array, REINFORCEState]:
        """
        Select action using current policy and return new state.
        
        During training, we store the observation and action for later
        policy gradient computation.
        
        Args:
            state: Current agent state
            observation: Current state
            key: Random key for stochastic action selection
            
        Returns:
            Tuple of (action, new_agent_state)
        """
        # Select action using current policy parameters
        action = self.policy(state.policy_params, observation, key)
        
        # Create new state with stored observation and action
        new_state = state._replace(
            episode_observations=state.episode_observations + [observation],
            episode_actions=state.episode_actions + [action]
        )
        
        return action, new_state
    
    def update(
        self, 
        state: REINFORCEState,
        obs: Array, 
        action: Array, 
        reward: float, 
        next_obs: Array, 
        done: bool, 
        key: Array
    ) -> REINFORCEState:
        """
        Store rewards and update policy at episode end, returning new state.
        
        REINFORCE waits until the episode is complete before updating,
        because it needs the full trajectory to compute returns.
        
        Args:
            state: Current agent state
            obs: Current observation (unused - we stored it in select_action)
            action: Action taken (unused - we stored it in select_action)
            reward: Reward received
            next_obs: Next observation (unused in REINFORCE)
            done: Whether episode ended
            key: Random key (unused in this update)
            
        Returns:
            New agent state after update
        """
        # Add reward to episode buffer
        new_state = state._replace(
            episode_rewards=state.episode_rewards + [reward]
        )
        
        # Only update when episode is complete
        if done:
            # Update policy and baseline, then clear buffers for next episode
            updated_params, updated_opt_state, updated_baseline = self._update_policy(new_state)
            return REINFORCEState(
                policy_params=updated_params,
                opt_state=updated_opt_state,
                episode_observations=[],
                episode_actions=[],
                episode_rewards=[],
                baseline=updated_baseline
            )
        else:
            return new_state
    
    def _compute_returns(self, state: REINFORCEState) -> Array:
        """
        Compute discounted returns for each timestep using JAX scan.
        
        G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γ^{T-t}*r_T
        
        Uses jax.lax.scan for efficient computation instead of Python loops.
        
        Returns:
            Array of returns, one per timestep
        """
        rewards = jnp.array(state.episode_rewards)
        
        def discount_step(carry, reward):
            """
            Compute one step of discounted return calculation.
            
            Args:
                carry: Accumulated return from future timesteps (starts at 0)
                reward: Current reward being processed
                
            Returns:
                (new_carry, current_return) where:
                - new_carry: Updated accumulated return for next iteration
                - current_return: The return value for this timestep
            """
            # G_t = r_t + γ * G_{t+1}
            current_return = reward + self.gamma * carry
            return current_return, current_return
        
        # Process rewards backwards (from end of episode)
        # rewards[::-1] reverses the array so we start from the last reward
        _, returns = jax.lax.scan(
            discount_step,
            0.0,  # Initial carry: no future rewards beyond episode end
            rewards[::-1]  # Process rewards in reverse order
        )
        
        # Reverse returns back to chronological order
        return returns[::-1]
    
    def _update_policy(self, state: REINFORCEState) -> Tuple[Any, Any, float]:
        """
        Update policy parameters using REINFORCE gradient estimator with baseline.
        
        The REINFORCE gradient with baseline is:
        ∇J(θ) = E[∑_t ∇log π(a_t|s_t; θ) * (G_t - b)]
        
        Where G_t is the return from timestep t and b is the baseline.
        
        Args:
            state: Current agent state
            
        Returns:
            Tuple of (updated_policy_params, updated_opt_state, updated_baseline)
        """
        # Compute returns for all timesteps
        returns = self._compute_returns(state)
        
        # Update baseline using exponential moving average of episode return
        episode_return = returns[0]  # Return from episode start
        updated_baseline = (1 - self.baseline_alpha) * state.baseline + self.baseline_alpha * episode_return
        
        # Convert episode data to JAX arrays
        observations = jnp.stack(state.episode_observations)
        actions = jnp.stack(state.episode_actions)
        
        # Compute advantages by subtracting baseline from returns
        advantages = returns - state.baseline
        
        # Define loss function for policy gradient
        def policy_loss(params):
            """
            Compute negative log probability weighted by returns.
            
            We minimize the negative of the policy gradient objective,
            which is equivalent to maximizing expected returns.
            """
            # Get log probabilities for all observations and actions
            log_probs = jax.vmap(
                lambda obs, act: self.policy.get_log_prob(params, obs, act)
            )(observations, actions)
            
            # REINFORCE loss with baseline: -log_prob * advantage
            # Negative because we want to maximize returns (minimize negative returns)
            loss = -jnp.mean(log_probs * advantages)
            return loss
        
        # Compute gradients
        grads = jax.grad(policy_loss)(state.policy_params)
        
        # Apply gradients to update parameters
        updates, new_opt_state = self.optimizer.update(grads, state.opt_state)
        new_policy_params = optax.apply_updates(state.policy_params, updates)
        
        return new_policy_params, new_opt_state, updated_baseline