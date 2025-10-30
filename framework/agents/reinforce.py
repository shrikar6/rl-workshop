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
from typing import NamedTuple, Any, Tuple, Dict
from jax import Array
from ..networks.policy import PolicyNetworkABC
from .base import AgentABC


class REINFORCEState(NamedTuple):
    """
    Immutable state for REINFORCE agent.

    Tracks policy parameters, optimizer state, episode buffers, and baseline value.
    State is updated functionally - methods return new state objects.
    Episode buffers are pre-allocated JAX arrays for O(n) complexity.
    """
    policy_params: Any
    opt_state: Any
    episode_observations: Array  # Pre-allocated buffer [max_episode_length, obs_shape]
    episode_actions: Array  # Pre-allocated buffer [max_episode_length, action_shape]
    episode_rewards: Array  # Pre-allocated buffer [max_episode_length]
    episode_length: int  # Current number of transitions stored in buffers
    baseline: float  # Moving average of episode returns for variance reduction


class REINFORCEAgent(AgentABC):
    """
    REINFORCE agent implementation.

    Collects complete episodes and updates the policy using Monte Carlo
    returns to estimate policy gradients. Maintains episode buffers in
    state and updates only when episodes complete.
    """
    
    def __init__(
        self,
        policy: PolicyNetworkABC,
        observation_space: gym.Space,
        action_space: gym.Space,
        max_episode_length: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        baseline_alpha: float = 0.01,
        normalize_advantages: bool = False
    ):
        """
        Initialize REINFORCE agent.

        Args:
            policy: Policy network to optimize
            observation_space: Environment observation space
            action_space: Environment action space
            max_episode_length: Maximum episode length for pre-allocating buffers
            learning_rate: Step size for gradient updates
            gamma: Discount factor for future rewards
            baseline_alpha: Exponential moving average coefficient for baseline update
            normalize_advantages: Whether to normalize advantages by std (default: False)
        """
        # Validate hyperparameters
        if not (0 <= gamma <= 1):
            raise ValueError(f"gamma must be in [0,1], got {gamma}")
        if not (0 <= baseline_alpha <= 1):
            raise ValueError(f"baseline_alpha must be in [0,1], got {baseline_alpha}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if max_episode_length <= 0:
            raise ValueError(f"max_episode_length must be positive, got {max_episode_length}")

        self.policy = policy
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.baseline_alpha = baseline_alpha
        self.observation_space = observation_space
        self.action_space = action_space
        self.max_episode_length = max_episode_length
        self.normalize_advantages = normalize_advantages

        # Initialize optimizer
        self.optimizer = optax.adam(learning_rate)

    def init_state(self, key: Array) -> REINFORCEState:
        """
        Create initial agent state.

        Args:
            key: JAX random key for parameter initialization

        Returns:
            Initial REINFORCEState with randomly initialized parameters
        """
        policy_params = self.policy.init_params(key, self.observation_space, self.action_space)
        opt_state = self.optimizer.init(policy_params)

        # Pre-allocate episode buffers
        obs_shape = self.observation_space.shape
        action_shape = (1,)  # Actions are single scalars wrapped in arrays

        episode_observations = jnp.zeros((self.max_episode_length, *obs_shape))
        episode_actions = jnp.zeros((self.max_episode_length, *action_shape))
        episode_rewards = jnp.zeros(self.max_episode_length)

        return REINFORCEState(
            policy_params=policy_params,
            opt_state=opt_state,
            episode_observations=episode_observations,
            episode_actions=episode_actions,
            episode_rewards=episode_rewards,
            episode_length=0,
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
        action = self.policy.sample_action(state.policy_params, observation, key)

        # Store observation and action in pre-allocated buffers
        new_state = state._replace(
            episode_observations=state.episode_observations.at[state.episode_length].set(observation),
            episode_actions=state.episode_actions.at[state.episode_length].set(action),
            episode_length=state.episode_length + 1
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
    ) -> Tuple[REINFORCEState, Dict[str, float]]:
        """
        Store rewards and update policy at episode end, returning new state and metrics.

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
            Tuple of (new agent state, metrics dict)
        """
        # Store reward in pre-allocated buffer at the current index
        # episode_length was already incremented in select_action, so we use episode_length-1
        reward_idx = state.episode_length - 1
        new_state = state._replace(
            episode_rewards=state.episode_rewards.at[reward_idx].set(reward)
        )

        # Define branches for episode completion using JAX conditional
        # (required for JIT compilation - can't use Python if with traced values)
        def update_and_reset(s):
            """Branch: Episode is complete, update policy and reset buffers."""
            updated_params, updated_opt_state, updated_baseline, metrics = self._update_policy(s)

            # Reuse existing buffers instead of creating new ones
            # Old data doesn't matter - we track valid data via episode_length
            return s._replace(
                policy_params=updated_params,
                opt_state=updated_opt_state,
                episode_length=0,
                baseline=updated_baseline
            ), metrics

        def continue_episode(s):
            """Branch: Episode continues, return state unchanged with empty metrics."""
            # Return same structure as update_and_reset but with dummy metrics
            # (jax.lax.cond requires both branches to have same pytree structure)
            empty_metrics = {
                "policy_loss": 0.0,
                "baseline": 0.0,
                "mean_advantage": 0.0,
                "grad_norm": 0.0
            }
            return s, empty_metrics

        # Use JAX conditional instead of Python if for JIT compatibility
        return jax.lax.cond(done, update_and_reset, continue_episode, new_state)

    def compute_baseline_and_advantages(
        self,
        rewards: Array,
        gamma: float,
        old_baseline: float,
        baseline_alpha: float,
        mask: Array,
        episode_length: int
    ):
        """
        Compute updated baseline and advantages for policy gradient.

        Uses masking to handle variable-length episodes in pre-allocated buffers.
        Returns advantages with padding positions set to 0 (fully masked).

        Args:
            rewards: Full pre-allocated rewards array (includes padding)
            gamma: Discount factor
            old_baseline: Current baseline value
            baseline_alpha: Learning rate for baseline update
            mask: Boolean mask indicating valid episode steps vs padding
            episode_length: Number of valid steps in episode

        Returns:
            Tuple of (updated_baseline, advantages where padding = 0)
        """
        def discount_step(carry, reward_and_mask):
            reward, is_valid = reward_and_mask
            # Only accumulate return for valid timesteps
            current_return = jnp.where(is_valid, reward + gamma * carry, 0.0)
            return current_return, current_return

        # Compute returns using scan (efficient for JAX)
        # Process rewards in reverse to accumulate discounted returns from end to start
        # Stack rewards with mask so scan can handle both
        rewards_masked = jnp.stack([rewards, mask], axis=1)
        _, returns = jax.lax.scan(
            discount_step,
            0.0,
            rewards_masked[::-1]
        )
        returns = returns[::-1]  # Reverse back to chronological order

        # Update baseline using exponential moving average of episode return
        episode_return = returns[0]  # Return from episode start
        updated_baseline = (1 - baseline_alpha) * old_baseline + baseline_alpha * episode_return

        # Compute advantages (returns minus baseline for variance reduction)
        # Mask advantages so padding positions are 0 (encapsulate masking here)
        advantages = jnp.where(mask, returns - old_baseline, 0.0)

        return updated_baseline, advantages

    
    def _update_policy(self, state: REINFORCEState) -> Tuple[Any, Any, float, Dict[str, float]]:
        """
        Update policy parameters using REINFORCE gradient estimator with baseline.

        The REINFORCE gradient with baseline is:
        ∇J(θ) = E[∑_t ∇log π(a_t|s_t; θ) * (G_t - b)]

        Where G_t is the return from timestep t and b is the baseline.

        Args:
            state: Current agent state

        Returns:
            Tuple of (updated_policy_params, updated_opt_state, updated_baseline, metrics)
        """
        # Use full pre-allocated buffers (JIT-compatible, no dynamic slicing)
        # Create mask to identify valid episode data vs padding
        mask = jnp.arange(self.max_episode_length) < state.episode_length

        rewards = state.episode_rewards
        observations = state.episode_observations
        actions = state.episode_actions

        # Compute returns, updated baseline, and advantages with masking
        # advantages are returned already masked (padding positions = 0)
        updated_baseline, advantages = self.compute_baseline_and_advantages(
            rewards, self.gamma, state.baseline, self.baseline_alpha, mask, state.episode_length
        )

        # Optionally normalize advantages by std to reduce variance
        if self.normalize_advantages:
            # Compute std only over valid advantages (padding already = 0)
            mean_adv = jnp.sum(advantages) / state.episode_length
            squared_diff = (advantages - mean_adv) ** 2
            variance = jnp.sum(squared_diff) / state.episode_length
            std = jnp.sqrt(variance)
            std_clamped = jnp.maximum(std, 1e-3)  # Clamp to prevent division by ~0
            advantages = advantages / std_clamped

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

            # Multiply by advantages (padding naturally contributes 0)
            weighted_log_probs = log_probs * advantages

            # REINFORCE loss with baseline: -log_prob * advantage
            # Sum over all timesteps (padding contributes 0) and divide by episode length
            loss = -jnp.sum(weighted_log_probs) / state.episode_length
            return loss

        # Compute loss value and gradients efficiently in single pass
        loss_value, grads = jax.value_and_grad(policy_loss)(state.policy_params)

        # Apply gradients to update parameters
        updates, new_opt_state = self.optimizer.update(grads, state.opt_state)
        new_policy_params = optax.apply_updates(state.policy_params, updates)

        # Compute metrics to track
        grad_norm = optax.global_norm(grads)
        mean_advantage = jnp.sum(advantages) / state.episode_length

        metrics = {
            "policy_loss": loss_value,
            "baseline": updated_baseline,
            "mean_advantage": mean_advantage,
            "grad_norm": grad_norm
        }

        return new_policy_params, new_opt_state, updated_baseline, metrics