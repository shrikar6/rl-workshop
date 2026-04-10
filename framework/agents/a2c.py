"""
A2C: Advantage Actor-Critic

An episode-based actor-critic algorithm that uses Generalized Advantage
Estimation (GAE) and separate policy and value networks. Collects complete
episodes, then computes TD-based advantages and updates both networks.

Reference: Mnih et al. (2016) "Asynchronous Methods for Deep Reinforcement Learning"
"""

import jax
import jax.numpy as jnp
import optax
import gymnasium as gym
from typing import NamedTuple, Any, Tuple, Dict
from jax import Array
from ..networks.policy.base import PolicyNetworkABC
from ..networks.value.base import ValueNetworkABC
from .base import AgentABC


class A2CState(NamedTuple):
    """
    Immutable state for A2C agent.

    Tracks policy and value network parameters, optimizer states, and episode buffers.
    The learned value function serves as the baseline, so no separate baseline field
    is needed.
    """
    policy_params: Any
    value_params: Any
    policy_opt_state: Any
    value_opt_state: Any
    episode_observations: Array
    episode_actions: Array
    episode_rewards: Array
    episode_length: int


class A2CAgent(AgentABC):
    """
    A2C (Advantage Actor-Critic) agent implementation.

    Collects complete episodes and updates both policy and value networks using
    GAE (Generalized Advantage Estimation) for advantage computation. Uses separate
    policy and value networks with independent optimizers.
    """

    def __init__(
        self,
        policy: PolicyNetworkABC,
        value_network: ValueNetworkABC,
        observation_space: gym.Space,
        action_space: gym.Space,
        max_episode_length: int,
        policy_lr: float = 3e-4,
        value_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantages: bool = False
    ):
        """
        Initialize A2C agent.

        Args:
            policy: Policy network (actor) for action selection
            value_network: Value network (critic) for state value estimation
            observation_space: Environment observation space
            action_space: Environment action space
            max_episode_length: Maximum episode length for pre-allocating buffers
            policy_lr: Learning rate for policy network
            value_lr: Learning rate for value network
            gamma: Discount factor for future rewards
            gae_lambda: GAE lambda parameter (0=pure TD, 1=Monte Carlo)
            normalize_advantages: Whether to normalize advantages by std
        """
        if not (0 <= gamma <= 1):
            raise ValueError(f"gamma must be in [0,1], got {gamma}")
        if not (0 <= gae_lambda <= 1):
            raise ValueError(f"gae_lambda must be in [0,1], got {gae_lambda}")
        if policy_lr <= 0:
            raise ValueError(f"policy_lr must be positive, got {policy_lr}")
        if value_lr <= 0:
            raise ValueError(f"value_lr must be positive, got {value_lr}")
        if max_episode_length <= 0:
            raise ValueError(f"max_episode_length must be positive, got {max_episode_length}")

        self.policy = policy
        self.value_network = value_network
        self.observation_space = observation_space
        self.action_space = action_space
        self.max_episode_length = max_episode_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

        self.policy_optimizer = optax.adam(policy_lr)
        self.value_optimizer = optax.adam(value_lr)

    def init_state(self, key: Array) -> A2CState:
        """
        Create initial agent state.

        Args:
            key: JAX random key for parameter initialization

        Returns:
            Initial A2CState with randomly initialized parameters
        """
        policy_key, value_key = jax.random.split(key)

        policy_params = self.policy.init_params(
            policy_key, self.observation_space, self.action_space
        )
        value_params = self.value_network.init_params(
            value_key, self.observation_space
        )

        policy_opt_state = self.policy_optimizer.init(policy_params)
        value_opt_state = self.value_optimizer.init(value_params)

        obs_shape = self.observation_space.shape
        action_shape = (1,)

        episode_observations = jnp.zeros((self.max_episode_length, *obs_shape))
        episode_actions = jnp.zeros((self.max_episode_length, *action_shape))
        episode_rewards = jnp.zeros(self.max_episode_length)

        return A2CState(
            policy_params=policy_params,
            value_params=value_params,
            policy_opt_state=policy_opt_state,
            value_opt_state=value_opt_state,
            episode_observations=episode_observations,
            episode_actions=episode_actions,
            episode_rewards=episode_rewards,
            episode_length=0
        )

    def select_action(
        self, state: A2CState, observation: Array, key: Array
    ) -> Tuple[Array, A2CState]:
        """
        Select action using current policy and return new state.

        During training, stores the observation and action in pre-allocated
        buffers for later use in the end-of-episode update.

        Args:
            state: Current agent state
            observation: Current state observation
            key: Random key for stochastic action selection

        Returns:
            Tuple of (action, new_agent_state)
        """
        action = self.policy.sample_action(state.policy_params, observation, key)

        new_state = state._replace(
            episode_observations=state.episode_observations.at[state.episode_length].set(observation),
            episode_actions=state.episode_actions.at[state.episode_length].set(action),
            episode_length=state.episode_length + 1
        )

        return action, new_state

    def update(
        self,
        state: A2CState,
        obs: Array,
        action: Array,
        reward: float,
        next_obs: Array,
        done: bool,
        key: Array
    ) -> Tuple[A2CState, Dict[str, float]]:
        """
        Store reward and update both networks at episode end.

        A2C waits until the episode is complete before updating, because GAE
        needs the full trajectory of rewards and value estimates to compute
        advantages.

        Args:
            state: Current agent state
            obs: Current observation (unused - stored in select_action)
            action: Action taken (unused - stored in select_action)
            reward: Reward received
            next_obs: Next observation (unused)
            done: Whether episode ended
            key: Random key (unused)

        Returns:
            Tuple of (new agent state, metrics dict)
        """
        # Store reward at the current index (episode_length was incremented in select_action)
        reward_idx = state.episode_length - 1
        new_state = state._replace(
            episode_rewards=state.episode_rewards.at[reward_idx].set(reward)
        )

        def update_and_reset(s):
            """Branch: Episode complete - update both networks and reset buffers."""
            updated_policy_params, updated_value_params, \
                updated_policy_opt, updated_value_opt, metrics = self._update_networks(s)

            return s._replace(
                policy_params=updated_policy_params,
                value_params=updated_value_params,
                policy_opt_state=updated_policy_opt,
                value_opt_state=updated_value_opt,
                episode_length=0
            ), metrics

        def continue_episode(s):
            """Branch: Episode continues - return state unchanged with empty metrics."""
            # Must match structure of update_and_reset metrics for jax.lax.cond
            empty_metrics = {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "mean_advantage": 0.0,
                "policy_grad_norm": 0.0,
                "value_grad_norm": 0.0,
                "mean_value": 0.0,
            }
            return s, empty_metrics

        return jax.lax.cond(done, update_and_reset, continue_episode, new_state)

    def compute_gae(
        self,
        rewards: Array,
        values: Array,
        gamma: float,
        gae_lambda: float,
        mask: Array,
        episode_length: int
    ) -> Tuple[Array, Array]:
        """
        Compute Generalized Advantage Estimation (GAE).

        GAE blends TD errors across multiple timesteps:
            A_t = delta_t + (gamma * lambda) * delta_{t+1} + ...
        where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t).

        At lambda=0 this reduces to one-step TD: A_t = delta_t.
        At lambda=1 this reduces to Monte Carlo: A_t = G_t - V(s_t).

        Uses masking to handle variable-length episodes in pre-allocated buffers.

        Args:
            rewards: Pre-allocated rewards array (includes padding)
            values: Value estimates for all observations (includes padding)
            gamma: Discount factor
            gae_lambda: GAE lambda (0=pure TD, 1=Monte Carlo)
            mask: Boolean mask for valid timesteps
            episode_length: Number of valid steps

        Returns:
            Tuple of (advantages, returns) where returns = advantages + values.
            Padding positions are 0 in advantages.
        """
        # Compute V(s_{t+1}) for each timestep.
        # For the last valid step: next_value = 0 (terminal state).
        # For padding positions: next_value = 0.
        shifted_values = jnp.roll(values, -1)
        not_last_step = jnp.arange(self.max_episode_length) < (episode_length - 1)
        next_values = jnp.where(not_last_step, shifted_values, 0.0)

        # TD residuals: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        # Padding positions are zeroed so they don't contribute to GAE accumulation.
        deltas = jnp.where(mask, rewards + gamma * next_values - values, 0.0)

        # Reverse scan for GAE: A_t = delta_t + gamma * lambda * A_{t+1}
        def gae_step(carry, delta_and_mask):
            delta, is_valid = delta_and_mask
            advantage = jnp.where(is_valid, delta + gamma * gae_lambda * carry, 0.0)
            return advantage, advantage

        deltas_masked = jnp.stack([deltas, mask], axis=1)
        _, advantages = jax.lax.scan(
            gae_step,
            0.0,
            deltas_masked[::-1]
        )
        advantages = advantages[::-1]  # Reverse back to chronological order

        # Returns are GAE targets for value function training (A_t + V(s_t))
        returns = advantages + values

        return advantages, returns

    def _update_networks(
        self, state: A2CState
    ) -> Tuple[Any, Any, Any, Any, Dict[str, float]]:
        """
        Update both policy and value network parameters.

        Computes GAE advantages, then performs separate gradient updates for:
        - Policy: minimize -sum(log_prob * advantage) / episode_length
        - Value: minimize sum((V(s) - returns)^2) / episode_length

        Args:
            state: Current agent state

        Returns:
            Tuple of (policy_params, value_params, policy_opt_state, value_opt_state, metrics)
        """
        mask = jnp.arange(self.max_episode_length) < state.episode_length

        observations = state.episode_observations
        actions = state.episode_actions
        rewards = state.episode_rewards

        # Compute V(s_t) for every stored observation (includes padding).
        # Padding positions hold V(zeros), which will be masked out in the loss.
        values = jax.vmap(self.value_network.forward, in_axes=(None, 0))(
            state.value_params, observations
        )

        # Compute GAE advantages and value function targets
        advantages, returns = self.compute_gae(
            rewards, values, self.gamma, self.gae_lambda, mask, state.episode_length
        )

        # Optional full z-score normalization: (advantages - mean) / std.
        # A2C can safely subtract the mean because the baseline is a learned
        # value function (not a scalar that would cancel with mean subtraction).
        # Squared diff and mean-centered advantages must both be re-masked because
        # the mean is non-zero at padding positions after subtraction.
        if self.normalize_advantages:
            mean_adv = jnp.sum(advantages) / state.episode_length
            squared_diff = jnp.where(mask, (advantages - mean_adv) ** 2, 0.0)
            variance = jnp.sum(squared_diff) / state.episode_length
            std = jnp.sqrt(variance)
            std_clamped = jnp.maximum(std, 1e-3)
            advantages = jnp.where(mask, (advantages - mean_adv) / std_clamped, 0.0)

        # --- Policy update ---
        def policy_loss(params):
            log_probs = jax.vmap(
                lambda obs, act: self.policy.get_log_prob(params, obs, act)
            )(observations, actions)
            weighted_log_probs = log_probs * advantages
            return -jnp.sum(weighted_log_probs) / state.episode_length

        policy_loss_value, policy_grads = jax.value_and_grad(policy_loss)(state.policy_params)
        policy_updates, new_policy_opt_state = self.policy_optimizer.update(
            policy_grads, state.policy_opt_state
        )
        new_policy_params = optax.apply_updates(state.policy_params, policy_updates)

        # --- Value update ---
        # stop_gradient on returns: although jax.grad only differentiates through
        # the function argument, this makes the intent explicit and protects against
        # refactoring that might move returns computation inside the loss function.
        stopped_returns = jax.lax.stop_gradient(returns)

        def value_loss(params):
            pred_values = jax.vmap(self.value_network.forward, in_axes=(None, 0))(
                params, observations
            )
            squared_errors = (pred_values - stopped_returns) ** 2
            masked_errors = jnp.where(mask, squared_errors, 0.0)
            return jnp.sum(masked_errors) / state.episode_length

        value_loss_value, value_grads = jax.value_and_grad(value_loss)(state.value_params)
        value_updates, new_value_opt_state = self.value_optimizer.update(
            value_grads, state.value_opt_state
        )
        new_value_params = optax.apply_updates(state.value_params, value_updates)

        # --- Metrics ---
        policy_grad_norm = optax.global_norm(policy_grads)
        value_grad_norm = optax.global_norm(value_grads)
        mean_advantage = jnp.sum(advantages) / state.episode_length
        mean_value = jnp.sum(values * mask) / state.episode_length

        metrics = {
            "policy_loss": policy_loss_value,
            "value_loss": value_loss_value,
            "mean_advantage": mean_advantage,
            "policy_grad_norm": policy_grad_norm,
            "value_grad_norm": value_grad_norm,
            "mean_value": mean_value,
        }

        return new_policy_params, new_value_params, new_policy_opt_state, new_value_opt_state, metrics
