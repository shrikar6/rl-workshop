"""
Shared pytest fixtures for the RL framework tests.

This file contains common test fixtures that can be used across multiple test modules.
"""

import pytest
import jax
import jax.numpy as jnp
from framework import (
    CartPoleEnv,
    MLPBackbone,
    DiscreteHead, 
    ComposedPolicy
)


@pytest.fixture
def random_key():
    """Provides a consistent JAX random key for reproducible tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def cartpole_env():
    """Provides a CartPole environment for testing."""
    return CartPoleEnv()


@pytest.fixture
def sample_observation():
    """Provides a sample CartPole observation for testing."""
    return jnp.array([0.1, -0.2, 0.05, 0.3])


@pytest.fixture
def mlp_backbone():
    """Provides an MLP backbone for testing."""
    return MLPBackbone(hidden_dims=[64, 32], output_dim=16)


@pytest.fixture
def discrete_head():
    """Provides a discrete head for testing."""
    return DiscreteHead(input_dim=16)


@pytest.fixture
def composed_policy(mlp_backbone, discrete_head):
    """Provides a composed MLP policy for testing."""
    return ComposedPolicy(mlp_backbone, discrete_head)


@pytest.fixture
def policy_params(composed_policy, cartpole_env, random_key):
    """Provides initialized policy parameters for testing."""
    return composed_policy.init_params(
        random_key, 
        cartpole_env.observation_space, 
        cartpole_env.action_space
    )