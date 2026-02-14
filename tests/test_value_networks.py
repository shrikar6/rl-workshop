"""
Tests for value network implementations.
"""

import pytest
import jax
import jax.numpy as jnp
from framework import MLPBackbone, ScalarValueHead, ComposedValueNetwork


class TestScalarValueHead:
    """Tests for ScalarValueHead implementation."""

    def test_head_creation(self):
        """Test scalar value head creation."""
        head = ScalarValueHead(input_dim=16)
        assert head.input_dim == 16

    def test_param_initialization(self, scalar_value_head, random_key):
        """Test parameter initialization shapes."""
        params = scalar_value_head.init_params(random_key)

        w, b = params
        assert w.shape == (16,)
        assert b.shape == ()

    def test_forward_returns_scalar(self, scalar_value_head, random_key):
        """Test that forward pass returns a true scalar."""
        params = scalar_value_head.init_params(random_key)
        features = jnp.ones(16)

        value = scalar_value_head.forward(params, features)

        assert value.shape == ()
        assert jnp.isfinite(value)

    def test_forward_deterministic(self, scalar_value_head, random_key):
        """Test that forward pass is deterministic."""
        params = scalar_value_head.init_params(random_key)
        features = jnp.array([0.1, -0.2, 0.5, 0.0, 0.3, -0.1, 0.2, 0.4,
                             -0.3, 0.1, 0.0, -0.4, 0.2, 0.1, -0.2, 0.3])

        value1 = scalar_value_head.forward(params, features)
        value2 = scalar_value_head.forward(params, features)

        assert jnp.array_equal(value1, value2)


class TestComposedValueNetwork:
    """Tests for ComposedValueNetwork implementation."""

    def test_creation(self, mlp_backbone, scalar_value_head):
        """Test composed value network creation."""
        network = ComposedValueNetwork(mlp_backbone, scalar_value_head)

        assert network.backbone == mlp_backbone
        assert network.head == scalar_value_head

    def test_dimension_mismatch(self):
        """Test that mismatched backbone/head dimensions are caught."""
        backbone = MLPBackbone([64], output_dim=32)
        head = ScalarValueHead(input_dim=16)

        with pytest.raises(ValueError, match="must match"):
            ComposedValueNetwork(backbone, head)

    def test_param_initialization(self, composed_value_network, cartpole_env, random_key):
        """Test value network parameter initialization."""
        params = composed_value_network.init_params(
            random_key, cartpole_env.observation_space
        )

        backbone_params, head_params = params

        # Backbone params: 3 layers (4->64, 64->32, 32->16)
        assert len(backbone_params) == 3

        # Head params: weight vector and scalar bias
        w, b = head_params
        assert w.shape == (16,)
        assert b.shape == ()

    def test_forward_returns_scalar(self, composed_value_network, value_params, sample_observation):
        """Test that full network returns a scalar value."""
        value = composed_value_network.forward(value_params, sample_observation)

        assert value.shape == ()
        assert jnp.isfinite(value)

    def test_forward_deterministic(self, composed_value_network, value_params, sample_observation):
        """Test that value network is deterministic."""
        value1 = composed_value_network.forward(value_params, sample_observation)
        value2 = composed_value_network.forward(value_params, sample_observation)

        assert jnp.array_equal(value1, value2)
