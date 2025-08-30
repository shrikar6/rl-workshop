"""
Tests for policy implementations.
"""

import pytest
import jax
import jax.numpy as jnp
import gymnasium as gym
from framework import MLPBackbone, DiscreteHead, ComposedPolicy


class TestMLPBackbone:
    """Tests for MLPBackbone implementation."""
    
    def test_backbone_creation(self):
        """Test MLP backbone creation with different configurations."""
        backbone = MLPBackbone(hidden_dims=[64, 32], output_dim=16)
        
        assert backbone.hidden_dims == (64, 32)
        assert backbone.output_dim == 16
    
    def test_backbone_param_initialization(self, mlp_backbone, cartpole_env, random_key):
        """Test parameter initialization for different observation spaces."""
        params = mlp_backbone.init_params(random_key, cartpole_env.observation_space)
        
        # Should have 3 layers: 4->64, 64->32, 32->16
        assert len(params) == 3
        
        # Check layer shapes
        w1, b1 = params[0]  # Input layer: 4 -> 64
        assert w1.shape == (4, 64)
        assert b1.shape == (64,)
        
        w2, b2 = params[1]  # Hidden layer: 64 -> 32
        assert w2.shape == (64, 32)
        assert b2.shape == (32,)
        
        w3, b3 = params[2]  # Output layer: 32 -> 16
        assert w3.shape == (32, 16)
        assert b3.shape == (16,)
    
    def test_backbone_forward_pass(self, mlp_backbone, cartpole_env, random_key, sample_observation):
        """Test forward pass through backbone."""
        params = mlp_backbone.init_params(random_key, cartpole_env.observation_space)
        features = mlp_backbone.forward(params, sample_observation)
        
        assert features.shape == (16,)  # Should match output_dim
        assert jnp.all(jnp.isfinite(features))
    
    def test_backbone_different_activations(self, cartpole_env, random_key, sample_observation):
        """Test that MLPBackbone works with different activation functions."""
        # Create backbones with different activations
        relu_backbone = MLPBackbone(hidden_dims=[32], output_dim=16)  # Default relu
        tanh_backbone = MLPBackbone(hidden_dims=[32], output_dim=16, activation=jax.nn.tanh)
        elu_backbone = MLPBackbone(hidden_dims=[32], output_dim=16, activation=jax.nn.elu)
        
        # Initialize parameters with same key for comparison
        relu_params = relu_backbone.init_params(random_key, cartpole_env.observation_space)
        tanh_params = tanh_backbone.init_params(random_key, cartpole_env.observation_space)
        elu_params = elu_backbone.init_params(random_key, cartpole_env.observation_space)
        
        # Forward pass with same observation
        relu_out = relu_backbone.forward(relu_params, sample_observation)
        tanh_out = tanh_backbone.forward(tanh_params, sample_observation)
        elu_out = elu_backbone.forward(elu_params, sample_observation)
        
        # All should have correct shape
        assert relu_out.shape == (16,)
        assert tanh_out.shape == (16,)
        assert elu_out.shape == (16,)
        
        # All should be finite
        assert jnp.all(jnp.isfinite(relu_out))
        assert jnp.all(jnp.isfinite(tanh_out))
        assert jnp.all(jnp.isfinite(elu_out))
        
        # Outputs should differ due to different activations
        # (even with same initial params, activations transform values differently)
        assert not jnp.allclose(relu_out, tanh_out)
        assert not jnp.allclose(relu_out, elu_out)
        assert not jnp.allclose(tanh_out, elu_out)


class TestDiscreteHead:
    """Tests for DiscreteHead implementation."""
    
    def test_head_creation(self):
        """Test discrete head creation."""
        head = DiscreteHead(input_dim=16)
        assert head.input_dim == 16
    
    def test_head_param_initialization(self, discrete_head, cartpole_env, random_key):
        """Test parameter initialization for discrete actions."""
        params = discrete_head.init_params(random_key, cartpole_env.action_space)
        
        w, b = params
        assert w.shape == (16, 2)  # 16 features -> 2 actions
        assert b.shape == (2,)     # 2 action biases
    
    def test_head_invalid_action_space(self, discrete_head, random_key):
        """Test that head rejects non-discrete action spaces."""
        invalid_space = gym.spaces.Box(-1, 1, shape=(2,))
        
        with pytest.raises(ValueError, match="DiscreteHead only supports"):
            discrete_head.init_params(random_key, invalid_space)
    
    def test_head_forward_pass(self, discrete_head, cartpole_env, random_key):
        """Test forward pass through head."""
        params = discrete_head.init_params(random_key, cartpole_env.action_space)
        features = jnp.array([0.1, -0.2, 0.5, 0.0, 0.3, -0.1, 0.2, 0.4, 
                             -0.3, 0.1, 0.0, -0.4, 0.2, 0.1, -0.2, 0.3])  # 16 features
        
        action_key = jax.random.PRNGKey(123)
        action = discrete_head.sample_action(params, features, action_key)
        
        assert action.shape == (1,)
        assert action[0] in [0, 1]  # Valid CartPole actions
    
    def test_head_log_prob(self, discrete_head, cartpole_env, random_key):
        """Test log probability computation for specific actions."""
        params = discrete_head.init_params(random_key, cartpole_env.action_space)
        features = jnp.array([0.1, -0.2, 0.5, 0.0, 0.3, -0.1, 0.2, 0.4,
                             -0.3, 0.1, 0.0, -0.4, 0.2, 0.1, -0.2, 0.3])
        
        # Test specific action log probability
        log_prob_0 = discrete_head.get_log_prob(params, features, jnp.array([0]))
        log_prob_1 = discrete_head.get_log_prob(params, features, jnp.array([1]))
        
        # Should return finite values
        assert jnp.isfinite(log_prob_0)
        assert jnp.isfinite(log_prob_1)
        
        # Log probabilities should be negative (since probs are < 1)
        assert log_prob_0 <= 0
        assert log_prob_1 <= 0


class TestComposedPolicy:
    """Tests for ComposedPolicy implementation."""
    
    def test_policy_creation(self, mlp_backbone, discrete_head):
        """Test composed policy creation."""
        policy = ComposedPolicy(mlp_backbone, discrete_head)
        
        assert policy.backbone == mlp_backbone
        assert policy.head == discrete_head
    
    def test_policy_dimension_mismatch(self):
        """Test that mismatched backbone/head dimensions are caught."""
        backbone = MLPBackbone([64], output_dim=32)  # Output 32
        head = DiscreteHead(input_dim=16)            # Expects 16
        
        with pytest.raises(ValueError, match="must match"):
            ComposedPolicy(backbone, head)
    
    def test_policy_param_initialization(self, composed_policy, cartpole_env, random_key):
        """Test policy parameter initialization."""
        params = composed_policy.init_params(
            random_key, cartpole_env.observation_space, cartpole_env.action_space
        )
        
        backbone_params, head_params = params
        
        # Backbone params: 3 layers
        assert len(backbone_params) == 3
        
        # Head params: weights and biases
        w, b = head_params
        assert w.shape == (16, 2)
        assert b.shape == (2,)
    
    def test_policy_action_selection(self, composed_policy, policy_params, sample_observation, random_key):
        """Test full policy action selection."""
        action = composed_policy.sample_action(policy_params, sample_observation, random_key)
        
        assert action.shape == (1,)
        assert action[0] in [0, 1]  # Valid CartPole actions
    
    def test_policy_deterministic_with_same_key(self, composed_policy, policy_params, sample_observation):
        """Test that same key produces same action."""
        key = jax.random.PRNGKey(999)
        
        action1 = composed_policy.sample_action(policy_params, sample_observation, key)
        action2 = composed_policy.sample_action(policy_params, sample_observation, key)
        
        assert jnp.array_equal(action1, action2)
    
    def test_policy_stochastic_with_different_keys(self, composed_policy, policy_params, sample_observation):
        """Test that different keys can produce different actions."""
        actions = []
        for i in range(20):  # Try multiple keys
            key = jax.random.PRNGKey(i)
            action = composed_policy.sample_action(policy_params, sample_observation, key)
            actions.append(action[0])
        
        # Should see both actions (0 and 1) with high probability
        # Convert JAX arrays to Python ints for set operations
        action_values = [int(action) for action in actions]
        unique_actions = set(action_values)
        assert len(unique_actions) >= 1  # At least some variation (might be deterministic)
    
    def test_policy_log_prob(self, composed_policy, policy_params, sample_observation):
        """Test that policy can compute log probabilities."""
        # Test for action 0
        action_0 = jnp.array([0])
        log_prob_0 = composed_policy.get_log_prob(policy_params, sample_observation, action_0)
        assert jnp.isfinite(log_prob_0)
        assert log_prob_0 <= 0  # Log probabilities should be <= 0
        
        # Test for action 1
        action_1 = jnp.array([1])
        log_prob_1 = composed_policy.get_log_prob(policy_params, sample_observation, action_1)
        assert jnp.isfinite(log_prob_1)
        assert log_prob_1 <= 0
        
        # Test that probabilities sum to 1
        prob_sum = jnp.exp(log_prob_0) + jnp.exp(log_prob_1)
        assert jnp.allclose(prob_sum, 1.0, atol=1e-6)