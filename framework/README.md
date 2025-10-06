# Framework Design

This document explains the core design decisions that apply across all framework components.

## Core Architectural Decisions

### 1. Separation of Configuration and Parameters

**What:** Network components store architectural configuration (layer sizes, activation functions, etc.) as instance attributes, while learned parameters (weights, biases) are kept separate and passed to methods functionally.

**Example:**
```python
# Configuration stored in instance
backbone = MLPBackbone(
    hidden_dims=[64, 32],
    output_dim=16,
    activation=jax.nn.relu
)

# Parameters initialized separately
params = backbone.init_params(key, observation_space)

# Parameters passed to computation methods
features = backbone.forward(params, observation)
```

**Why:** Configuration defines the architecture (doesn't change during training), while parameters are the learned weights (updated every step). Keeping them separate enables JAX's functional paradigm for params while maintaining OOP convenience for configuration. This makes it easy to swap architectures (just create a new instance) and enables JAX pytree operations on params.

### 2. Instance Methods with JIT Delegation Pattern

**What:** Public API methods are instance methods that can access configuration, and they delegate to static JIT-compiled helpers for performance.

**Example:**
```python
class MLPBackbone:
    def forward(self, params, observation):
        # Instance method - accesses self.activation
        return self._forward_jit(params, observation, self.activation)

    @staticmethod
    @partial(jax.jit, static_argnums=(2,))
    def _forward_jit(params, observation, activation):
        # Pure function - JIT compilable
        # ... computation ...
```

**Why:** Instance methods allow components to access their configuration (like activation functions), enabling flexible, parameterizable behavior. Delegation to static JIT'd helpers ensures JAX gets pure functions for compilation. This pattern follows established JAX libraries like Flax and Haiku.

### 3. Functional State Management

**What:** Agent state is immutable. Methods return new state objects rather than mutating existing state. All randomness is explicit via JAX PRNGKey splitting.

**Example:**
```python
# Agent state is a NamedTuple (immutable)
class REINFORCEState(NamedTuple):
    policy_params: Any
    opt_state: Any
    episode_observations: List[Array]
    # ... other fields

# Methods return new state
def select_action(self, state, observation, key):
    action = self.policy.sample_action(state.policy_params, observation, key)
    new_state = state._replace(
        episode_observations=state.episode_observations + [observation]
    )
    return action, new_state
```

**Why:** Immutable state is easier to reason about, debug, and makes training reproducible. This is core to JAX's functional programming paradigm. Explicit randomness via PRNGKey ensures experiments are fully reproducible.

### 4. Composition Over Inheritance

**What:** Complex components are built by composing simpler components rather than deep inheritance hierarchies. For example, `ComposedPolicyNetwork` composes a `Backbone` + `Head`.

**Example:**
```python
# Compose instead of inherit
policy = ComposedPolicyNetwork(
    backbone=MLPBackbone(hidden_dims=[64, 32], output_dim=16),
    head=DiscretePolicyHead(input_dim=16)
)

# Easy to swap parts
policy_tanh = ComposedPolicyNetwork(
    backbone=MLPBackbone(hidden_dims=[64, 32], output_dim=16, activation=jax.nn.tanh),
    head=DiscretePolicyHead(input_dim=16)
)
```

**Why:** Composition maximizes plug-and-play flexibility. Any backbone can work with any head. Shallow hierarchies are easier to understand than deep inheritance trees.

### 5. JIT Compilation Strategy: Optimize the Math, Keep the Composition Simple

**What:** JIT compilation is applied to computational primitives (the math), but not to orchestration/composition logic (calling and combining components). We accept marginal performance loss in exchange for simplicity and clarity.

**Computational Primitives (JIT'd):**
- Neural network forward passes (matrix multiplies, activations, etc.)
- Sampling operations (categorical sampling, etc.)
- Specific computational kernels (advantage computation, etc.)

**Orchestration/Composition (NOT JIT'd):**
- Calling one component then another
- Combining results from multiple components
- State management and bookkeeping (Python list operations, NamedTuple updates, etc.)

**Example:**
```python
# Computational primitive - JIT'd
class MLPBackbone:
    def forward(self, params, observation):
        return self._forward_jit(params, observation, self.activation)

    @staticmethod
    @partial(jax.jit, static_argnums=(2,))
    def _forward_jit(params, observation, activation):
        # The math: matrix multiplies, activations
        x = observation
        for w, b in params[:-1]:
            x = activation(jnp.dot(x, w) + b)
        w_final, b_final = params[-1]
        return jnp.dot(x, w_final) + b_final

# Orchestration - NOT JIT'd
class ComposedPolicyNetwork:
    def forward(self, params, observation):
        backbone_params, head_params = params
        # Orchestration: call backbone, then call head
        features = self.backbone.forward(backbone_params, observation)
        return self.head.forward(head_params, features)

class REINFORCEAgent:
    def select_action(self, state, observation, key):
        # Orchestration: call policy (already JIT'd) + state updates (Python ops)
        action = self.policy.sample_action(state.policy_params, observation, key)
        new_state = state._replace(
            episode_observations=state.episode_observations + [observation],
            episode_actions=state.episode_actions + [action]
        )
        return action, new_state
```

**Why:** This strategy captures ~95% of the performance gains (by JIT'ing the computational primitives) while keeping the codebase simple and maintainable. Alternative approaches like adding JIT to orchestration layers could squeeze out an additional 1-5% performance, but at the cost of moderate complexity increase, less elegant code, and marginal real-world benefit.

## Implementation Conventions

### Parameter Initialization: Xavier/Glorot

Use Xavier initialization for weight matrices: `scale = sqrt(2 / (fan_in + fan_out))`

Maintains variance across layers for better gradient flow. More general than He initialization.

### Parameter Structure: Tuples

Parameters are stored as tuples: `(w, b)` for a single layer, or lists of tuples for multi-layer networks.

Simple, minimal overhead, works naturally with JAX pytrees.

### Random Key Management

JAX PRNGKeys are explicitly split and passed to functions that need randomness.

```python
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
action = policy.sample_action(params, observation, subkey)
```

JAX requires explicit randomness for reproducibility and parallelization.

### Type Annotations: Flexible Interfaces, Typed Implementations

**What:** Abstract base classes use `Any` for parameter and state types to maintain plug-and-play flexibility. Concrete implementations use specific types (like `MLPParams`, `Tuple[Array, Array]`, or `REINFORCEState`) to provide IDE support and type checking.

**Example:**
```python
# Abstract base class - flexible interface
class BackboneABC(ABC):
    @abstractmethod
    def forward(self, params: Any, observation: Array) -> Array:
        pass

# Concrete implementation - specific types
MLPParams = List[Tuple[Array, Array]]

class MLPBackbone(BackboneABC):
    def forward(self, params: MLPParams, observation: Array) -> Array:
        # IDE knows params structure, provides autocomplete
        pass
```

**Why:** Abstract interfaces use `Any` so different implementations (MLPBackbone, CNNBackbone) can have different parameter structures without type conflicts - this preserves plug-and-play modularity (Priority 1). Concrete implementations use specific types for developer experience - IDE autocomplete, type checking, and discoverability (Priority 3). This balances flexibility at composition boundaries with safety within implementations.

## Subsystem Documentation

For design decisions specific to subsystems:
- Networks: See `framework/networks/README.md`
- Agents: (To be created when patterns stabilize)
- Environments: (To be created when patterns stabilize)
