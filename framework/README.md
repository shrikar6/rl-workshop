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

### 2. Functional State Management

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

### 3. Composition Over Inheritance

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

### 4. JIT Compilation Strategy

**What:** JIT at the highest level possible - the agent level.

**Why:** JIT'ing at the top level allows JAX to trace through the entire call chain and fuse all operations into optimized kernels, maximizing performance (Priority 3).

## Implementation Conventions

### Parameter Initialization: Xavier/Glorot

Use Xavier initialization for weight matrices: `scale = sqrt(2 / (fan_in + fan_out))`

Maintains variance across layers for better gradient flow. More general than He initialization.

### Parameter Structure: Tuples

**What:** Parameters use tuples. Single layers use `(w, b)`, multi-layer networks use lists of tuples `[(w, b), (w, b), ...]`, and composed networks use tuples of component params `(backbone_params, head_params)`.

**Why:** Tuples provide the right balance for our workshop:

- **Elegance (Priority 4):** Minimal syntax with clean unpacking (`w, b = params`)
- **Freedom from conventions (Priority 1):** No coupling to standardized naming - each component freely chooses its structure
- **Functional style (Priority 2):** Aligns with functional programming patterns and JAX treats them as pytrees naturally

**Alternative considered:** Dict-based structures `{"weight": w, "bias": b}` offer self-documenting keys and backward-compatible evolution (can add optional fields via `.get()`). Production libraries like Flax/Haiku use dicts because they need backward compatibility with saved model checkpoints across versions. Our workshop retrains from scratch each experiment and allows breaking changes, so tuple simplicity wins over dict evolvability.

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
- Agents: See `framework/agents/README.md`
- Environments: See `framework/environments/README.md`
