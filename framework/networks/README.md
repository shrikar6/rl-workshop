# Networks Design

Design decisions specific to the networks subsystem. For framework-wide patterns (config/params separation, instance methods, etc.), see `framework/README.md`.

## Code Organization

**Directory Structure Principle:** Three-tier hierarchy separating framework concepts, domain abstractions, and concrete implementations.

- **Tier 1 (Framework ABCs):** `networks/base.py` defines fundamental building blocks: `NetworkABC`, `BackboneABC`, `HeadABC`
- **Tier 2 (Domain ABCs):** Domain-specific abstractions like `PolicyNetworkABC`, `PolicyHeadABC` live at the domain level (e.g., `policy/base.py`)
- **Tier 3 (Implementations):** Concrete classes in subdirectories (e.g., `backbones/mlp.py`, `policy/heads/discrete.py`)

This structure clarifies what's a fundamental framework concept versus what's domain-specific, and keeps related domain abstractions together.

## Policy vs Value Network Hierarchies

**What:** Networks are split into two parallel hierarchies, `networks/policy/` and `networks/value/`, each with its own domain ABC pair (`PolicyNetworkABC`/`PolicyHeadABC`, `ValueNetworkABC`/`ValueHeadABC`).

**Why not reuse policy classes for value networks?**

Policy and value networks have genuinely different APIs:

- Policy networks expose `sample_action(params, obs, key)` and `get_log_prob(params, obs, action)`. Value networks only need `forward(params, obs)`.
- `PolicyHeadABC.init_params(key, action_space)` requires the action space to size the output. `ValueHeadABC.init_params(key)` does not — value heads always output a scalar (or vector, for distributional RL) that is independent of the action space.
- `PolicyNetworkABC.init_params(key, obs_space, action_space)` vs `ValueNetworkABC.init_params(key, obs_space)`.

Forcing a shared type would require either leaky optional parameters (`action_space=None` for value heads) or a lowest-common-denominator API that hides the real semantics. Two parallel hierarchies keep each API honest.

**What IS shared:** Backbones. `MLPBackbone` is used by both policy and value networks — feature extraction is task-agnostic and should not be duplicated. This is the mirror-image of the "same backbone, different heads" design philosophy from the next section: policies and values share backbones and diverge at the head.

## Architecture: Backbone/Head Separation

**What:** Networks are composed from two independent pieces:

```
Network = Backbone + Head

Backbone: observation → features
Head: features → output (actions, Q-values, etc.)
```

**Example:**
```python
# Create components
backbone = MLPBackbone(hidden_dims=[64, 32], output_dim=16)
head = DiscretePolicyHead(input_dim=16)

# Compose into network
policy = ComposedPolicyNetwork(backbone, head)

# Easy to swap - try different backbone
backbone_deep = MLPBackbone(hidden_dims=[128, 64, 32], output_dim=16)
policy_deep = ComposedPolicyNetwork(backbone_deep, head)

# Or try different head
head_continuous = ContinuousPolicyHead(input_dim=16)  # (future)
policy_continuous = ComposedPolicyNetwork(backbone, head_continuous)
```

**Why:** Separating feature extraction from task-specific output maximizes reusability. The same backbone can be used for discrete policies, continuous policies, or value functions - just swap the head.

## Component Responsibilities

**Backbone:**
- Input: Raw observations from environment
- Output: Feature representations
- Examples: MLPBackbone, CNNBackbone (future)

**Head:**
- Input: Features from backbone
- Output: Task-specific outputs (action logits, state values, Q-values, etc.)
- Examples: DiscretePolicyHead, ScalarValueHead, ContinuousPolicyHead (future)

**ComposedNetwork (ComposedPolicyNetwork, ComposedValueNetwork):**
- Orchestrates backbone and head
- Validates dimension compatibility (backbone.output_dim must equal head.input_dim)
- Parameters structure: Tuple of (backbone_params, head_params)

## JIT Compilation Strategy

**Pattern:** Network components are NOT individually JIT-compiled. Instead, they are JIT-compiled as part of higher-level agent methods (see `framework/README.md` JIT strategy).

**How it works:**
1. Agent methods (`select_action`, `update`) are JIT-compiled at the agent level
2. When JAX traces these methods, it follows calls through the entire network stack
3. Instance attributes (like `self.activation` in MLPBackbone) are traced and baked into the compiled code
4. The entire computation graph is optimized as a single unit

**Why this is good:**
- Allows JAX to fuse operations across the entire computation (backbone → head → loss)
- Simpler implementation - no need for static helper methods
- Configuration (activation functions, etc.) is immutable after initialization, so tracing it is fine

**Example of what gets JIT'd together:**
```python
# In Trainer, agent methods are JIT-compiled:
self.select_action_jit = jax.jit(agent.select_action)

# When traced, JAX compiles the entire chain:
agent.select_action → policy.sample_action → backbone.forward + head.forward
# All as a single optimized computation
```

## Implementation Patterns

### Creating a New Backbone

```python
class MyBackbone(BackboneABC):
    def __init__(self, output_dim, my_config):
        super().__init__(output_dim)
        self.my_config = my_config  # Immutable after init

    def init_params(self, key, observation_space):
        # Use self.my_config to create params
        return params

    def forward(self, params, observation):
        # Access instance attributes directly - JAX will trace them
        result = some_computation(observation, self.my_config)
        return result
```

### Creating a New Head

```python
class MyPolicyHead(PolicyHeadABC):
    def __init__(self, input_dim, my_config):
        super().__init__(input_dim)
        self.my_config = my_config  # Immutable after init

    def init_params(self, key, action_space):
        # Use self.input_dim and action_space
        return params

    def forward(self, params, features):
        return output


class MyValueHead(ValueHeadABC):
    def __init__(self, input_dim, my_config):
        super().__init__(input_dim)
        self.my_config = my_config

    def init_params(self, key):
        # Value heads do NOT take action_space - value is a function of state only
        return params

    def forward(self, params, features):
        return output
```

**Key principle:** Configuration lives in instance attributes (set once during `__init__`). Parameters live in the `params` argument (updated during training). JAX traces through both.

## Current Implementations

- **Backbones:** MLPBackbone
- **Policy heads:** DiscretePolicyHead
- **Value heads:** ScalarValueHead
- **Networks:** ComposedPolicyNetwork, ComposedValueNetwork

See docstrings for detailed usage.
