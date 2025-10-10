# Agents Design Documentation

This document explains the design decisions and architectural choices for the agents subsystem.

---

## Core Architectural Decisions

### Agent State Management (Functional Style)

All agents use immutable `NamedTuple` state objects that are passed through methods and returned with updates. This functional approach:
- Aligns with JAX's functional programming paradigm
- Makes state changes explicit and traceable
- Enables JIT compilation of update logic
- Prevents subtle bugs from mutable state

Example:
```python
new_state = agent.select_action(state, observation, key)
updated_state, metrics = agent.update(state, obs, action, reward, next_obs, done, key)
```

---

## Implementation Conventions

### REINFORCE: Advantage Normalization Design

**Decision:** When `normalize_advantages=True`, we normalize by **std only**, not full z-score (mean + std).

**Why not full z-score normalization?**

Standard z-score normalization is: `(x - mean(x)) / std(x)`

However, REINFORCE uses a **scalar baseline** (moving average of returns):
```python
advantages = returns - scalar_baseline
```

If we then subtract the mean:
```python
advantages = (returns - scalar_baseline) - mean(returns - scalar_baseline)
           = returns - scalar_baseline - mean(returns) + scalar_baseline
           = returns - mean(returns)  # Baseline cancels!
```

The scalar baseline would become completely useless. We'd lose its cross-episode learning benefits.

**Our approach:** Std-only normalization
```python
if normalize_advantages:
    std_clamped = max(std(advantages), 1e-3)  # Clamp to prevent division by ~0
    advantages = advantages / std_clamped
```

This preserves the scalar baseline's variance reduction while adding adaptive gradient scaling.

**Tradeoff analysis:**

- **No normalization:** Theoretically pure REINFORCE with baseline. Unbiased but higher variance. Good for understanding the algorithm and environments with naturally low variance.
- **Std-only normalization:** Small bias from std division (O(1/T) for T-step episodes), but preserves the scalar baseline. Reduces variance while keeping cross-episode learning benefits. Good balance for most cases.
- **Full z-score normalization:** Also has small bias from std division, achieves lowest variance, but completely cancels the scalar baseline (mean subtraction removes it). Only makes sense without a separate baseline.

**Our choice:** Default to no normalization (pure algorithm), offer std-only as configurable option for empirical performance.
