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

### A2C: Separate Policy and Value Networks

**Decision:** A2C uses **two fully independent networks** — one for the policy and one for the value function — each with its own backbone, head, and optimizer. The backbones are not shared.

**Why not share the backbone?**

A shared backbone is the common default in many A2C/PPO implementations because it saves parameters and lets the two tasks co-train feature representations. However, it also introduces gradient interference: policy and value gradients flow through the same parameters with potentially opposing objectives, and tuning their relative weighting (via a value_loss_coef hyperparameter) becomes a new source of fragility.

For this workshop, independent networks are simpler and more explicit:
- No shared-backbone gradient coupling to reason about
- Independent learning rates (`policy_lr`, `value_lr`) without requiring a separate balance coefficient
- Plug-and-play with the existing `ComposedPolicyNetwork`/`ComposedValueNetwork` abstractions — no new "two-headed" network type needed
- Matches the framework's Priority 1 (modularity over optimization)

If parameter count becomes a concern for larger environments, a shared-backbone variant could be added later as an alternative agent without changing this one.

### A2C: GAE over Pure TD or Monte Carlo

**Decision:** A2C uses **Generalized Advantage Estimation (GAE)** with a configurable `gae_lambda` parameter, instead of hardcoding pure one-step TD (λ=0) or pure Monte Carlo (λ=1).

**Why GAE?**

GAE's λ parameter is a bias-variance knob:
- **λ=0** (pure TD): low variance, high bias — only uses one-step lookahead, so biased by the accuracy of V(s_{t+1}).
- **λ=1** (pure Monte Carlo): low bias, high variance — uses the full return, exactly recovering REINFORCE-with-learned-baseline.
- **Intermediate λ** (e.g., 0.95): blends both, typically the best empirical choice.

Exposing this as a parameter has two benefits:
1. **Sanity comparisons:** With `gae_lambda=1.0`, A2C should behave similarly to REINFORCE with a learned baseline (same algorithmic target, different baseline source). This is a free correctness check.
2. **Experimental flexibility:** Sweeping λ across experiments reveals how much the bias-variance tradeoff matters for a given environment.

### A2C: Episode-based Updates

**Decision:** A2C waits for a **full episode** before performing an update, just like REINFORCE. It does not update every step or every N steps.

**Why episode-based?**

This is not the only valid choice. Classical A2C can update continuously (step-by-step) or in fixed-length rollouts (N-step). Episode-based updates were chosen because:
- They match the existing Trainer's per-episode abstraction, so no trainer changes were needed
- They match REINFORCE's update cadence, making A2C a cleaner drop-in comparison
- GAE can be computed exactly (no bootstrap truncation at arbitrary N-step boundaries)

The main downside is that very long episodes delay updates, and environments that never terminate are incompatible. For this workshop's bounded episodic environments (CartPole, Acrobot, LunarLander), that's not a concern.

### Padding-masking Invariant for Variance Computations

**Rule:** When computing variance (or any non-linear reduction like std, mean-of-squares, etc.) over an array that was produced by masked computation against a pre-allocated buffer, you **must re-mask the intermediate quantity**.

**The bug this prevents:**

Both REINFORCE and A2C store episodes in pre-allocated buffers padded with zeros beyond `episode_length`. `compute_gae` and `compute_baseline_and_advantages` both produce `advantages` arrays where padding positions are explicitly 0. Summing `advantages` directly is safe — zeros contribute nothing.

However, when normalizing:
```python
# WRONG:
mean_adv = jnp.sum(advantages) / episode_length  # OK: ignores padding
squared_diff = (advantages - mean_adv) ** 2      # BUG: padding becomes mean_adv^2
variance = jnp.sum(squared_diff) / episode_length  # pollutes with padding
```

At padding positions, `advantages = 0`, so `(0 - mean_adv)^2 = mean_adv^2`, which is non-zero whenever the episode has non-zero mean advantage. This silently inflates the variance by a factor that depends on the ratio of padding to valid data — so results become dependent on `max_episode_length`, which should be irrelevant.

**The fix:**
```python
# CORRECT:
mean_adv = jnp.sum(advantages) / episode_length
squared_diff = jnp.where(mask, (advantages - mean_adv) ** 2, 0.0)  # re-mask
variance = jnp.sum(squared_diff) / episode_length
```

And likewise for any subsequent operation that mean-centers padded data:
```python
# A2C full z-score normalization also re-masks after mean subtraction:
advantages = jnp.where(mask, (advantages - mean_adv) / std_clamped, 0.0)
```

**Testing convention:** For any code that does variance-like computation over padded buffers, write a **padding invariance test** — construct two agents with different `max_episode_length` values, feed the same valid episode data to both, and assert the result is identical. See `tests/test_a2c.py::test_normalization_padding_invariance` and `tests/test_reinforce.py::test_normalization_padding_invariance` for the pattern.

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

**Contrast with A2C:** A2C uses a learned value function as its baseline, not a scalar. Mean subtraction does not cancel a learned baseline, so A2C's `normalize_advantages` uses **full z-score** normalization (the standard A2C/PPO convention). See `A2CAgent._update_networks`.
