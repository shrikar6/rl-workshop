# Environment Design

## Core Design Decisions

### 1. Explicit Environment Wrappers Over Abstraction

**What:** Each Gymnasium environment (CartPole, Acrobot, LunarLander) has its own standalone wrapper class, even though they share ~95% identical code.

**Why:** We don't yet know if future environments will follow this pattern. Atari environments might need frame stacking, continuous control might handle actions differently, and custom environments might not wrap Gymnasium at all. Creating an intermediate `GymnasiumEnv` base class now would be premature abstraction (YAGNI). The ~150 lines of duplication is minimal, stable, and easy to customize per-environment if needed.
