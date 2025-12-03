# RL Workshop

A personal reinforcement learning workshop designed for rapid experimentation with maximum flexibility.

## Philosophy

This is a **learning and experimentation workshop**, not a production library. The goal is to minimize friction when trying new RL algorithms, network architectures, and environment configurations while maintaining complete flexibility to explore novel ideas.

## Design Priorities

All design decisions in this codebase are guided by these priorities (in order):

### Priority 1: Maximal Modularity & Plug-and-Playability
The highest priority. You should be able to swap out agents, network architectures, environments, and any other component with minimal code changes. Experimentation should be fast and frictionless.

### Priority 2: JAX Idiomaticity
Leverage JAX's strengths: functional programming, JIT compilation, vectorization, and automatic differentiation. Avoid patterns that work against JAX's design.

### Priority 3: Performance
Ensure code is as performant as possible by using JIT compilation wherever appropriate. Optimize the critical path (action selection, gradient computation) while maintaining the first two priorities.

### Priority 4: Excellent Software Engineering Practices
Well-structured code, clear interfaces, good separation of concerns, type hints, comprehensive tests.

### Priority 5: Elegance & Minimalism
Simple, concise code without sacrificing clarity. Avoid over-engineering and premature optimization (YAGNI principle).

**Note:** When Priority 4 and 5 conflict, Priority 5 wins (favor simple over "proper" when appropriate).

## Core Architecture

The codebase is organized around composable components that interact during training:

```
┌─────────────┐
│ Experiment  │  Configures and launches training
└──────┬──────┘
       │
       ├──────> Trainer ────────> coordinates training loop
       │           │
       │           ├──────> Agent (e.g., REINFORCE)
       │           │           └──> Network (policy)
       │           │                   └──> Backbone + Head
       │           │
       │           └──────> Environment (e.g., CartPole)
       │
       └──────> Tracker (optional) ──> logs metrics and videos
```

During each training step:
1. **Agent** selects actions using its **Network** (composed from **Backbone** + **Head**)
2. **Environment** returns observations and rewards
3. **Agent** updates its network parameters based on experience
4. **Tracker** logs progress (optional)

Components are designed to be independently swappable - change the environment, agent, or network architecture without touching other parts.

## Structure

```
rl-workshop/
├── README.md                  # This file - global philosophy and architecture
├── framework/                 # Core RL components
│   ├── README.md             # Framework-level design decisions
│   ├── agents/               # RL algorithms (REINFORCE, etc.)
│   ├── networks/             # Neural network architectures
│   │   ├── README.md        # Networks subsystem design
│   │   ├── backbones/       # Feature extraction (MLP, CNN, etc.)
│   │   └── policy/          # Policy-specific heads and networks
│   ├── environments/         # Environment wrappers
│   ├── trainer.py           # Training loop orchestration
│   ├── tracking.py          # Metrics and visualization
│   └── utils.py             # Shared utilities
├── experiments/              # Training scripts for specific env/agent combinations
├── tests/                    # Unit tests
└── results/                  # Training outputs (videos, plots, logs)
```

## Documentation Structure

This codebase follows a three-level documentation framework:

- **Level 1 (Inline Comments):** Explain non-obvious code mechanisms
- **Level 2 (Docstrings):** Explain what each class/method does and how to use it
- **Level 3 (READMEs):** Explain why the codebase is designed this way

You're currently reading Level 3 documentation. For deeper understanding:
- See `framework/README.md` for framework-level design decisions
- See `framework/networks/README.md` for the networks subsystem design
- See docstrings for usage information on specific components

## Quick Start

### Installation
```bash
pip install -e .
```

### Run an Experiment
```bash
python experiments/cartpole_reinforce.py
```

See `experiments/cartpole_reinforce.py` for an example of how to compose agents, networks, and environments.

### Running Tests
```bash
pytest tests/
```
