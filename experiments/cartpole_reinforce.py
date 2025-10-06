"""
CartPole experiment using REINFORCE agent.

Trains a REINFORCE agent on the CartPole-v1 environment
using an MLP policy with a discrete action head.
"""

import jax
from framework import (
    REINFORCEAgent,
    ComposedPolicyNetwork,
    MLPBackbone,
    DiscretePolicyHead,
    CartPoleEnv,
    Trainer,
    Tracker
)


def main():
    """Run the CartPole REINFORCE experiment."""
    # Configuration
    seed = 1
    num_episodes = 500
    learning_rate = 1e-3
    gamma = 0.99
    baseline_alpha = 0.01
    hidden_dims = [64]
    backbone_output_dim = 32
    
    # Initialize environment
    env = CartPoleEnv(seed=seed)
    
    # Create policy: MLP backbone + discrete head
    backbone = MLPBackbone(hidden_dims=hidden_dims, output_dim=backbone_output_dim)
    head = DiscretePolicyHead(input_dim=backbone_output_dim)
    policy = ComposedPolicyNetwork(backbone, head)
    
    # Create agent
    agent = REINFORCEAgent(
        policy=policy,
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=learning_rate,
        gamma=gamma,
        baseline_alpha=baseline_alpha
    )

    # Initialize agent state
    agent_key = jax.random.PRNGKey(seed)
    agent_state = agent.init_state(agent_key)

    # Initialize tracker with video recording
    tracker = Tracker(
        log_interval=10,
        window=10,
        video_interval=num_episodes // 10,  # Record video every 10% of training
        experiment_name="cartpole_reinforce"
    )

    # Create trainer with integrated tracker
    trainer = Trainer(environment=env, agent=agent, seed=seed, tracker=tracker)

    # Log experiment configuration
    print("Starting CartPole REINFORCE experiment")
    print(f"Episodes: {num_episodes}")
    print(f"Learning rate: {learning_rate}")
    print(f"Discount factor: {gamma}")
    print(f"Baseline alpha: {baseline_alpha}")
    print(f"Policy architecture: MLP({hidden_dims}) -> {backbone_output_dim} -> DiscretePolicyHead(2)")
    print()

    # Training loop
    final_state, final_key = trainer.train(agent_state, num_episodes)
    
    # Final results
    tracker.log_final(metric="return", success_threshold=450.0, window=num_episodes//10)
    
    # Generate plot
    tracker.plot()
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()