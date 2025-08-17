"""
CartPole experiment using REINFORCE agent.

Trains a REINFORCE agent on the CartPole-v1 environment
using an MLP policy with a discrete action head.
"""

import jax
from framework import (
    REINFORCEAgent,
    ComposedPolicy,
    MLPBackbone,
    DiscreteHead,
    CartPoleEnv,
    Trainer,
    Tracker
)


def main():
    """Run the CartPole REINFORCE experiment."""
    # Configuration
    seed = 42
    num_episodes = 500
    learning_rate = 1e-3
    gamma = 0.99
    hidden_dims = [64, 64]
    backbone_output_dim = 32
    
    # Initialize environment
    env = CartPoleEnv(seed=seed)
    
    # Create policy: MLP backbone + discrete head
    backbone = MLPBackbone(hidden_dims=hidden_dims, output_dim=backbone_output_dim)
    head = DiscreteHead(input_dim=backbone_output_dim)
    policy = ComposedPolicy(backbone, head)
    
    # Create agent
    agent = REINFORCEAgent(
        policy=policy,
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=learning_rate,
        gamma=gamma,
        seed=seed
    )
    
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
    print(f"Policy architecture: MLP({hidden_dims}) -> {backbone_output_dim} -> DiscreteHead(2)")
    print()
    
    # Training loop
    trainer.train(num_episodes)
    
    # Final results
    tracker.log_final(success_threshold=195.0)
    
    # Generate plot
    tracker.plot()


if __name__ == "__main__":
    main()