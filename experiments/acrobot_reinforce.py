"""
Acrobot experiment using REINFORCE agent.

Trains a REINFORCE agent on the Acrobot-v1 environment
using an MLP policy with a discrete action head.
"""

from framework import (
    REINFORCEAgent,
    ComposedNetwork,
    MLPBackbone,
    DiscretePolicyHead,
    AcrobotEnv,
    Trainer,
    Tracker
)


def main():
    """Run the Acrobot REINFORCE experiment."""
    # Configuration
    seed = 1
    num_episodes = 1000
    learning_rate = 1e-3
    gamma = 0.99
    baseline_alpha = 0.01
    hidden_dims = [64, 64]
    backbone_output_dim = 32

    # Initialize environment
    env = AcrobotEnv(seed=seed)
    
    # Create policy: MLP backbone + discrete head (3 actions for Acrobot)
    backbone = MLPBackbone(hidden_dims=hidden_dims, output_dim=backbone_output_dim)
    head = DiscretePolicyHead(input_dim=backbone_output_dim)
    policy = ComposedNetwork(backbone, head)
    
    # Create agent
    agent = REINFORCEAgent(
        policy=policy,
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=learning_rate,
        gamma=gamma,
        baseline_alpha=baseline_alpha,
        seed=seed
    )
    
    # Initialize tracker with video recording
    tracker = Tracker(
        log_interval=25, 
        window=25,
        video_interval=num_episodes // 10,  # Record video every 10% of training
        experiment_name="acrobot_reinforce"
    )
    
    # Create trainer with integrated tracker
    trainer = Trainer(environment=env, agent=agent, seed=seed, tracker=tracker)
    
    # Log experiment configuration
    print("Starting Acrobot REINFORCE experiment")
    print(f"Episodes: {num_episodes}")
    print(f"Learning rate: {learning_rate}")
    print(f"Discount factor: {gamma}")
    print(f"Baseline alpha: {baseline_alpha}")
    print(f"Policy architecture: MLP({hidden_dims}) -> {backbone_output_dim} -> DiscretePolicyHead(3)")
    print()
    
    # Training loop
    trainer.train(num_episodes)
    
    # Final results
    tracker.log_final(metric="return", success_threshold=-100.0, window=num_episodes//10)  # Acrobot success is around -100 to -120
    
    # Generate plot
    tracker.plot()
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()