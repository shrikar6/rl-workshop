"""
LunarLander experiment using REINFORCE agent.

Trains a REINFORCE agent on the LunarLander-v3 environment
using an MLP policy with a discrete action head.
"""

from framework import (
    REINFORCEAgent,
    ComposedPolicy,
    MLPBackbone,
    DiscreteHead,
    LunarLanderEnv,
    Trainer,
    Tracker
)


def main():
    """Run the LunarLander REINFORCE experiment."""
    # Configuration
    seed = 1
    num_episodes = 15000
    learning_rate = 1e-4
    gamma = 0.99
    baseline_alpha = 0.003
    hidden_dims = [128, 128]
    backbone_output_dim = 64
    
    # Initialize environment
    env = LunarLanderEnv(seed=seed)
    
    # Create policy: MLP backbone + discrete head (4 actions for LunarLander)
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
        baseline_alpha=baseline_alpha,
        seed=seed
    )
    
    # Initialize tracker with video recording
    tracker = Tracker(
        log_interval=50, 
        window=50,
        video_interval=num_episodes // 10,  # Record video every 10% of training
        experiment_name="lunarlander_reinforce"
    )
    
    # Create trainer with integrated tracker
    trainer = Trainer(environment=env, agent=agent, seed=seed, tracker=tracker)
    
    # Log experiment configuration
    print("Starting LunarLander REINFORCE experiment")
    print(f"Episodes: {num_episodes}")
    print(f"Learning rate: {learning_rate}")
    print(f"Discount factor: {gamma}")
    print(f"Baseline alpha: {baseline_alpha}")
    print(f"Policy architecture: MLP({hidden_dims}) -> {backbone_output_dim} -> DiscreteHead(4)")
    print()
    
    # Training loop
    trainer.train(num_episodes)
    
    # Final results
    tracker.log_final(metric="return", success_threshold=200.0, window=num_episodes//10)  # LunarLander success is around 200
    
    # Generate plot
    tracker.plot()
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()