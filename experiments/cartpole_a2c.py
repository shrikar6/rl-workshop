"""
CartPole experiment using A2C agent.

Trains an A2C (Advantage Actor-Critic) agent on the CartPole-v1 environment
using separate MLP policy and value networks. Policy uses a discrete action
head; value network uses a scalar value head.
"""

import jax
from framework import (
    A2CAgent,
    ComposedPolicyNetwork,
    ComposedValueNetwork,
    MLPBackbone,
    DiscretePolicyHead,
    ScalarValueHead,
    CartPoleEnv,
    Trainer,
    Tracker
)


def main():
    """Run the CartPole A2C experiment."""
    # Configuration
    seed = 1
    num_episodes = 500
    policy_lr = 1e-3
    value_lr = 1e-3
    gamma = 0.99
    gae_lambda = 0.95
    hidden_dims = [64]
    backbone_output_dim = 32

    # Initialize environment
    env = CartPoleEnv(seed=seed)

    # Create policy: MLP backbone + discrete head
    policy = ComposedPolicyNetwork(
        backbone=MLPBackbone(hidden_dims=hidden_dims, output_dim=backbone_output_dim),
        head=DiscretePolicyHead(input_dim=backbone_output_dim)
    )

    # Create value network: separate MLP backbone + scalar head
    value_network = ComposedValueNetwork(
        backbone=MLPBackbone(hidden_dims=hidden_dims, output_dim=backbone_output_dim),
        head=ScalarValueHead(input_dim=backbone_output_dim)
    )

    # Create agent
    agent = A2CAgent(
        policy=policy,
        value_network=value_network,
        observation_space=env.observation_space,
        action_space=env.action_space,
        max_episode_length=env.max_episode_length,
        policy_lr=policy_lr,
        value_lr=value_lr,
        gamma=gamma,
        gae_lambda=gae_lambda
    )

    # Initialize agent state
    agent_key = jax.random.PRNGKey(seed)
    agent_state = agent.init_state(agent_key)

    # Initialize tracker with video recording
    tracker = Tracker(
        log_interval=10,
        window=10,
        video_interval=num_episodes // 10,  # Record video every 10% of training
        experiment_name="cartpole_a2c"
    )

    # Create trainer with integrated tracker
    trainer = Trainer(environment=env, agent=agent, tracker=tracker)

    # Log experiment configuration
    print("Starting CartPole A2C experiment")
    print(f"Episodes: {num_episodes}")
    print(f"Policy LR: {policy_lr}")
    print(f"Value LR: {value_lr}")
    print(f"Discount factor: {gamma}")
    print(f"GAE lambda: {gae_lambda}")
    print(f"Policy architecture: MLP({hidden_dims}) -> {backbone_output_dim} -> DiscretePolicyHead(2)")
    print(f"Value architecture:  MLP({hidden_dims}) -> {backbone_output_dim} -> ScalarValueHead")
    print()

    # Training loop
    trainer_key = jax.random.PRNGKey(seed)
    final_state, final_key = trainer.train(agent_state, trainer_key, num_episodes)

    # Final results
    tracker.log_final(metric="return", success_threshold=450.0, window=num_episodes // 10)

    # Generate plot
    tracker.plot()

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
