from agent.ddqn import DDQNAgent
from agent.dueling_dqn import DuelingDQNAgent

# Initialize environment and agent
env_name = "MountainCar-v0"  # Acrobot-v1, CartPole-v1, MountainCar-v0
agent_class = DDQNAgent  # DDQNAgent, DuelingDQNAgent
n_episodes = 1000               # Number of episodes
target_update = 5               # Update target network after every _ episodes
horizon = 200                   # 200 for MountainCar, 500 for rest

agent_args = {
	"buffer_capacity": 20000,
	"hidden_dim": 64,
	"lr": 1e-3,  # 1e-3, 1e-4
	"gamma": 0.99,
	"epsilon_start": 1.0,
	"epsilon_decay": 0.994,
	"epsilon_end": 0.01,
	"tau": 0.01,  # Target network update rate, Use 1 for hard update
	"batch_size": 256  # 64 128 256
}

# Update with hashing-specific arguments
agent_args.update({
	"count_based": False,  # Use count-based exploration
	"k": 64,  # Number of local centroids
	"alpha": 0.1,  # Exploration bonus factor
	"use_per": True,  # Use prioritized experience replay
	"beta": 0.1,  # Exponent for sampling probabilities. Lower value suggests more uniform sampling
})
# if agent_args["count_based"]:
# 	# No epsilon-greedy exploration
# 	agent_args["epsilon_start"] = 0.0
# 	agent_args["epsilon_end"] = 0.0

# Set 5 random seeds
seeds = [0, 1, 2]
