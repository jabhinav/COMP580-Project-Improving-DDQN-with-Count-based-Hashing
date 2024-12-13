import gymnasium as gym
from collections import defaultdict
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


def train_rl_agent(env_name, agent_class, agent_args, n_episodes=500, horizon=200, update_target_every=10, results_dir=None):
	"""
	Train an RL agent in a specified Gym environment.

	Args:
		env_name (str): Name of the Gym environment.
		agent_class (class): The RL agent class (e.g., DuelingDQNAgentWithExploration).
		agent_args (dict): Arguments to initialize the agent.
		n_episodes (int): Number of training episodes.
		horizon (int): Maximum number of steps per episode.
		update_target_every (int): Frequency of target network updates.
		results_dir (str): Directory to save results.
	Returns:
		dict: Training logs containing rewards, steps, and loss for each episode.
	"""
	# Initialize environment
	env = gym.make(env_name, render_mode="rgb_array")
	env = RecordVideo(env, video_folder=results_dir, name_prefix="training",
					  episode_trigger=lambda x: x % (n_episodes//5) == 0)
	env = RecordEpisodeStatistics(env)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.n
	
	# Initialize agent
	agent_args.update({"state_dim": state_dim, "action_dim": action_dim})
	agent = agent_class(**agent_args)
	
	# Logs for tracking performance
	logs = defaultdict(list)
	
	for episode in range(n_episodes):
		state, info = env.reset()
		episode_reward = 0
		episode_steps = 0
		episode_loss = 0
		
		for t in range(horizon):
			# Select action
			action = agent.act(state)
			
			# Step in the environment
			next_state, reward, terminated, truncated, info = env.step(action)
			episode_reward += reward
			
			# Store transition in buffer
			agent.buffer.add((state, action, reward, next_state, int(terminated)))
			state = next_state
			episode_steps += 1
			
			# Learn from experience
			if len(agent.buffer) >= agent_args.get("batch_size", 32):
				loss = agent.learn(agent_args.get("batch_size", 32))
				episode_loss += loss
			
			if terminated or truncated:
				break
				
		# Decay epsilon
		agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)
		
		# # Update target network periodically
		# if episode % update_target_every == 0:
		# 	agent.update_target_network()
		
		# Log results
		logs["epsilon"].append(agent.epsilon)
		logs["episode"].append(episode)
		logs["rewards"].append(episode_reward)
		logs["steps"].append(episode_steps)
		logs["loss"].append(episode_loss / max(1, episode_steps))  # Average loss per step
		
		# Display progress
		print(
			f"Episode {episode + 1}/{n_episodes} | Reward: {episode_reward:.2f} | Steps: {episode_steps} | Loss: {episode_loss:.4f} | Epsilon: {agent.epsilon:2f}"
		)
	
	env.close()
	return logs
