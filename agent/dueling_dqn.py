import random
import torch
import torch.nn as nn
from utils.networks import DuelingQNetwork
from utils.buffer import ReplayBuffer, PrioritizedReplayBuffer
from utils.hashing import SimHash, GlobalSimHash


class DuelingDQNAgent:
	def __init__(self, state_dim, action_dim, buffer_capacity=10000, hidden_dim=128,
                 lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500, k=None, alpha=0.01, beta=0.4, tau=0.01,
				 count_based=False, use_per=False, **kwargs):
		self.action_dim = action_dim
		self.gamma = gamma
		self.epsilon = epsilon_start
		self.epsilon_end = epsilon_end
		self.epsilon_decay = epsilon_decay
		self.step_count = 0
		
		# Replay buffer
		if use_per:
			self.buffer = PrioritizedReplayBuffer(buffer_capacity, GlobalSimHash(state_dim, k), beta=beta)
		else:
			self.buffer = ReplayBuffer(buffer_capacity)
		
		# Hashing
		self.hash = SimHash(state_dim, k) if k is not None else None
		self.count_based = count_based
		self.alpha = alpha
		
		self.q_network = DuelingQNetwork(state_dim, action_dim, hidden_dim)
		self.target_network = DuelingQNetwork(state_dim, action_dim, hidden_dim)
		self.tau = tau
		self.target_network.load_state_dict(self.q_network.state_dict())
		self.target_network.eval()
		
		self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
		self.loss_fn = nn.MSELoss()
	
	def act(self, state):
		self.step_count += 1
		self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
		
		if random.random() < self.epsilon:
			return random.randint(0, self.action_dim - 1)
		else:
			state_tensor = torch.FloatTensor(state).unsqueeze(0)
			with torch.no_grad():
				q_values = self.q_network(state_tensor)
			return q_values.argmax().item()
	
	def learn(self, batch_size):
		if len(self.buffer) < batch_size:
			return
		
		# Sample a batch
		states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
		
		# Convert to tensors
		states = torch.FloatTensor(states)
		actions = torch.LongTensor(actions).unsqueeze(1)
		rewards = torch.FloatTensor(rewards).unsqueeze(1)
		next_states = torch.FloatTensor(next_states)
		dones = torch.FloatTensor(dones).unsqueeze(1)
		
		# Compute current Q values
		q_values = self.q_network(states).gather(1, actions)
		# q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
		
		# Compute targets
		with torch.no_grad():
			if self.count_based:
				# Add the counting based parts
				counts = self.hash.count(states)
				# Compute the new rewards
				count_reward = self.alpha / torch.sqrt(counts)
				count_reward = count_reward.unsqueeze(1)
			else:
				count_reward = 0
			rewards = rewards + count_reward
			
			next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
			target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
		
		# Compute loss
		loss = self.loss_fn(q_values, target_q_values)
		
		# Optimize the Q-network
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		
		return loss.detach().item()
		
	def update_target_network(self):
		# Periodically update target network using Polyak averaging
		state_dict = self.tau * self.q_network.state_dict() + (1 - self.tau) * self.target_network.state_dict()
		self.target_network.load_state_dict(state_dict)
		
		# self.target_network.load_state_dict(self.q_network.state_dict())
		
	def __str__(self):
		return "DuelingDQN_Agent"
	
	def __repr__(self):
		return "DuelingDQN_Agent"