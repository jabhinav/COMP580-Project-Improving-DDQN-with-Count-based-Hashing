from collections import deque
import numpy as np
from utils.hashing import GlobalSimHash


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in batch])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, hasher: GlobalSimHash, beta=0.4, delta=1e-5):
        """
        Implements a prioritized replay buffer with hashing-based state visitation count.

        Args:
            capacity (int): Maximum buffer size.
            hasher (StateHasher): Instance of the StateHasher for state visitation counting.
            beta (float): Exponent for sampling probabilities.
            delta (float): Small value to avoid division by zero.
        """
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.beta = beta
        self.delta = delta
        self.hasher = hasher
    
    def add(self, transition):
        """
        Adds a new transition and updates priorities.

        Args:
            transition (tuple): (state, action, reward, next_state, done)
        """
        
        # Update state visitation count
        state = transition[0]
        self.hasher.insert(state)
        count = self.hasher.lookup(state)
        priority = 1 / (count + self.delta)  # Calculate priority based on state visitation count
        
        # If buffer exceeds capacity, remove the oldest entry
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)
        
        self.buffer.append(transition)
        self.priorities.append(priority)
    
    def sample(self, batch_size):
        """
        Samples a batch of transitions based on priority distribution.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        priorities = np.array(self.priorities) ** self.beta
        probabilities = priorities / np.sum(priorities)  # Normalize priorities to get probabilities
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probabilities)
        sampled_transitions = [self.buffer[i] for i in indices]
        
        # Separate components of transitions
        states, actions, rewards, next_states, dones = zip(*sampled_transitions)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)