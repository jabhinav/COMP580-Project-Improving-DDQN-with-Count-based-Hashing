import numpy as np
import torch


class SimHash():
	"""
		Implementation of SimHash. Will compute and return the count of the states within a batch
	"""
	def __init__(self, state_size, k):
		self.hash = {}
		self.A = np.random.normal(0, 1, (k, state_size))
	
	def count(self, states):
		counts = []
		for state in states:
			key = str(np.sign(self.A @ state.detach().cpu().numpy()).tolist())
			if key in self.hash:
				self.hash[key] += 1
			else:
				self.hash[key] = 1
			counts.append(self.hash[key])
		
		return torch.from_numpy(np.array(counts))
	
	
	
class GlobalSimHash():
	"""
		Implementation of Global SimHash. Will store the count of the states within the entire buffer
	"""
	def __init__(self, state_size, k):
		self.hash = {}
		self.A = np.random.normal(0, 1, (k, state_size))
		
	def insert(self, state):
		key = str(np.sign(self.A @ state).tolist())
		if key in self.hash:
			self.hash[key] += 1
		else:
			self.hash[key] = 1
			
	def lookup(self, state):
		key = str(np.sign(self.A @ state).tolist())
		if key in self.hash:
			return self.hash[key]
		else:
			return 0
	
	