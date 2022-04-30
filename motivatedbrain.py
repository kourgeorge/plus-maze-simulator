__author__ = 'gkour'

import numpy as np
import torch
import torch.optim as optim
from abstractbrain import AbstractBrain
import os.path

torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MotivatedBrain(AbstractBrain):
	BATCH_SIZE = 20

	def __init__(self, network, optimizer, reward_discount=1):
		super(MotivatedBrain, self).__init__()
		self.network = network
		self.optimizer = optimizer
		self.reward_discount = reward_discount
		self.num_optimizations = 0
		print("{}. Num parameters: {}".format(str(self),self.num_trainable_parameters()))

	def think(self, obs, agent):
		action_probs = self.network(torch.FloatTensor(obs))
		return action_probs

	def consolidate(self, memory, agent, batch_size=BATCH_SIZE):
		minibatch_size = min(batch_size, len(memory))
		if minibatch_size == 0:
			return
		self.num_optimizations += 1

		minibatch = memory.sample(minibatch_size)
		state_batch = torch.from_numpy(np.stack([np.stack(data[0]) for data in minibatch])).float()
		action_batch = torch.FloatTensor([data[1] for data in minibatch])
		reward_batch = torch.FloatTensor([data[2] for data in minibatch])
		nextstate_batch = torch.from_numpy(np.stack([data[4] for data in minibatch])).float()

		action_values = self.think(state_batch, agent)

		return self.train(state_batch, action_batch, reward_batch, action_values, nextstate_batch)

	def train(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch):
		'''Given a set of states (s), actions (a), and obtained rewards (r) and  state-action values under current
		policy pi_t(s,a), improve the policy.'''
		raise NotImplementedError()

	def save_model(self, path):
		torch.save(self.network.state_dict(), path)

	def load_model(self, path):
		if os.path.exists(path):
			self.network.load_state_dict(torch.load(path))

	def num_trainable_parameters(self):
		return sum(p.numel() for p in self.network.parameters())

	def get_network(self):
		return self.network
