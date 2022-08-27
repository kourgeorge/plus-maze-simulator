__author__ = 'gkour'

import numpy as np
import torch
from abstractbrain import AbstractBrain
import os.path
from standardbrainnetwork import AbstractNetwork
from learner import AbstractLearner
import config

torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConsolidationBrain(AbstractBrain):

	def __init__(self, learner: AbstractLearner, reward_discount=1):
		super().__init__(reward_discount)
		self.learner = learner
		self.consolidation_counter = 0
		print("{}. Num parameters: {}".format(str(self),self.num_trainable_parameters()))

	def network(self) -> AbstractNetwork:
		return self.learner.get_model()

	def think(self, obs, agent):
		action_probs = self.learner.get_model()(torch.FloatTensor(obs))
		return action_probs

	def consolidate(self, memory, agent, batch_size=config.BATCH_SIZE, iterations=config.CONSOLIDATION_REPLAYS):
		minibatch_size = min(batch_size, len(memory))
		if minibatch_size == 0:
			return
		self.consolidation_counter += 1

		losses = []
		for _ in range(iterations):
			minibatch = memory.sample(minibatch_size)
			state_batch = torch.from_numpy(np.stack([np.stack(data[0]) for data in minibatch])).float()
			action_batch = torch.FloatTensor([data[1] for data in minibatch])
			reward_batch = torch.FloatTensor([data[2] for data in minibatch])
			nextstate_batch = torch.from_numpy(np.stack([data[4] for data in minibatch])).float()

			action_values = self.think(state_batch, agent)

			losses += [self.learner.learn(state_batch, action_batch, reward_batch, action_values, nextstate_batch)]

		return np.mean(losses)

	def save_model(self, path):
		torch.save(self.network().state_dict(), path)

	def load_model(self, path):
		if os.path.exists(path):
			self.network().load_state_dict(torch.load(path))

	def num_trainable_parameters(self):
		return sum(p.numel() for p in self.network().parameters())

	def get_network(self):
		return self.network()


class RandomBrain(AbstractBrain):

	def __init__(self, learner: AbstractLearner, reward_discount=1):
		super().__init__(reward_discount)
		self.learner = learner
		self.consolidation_counter = 0
		print("{}. Num parameters: {}".format(str(self), self.num_trainable_parameters()))

	def network(self) -> AbstractNetwork:
		return self.learner.get_model()

	def think(self, obs, agent):
		return torch.softmax(torch.rand(1,4),  dim=1)

	def consolidate(self, memory, agent, batch_size=config.BATCH_SIZE, iterations=config.CONSOLIDATION_REPLAYS):
		return 0

	def save_model(self, path):
		torch.save(self.network().state_dict(), path)

	def load_model(self, path):
		if os.path.exists(path):
			self.network().load_state_dict(torch.load(path))

	def num_trainable_parameters(self):
		return sum(p.numel() for p in self.network().parameters())

	def get_network(self):
		return self.network()