__author__ = 'gkour'

import numpy as np
import torch
from brains.abstractbrain import AbstractBrain
import os.path
from learners.networklearners import AbstractLearner
import config

torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConsolidationBrain(AbstractBrain):

	def __init__(self, learner: AbstractLearner, beta=1, reward_discount=1, batch_size=config.BATCH_SIZE):
		super().__init__(reward_discount)
		self.learner = learner
		self.batch_size = batch_size
		self.consolidation_counter = 0
		self.beta = beta
		print("{}.{}: Num parameters: {}".format(str(self),str(learner.get_model()),self.num_trainable_parameters()))

	def think(self, obs, agent):
		action_value = self.get_model()(torch.FloatTensor(obs))
		action_dist = torch.softmax(self.beta*action_value, axis=-1)
		return action_dist

	def get_learner(self):
		return self.learner

	def consolidate(self, memory, agent, iterations=config.CONSOLIDATION_REPLAYS):
		minibatch_size = min(self.batch_size, len(memory))
		if minibatch_size == 0:
			return
		self.consolidation_counter += 1

		losses = []
		for _ in range(iterations):
			minibatch = memory.last(minibatch_size)
			state_batch = torch.from_numpy(np.stack([np.stack(data[0]) for data in minibatch])).float()
			action_batch = torch.from_numpy(np.stack([data[1] for data in minibatch], axis=0)).float()
			reward_batch = torch.from_numpy(np.stack([data[2] for data in minibatch])).float()
			nextstate_batch = torch.from_numpy(np.stack([data[4] for data in minibatch])).float()

			action_values = self.think(state_batch, agent)

			losses += [self.learner.learn(state_batch, action_batch, reward_batch, action_values, nextstate_batch, agent.get_motivation())]

		return np.mean(losses)

	def save_model(self, path):
		torch.save(self.get_model().state_dict(), path)

	def load_model(self, path):
		if os.path.exists(path):
			self.get_model().load_state_dict(torch.load(path))

	def num_trainable_parameters(self):
		return sum(p.numel() for p in self.get_model().parameters())

	def get_model(self):
		return self.learner.get_model()

