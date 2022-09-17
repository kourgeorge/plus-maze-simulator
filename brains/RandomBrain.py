__author__ = 'gkour'

import numpy as np
import torch

import config
import utils
from brains.abstractbrain import AbstractBrain
from learners.abstractlearner import AbstractLearner


class RandomBrain(AbstractBrain):

	def __init__(self, learner: AbstractLearner, beta=1, reward_discount=1, batch_size=config.BATCH_SIZE):
		super().__init__(reward_discount)
		self.batch_size = batch_size
		self.learner = learner
		self.consolidation_counter = 0
		print("{}. Num parameters: {}".format(str(self), self.num_trainable_parameters()))

	def think(self, obs, agent):
		return torch.softmax(torch.rand(1,4),  dim=1)

	def consolidate(self, memory, agent, iterations=config.CONSOLIDATION_REPLAYS):
		return 0

	def num_trainable_parameters(self):
		return 0

	def get_model(self):
		return self.learner.get_model()