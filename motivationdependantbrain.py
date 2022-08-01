__author__ = 'gkour'

import torch
from consolidationbrain import ConsolidationBrain


class MotivationDependantBrain(ConsolidationBrain):
	BATCH_SIZE = 20

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def think(self, obs, agent):
		action_probs = self.network()(torch.FloatTensor(obs), agent.get_motivation().value)
		return action_probs







