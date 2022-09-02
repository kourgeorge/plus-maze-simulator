__author__ = 'gkour'

import torch
from consolidationbrain import ConsolidationBrain


class MotivationDependantBrain(ConsolidationBrain):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def think(self, obs, agent):
		action_probs = self.get_model()(torch.FloatTensor(obs), agent.get_motivation().value)
		return action_probs







