__author__ = 'gkour'

import torch
import config
from motivatedagent import MotivatedAgent
from motivatedbrain import MotivatedBrain


class FixedDoorAttentionBrain(MotivatedBrain):
	def __init__(self, *args, **kwargs):
		super(FixedDoorAttentionBrain, self).__init__(*args, **kwargs)

	def think(self, obs, agent: MotivatedAgent):
		if agent.get_motivation() == config.RewardType.WATER:
			attention_vec = [1, 1, 0, 0]
		else:
			attention_vec = [0, 0, 1, 1]
		action_probs = self.network(torch.FloatTensor(obs), attention_vec)

		return action_probs
