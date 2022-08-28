__author__ = 'gkour'

import torch
from motivatedagent import MotivatedAgent
from motivationdependantbrain import MotivationDependantBrain
from rewardtype import RewardType


class FixedDoorAttentionBrain(MotivationDependantBrain):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def think(self, obs, agent: MotivatedAgent):
		if agent.get_motivation() == RewardType.WATER:
			attention_vec = [0, 0, 1, 1]
		else:
			attention_vec = [1, 1, 0, 0]
		action_probs = self.network()(torch.FloatTensor(obs), attention_vec)

		return action_probs