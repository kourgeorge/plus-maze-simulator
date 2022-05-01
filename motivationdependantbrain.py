__author__ = 'gkour'

import torch

from braindqn import BrainDQN
from brainpg import BrainPG
from torchbrain import TorchBrain


class MotivationDependantBrain(TorchBrain):
	BATCH_SIZE = 20

	def __init__(self, network, reward_discount=1):
		super().__init__(network,reward_discount)

	def think(self, obs, agent):
		action_probs = self.network(torch.FloatTensor(obs), agent.get_motivation().value)
		return action_probs


class MotivationDependantBrainPG(MotivationDependantBrain, BrainPG):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)





class MotivationDependantBrainDQN(MotivationDependantBrain, BrainDQN):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)







