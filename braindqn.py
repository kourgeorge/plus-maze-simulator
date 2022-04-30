__author__ = 'gkour'

import torch
import torch.nn.functional as F
import torch.optim as optim

from fixeddoorattentionbrain import FixedDoorAttentionBrain
from motivationdependantbrain import MotivationDependantBrain, NetworkBrain
from lateoutcomeevaluationbrain import LateOutcomeEvaluationBrain

torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BrainDQN(NetworkBrain):

	def __init__(self, network, reward_discount=1, learning_rate=0.01):
		super().__init__(network, optim.Adam(network.parameters(), lr=learning_rate), reward_discount)

	def optimize(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch):
		state_action_values, _ = torch.max(action_values * action_batch, dim=1)

		expected_state_action_values = []
		for i in range(0, len(reward_batch)):
			expected_state_action_values.append(reward_batch[i])

		# Compute Huber loss
		loss = F.mse_loss(state_action_values, torch.stack(expected_state_action_values).detach())

		# Optimize the model
		self.optimizer.zero_grad()
		loss.backward()

		self.optimizer.step()
		return loss.item()


class MotivationDependantBrainDQN(MotivationDependantBrain, BrainDQN):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)


class MotivationDependantBrainDQNLateOutcomeEvaluation(LateOutcomeEvaluationBrain, MotivationDependantBrainDQN):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

class BrainDQNFixedDoorAttention(FixedDoorAttentionBrain, MotivationDependantBrain, BrainDQN):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)