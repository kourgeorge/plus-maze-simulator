__author__ = 'gkour'

import torch
import torch.nn.functional as F
import torch.optim as optim

from torchbrain import TorchBrain

torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BrainDQN(TorchBrain):

	def __init__(self, network, reward_discount=1, learning_rate=0.01):
		super().__init__(network, optim.Adam(network.parameters(), lr=learning_rate), reward_discount)

	def optimize(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch):
		selected_action_value, _ = torch.max(action_values * action_batch, dim=1)

		# Compute Huber loss
		loss = F.mse_loss(selected_action_value, reward_batch.detach())

		# Optimize the model
		self.optimizer.zero_grad()
		loss.backward()

		self.optimizer.step()
		return loss.item()


