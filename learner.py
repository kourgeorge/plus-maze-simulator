__author__ = 'gkour'

import torch
import torch.nn.functional as F
import torch.optim as optim
from standardbrainnetwork import AbstractNetwork

torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AbstractLearner:
	def __init__(self, model, optimizer):
		super().__init__()
		self.model = model
		self.optimizer = optimizer

	def get_model(self):
		return self.model

	def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch):
		raise NotImplementedError()


class DQN(AbstractLearner):
	def __init__(self, network: AbstractNetwork, optimizer=optim.Adam, learning_rate=0.01):
		super().__init__(network, optimizer)
		self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)

	def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch):
		selected_action_value, _ = torch.max(action_values * action_batch, dim=1)

		# Compute Huber loss
		loss = F.mse_loss(selected_action_value, reward_batch.detach())

		# Optimize the model
		self.optimizer.zero_grad()
		loss.backward()

		self.optimizer.step()
		return loss.item()


class PG(AbstractLearner):

	def __init__(self, network: AbstractNetwork, optimizer=optim.Adam, learning_rate=0.01):
		super().__init__(network, optimizer)
		self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)

	def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch):
		state_action_values, _ = torch.max(action_values * action_batch, dim=1)
		log_prob_actions = torch.log(state_action_values)

		# Calculate loss
		loss = (torch.mean(torch.mul(log_prob_actions, reward_batch).mul(-1), -1))

		# Optimize the model
		self.optimizer.zero_grad()
		loss.backward()

		torch.nn.utils.clip_grad_norm_(self.get_model().parameters(), max_norm=1)

		self.optimizer.step()
		return loss.item()
