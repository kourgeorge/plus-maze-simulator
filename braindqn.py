__author__ = 'gkour'

import torch
import torch.nn.functional as F
import torch.optim as optim
import config
from motivatedbrain import MotivatedBrain
from motivatedagent import MotivatedAgent
import numpy as np

torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BrainDQN(MotivatedBrain):

	def __init__(self, network, reward_discount=1, learning_rate=0.01):
		super(BrainDQN, self).__init__(network, optim.Adam(network.parameters(), lr=learning_rate), reward_discount)

	def train(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch):
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


class BrainDQNFixedDoorAttention(BrainDQN):
	def __init__(self, *args, **kwargs):
		super(BrainDQNFixedDoorAttention, self).__init__(*args, **kwargs)

	def think(self, obs, agent: MotivatedAgent):
		if agent.get_motivation() == config.RewardType.WATER:
			attention_vec = [1, 1, 0, 0]
		else:
			attention_vec = [0, 0, 1, 1]
		action_probs = self.network(torch.FloatTensor(obs), attention_vec)

		return action_probs


class BrainDQNSeparateNetworks(BrainDQN):
	def __init__(self, *args, **kwargs):
		super(BrainDQNSeparateNetworks, self).__init__(*args, **kwargs)

	def think(self, obs, agent: MotivatedAgent):
		action_probs = self.network(torch.FloatTensor(obs), agent.get_motivation().value)
		return action_probs

class BrainDQNSeparateNetworksLateOutcomeEvaluation(BrainDQNSeparateNetworks):
    BATCH_SIZE = 20
    def __init__(self,  *args, **kwargs):
        super(BrainDQNSeparateNetworksLateOutcomeEvaluation, self).__init__(*args, **kwargs)

    def consolidate(self, memory, agent:MotivatedAgent, batch_size=BATCH_SIZE):
        minibatch_size = min(batch_size, len(memory))
        if minibatch_size == 0:
            return
        self.num_optimizations += 1

        minibatch = memory.sample(minibatch_size)
        state_batch = torch.from_numpy(np.stack([np.stack(data[0]) for data in minibatch])).float()
        action_batch = torch.FloatTensor([data[1] for data in minibatch])
        outcome_batch = [data[3] for data in minibatch]
        reward_batch = torch.FloatTensor([agent.evaluate_outcome(outcome) for outcome in outcome_batch])

        action_values = self.think(state_batch, agent)

        return self.train(state_batch, action_batch, reward_batch, action_values, None)