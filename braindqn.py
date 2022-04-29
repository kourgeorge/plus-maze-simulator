__author__ = 'gkour'

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import config
from abstractbrain import AbstractBrain
from standardbrainnetwork import StandardBrainNetworkAttention, StandardBrainNetworkOrig

torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BrainDQN(AbstractBrain):
    BATCH_SIZE = 20

    def __init__(self, network:torch.nn.Mod, reward_discount=1, learning_rate=0.01):
        super(BrainDQN, self).__init__()
        self.network = network
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.reward_discount = reward_discount
        self.num_optimizations = 0
        print("Pytorch DQN. Num parameters: " + str(self.num_trainable_parameters()))

    def think(self, obs, agent_state):
        action_probs = self.network(torch.from_numpy(obs).float().unsqueeze_(0))
        return action_probs

    def train(self, memory, agent_state):
        minibatch_size = min(BrainDQN.BATCH_SIZE, len(memory))
        if minibatch_size == 0:
            return
        self.num_optimizations += 1

        minibatch = memory.sample(minibatch_size)
        state_batch = torch.from_numpy(np.stack([np.stack(data[0]) for data in minibatch])).float()
        action_batch = torch.FloatTensor([data[1] for data in minibatch])
        reward_batch = torch.FloatTensor([data[2] for data in minibatch])
        nextstate_batch = torch.from_numpy(np.stack([data[3] for data in minibatch])).float()

        state_action_values, _ = torch.max(self.think(state_batch, agent_state) * action_batch, dim=1)

        expected_state_action_values = []
        for i in range(0, minibatch_size):
            expected_state_action_values.append(reward_batch[i])

        # Compute Huber loss
        loss = F.mse_loss(state_action_values, torch.stack(expected_state_action_values).detach())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        return loss.item()

    def save_model(self, path):
        torch.save(self.network.state_dict(), path)

    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.network.parameters())

    def get_network(self):
        return self.network


class BrainDQNFixedDoorAttention(BrainDQN):
    def __init__(self,  *args, **kwargs):
        super(BrainDQNFixedDoorAttention, self).__init__(*args, **kwargs)

    def think(self, obs, motivation):
        if motivation == config.RewardType.WATER:
            attention_vec = [1, 1, 0, 0]
        else:
            attention_vec = [0, 0, 1, 1]
        action_probs = self.network(torch.FloatTensor(obs), attention_vec)

        return action_probs

class BrainDQNSeparateNetworks(BrainDQN):
    def __init__(self,  *args, **kwargs):
        super(BrainDQNSeparateNetworks, self).__init__(*args, **kwargs)

    def think(self, obs, motivation):
        action_probs = self.network(torch.FloatTensor(obs), motivation.value)

        return action_probs
