__author__ = 'gkour'

import config
import numpy as np
import torch
import torch.optim as optim
from abstractbrain import AbstractBrain
import os.path

torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BrainPG(AbstractBrain):
    BATCH_SIZE = 20

    def __init__(self, network, reward_discount=1, learning_rate=0.01):
        super(BrainPG, self).__init__()
        self.network = network
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.reward_discount = reward_discount
        self.num_optimizations = 0
        print("Pytorch PG. Num parameters: " + str(self.num_trainable_parameters()))

    def think(self, obs, agent_state):
        action_probs = self.network(torch.FloatTensor(obs))
        return action_probs

    def train(self, memory, agent_state):
        minibatch_size = min(BrainPG.BATCH_SIZE, len(memory))
        if minibatch_size == 0:
            return
        self.num_optimizations += 1

        minibatch = memory.sample(minibatch_size)
        state_batch = torch.from_numpy(np.stack([np.stack(data[0]) for data in minibatch])).float()
        action_batch = torch.FloatTensor([data[1] for data in minibatch])
        reward_batch = torch.FloatTensor([data[2] for data in minibatch])
        nextstate_batch = torch.from_numpy(np.stack([data[3] for data in minibatch])).float()

        log_prob_actions = torch.log(torch.max(self.think(state_batch, agent_state).mul(action_batch), dim=1)[0])

        # Calculate loss
        loss = (torch.mean(torch.mul(log_prob_actions, reward_batch).mul(-1), -1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1)

        self.optimizer.step()
        return loss.item()

    def save_model(self, path):
        torch.save(self.network.state_dict(), path)

    def load_model(self, path):
        if os.path.exists(path):
            self.network.load_state_dict(torch.load(path))

    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.network.parameters())

    def get_network(self):
        return self.network


class BrainPGFixedDoorAttention(BrainPG):
    def __init__(self,  *args, **kwargs):
        super(BrainPGFixedDoorAttention, self).__init__(*args, **kwargs)

    def think(self, obs, motivation):
        if motivation == config.RewardType.WATER:
            attention_vec = [1, 1, 0, 0]
        else:
            attention_vec = [0, 0, 1, 1]
        action_probs = self.network(torch.FloatTensor(obs), attention_vec)
        return action_probs


class BrainPGSeparateNetworks(BrainPG):
    def __init__(self,  *args, **kwargs):
        super(BrainPGSeparateNetworks, self).__init__(*args, **kwargs)

    def think(self, obs, motivation):
        action_probs = self.network(torch.FloatTensor(obs), motivation.value)
        return action_probs