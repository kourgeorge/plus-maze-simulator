__author__ = 'gkour'

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from abstractbrain import AbstractBrain
import os.path
import copy

device = "cpu"


class BrainDQN(AbstractBrain):
    BATCH_SIZE = 20

    def __init__(self, observation_size, num_actions, reward_discount, learning_rate=0.01):
        super(BrainDQN, self).__init__(observation_size, num_actions)
        self.policy = DQN(observation_size, num_actions).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.reward_discount = reward_discount
        self.num_optimizations = 0
        print("Pytorch DQN. Num parameters: " + str(self.num_trainable_parameters()))

    def think(self, obs):
        with torch.no_grad():
            action = self.policy(torch.from_numpy(obs).float().unsqueeze_(0)).argmax().item()
            distribution = np.zeros(self.num_actions())
            distribution[action] = 1
            return distribution

    def train(self, memory):

        minibatch_size = min(BrainDQN.BATCH_SIZE, len(memory))
        if minibatch_size == 0:
            return
        self.num_optimizations += 1

        minibatch = memory.last(minibatch_size)
        state_batch = torch.from_numpy(np.stack([np.stack(data[0]) for data in minibatch])).float()
        action_batch = torch.FloatTensor([data[1] for data in minibatch])
        reward_batch = torch.FloatTensor([data[2] for data in minibatch])
        nextstate_batch = torch.from_numpy(np.stack([data[3] for data in minibatch])).float()

        state_action_values, _ = torch.max(self.policy(state_batch) * action_batch, dim=1)

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
        torch.save(self.policy.state_dict(), path)

    # def load_model(self, path):
    #     if os.path.exists(path):
    #         self.policy_net.load_state_dict(torch.load(path))
    #         self.target_net.load_state_dict(torch.load(path))

    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.policy.parameters())


class DQN(nn.Module):
    def __init__(self, num_channels, num_actions):
        super(DQN, self).__init__()
        self.affine = nn.Linear(num_channels, 16, bias=False)
        self.controller = nn.Linear(16, num_actions, bias=False)
        self.model = torch.nn.Sequential(
            self.affine,
            nn.Dropout(p=0.6),
            nn.Sigmoid(),
            self.controller,
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)
