__author__ = 'gkour'

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from abstractbrain import AbstractBrain
import os.path

device = "cpu"


class BrainDQN(AbstractBrain):
    BATCH_SIZE = 512

    def __init__(self, observation_size, num_actions, reward_discount, learning_rate=0.01):
        super(BrainDQN, self).__init__(observation_size, num_actions)
        self.policy_net = DQN(observation_size, num_actions).to(device)
        self.target_net = DQN(observation_size, num_actions).to(device)
        self.optimizer = optim.SGD(self.policy_net.parameters(), lr=learning_rate)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.reward_discount = reward_discount
        self.num_optimizations = 0
        print("Pytorch DQN. Num parameters: " + str(self.num_trainable_parameters()))

    def think(self, obs):
        with torch.no_grad():
            action = self.policy_net(torch.from_numpy(obs).float().unsqueeze_(0)).argmax().item()
            distribution = np.zeros(self.num_actions())
            distribution[action] = 1
            return distribution

    def train(self, memory):

        minibatch_size = min(BrainDQN.BATCH_SIZE, len(memory))
        if minibatch_size == 0:
            return
        self.num_optimizations += 1

        minibatch = memory.sample(minibatch_size)
        state_batch = torch.from_numpy(np.stack([np.stack(data[0]) for data in minibatch])).float()
        action_batch = torch.FloatTensor([data[1] for data in minibatch])
        reward_batch = torch.FloatTensor([data[2] for data in minibatch])
        nextstate_batch = torch.from_numpy(np.stack([data[3] for data in minibatch])).float()

        state_action_values, _ = torch.max(self.policy_net(state_batch) * action_batch, dim=1)
        # Compute V(s_{t+1}) for all next states.
        qvalue_batch = self.target_net(nextstate_batch)
        expected_state_action_values = []
        for i in range(0, minibatch_size):
            terminal = minibatch[i][4]
            if terminal:
                expected_state_action_values.append(reward_batch[i])
            else:
                expected_state_action_values.append(reward_batch[i] + self.reward_discount * torch.max(qvalue_batch[i]))

        # Compute Huber loss
        loss = F.mse_loss(state_action_values, torch.stack(expected_state_action_values).detach())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1)
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # if self.num_optimizations % 10 == 0:
        #     self.target_net.load_state_dict(self.policy_net.state_dict())
        #     for name, param in self.policy_net.state_dict().items():
        #         if name == 'lin1.weight':
        #             print(param)

        return {'loss': loss.item()}

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        if os.path.exists(path):
            self.policy_net.load_state_dict(torch.load(path))
            self.target_net.load_state_dict(torch.load(path))

    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.policy_net.parameters())


class DQN(nn.Module):
    def __init__(self, num_channels, num_actions):
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(num_channels, num_channels)
        self.lin2 = nn.Linear(num_channels, num_channels)
        self.head = nn.Linear(num_channels, num_actions)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.head(x.view(x.size(0), -1))
