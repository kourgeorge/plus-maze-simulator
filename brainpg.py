__author__ = 'gkour'

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from abstractbrain import AbstractBrain
import os.path
from standardbrainnetwork import StandardBrainNetwork

device = "cpu"


class BrainPG(AbstractBrain):
    BATCH_SIZE = 20

    def __init__(self, observation_size, num_actions, reward_discount, learning_rate=0.01):
        super(BrainPG, self).__init__(observation_size, num_actions)
        self.policy = StandardBrainNetwork(observation_size, num_actions).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.reward_discount = reward_discount
        self.num_optimizations = 0
        print("Pytorch PG. Num parameters: " + str(self.num_trainable_parameters()))

    def think(self, obs):
        with torch.no_grad():
            action_probs = self.policy(torch.FloatTensor(obs))
        return action_probs.tolist()

    def train(self, memory):
        minibatch_size = min(BrainPG.BATCH_SIZE, len(memory))
        if minibatch_size == 0:
            return
        self.num_optimizations += 1

        minibatch = memory.last(minibatch_size)
        state_batch = torch.from_numpy(np.stack([np.stack(data[0]) for data in minibatch])).float()
        action_batch = torch.FloatTensor([data[1] for data in minibatch])
        reward_batch = torch.FloatTensor([data[2] for data in minibatch])
        nextstate_batch = torch.from_numpy(np.stack([data[3] for data in minibatch])).float()

        log_prob_actions = torch.log(torch.max(self.policy(state_batch).mul(action_batch), dim=1)[0])

        # Calculate loss
        loss = (torch.mean(torch.mul(log_prob_actions, reward_batch).mul(-1), -1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1)

        self.optimizer.step()
        return loss.item()

    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)

    def load_model(self, path):
        if os.path.exists(path):
            self.policy.load_state_dict(torch.load(path))

    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.policy.parameters())