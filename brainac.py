__author__ = 'gkour'

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from abstractbrain import AbstractBrain
import os.path

#torch.manual_seed(0)

device = "cpu"


def has_err(x):
    return bool(((x != x) | (x == float("inf")) | (x == float("-inf"))).any().item())


class BrainAC(AbstractBrain):
    BATCH_SIZE = 1

    def __init__(self, observation_size, num_actions, reward_discount, learning_rate=0.01):
        super(BrainAC, self).__init__(observation_size, num_actions)
        self.policy = Policy(observation_size, num_actions).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.reward_discount = reward_discount
        self.num_optimizations = 0
        print("Pytorch PG. Num parameters: " + str(self.num_trainable_parameters()))

    def think(self, obs):
        with torch.no_grad():
            action_probs, _ = self.policy(torch.FloatTensor(obs))
        return action_probs.tolist()

    def train(self, memory):
        minibatch_size = min(BrainAC.BATCH_SIZE, len(memory))
        if minibatch_size == 0:
            return
        self.num_optimizations += 1

        minibatch = memory.last(minibatch_size)
        state_batch = torch.from_numpy(np.stack([np.stack(data[0]) for data in minibatch])).float()
        action_batch = torch.FloatTensor([data[1] for data in minibatch])
        reward_batch = torch.FloatTensor([data[2] for data in minibatch])
        nextstate_batch = torch.from_numpy(np.stack([data[3] for data in minibatch])).float()

        # Scale rewards
        # reward_std = 1 if torch.isnan(reward_batch.std()) else reward_batch.std()
        # rewards = (reward_batch - reward_batch.mean()) / (reward_std  + np.finfo(np.float32).eps)

        action_probs, state_value = self.policy(state_batch)
        log_prob_actions = torch.log(torch.max(action_probs.mul(action_batch), dim=1)[0])

        # Calculate critic loss
        advantage = reward_batch - state_value.squeeze(1)
        value_loss = F.smooth_l1_loss(state_value, reward_batch, reduction='mean')

        # Calculate actor loss
        loss = (torch.mean(torch.mul(log_prob_actions, reward_batch).mul(-1), -1))

        #loss += value_loss

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        #assert not has_err(self.policy.l2.weight.grad)
        #assert not has_err(self.policy.l1.weight.grad)

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1)
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()

    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)

    def load_model(self, path):
        if os.path.exists(path):
            self.policy.load_state_dict(torch.load(path))

    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.policy.parameters())


class Policy(nn.Module):
    def __init__(self, num_channels, num_actions):
        super(Policy, self).__init__()
        self.affine = nn.Linear(num_channels, 16, bias=False)

        self.affine2 = nn.Linear(num_channels, 16, bias=False)
        self.controller = nn.Linear(16, num_actions, bias=False)
        self.state_value = nn.Linear(16, 1, bias=False)

        self.actor = torch.nn.Sequential(
            self.affine,
            nn.Dropout(p=0.6),
            nn.Sigmoid(),
            self.controller,
            nn.Softmax(dim=-1)
        )

        self.critic = torch.nn.Sequential(
            self.affine2,
            nn.Dropout(p=0.6),
            nn.Sigmoid(),
            self.state_value
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)
