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
from torch.autograd import Variable

device = "cpu"

def has_err(x):
    return bool(((x != x) | (x == float("inf")) | (x == float("-inf"))).any().item())

class BrainPG(AbstractBrain):
    BATCH_SIZE = 512

    def __init__(self, observation_size, num_actions, reward_discount, learning_rate=0.01):
        super(BrainPG, self).__init__(observation_size, num_actions)
        self.policy = Policy(observation_size, num_actions).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.reward_discount = reward_discount
        self.num_optimizations = 0
        print("Pytorch PG. Num parameters: " + str(self.num_trainable_parameters()))

    def think(self, obs):
        with torch.no_grad():
            action_probs= self.policy(torch.FloatTensor(obs))
            return action_probs.tolist()
            #c = torch.distributions.Categorical(action_probs)
            #action = c.sample()
        #return action.item()

    def train(self, memory):

        minibatch_size = min(BrainPG.BATCH_SIZE, len(memory))
        if minibatch_size == 0:
            return
        self.num_optimizations += 1

        minibatch = memory.last(100)
        state_batch = torch.from_numpy(np.stack([np.stack(data[0]) for data in minibatch])).float()
        action_batch = torch.FloatTensor([data[1] for data in minibatch])
        reward_batch = torch.FloatTensor([data[2] for data in minibatch])
        nextstate_batch = torch.from_numpy(np.stack([data[3] for data in minibatch])).float()

        # Scale rewards
        #reward_std = 1 if torch.isnan(reward_batch.std()) else reward_batch.std()
        #rewards = (reward_batch - reward_batch.mean()) / (reward_std  + np.finfo(np.float32).eps)

        log_prob_actions = torch.log(torch.max(self.policy(state_batch).mul(action_batch), dim=1)[0])

        # Calculate loss
        loss = (torch.mean(torch.mul(log_prob_actions, reward_batch).mul(-1), -1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        assert not has_err(self.policy.l2.weight.grad)
        assert not has_err(self.policy.l1.weight.grad)
        #torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1)
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


        # if self.num_optimizations % 10 == 0:
        #     self.target_net.load_state_dict(self.policy_net.state_dict())
        #     for name, param in self.policy_net.state_dict().items():
        #         if name == 'lin1.weight':
        #             print(param)

        #self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)

    # def load_model(self, path):
    #     if os.path.exists(path):
    #         self.policy_net.load_state_dict(torch.load(path))
    #         self.target_net.load_state_dict(torch.load(path))

    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.policy.parameters())


class Policy(nn.Module):
    def __init__(self, num_channels, num_actions):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(num_channels, 128, bias=False)
        self.l2 = nn.Linear(128, num_actions, bias=False)

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(x)
