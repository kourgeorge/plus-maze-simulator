__author__ = 'gkour'

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from abstractbrain import AbstractBrain
from standardbrainnetwork import StandardBrainNetwork

torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BrainDQN(AbstractBrain):
    BATCH_SIZE = 20

    def __init__(self, observation_size, num_actions, reward_discount, learning_rate=0.01):
        super(BrainDQN, self).__init__(observation_size, num_actions)
        self.network = StandardBrainNetwork(observation_size, num_actions)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.reward_discount = reward_discount
        self.num_optimizations = 0
        print("Pytorch DQN. Num parameters: " + str(self.num_trainable_parameters()))

    def think(self, obs):
        with torch.no_grad():
            action = self.network(torch.from_numpy(obs).float().unsqueeze_(0)).argmax().item()
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

        state_action_values, _ = torch.max(self.network(state_batch) * action_batch, dim=1)

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

    # def load_model(self, path):
    #     if os.path.exists(path):
    #         self.policy_net.load_state_dict(torch.load(path))
    #         self.target_net.load_state_dict(torch.load(path))

    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.network.parameters())

    def get_network(self) -> StandardBrainNetwork:
        return self.network
