__author__ = 'gkour'

import numpy as np
import config
import torch
import torch.optim as optim
from motivatedbrain import MotivatedBrain
from motivatedagent import MotivatedAgent
torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class BrainPG(MotivatedBrain):

    def __init__(self, network, reward_discount=1, learning_rate=0.01):
        super(BrainPG, self).__init__(network,optim.Adam(network.parameters(), lr=learning_rate), reward_discount)

    def train(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch):
        state_action_values, _ = torch.max(action_values * action_batch, dim=1)
        log_prob_actions = torch.log(state_action_values)

        # Calculate loss
        loss = (torch.mean(torch.mul(log_prob_actions, reward_batch).mul(-1), -1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1)

        self.optimizer.step()
        return loss.item()


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

    def think(self, obs,  agent:MotivatedAgent):
        action_probs = self.network(torch.FloatTensor(obs), agent.get_motivation().value)
        return action_probs


class BrainPGSeparateNetworksLateOutcomeEvaluation(BrainPGSeparateNetworks):
    BATCH_SIZE = 20
    def __init__(self,  *args, **kwargs):
        super(BrainPGSeparateNetworksLateOutcomeEvaluation, self).__init__(*args, **kwargs)

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