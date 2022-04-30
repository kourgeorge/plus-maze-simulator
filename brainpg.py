__author__ = 'gkour'

import torch
import torch.optim as optim

from fixeddoorattentionbrain import FixedDoorAttentionBrain
from motivationdependantbrain import NetworkBrain, MotivationDependantBrain
from lateoutcomeevaluationbrain import LateOutcomeEvaluationBrain
torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BrainPG(NetworkBrain):

    def __init__(self, network, reward_discount=1, learning_rate=0.01):
        super().__init__(network, optim.Adam(network.parameters(), lr=learning_rate), reward_discount)


    def optimize(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch):
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


class MotivationDependantBrainPG(MotivationDependantBrain, BrainPG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MotivationDependantBrainPGLateOutcomeEvaluation(LateOutcomeEvaluationBrain, MotivationDependantBrainPG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BrainPGFixedDoorAttention(MotivationDependantBrainPG, BrainPG):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)