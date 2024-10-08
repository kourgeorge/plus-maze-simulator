__author__ = 'gkour'

import numpy as np
import config
from brains.abstractbrain import AbstractBrain
import torch

from learners.abstractlearner import AbstractLearner
from motivatedagent import MotivatedAgent

class TDBrain(AbstractBrain):

    def __init__(self, learner: AbstractLearner, beta=5, reward_discount=0, batch_size=config.BATCH_SIZE, *args, **kwargs):
        super().__init__(reward_discount, *args, **kwargs)
        self.learner = learner
        self.batch_size = batch_size
        self.beta = beta
        self._reward_discount = reward_discount
        self.num_optimizations = 0

    def think(self, obs, agent):
        # Get action values from the model
        action_values = self.get_model()(obs, agent.get_motivation())  # Should return (batch_size, num_actions)

        # Convert action values to torch tensor
        action_values_tensor = torch.from_numpy(action_values).float()

        # Compute action probabilities using softmax
        action_dist = torch.softmax(self.beta * action_values_tensor, dim=-1)

        return action_dist

    def get_model(self):
        return self.learner.model

    def get_learner(self):
        return self.learner

    def consolidate(self, memory, agent: MotivatedAgent, replays=config.CONSOLIDATION_REPLAYS):
        minibatch_size = min(self.batch_size, len(memory))
        if minibatch_size == 0:
            return

        for _ in range(replays):
            minibatch = memory.last(minibatch_size)
            state_batch = np.stack([np.stack(data[0]) for data in minibatch])
            action_batch = np.stack([data[1] for data in minibatch])
            outcome_batch = np.stack([data[3] for data in minibatch])
            reward_batch = np.stack([agent.evaluate_outcome(outcome) for outcome in outcome_batch])
            nextstate_batch = np.stack([data[4] for data in minibatch])

            # Think method returns torch tensor; convert to numpy if necessary
            action_values = self.think(state_batch, agent).detach().numpy()

            optimization_data = self.learner.learn(state_batch, action_batch, reward_batch, action_values, nextstate_batch,
                                                   agent.get_motivation())

        return optimization_data

    def num_trainable_parameters(self):
        # If your Bayesian model has parameters that can be counted, implement this method accordingly
        return 0

    def get_parameters(self):
        return self.get_learner().get_parameters()
