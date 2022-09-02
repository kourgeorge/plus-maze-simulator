
import numpy as np
import config
from abstractbrain import AbstractBrain
from learner import TD
import torch


class TDBrain(AbstractBrain):

    def __init__(self, learner:TD, reward_discount=0):
        super().__init__(reward_discount)
        self.learner = learner

        self._reward_discount = reward_discount
        self.num_optimizations = 0

    def think(self, state, agent):
        state_actions_value = self.get_model()(state)
        return torch.from_numpy(np.stack(state_actions_value))

    def get_model(self):
        return self.learner.model

    def consolidate(self, memory, agent, batch_size=config.BATCH_SIZE, replays=config.CONSOLIDATION_REPLAYS):
        minibatch_size = min(batch_size, len(memory))
        if minibatch_size == 0:
            return

        losses = []
        for _ in range(replays):
            minibatch = memory.last(minibatch_size)
            state_batch = np.stack([np.stack(data[0]) for data in minibatch])
            action_batch = np.stack([data[1] for data in minibatch])
            outcome_batch = np.stack([data[3] for data in minibatch])
            reward_batch = np.stack([agent.evaluate_outcome(outcome) for outcome in outcome_batch])
            nextstate_batch = np.stack([data[4] for data in minibatch])

            action_values = self.think(state_batch, agent)

            losses += [self.learner.learn(state_batch, action_batch, reward_batch, action_values, nextstate_batch)]

        return np.mean(losses)

    def num_trainable_parameters(self):
        return 0



