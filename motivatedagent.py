__author__ = 'gkour'

#import config
from config import get_config
config = get_config()
from utils import epsilon_greedy
from ReplayMemory import ReplayMemory
from abstractbrain import AbstractBrain
import numpy as np


class MotivatedAgent:
    def __init__(self, brain: AbstractBrain, memory_size=10000, motivation=1,
                 motivated_reward_value=1, non_motivated_reward_value=0.3, exploration_param=0.2):

        self._brain = brain
        self._memory_size = memory_size
        self._memory = ReplayMemory(self._memory_size)
        self._motivation = motivation
        self._motivated_reward_value = motivated_reward_value
        self._non_motivated_reward_value = non_motivated_reward_value
        self._exploration_param = exploration_param

    def get_brain(self) -> AbstractBrain:
        return self._brain

    def decide(self, state):
        decision = self._brain.think(np.expand_dims(state,0), self).squeeze().detach().numpy()
        action = epsilon_greedy(self._exploration_param, decision)
        return action

    def evaluate_outcome(self, outcome):
        if outcome == config.RewardType.NONE:
            return 0
        return self._motivated_reward_value if outcome == self._motivation else self._non_motivated_reward_value

    def add_experience(self, *experience):
        self._memory.push(*experience)

    def smarten(self):
        return self._brain.consolidate(self._memory, self)

    def set_motivation(self, motivation):
        self._motivation = motivation

    def get_motivation(self):
        return self._motivation

    def get_memory(self) -> ReplayMemory:
        return self._memory

    def clear_memory(self):
        self._memory = ReplayMemory(self._memory_size)

    def get_internal_state(self):
        if self._motivation == config.RewardType.WATER:
            return [-1]
        else:
            return [1]