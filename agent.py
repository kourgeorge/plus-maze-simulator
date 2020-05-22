from braindqntorch import BrainDQN
from environment import PlusMaze
from collections import deque
import config
import utils


class Agent:
    def __init__(self, observation_size=PlusMaze.state_shape(), num_actions=PlusMaze.num_actions(),
                 motivation=config.RewardType.WATER, motivated_reward_value=1, non_motivated_reward_value=0):
        self._brain = BrainDQN(observation_size=observation_size,
                               num_actions=num_actions,
                               reward_discount=0,
                               learning_rate=config.BASE_LEARNING_RATE)

        self._memory_size = 1024
        self._memory = utils.NStepsReplayMemory(self._memory_size, 1, 0.99)
        self._motivation = motivation
        self._motivated_reward_value = motivated_reward_value
        self._non_motivated_reward_value = non_motivated_reward_value

    def decide(self, state):
        eps = config.BASE_EPSILON
        brain_actions_prob = self._brain.think(state)
        # action_prob = utils.softmax(brain_actions_prob)
        decision = utils.epsilon_greedy(eps, brain_actions_prob)

        # action_prob = utils.normalize_dist((1-eps)*self.fitrah() + eps*brain_actions_prob)
        # decision = utils.epsilon_greedy(0, action_prob)
        return decision

    def evaluate_reward(self, reward_type):
        if reward_type == config.RewardType.NONE:
            return 0
        return self._motivated_reward_value if reward_type == self._motivation else self._non_motivated_reward_value

    def add_experience(self, *experience):
        self._memory.push(*experience)

    def smarten(self):
        return self._brain.train(self._memory)

    def set_motivation(self, motivation):
        self._motivation = motivation
