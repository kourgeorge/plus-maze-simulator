import config
import utils


class Agent:
    def __init__(self, brain, motivation=config.RewardType.WATER, motivated_reward_value=1, non_motivated_reward_value=0.3):

        self._brain = brain
        self._memory_size = 10240
        self._memory = utils.ReplayMemory(self._memory_size)
        self._motivation = motivation
        self._motivated_reward_value = motivated_reward_value
        self._non_motivated_reward_value = non_motivated_reward_value

    def get_brain(self):
        return self._brain

    def decide(self, state):
        decision = self._brain.think(state)
        # action_prob = utils.normalize_dist((1-eps)*self.fitrah() + eps*brain_actions_prob)
        #action = utils.dist_selection(decision)
        action = utils.epsilon_greedy(config.EXPLORATION_EPSILON, decision)
        return action

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

    def get_motivation(self):
        return self._motivation

    def get_memory(self):
        return self._memory

    def get_internal_state(self):
        if self._motivation == config.RewardType.WATER:
            return [-1]
        else:
            return [1]