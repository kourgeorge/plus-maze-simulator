import numpy as np
import utils


class TableDQNBrain():
    BATCH_SIZE = 10

    def __init__(self, num_actions, reward_discount=0, learning_rate=0.01):

        self._qvalue = dict()
        self._num_actions = num_actions
        self._reward_discount = reward_discount
        self._learning_rate = learning_rate
        self.num_optimizations = 0

    def think(self, state):
        if state.tostring() not in self._qvalue.keys():
            self._qvalue[state.tostring()] = 0.5 * np.ones(self._num_actions)
        state_actions_value = self._qvalue[state.tostring()]
        return utils.softmax(state_actions_value)

    def train(self, memory):
        minibatch_size = min(TableDQNBrain.BATCH_SIZE, len(memory))
        if minibatch_size == 0:
            return
        self.num_optimizations += 1

        minibatch = memory.sample(minibatch_size)
        # minibatch = memory.last(minibatch_size)

        # states_q_value = [self._qvalue[state.tostring()] for state in state_batch]
        # state_action_values =[utils.dot_lists(states_q_value[i], action_batch[i]) for i in range(minibatch_size)]
        # Compute V(s_{t+1}) for all next states.

        loss = 0
        for i in range(minibatch_size):
            state = minibatch[i][0]
            action = np.argmax(minibatch[i][1])
            reward = minibatch[i][2]
            nextstate = minibatch[i][3]
            terminal = minibatch[i][4]

            current_action_value = self._qvalue[state.tostring()][action]
            if not terminal:
                current_action_value = (1 - self._reward_discount) * current_action_value + \
                                       self._reward_discount * np.max(self._qvalue[nextstate.tostring()])

            self._qvalue[state.tostring()][action] = (1 - self._learning_rate) * current_action_value + \
                                                     self._learning_rate * reward
            loss += (current_action_value - reward) ** 2

        return loss/minibatch_size
