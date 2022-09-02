import numpy as np
import config
from abstractbrain import AbstractBrain
from learner import AbstractLearner
import torch


class TD(AbstractLearner):

    def __init__(self, model,learning_rate=0.01):
        super().__init__(model=model, optimizer={'learning_rate':learning_rate})

    def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch):
        # states_q_value = [self._qvalue[state.tostring()] for state in state_batch]
        # q_values =[utils.dot_lists(states_q_value[i], action_batch[i]) for i in range(minibatch_size)]
        # Compute V(s_{t+1}) for all next states.

        learning_rate = self.optimizer['learning_rate']
        actions = np.argmax(action_batch, axis=1)
        q_values = np.max(np.multiply(self.model(state_batch), action_batch), axis=1)
        deltas = (reward_batch - q_values)
        updated_q_values = q_values + learning_rate * deltas

        for state, action, update_q_value in zip(state_batch, actions, updated_q_values):
            self.model.set_state_action_value(state, action, update_q_value)

        return np.mean(deltas)


class TableDQNBrain(AbstractBrain):

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


