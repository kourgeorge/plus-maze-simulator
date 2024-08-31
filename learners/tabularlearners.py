__author__ = 'gkour'

from abc import ABC

import numpy as np

import config
import utils
from learners.abstractlearner import AbstractLearner, SymmetricLearner
from models.tabularmodels import QTable, FTable, ACFTable
from rewardtype import RewardType


class AsymmetricQLearner(AbstractLearner):
    def __init__(self, model: QTable, lr=config.LEARNING_RATE, lr_nr=config.LEARNING_RATE, *args, **kwargs):
        super().__init__(model=model, optimizer={'learning_rate_rewarded': lr, 'learning_rate_nonrewarded':lr_nr}, *args, **kwargs)

    def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation):
        learning_rate_rewarded = self.optimizer['learning_rate_rewarded']
        learning_rate_nonrewarded = self.optimizer['learning_rate_nonrewarded']
        actions = np.argmax(action_batch, axis=1)
        all_action_values = self.model(state_batch, motivation)
        selected_action_value = all_action_values[np.arange(len(all_action_values)), actions]
        deltas = (reward_batch - selected_action_value)
        # Apply the learning rate based on whether the reward was given (1) or not
        learning_rates = np.where(reward_batch == 1, learning_rate_rewarded, learning_rate_nonrewarded)

        # Update the Q-values using the appropriate learning rate for each sample
        updated_q_values = selected_action_value + deltas * learning_rates

        self.model.set_actual_state_value(state_batch, actions, updated_q_values, motivation)

        return {'delta': deltas[0]}

    def get_parameters(self):
        return {
            'learning_rate': self.optimizer['learning_rate']
        }


class QLearner(AsymmetricQLearner, SymmetricLearner):
    def __init__(self, model: QTable, lr=config.LEARNING_RATE, *args, **kwargs):
        super().__init__(model=model, lr=lr, lr_nr=lr, *args, **kwargs)


class AsymmetricIALearner(AbstractLearner):
    """Immutable Attention Learner"""

    def __init__(self, model: FTable, lr=config.LEARNING_RATE, lr_nr=config.LEARNING_RATE, *args, **kwargs):
        super().__init__(model=model, optimizer={'learning_rate_rewarded': lr,
                                                 'learning_rate_nonrewarded': lr_nr})

    def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation):

        learning_rate_rewarded = self.optimizer['learning_rate_rewarded']
        learning_rate_nonrewarded = self.optimizer['learning_rate_nonrewarded']

        actions = np.argmax(action_batch, axis=1)

        # Calculate the Q function for all actions
        all_action_values = self.model(state_batch, motivation)

        # calculate the Q function for selected action
        selected_action_value = all_action_values[np.arange(len(all_action_values)), actions]

        if np.any(np.isinf(selected_action_value)):
            print('Warning! rat Selected inactive door!')
            return 0
        delta = (reward_batch - selected_action_value)
        selected_odors, selected_colors = self.model.get_selected_door_stimuli(state_batch, actions)

        phi = self.model.phi()
        learning_rate = np.where(reward_batch == 1, learning_rate_rewarded, learning_rate_nonrewarded)
        for odor in set(np.unique(selected_odors)).difference([self.model.encoding_size]):
            self.model.update_stimulus_value('odors', odor, motivation,
                                             learning_rate * phi[0] * np.nanmean(delta[selected_odors == odor]))
        for color in set(np.unique(selected_colors)).difference([self.model.encoding_size]):
            self.model.update_stimulus_value('colors', color, motivation,
                                             learning_rate * phi[1] * np.nanmean(delta[selected_colors == color]))
        for door in np.unique(actions):
            self.model.update_stimulus_value('spatial', door, motivation,
                                             learning_rate * phi[2] * np.nanmean(delta[actions == door]))

        return {"delta": delta[0]}

    def get_parameters(self):
        return {
            'lr': self.optimizer['learning_rate_rewarded'],
            'lr_nr': self.optimizer['learning_rate_nonrewarded']
        }


class IALearner(AsymmetricIALearner, SymmetricLearner):
    """Immutable Attention Learner"""

    def __init__(self, model: FTable, lr=config.LEARNING_RATE, *args, **kwargs):
        super().__init__(model=model, lr=lr, lr_nr=lr, *args, **kwargs)

    def get_parameters(self):
        return {
            'lr': self.optimizer['learning_rate_rewarded'],
        }


class AsymmetricMALearner(AsymmetricIALearner):
    """Mutable Attention Learner"""

    def __init__(self, model: ACFTable, attention_lr=config.LEARNING_RATE, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.alpha_phi = attention_lr

    def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation):
        actions = np.argmax(action_batch, axis=1)

        all_action_values = self.model(state_batch, motivation)
        selected_action_value = all_action_values[np.arange(len(all_action_values)), actions]

        deltas = (reward_batch - selected_action_value)
        selected_odors, selected_colors = self.model.get_selected_door_stimuli(state_batch, actions)

        phi = self.model.phi()

        optimization_data = {}
        delta_phi = np.empty((0, 3))
        V = np.empty((0, 3))
        regrets = np.empty((0, 3))

        # Calculate the attention updates
        for choice_value, selected_odor, selected_color, selected_door, delta, reward in \
                zip(selected_action_value, selected_odors, selected_colors, actions, deltas, reward_batch):
            V = np.vstack([V, [self.model.Q[motivation.value]['odors'][selected_odor],
                               self.model.Q[motivation.value]['colors'][selected_color],
                               self.model.Q[motivation.value]['spatial'][selected_door]]])
            attention_regret = self.calc_attention_regret(choice_value, reward, V)
            delta_phi = np.vstack([delta_phi, delta * phi * attention_regret])
            regrets = np.vstack([regrets, attention_regret])

        # update values using old attentions
        super().learn(state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation)

        # update attention importance
        self.model.attn_importance += self.alpha_phi * np.mean(delta_phi, axis=0)

        optimization_data['delta'] = np.mean(deltas)
        optimization_data['odor_regret'] = np.mean(regrets[:, 0])
        optimization_data['color_regret'] = np.mean(regrets[:, 1])
        optimization_data['spatial_regret'] = np.mean(regrets[:, 2])
        optimization_data['Q'] = np.mean(selected_action_value)
        optimization_data['odor_V'] = np.mean(V[:, 0])
        optimization_data['color_V'] = np.mean(V[:, 1])
        optimization_data['spatial_V'] = np.mean(V[:, 2])

        optimization_data['odor_importance'] = self.model.attn_importance[0]
        optimization_data['color_importance'] = self.model.attn_importance[1]
        optimization_data['spatial_importance'] = self.model.attn_importance[2]

        phi = self.model.phi()
        optimization_data['odor_phi'] = phi[0]
        optimization_data['color_phi'] = phi[1]
        optimization_data['spatial_phi'] = phi[2]

        optimization_data['odor_delta_phi'] = np.mean(delta_phi[:, 0])
        optimization_data['color_delta_phi'] = np.mean(delta_phi[:, 1])
        optimization_data['spatial_delta_phi'] = np.mean(delta_phi[:, 2])

        return optimization_data

    def calc_attention_regret(self, Q, reward, V):
        return V - Q


class MALearner(AsymmetricMALearner, SymmetricLearner):
    def __init__(self, model: ACFTable, lr=config.LEARNING_RATE, *args, **kwargs):
        super().__init__(model,  lr=lr, lr_nr=lr, *args, **kwargs)

    def get_parameters(self):
        return {
            'lr': self.optimizer['learning_rate_rewarded'],
            'lr_nr': self.optimizer['learning_rate_nonrewarded']
        }


class ABIALearner(IALearner):
    def __init__(self, model: FTable, lr=config.LEARNING_RATE, alpha_bias=None, *args, **kwargs):
        super().__init__(model=model, learning_rate=lr, *args, **kwargs)
        self.alpha_bias = alpha_bias if alpha_bias is not None else lr

    def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation):
        deltas = super().learn(state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation)
        actions = np.argmax(action_batch, axis=1)

        for action in np.unique(actions):
            new_action_bias_value = self.model.get_bias_values(motivation)[action] + \
                                    self.alpha_bias * np.mean(deltas[actions == action])
            self.model.set_bias_value(motivation, action, new_action_bias_value)

        return deltas


class UABIALearner(IALearner):
    def __init__(self, model: FTable, alpha_bias=config.LEARNING_RATE, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        self.alpha_bias = alpha_bias

    def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation):
        deltas = super().learn(state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation)
        actions = np.argmax(action_batch, axis=1)

        for action in np.unique(actions):
            new_action_bias_value = self.model.get_bias_values(motivation)[action] + \
                                    self.alpha_bias * np.mean(deltas[actions == action])
            self.model.set_bias_value(RewardType.FOOD, action, new_action_bias_value)
            self.model.set_bias_value(RewardType.WATER, action, new_action_bias_value)

        return deltas


class ABQLearner(QLearner):
    def __init__(self, model: QTable, lr=config.LEARNING_RATE, alpha_bias=None, *args, **kwargs):
        super().__init__(model=model, learning_rate=lr, *args, **kwargs)
        self.alpha_bias = alpha_bias if alpha_bias is not None else lr

    def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation):
        deltas = super().learn(state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation)
        actions = np.argmax(action_batch, axis=1)

        for action in np.unique(actions):
            new_action_bias_value = self.model.get_bias_values(motivation)[action] + \
                                    self.alpha_bias * np.mean(deltas[actions == action])
            self.model.set_bias_value(motivation, action, new_action_bias_value)

        return deltas


class UABQLearner(QLearner):
    def __init__(self, model: QTable, learning_rate=config.LEARNING_RATE, alpha_bias=None, *args, **kwargs):
        super().__init__(model=model, learning_rate=learning_rate, alpha_bias=None, *args, **kwargs)
        self.alpha_bias = alpha_bias if alpha_bias is not None else learning_rate

    def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation):
        deltas = super().learn(state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation)
        actions = np.argmax(action_batch, axis=1)

        for action in np.unique(actions):
            self.model.action_bias[RewardType.WATER.value][action] += self.alpha_bias * np.mean(
                deltas[actions == action])
            self.model.action_bias[RewardType.FOOD.value][action] += self.alpha_bias * np.mean(
                deltas[actions == action])

        return deltas


class IAAluisiLearner(AbstractLearner):
    def __init__(self, model: FTable, lr=config.LEARNING_RATE):
        super().__init__(model=model, optimizer={'learning_rate': lr})

    def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation):

        learning_rate = self.optimizer['learning_rate']
        actions = np.argmax(action_batch, axis=1)

        # Calculate the Q function for all actions
        all_action_values = self.model(state_batch, motivation)

        # calculate the Q function for selected action
        selected_action_value = all_action_values[np.arange(len(all_action_values)), actions]

        if np.any(np.isinf(selected_action_value)):
            print('Warning! rat Selected inactive door!')
            return 0
        delta = (reward_batch - selected_action_value)
        selected_odors, selected_colors = self.model.get_selected_door_stimuli(state_batch, actions)

        for odor in set(np.unique(selected_odors)).difference([self.model.encoding_size]):
            self.model.Q['odors'][odor] += \
                learning_rate * np.nanmean(reward_batch[selected_odors == odor] - self.model.Q['odors'][odor])
        for color in set(np.unique(selected_colors)).difference([self.model.encoding_size]):
            self.model.Q['colors'][color] += \
                learning_rate * np.nanmean(reward_batch[selected_colors == color] - self.model.Q['colors'][color])
        for door in np.unique(actions):
            self.model.Q['spatial'][door] += \
                learning_rate * np.nanmean(reward_batch[actions == door] - self.model.Q['spatial'][door])

        return delta





class ABMALearner(MALearner):

    def __init__(self, model: ACFTable, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)

    def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation):
        deltas = super().learn(state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation)
        actions = np.argmax(action_batch, axis=1)
        learning_rate = self.optimizer['learning_rate']

        for action in np.unique(actions):
            self.model.action_bias[motivation.value][action] += learning_rate * np.mean(deltas[actions == action])

        return deltas


class UABMALearner(MALearner):

    def __init__(self, model: ACFTable, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)

    def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation):
        deltas = super().learn(state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation)
        actions = np.argmax(action_batch, axis=1)
        learning_rate = self.optimizer['learning_rate']

        for action in np.unique(actions):
            self.model.action_bias[RewardType.WATER.value][action] += learning_rate * np.mean(deltas[actions == action])
            self.model.action_bias[RewardType.FOOD.value][action] += learning_rate * np.mean(deltas[actions == action])

        return deltas


class MALearnerSimple(MALearner):
    def calc_attention_regret(self, Q, reward, V):
        return V - reward
