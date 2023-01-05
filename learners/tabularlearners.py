__author__ = 'gkour'

import numpy as np
import copy

import utils
from learners.abstractlearner import AbstractLearner
from models.tabularmodels import QTable, FTable, ACFTable, OptionsTable


class QLearner(AbstractLearner):
	def __init__(self, model: QTable, learning_rate=0.01):
		super().__init__(model=model, optimizer={'learning_rate': learning_rate})

	def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch):
		learning_rate = self.optimizer['learning_rate']
		actions = np.argmax(action_batch, axis=1)
		all_action_values = self.model(state_batch)
		selected_action_value = all_action_values[np.arange(len(all_action_values)), actions]
		deltas = (reward_batch - selected_action_value)
		updated_q_values = selected_action_value + learning_rate * deltas

		for state, action, update_q_value in zip(state_batch, actions, updated_q_values):
			self.model.set_state_action_value(state, action, update_q_value)

		return np.mean(deltas)


class OptionsLearner(AbstractLearner):
	def __init__(self, model: OptionsTable, learning_rate=0.01):
		super().__init__(model=model, optimizer={'learning_rate': learning_rate})

	def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch):
		learning_rate = self.optimizer['learning_rate']
		actions = np.argmax(action_batch, axis=1)
		all_action_values = self.model(state_batch)
		selected_action_value = all_action_values[np.arange(len(all_action_values)), actions]
		deltas = (reward_batch - selected_action_value)
		updated_q_values = selected_action_value + learning_rate * deltas

		for state, action, update_q_value in zip(state_batch, actions, updated_q_values):
			self.model.set_option_value(state, action, update_q_value)
		return np.mean(deltas)


class IALearner(AbstractLearner):
	def __init__(self, model: FTable, learning_rate=0.01):
		super().__init__(model=model, optimizer={'learning_rate': learning_rate})

	def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch):

		learning_rate = self.optimizer['learning_rate']
		actions = np.argmax(action_batch, axis=1)

		# Calculate the Q function for all actions
		all_action_values = self.model(state_batch)

		# calculate the Q function for selected action
		selected_action_value = all_action_values[np.arange(len(all_action_values)), actions]

		if np.any(np.isinf(selected_action_value)):
			print('Warning! rat Selected inactive door!')
			return 0
		delta = (reward_batch - selected_action_value)
		selected_odors, selected_colors = self.model.get_selected_door_stimuli(state_batch, actions)

		phi = utils.softmax(self.model.phi) if isinstance(self.model, ACFTable) else [1/3, 1/3, 1/3]
		#phi = self.model.phi # self.model.phi if isinstance(self.model, ACFTable) else [1, 1, 1]
		for odor in set(np.unique(selected_odors)).difference([self.model.encoding_size]):
			self.model.V['odors'][odor] += learning_rate * phi[0] * np.nanmean(delta[selected_odors == odor])
		for color in set(np.unique(selected_colors)).difference([self.model.encoding_size]):
			self.model.V['colors'][color] += learning_rate * phi[1] * np.nanmean(delta[selected_colors == color])
		for door in np.unique(actions):
			self.model.V['spatial'][door] += learning_rate * phi[2] * np.nanmean(delta[actions == door])

		return delta


class IAAluisiLearner(AbstractLearner):
	def __init__(self, model: FTable, learning_rate=0.01):
		super().__init__(model=model, optimizer={'learning_rate': learning_rate})

	def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch):

		learning_rate = self.optimizer['learning_rate']
		actions = np.argmax(action_batch, axis=1)

		# Calculate the Q function for all actions
		all_action_values = self.model(state_batch)

		# calculate the Q function for selected action
		selected_action_value = all_action_values[np.arange(len(all_action_values)), actions]

		if np.any(np.isinf(selected_action_value)):
			print('Warning! rat Selected inactive door!')
			return 0
		delta = (reward_batch - selected_action_value)
		selected_odors, selected_colors = self.model.get_selected_door_stimuli(state_batch, actions)

		for odor in set(np.unique(selected_odors)).difference([self.model.encoding_size]):
			self.model.V['odors'][odor] += \
				learning_rate * np.nanmean(reward_batch[selected_odors == odor] - self.model.V['odors'][odor])
		for color in set(np.unique(selected_colors)).difference([self.model.encoding_size]):
			self.model.V['colors'][color] += \
				learning_rate * np.nanmean(reward_batch[selected_colors == color] - self.model.V['colors'][color])
		for door in np.unique(actions):
			self.model.V['spatial'][door] += \
				 learning_rate * np.nanmean(reward_batch[actions == door] - self.model.V['spatial'][door])

		return delta


class MALearner(IALearner):
	def __init__(self, model, alpha_phi=0.005, *args, **kwargs):
		super().__init__(model, *args, **kwargs)
		self.alpha_phi = alpha_phi

	def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch):
		oldV = copy.deepcopy(self.model.V)

		super().learn(state_batch, action_batch, reward_batch, action_values, nextstate_batch)
		# learning_rate = self.optimizer['learning_rate']
		# delta_value = [1/learning_rate*np.linalg.norm(self.model.V[dimension]-oldV[dimension], ord=1) for dimension in oldV.keys()]

		learning_rate = self.optimizer['learning_rate']
		actions = np.argmax(action_batch, axis=1)

		all_action_values = self.model(state_batch)
		selected_action_value = all_action_values[np.arange(len(all_action_values)), actions]

		deltas = (reward_batch - selected_action_value)
		selected_odors, selected_colors = self.model.get_selected_door_stimuli(state_batch, actions)

		phi_s = utils.softmax(self.model.phi)

		for selected_odor, selected_color, selected_door, delta, reward in zip(selected_odors, selected_colors, actions, deltas, reward_batch):
			V = np.array([self.model.V['odors'][selected_odor],
						  self.model.V['colors'][selected_color],
						  self.model.V['spatial'][selected_door]])
			delta_phi = self.calc_delta_phi(delta, reward, V, phi_s)

			self.model.phi += self.alpha_phi * delta * phi_s * delta_phi

		return deltas

	def calc_delta_phi(self, delta_v, reward, V, phi_s):
		"""
		:param delta_v: r-Q where Q is the total value
		:param reward: r
		:param V: the feature value
		:param phi_s: the normalized attention
		:return: delta_phi
		"""
		# return np.array([2 * delta_v * np.matmul(V, [1 - phi_s[0], -phi_s[1], -phi_s[2]]),
		# 				 2 * delta_v * np.matmul(V, [-phi_s[0], 1 - phi_s[1], -phi_s[2]]),
		# 				 2 * delta_v * np.matmul(V, [-phi_s[0], -phi_s[1], 1 - phi_s[2]])])
		# #
		return np.array([np.matmul(V, [1 - phi_s[0], -phi_s[1], -phi_s[2]]),
								 np.matmul(V, [-phi_s[0], 1 - phi_s[1], -phi_s[2]]),
								 np.matmul(V, [-phi_s[0], -phi_s[1], 1 - phi_s[2]])])


class MALearnerSimple(MALearner):
	def calc_delta_phi(self, delta_v, reward, V, phi_s):
		return -delta_v * (reward - V)


