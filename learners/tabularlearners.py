__author__ = 'gkour'

import numpy as np
import copy

import utils
from learners.abstractlearner import AbstractLearner
from models.tabularmodels import TabularQ, UniformAttentionTabular, AttentionAtChoiceAndLearningTabular


class TD(AbstractLearner):
	def __init__(self, model: TabularQ, learning_rate=0.01):
		super().__init__(model=model, optimizer={'learning_rate': learning_rate})

	def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch):
		learning_rate = self.optimizer['learning_rate']
		actions = np.argmax(action_batch, axis=1)
		q_values = np.max(np.multiply(self.model(state_batch), action_batch), axis=1)
		deltas = (reward_batch - q_values)
		updated_q_values = q_values + learning_rate * deltas

		for state, action, update_q_value in zip(state_batch, actions, updated_q_values):
			self.model.set_state_action_value(state, action, update_q_value)

		return np.mean(deltas)


class TDUniformAttention(AbstractLearner):
	def __init__(self, model: UniformAttentionTabular, learning_rate=0.01):
		super().__init__(model=model, optimizer={'learning_rate': learning_rate})

	def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch):
		learning_rate = self.optimizer['learning_rate']
		actions = np.argmax(action_batch, axis=1)

		a = self.model(state_batch)
		v_all_dims = a[np.arange(len(a)), actions]

		deltas = (reward_batch - v_all_dims)
		selected_odors, selected_colors = self.model.get_selected_door_stimuli(state_batch, actions)

		phi = utils.softmax(self.model.phi)
		for odor in set(np.unique(selected_odors)).difference([self.model.encoding_size]):
			self.model.V['odors'][odor] = self.model.V['odors'][odor] + \
										  np.nanmean(learning_rate * phi[0] * deltas[selected_odors == odor])
		for color in set(np.unique(selected_colors)).difference([self.model.encoding_size]):
			self.model.V['colors'][color] = self.model.V['colors'][color] + \
											np.nanmean(learning_rate * phi[1] * deltas[selected_colors == color])
		for door in np.unique(actions):
			self.model.V['spatial'][door] = self.model.V['spatial'][door] + \
											np.nanmean(learning_rate * phi[2] * deltas[actions == door])

		return deltas


class TDAttentionAtLearning(TDUniformAttention):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch):

		super().learn(state_batch, action_batch, reward_batch, action_values, nextstate_batch)
		learning_rate = self.optimizer['learning_rate']
		actions = np.argmax(action_batch, axis=1)

		a = self.model(state_batch)
		v_all_dims = a[np.arange(len(a)), actions]

		deltas = (reward_batch - v_all_dims)
		selected_odors, selected_colors = self.model.get_selected_door_stimuli(state_batch, actions)

		phi_s = utils.softmax(self.model.phi)

		for selected_odor, selected_color, selected_door, delta in zip(selected_odors, selected_colors, actions, deltas):
			v = np.array([self.model.V['odors'][selected_odor],
						  self.model.V['colors'][selected_color],
						  self.model.V['spatial'][selected_door]])

			# delta_phi = 2 * phi_s * v * delta * (1 - phi_s)

			delta_phi = np.array([2  * delta * np.matmul(v, [1-phi_s[0], -phi_s[1], -phi_s[2]]),
						2 * delta * np.matmul(v, [-phi_s[0], 1-phi_s[1], -phi_s[2]]),
						2  * delta * np.matmul(v, [-phi_s[0], -phi_s[1], 1-phi_s[2]])])

			alpha = learning_rate
			self.model.phi = (1-alpha) *self.model.phi + alpha * delta_phi

		return deltas
