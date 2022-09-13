__author__ = 'gkour'

import torch

from learners.abstractlearner import AbstractLearner
from models.tabularmodels import *

torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

		phi = self.model.phi
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
		deltas = super().learn(state_batch, action_batch, reward_batch, action_values, nextstate_batch)
		delta = np.mean(deltas)
		learning_rate = self.optimizer['learning_rate']
		beta = 0.5 * (delta + 1)
		delta_phi = (1 - beta) * np.ones([3]) / 3 + beta * np.eye(3)[
			np.argmax(self.model.phi)]  # np.softmax(np.eye(np.argmax(self.model._phi)), beta)
		# print("delta:{}, beta:{}, delta_phi:{} ".format(delta,beta, delta_phi ))
		old_phi = self.model.phi
		self.model.phi = (1 - learning_rate) * self.model.phi + learning_rate * delta_phi
		diff_phi = self.model.phi - old_phi
		return delta
