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

		phi = utils.softmax(self.model._phi)
		for odor in np.unique(selected_odors):
			self.model.V['odors'][odor] = self.model.V['odors'][odor] + \
										  np.mean(learning_rate * phi[0] * deltas[selected_odors == odor])
		for color in np.unique(selected_colors):
			self.model.V['colors'][color] = self.model.V['colors'][color] + \
											np.mean(learning_rate * phi[1] * deltas[selected_colors == color])
		for door in np.unique(actions):
			self.model.V['spatial'][door] = self.model.V['spatial'][door] + \
											np.mean(learning_rate * phi[2] * deltas[actions == door])

		if isinstance(self.model, AttentionAtChoiceAndLearningTabular):
			self.model._phi = self.model._phi + utils.softmax(self.model._phi) * learning_rate * np.mean(deltas)

		return np.mean(deltas**2)
