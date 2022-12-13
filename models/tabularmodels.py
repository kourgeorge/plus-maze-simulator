__author__ = 'gkour'

import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

import utils

norm = 'fro'


class QTable:

	def __init__(self, encoding_size, num_channels, num_actions):
		self._num_actions = num_actions
		self.Q = dict()
		self.encoding_size = encoding_size

	def __call__(self, *args, **kwargs):
		state = args[0]
		state_actions_value = []
		for obs in utils.states_encoding_to_cues(state, self.encoding_size):
			if obs.tostring() not in self.Q.keys():
				self.Q[obs.tostring()] = 0.5 * np.ones(self._num_actions)
			state_actions_value.append(self.Q[obs.tostring()])
		cues = utils.states_encoding_to_cues(state, self.encoding_size)
		odor = cues[:, 0]  # odor for each door
		state_actions_value = np.array(state_actions_value)
		state_actions_value[odor == self.encoding_size] = -np.inf
		return state_actions_value

	def set_state_action_value(self, state, action, value):
		obs = utils.states_encoding_to_cues(state, self.encoding_size)
		self.Q[obs.tostring()][action] = value

	def get_model_metrics(self):
		return {'num_entries': len(self.Q.keys())}

	def get_model_diff(self, brain2):
		diff = [np.linalg.norm(self.Q[state] - brain2.Q[state]) for state in
				set(self.Q.keys()).intersection(brain2.Q.keys())]
		return {'table_change': np.mean(diff) * 100}

	def __str__(self):
		return self.__class__.__name__


class FTable:

	def __init__(self, encoding_size, num_channels, num_actions):
		self.encoding_size = encoding_size
		self._num_actions = num_actions
		self.V = dict()
		self.V['odors'] = np.zeros([encoding_size + 1])
		self.V['colors'] = np.zeros([encoding_size + 1])
		self.V['spatial'] = np.zeros([4])

	def __call__(self, *args, **kwargs):
		states = args[0]

		cues = utils.states_encoding_to_cues(states, self.encoding_size)
		odor = cues[:, 0]  # odor for each door
		color = cues[:, 1]  # color for each door
		door = np.array(range(4))
		action_values = (self.stimuli_value('odors', odor) + \
						self.stimuli_value('colors', color) + \
						self.stimuli_value('spatial', door))
		action_values[odor == self.encoding_size]=-np.inf #avoid selecting inactive doors.
		return action_values

	def get_selected_door_stimuli(self, states, doors):
		cues = utils.states_encoding_to_cues(states, self.encoding_size)
		selected_cues = cues[np.arange(len(states)), :, doors]
		return selected_cues[:, 0], selected_cues[:, 1]

	def stimuli_value(self, dimension, feature):
		return self.V[dimension][feature]

	def get_model_metrics(self):
		return {'odor': np.linalg.norm(self.V['odors']),
				'color': np.linalg.norm(self.V['colors']),
				'spatial': np.linalg.norm(self.V['spatial'])
				}

	def get_model_diff(self, brain2):
		return {'odor_diff': jensenshannon(self.V['odors'], brain2.V['odors']),
				'color_diff': entropy(self.V['colors'], brain2.V['colors']),
				'spatial_diff': jensenshannon(self.V['spatial'], brain2.V['spatial'])}


class ACFTable(FTable):
	def __init__(self, encoding_size, num_channels, num_actions):
		super().__init__(encoding_size, num_channels, num_actions)
		self.phi = np.ones([3]) / 3

	def __call__(self, *args, **kwargs):
		states = args[0]
		batch = states.shape[0]

		cues = utils.states_encoding_to_cues(states, self.encoding_size)
		odor = cues[:, 0]
		color = cues[:, 1]
		data = np.stack([self.stimuli_value('odors', odor), self.stimuli_value('colors', color),
						 np.repeat(np.expand_dims(self.stimuli_value('spatial', np.array(range(4))), axis=0),
								   repeats=batch, axis=0)])
		attention = np.expand_dims(utils.softmax(self.phi), axis=0)
		doors_value = np.matmul(attention, np.transpose(data, axes=(1, 0, 2)))
		doors_value = np.squeeze(doors_value, axis=1)
		doors_value[odor == self.encoding_size] = -np.inf  # avoid selecting inactive doors.

		return doors_value

	def get_selected_door_stimuli(self, states, doors):
		cues = utils.states_encoding_to_cues(states, self.encoding_size)
		selected_cues = cues[np.arange(len(states)), :, doors]
		return selected_cues[:, 0], selected_cues[:, 1]

	def get_model_metrics(self):
		phi = utils.softmax(self.phi)
		return {'odor_attn': phi[0],
				'color_attn': phi[1],
				'spatial_attn': phi[2],
				}

	def get_model_diff(self, brain2):
		return {'odor_attn_diff': self.phi[0]-brain2.phi[0],
				'color_attn_diff': self.phi[1]-brain2.phi[1],
				'spatial_attn_diff': self.phi[2]-brain2.phi[2],}


class PCFTable(ACFTable):
	def __init__(self, encoding_size, num_channels, num_actions):
		super().__init__(encoding_size, num_channels, num_actions)

	def __call__(self, *args, **kwargs):
		states = args[0]
		batch = states.shape[0]

		cues = utils.states_encoding_to_cues(states, self.encoding_size)
		odor = cues[:, 0]
		color = cues[:, 1]

		selected_dim = utils.epsilon_greedy(0.1,self.phi)
		doors_value = self.stimuli_value('odors', odor) if selected_dim == 0 else \
			self.stimuli_value('colors', color) if selected_dim == 1 \
				else self.stimuli_value('spatial', [np.array(range(4))])

		doors_value[odor == self.encoding_size] = -np.inf  # avoid selecting inactive doors.

		return doors_value

	def get_selected_door_stimuli(self, states, doors):
		cues = utils.states_encoding_to_cues(states, self.encoding_size)
		selected_cues = cues[np.arange(len(states)), :, doors]
		return selected_cues[:, 0], selected_cues[:, 1]

