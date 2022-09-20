__author__ = 'gkour'

import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

import utils
from utils import compress

norm = 'fro'


class TabularQ:

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
		return state_actions_value

	def set_state_action_value(self, state, action, value):
		obs = utils.states_encoding_to_cues(state, self.encoding_size)
		self.Q[obs.tostring()][action] = value

	def get_model_metrics(self):
		return {'num_entries': len(self.Q.keys())}

	def get_model_diff(self, brain2):
		diff = [np.linalg.norm(self.Q[state] - brain2.Q[state]) for state in
				set(self.Q.keys()).intersection(brain2.Q.keys())]
		return {'table_change': np.mean(diff)*100}

	def __str__(self):
		return self.__class__.__name__


class UniformAttentionTabular:
	def __str__(self):
		return self.__class__.__name__

	def __init__(self, encoding_size, num_channels, num_actions):
		self.encoding_size=encoding_size
		self._num_actions = num_actions
		self.V = dict()
		self.V['odors'] = np.zeros([encoding_size+1])
		self.V['colors'] = np.zeros([encoding_size+1])
		self.V['spatial'] = np.zeros([4])
		self.phi = np.ones([3])/3

	def __call__(self, *args, **kwargs):
		states = args[0]
		batch = states.shape[0]

		cues = utils.states_encoding_to_cues(states, self.encoding_size)
		odor = cues[:, 0]
		color = cues[:, 1]
		data = np.stack([self.odor_value(odor), self.color_value(color),
						 np.repeat(np.expand_dims(self.spatial_value(np.array(range(4))), axis=0), repeats=batch, axis=0)])
		doors_value = np.matmul(np.expand_dims(utils.softmax(self.phi), axis=0), np.transpose(data, axes=(1, 0, 2)))
		return np.squeeze(doors_value, axis=1)

	def get_selected_door_stimuli(self, states, doors):
		cues = utils.states_encoding_to_cues(states, self.encoding_size)
		selected_cues = cues[np.arange(len(states)), :, doors]
		return selected_cues[:, 0], selected_cues[:, 1]

	def odor_value(self, odors):
		return self.V['odors'][odors]

	def color_value(self, colors):
		return np.array(self.V['colors'])[colors]

	def spatial_value(self, doors):
		return np.array(self.V['spatial'])[doors]

	def get_model_metrics(self):
		return {'color': np.linalg.norm(self.V['colors']),
				'odor':  np.linalg.norm(self.V['odors']),
				'spatial':  np.linalg.norm(self.V['spatial'])}

	def get_model_diff(self, brain2):
		return {'color_change': entropy(self.V['colors'], brain2.V['colors']),
				'odor_change':  jensenshannon(self.V['odors'], brain2.V['odors']),
				'spatial_change':  jensenshannon(self.V['spatial'], brain2.V['spatial'])}


class AttentionAtChoiceAndLearningTabular(UniformAttentionTabular):
	def __init__(self,encoding_size, num_channels, num_actions):
		super().__init__(encoding_size, num_channels, num_actions)
		#self.phi = np.random.normal(loc=0.0, scale=1.0, size=3)

	def get_model_metrics(self):
		return {'color_attn': self.phi[0],
				'odor_attn': self.phi[1],
				'spatial_attn': self.phi[2]}

