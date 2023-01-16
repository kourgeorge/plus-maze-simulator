__author__ = 'gkour'

import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

import config
import utils
from collections import defaultdict

from rewardtype import RewardType

norm = 'fro'


class QTable:

	def __init__(self, encoding_size, num_channels, num_actions, initial_value=config.INITIAL_FEATURE_VALUE):
		self._num_actions = num_actions
		self.Q = defaultdict(lambda: initial_value * np.ones(self._num_actions))
		self.action_bias = np.zeros(self._num_actions)
		self.encoding_size = encoding_size

	def __call__(self, *args, **kwargs):
		state = args[0]
		state_actions_value = []
		for obs in utils.states_encoding_to_cues(state, self.encoding_size):
			#state_actions_value.append(self.Q[obs.tostring()])
			state_actions_value.append(self.Q[np.array2string(obs)])
		cues = utils.states_encoding_to_cues(state, self.encoding_size)
		odor = cues[:, 0]  # odor for each door
		state_actions_value = np.array(state_actions_value)
		state_actions_value += self.action_bias
		state_actions_value[odor == self.encoding_size] = -np.inf
		return state_actions_value

	def set_actual_state_value(self, state, action, value):
		obs = utils.states_encoding_to_cues(state, self.encoding_size)
		self.Q[np.array2string(obs)][action] = value

	def get_model_metrics(self):
		return {'num_entries': len(self.Q.keys())}

	def get_model_diff(self, brain2):
		diff = [np.linalg.norm(self.Q[state] - brain2.Q[state]) for state in
				set(self.Q.keys()).intersection(brain2.Q.keys())]
		return {'table_change': np.mean(diff) * 100}

	def __str__(self):
		return self.__class__.__name__


class OptionsTable:

	def __init__(self, encoding_size, num_channels, num_actions, use_location_cue=True, initial_value=config.INITIAL_FEATURE_VALUE):
		self._num_actions = num_actions
		self.C = defaultdict(lambda: float(initial_value)) # familiar options are stored as tupples (color, odor and possibly, location).
		self.action_bias = np.zeros(self._num_actions)
		self.encoding_size = encoding_size
		self.use_location_cue = use_location_cue

	def __call__(self, *args, **kwargs):
		states = args[0]
		state_actions_value = []
		for state in states:
			obs_action_value = []
			for option in self.get_cues_combinations(state):
				obs_action_value += [self.C[option]] if option[0] != self.encoding_size else [-np.inf]
			obs_action_value += self.action_bias
			state_actions_value.append(obs_action_value)
		return np.array(state_actions_value)

	def set_actual_state_value(self, state, action, value):
		option = self.get_cues_combinations(state)
		self.C[option[action]] = value

	def get_cues_combinations(self, state):
		state_cues = utils.states_encoding_to_cues(state, self.encoding_size)
		cues = list(zip(*state_cues))
		if self.use_location_cue:
			cues = [option+(a+1,) for a, option in enumerate(cues)]
		return cues

	def get_model_metrics(self):
		return {'action_bias': self.action_bias.tolist()}

	def get_model_diff(self, brain2):
		diff = [np.linalg.norm(self.C[state] - brain2.C[state]) for state in
				set(self.C.keys()).intersection(brain2.C.keys())]
		return {'table_change': np.mean(diff) * 100}

	def __str__(self):
		return self.__class__.__name__


class FTable:
	def __init__(self, encoding_size, num_channels, num_actions, initial_value=config.INITIAL_FEATURE_VALUE):
		self.encoding_size = encoding_size
		self._num_actions = num_actions
		self.V = dict()
		self.V['odors'] = initial_value*np.ones([encoding_size + 1])
		self.V['colors'] = initial_value*np.ones([encoding_size + 1])
		self.V['spatial'] = initial_value*np.ones([4])
		self.V['bias_W'] = np.zeros([4])
		self.V['bias_F'] = np.zeros([4])
		self.initial_value = initial_value

	def __call__(self, *args, **kwargs):
		states = args[0]
		motivation = args[1]

		cues = utils.states_encoding_to_cues(states, self.encoding_size)
		odor = cues[:, 0]  # odor for each door
		color = cues[:, 1]  # color for each door
		door = np.array(range(4))
		action_bias = self.V['bias_W'] if motivation==RewardType.WATER else self.V['bias_F']
		action_values = (self.stimuli_value('odors', odor) + \
						self.stimuli_value('colors', color) + \
						self.stimuli_value('spatial', door)) + action_bias
		action_values[odor == self.encoding_size]=-np.inf #avoid selecting inactive doors.
		return action_values


	def get_selected_door_stimuli(self, states, doors):
		cues = utils.states_encoding_to_cues(states, self.encoding_size)
		selected_cues = cues[np.arange(len(states)), :, doors]
		return selected_cues[:, 0], selected_cues[:, 1]

	def stimuli_value(self, dimension, feature):
		return self.V[dimension][feature]

	def new_stimuli_context(self):
		pass

	def get_model_metrics(self):
		#print("odor:{}\ncolor:{},\nspatial:{}".format(self.V['odors'], self.V['colors'], self.V['spatial']))
		# {'odor': entropy(utils.softmax(self.V['odors']))/(entropy(np.ones_like(self.V['odors']))/np.count_nonzero(self.V['odors'])),
		# 		'color': entropy(utils.softmax(self.V['colors']))/(entropy(np.ones_like(self.V['colors']))/np.count_nonzero(self.V['colors'])),
		# 		'spatial': entropy(utils.softmax(self.V['spatial']))/(entropy(np.ones_like(self.V['spatial']))/np.count_nonzero(self.V['spatial']))
		# 		}

		return {
			'odors': np.linalg.norm(self.V['odors']),
			'colors': np.linalg.norm(self.V['colors']),
			'spatial': np.linalg.norm(self.V['spatial'])
		}


	def get_model_diff(self, brain2):
		return {'odor_diff': jensenshannon(self.V['odors'], brain2.V['odors']),
				'color_diff': jensenshannon(self.V['colors'], brain2.V['colors']),
				'spatial_diff': jensenshannon(self.V['spatial'], brain2.V['spatial'])}

	def __str__(self):
		return self.__class__.__name__


class ACFTable(FTable):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.phi = np.ones([3]) / 3

	def __call__(self, *args, **kwargs):
		states = args[0]
		batch = states.shape[0]
		motivation = args[1]
		cues = utils.states_encoding_to_cues(states, self.encoding_size)
		odor = cues[:, 0]
		color = cues[:, 1]
		data = np.stack([self.stimuli_value('odors', odor), self.stimuli_value('colors', color),
						 np.repeat(np.expand_dims(self.stimuli_value('spatial', np.array(range(4))), axis=0),
								   repeats=batch, axis=0)])
		attention = np.expand_dims(utils.softmax(self.phi), axis=0)
		action_bias = self.V['bias_W'] if motivation == RewardType.WATER else self.V['bias_F']
		doors_value = np.matmul(attention, np.transpose(data, axes=(1, 0, 2))) + action_bias
		doors_value = np.squeeze(doors_value, axis=1)
		doors_value[odor == self.encoding_size] = -np.inf  # avoid selecting inactive doors.

		return doors_value

	def get_model_metrics(self):
		#print("odor:{}\ncolor:{},\nspatial:{}".format(self.V['odors'], self.V['colors'], self.V['spatial']))
		phi = utils.softmax(self.phi)
		return {'odor_attn': phi[0],
				'color_attn': phi[1],
				'spatial_attn': phi[2],
				}

	def get_model_diff(self, brain2):
		return {'odor_attn_diff': self.phi[0]-brain2.phi[0],
				'color_attn_diff': self.phi[1]-brain2.phi[1],
				'spatial_attn_diff': self.phi[2]-brain2.phi[2],}

	def new_stimuli_context(self):
		self.V['odors'] = self.initial_value * np.ones([self.encoding_size + 1])
		self.V['colors'] = self.initial_value * np.ones([self.encoding_size + 1])
		self.V['spatial'] = self.initial_value * np.ones([4])


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

