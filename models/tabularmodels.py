__author__ = 'gkour'

import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

import config
import utils
from collections import defaultdict

from rewardtype import RewardType

norm = 'fro'


class SCDependantV():
	'''
	Stimuli dependant Tabular model,
	It initializes the state value data when a new stimuli is encountered.
	'''
	def new_stimuli_context(self):
		self.initialize_state_values()

class SCDependantVB():
	'''
	Motivation dependant Tabular model.
	It initializes the state value and action bias data when a new stimuli is encountered.
	'''

	def new_stimuli_context(self):
		self.initialize_state_values()
		self.action_bias = {'water': np.zeros(self._num_actions), 'food': np.zeros(self._num_actions)}


class AbstractTabularModel:

	def __call__(self, *args, **kwargs):
		raise NotImplementedError()

	def get_model_metrics(self):
		raise NotImplementedError()

	def get_model_diff(self, brain2):
		raise NotImplementedError()

	def __str__(self):
		return self.__class__.__name__


class QTable(AbstractTabularModel):
	def __init__(self, encoding_size, num_channels, num_actions, initial_value=config.INITIAL_FEATURE_VALUE):
		super().__init__()
		self._num_actions = num_actions
		self.Q = defaultdict(lambda: initial_value * np.ones(self._num_actions))
		self.action_bias = {'water': np.zeros(self._num_actions), 'food': np.zeros(self._num_actions)}
		self.encoding_size = encoding_size

	def __call__(self, *args, **kwargs):
		env_obs = args[0]
		motivation = args[1]
		state_actions_value = []
		for state in self.state_representation(env_obs, motivation):
			state_actions_value.append(self.Q[np.array2string(state)])
		state_actions_value = np.array(state_actions_value)
		state_actions_value += self.action_bias[motivation.value]
		inactive_doors = utils.get_inactive_doors(env_obs)
		state_actions_value[inactive_doors] = -np.inf
		return state_actions_value

	def state_representation(self, obs, motivation):
		return utils.stimuli_1hot_to_cues(obs, self.encoding_size)

	def set_actual_state_value(self, obs, actions, values, motivation):
		states = self.state_representation(obs, motivation)
		for state, action, value in zip(states, actions, values):
			self.Q[np.array2string(state)][action] = value

	def get_model_metrics(self):
		res =  {'WMotivation_bias': utils.negentropy(self.action_bias['water']),
				'FMotivation_bias': utils.negentropy(self.action_bias['food'])}

		return res

	def get_model_diff(self, brain2):
		diff = [np.linalg.norm(self.Q[state] - brain2.Q[state]) for state in
				set(self.Q.keys()).intersection(brain2.Q.keys())]
		return {'table_change': np.mean(diff) * 100}


class MQTable(QTable):
	def __init__(self, encoding_size, num_channels, num_actions):
		super().__init__(encoding_size, num_channels, num_actions)

	def state_representation(self, obs, motivation):
		motivation_dimension = np.tile([motivation.value], (obs.shape[0], 1, 4))
		return np.append(utils.stimuli_1hot_to_cues(obs, self.encoding_size), motivation_dimension, axis=1)


class OptionsTable(AbstractTabularModel):

	def __init__(self, encoding_size, num_channels, num_actions, use_location_cue=True,
				 initial_value=config.INITIAL_FEATURE_VALUE):
		super().__init__()
		self._num_actions = num_actions
		self.C = defaultdict(lambda: float(
			initial_value))  # familiar options are stored as tupples (color, odor and possibly, location).
		self.action_bias = {'water': np.zeros(self._num_actions), 'food': np.zeros(self._num_actions)}
		self.encoding_size = encoding_size
		self.use_location_cue = use_location_cue

	def __call__(self, *args, **kwargs):
		states = args[0]
		motivation = args[1]
		state_actions_value = []
		for state in self.state_representation(states, motivation):
			obs_action_value = []
			for option in self.get_cues_combinations(state):
				obs_action_value += [self.C[option]] if option[0] != self.encoding_size else [-np.inf]
			obs_action_value += self.action_bias[motivation.value]
			state_actions_value.append(obs_action_value)
		return np.array(state_actions_value)

	def state_representation(self, obs, motivation):
		return utils.stimuli_1hot_to_cues(obs, self.encoding_size)

	def set_actual_state_value(self, obs, actions, values, motivation):
		states = self.state_representation(obs, motivation)
		for state, action, value in zip(states, actions, values):
			option = self.get_cues_combinations(state)
			self.C[option[action]] = value

	def get_cues_combinations(self, state):
		cues = list(zip(*state))
		if self.use_location_cue:
			cues = [option + (a + 1,) for a, option in enumerate(cues)]
		return cues

	def get_model_metrics(self):
		return {'WMotivation_bias': utils.negentropy(self.action_bias['water'].tolist()),
				'FMotivation_bias': utils.negentropy(self.action_bias['food'].tolist())}

	def get_model_diff(self, brain2):
		diff = [np.linalg.norm(self.C[state] - brain2.C[state]) for state in
				set(self.C.keys()).intersection(brain2.C.keys())]
		return {'table_change': np.mean(diff) * 100}

	def __str__(self):
		return self.__class__.__name__


class MOptionsTable(OptionsTable):
	def __init__(self, encoding_size, num_channels, num_actions):
		super().__init__(encoding_size, num_channels, num_actions)

	def state_representation(self, obs, motivation):
		motivation_dimension = np.tile([motivation.value], (obs.shape[0], 1, 4))
		return np.append(utils.stimuli_1hot_to_cues(obs, self.encoding_size), motivation_dimension, axis=1)


class FTable(AbstractTabularModel):
	def __init__(self, encoding_size, num_channels, num_actions, initial_value=config.INITIAL_FEATURE_VALUE):
		super().__init__()
		self.encoding_size = encoding_size
		self._num_actions = num_actions
		self.initial_value = initial_value
		self.Q = dict()
		self.action_bias = dict()
		self.initialize_state_values()
		for motivation in RewardType:
			self.action_bias[motivation.value] = np.zeros(self._num_actions)

	def initialize_state_values(self):
		for motivation in RewardType:
			self.Q[motivation.value] = dict()
			for stimuli in ['odors', 'colors', 'spatial']:
				self.Q[motivation.value][stimuli] = self.initial_value * np.ones([self.encoding_size + 1])

	def __call__(self, *args, **kwargs):
		states = args[0]
		motivation = args[1]

		cues = utils.stimuli_1hot_to_cues(states, self.encoding_size)
		odor = cues[:, 0]  # odor for each door
		color = cues[:, 1]  # color for each door
		door = np.array(range(4))
		action_values = (self.get_stimulus_value('odors', odor, motivation) +
						 self.get_stimulus_value('colors', color, motivation) +
						 self.get_stimulus_value('spatial', door, motivation)) + self.action_bias[motivation.value]
		action_values[odor == self.encoding_size] = -np.inf  # avoid selecting inactive doors.
		return action_values

	def get_selected_door_stimuli(self, states, doors):
		cues = utils.stimuli_1hot_to_cues(states, self.encoding_size)
		selected_cues = cues[np.arange(len(states)), :, doors]
		return selected_cues[:, 0], selected_cues[:, 1]

	def get_stimulus_value(self, dimension, feature, motivation):
		return self.Q[RewardType.NONE.value][dimension][feature]

	def set_stimulus_value(self, dimension, feature, motivation, new_value):
		self.Q[RewardType.NONE.value][dimension][feature] = new_value

	def update_stimulus_value(self, dimension, feature, motivation, delta):
		#if dimension=='odors':
			#print('{}:{:.2},{:.2}'.format(feature, self.Q[RewardType.NONE.value][dimension][feature], self.Q[RewardType.NONE.value][dimension][feature] + delta))
		self.Q[RewardType.NONE.value][dimension][feature] += delta

	def get_model_metrics(self):
		flattened_biases_values = utils.flatten_dict(self.action_bias)
		res = {k: np.sum(v.tolist()[0:1])-np.sum(v.tolist()[2:3]) for k, v in flattened_biases_values.items()}
		#res['odor_value']= np.max(self.Q[RewardType.NONE.value]['odors'])
		return res

	def get_model_diff(self, brain2):
		flattened_biases_values1 = utils.flatten_dict(self.action_bias)
		flattened_stimulus_values1 = utils.flatten_dict(self.Q)

		flattened_biases_values2 = utils.flatten_dict(brain2.action_bias)
		flattened_stimulus_values2 = utils.flatten_dict(brain2.Q)

		result = {}
		for key in flattened_stimulus_values1.keys():
			result[key] = 0 if True else jensenshannon(flattened_stimulus_values1[key], flattened_stimulus_values2[key])
		for key in flattened_biases_values1.keys():
			result[key] = 0 if True else jensenshannon(flattened_biases_values1[key], flattened_biases_values2[key])

		return result

	def __str__(self):
		return self.__class__.__name__


class SCFTable(FTable, SCDependantV):
	pass

class SCVBFTable(FTable, SCDependantVB):
	pass

class MFTable(FTable):
	"""The state action value is dependent on the current motivation.
	Different models for diffrent motivations"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def __call__(self, *args, **kwargs):
		states = args[0]
		motivation = args[1]

		cues = utils.stimuli_1hot_to_cues(states, self.encoding_size)
		odor = cues[:, 0]  # odor for each door
		color = cues[:, 1]  # color for each door
		door = np.array(range(4))
		action_values = (self.get_stimulus_value('odors', odor, motivation) +
						 self.get_stimulus_value('colors', color, motivation) +
						 self.get_stimulus_value('spatial', door, motivation)) + self.action_bias[motivation.value]
		action_values[odor == self.encoding_size] = -np.inf  # avoid selecting inactive doors.
		return action_values

	def get_stimulus_value(self, dimension, feature, motivation):
		return self.Q[motivation.value][dimension][feature]

	def set_stimulus_value(self, dimension, feature, motivation, new_value):
		self.Q[motivation][dimension][feature] = new_value

	def update_stimulus_value(self, dimension, feature, motivation, delta):
		self.Q[motivation.value][dimension][feature] += delta


class ACFTable(FTable):
	"""This model implements attention in addition to stimuli reset on SC.
	However, whether attention is updated is up to the learner."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.phi = np.ones([3]) / 3

	def __call__(self, *args, **kwargs):
		states = args[0]
		batch = states.shape[0]
		motivation = args[1]
		cues = utils.stimuli_1hot_to_cues(states, self.encoding_size)
		odor = cues[:, 0]
		color = cues[:, 1]

		data = np.stack([self.get_stimulus_value('odors', odor, motivation),
						 self.get_stimulus_value('colors', color, motivation),
						 np.repeat(np.expand_dims(self.get_stimulus_value('spatial', np.array(range(4)), motivation), axis=0),
								   repeats=batch, axis=0)])
		attention = np.expand_dims(utils.softmax(self.phi), axis=0)
		doors_value = np.matmul(attention, np.transpose(data, axes=(1, 0, 2))) + self.action_bias[motivation.value]
		doors_value = np.squeeze(doors_value, axis=1)
		doors_value[odor == self.encoding_size] = -np.inf  # avoid selecting inactive doors.

		return doors_value

	def get_model_metrics(self):
		# print("odor:{}\ncolor:{},\nspatial:{}".format(self.V['odors'], self.V['colors'], self.V['spatial']))
		phi = utils.softmax(self.phi)
		return {'odor_attn': phi[0],
				'color_attn': phi[1],
				'spatial_attn': phi[2],
				}

	def get_model_diff(self, brain2):
		return {'odor_attn_diff': self.phi[0] - brain2.phi[0],
				'color_attn_diff': self.phi[1] - brain2.phi[1],
				'spatial_attn_diff': self.phi[2] - brain2.phi[2], }

	def new_stimuli_context(self):
		self.Q['odors'] = self.initial_value * np.ones([self.encoding_size + 1])
		self.Q['colors'] = self.initial_value * np.ones([self.encoding_size + 1])
		self.Q['spatial'] = self.initial_value * np.ones([4])


class PCFTable(ACFTable):
	def __init__(self, encoding_size, num_channels, num_actions):
		super().__init__(encoding_size, num_channels, num_actions)

	def __call__(self, *args, **kwargs):
		states = args[0]
		batch = states.shape[0]

		cues = utils.stimuli_1hot_to_cues(states, self.encoding_size)
		odor = cues[:, 0]
		color = cues[:, 1]

		selected_dim = utils.epsilon_greedy(0.1, self.phi)
		doors_value = self.get_stimulus_value('odors', odor) if selected_dim == 0 else \
			self.get_stimulus_value('colors', color) if selected_dim == 1 \
				else self.get_stimulus_value('spatial', [np.array(range(4))])

		doors_value[odor == self.encoding_size] = -np.inf  # avoid selecting inactive doors.

		return doors_value
