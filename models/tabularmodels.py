__author__ = 'gkour'

import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

import config
import utils
from collections import defaultdict

from rewardtype import RewardType

norm = 'fro'


class AbstractTabularModel:

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def get_model_metrics(self):
        raise NotImplementedError()

    def get_model_diff(self, brain2):
        raise NotImplementedError()

    def get_observations_values(self, observations, motivation):
        raise NotImplementedError()

    def __str__(self):
        return self.__class__.__name__


class QTable(AbstractTabularModel):
    def __init__(self, encoding_size, num_channels, num_actions, initial_value=config.INITIAL_FEATURE_VALUE, *args,
                 **kwargs):
        super().__init__()
        self._num_actions = num_actions
        self.Q = defaultdict(lambda: initial_value * np.ones(self._num_actions))
        self.action_bias = {'water': np.zeros(self._num_actions), 'food': np.zeros(self._num_actions),
                            'none': np.zeros(self._num_actions)}
        self.encoding_size = encoding_size

    def __call__(self, *args, **kwargs):
        env_obs = args[0]
        motivation = args[1]
        state_actions_value = self.get_observations_values(env_obs, motivation)
        state_actions_value += self.get_bias_values(motivation)
        inactive_doors = utils.get_inactive_doors(env_obs)
        state_actions_value[inactive_doors] = -np.inf
        return state_actions_value

    def get_observations_values(self, observations, motivation):
        state_actions_value = []
        for state in self.state_representation(observations, motivation):
            state_actions_value.append(self.Q[np.array2string(state)])
        state_actions_value = np.array(state_actions_value)
        return state_actions_value

    def get_bias_values(self, motivation):
        return self.action_bias[motivation.value]

    def set_bias_value(self, motivation, action, value):
        self.action_bias[motivation.value][action] = value

    def state_representation(self, obs, motivation):
        return utils.stimuli_1hot_to_cues(obs, self.encoding_size)

    def set_actual_state_value(self, obs, actions, values, motivation):
        states = self.state_representation(obs, motivation)
        for state, action, value in zip(states, actions, values):
            self.Q[np.array2string(state)][action] = value

    def get_model_metrics(self):
        flattened_biases_values = utils.flatten_dict(self.action_bias)
        res = {k: np.round(np.sum(v.tolist()[0:2]) - np.sum(v.tolist()[2:4]), 4) for k, v in
               flattened_biases_values.items()}
        return res

    def get_model_diff(self, brain2):
        diff = [np.linalg.norm(self.Q[state] - brain2.Q[state]) for state in
                set(self.Q.keys()).intersection(brain2.Q.keys())]
        return {'table_change': np.mean(diff) * 100}


class OptionsTable(AbstractTabularModel):

    def __init__(self, encoding_size, num_channels, num_actions, use_location_cue=True,
                 initial_value=config.INITIAL_FEATURE_VALUE, *args, **kwargs):
        super().__init__()
        self._num_actions = num_actions
        self.C = defaultdict(lambda: float(
            initial_value))  # familiar options are stored as tupples (color, odor and possibly, location).
        self.action_bias = {'water': np.zeros(self._num_actions), 'food': np.zeros(self._num_actions),
                            'none': np.zeros(self._num_actions)}
        self.encoding_size = encoding_size
        self.use_location_cue = use_location_cue

    def __call__(self, *args, **kwargs):
        observations = args[0]
        motivation = args[1]
        state_actions_value = []
        for state in self.state_representation(observations, motivation):
            obs_action_value = []
            for option in self.get_cues_combinations(state):
                obs_action_value += [self.C[option]] if option[0] != self.encoding_size else [-np.inf]
            obs_action_value += self.get_bias_values(motivation)
            state_actions_value.append(obs_action_value)
        return np.array(state_actions_value)

    def get_bias_values(self, motivation):
        return self.action_bias[motivation.value]

    def set_bias_value(self, motivation, action, value):
        self.action_bias[motivation.value][action] = value

    def get_observations_values(self, observations, motivation):
        state_actions_value = []
        for state in self.state_representation(observations, motivation):
            obs_action_value = []
            for option in self.get_cues_combinations(state):
                obs_action_value += [self.C[option]] if option[0] != self.encoding_size else [-np.inf]
            state_actions_value.append(obs_action_value)
        return state_actions_value

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
        flattened_biases_values = utils.flatten_dict(self.action_bias)
        return {k: np.round(np.sum(v.tolist()[0:2]) - np.sum(v.tolist()[2:4]), 4) for k, v in
                flattened_biases_values.items()}

    def get_model_diff(self, brain2):
        diff = [np.linalg.norm(self.C[state] - brain2.C[state]) for state in
                set(self.C.keys()).intersection(brain2.C.keys())]
        return {'table_change': np.mean(diff) * 100}

    def __str__(self):
        return self.__class__.__name__


class FTable(AbstractTabularModel):
    def __init__(self, encoding_size, num_actions, initial_value=config.INITIAL_FEATURE_VALUE, num_channels=2):
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
        observations = args[0]
        motivation = args[1]

        action_values = self.get_observations_values(observations, motivation) + self.get_bias_values(motivation)
        inactive_doors = utils.get_inactive_doors(observations)
        action_values[inactive_doors] = -np.inf  # avoid selecting inactive doors.
        return action_values

    def get_observations_values(self, observations, motivation):
        cues = utils.stimuli_1hot_to_cues(observations, self.encoding_size)
        odor = cues[:, 0]  # odor for each door
        color = cues[:, 1]  # color for each door
        door = np.array(range(4))
        action_values = (self.get_stimulus_value('odors', odor, motivation) +
                         self.get_stimulus_value('colors', color, motivation) +
                         self.get_stimulus_value('spatial', door, motivation))
        return action_values

    def get_bias_values(self, motivation: RewardType):
        return self.action_bias[motivation.value]

    def set_bias_value(self, motivation: RewardType, action, value):
        self.action_bias[motivation.value][action] = value

    def get_selected_door_stimuli(self, states, doors):
        cues = utils.stimuli_1hot_to_cues(states, self.encoding_size)
        selected_cues = cues[np.arange(len(states)), :, doors]
        return selected_cues[:, 0], selected_cues[:, 1]

    def get_stimulus_value(self, dimension, feature, motivation):
        return self.Q[RewardType.NONE.value][dimension][feature]

    def set_stimulus_value(self, dimension, feature, motivation, new_value):
        self.Q[motivation][dimension][feature] = new_value

    def update_stimulus_value(self, dimension, feature, motivation, delta):
        # if dimension=='odors':
        # print('{}:{:.2},{:.2}'.format(feature, self.Q[RewardType.NONE.value][dimension][feature], self.Q[RewardType.NONE.value][dimension][feature] + delta))
        self.Q[motivation.value][dimension][feature] += delta

    def get_model_metrics(self):
        flattened_biases_values = utils.flatten_dict(self.action_bias)
        res = {k: np.round(np.sum(v.tolist()[0:2]) - np.sum(v.tolist()[2:4]), 4) for k, v in
               flattened_biases_values.items()}
        # res['odor']=np.sum(np.abs(self.Q['none']['odors']))
        # res['spatial'] = np.sum(np.abs(self.Q['none']['spatial']))
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


class ACFTable(FTable):
    """This model implements attention in addition to stimuli reset on SC.
    However, whether attention is updated is up to the learner."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_importance = np.ones([3]) / 3

    def __call__(self, *args, **kwargs):
        states = args[0]
        motivation = args[1]
        cues = utils.stimuli_1hot_to_cues(states, self.encoding_size)
        odor = cues[:, 0]
        doors_value = self.get_observations_values(states, motivation) + self.action_bias[motivation.value]
        doors_value[odor == self.encoding_size] = -np.inf
        return doors_value

    def phi(self):
        return utils.softmax(self.attn_importance)

    def get_observations_values(self, observations, motivation):
        cues = utils.stimuli_1hot_to_cues(observations, self.encoding_size)
        batch = observations.shape[0]

        odor = cues[:, 0]  # odor for each door
        color = cues[:, 1]  # color for each door

        data = np.stack([self.get_stimulus_value('odors', odor, motivation),
                         self.get_stimulus_value('colors', color, motivation),
                         np.repeat(
                             np.expand_dims(self.get_stimulus_value('spatial', np.array(range(4)), motivation), axis=0),
                             repeats=batch, axis=0)])
        attention = np.expand_dims(self.phi(), axis=0)
        action_values = np.matmul(attention, np.transpose(data, axes=(1, 0, 2)))
        action_values = np.squeeze(action_values, axis=1)
        return action_values

    def get_model_metrics(self):
        # print("odor:{}\ncolor:{},\nspatial:{}".format(self.V['odors'], self.V['colors'], self.V['spatial']))
        phi = self.phi()
        return {'odor_importance': self.attn_importance[0],
                'color_importance': self.attn_importance[1],
                'spatial_importance': self.attn_importance[2],
                'odor_weight': phi[0],
                'color_weight': phi[1],
                'spatial_weight': phi[2],
                }

    def get_model_diff(self, brain2):
        return {'odor_attn_diff': self.attn_importance[0] - brain2.attn_importance[0],
                'color_attn_diff': self.attn_importance[1] - brain2.attn_importance[1],
                'spatial_attn_diff': self.attn_importance[2] - brain2.attn_importance[2]}

    def new_stimuli_context(self, motivation):
        self.Q[motivation]['odors'] = self.initial_value * np.ones([self.encoding_size + 1])
        self.Q[motivation]['colors'] = self.initial_value * np.ones([self.encoding_size + 1])
        self.Q[motivation]['spatial'] = self.initial_value * np.ones([self.encoding_size + 1])


class FixedACFTable(ACFTable):
    """This model implements attention in addition to stimuli reset on SC.
    However, whether attention is updated is up to the learner."""

    def __init__(self, attn_importance=np.ones([3]) / 3, *args, **kwargs):
        if any(item < 0 for item in attn_importance) or np.abs(np.sum(attn_importance) - 1) > 1e-9:
            raise Exception("Illigal attention arguments, should be positive and sum to 1!")
        super().__init__(*args, **kwargs)

        self.attn_importance = np.abs(attn_importance) / np.sum(
            np.abs(attn_importance))  # notmalize attention parameters

    def phi(self):
        return self.attn_importance
