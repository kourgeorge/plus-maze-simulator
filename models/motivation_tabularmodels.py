__author__ = 'gkour'

import numpy as np
import utils
from models.tabularmodels import QTable, OptionsTable, FTable
from rewardtype import RewardType

norm = 'fro'


class MQTable(QTable):
    def __init__(self, encoding_size, num_channels, num_actions):
        super().__init__(encoding_size, num_channels, num_actions)

    def state_representation(self, obs, motivation):
        motivation_dimension = np.tile([motivation.value], (obs.shape[0], 1, 4))
        return np.append(utils.stimuli_1hot_to_cues(obs, self.encoding_size), motivation_dimension, axis=1)


class MOptionsTable(OptionsTable):
    def __init__(self, encoding_size, num_channels, num_actions):
        super().__init__(encoding_size, num_channels, num_actions)

    def state_representation(self, obs, motivation):
        motivation_dimension = np.tile([motivation.value], (obs.shape[0], 1, 4))
        return np.append(utils.stimuli_1hot_to_cues(obs, self.encoding_size), motivation_dimension, axis=1)


class SCDependantV():
    '''
    Stimuli dependant Tabular model,
    It initializes the state value data when a new stimuli is encountered.
    '''

    def new_stimuli_context(self):
        self.initialize_state_values()


class SCDependantB():
    '''
    Motivation dependant Tabular model.
    It initializes the state value and action bias data when a new stimuli is encountered.
    '''

    def new_stimuli_context(self):
        for motivation in RewardType:
            self.action_bias[motivation.value] = np.zeros(self._num_actions)


class SCDependantVB():
    '''
    Motivation dependant Tabular model.
    It initializes the state value and action bias data when a new stimuli is encountered.
    '''

    def new_stimuli_context(self):
        self.initialize_state_values()
        for motivation in RewardType:
            self.action_bias[motivation.value] = np.zeros(self._num_actions)


class MFTable(FTable):
    """The state action value is dependent on the current motivation.
    Different models for different motivations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_stimulus_value(self, dimension, feature, motivation):
        return self.Q[motivation.value][dimension][feature]

    def set_stimulus_value(self, dimension, feature, motivation, new_value):
        self.Q[motivation][dimension][feature] = new_value

    def update_stimulus_value(self, dimension, feature, motivation, delta):
        self.Q[motivation.value][dimension][feature] += delta


class MSCFTable(MFTable, SCDependantV):
    pass


class SCFTable(FTable, SCDependantV):
    pass


class SCVBFTable(FTable, SCDependantVB):
    pass


class SCBFTable(FTable, SCDependantB):
    pass


class SCBQTable(QTable, SCDependantB):
    pass


class SCBOptionsTable(OptionsTable, SCDependantB):
    pass
