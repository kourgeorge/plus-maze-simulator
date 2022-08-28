__author__ = 'gkour'

import random
import numpy as np
import config
import utils
from rewardtype import RewardType
from enum import Enum


class CueType(Enum):
    ODOR = 0
    LIGHT = 1
    SPATIAL = 2


class PlusMaze:
    def __init__(self, relevant_cue: CueType):
        # relevant_cue: 0 is odor, 1 is light
        # correct_value: 1/-1 the identity of the correct cue

        self._state = None
        self._stimuli_encoding_size = 2
        self._relevant_cue:CueType = relevant_cue
        self._odor_cues = None
        self._light_cues = None
        self._correct_spatial_cues = [1,3]
        self._stage = 0
        self.set_random_odor_set()
        self.set_random_light_set()

    def reset(self):
        self._state = self.random_state()
        return self._state

    def set_state(self, state):
        self._state = state

    def step(self, action):
        # returns reward, new state, done

        option_cues_length = 2*self.stimuli_encoding_size()
        selected_arm_cues_ind = option_cues_length * action
        selected_cues = self._state[selected_arm_cues_ind:selected_arm_cues_ind + option_cues_length]
        #        print('state {}, action:{} selected arm cues:{}'.format(self._state,action,selected_cues ))
        self._state = np.asarray([-1] * self.state_shape())
        selected_cues_items = [selected_cues[0:self.stimuli_encoding_size()]] + \
                              [selected_cues[self.stimuli_encoding_size():2*self.stimuli_encoding_size()]]

        if (self._relevant_cue == CueType.SPATIAL and action in self._correct_spatial_cues) or\
                (self._relevant_cue != CueType.SPATIAL and
                 np.array_equal(selected_cues_items[self._relevant_cue.value], self.get_correct_cue_value())):
            outcome = RewardType.FOOD if action in [0, 1] else RewardType.WATER
            return self._state, outcome, 1, self._get_step_info(outcome)
        return self._state, RewardType.NONE, 1, self._get_step_info(RewardType.NONE)

    def state(self):
        # Odor Arm 1, Light Arm 1,Odor Arm 2, Light Arm 2,Odor Arm 3, Light Arm 3,Odor Arm 4, Light Arm 4
        return self._state

    def get_stage(self):
        return self._stage

    def set_stage(self, stage):
        self._stage = stage

    def _get_step_info(self, outcome):
        info = utils.Object()
        info.relevant_cue = self._relevant_cue
        info.correct_cue_value = self.get_correct_cue_value()
        info.odor_options = self._odor_cues
        info.light_options = self._light_cues
        info.outcome = outcome
        info.stage = self._stage

        return info

    def action_space(self):
        return [0, 1, 2, 3]

    def num_actions(self):
        return len(self.action_space())

    def state_shape(self):
        return (self.stimuli_encoding_size(), 8)

    def set_random_odor_set(self):
        new_odor1 = PlusMaze.random_stimuli_encoding(self.stimuli_encoding_size())
        new_odor2 = list(np.round(utils.get_orthogonal(new_odor1),2))
        while(np.linalg.norm(np.array(new_odor1)-np.array(new_odor2))<0.2):
            new_odor1 = PlusMaze.random_stimuli_encoding(self.stimuli_encoding_size())
            new_odor2 = PlusMaze.random_stimuli_encoding(self.stimuli_encoding_size())
        self.set_odor_cues([new_odor1,new_odor2])

    def set_random_light_set(self):
        new_light1 = PlusMaze.random_stimuli_encoding(self.stimuli_encoding_size())
        new_light2 = PlusMaze.random_stimuli_encoding(self.stimuli_encoding_size())
        while (np.linalg.norm(np.array(new_light1) - np.array(new_light2)) < 0.2):
            new_light1 = PlusMaze.random_stimuli_encoding(self.stimuli_encoding_size())
            new_light2 = PlusMaze.random_stimuli_encoding(self.stimuli_encoding_size())
        self.set_light_cues([new_light1,new_light2])

    @staticmethod
    def random_stimuli_encoding(encoding_size):
        point = np.random.rand(encoding_size) * 2 - 1
        point=np.round(point/np.linalg.norm(point),2)
        return list(point)

    def stimuli_encoding_size(self):
        return self._stimuli_encoding_size

    def get_odor_cues(self):
        return self._odor_cues

    def get_light_cues(self):
        return self._light_cues

    def set_odor_cues(self, options):
        self._odor_cues = options

    def set_light_cues(self, options):
        self._light_cues = options

    def set_relevant_cue(self, relevant_cue:CueType):
        self._relevant_cue = relevant_cue

    def get_correct_cue_value(self):
        if self._relevant_cue == CueType.ODOR:
            return self.get_odor_cues()[1]
        if self._relevant_cue == CueType.LIGHT:
            return self.get_light_cues()[1]
        if self._relevant_cue == CueType.SPATIAL:
            return self._correct_spatial_cues

    def random_state(self):
        arm1O = random.choice([0, 1])
        arm1L = random.choice([0, 1])

        # use the same combination for arms 3 and arms 4. select randomly.
        arm3O = random.choice([0, 1])
        arm3L = random.choice([0, 1])

        return np.asarray(self._odor_cues[arm1O] + self._light_cues[arm1L] +
                          self._odor_cues[1 - arm1O] + self._light_cues[1 - arm1L] +
                          self._odor_cues[arm3O] + self._light_cues[arm3L] +
                          self._odor_cues[1 - arm3O] + self._light_cues[1 - arm3L])


class PlusMazeOneHotCues(PlusMaze):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stimuli_encoding_size = 6
        self.set_random_odor_set()
        self.set_random_light_set()

    def set_state(self, state):
        arm1O = int(state['A1o'])
        arm1L = int(state['A1c'])

        # use the same combination for arms 3 and arms 4. select randomly.
        arm3O = int(state['A3o'])
        arm3L = int(state['A3c'])
        state = np.ndarray(shape=[2, 4, self.stimuli_encoding_size()])
        state[0, 0, :] = self._odor_cues[arm1O]
        state[0, 1, :] = self._odor_cues[1 - arm1O]
        state[0, 2, :] = self._odor_cues[arm3O]
        state[0, 3, :] = self._odor_cues[1 - arm3O]

        state[1, 0, :] = self._light_cues[arm1L]
        state[1, 1, :] = self._light_cues[1 - arm1L]
        state[1, 2, :] = self._light_cues[arm3L]
        state[1, 3, :] = self._light_cues[1 - arm3L]

        self._state = state
        return state

    def step(self, action):
        # returns reward, new state, done

        option_cues_length = 2*self.stimuli_encoding_size()
        selected_arm_cues_ind = option_cues_length * action
        selected_cues = self._state[:, action, :]
        #        print('state {}, action:{} selected arm cues:{}'.format(self._state,action,selected_cues ))
        selected_cues_items = [selected_cues[0:self.stimuli_encoding_size()]] + \
                              [selected_cues[
                               self.stimuli_encoding_size():2*self.stimuli_encoding_size() ]]

        self._state = np.ones(self.state_shape())
        if (self._relevant_cue == CueType.SPATIAL and action in self._correct_spatial_cues) or \
                (self._relevant_cue != CueType.SPATIAL and np.array_equal(selected_cues[self._relevant_cue.value, :],
                                                                          self.get_correct_cue_value())):
            outcome = RewardType.FOOD if action in [0, 1] else RewardType.WATER
            return self._state, outcome, 1, self._get_step_info(outcome)
        return self._state, RewardType.NONE, 1, self._get_step_info(RewardType.NONE)

    def state_shape(self):
        return (self.num_actions(), self.stimuli_encoding_size(), 2)

    def stimuli_encoding_size(self):
        return self._stimuli_encoding_size

    def set_random_odor_set(self):
        self._odor_cues = PlusMazeOneHotCues._choose_new_cue_set(self._odor_cues, self._stimuli_encoding_size)

    def set_random_light_set(self):
        self._light_cues = PlusMazeOneHotCues._choose_new_cue_set(self._light_cues, self._stimuli_encoding_size)

    def random_state(self):
        arm1O = random.choice([0, 1])
        arm1L = random.choice([0, 1])

        # use the same combination for arms 3 and arms 4. select randomly.
        arm3O = random.choice([0, 1])
        arm3L = random.choice([0, 1])
        state = np.ndarray(shape=[2, 4, self.stimuli_encoding_size()])
        state[0, 0, :] = self._odor_cues[arm1O]
        state[0, 1, :] = self._odor_cues[1 - arm1O]
        state[0, 2, :] = self._odor_cues[arm3O]
        state[0, 3, :] = self._odor_cues[1 - arm3O]

        state[1, 0, :] = self._light_cues[arm1L]
        state[1, 1, :] = self._light_cues[1 - arm1L]
        state[1, 2, :] = self._light_cues[arm3L]
        state[1, 3, :] = self._light_cues[1 - arm3L]

        return state

    def state_shape(self):
        return (2, self.num_actions(), self.stimuli_encoding_size())

    @staticmethod
    def _choose_new_cue_set(old_cues_one_hot, encoding_size):
        if old_cues_one_hot is None:
            c1, c2 = random.sample(range(0, encoding_size), 2)
            return [np.eye(encoding_size)[c1], np.eye(encoding_size)[c2]]

        old_cues = [np.argmax(old_cues_one_hot[0]), np.argmax(old_cues_one_hot[1])]
        c1, c2 = random.sample(range(0, encoding_size), 2)
        while c1 in old_cues or c2 in old_cues:
            c1, c2 = random.sample(range(0, encoding_size), 2)
        return [np.eye(encoding_size)[c1], np.eye(encoding_size)[c2]]