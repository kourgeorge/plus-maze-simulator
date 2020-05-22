import random
import numpy as np
import config


class PlusMaze:
    def __init__(self, relevant_cue: config.CueType, correct_value):
        # relevant_cue: 0 is odor, 1 is light
        # correct_value: 1/-1 the identity of the correct cue

        self._state = PlusMaze.random_state()
        self._relevant_cue = relevant_cue
        self._correct_value = correct_value

    def reset(self):
        self._state = PlusMaze.random_state()
        return self._state

    def step(self, action):
        # returns reward, new state, done
        selected_arm_cues_ind = 2 * action
        selected_cues = [self._state[selected_arm_cues_ind], self._state[selected_arm_cues_ind + 1]]
        #        print('state {}, action:{} selected arm cues:{}'.format(self._state,action,selected_cues ))
        self._state = np.asarray([-1, -1, -1, -1, -1, -1, -1, -1])
        if selected_cues[self._relevant_cue.value] == self._correct_value:
            outcome = config.RewardType.WATER if action in [0, 1] else config.RewardType.FOOD
            return self._state, outcome, 1, None
        return self._state, config.RewardType.NONE, 1, None

    def state(self):
        # Odor Arm 1, Light Arm 1,Odor Arm 2, Light Arm 2,Odor Arm 3, Light Arm 3,Odor Arm 4, Light Arm 4
        return self._state

    @staticmethod
    def action_space():
        return [0, 1, 2, 3]

    @staticmethod
    def num_actions():
        return len(PlusMaze.action_space())

    @staticmethod
    def state_shape():
        return 8

    @staticmethod
    def random_state():
        arm1O = random.choice([-1, 1])
        arm1L = random.choice([-1, 1])

        arm3O = random.choice([-1, 1])
        arm3L = random.choice([-1, 1])

        return np.asarray([arm1O, arm1L, -arm1O, -arm1L, arm3O, arm3L, -arm3O, -arm3L])
