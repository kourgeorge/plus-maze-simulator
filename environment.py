import random
import numpy as np
import config
import utils


class PlusMaze:
    def __init__(self, relevant_cue: config.CueType):
        # relevant_cue: 0 is odor, 1 is light
        # correct_value: 1/-1 the identity of the correct cue

        self._state = None
        self._relevant_cue = relevant_cue
        self._correct_cue_value = -1
        self._odor_options = [-1, 1]
        self._light_options = [-1, 1]
        self.stage = 1

    def reset(self):
        self._state = self.random_state()
        return self._state

    def step(self, action):
        # returns reward, new state, done
        selected_arm_cues_ind = 2 * action
        selected_cues = [self._state[selected_arm_cues_ind], self._state[selected_arm_cues_ind + 1]]
        #        print('state {}, action:{} selected arm cues:{}'.format(self._state,action,selected_cues ))
        self._state = np.asarray([-1, -1, -1, -1, -1, -1, -1, -1])
        if selected_cues[self._relevant_cue.value] == self._correct_cue_value:
            outcome = config.RewardType.WATER if action in [0, 1] else config.RewardType.FOOD
            return self._state, outcome, 1, self._get_step_info(outcome)
        return self._state, config.RewardType.NONE, 1, self._get_step_info(config.RewardType.NONE)

    # def step(self, action):
    #     # returns reward, new state, done
    #     self._state = np.asarray([-1, -1, -1, -1, -1, -1, -1, -1])
    #
    #     reward = 1 if action == 0 else 0
    #     return self._state, reward, True, None

    def state(self):
        # Odor Arm 1, Light Arm 1,Odor Arm 2, Light Arm 2,Odor Arm 3, Light Arm 3,Odor Arm 4, Light Arm 4
        return self._state

    def set_stage(self, stage):
        self._stage = stage

    def _get_step_info(self, outcome):
        info = utils.Object()
        info.relevant_cue = self._relevant_cue
        info.correct_cue_value = self._correct_cue_value
        info.relevant_cue = self._relevant_cue
        info.odor_options = self._odor_options
        info.light_options = self._light_options
        info.outcome = outcome
        info.stage = self.stage

        return info

    @staticmethod
    def action_space():
        return [0, 1, 2, 3]

    @staticmethod
    def num_actions():
        return len(PlusMaze.action_space())

    @staticmethod
    def state_shape():
        return 8

    def set_odor_options(self, options):
        self._odor_options = options

    def set_light_options(self, options):
        self._light_options = options

    def set_relevant_cue(self, relevant_cue):
        self._relevant_cue = relevant_cue

    def set_correct_cue_value(self, correct_cue_value):
        self._correct_cue_value = correct_cue_value

    def random_state(self):
        arm1O = random.choice([0, 1])
        arm1L = random.choice([0, 1])

        # use the same combination for arms 3 and arms 4. select randomly.
        arm3O = random.choice([0, 1])
        arm3L = random.choice([0, 1])

        return np.asarray([self._odor_options[arm1O], self._light_options[arm1L],
                           self._odor_options[1 - arm1O], self._light_options[1 - arm1L],
                           self._odor_options[arm3O], self._light_options[arm3L],
                           self._odor_options[1 - arm3O], self._light_options[1 - arm3L]])
