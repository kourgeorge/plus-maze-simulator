import random
import numpy as np
import config
import utils

#np.random.seed(seed=1234)

class PlusMaze:
    def __init__(self, relevant_cue: config.CueType):
        # relevant_cue: 0 is odor, 1 is light
        # correct_value: 1/-1 the identity of the correct cue

        self._state = None
        self.stimuli_encoding_size = 2
        self._relevant_cue = relevant_cue
        self._odor_cues = None
        self._light_cues = None
        self._stage = 0
        self.set_random_odor_set()
        self.set_random_light_set()

    def reset(self):
        self._state = self.random_state()
        return self._state

    def step(self, action):
        # returns reward, new state, done

        option_cues_length = self.light_encoding_size() + self.odor_encoding_size()
        selected_arm_cues_ind = option_cues_length * action
        selected_cues = self._state[selected_arm_cues_ind:selected_arm_cues_ind + option_cues_length]
        #        print('state {}, action:{} selected arm cues:{}'.format(self._state,action,selected_cues ))
        self._state = np.asarray([-1] * self.state_shape())
        selected_cues_items = [selected_cues[0:self.odor_encoding_size()]] + \
                        [selected_cues[self.odor_encoding_size():self.odor_encoding_size() + self.light_encoding_size()]]

        if np.array_equal(selected_cues_items[self._relevant_cue.value], self.get_correct_cue_value()):
            outcome = config.RewardType.WATER if action in [0, 1] else config.RewardType.FOOD
            return self._state, outcome, 1, self._get_step_info(outcome)
        return self._state, config.RewardType.NONE, 1, self._get_step_info(config.RewardType.NONE)

    def state(self):
        # Odor Arm 1, Light Arm 1,Odor Arm 2, Light Arm 2,Odor Arm 3, Light Arm 3,Odor Arm 4, Light Arm 4
        return self._state

    def set_stage(self, stage):
        self._stage = stage

    def _get_step_info(self, outcome):
        info = utils.Object()
        info.relevant_cue = self._relevant_cue
        info.correct_cue_value = self.get_correct_cue_value()
        info.relevant_cue = self._relevant_cue
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
        return 4 * (self.odor_encoding_size() + self.light_encoding_size())

    def odor_encoding_size(self):
        return self.stimuli_encoding_size

    def set_random_odor_set(self):
        new_odor1 = PlusMaze.random_stimuli_encoding(self.odor_encoding_size())
        new_odor2 = PlusMaze.random_stimuli_encoding(self.odor_encoding_size())
        while(np.linalg.norm(np.array(new_odor1)-np.array(new_odor2))<0.2):
            new_odor1 = PlusMaze.random_stimuli_encoding(self.odor_encoding_size())
            new_odor2 = PlusMaze.random_stimuli_encoding(self.odor_encoding_size())
        self.set_odor_cues([new_odor1,new_odor2])

    def set_random_light_set(self):
        new_light1 = PlusMaze.random_stimuli_encoding(self.light_encoding_size())
        new_light2 = PlusMaze.random_stimuli_encoding(self.light_encoding_size())
        while (np.linalg.norm(np.array(new_light1) - np.array(new_light2)) < 0.2):
            new_light1 = PlusMaze.random_stimuli_encoding(self.odor_encoding_size())
            new_light2 = PlusMaze.random_stimuli_encoding(self.odor_encoding_size())
        self.set_light_cues([new_light1,new_light2])

    @staticmethod
    def random_stimuli_encoding(encoding_size):
        return list(np.round(np.random.rand(encoding_size)*2-1,2))

    def light_encoding_size(self):
        return self.stimuli_encoding_size

    def get_odor_cues(self):
        return self._odor_cues

    def get_light_cues(self):
        return self._light_cues

    def set_odor_cues(self, options):
        self._odor_cues = options

    def set_light_cues(self, options):
        self._light_cues = options

    def set_relevant_cue(self, relevant_cue):
        self._relevant_cue = relevant_cue

    def get_correct_cue_value(self):
        return self.get_odor_cues()[0] if self._relevant_cue==config.CueType.ODOR else self.get_light_cues()[0]

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
