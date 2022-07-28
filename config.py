__author__ = 'gkour'

from enum import Enum

REPORTING_INTERVAL = 100

LEARNING_RATE = 1e-2
EXPLORATION_EPSILON = 0.2
BATCH_SIZE = 20

SUCCESS_CRITERION_THRESHOLD = 0.80
MEMORY_SIZE = [10000]
STIMULIENCODINGSIZE = 6
FORGETTING = [0.05, 0.1, 0.2]

MOTIVATED_REWARD = [1]
NON_MOTIVATED_REWARD = [0, 0.3]


class RewardType(Enum):
    WATER = 'water'
    FOOD = 'food'
    NONE = 'none'


class CueType(Enum):
    ODOR = 0
    LIGHT = 1
    SPATIAL = 2


def initial_config():
    return {
        'REPORTING_INTERVAL': REPORTING_INTERVAL,
        'LEARNING_RATE': LEARNING_RATE,
        'EXPLORATION_EPSILON': EXPLORATION_EPSILON,
        'BATCH_SIZE': BATCH_SIZE,
        'SUCCESS_CRITERION_THRESHOLD': SUCCESS_CRITERION_THRESHOLD,
        'MEMORY_SIZE': MEMORY_SIZE[0],
        'STIMULIENCODINGSIZE': STIMULIENCODINGSIZE,
        'FORGETTING': FORGETTING[0],
        'MOTIVATED_REWARD': MOTIVATED_REWARD[0],
        'NON_MOTIVATED_REWARD': NON_MOTIVATED_REWARD[0],
        'CueType': CueType,
        'RewardType': RewardType
    }


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


current_config = dotdict(initial_config())


def create_dir_name(config):
    return '_'.join(map(str, [
        config.MEMORY_SIZE,
        config.FORGETTING,
        config.MOTIVATED_REWARD,
        config.NON_MOTIVATED_REWARD
    ]))


def gen_get_config():
    for MEMORY_SIZEi in range(len(MEMORY_SIZE)):
        for FORGETTINGi in range(len(FORGETTING)):
            for MOTIVATED_REWARDi in range(len(MOTIVATED_REWARD)):
                for NON_MOTIVATED_REWARDi in range(len(NON_MOTIVATED_REWARD)):
                    current_config = dotdict({
                        'REPORTING_INTERVAL': REPORTING_INTERVAL,
                        'BATCH_SIZE': BATCH_SIZE,
                        'LEARNING_RATE': LEARNING_RATE,
                        'EXPLORATION_EPSILON': EXPLORATION_EPSILON,
                        'SUCCESS_CRITERION_THRESHOLD': SUCCESS_CRITERION_THRESHOLD,
                        'MEMORY_SIZE': MEMORY_SIZE[MEMORY_SIZEi],
                        'STIMULIENCODINGSIZE': STIMULIENCODINGSIZE,
                        'FORGETTING': FORGETTING[FORGETTINGi],
                        'MOTIVATED_REWARD': MOTIVATED_REWARD[MOTIVATED_REWARDi],
                        'NON_MOTIVATED_REWARD': NON_MOTIVATED_REWARD[NON_MOTIVATED_REWARDi],
                        'CueType': CueType,
                        'RewardType': RewardType
                    })
                    dirname = create_dir_name(current_config)
                    yield current_config, dirname


def get_config():
    return current_config
