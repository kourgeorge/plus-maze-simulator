from enum import Enum

REPORTING_INTERVAL = 100

LEARNING_RATE = 1e-2
EXPLORATION_EPSILON = 0.2
SUCCESS_CRITERION_THRESHOLD = 0.80
MEMORY_SIZE = 200
STIMULIENCODINGSIZE = 6
FORGETTING = 0.1

MOTIVATED_REWARD = 1
NON_MOTIVATED_REWARD = 0.3


class RewardType(Enum):
    WATER = 'water'
    FOOD = 'food'
    NONE = 'none'


class CueType(Enum):
    ODOR = 0
    LIGHT = 1

stage_names = ['Baseline', 'IDshift', 'Mshift(Food)', 'MShift(Water)+IDshift', 'EDShift(Light)']