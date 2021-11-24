from enum import Enum

REPORTING_INTERVAL = 100

LEARNING_RATE = 1e-2
EXPLORATION_EPSILON = 0.2
SUCCESS_CRITERION_THRESHOLD = 0.85

MOTIVATED_REWARD = 1
NON_MOTIVATED_REWARD = 0.3


class RewardType(Enum):
    WATER = 'water'
    FOOD = 'food'
    NONE = 'none'


class CueType(Enum):
    ODOR = 0
    LIGHT = 1