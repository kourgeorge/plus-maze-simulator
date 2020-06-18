from enum import Enum

BASE_LEARNING_RATE = 1e-2
BASE_BRAIN_STRUCTURE_PARAM = 4
BASE_EPSILON = 0.1
BASE_LEARNING_FREQ = 500


class RewardType(Enum):
    WATER = 'water'
    FOOD = 'food'
    NONE = 'none'


class CueType(Enum):
    ODOR = 0
    LIGHT = 1