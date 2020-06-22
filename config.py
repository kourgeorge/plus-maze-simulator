from enum import Enum

BASE_LEARNING_RATE = 1e-2
BASE_BRAIN_STRUCTURE_PARAM = 4
EXPLORATION_EPSILON = 0.1
SUCCESS_CRITERION_THRESHOLD = 0.85

class RewardType(Enum):
    WATER = 'water'
    FOOD = 'food'
    NONE = 'none'


class CueType(Enum):
    ODOR = 0
    LIGHT = 1