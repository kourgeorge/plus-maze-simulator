__author__ = 'gkour'

import itertools

from skopt.space import Real, Integer

from brains.consolidationbrain import ConsolidationBrain
from brains.RandomBrain import RandomBrain
from brains.fixeddoorattentionbrain import FixedDoorAttentionBrain
from brains.lateoutcomeevaluationbrain import LateOutcomeEvaluationBrain
from brains.tdbrain import TDBrain
from brains.motivationdependantbrain import MotivationDependantBrain
from models.networkmodels import *
from learners.networklearners import *
from learners.tabularlearners import *

REPETITIONS = 3
MOTIVATED_ANIMAL_DATA_PATH = './fitting/motivation_behavioral_data'
MAZE_ANIMAL_DATA_PATH = './fitting/maze_behavioral_data'

MOTIVATED_ANIMAL_BATCHES = {1: [1, 2], 2: [1], 4: [6, 7, 8], 5: [1, 2], 6: [10, 11]}
MAZE_ANIMALS = [0, 1, 2, 3, 4, 5, 6, 7, 8]

nmr = Real(name='nmr', low=-1, high=1)
lr = Real(name='lr', low=0.0001, high=0.7, prior='log-uniform')
attention_lr = Real(name='lr', low=0.0001, high=0.7, prior='log-uniform')
batch_size = Integer(name='batch_size', low=1, high=20)
beta = Real(name='beta', low=0.1, high=10)

maze_models = [((TDBrain, QLearner, QTable), (beta, lr)),
			   ((TDBrain, IALearner, FTable), (beta, lr)),
			   ((TDBrain, MALearnerSimple, ACFTable), (beta, lr, attention_lr)),
			   ((TDBrain, MALearner, ACFTable), (beta, lr, attention_lr)),
			   # ((ConsolidationBrain, DQN, FullyConnectedNetwork), (beta, lr)),
			   # ((ConsolidationBrain, DQN, UniformAttentionNetwork), (beta, lr)),
			   # ((ConsolidationBrain, DQN, AttentionAtChoiceAndLearningNetwork), (beta, lr)),
			   # ((ConsolidationBrain, DQN, FullyConnectedNetwork2Layers), (beta, lr)),
				# ((ConsolidationBrain, PG, FullyConnectedNetwork), (beta, lr)),
				# ((ConsolidationBrain, PG, UniformAttentionNetwork), (beta, lr)),
				# ((ConsolidationBrain, PG, AttentionAtChoiceAndLearningNetwork), (beta, lr)),
				# ((ConsolidationBrain, PG, FullyConnectedNetwork2Layers), (beta, lr)),
			   # ((RandomBrain, DQN, Random), (beta, lr))
			   ]

motivational_models = maze_models + [((ConsolidationBrain, DQN, EfficientNetwork), (beta, lr, batch_size)),
					   ((FixedDoorAttentionBrain, DQN, EfficientNetwork), (nmr, lr, batch_size)),
					   ((MotivationDependantBrain, DQN, SeparateMotivationAreasNetwork), (nmr, lr, batch_size)),
					   ((MotivationDependantBrain, DQN, SeparateMotivationAreasFCNetwork), (nmr, lr, batch_size)),
					   ((LateOutcomeEvaluationBrain, DQN, SeparateMotivationAreasNetwork), (nmr, lr, batch_size))
									 ]

non_motivated_reward = [0, 0.3]
learning_rates = [0.001, 0.01, 0.2]
combinations = list(itertools.product(maze_models, learning_rates, non_motivated_reward))
configs = [{'br': br, 'lr': lr, 'non_motivated_reward': nmr} for br, lr, nmr in combinations]


def extract_configuration_params(params):
	return params['br'][0], params['br'][1], params['br'][2], params['lr'], params['non_motivated_reward']
