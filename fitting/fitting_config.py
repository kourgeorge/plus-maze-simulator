__author__ = 'gkour'

import itertools

from skopt.space import Real, Integer

from brains.consolidationbrain import ConsolidationBrain, RandomBrain
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
#ANIMAL_BATCHES = {1: [1, 2],  2: [1], 4: [6, 7]}

def motivational_animals_generator():
	for animal_batch in MOTIVATED_ANIMAL_BATCHES:
		for rat in MOTIVATED_ANIMAL_BATCHES[animal_batch]:
			yield (animal_batch, rat)


nmr = Real(name='nmr', low=-1, high=1)
lr = Real(name='lr', low=0.0001, high=0.2, prior='log-uniform')
batch_size = Integer(name='batch_size', low=1, high=20)
beta = Real(name='beta', low=0.1, high=5)

non_motivated_reward = [0, 0.3]
learning_rates = [0.001, 0.01, 0.2]
maze_models = [((TDBrain, TD, TabularQ), (beta, lr, batch_size)),
			   ((TDBrain, TDUniformAttention, UniformAttentionTabular), (beta,lr,batch_size)),
			   ((TDBrain, TDUniformAttention, AttentionAtChoiceAndLearningTabular),(nmr,lr,batch_size)),
			   ((ConsolidationBrain, DQN, FullyConnectedNetwork),(beta,lr,batch_size)),
			   ((ConsolidationBrain, DQN, UniformAttentionNetwork),(beta,lr,batch_size)),
			   ((ConsolidationBrain, DQN, AttentionAtChoiceAndLearningNetwork),(beta,lr,batch_size)),
			   ((ConsolidationBrain, DQN, FullyConnectedNetwork2Layers),(beta,lr,batch_size)),
			   ((RandomBrain, DQN, RandomNetwork),(beta,lr,batch_size))
			   ]

motivational_models = maze_models + \
					  [((ConsolidationBrain, DQN, EfficientNetwork), (beta, lr, batch_size)),
					   ((FixedDoorAttentionBrain, DQN, EfficientNetwork), (nmr, lr, batch_size)),
					   ((MotivationDependantBrain, DQN, SeparateMotivationAreasNetwork), (nmr, lr, batch_size)),
					   ((MotivationDependantBrain, DQN, SeparateMotivationAreasFCNetwork), (nmr, lr, batch_size)),
					   ((LateOutcomeEvaluationBrain, DQN, SeparateMotivationAreasNetwork), (nmr, lr, batch_size))]

combinations = list(itertools.product(maze_models, learning_rates, non_motivated_reward))
configs = [{'br': br, 'lr': lr, 'non_motivated_reward': nmr} for br, lr, nmr in combinations]


def extract_configuration_params(params):
	return params['br'][0], params['br'][1], params['br'][2], params['lr'], params['non_motivated_reward']
