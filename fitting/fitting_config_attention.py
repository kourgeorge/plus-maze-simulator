__author__ = 'gkour'

import itertools

from skopt.space import Real, Integer

from brains.consolidationbrain import ConsolidationBrain
from brains.fixeddoorattentionbrain import FixedDoorAttentionBrain
from brains.lateoutcomeevaluationbrain import LateOutcomeEvaluationBrain
from brains.motivationdependantbrain import MotivationDependantBrain
from brains.tdbrain import TDBrain
from learners.networklearners import *
from learners.tabularlearners import *
from models.networkmodels import *
from models.tabularmodels import *

MAZE_ANIMAL_DATA_PATH = '/Users/georgekour/repositories/plus-maze-simulator/fitting/maze_behavioral_data'
OPTIMIZATION_METHOD = 'Hybrid' #  'Hybrid','Newton', 'Bayesian'
MAZE_ANIMALS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
FITTING_ITERATIONS = 75

lr = Real(name='lr', low=0.001, high=0.4, prior='log-uniform')
attention_lr = Real(name='attention_lr', low=0.001, high=0.4, prior='log-uniform')
beta = Real(name='beta', low=0.1, high=10, prior='log-uniform')
initial_value = Real(name='initial', low=0, high=0.1, prior='uniform')
batch_size = Integer(name='batch_size', low=1, high=20)


maze_models = [
	# ((TDBrain, QLearner, QTable), (beta, lr)),
	# 		    ((TDBrain, QLearner, OptionsTable), (beta, lr)),
	# 		    ((TDBrain, IALearner, ACFTable), (beta, lr)),
	 			((TDBrain, MALearner, ACFTable), (beta, lr, attention_lr)),

			   # ((TDBrain, MALearnerSimple, ACFTable), (beta, lr, attention_lr)),
			   # ((TDBrain, MALearner, PCFTable), (beta, lr, attention_lr)),

			   # ((ConsolidationBrain, DQN, FCNet), (beta, lr)),
			   # ((ConsolidationBrain, DQN, UANet), (beta, lr)),
			   # ((ConsolidationBrain, DQN, ACLNet), (beta, lr)),
			   # ((ConsolidationBrain, DQNAtt, ACLNet), (beta, lr, attention_lr)),
			   # ((ConsolidationBrain, DQN, FC2LayersNet), (beta, lr)),

				#((ConsolidationBrain, DQN, EfficientNetwork),(beta, lr)),
				# ((ConsolidationBrain, PG, FullyConnectedNetwork), (beta, lr)),
				# ((ConsolidationBrain, PG, UniformAttentionNetwork), (beta, lr)),
				# ((ConsolidationBrain, PG, AttentionAtChoiceAndLearningNetwork), (beta, lr)),
				# ((ConsolidationBrain, PG, FullyConnectedNetwork2Layers), (beta, lr)),
			   # ((RandomBrain, DQN, Random), (beta, lr))
			   ]



def extract_configuration_params(params):
	return params['br'][0], params['br'][1], params['br'][2], params['lr'], params['non_motivated_reward']


friendly_models_name_map = {'QLearner.QTable': 'SARL',
							'UABQLearner.QTable': 'B-SARL',
							'ABQLearner.QTable': 'M(B)-SARL',
							'QLearner.MQTable': 'M(V)-SARL',
							'ABQLearner.MQTable': 'M(VB)-SARL',
							'QLearner.SCBQTable': 'E(B)-SARL',
							'ABQLearner.SCBQTable': 'E(B)-M(B)-SARL',

							'QLearner.OptionsTable': 'ORL',
							'OptionsLearner.OptionsTable': 'ORL',

							'IALearner.FTable': 'FRL',
							'IALearner.ACFTable': 'FRL',

							'MALearner.ACFTable': 'AARL'

							}