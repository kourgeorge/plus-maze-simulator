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
MAZE_ANIMAL_LED_FIRST_DATA_PATH = '/Users/georgekour/repositories/plus-maze-simulator/fitting/maze_behavioral_led_first_data'
MAZE_ANIMALS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
MAZE_LED_FIRST_ANIMALS = [33, 34, 35, 36, 37, 38, 39, 40, 41, 42]

OPTIMIZATION_METHOD = 'Hybrid' #  'Hybrid','Newton', 'Bayesian'
FITTING_ITERATIONS = 75

lr = Real(name='lr', low=0.001, high=0.4, prior='log-uniform')
attention_lr = Real(name='attention_lr', low=0.001, high=0.4, prior='log-uniform')
beta = Real(name='beta', low=0.1, high=10, prior='log-uniform')
attention_color = Real(name='attention_color', low=0, high=1, prior='uniform')
attention_odor = Real(name='attention_odor', low=0, high=1, prior='uniform')
initial_value = Real(name='initial', low=0, high=0.1, prior='uniform')
batch_size = Integer(name='batch_size', low=1, high=20)

maze_models = [
	((TDBrain, QLearner, QTable), (beta, lr)),
	((TDBrain, QLearner, OptionsTable), (beta, lr)),
	((TDBrain, IALearner, ACFTable), (beta, lr)),
	((TDBrain, MALearner, ACFTable), (beta, lr, attention_lr)),
	((TDBrain, IALearner, FixedACFTable), (beta, lr, attention_odor, attention_color))

			   # ((TDBrain, MALearnerSimple, ACFTable), (beta, lr, attention_lr)),
			   # ((TDBrain, MALearner, PCFTable), (beta, lr, attention_lr)),

				# Neural Network
			   # ((ConsolidationBrain, DQN, UANet), (beta, lr)),
			   # ((ConsolidationBrain, DQN, ACLNet), (beta, lr)),
			   # ((ConsolidationBrain, DQNAtt, ACLNet), (beta, lr, attention_lr)),

				# ((ConsolidationBrain, DQN, FCNet), (beta, lr)),
			   # ((ConsolidationBrain, DQN, FC2LayersNet), (beta, lr)),
				# ((ConsolidationBrain, DQN, EfficientNetwork),(beta, lr)),
			   #
				# ((ConsolidationBrain, PG, FCNet), (beta, lr)),
				# ((ConsolidationBrain, PG, UANet), (beta, lr)),
				# ((ConsolidationBrain, PG, ACLNet), (beta, lr)),
				# ((ConsolidationBrain, PG, EfficientNetwork), (beta, lr)),
			   # ((RandomBrain, DQN, Random), (beta, lr))
			   ]



def extract_configuration_params(params):
	return params['br'][0], params['br'][1], params['br'][2], params['lr'], params['non_motivated_reward']


friendly_models_name_map = {
							'QLearner.QTable': 'SARL',
							'QLearner.OptionsTable': 'ORL',
							'IALearner.ACFTable': 'FRL',
							'IALearner.FixedACFTable': 'FARL',
							'MALearner.ACFTable': 'AARL',


							'DQN.FCNet':'FCNet-D',
							'DQN.UANet':'UANet-D',
							'DQN.ACLNet':'ACLNet-D',
							'DQNAtt.ACLNet':'ACLNet-DA',

							'PG.FCNet':'FCNet-P',
							'P.UANet':'UANet-P',
							'PG.ACLNet':'ACLNet-P'

							}