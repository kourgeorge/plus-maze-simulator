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
from models.tabularmodels import PCFTable

REPETITIONS = 50
MOTIVATED_ANIMAL_DATA_PATH = './fitting/motivation_behavioral_data'
MAZE_ANIMAL_DATA_PATH = './fitting/maze_behavioral_data'
BAYESIAN_OPTIMIZATION = True
MOTIVATED_ANIMAL_BATCHES = {1: [1, 2], 2: [1], 4: [6, 7, 8], 5: [1, 2], 6: [10, 11]}
MAZE_ANIMALS = [0, 1, 2, 3, 4, 5, 6, 7, 8]

nmr = Real(name='nmr', low=-1, high=1)
batch_size = Integer(name='batch_size', low=1, high=20)

if BAYESIAN_OPTIMIZATION:
	lr = Real(name='lr', low=0.001, high=0.4, prior='log-uniform')
	attention_lr = Real(name='attention_lr', low=0.001, high=0.4, prior='log-uniform')
	beta = Real(name='beta', low=0.1, high=30, prior='log-uniform')
	initial_value = Real(name='initial', low=0, high=0.1, prior='uniform')
else:
	lr = (0.001, 0.2)
	attention_lr = (0.001, 0.2)
	beta = (0, 30)
	initial_value = (0, 0.15)

FITTING_ITERATIONS = 50

maze_models = [
				# ((TDBrain, QLearner, QTable), (beta, lr)),
				# ((TDBrain, ActionBiasedQLearner, QTable), (beta, lr)),
				#
			    # ((TDBrain, QLearner, OptionsTable), (beta, lr)),
				# ((TDBrain, ActionBiasedQLearner, OptionsTable), (beta, lr)),
				#
			    # ((TDBrain, IALearner, ACFTable), (beta, lr)),
			    # ((TDBrain, IAAluisiLearner, ACFTable), (beta, lr)),
				#
				# ((TDBrain, IALearner, FTable), (beta, lr)),
			    #((TDBrain, IALearner, ACFTable), (beta, lr)),

				#((TDBrain, MALearner, ACFTable), (beta, lr, attention_lr)),

			   # ((TDBrain, MALearnerSimple, ACFTable), (beta, lr, attention_lr)),
			   # ((TDBrain, MALearner, PCFTable), (beta, lr, attention_lr)),

			   ((ConsolidationBrain, DQN, FCNet), (beta, lr)),
			   ((ConsolidationBrain, DQN, UANet), (beta, lr)),
			   #((ConsolidationBrain, DQN, ACLNet), (beta, lr)),
			   #((ConsolidationBrain, DQNAtt, ACLNet), (beta, lr, attention_lr)),
			   #((ConsolidationBrain, DQN, FC2LayersNet), (beta, lr)),

				#((ConsolidationBrain, DQN, EfficientNetwork),(beta, lr)),
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


friendly_models_name_map = {'QLearner.QTable': 'SARL',
							'ActionBiasedQLearner.QTable': 'ABSARL',
							'QLearner.OptionsTable': 'ORL',
							'ActionBiasedQLearner.OptionsTable': 'ABORL',
							'IALearner.FTable': 'FRL',
							'IALearner.ACFTable': 'SCFRL',
							'IAAluisiLearner.ACFTable': 'MFRL',
							'MALearner.ACFTable': 'AARL',
							'MALearnerSimple.ACFTable':'MAARL',
							'DQN.FCNet':'FCNet',
							'DQN.FC2LayersNet':'FC2Net',
							'DQN.UANet':'UANet',
							'DQN.ACLNet':'ACLNet',
							'DQNAtt.ACLNet':'ACLNet2'
							}