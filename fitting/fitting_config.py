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
from models.tabularmodels import PCFTable, MFTable, MQTable, MOptionsTable, SCFTable, SCVBFTable

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
	beta = Real(name='beta', low=0.1, high=30, prior='uniform')
	initial_value = Real(name='initial', low=0, high=0.1, prior='uniform')
else:
	lr = (0.001, 0.4)
	attention_lr = (0.001, 0.4)
	beta = (0, 30)
	initial_value = (0, 0.15)

FITTING_ITERATIONS = 25

maze_models = [
				# ((TDBrain, QLearner, QTable), (beta, lr)),
				((TDBrain, ABQLearner, QTable), (beta, lr)),
				#
			    # ((TDBrain, QLearner, OptionsTable), (beta, lr)),
				# ((TDBrain, ABQLearner, OptionsTable), (beta, lr)),
				#
			    # ((TDBrain, IALearner, ACFTable), (beta, lr)),
			    #((TDBrain, IAAluisiLearner, ACFTable), (beta, lr)),
				#
				# ((TDBrain, IALearner, FTable), (beta, lr)),
			    # ((TDBrain, IALearner, ACFTable), (beta, lr)),

				# ((TDBrain, MALearner, ACFTable), (beta, lr, attention_lr)),

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


maze_models_action_bias = [
				# ((TDBrain, QLearner, QTable), (beta, lr)),
				# ((TDBrain, UABQLearner, QTable), (beta, lr)),
				# ((TDBrain, ABQLearner, QTable), (beta, lr)),
				#
			    # ((TDBrain, QLearner, OptionsTable), (beta, lr)),
				# ((TDBrain, UABQLearner, OptionsTable), (beta, lr)),
				# ((TDBrain, ABQLearner, OptionsTable), (beta, lr)),
				#
				# ((TDBrain, IALearner, FTable), (beta, lr)),
				 #((TDBrain, ABIALearner, SCFTable), (beta, lr)),
				# ((TDBrain, ABIALearner, FTable), (beta, lr)),


				# ((TDBrain, IALearner, FTable), (beta, lr)),
				# ((TDBrain, IALearner, ACFTable), (beta, lr)),

				((TDBrain, QLearner, MQTable), (beta, lr)),
				((TDBrain, ABQLearner, MQTable), (beta, lr)),

				((TDBrain, QLearner, MOptionsTable), (beta, lr)),
				((TDBrain, ABQLearner, MOptionsTable), (beta, lr)),

				((TDBrain, IALearner, MFTable), (beta, lr)),
				((TDBrain, ABIALearner, MFTable), (beta, lr)),

	]

maze_SCFRL = [
	((TDBrain, ABIALearner, FTable), (beta, lr)), #AB-FRL
	((TDBrain, ABIALearner, SCFTable), (beta, lr)), #AB-SC-FRL
	((TDBrain, ABIALearner, SCVBFTable), (beta, lr)) #SCVB-FRL

]

maze_models = maze_SCFRL


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
							'UABQLearner.QTable': 'AB-SARL',
							'ABQLearner.QTable': 'MAB-SARL',
							'QLearner.MQTable': 'MD-SARL',
							'ABQLearner.MQTable': 'MAB-MD-SARL',

							'QLearner.OptionsTable': 'ORL',
							'UABQLearner.OptionsTable': 'AB-ORL',
							'ABQLearner.OptionsTable': 'MAB-ORL',
							'QLearner.MOptionsTable': 'MD-ORL',
							'ABQLearner.MOptionsTable': 'MAB-MD-ORL',

							'IALearner.FTable': 'FRL',
							'UABIALearner.FTable': 'AB-FRL',
							'ABIALearner.FTable': 'MAB-FRL',
							'IALearner.MFTable': 'MD-FRL',
							'ABIALearner.MFTable': 'MAB-MD-FRL',

							'ABIALearner.SCFTable': 'SDV-MAB-FRL',
							'ABIALearner.SCVBFTable': 'SDVAB-MAB-FRL',

							'IALearner.SCFTable': 'SDV-FRL',
							'UABIALearner.SCFTable': 'AB-SD-FRL',

							'IALearner.SCVBFTable': 'VAB-SD-FRL',

							'IAAluisiLearner.ACFTable': 'MFRL',
							'MALearner.ACFTable': 'AARL',
							'UABMALearner.ACFTable': 'AB-AARL',
							'ABMALearner.ACFTable': 'MAB-AARL',


							'MALearnerSimple.ACFTable':'MAARL',
							'DQN.FCNet':'FCNet',
							'DQN.FC2LayersNet':'FC2Net',
							'DQN.UANet':'UANet',
							'DQN.ACLNet':'ACLNet',
							'DQNAtt.ACLNet':'ACLNet2'
							}