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

MOTIVATED_ANIMAL_DATA_PATH = './fitting/motivation_behavioral_data_all'
#MOTIVATED_ANIMAL_DATA_PATH = './fitting/motivation_behavioral_data_food'
MAZE_ANIMAL_DATA_PATH = './fitting/maze_behavioral_data'
OPTIMIZATION_METHOD = 'Hybrid' # 'Newton', 'Bayesian'
MOTIVATED_ANIMAL_BATCHES = {1: [1, 2], 2: [1], 4: [6, 7, 8], 5: [1, 2], 6: [10, 11]}
MAZE_ANIMALS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
FITTING_ITERATIONS = 50

lr = Real(name='lr', low=0.001, high=0.4, prior='log-uniform')
attention_lr = Real(name='attention_lr', low=0.001, high=0.4, prior='log-uniform')
bias_lr = Real(name='bias_lr', low=0.001, high=0.4, prior='log-uniform')
beta = Real(name='beta', low=0.1, high=10, prior='log-uniform')
initial_value = Real(name='initial', low=0, high=0.1, prior='uniform')
nmr = Real(name='nmr', low=-1, high=1)
batch_size = Integer(name='batch_size', low=1, high=20)


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
				((TDBrain, QLearner, QTable), (nmr, beta, lr, bias_lr)), #SARL
				((TDBrain, UABQLearner, QTable), (nmr, beta, lr, bias_lr)), #B-SARL
				((TDBrain, ABQLearner, QTable), (nmr, beta, lr, bias_lr)), # M(B)-SARL

			    ((TDBrain, QLearner, OptionsTable), (nmr, beta, lr, bias_lr)),#ORL
				((TDBrain, UABQLearner, OptionsTable), (nmr, beta, lr, bias_lr)),#B-ORL
				((TDBrain, ABQLearner, OptionsTable), (nmr, beta, lr, bias_lr)),# M(B)-ORL

				((TDBrain, IALearner, FTable), (nmr, beta, lr, bias_lr)),#FRL
				((TDBrain, UABIALearner, FTable), (nmr, beta, lr, bias_lr)),#B-FRL
				((TDBrain, ABIALearner, FTable), (nmr, beta, lr, bias_lr)), # M(B)-FRL
]

maze_MD = [
		((TDBrain, ABQLearner, QTable), (nmr, beta, lr, bias_lr)),  # M(B)-SARL
		((TDBrain, ABQLearner, OptionsTable), (nmr, beta, lr, bias_lr)), # M(B)-ORL
		((TDBrain, ABIALearner, FTable), (nmr, beta, lr, bias_lr)), #M(B)-FRL

		((TDBrain, QLearner, MQTable), (nmr, beta, lr, bias_lr)),  # M(V)-SARL
		((TDBrain, QLearner, MOptionsTable), (nmr, beta, lr, bias_lr)),  # M(V)-ORL
		((TDBrain, IALearner, MFTable), (nmr, beta, lr, bias_lr)),  # M(V)-FRL

		((TDBrain, ABQLearner, MQTable), (nmr, beta, lr, bias_lr)),  # M(VB)-SARL
		((TDBrain, ABQLearner, MOptionsTable), (nmr, beta, lr, bias_lr)),  # M(VB)-ORL
		((TDBrain, ABIALearner,MFTable), (nmr, beta, lr, bias_lr)) #M(VB)-FRL

]

bias_vs_stimuli=[
	((TDBrain, ABQLearner, QTable), (beta, lr)),  # M(B)-SARL
	((TDBrain, ABQLearner, OptionsTable), (beta, lr)),  # M(B)-ORL
	((TDBrain, ABIALearner, FTable), (beta, lr)),  # M(B)-FRL

]


maze_SCFRL = [
	((TDBrain, ABIALearner, FTable), (nmr, beta, lr, bias_lr)), #M(B)-FRL
	((TDBrain, IALearner, SCFTable), (nmr, beta, lr, bias_lr)),  # E(V)-FRL
	((TDBrain, IALearner, SCVBFTable), (nmr, beta, lr, bias_lr)),  # E(VB)-FRL
	((TDBrain, ABIALearner, SCFTable), (nmr, beta, lr, bias_lr)), #M(B)-E(V)-FRL
	((TDBrain, ABIALearner, SCVBFTable), (nmr, beta, lr, bias_lr)), #M(B)-E(VB)-FRL
	((TDBrain, ABIALearner, MSCFTable), (nmr, beta, lr, bias_lr)) #M(VB)-E(V)-FRL
]

maze_EC=[
	((TDBrain, QLearner, SCBQTable), (nmr, beta, lr, bias_lr)),  # E(B)-SARL
	((TDBrain, ABQLearner, SCBQTable), (nmr, beta, lr, bias_lr)),  # E(B)-M(B)-SARL

	((TDBrain, QLearner, SCBOptionsTable), (nmr, beta, lr, bias_lr)),  # E(B)-ORL
	((TDBrain, ABQLearner, SCBOptionsTable), (nmr, beta, lr, bias_lr)),  # E(B)-M(B)-ORL

	((TDBrain, IALearner, SCBFTable), (nmr, beta, lr, bias_lr)),  # E(B)-FRL
	((TDBrain, ABIALearner, SCBFTable), (nmr, beta, lr, bias_lr)),  # E(B)-M(B)-FRL
]

maze_models = maze_models_action_bias + maze_MD[3:] + maze_SCFRL[1:] + maze_EC


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
							'UABQLearner.QTable': 'B-SARL',
							'ABQLearner.QTable': 'M(B)-SARL',
							'QLearner.MQTable': 'M(V)-SARL',
							'ABQLearner.MQTable': 'M(VB)-SARL',
							'QLearner.SCBQTable': 'E(B)-SARL',
							'ABQLearner.SCBQTable': 'E(B)-M(B)-SARL',

							'QLearner.OptionsTable': 'ORL',
							'UABQLearner.OptionsTable': 'B-ORL',
							'ABQLearner.OptionsTable': 'M(B)-ORL',
							'QLearner.MOptionsTable': 'M(V)-ORL',
							'ABQLearner.MOptionsTable': 'M(VB)-ORL',
							'QLearner.SCBOptionsTable': 'E(B)-ORL',
							'ABQLearner.SCBOptionsTable': 'E(B)-M(B)-ORL',

							'IALearner.FTable': 'FRL',
							'UABIALearner.FTable': 'B-FRL',
							'ABIALearner.FTable': 'M(B)-FRL',
							'IALearner.MFTable': 'M(V)-FRL',
							'ABIALearner.MFTable': 'M(VB)-FRL',
							'IALearner.SCBFTable': 'E(B)-FRL',
							'ABIALearner.SCBFTable': 'E(B)-M(B)-FRL',

							'IALearner.SCFTable': 'E(V)-FRL',
							'UABIALearner.SCFTable': 'E(V)-B-FRL',
							'IALearner.SCVBFTable': 'E(VB)-FRL',

							'ABIALearner.SCFTable': 'E(V)-M(B)-FRL',
							'ABIALearner.SCVBFTable': 'E(VB)-M(B)-FRL',
							'ABIALearner.MSCFTable': 'E(V)-M(VB)-FRL',

							'IAAluisiLearner.ACFTable': 'MFRL',
							'MALearner.ACFTable': 'AARL',
							'UABMALearner.ACFTable': 'B-AARL',
							'ABMALearner.ACFTable': 'M(B)-AARL',


							'MALearnerSimple.ACFTable':'MAARL',
							'DQN.FCNet':'FCNet',
							'DQN.FC2LayersNet':'FC2Net',
							'DQN.UANet':'UANet',
							'DQN.ACLNet':'ACLNet',
							'DQNAtt.ACLNet':'ACLNet2'
							}