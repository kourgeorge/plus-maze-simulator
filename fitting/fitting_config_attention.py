__author__ = 'gkour'

import itertools

from skopt.space import Real, Integer

from brains.consolidationbrain import ConsolidationBrain
from brains.fixeddoorattentionbrain import FixedDoorAttentionBrain
from brains.lateoutcomeevaluationbrain import LateOutcomeEvaluationBrain
from brains.motivationdependantbrain import MotivationDependantBrain
from brains.tdbrain import TDBrain
from fitting.skopt_priors import Exponential, Beta, Dirichlet
from learners.networklearners import *
from learners.tabularbayesianlearners import BayesianAttentionLearner, BayesianFeatureLearner
from learners.tabularlearners import *
from models.networkmodels import *
from models.non_directional_tabularmodels import NonDirectionalOptionsTable, \
	NonDirectionalACFTable, NonDirectionalFixedACFTable
from models.tabularbayesianmodels import BayesianACFTable, BayesianFTable
from models.tabularmodels import *

MAZE_ANIMAL_DATA_PATH = '/Users/georgekour/repositories/plus-maze-simulator/fitting/maze_behavioral_data'
MAZE_ANIMAL_LED_FIRST_DATA_PATH = '/Users/georgekour/repositories/plus-maze-simulator/fitting/maze_behavioral_led_first_data'
MAZE_ANIMALS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
MAZE_LED_FIRST_ANIMALS = [33, 34, 35, 36, 37, 38, 39, 40, 41, 42]

OPTIMIZATION_METHOD = 'Hybrid' #  'Hybrid','Newton', 'Bayesian'
FITTING_ITERATIONS = 200

lr = Real(name='lr', low=0.0001, high=0.4, prior='log-uniform')
lr_nr = Real(name='lr_nr', low=0.0001, high=0.4, prior='log-uniform')
attention_lr = Real(name='attention_lr', low=0.0001, high=0.4, prior='log-uniform')
beta = Real(name='beta', low=0.1, high=10, prior='log-uniform')
# beta = Beta(scale=10, alpha=2, beta_param=5, name='beta')
attention_color = Real(name='attention_color', low=0, high=1, prior='uniform')
attention_odor = Real(name='attention_odor', low=0, high=1, prior='uniform')
initial_feature_value = Real(name='initial_feature_value', low=0, high=1, prior='uniform')
batch_size = Integer(name='batch_size', low=1, high=20)

attention_distribution = Dirichlet(alpha=[1, 1, 1], name='dirichlet_param')

# New hyperparameters for Bayesian models
initial_variance = Real(name='initial_variance', low=0.1, high=100, prior='log-uniform')
observation_variance = Real(name='observation_variance', low=0.1, high=10, prior='log-uniform')
initial_alpha = Real(name='initial_alpha', low=0.1, high=10, prior='log-uniform')
attention_beta = Real(name='attention_beta', low=0.001, high=1.0, prior='log-uniform')


maze_models = [
	((TDBrain, QLearner, QTable), (initial_feature_value, beta, lr)),
	((TDBrain, QLearner, OptionsTable), (initial_feature_value, beta, lr)),
	((TDBrain, IALearner, ACFTable), (initial_feature_value, beta, lr)),
	((TDBrain, IALearner, FixedACFTable), (initial_feature_value, beta, lr, attention_distribution)),
	((TDBrain, MALearner, ACFTable), (initial_feature_value, beta, lr, attention_lr)),

	# # #
	# ((TDBrain, AsymmetricQLearner, QTable), (initial_feature_value, beta, lr, lr_nr)),
	# ((TDBrain, AsymmetricQLearner, OptionsTable), (initial_feature_value, beta, lr, lr_nr)),
	# ((TDBrain, AsymmetricIALearner, ACFTable), (initial_feature_value, beta, lr, lr_nr)),
	# ((TDBrain, AsymmetricIALearner, FixedACFTable), (initial_feature_value, beta, lr, lr_nr, attention_distribution)),
	# ((TDBrain, AsymmetricMALearner, ACFTable), (initial_feature_value, beta, lr, lr_nr, attention_lr)),
	#
	# ((TDBrain, QLearner, NonDirectionalOptionsTable), (initial_feature_value, beta, lr)),
	# ((TDBrain, IALearner, NonDirectionalACFTable), (initial_feature_value, beta, lr)),
	# ((TDBrain, IALearner, NonDirectionalFixedACFTable), (initial_feature_value, beta, lr, attention_odor)),
	# ((TDBrain, MALearner, NonDirectionalACFTable), (initial_feature_value, beta, lr, attention_lr)),

	# ((TDBrain, BayesianFeatureLearner, BayesianFTable), (initial_feature_value, beta, initial_variance, observation_variance)),
	# Bayesian Attention Learner
	# ((TDBrain, BayesianAttentionLearner, BayesianACFTable), (initial_feature_value, beta, initial_variance, observation_variance, initial_alpha, attention_beta)),

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


map_maze_models = {f'{model[0][1].__name__}.{model[0][2].__name__}':model for model in maze_models}


def extract_configuration_params(params):
	return params['br'][0], params['br'][1], params['br'][2], params['lr'], params['non_motivated_reward']


friendly_models_name_map = {
							'QLearner.QTable': 'SARL',
							'QLearner.OptionsTable': 'ORL',
							'IALearner.ACFTable': 'FRL',
							'IALearner.FixedACFTable': 'FARL',
							'MALearner.ACFTable': 'AARL',

							'AsymmetricQLearner.QTable': 'As. SARL',
							'AsymmetricQLearner.OptionsTable': 'As. ORL',
							'AsymmetricIALearner.ACFTable': 'As. FRL',
							'AsymmetricIALearner.FixedACFTable': 'As. FARL',
							'AsymmetricMALearner.ACFTable': 'As. AARL',

							'QLearner.NonDirectionalOptionsTable': 'N.D. ORL',
							'IALearner.NonDirectionalACFTable': 'N.D. FRL',
							'IALearner.NonDirectionalFixedACFTable': 'N.D. FARL',
							'MALearner.NonDirectionalACFTable': 'N.D. AARL',

							# Add friendly names for Bayesian models
							'BayesianFeatureLearner.BayesianFTable': 'Bayesian FRL',
							'BayesianAttentionLearner.BayesianACFTable': 'Bayesian AARL',

							'DQN.FCNet':'FCNet-D',
							'DQN.UANet':'UANet-D',
							'DQN.ACLNet':'ACLNet-D',
							'DQNAtt.ACLNet':'ACLNet-DA',

							'PG.FCNet':'FCNet-P',
							'P.UANet':'UANet-P',
							'PG.ACLNet':'ACLNet-P'
							}


def get_parameter_names(model):
	parameter_names = {
		'SARL': ['initial_feature_value','beta', 'alpha'],
		'ORL': ['initial_feature_value', 'beta', 'alpha'],
		'FRL': ['initial_feature_value', 'beta', 'alpha'],
		'AARL': ['initial_feature_value', 'beta', 'alpha', 'alpha_phi'],
		'FARL': ['initial_feature_value', 'beta', 'alpha', 'att_odor', 'att_color'],

		'N.D. ORL': ['initial_feature_value', 'beta', 'alpha'],
		'N.D. FRL': ['initial_feature_value', 'beta', 'alpha'],
		'N.D. AARL': ['initial_feature_value', 'beta', 'alpha', 'alpha_phi'],
		'N.D. FARL': ['initial_feature_value', 'beta', 'alpha', 'att_odor'],

		'As. SARL': ['initial_feature_value', 'beta', 'alpha_rew', 'alpha_no_rew'],
		'As. ORL': ['initial_feature_value', 'beta', 'alpha_rew', 'alpha_no_rew'],
		'As. FRL': ['initial_feature_value', 'beta', 'alpha_rew', 'alpha_no_rew'],
		'As. AARL': ['initial_feature_value', 'beta', 'alpha_rew', 'alpha_no_rew', 'alpha_phi'],
		'As. FARL': ['initial_feature_value', 'beta', 'alpha_rew', 'alpha_no_rew', 'att_odor', 'att_color'],


		# Add parameter names for Bayesian models
		'Bayesian FRL': ['initial_feature_value', 'beta', 'observation_variance'],
		'Bayesian AARL': ['initial_feature_value', 'beta',  'observation_variance', 'initial_alpha', 'attention_beta'],

	}
	return parameter_names.get(model, [])
