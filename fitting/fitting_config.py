__author__ = 'gkour'

import itertools
from consolidationbrain import ConsolidationBrain, RandomBrain
from fixeddoorattentionbrain import FixedDoorAttentionBrain
from lateoutcomeevaluationbrain import LateOutcomeEvaluationBrain
from tabularmodels import UniformAttentionTabular, TabularQ
from tdbrain import TDBrain
from learner import DQN, TD, TDUniformAttention
from motivationdependantbrain import MotivationDependantBrain
from standardbrainnetwork import SeparateMotivationAreasNetwork, EfficientNetwork, FullyConnectedNetwork, \
	FullyConnectedNetwork2Layers, TabularQ, SeparateMotivationAreasFCNetwork, UniformAttentionTabular

REPETITIONS = 3
ANIMAL_DATA_PATH = './fitting/behavioral_data'
ANIMAL_BATCHES = {1: [1, 2], 2: [1], 4: [6, 7, 8], 5: [1, 2], 6: [10, 11]}
#ANIMAL_BATCHES = {1: [1, 2],  2: [1], 4: [6, 7]}

non_motivated_reward = [0, 0.3]
learning_rates = [0.001, 0.01, 0.2]
brains = [#(TDBrain, TD, TabularQ),
		  (TDBrain, TDUniformAttention, UniformAttentionTabular),
		  (ConsolidationBrain, DQN, FullyConnectedNetwork),
		  # (ConsolidationBrain, DQN, FullyConnectedNetwork2Layers),
		  # (ConsolidationBrain, DQN, EfficientNetwork),
		  # #(FixedDoorAttentionBrain, DQN, EfficientNetwork),
		  # (MotivationDependantBrain, DQN, SeparateMotivationAreasNetwork),
		  # (MotivationDependantBrain, DQN, SeparateMotivationAreasFCNetwork),
		  (LateOutcomeEvaluationBrain, DQN, SeparateMotivationAreasNetwork),
		  # (RandomBrain, DQN, EfficientNetwork)
		  ]

combinations = list(itertools.product(brains, learning_rates, non_motivated_reward))
configs = [{'br': br, 'lr': lr, 'non_motivated_reward': nmr} for br, lr, nmr in combinations]


def extract_configuration_params(params):
	return params['br'][0], params['br'][1], params['br'][2], params['lr'], params['non_motivated_reward']
