import os
import random

import pandas as pd
import numpy as np
import torch

from fitting.PlusMazeExperimentFitting import PlusMazeExperimentFitting
from fitting.fitting_config import ANIMAL_BATCHES, ANIMAL_DATA_PATH
from standardbrainnetwork import FullyConnectedNetwork, EfficientNetwork, SeparateMotivationAreasNetwork, \
	FullyConnectedNetwork2Layers
from learner import DQN, PG
from scipy.optimize import minimize
from consolidationbrain import ConsolidationBrain
import config
from motivatedagent import MotivatedAgent
from environment import PlusMazeOneHotCues,CueType
from rewardtype import RewardType


def llik_td(x, *args):
	(nmr, lr, batch_size) = x
	rat_data_file = args[0]
	rat_data = pd.read_csv(rat_data_file)

	torch.manual_seed(0)
	np.random.seed(0)
	random.seed(0)

	env = PlusMazeOneHotCues(relevant_cue=CueType.ODOR)
	[brain, learner, network, motivated_reward_value, non_motivated_reward_value] = \
		[ConsolidationBrain, DQN, FullyConnectedNetwork2Layers, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD]

	print("Current VARIABLE:{}".format(x))
	agent = MotivatedAgent(brain(
		learner(network(env.stimuli_encoding_size(), 2, env.num_actions()), learning_rate=lr), batch_size),
		motivation=RewardType.WATER,
		motivated_reward_value=motivated_reward_value,
		non_motivated_reward_value=nmr)

	_, all_experiment_likelihoods = PlusMazeExperimentFitting(agent, dashboard=False, rat_data=rat_data)
	return np.median(all_experiment_likelihoods)



if __name__ == '__main__':
	# load behavioural data
	nmr = 0.5
	lr = 0.005
	batch_size = 2
	bnds = ((0, 0.7),(0.001, 0.01), (1,20))
	for animal_batch in ANIMAL_BATCHES:
		for rat in ANIMAL_BATCHES[animal_batch]:
			rat_data_path = os.path.join(ANIMAL_DATA_PATH, 'output_expr{}_rat{}.csv'.format(animal_batch, rat))
			rat_id = '{}_{}'.format(animal_batch, rat)
			result = minimize(llik_td, [nmr, lr, batch_size], rat_data_path, bounds=bnds)
			print(result)
