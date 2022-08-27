import pandas as pd
import numpy as np

from standardbrainnetwork import FullyConnectedNetwork, EfficientNetwork, SeparateMotivationAreasNetwork, \
	FullyConnectedNetwork2Layers
from learner import DQN, PG
from scipy.special import logsumexp
from scipy.optimize import minimize
from consolidationbrain import ConsolidationBrain
import config
from PlusMazeExperiment import PlusMazeExperiment, ExperimentStatus
from motivatedagent import MotivatedAgent
from environment import PlusMazeOneHotCues,CueType
from rewardtype import RewardType


def llik_td(x, *args):
	results = []
	repetitions = 3
	for i in range(repetitions):
		env = PlusMazeOneHotCues(relevant_cue=CueType.ODOR)
		[brain, learner, network, motivated_reward_value, non_motivated_reward_value] = \
			[ConsolidationBrain, DQN, FullyConnectedNetwork, config.MOTIVATED_REWARD, config.NON_MOTIVATED_REWARD]

		non_motivated_reward_value = x[0]
		print("Current VARIABLE:{}".format(non_motivated_reward_value))
		agent = MotivatedAgent(brain(
			learner(network(env.stimuli_encoding_size(), 2, env.num_actions()), learning_rate=config.LEARNING_RATE)),
							   motivation=RewardType.WATER,
							   motivated_reward_value=motivated_reward_value,
							   non_motivated_reward_value=non_motivated_reward_value)

		_, all_experiment_likelihoods = PlusMazeExperiment(agent, dashboard=False, rat_data_file=rat_data_file)
		results += [np.median(all_experiment_likelihoods)]

	return np.mean(results)


if __name__ == '__main__':
	# load behavioural data
	expr_data = {1: [1, 2], 2: [1], 4: [6, 7, 8], 5: [1, 2], 6: [10, 11]}
	rat_data_file = './output_expr{}_rat{}.csv'.format(1, 1)
	num_trials = len((pd.read_csv(rat_data_file)).index)
	x0 = 0.3
	bnds = ((0, 0.5),)
	result = minimize(llik_td, x0, bounds=bnds)
	print(result)
