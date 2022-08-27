__author__ = 'gkour'

import os
import numpy as np

import pandas as pd
from fitting.fitting_utils import get_timestamp
from motivatedagent import MotivatedAgent
from environment import PlusMazeOneHotCues, CueType
from rewardtype import RewardType

import config
from standardbrainnetwork import FullyConnectedNetwork, EfficientNetwork, SeparateMotivationAreasNetwork, \
	FullyConnectedNetwork2Layers
from learner import DQN
from PlusMazeExperimentFitting import PlusMazeExperimentFitting
from behavioral_analysis import plot_days_per_stage, plot_behavior_results
from consolidationbrain import ConsolidationBrain

expr_data = {1: [1, 2], 2: [1], 4: [6, 7, 8], 5: [1, 2], 6: [10, 11]}


def override_params(params):
	return params['brain'], params['learner'], params['network'], params['nonmotivated_reward'], params['learning_rate']


def run_fitting(model_params, rat_data_file=None, rat_id=None, results_df=None, repetitions=3):

	# Read behavioral data
	rat_data = pd.read_csv(rat_data_file)

	# Initialize environment
	env = PlusMazeOneHotCues(relevant_cue=CueType.ODOR)

	# Initialize agents
	(brain, learner, network, non_motivated_reward_value, learning_rate) = override_params(model_params)

	for rep in range(repetitions):
		agent = MotivatedAgent(
			brain(learner(network(env.stimuli_encoding_size(), 2, env.num_actions()), learning_rate=learning_rate)),
			motivation=RewardType.WATER,
			motivated_reward_value=config.MOTIVATED_REWARD, non_motivated_reward_value=non_motivated_reward_value)

		# Run the Model
		experiment_stats, all_experiment_likelihoods = PlusMazeExperimentFitting(agent, rat_data=rat_data)

		# Report results
		likelihoods = experiment_stats.epoch_stats_df.Likelihood
		stages = experiment_stats.epoch_stats_df.Stage
		likelihood_stage = np.zeros([5])
		for stage in range(5):
			stage_likelihood = [likelihoods[i] for i in range(len(likelihoods)) if stages[i] == stage]
			if len(stage_likelihood) == 0:
				likelihood_stage[stage] = None
			else:
				likelihood_stage[stage] = np.mean(stage_likelihood)

		dict = {'rat': rat_id, 'brain': brain.__name__, 'learner': learner.__name__,
				'network': network.__name__,
				'forgetting': config.FORGETTING, 'motivated_reward': config.MOTIVATED_REWARD,
				'non_motivated_reward': config.NON_MOTIVATED_REWARD,
				'memory_size': config.MEMORY_SIZE, 'learning_rate': config.LEARNING_RATE, 'trials': len(rat_data),
				'median_full_likelihood': np.median(all_experiment_likelihoods),
				'average_full_likelihood': np.mean(all_experiment_likelihoods),
				'average_likelihood_s1': likelihood_stage[0],
				'average_likelihood_s2': likelihood_stage[1],
				'average_likelihood_s3': likelihood_stage[2],
				'average_likelihood_s4': likelihood_stage[3],
				'average_likelihood_s5': likelihood_stage[4],
				'param_number': agent.get_brain().num_trainable_parameters(), 'repetition': rep}
		results_df = results_df.append(dict, ignore_index=True)
	return results_df


if __name__ == '__main__':

	behavioral_data_path = './behavioral_data'
	params = {
		'brain': ConsolidationBrain,
		'network': EfficientNetwork,
		'learner': DQN,
		'nonmotivated_reward': 0.3,
		'learning_rate': 0.01,
		'forgetting': 0.1
	}

	params_list = [params]

	# create results folder and dataframe
	results_path = os.path.join('Results', 'Rats-Results', get_timestamp())
	if not os.path.exists(results_path):
		os.makedirs(results_path)
	df = pd.DataFrame(columns=['rat', 'brain', 'network', 'forgetting', 'motivated_reward', 'non_motivated_reward',
							   'memory_size', 'learning_rate', 'likelihood', 'trials', 'param_number'])

	# go over all model_params and animal
	for config_index, params in enumerate(params_list):
		for expr in expr_data:
			for rat in expr_data[expr]:
				run_df = run_fitting(params, os.path.join(behavioral_data_path,'output_expr{}_rat{}.csv'.format(expr, rat)),
									 '{}_{}'.format(expr, rat), df, repetitions=3)
	df.to_csv(os.path.join(results_path, 'outputForAll.csv'), index=False)
