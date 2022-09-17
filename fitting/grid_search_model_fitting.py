__author__ = 'gkour'

import os
import numpy as np
import pandas as pd
import torch
import random

from fitting.fitting_config import configs, extract_configuration_params, REPETITIONS, MOTIVATED_ANIMAL_DATA_PATH, \
	MOTIVATED_ANIMAL_BATCHES, MAZE_ANIMAL_DATA_PATH, MAZE_ANIMALS
from fitting.fitting_utils import get_timestamp
from motivatedagent import MotivatedAgent
from environment import PlusMazeOneHotCues2ActiveDoors, CueType, PlusMazeOneHotCues
from rewardtype import RewardType
import config
from PlusMazeExperimentFitting import PlusMazeExperimentFitting


def run_fitting(env, model_params, rat_data_file=None, rat_id=None, repetitions=3):

	# Read behavioral data
	rat_data = pd.read_csv(rat_data_file)

	# Initialize agents
	(brain, learner, network, learning_rate, non_motivated_reward_value) = extract_configuration_params(model_params)

	results_df = pd.DataFrame()
	for rep in range(repetitions):
		# Initialize environment
		env.init()
		agent = MotivatedAgent(
			brain(learner(network(env.stimuli_encoding_size(), 2, env.num_actions()), learning_rate=learning_rate)),
			motivation=RewardType.WATER,
			motivated_reward_value=config.MOTIVATED_REWARD, non_motivated_reward_value=non_motivated_reward_value)

		# Run the fitting process
		experiment_stats, rat_data_likelihood = PlusMazeExperimentFitting(env, agent, rat_data=rat_data)

		# Report results
		#plot_behavior_results([experiment_stats])
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
				'forgetting': config.FORGETTING,
				'motivated_reward': config.MOTIVATED_REWARD,
				'non_motivated_reward': non_motivated_reward_value,
				'memory_size': config.MEMORY_SIZE, 'learning_rate': learning_rate,
				'trials': len(rat_data),
				'median_full_likelihood': np.median(rat_data_likelihood.likelihood),
				'average_full_likelihood': np.mean(rat_data_likelihood.likelihood),
				'average_likelihood_s1': likelihood_stage[0],
				'average_likelihood_s2': likelihood_stage[1],
				'average_likelihood_s3': likelihood_stage[2],
				'average_likelihood_s4': likelihood_stage[3],
				'average_likelihood_s5': likelihood_stage[4],
				'param_number': agent.get_brain().num_trainable_parameters(), 'repetition': int(rep)}
		results_df = results_df.append(dict, ignore_index=True)
	return results_df


def fit_motivation_rat_data():
	# create results folder and dataframe
	results_path = os.path.join('Results', 'Rats-Results', get_timestamp())
	if not os.path.exists(results_path):
		os.makedirs(results_path)

	df = pd.DataFrame()
	# go over all model_params and all animals
	for animal_batch in MOTIVATED_ANIMAL_BATCHES:
		for animal in MOTIVATED_ANIMAL_BATCHES[animal_batch]:
			rat_id = '{}_{}'.format(animal_batch, animal)
			rat_data_path = os.path.join(MOTIVATED_ANIMAL_DATA_PATH, 'output_expr{}_rat{}.csv'.format(animal_batch, animal))
			for config_index, params in enumerate(configs):
				print("Batch:{}, Animal:{}.\nParameters:{}".format(animal_batch, animal, params))
				env = PlusMazeOneHotCues(relevant_cue=CueType.ODOR)
				run_df = run_fitting(env, params, rat_data_path, rat_id, repetitions=REPETITIONS)
				df = df.append(run_df, ignore_index=True)
		df.to_csv(os.path.join(results_path, 'results_until_batch_{}.csv'.format(animal_batch)), index=False)
	df.to_csv(os.path.join(results_path, 'outputForAll.csv'), index=False)


def fit_maze_rat_data():
	# create results folder and dataframe
	results_path = os.path.join('Results', 'Rats-Results', get_timestamp())
	if not os.path.exists(results_path):
		os.makedirs(results_path)

	df = pd.DataFrame()
	# go over all model_params and all animals
	for rat_id in MAZE_ANIMALS:
		rat_data_path = os.path.join(MAZE_ANIMAL_DATA_PATH, 'output_expr_rat{}.csv'.format(rat_id))
		for config_index, params in enumerate(configs):
			print("Animal:{}.\nParameters:{}".format(rat_id, params))
			env = PlusMazeOneHotCues2ActiveDoors(relevant_cue=CueType.ODOR)
			run_df = run_fitting(env, params, rat_data_path, rat_id, repetitions=REPETITIONS)
			df = df.append(run_df, ignore_index=True)
		df.to_csv(os.path.join(results_path, 'results_until_rat_{}.csv'.format(rat_id)), index=False)


	df.to_csv(os.path.join(results_path, 'outputForAll.csv'), index=False)


if __name__ == '__main__':
	#fit_motivation_rat_data()
	fit_maze_rat_data()