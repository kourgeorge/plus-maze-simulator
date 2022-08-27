__author__ = 'gkour'

import os
import numpy as np

import pandas as pd

from fitting.fitting_config import configs, extract_configuration_params, REPETITIONS, ANIMAL_DATA_PATH, ANIMAL_BATCHES
from fitting.fitting_utils import get_timestamp
from motivatedagent import MotivatedAgent
from environment import PlusMazeOneHotCues, CueType
from rewardtype import RewardType
import config
from PlusMazeExperimentFitting import PlusMazeExperimentFitting


def run_fitting(model_params, rat_data_file=None, rat_id=None, repetitions=3):

	# Read behavioral data
	rat_data = pd.read_csv(rat_data_file)

	# Initialize environment
	env = PlusMazeOneHotCues(relevant_cue=CueType.ODOR)

	# Initialize agents
	(brain, learner, network, non_motivated_reward_value, learning_rate) = extract_configuration_params(model_params)

	results_df = pd.DataFrame()
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
				'forgetting': config.FORGETTING,
				'motivated_reward': config.MOTIVATED_REWARD,
				'non_motivated_reward': non_motivated_reward_value,
				'memory_size': config.MEMORY_SIZE, 'learning_rate': learning_rate,
				'trials': len(rat_data),
				'median_full_likelihood': np.median(all_experiment_likelihoods),
				'average_full_likelihood': np.mean(all_experiment_likelihoods),
				'average_likelihood_s1': likelihood_stage[0],
				'average_likelihood_s2': likelihood_stage[1],
				'average_likelihood_s3': likelihood_stage[2],
				'average_likelihood_s4': likelihood_stage[3],
				'average_likelihood_s5': likelihood_stage[4],
				'param_number': agent.get_brain().num_trainable_parameters(), 'repetition': int(rep)}
		results_df = results_df.append(dict, ignore_index=True)
	return results_df


if __name__ == '__main__':

	# create results folder and dataframe
	results_path = os.path.join('Results', 'Rats-Results', get_timestamp())
	if not os.path.exists(results_path):
		os.makedirs(results_path)

	df = pd.DataFrame()
	# go over all model_params and all animals
	for animal_batch in ANIMAL_BATCHES:
		for rat in ANIMAL_BATCHES[animal_batch]:
			rat_id = '{}_{}'.format(animal_batch, rat)
			rat_data_path = os.path.join(ANIMAL_DATA_PATH, 'output_expr{}_rat{}.csv'.format(animal_batch, rat))
			for config_index, params in enumerate(configs):
				run_df = run_fitting(params, rat_data_path, rat_id, repetitions=REPETITIONS)
				df = df.append(run_df, ignore_index=True)
	df.to_csv(os.path.join(results_path, 'outputForAll.csv'), index=False)
