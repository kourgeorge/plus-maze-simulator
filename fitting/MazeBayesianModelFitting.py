__author__ = 'gkour'

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from skopt import gp_minimize

import config
from environment import PlusMazeOneHotCues2ActiveDoors, CueType
from fitting import fitting_utils
from fitting.fitting_config import maze_models, MAZE_ANIMAL_DATA_PATH
from fitting.PlusMazeExperimentFitting import PlusMazeExperimentFitting
from fitting.fitting_utils import blockPrint, enablePrint
from motivatedagent import MotivatedAgent
from rewardtype import RewardType

warnings.filterwarnings("ignore")


class MazeBayesianModelFitting:
	def __init__(self, env, experiment_data, model, parameters_space, n_calls):
		self.env = env
		self.experiment_data = experiment_data
		self.model = model
		self.parameters_space = parameters_space
		self.n_calls = n_calls

	def _run_model(self, parameters):
		(brain, learner, model) = self.model
		(beta, lr, batch_size) = parameters
		blockPrint()
		self.env.init()
		agent = MotivatedAgent(brain(
			learner(model(self.env.stimuli_encoding_size(), 2, self.env.num_actions()), learning_rate=lr),
			batch_size=batch_size, beta=beta),
			motivation=RewardType.WATER,
			motivated_reward_value=config.MOTIVATED_REWARD,
			non_motivated_reward_value=0)

		experiment_stats, rat_data_with_likelihood = PlusMazeExperimentFitting(self.env, agent, dashboard=False,
																			   experiment_data=self.experiment_data)
		enablePrint()

		return experiment_stats, rat_data_with_likelihood

	def _calc_experiment_likelihood(self, parameters):
		model = self.model
		experiment_stats, rat_data_with_likelihood = self._run_model(parameters)
		likelihood_stage = list(rat_data_with_likelihood.groupby('stage').mean()['likelihood'])
		y = np.nanmean(likelihood_stage)

		print("{}.\tx={},\t\ty={:.3f},\tstages={} \toverall_mean={:.3f}".format(fitting_utils.brain_name(model),
																				list(np.round(parameters, 4)), y,
																				np.round(likelihood_stage, 3),
																				np.mean(rat_data_with_likelihood.likelihood)))
		return np.clip(y, a_min=0, a_max=50)

	def optimize(self):
		search_result = gp_minimize(self._calc_experiment_likelihood, self.parameters_space,
									n_calls=self.n_calls)
		experiment_stats, rat_data_with_likelihood = self._run_model(search_result.x)
		return search_result, experiment_stats, rat_data_with_likelihood

	def _fake_optimize(self):
		class Object(object):
			pass

		print("Warning!! You are running Fake Optimization. This should be used for development purposes only!!")

		x = (1.5, 0.05, 15)
		obj = Object()
		obj.x = x
		experiment_stats, rat_data_with_likelihood = self._run_model(x)
		return obj, experiment_stats, rat_data_with_likelihood

	@staticmethod
	def all_subjects_all_models_optimization(env, animals_data_folder, all_models, n_calls=35):
		animal_data = [pd.read_csv(os.path.join(animals_data_folder, rat_file))
					   for rat_file in list(np.sort(os.listdir(animals_data_folder)))]

		timestamp=fitting_utils.get_timestamp()
		fitting_results = {}
		results_df = pd.DataFrame()
		for subject_id, curr_rat in enumerate(animal_data):
			print("Optimizing models for subject: {}".format(subject_id))
			fitting_results[subject_id] = {}
			for curr_model in all_models:
				model, parameters_space = curr_model
				search_result, experiment_stats, rat_data_with_likelihood = \
					MazeBayesianModelFitting(env, curr_rat, model, parameters_space, n_calls).optimize()
				rat_data_with_likelihood['subject'] = subject_id
				rat_data_with_likelihood["model"] = fitting_utils.brain_name(model)
				rat_data_with_likelihood["parameters"] = [search_result.x] * len(rat_data_with_likelihood)

				results_df = results_df.append(rat_data_with_likelihood, ignore_index=True)
			results_df.to_csv('fitting/Results/Rats-Results/fitting_results{}_tmp.csv'.format(timestamp))
		results_df.to_csv('fitting/Results/Rats-Results/fitting_results{}.csv'.format(timestamp))
		return fitting_results


if __name__ == '__main__':

	MazeBayesianModelFitting.all_subjects_all_models_optimization(
		PlusMazeOneHotCues2ActiveDoors(relevant_cue=CueType.ODOR, stimuli_encoding=10),
										 MAZE_ANIMAL_DATA_PATH, maze_models, n_calls=35)
