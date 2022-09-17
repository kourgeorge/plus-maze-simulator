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
import seaborn as sns

warnings.filterwarnings("ignore")


class MazeBayesianModelFitting():
	def __init__(self, env, experiment_data, model, parameters_space, n_calls):
		self.env = env
		self.experiment_data = experiment_data
		self.model = model
		self.parameters_space = parameters_space
		self.n_calls = n_calls

	def _run_model(self, parameters):
		(brain, learner, network) = self.model
		(beta, lr, batch_size) = parameters
		blockPrint()
		self.env.init()
		agent = MotivatedAgent(brain(
			learner(network(self.env.stimuli_encoding_size(), 2, self.env.num_actions()), learning_rate=lr),
			batch_size=batch_size, beta=beta),
			motivation=RewardType.WATER,
			motivated_reward_value=config.MOTIVATED_REWARD,
			non_motivated_reward_value=0)

		experiment_stats, rat_data_with_likelihood = PlusMazeExperimentFitting(self.env, agent, dashboard=False,
																				 rat_data=self.experiment_data)
		enablePrint()

		return experiment_stats, rat_data_with_likelihood

	def _calc_experiment_likelihood(self, parameters):
		model = self.model
		experiment_stats, rat_data_with_likelihood = self._run_model(parameters)
		likelihood_stage = rat_data_with_likelihood.groupby('stage').mean()['likelihood']
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

		x = (1.5, 0.05, 15)
		obj = Object()
		obj.x = x
		experiment_stats, all_experiment_likelihoods = self._run_model(x)
		return obj, experiment_stats, all_experiment_likelihoods


def all_subjects_all_models_optimization(env, animals_data_folder, all_models, n_calls=35):
	animal_data = [pd.read_csv(os.path.join(animals_data_folder, rat_file))
				   for rat_file in list(np.sort(os.listdir(animals_data_folder)))]

	fitting_results = {}
	results_df = pd.DataFrame()
	for animal_id, curr_rat in enumerate(animal_data):
		print(animal_id)
		fitting_results[animal_id] = {}
		for curr_model in all_models:
			model, parameters_space = curr_model
			search_result, experiment_stats, rat_data_with_likelihood = \
				MazeBayesianModelFitting(env, curr_rat, model, parameters_space, n_calls).optimize()
			rat_data_with_likelihood['subject'] = animal_id
			rat_data_with_likelihood["model"] = fitting_utils.brain_name(model)
			rat_data_with_likelihood["parameters"] = [search_result.x] * len(rat_data_with_likelihood)

			results_df = results_df.append(rat_data_with_likelihood, ignore_index=True)

	results_df.to_csv('fitting/Results/Rats-Results/fitting_results_{}.csv'.format(fitting_utils.get_timestamp()))
	# with open('fitting/Results/Rats-Results/fitting_results_{}.pkl'.format(fitting_utils.get_timestamp()),
	# 		  'wb') as f:
	# 	pickle.dump(fitting_results, f)
	return fitting_results


def plot_all_subjects_fitting_results(env, results_file_path):
	# for fitness, x in sorted(zip(search_result.func_vals, search_result.x_iters)):
	# 	print(fitness, x)
	with open(results_file_path, 'rb') as f:
		results = pickle.load(f)

	ds = pd.DataFrame()
	for rat, rat_value in results.items():
		for brain, brain_results in rat_value.items():
			row = {'brain': brain,
				   'animal': rat,
				   'stages_likl': np.round(brain_results['stages_likl'], 3),
				   'exp_likl': np.round(brain_results['exp_likl'], 3),
				   'parameters': np.round(brain_results['parameters'], 3)}
			ds = ds.append(row, ignore_index=True)
			for stage in range(len(env.stage_names)):
				ds[env.stage_names[stage]] = np.stack(list(ds['stages_likl']))[:, stage]

	# params = "(nmr:{}, lr:{}, bs:{})".format(*np.round(brain_results['values'], 3))
	# print("{}, {}: \t{} {}".format(rat, brain, np.round(brain_results['likelihood'],3), params))
	# plots.plot_histogram(result=brain_results, dimension_identifier='lr', bins=20)
	# plots.plot_objective_2D(brain_results['results'], 'lr', 'batch_size')
	# plots.plot_objective(brain_results['results'], plot_dims=['nmr', 'lr'])

	sns.set_theme(style="darkgrid")

	df_unpivot = pd.melt(ds, id_vars=['brain', 'animal'], value_vars=['exp_likl'] + env.stage_names)
	df_unpivot['exp_likl'] = df_unpivot['value']
	sns.boxplot(x='variable', y='exp_likl', hue='brain', data=df_unpivot)
	del ds['stages_likl']


if __name__ == '__main__':
	# subject_expr_data = pd.read_csv('fitting/maze_behavioral_data/output_expr_rat0.csv')
	# model, parameters_space = maze_models[0]
	# fitter = MazeBayesianModelFitting(PlusMazeOneHotCues2ActiveDoors(relevant_cue=CueType.ODOR, stimuli_encoding=10),
	# 								  subject_expr_data, model, parameters_space, n_calls=15)
	# search_result, experiment_stats, likelihood_stage, all_experiment_likelihoods = fitter.optimize()

	# # fitting = MazeBayesianModelFitting(PlusMazeOneHotCues(relevant_cue=CueType.ODOR, stimuli_encoding=10),
	# # 								   animals_data_folder=MOTIVATED_ANIMAL_DATA_PATH, brains=brains)
	# #

	all_subjects_all_models_optimization(PlusMazeOneHotCues2ActiveDoors(relevant_cue=CueType.ODOR, stimuli_encoding=10),
										 MAZE_ANIMAL_DATA_PATH, maze_models, n_calls=11)

	# fitting = plot_all_subjects_fitting_results(PlusMazeOneHotCues2ActiveDoors(relevant_cue=CueType.ODOR, stimuli_encoding=10),
	# 											 animals_data_folder=MAZE_ANIMAL_DATA_PATH, all_models=maze_models,
	# 											 n_calls=10)

	# plot_all_subjects_fitting_results(PlusMazeOneHotCues2ActiveDoors(relevant_cue=CueType.ODOR, stimuli_encoding=10),
	# 								  'fitting/Results/Rats-Results/fitting_results_2022_09_15_05_17.pkl')
