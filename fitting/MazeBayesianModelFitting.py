import os
import pickle
import sys

import pandas as pd
import numpy as np
from skopt import gp_minimize
from skopt import plots

from fitting.PlusMazeExperimentFitting import PlusMazeExperimentFitting
from fitting.fitting_config import MAZE_ANIMAL_DATA_PATH, MOTIVATED_ANIMAL_DATA_PATH, maze_models
import fitting_utils
import config
from motivatedagent import MotivatedAgent
from environment import PlusMazeOneHotCues, CueType, PlusMazeOneHotCues2ActiveDoors
from rewardtype import RewardType
import PlusMazeExperiment as PlusMazeExperiment
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

def blockPrint():
	sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
	sys.stdout = sys.__stdout__


class MazeBayesianModelFitting:
	def __init__(self, env, animals_data_folder, models, n_calls=35):
		self.n_calls = n_calls
		self.env = env
		self.animal_data = [pd.read_csv(os.path.join(animals_data_folder, rat_file))
							for rat_file in list(np.sort(os.listdir(animals_data_folder)))]
		self.models = models
		self.curr_rat = None
		self.curr_model = None

	def run_fitting(self, parameters):
		(brain, learner, network),_ = self.curr_model
		(beta, lr, batch_size) = parameters

		blockPrint()
		self.env.init()
		agent = MotivatedAgent(brain(
			learner(network(self.env.stimuli_encoding_size(), 2, self.env.num_actions()), learning_rate=lr),
			batch_size=batch_size, beta=beta),
			motivation=RewardType.WATER,
			motivated_reward_value=config.MOTIVATED_REWARD,
			non_motivated_reward_value=0)

		experiment_stats, all_experiment_likelihoods = PlusMazeExperimentFitting(self.env, agent, dashboard=False,
																				 rat_data=self.curr_rat)
		enablePrint()

		daily_likelihoods = experiment_stats.epoch_stats_df.Likelihood
		stages = experiment_stats.epoch_stats_df.Stage
		all_stages = len(self.env.stage_names)
		likelihood_stage = np.zeros([all_stages])
		for stage in range(all_stages):
			stage_likelihood = [daily_likelihoods[i] for i in range(len(daily_likelihoods)) if stages[i] == stage]
			if len(stage_likelihood) == 0:
				likelihood_stage[stage] = None
			else:
				likelihood_stage[stage] = np.nanmean(stage_likelihood)
		return likelihood_stage, all_experiment_likelihoods

	def calc_experiment_likelihood(self, parameters):
		model, params_space = self.curr_model
		likelihood_stage, all_experiment_likelihoods = self.run_fitting(parameters)
		y = np.nanmean(likelihood_stage)

		print("{}.\tx={},\t\ty={:.3f},\tstages={} \toverall_mean={:.3f}".format(fitting_utils.brain_name(model),
																		list(np.round(parameters, 4)), y,
														  				np.round(likelihood_stage, 3),
			  															np.mean(all_experiment_likelihoods)))
		return np.clip(y, a_min=0, a_max=50)

	def run_bayesian_optimization(self):

		fitting_results = {}
		for animal_id, self.curr_rat in enumerate(self.animal_data):
			print(animal_id)
			fitting_results[animal_id] = {}
			for self.curr_model in self.models:
				model, space = self.curr_model
				search_result = gp_minimize(self.calc_experiment_likelihood, space,
											n_calls=self.n_calls)
				likelihood_stage, all_experiment_likelihoods = self.run_fitting(search_result.x)
				fitting_results[animal_id][fitting_utils.brain_name(model)] = \
																			{"exp_likl": np.mean(all_experiment_likelihoods),
																			 "stages_likl": likelihood_stage,
																			"trial_likl": all_experiment_likelihoods,
																			"parameters": search_result.x,
																			"results": search_result}

		with open('fitting/Results/Rats-Results/fitting_results_{}.pkl'.format(fitting_utils.get_timestamp()),
				  'wb') as f:
			pickle.dump(fitting_results, f)
		return fitting_results

	@staticmethod
	def plot_results(env, results_file_path):
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

		df_unpivot = pd.melt(ds, id_vars=['brain', 'animal'], value_vars=['exp_likl']+env.stage_names)
		df_unpivot['exp_likl'] = df_unpivot['value']
		sns.boxplot(x='variable', y='exp_likl', hue='brain', data=df_unpivot)
		del ds['stages_likl']
		# f, axes = plt.subplots(3, 3)

		# for i, animal in enumerate(np.unique(ds['animal'])):
		# 	plt.figure(figsize=(3.5, 3.5))
		# 	ds_filtered = ds[ds['animal'] == animal]
		# 	g = sns.scatterplot(y="lr", x="nmr",
		# 						hue="brain", size='likelihood',
		# 						data=ds_filtered, sizes=(20, 200), alpha=0.7)
		#
		# 	# g.set(yscale="log")
		# 	g.yaxis.grid(True, "minor", linewidth=.25)


if __name__ == '__main__':
	# fitting = MazeBayesianModelFitting(PlusMazeOneHotCues(relevant_cue=CueType.ODOR, stimuli_encoding=10),
	# 								   animals_data_folder=MOTIVATED_ANIMAL_DATA_PATH, brains=brains)

	fitting = MazeBayesianModelFitting(PlusMazeOneHotCues2ActiveDoors(relevant_cue=CueType.ODOR, stimuli_encoding=10),
									   animals_data_folder=MAZE_ANIMAL_DATA_PATH, models=maze_models).run_bayesian_optimization()


	# MazeBayesianModelFitting.plot_results(PlusMazeOneHotCues2ActiveDoors(relevant_cue=CueType.ODOR, stimuli_encoding=10),
	# 									  'fitting/Results/Rats-Results/fitting_results_2022_09_13_21_34.pkl')

	x=1
