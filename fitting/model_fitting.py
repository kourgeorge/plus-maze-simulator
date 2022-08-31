import os
import pickle
import random
import sys

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from skopt import gp_minimize
from skopt import plots
from skopt.space import Real, Integer

from fitting.PlusMazeExperimentFitting import PlusMazeExperimentFitting
from fitting.fitting_config import ANIMAL_BATCHES, ANIMAL_DATA_PATH, brains
from standardbrainnetwork import FullyConnectedNetwork, EfficientNetwork, SeparateMotivationAreasNetwork, \
	FullyConnectedNetwork2Layers
from learner import DQN, PG
from scipy.optimize import minimize
from consolidationbrain import ConsolidationBrain
import config
from motivatedagent import MotivatedAgent
from environment import PlusMazeOneHotCues, CueType
from rewardtype import RewardType

import seaborn as sns

animal_batch = None
rat = None
architecture = None

results_file_name = 'results_mean_stage_likelihood.pkl'

env = PlusMazeOneHotCues(relevant_cue=CueType.ODOR)

# Disable
def blockPrint():
	sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
	sys.stdout = sys.__stdout__

def llik_td(x):
	# torch.manual_seed(0)
	# np.random.seed(0)
	# random.seed(0)

	(nmr, lr, batch_size) = x
	rat_data_path = os.path.join(ANIMAL_DATA_PATH, 'output_expr{}_rat{}.csv'.format(animal_batch, rat))
	rat_data = pd.read_csv(rat_data_path)

	brain, learner, network = architecture
	agent = MotivatedAgent(brain(
		learner(network(env.stimuli_encoding_size(), 2, env.num_actions()), learning_rate=lr), batch_size=batch_size),
		motivation=RewardType.WATER,
		motivated_reward_value=config.MOTIVATED_REWARD,
		non_motivated_reward_value=nmr)

	blockPrint()
	experiment_stats, all_experiment_likelihoods = PlusMazeExperimentFitting(agent, dashboard=False, rat_data=rat_data)
	enablePrint()

	daily_likelihoods = experiment_stats.epoch_stats_df.Likelihood
	stages = experiment_stats.epoch_stats_df.Stage
	likelihood_stage = np.zeros([5])
	for stage in range(5):
		stage_likelihood = [daily_likelihoods[i] for i in range(len(daily_likelihoods)) if stages[i] == stage]
		if len(stage_likelihood) == 0:
			likelihood_stage[stage] = None
		else:
			likelihood_stage[stage] = np.mean(stage_likelihood)
	return likelihood_stage, all_experiment_likelihoods


def fun(x):
	likelihood_stage, all_experiment_likelihoods = llik_td(x)
	y = np.nanmean(likelihood_stage)
	print("x={} ,y={}, stages={}".format(x, y, likelihood_stage))
	return y


def brain_name(architecture):
	return "{}.{}.{}".format(architecture[0].__name__, architecture[1].__name__, architecture[2].__name__)


def run_all_data():
	global animal_batch, rat, architecture
	# load behavioural data
	x0_nmr = 0.3
	x0_lr = 0.005
	x0_batch_size = 10

	nmr = Real(name='nmr', low=0.0, high=1)
	lr = Real(name='lr', low=0.0001, high=0.1, prior='log-uniform')
	batch_size = Integer(name='batch_size', low=1, high=20)
	space = [nmr, lr, batch_size]

	results = {}
	for animal_batch in ANIMAL_BATCHES:
		for rat in ANIMAL_BATCHES[animal_batch]:
			animal_id = "{}_{}".format(animal_batch, rat)
			results[animal_id] = {}
			for architecture in brains:
				search_result = gp_minimize(fun, space, x0=(x0_nmr, x0_lr, x0_batch_size), n_calls=30)
				likelihood_stage, all_experiment_likelihoods = llik_td(search_result.x)
				results[animal_id][brain_name(architecture)] = {"likelihood": search_result.fun,
																"values": search_result.x,
																"stages": likelihood_stage,
																"results": search_result}

	with open(results_file_name, 'wb') as f:
		pickle.dump(results, f)
# for fitness, x in sorted(zip(search_result.func_vals, search_result.x_iters)):
# 	print(fitness, x)


def plot_resuls():
	with open(results_file_name, 'rb') as f:
		results = pickle.load(f)

	ds = pd.DataFrame()
	for rat, rat_value in results.items():
		for brain, brain_results in rat_value.items():
			row = {'brain': brain,
				   'animal': rat,
				   'likelihood': np.round(1.5-brain_results['likelihood'], 3),
				   'nmr': np.round(brain_results['values'][0], 3),
				   'lr': np.round(brain_results['values'][1], 3),
				   'bs': np.round(brain_results['values'][2], 3)
				   }
			ds = ds.append(row, ignore_index=True)
	# params = "(nmr:{}, lr:{}, bs:{})".format(*np.round(brain_results['values'], 3))
	# print("{}, {}: \t{} {}".format(rat, brain, np.round(brain_results['likelihood'],3), params))
	# plots.plot_histogram(result=brain_results, dimension_identifier='lr', bins=20)
	# plots.plot_objective_2D(brain_results['results'], 'lr', 'batch_size')
	# plots.plot_objective(brain_results['results'], plot_dims=['nmr', 'lr'])

	sns.set_theme(style="darkgrid")

	#f, axes = plt.subplots(3, 3)

	for i, animal in enumerate(np.unique(ds['animal'])):
		plt.figure(figsize=(3.5,3.5))
		ds_filtered = ds[ds['animal'] == animal]
		g = sns.scatterplot(y="lr", x="nmr",
						hue="brain", size='likelihood',
						data=ds_filtered, sizes=(20, 200),  alpha= 0.7)

		#g.set(yscale="log")
		g.yaxis.grid(True, "minor", linewidth=.25)

	x = 1


if __name__ == '__main__':
	run_all_data()
	# plot_resuls()
