__author__ = 'gkour'

import os
import warnings

import numpy as np
import pandas as pd
from skopt import gp_minimize
import scipy
import fitting.fitting_config as fitting_config

import config
import utils
from environment import PlusMazeOneHotCues2ActiveDoors, CueType, PlusMazeOneHotCues
from fitting import fitting_utils
from fitting.fitting_config import maze_models, MAZE_ANIMAL_DATA_PATH
from fitting.PlusMazeExperimentFitting import PlusMazeExperimentFitting
from fitting.fitting_utils import blockPrint, enablePrint
from learners.networklearners import DQNAtt
from learners.tabularlearners import MALearner
from motivatedagent import MotivatedAgent
from rewardtype import RewardType

warnings.filterwarnings("ignore")


class MazeBayesianModelFitting:
	LvsNLL = []

	def __init__(self, env, experiment_data, model, parameters_space, n_calls):
		self.env = env
		self.experiment_data = experiment_data
		self.model = model
		self.parameters_space = parameters_space
		self.n_calls = n_calls

	def _run_model(self, parameters):
		(brain, learner, model) = self.model
		model_instance = model(self.env.stimuli_encoding_size(), 2, self.env.num_actions())

		if issubclass(learner, MALearner) or issubclass(learner, DQNAtt):
			(beta, lr, attention_lr) = parameters
			learner_instance = learner(model_instance, learning_rate=lr, alpha_phi=attention_lr)
		else:
			(beta, lr) = parameters
			learner_instance = learner(model_instance, learning_rate=lr)

		blockPrint()
		self.env.init()
		agent = MotivatedAgent(brain(learner_instance, beta=beta),
			motivation=RewardType.WATER,
			motivated_reward_value=config.MOTIVATED_REWARD,
			non_motivated_reward_value=config.NON_MOTIVATED_REWARD, exploration_param=0)

		experiment_stats, rat_data_with_likelihood = PlusMazeExperimentFitting(self.env, agent, dashboard=False,
																			   experiment_data=self.experiment_data)
		enablePrint()

		return experiment_stats, rat_data_with_likelihood

	def _calc_experiment_likelihood(self, parameters):
		model = self.model

		experiment_stats, rat_data_with_likelihood = self._run_model(parameters)
		rat_data_with_likelihood['NLL'] = -np.log(rat_data_with_likelihood.likelihood)
		likelihood_day = rat_data_with_likelihood.groupby(['stage', 'day in stage']).mean().reset_index()
		likelihood_stage = likelihood_day.groupby('stage').mean()

		NLL = rat_data_with_likelihood.NLL.to_numpy()
		n = len(NLL)
		L = rat_data_with_likelihood.likelihood.to_numpy()
		meanNLL = np.nanmean(NLL)
		meanL = np.nanmean(rat_data_with_likelihood.likelihood)
		geomeanL = scipy.stats.mstats.gmean(rat_data_with_likelihood.likelihood, nan_policy='omit')
		np.testing.assert_almost_equal(np.exp(-meanNLL), geomeanL)
		aic = 2 * np.sum(NLL)/n + 2 * len(parameters) / n
		y = meanNLL

		print("{}. x={}, AIC:{:2f} (meanNLL={:.2f}, medianNLL={:.2f}, sumNLL={:.2f}, stages={}), \t(meanL={:.2f}, "
			  "medianL={:.3f}, gmeanL={:.3f}, stages={})".format(utils.brain_name(model),
																 list(np.round(parameters, 4)),
																 aic,
																 meanNLL, np.nanmedian(NLL), np.nansum(NLL),
																 np.round(likelihood_stage.NLL.to_numpy(), 2),
																 meanL, np.nanmedian(L),
																 geomeanL,
																 np.round(likelihood_stage.likelihood.to_numpy(), 2)))

		MazeBayesianModelFitting.LvsNLL+=[[meanL, meanNLL]]

		return np.clip(y, a_min=-5000, a_max=5000)

	def optimize(self):
		if fitting_config.BAYESIAN_OPTIMIZATION:
			search_result = gp_minimize(self._calc_experiment_likelihood, self.parameters_space, n_calls=self.n_calls)
		else:
			x0 = np.array([5,0.01,0.15])
			search_result = scipy.optimize.minimize(
				self._calc_experiment_likelihood, x0=x0[:len(self.parameters_space)],
				bounds=self.parameters_space,
				options={'maxiter':self.n_calls})
		experiment_stats, rat_data_with_likelihood = self._run_model(search_result.x)
		n = len(rat_data_with_likelihood)
		aic = - 2 * np.sum(np.log(rat_data_with_likelihood.likelihood))/n + 2 * len(search_result.x)/n
		print("Best Parameters: {} - AIC:{:.3}".format(np.round(search_result.x, 4), aic))

		return search_result, experiment_stats, rat_data_with_likelihood

	def _fake_optimize(self):
		"""Use this to validate that the data of all rats are sensible and that
		there is no discrepancy between the data and the behaviour of simulation"""
		class Object(object):
			pass

		print("Warning!! You are running Fake Optimization. This should be used for development purposes only!!")
		(brain, learner, model) = self.model
		if issubclass(learner, MALearner) or issubclass(learner, DQNAtt):
			x = (1.5, 0.05, 15)
		else:
			x= (1.5, 0.05)

		obj = Object()
		obj.x = x
		experiment_stats, rat_data_with_likelihood = self._run_model(x)
		return obj, experiment_stats, rat_data_with_likelihood

	@staticmethod
	def all_subjects_all_models_optimization(env, animals_data_folder, all_models, n_calls=35):
		animal_data = [pd.read_csv(os.path.join(animals_data_folder, rat_file))
					   for rat_file in list(np.sort(os.listdir(animals_data_folder)))]

		timestamp = utils.get_timestamp()
		fitting_results = {}
		results_df = pd.DataFrame()
		for subject_id, curr_rat in enumerate(animal_data):
			print("{}".format(subject_id))
			curr_rat = fitting_utils.maze_experimental_data_preprocessing(curr_rat)
			fitting_results[subject_id] = {}
			for curr_model in all_models:
				model, parameters_space = curr_model
				search_result, experiment_stats, rat_data_with_likelihood = \
					MazeBayesianModelFitting(env, curr_rat, model, parameters_space, n_calls).optimize()

				rat_data_with_likelihood['subject'] = subject_id
				rat_data_with_likelihood["model"] = utils.brain_name(model)
				rat_data_with_likelihood["parameters"] = [search_result.x] * len(rat_data_with_likelihood)
				rat_data_with_likelihood["algorithm"] = "Bayesian" if fitting_config.BAYESIAN_OPTIMIZATION else "BFGS"

				results_df = results_df.append(rat_data_with_likelihood, ignore_index=True)
			results_df.to_csv('fitting/Results/Rats-Results/fitting_results_{}_{}_tmp.csv'.format(timestamp, n_calls))
		results_df.to_csv('fitting/Results/Rats-Results/fitting_results_{}_{}.csv'.format(timestamp, n_calls))
		return fitting_results


if __name__ == '__main__':
	# MazeBayesianModelFitting.all_subjects_all_models_optimization(
	# 	PlusMazeOneHotCues2ActiveDoors(relevant_cue=CueType.ODOR, stimuli_encoding=10),
	# 	MAZE_ANIMAL_DATA_PATH, maze_models, n_calls=30)

	MazeBayesianModelFitting.all_subjects_all_models_optimization(
		PlusMazeOneHotCues(relevant_cue=CueType.ODOR, stimuli_encoding=10), fitting_config.MOTIVATED_ANIMAL_DATA_PATH,
		maze_models, n_calls=fitting_config.FITTING_ITERATIONS)

