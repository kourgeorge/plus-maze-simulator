__author__ = 'gkour'

import os
import warnings

import numpy as np
import pandas as pd
from skopt import gp_minimize
import scipy
import fitting.fitting_config_attention as fitting_config

import config
import utils
from environment import PlusMazeOneHotCues2ActiveDoors, CueType, PlusMazeOneHotCues, StagesTransition
from fitting import fitting_utils
from learners.networklearners import DQNAtt
from learners.tabularlearners import MALearner
from models.tabularmodels import FixedACFTable

warnings.filterwarnings("ignore")


class MazeBayesianModelFitting:
	LvsNLL = []

	def __init__(self, env, experiment_data, model, parameters_space, n_calls):
		self.env = env
		self.experiment_data = experiment_data
		self.model = model
		self.parameters_space = parameters_space
		self.n_calls = n_calls

	def _calc_experiment_likelihood(self, parameters):
		(brain, learner, model) = self.model
		if model == FixedACFTable:
			att_o, att_c = parameters[2:]
			if att_o + att_c > 1:
				return 50

		experiment_stats, rat_data_with_likelihood = fitting_utils.run_model_on_animal_data(self.env, self.experiment_data, self.model,
																							parameters, silent=True)
		aic, likelihood_stage, meanL, meanNLL = fitting_utils.analyze_fitting(rat_data_with_likelihood, 'likelihood', len(parameters))

		y = meanNLL

		# print("{}. x={}, AIC:{:2f} (meanNLL={:.2f}, medianNLL={:.2f}, sumNLL={:.2f}, stages={}), \t(meanL={:.2f}, "
		# 	  "medianL={:.3f}, gmeanL={:.3f}, stages={})".format(utils.brain_name(self.model),
		# 														 list(np.round(parameters, 4)),
		# 														 aic,
		# 														 meanNLL, np.nanmedian(NLL), np.nansum(NLL),
		# 														 np.round(likelihood_stage.NLL.to_numpy(), 2),
		# 														 meanL, np.nanmedian(L),
		# 														 geomeanL,
		# 														 np.round(likelihood_stage.likelihood.to_numpy(), 2)))
		
		print("x={}, AIC:{:2f} (meanNLL={:.2f}, stages={}), \t(meanL={:.2f}, stages={})".format(
																 list(np.round(parameters, 4)),
																 aic,
																 meanNLL,
																 np.round(likelihood_stage.NLL.to_numpy(), 2),
																 meanL, 
																 np.round(likelihood_stage.likelihood.to_numpy(), 2)))

		MazeBayesianModelFitting.LvsNLL+=[[meanL, meanNLL]]

		return np.clip(y, a_min=-5000, a_max=5000)

	def optimize(self):
		if fitting_config.OPTIMIZATION_METHOD in ['Bayesian', 'Hybrid']:
			search_result = gp_minimize(self._calc_experiment_likelihood, self.parameters_space, n_calls=self.n_calls)
			x0 = search_result.x

		if fitting_config.OPTIMIZATION_METHOD in ['Newton', 'Hybrid']:
			bounds = [bound.bounds for bound in self.parameters_space]
			x0 = np.array([5, 0.01, 0.01]) if fitting_config.OPTIMIZATION_METHOD == 'Newton' else x0
			print("\t\t-----Finished Bayesian parameter estimation: {} -----".format(np.round(x0,4)))
			if isinstance(self.model, FixedACFTable):

				def constraint(params):
					x, y = params[2:]
					return 1 - (x + y)  # This should be >= 0 when x + y <= 1

				constraints = {'type': 'ineq', 'fun': constraint}  # Inequality constraint: x + y <= 1

				search_result = scipy.optimize.minimize(
					self._calc_experiment_likelihood, x0=x0,
					bounds=bounds,
					#		tol=0.002,
					options={'maxiter': self.n_calls},
					constraints=constraints)
			else:
				search_result = scipy.optimize.minimize(
						self._calc_experiment_likelihood, x0=x0,
						bounds=bounds,
				#		tol=0.002,
						options={'maxiter':self.n_calls})
		
		experiment_stats, rat_data_with_likelihood = fitting_utils.run_model_on_animal_data(self.env, self.experiment_data, self.model,
																							search_result.x)
		n = len(rat_data_with_likelihood)
		aic = - 2 * np.sum(np.log(rat_data_with_likelihood.likelihood)) + 2 * len(search_result.x)
		print("Best Parameters: {} - AIC:{:.3}\n".format(np.round(search_result.x, 4), aic))

		return search_result, experiment_stats, rat_data_with_likelihood

	def _fake_optimize(self):
		"""Use this to validate that the data of all rats are sensible and that
		there is no discrepancy between the data and the behaviour of simulation"""
		class Object(object):
			pass

		print("Warning!! You are running Fake Optimization. This should be used for development purposes only!!")
		(brain, learner, model) = self.model
		if issubclass(learner, MALearner) or issubclass(learner, DQNAtt):
			x = (1, 1.5, 0.05, 15)
		else:
			x= (config.NON_MOTIVATED_REWARD,config.BETA,config.LEARNING_RATE, config.LEARNING_RATE)

		obj = Object()
		obj.x = x
		experiment_stats, rat_data_with_likelihood = fitting_utils.run_model_on_animal_data(self.env, self.experiment_data, self.model, parameters=x)
		return obj, experiment_stats, rat_data_with_likelihood

	@staticmethod
	def all_subjects_all_models_optimization(env, animals_data_folder, all_models, n_calls=35):
		animal_data = [[rat_file, pd.read_csv(os.path.join(animals_data_folder, rat_file))]
					   for rat_file in list(np.sort(os.listdir(animals_data_folder)))]

		timestamp = utils.get_timestamp()
		fitting_results = {}
		results_df = pd.DataFrame()
		for subject_id, (file_name,curr_rat) in enumerate(animal_data):
			initial_motivation = curr_rat.iloc[0].initial_motivation if 'initial_motivation' in curr_rat.columns else 'water'

			print("\n#################### Subject: {} - {}. Env: {} #####################\n".format(subject_id, initial_motivation, str(env)))
			curr_rat = fitting_utils.maze_experimental_data_preprocessing(curr_rat)
			fitting_results[subject_id] = {}
			for curr_model in all_models:
				model, parameters_space = curr_model
				print("-----{}-----".format(utils.brain_name(model)))
				search_result, experiment_stats, rat_data_with_likelihood = \
					MazeBayesianModelFitting(env, curr_rat, model, parameters_space, n_calls).optimize()

				rat_data_with_likelihood["subject"] = subject_id
				rat_data_with_likelihood["model"] = utils.brain_name(model)
				# rat_data_with_likelihood["parameters"] = {name: round(value, 4) for name, value in zip([param.name for param in parameters_space], search_result.x)}
				rat_data_with_likelihood["parameters"] = [np.round(search_result.x,4)] * len(rat_data_with_likelihood)
				rat_data_with_likelihood["algorithm"] = \
					"{}_{}".format(fitting_config.OPTIMIZATION_METHOD, n_calls)

				results_df = results_df.append(rat_data_with_likelihood, ignore_index=True)
			results_df.to_csv('fitting/Results/Rats-Results/fitting_results_dimensional_led_first_{}_{}_tmp.csv'.format(timestamp, n_calls))
		results_df.to_csv('fitting/Results/Rats-Results/fitting_results_dimensional_led_first_{}_{}.csv'.format(timestamp, n_calls))
		return fitting_results


if __name__ == '__main__':

	# fit odor first animals
	# MazeBayesianModelFitting.all_subjects_all_models_optimization(
	#     PlusMazeOneHotCues2ActiveDoors(stages=led_first_stages, stimuli_encoding=10),
	# 	fitting_config.MAZE_ANIMAL_DATA_PATH, fitting_config.maze_models, n_calls=fitting_config.FITTING_ITERATIONS)


	#fit led first animals
	led_first_stages = [{'name': 'LED', 'transition_logic': StagesTransition.set_color_stage},
						{'name': 'Odor1', 'transition_logic': StagesTransition.set_odor_stage}]

	MazeBayesianModelFitting.all_subjects_all_models_optimization(
		PlusMazeOneHotCues2ActiveDoors(stages=led_first_stages, stimuli_encoding=10),
		fitting_config.MAZE_ANIMAL_LED_FIRST_DATA_PATH, fitting_config.maze_models, n_calls=fitting_config.FITTING_ITERATIONS)

	# MazeBayesianModelFitting.all_subjects_all_models_optimization(
	# 	PlusMazeOneHotCues(relevant_cue=CueType.ODOR, stimuli_encoding=10), fitting_config.MOTIVATED_ANIMAL_DATA_PATH,
	# 	fitting_config.maze_models, n_calls=fitting_config.FITTING_ITERATIONS)

