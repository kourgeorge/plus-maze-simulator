__author__ = 'gkour'

import os
import tempfile
import re
import numpy as np
import pandas as pd
import utils

from brains.consolidationbrain import ConsolidationBrain
from brains.tdbrain import TDBrain
from environment import PlusMazeOneHotCues, CueType, PlusMazeOneHotCues2ActiveDoors
from fitting.MazeResultsBehavioural import *
from fitting.fitting_utils import run_model_on_animal_data, extract_names_from_architecture
from learners.networklearners import *
from learners.tabularlearners import *
from models.networkmodels import *
from models.tabularmodels import *


def writecsvfiletotemp(rat_data_with_likelihood: pd.DataFrame):
	f = tempfile.NamedTemporaryFile(delete=False)
	rat_data_with_likelihood.to_csv(f)
	return f.name


def string2list(string):
	try:
		params= [float(x.strip()) for x in re.split(" +",string.strip(' ()]['))]
	except Exception:
		params= [float(x.strip()) for x in re.split(",",string.strip('()]['))]
	return params


def rerun_simulation():
	recalculated_df = pd.DataFrame()

	all_rat_data = pd.read_csv(
		'/fitting/Results/Rats-Results/reported_results_dimensional_shifting/main_results_reported_10_1_relevant.csv')

	# all_rat_data = pd.read_csv(
	# 	'/Users/georgekour/repositories/plus-maze-simulator/fitting/Results/Rats-Results/fitting_results_2023_09_12_AARL_best.csv')

	for model in all_rat_data['model'].unique():
		df = all_rat_data[all_rat_data['model'] == model]
		#df['parameters'] = df['parameters'].apply(string2list)
		learner_name, model_name = extract_names_from_architecture(df['model'].iloc[0])

		for subject in np.unique(df['subject']):
			learner_class = globals()[learner_name]
			model_class = globals()[model_name]

			model_arch = (TDBrain,learner_class, model_class)

			rat_data = df[df['subject'] == subject]
			parameters = tuple(string2list(rat_data['parameters'].iloc[0]))

			env = PlusMazeOneHotCues2ActiveDoors(relevant_cue=CueType.ODOR, stimuli_encoding=10)
			# env = PlusMazeOneHotCues(relevant_cue=CueType.ODOR, stimuli_encoding=10)
			# rat_data = fitting_utils.maze_experimental_data_preprocessing(rat_data)

			# experiment_stats, rat_data_with_likelihood = run_model_on_animal_data(env, rat_data, model_arch, parameters, RewardType(rat_data.iloc[0].initial_motivation))
			experiment_stats, rat_data_with_likelihood = run_model_on_animal_data(env, rat_data, model_arch, parameters,
																				  silent=False)
			rat_data_with_likelihood['model'] = utils.brain_name(model_arch)
			rat_data_with_likelihood['subject'] = [subject] * len(rat_data_with_likelihood)
			rat_data_with_likelihood['parameters'] = [parameters] * len(rat_data_with_likelihood)

			recalculated_df = recalculated_df.append(rat_data_with_likelihood, ignore_index=True)

	recalculated_df['algorithm'] = all_rat_data['algorithm']
	filename = writecsvfiletotemp(recalculated_df)
	return filename


if __name__ == '__main__':
	filename = rerun_simulation()

	#filename_softmax = '/var/folders/wy/czm63mcx4sx4_k3b740mrx0c0000gn/T/tmpc2mw_rwz'
	#filename_no_softmax_in_value_update = '/var/folders/wy/czm63mcx4sx4_k3b740mrx0c0000gn/T/tmprhwk54vf'
	#filename = filename_softmax

	#filename = '/fitting/Results/Rats-Results/fitting_results_2023_09_11_hybrid_optimization.csv'
	#filename = '/Users/georgekour/repositories/plus-maze-simulator/fitting/Results/Rats-Results/fitting_results_2023_09_12_AARL_best.csv'

	compare_fitting_criteria(filename)
	model_parameters_development(filename, reward_dependant_trials=None)
	investigate_regret_delta_relationship(filename)

	# bias_variables_in_stage(filename)
	# model_values_development(filename)
	# compare_model_subject_learning_curve_average(filename)
	plot_models_fitting_result_per_stage(filename)

	x=1