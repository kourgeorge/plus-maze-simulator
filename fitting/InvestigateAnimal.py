__author__ = 'gkour'

import os
import tempfile

import numpy as np
import pandas as pd

import utils
from brains.consolidationbrain import ConsolidationBrain
from brains.tdbrain import TDBrain
from environment import PlusMazeOneHotCues, CueType, PlusMazeOneHotCues2ActiveDoors
from fitting import fitting_config, fitting_utils
from fitting.MazeResultsBehaviouralLC import *
from fitting.fitting_utils import run_model_on_animal_data
from learners.networklearners import *
from learners.tabularlearners import *
from models.networkmodels import *
from models.tabularmodels import *


def writecsvfiletotemp(rat_data_with_likelihood: pd.DataFrame):
	f = tempfile.NamedTemporaryFile(delete=False)
	rat_data_with_likelihood.to_csv(f)
	return f.name


if __name__ == '__main__':
	subject = 7
	model_arch = (TDBrain, ABQLearner, QTable)
	#model_arch = (TDBrain, ABQLearner, OptionsTable)
	#model_arch = (TDBrain, ABIALearner, FTable)
	parameters = (0,1.3, 0.05, 0.03)
	rat_files = [rat_file for rat_file in list(np.sort(os.listdir(fitting_config.MOTIVATED_ANIMAL_DATA_PATH)))]
	rat_data = pd.read_csv(os.path.join(fitting_config.MOTIVATED_ANIMAL_DATA_PATH, rat_files[subject]))
	env = PlusMazeOneHotCues(relevant_cue=CueType.ODOR, stimuli_encoding=10)
	rat_data = fitting_utils.maze_experimental_data_preprocessing(rat_data)
	

	experiment_stats, rat_data_with_likelihood = run_model_on_animal_data(env, rat_data, model_arch, parameters, RewardType(rat_data.iloc[0].initial_motivation))
	rat_data_with_likelihood['model'] = utils.brain_name(model_arch)
	rat_data_with_likelihood['subject'] = [subject] * len(rat_data_with_likelihood)
	rat_data_with_likelihood['parameters'] = [parameters] * len(rat_data_with_likelihood)

	filename = writecsvfiletotemp(rat_data_with_likelihood)
	bias_variables_in_stage(filename)
	#model_values_development(filename)
	#compare_model_subject_learning_curve_average(filename)
	#plot_models_fitting_result_per_stage(filename)
	#model_parameters_development(filename)
	pass
