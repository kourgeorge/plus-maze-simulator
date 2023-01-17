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
from fitting.MazeResultsBehaviouralMS import compare_model_subject_learning_curve_average, \
	plot_models_fitting_result_per_stage
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
	subject = 6
	model_arch = (ConsolidationBrain, DQN, UANet)
	parameters = (3.72, 0.3485)
	rat_files = [rat_file for rat_file in list(np.sort(os.listdir(fitting_config.MOTIVATED_ANIMAL_DATA_PATH)))]
	rat_data = pd.read_csv(os.path.join(fitting_config.MOTIVATED_ANIMAL_DATA_PATH, rat_files[subject]))
	env = PlusMazeOneHotCues(relevant_cue=CueType.ODOR, stimuli_encoding=10)
	rat_data = fitting_utils.maze_experimental_data_preprocessing(rat_data)
	# test = rat_data.groupby(['A1o','A1c','A2o','A2c','A3o','A3c','A4o','A4c']).mean().reset_index()
	# experiment_stats, rat_data_with_likelihood = run_model_on_animal_data(PlusMazeOneHotCues(relevant_cue=CueType.ODOR, stimuli_encoding=10),
	# 						 rat_data, (TDBrain, MALearner, ACFTable), (8.5, 0.008, 0.4))

	experiment_stats, rat_data_with_likelihood = run_model_on_animal_data(env, rat_data, model_arch, parameters)
	rat_data_with_likelihood['model'] = utils.brain_name(model_arch)
	rat_data_with_likelihood['subject'] = [parameters] * len(rat_data_with_likelihood)

	filename = writecsvfiletotemp(rat_data_with_likelihood)

	compare_model_subject_learning_curve_average(filename)
	plot_models_fitting_result_per_stage(filename)
	pass
