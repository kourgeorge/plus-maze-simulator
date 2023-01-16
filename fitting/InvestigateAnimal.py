__author__ = 'gkour'

import os
import tempfile

import numpy as np
import pandas as pd

import utils
from brains.tdbrain import TDBrain
from environment import PlusMazeOneHotCues, CueType
from fitting import fitting_config, fitting_utils
from fitting.MazeResultsBehaviouralMS import compare_model_subject_learning_curve_average
from fitting.fitting_utils import run_model_on_animal_data
from learners.tabularlearners import MALearner, IALearner
from models.tabularmodels import ACFTable, FTable


def writecsvfiletotemp(rat_data_with_likelihood:pd.DataFrame):
	f = tempfile.NamedTemporaryFile()
	rat_data_with_likelihood.to_csv(f)
	return f.name


if __name__ == '__main__':
	subject = 0
	model_arch = (TDBrain, IALearner, FTable)
	parameters = (1.3704, 0.4)
	rat_files = [rat_file for rat_file in list(np.sort(os.listdir(fitting_config.MOTIVATED_ANIMAL_DATA_PATH)))]
	rat_data = pd.read_csv(os.path.join(fitting_config.MOTIVATED_ANIMAL_DATA_PATH ,rat_files[subject]))

	rat_data = fitting_utils.maze_experimental_data_preprocessing(rat_data)
	# test = rat_data.groupby(['A1o','A1c','A2o','A2c','A3o','A3c','A4o','A4c']).mean().reset_index()
	# experiment_stats, rat_data_with_likelihood = run_model_on_animal_data(PlusMazeOneHotCues(relevant_cue=CueType.ODOR, stimuli_encoding=10),
	# 						 rat_data, (TDBrain, MALearner, ACFTable), (8.5, 0.008, 0.4))

	experiment_stats, rat_data_with_likelihood = run_model_on_animal_data(
		PlusMazeOneHotCues(relevant_cue=CueType.ODOR, stimuli_encoding=10),
		rat_data, model_arch, parameters)
	rat_data_with_likelihood['model'] = utils.brain_name(model_arch)
	rat_data_with_likelihood['subject'] = [parameters] * len(rat_data_with_likelihood)

	compare_model_subject_learning_curve_average(writecsvfiletotemp(rat_data_with_likelihood))





