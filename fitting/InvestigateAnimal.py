__author__ = 'gkour'

import os

import numpy as np
import pandas as pd

from brains.tdbrain import TDBrain
from environment import PlusMazeOneHotCues, CueType
from fitting import fitting_config, fitting_utils
from fitting.fitting_utils import run_model_on_animal_data
from learners.tabularlearners import MALearner
from models.tabularmodels import ACFTable


if __name__ == '__main__':
	rat_files = [rat_file for rat_file in list(np.sort(os.listdir(fitting_config.MOTIVATED_ANIMAL_DATA_PATH)))]
	rat_data = pd.read_csv(os.path.join(fitting_config.MOTIVATED_ANIMAL_DATA_PATH ,rat_files[3]))
	rat_data = fitting_utils.maze_experimental_data_preprocessing(rat_data)
	experiment_stats, rat_data_with_likelihood = run_model_on_animal_data(PlusMazeOneHotCues(relevant_cue=CueType.ODOR, stimuli_encoding=10),
							 rat_data, (TDBrain, MALearner, ACFTable), (8.5, 0.008, 0.4))


