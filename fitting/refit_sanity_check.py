"""
Standalone sanity-check script: re-run the real parameter-fitting pipeline
from scratch (on one subject, using SARL and AARL) and compare against the
paper's already-published fitted parameters/metrics.

Bypasses MazeBayesianModelFitting.all_subjects_all_models_optimization(),
which calls the removed pandas DataFrame.append() API and crashes on the
installed pandas 3.0.0. Everything else (optimizer, objective function,
data preprocessing) is the same real pipeline used for the paper.
"""
__author__ = 'sanity-check'

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import PlusMazeOneHotCues2ActiveDoors
from fitting import fitting_config_attention as fitting_config
from fitting import fitting_utils
from fitting.MazeBayesianModelFitting import MazeBayesianModelFitting

SUBJECT_FILE = 'output_expr_rat0.csv'
N_CALLS = 30  # reduced from the paper's FITTING_ITERATIONS=200, for a fast sanity check

if __name__ == '__main__':
    env = PlusMazeOneHotCues2ActiveDoors(stimuli_encoding=10)
    rat_data = pd.read_csv(os.path.join(fitting_config.MAZE_ANIMAL_DATA_PATH, SUBJECT_FILE))
    rat_data = fitting_utils.maze_experimental_data_preprocessing(rat_data)

    models_to_test = {
        'SARL': fitting_config.maze_models[0],
        'AARL': fitting_config.maze_models[3],
    }

    for friendly_name, (model, parameters_space) in models_to_test.items():
        print(f"\n\n===== Fitting {friendly_name} on {SUBJECT_FILE} (n_calls={N_CALLS}) =====\n")
        fitter = MazeBayesianModelFitting(env, rat_data, model, parameters_space,
                                           fitting_config.OPTIMIZATION_METHOD, N_CALLS)
        parameters, experiment_stats, rat_data_with_likelihood = fitter.optimize()
        aic, bic, likelihood_stage, meanL, meanNLL = fitting_utils.analyze_fitting(
            rat_data_with_likelihood, 'likelihood', len(fitting_utils.flatten_list(parameters)))
        print(f"\n>>> RESULT {friendly_name}: parameters={fitting_utils.recursive_round(parameters)}, "
              f"AIC={aic:.2f}, BIC={bic:.2f}, meanL={meanL:.4f}, meanNLL={meanNLL:.4f}\n")
