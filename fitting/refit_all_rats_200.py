"""
Full re-run: all 4 models (SARL, ORL, FRL, AARL) x all 8 rats, using:
  - the widened parameter search bounds (fitting_config_attention.py)
  - a single, uniform n_calls=200 for every model (fairness: same iteration
    budget per model, matching the paper's own FITTING_ITERATIONS, per
    explicit instruction to not scale iterations by model complexity)

Bypasses MazeBayesianModelFitting.all_subjects_all_models_optimization()'s
own CSV-writing loop only to get incremental per-fit checkpointing (one row
appended to CSV as each fit finishes, so partial progress survives an
interruption). The actual fitting call (MazeBayesianModelFitting.optimize())
is the same, now-fixed pipeline code.
"""
__author__ = 'sanity-check'

import os
import sys
import csv
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import PlusMazeOneHotCues2ActiveDoors
from fitting import fitting_config_attention as fitting_config
from fitting import fitting_utils
from fitting.MazeBayesianModelFitting import MazeBayesianModelFitting

N_CALLS = 200  # same fair budget for every model, matches paper's FITTING_ITERATIONS
OUT_CSV = os.path.join(os.path.dirname(__file__), 'Results', 'Rats-Results', 'refit_all_rats_200_results.csv')

MODELS = [
    ('SARL', fitting_config.maze_models[0]),
    ('ORL', fitting_config.maze_models[1]),
    ('FRL', fitting_config.maze_models[2]),
    ('AARL', fitting_config.maze_models[3]),
]

FIELDS = ['subject', 'file', 'model', 'n', 'parameters', 'AIC', 'BIC', 'meanL', 'meanNLL']


def append_row(row):
    write_header = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == '__main__':
    rat_files = sorted(os.listdir(fitting_config.MAZE_ANIMAL_DATA_PATH))
    env = PlusMazeOneHotCues2ActiveDoors(stimuli_encoding=10)

    for subject_id, rat_file in enumerate(rat_files):
        raw = pd.read_csv(os.path.join(fitting_config.MAZE_ANIMAL_DATA_PATH, rat_file))
        rat_data = fitting_utils.maze_experimental_data_preprocessing(raw)

        for friendly_name, (model, parameters_space) in MODELS:
            print(f"\n\n===== Fitting {friendly_name} on {rat_file} (subject {subject_id}, n_calls={N_CALLS}) =====\n")
            try:
                fitter = MazeBayesianModelFitting(env, rat_data, model, parameters_space,
                                                   fitting_config.OPTIMIZATION_METHOD, N_CALLS)
                parameters, experiment_stats, rat_data_with_likelihood = fitter.optimize()
                num_params = len(fitting_utils.flatten_list(parameters))
                aic, bic, likelihood_stage, meanL, meanNLL = fitting_utils.analyze_fitting(
                    rat_data_with_likelihood, 'likelihood', num_params)
                row = {
                    'subject': subject_id, 'file': rat_file, 'model': friendly_name,
                    'n': len(rat_data_with_likelihood),
                    'parameters': fitting_utils.recursive_round(parameters),
                    'AIC': round(aic, 4), 'BIC': round(bic, 4),
                    'meanL': round(meanL, 4), 'meanNLL': round(meanNLL, 4),
                }
            except Exception as e:
                row = {
                    'subject': subject_id, 'file': rat_file, 'model': friendly_name,
                    'n': None, 'parameters': None, 'AIC': None, 'BIC': None,
                    'meanL': None, 'meanNLL': f'ERROR: {e}',
                }
                print(f"!!! ERROR fitting {friendly_name} on {rat_file}: {e}")
            append_row(row)
            print(f">>> DONE {friendly_name} subject {subject_id}: {row}")

    print("\n\nALL FITS COMPLETE\n")
