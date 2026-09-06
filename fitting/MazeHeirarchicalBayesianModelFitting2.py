__author__ = 'gkour'

import os
import warnings

import numpy as np
import pandas as pd
import pymc3 as pm
import fitting.fitting_config_attention as fitting_config

from skopt import gp_minimize

import utils
from environment import PlusMazeOneHotCues2ActiveDoors, CueType, PlusMazeOneHotCues, StagesTransition
from fitting import fitting_utils
from learners.networklearners import DQNAtt
from learners.tabularlearners import MALearner
from models.tabularmodels import FixedACFTable

warnings.filterwarnings("ignore")


class MazeBayesianOptimizationFitting:
    def __init__(self, env, experiment_data, model, parameters_space, n_iter):
        self.env = env
        self.experiment_data = experiment_data
        self.model = model
        self.parameters_space = parameters_space
        self.n_iter = n_iter
        self.n_subjects = len(experiment_data)

    def _build_hierarchical_model(self, mu_values, sigma_values):
        mu_values = np.array(mu_values)  # Ensure mu_values is a numpy array
        sigma_values = np.array(sigma_values)  # Ensure sigma_values is a numpy array

        with pm.Model() as hierarchical_model:
            # Group-level hyperparameters
            mu_param = pm.Normal('mu_param', mu=mu_values, sigma=10, shape=len(self.parameters_space))
            sigma_param = pm.HalfNormal('sigma_param', sigma=sigma_values, shape=len(self.parameters_space))

            # Individual parameters for each subject
            individual_params = pm.Normal('individual_params', mu=mu_param, sigma=sigma_param,
                                          shape=(self.n_subjects, len(self.parameters_space)))

            # Likelihood
            likelihoods = []
            for i in range(self.n_subjects):
                likelihood = self._calc_experiment_likelihood(individual_params[i])
                likelihoods.append(likelihood)

            total_likelihood = pm.Potential('total_likelihood', sum(likelihoods))

        return hierarchical_model

    def _calc_experiment_likelihood(self, parameters):
        # Replace with your likelihood calculation logic
        experiment_stats, rat_data_with_likelihood = fitting_utils.run_model_on_animal_data(
            self.env, self.experiment_data, self.model, parameters, silent=True)
        meanNLL = fitting_utils.analyze_fitting(rat_data_with_likelihood, 'likelihood', len(parameters))[3]
        return -meanNLL  # Return the negative log likelihood

    def optimize(self):
        # Bayesian optimization for group-level hyperparameters
        def objective(params):
            mu_values = params[:len(self.parameters_space)]
            sigma_values = params[len(self.parameters_space):]

            with self._build_hierarchical_model(mu_values, sigma_values) as model:
                trace = pm.sample(self.n_iter, chains=1, return_inferencedata=False)
                mean_likelihood = np.mean([self._calc_experiment_likelihood(p) for p in trace['individual_params']])
                return -mean_likelihood  # Minimize the negative log-likelihood

        # Define bounds for Bayesian optimization
        bounds = [(0, 10)] * len(self.parameters_space) + [(0.1, 5)] * len(self.parameters_space)

        # Run Bayesian optimization
        result = gp_minimize(objective, bounds, n_calls=self.n_iter)

        best_mu_params = result.x[:len(self.parameters_space)]
        best_sigma_params = result.x[len(self.parameters_space):]

        print(f"Best group-level mu parameters: {best_mu_params}")
        print(f"Best group-level sigma parameters: {best_sigma_params}")

        # Return the optimized group-level parameters
        return best_mu_params, best_sigma_params

    @staticmethod
    def all_subjects_all_models_optimization(env, animals_data_folder, all_models, n_walkers=50, n_steps=1000):
        animal_data = [[rat_file, pd.read_csv(os.path.join(animals_data_folder, rat_file))]
                       for rat_file in list(np.sort(os.listdir(animals_data_folder)))]

        timestamp = utils.get_timestamp()
        fitting_results = {}
        results_df = pd.DataFrame()
        for subject_id, (file_name, curr_rat) in enumerate(animal_data):
            initial_motivation = curr_rat.iloc[
                0].initial_motivation if 'initial_motivation' in curr_rat.columns else 'water'

            print("\n#################### Subject: {} - {}. Env: {} #####################\n".format(subject_id,
                                                                                                    initial_motivation,
                                                                                                    str(env)))
            curr_rat = fitting_utils.maze_experimental_data_preprocessing(curr_rat)
            fitting_results[subject_id] = {}
            for curr_model in all_models:
                model, parameters_space = curr_model
                print("-----{}-----".format(utils.brain_name(model)))
                search_result, experiment_stats, rat_data_with_likelihood = \
                    MazeBayesianOptimizationFitting(env, curr_rat, model, parameters_space, n_steps).optimize()

                rat_data_with_likelihood["subject"] = subject_id
                rat_data_with_likelihood["model"] = utils.brain_name(model)
                rat_data_with_likelihood["parameters"] = [np.round(search_result, 4)] * len(rat_data_with_likelihood)
                rat_data_with_likelihood["algorithm"] = \
                    "MCMC_{}".format(n_steps)

                results_df = results_df.append(rat_data_with_likelihood, ignore_index=True)
            results_df.to_csv(f'fitting/Results/Rats-Results/fitting_results_dimensional{timestamp}_{n_steps}_tmp.csv')
        results_df.to_csv(f'fitting/Results/Rats-Results/fitting_results_dimensional{timestamp}_{n_steps}.csv')
        return fitting_results


if __name__ == '__main__':
    # fit odor first animals
    MazeBayesianOptimizationFitting.all_subjects_all_models_optimization(
        PlusMazeOneHotCues2ActiveDoors(stimuli_encoding=10),
        fitting_config.MAZE_ANIMAL_DATA_PATH, fitting_config.maze_models, n_walkers=50,
        n_steps=fitting_config.FITTING_ITERATIONS)
