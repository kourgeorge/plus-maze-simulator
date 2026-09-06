__author__ = 'gkour'

import os
import warnings

import numpy as np
import pandas as pd
import pymc3 as pm
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


class MazePymc3Fitting:


    LvsNLL = []

    def __init__(self, env, experiment_data, model, parameters_space, n_steps):
        self.env = env
        self.experiment_data = experiment_data
        self.model = model
        self.parameters_space = parameters_space
        self.n_steps = n_steps

    def log_probability(self, parameters):
        lp = 0  # Add any priors here if needed
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._calc_experiment_likelihood(parameters)

    def _calc_experiment_likelihood(self, parameters):
        (brain, learner, model) = self.model
        if model == FixedACFTable:
            att_o, att_c = parameters[2:]
            if att_o + att_c > 1:
                return -np.inf  # Returning -inf makes MCMC reject this parameter set

        experiment_stats, rat_data_with_likelihood = fitting_utils.run_model_on_animal_data(self.env,
                                                                                            self.experiment_data,
                                                                                            self.model, parameters,
                                                                                            silent=True)
        aic, likelihood_stage, meanL, meanNLL = fitting_utils.analyze_fitting(rat_data_with_likelihood, 'likelihood',
                                                                              len(parameters))

        print("x={}, AIC:{:2f} (meanNLL={:.2f}, stages={}), \t(meanL={:.2f}, stages={})".format(
            list(np.round(parameters, 4)),
            aic,
            meanNLL,
            np.round(likelihood_stage.NLL.to_numpy(), 2),
            meanL,
            np.round(likelihood_stage.likelihood.to_numpy(), 2)))

        y = -meanNLL  # MCMC maximizes log-probability, so we minimize NLL by negating it

        MazePymc3Fitting.LvsNLL += [[meanL, meanNLL]]

        return np.clip(y, a_min=-5000, a_max=5000)

    def optimize(self):
        with pm.Model() as model:
            # Define priors for each parameter in the parameters_space
            # priors = [pm.Uniform(f'param_{i}', lower=low, upper=high)
            #           for i, (low, high) in enumerate([bound.bounds for bound in self.parameters_space])]

            # # Define a custom likelihood using Potential
            likelihood = pm.Potential('likelihood', self._calc_experiment_likelihood(priors))

            # Sample from the posterior using MCMC
            trace = pm.sample(self.n_steps, cores=1, tune=500, return_inferencedata=False)

        # Step 1: Extract parameter values from the trace for each parameter
        parameter_values = [trace[f'param_{i}'] for i in range(len(self.parameters_space))]

        # Step 2: Combine the parameter arrays into tuples
        parameter_tuples = list(zip(*parameter_values))

        # Step 3: Compute the log probabilities for each tuple of parameters
        log_probabilities = [self.log_probability(p) for p in parameter_tuples]

        # Step 4: Find the index of the maximum log probability
        best_index = np.argmax(log_probabilities)

        # Step 5: Extract the best set of parameters
        best_parameters = parameter_tuples[best_index]

        experiment_stats, rat_data_with_likelihood = fitting_utils.run_model_on_animal_data(self.env,
                                                                                            self.experiment_data,
                                                                                            self.model, best_parameters)
        aic = -2 * np.sum(np.log(rat_data_with_likelihood.likelihood)) + 2 * len(best_parameters)
        print("Best Parameters: {} - AIC:{:.3}\n".format(np.round(best_parameters, 4), aic))

        return best_parameters, experiment_stats, rat_data_with_likelihood

    @staticmethod
    def all_subjects_all_models_optimization(env, animals_data_folder, all_models, n_steps=1000):
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
                    MazePymc3Fitting(env, curr_rat, model, parameters_space, n_steps).optimize()

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
    MazePymc3Fitting.all_subjects_all_models_optimization(
        PlusMazeOneHotCues2ActiveDoors(stimuli_encoding=10),
        fitting_config.MAZE_ANIMAL_DATA_PATH, fitting_config.maze_models, n_steps=fitting_config.FITTING_ITERATIONS)
