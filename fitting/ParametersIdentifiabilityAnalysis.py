from random import random

import numpy as np
from matplotlib import pyplot as plt

from environment import PlusMazeOneHotCues2ActiveDoors, StagesTransition, CueType
from fitting import fitting_utils
from fitting.MazeBayesianModelFitting import MazeBayesianModelFitting
from fitting.fitting_config_attention import beta, lr, attention_lr
from learners.tabularlearners import QLearner, IALearner, MALearner
from models.tabularmodels import QTable, OptionsTable, ACFTable, FixedACFTable
from rewardtype import RewardType
from simulation import run_simulation_sampled_brain, run_single_simulation
import pandas as pd
from brains.tdbrain import TDBrain
import fitting_config_attention
import utils
from scipy.stats import pearsonr
import ast


# for each type of brain in a given list. (friendly name)
# create an agent with the suitable brain.
# sample parameters in the allowed range
# run a simulation with this agent on the suitable environment
# write down the simulation output in a file
# estimate the parameters ysung the Bayesian OPS
# for each execution, write in a cv the model name, the repetition, the set of true parameters and the set of estimated param,eters.
# do the analysis.


def run_simulation_on_models(rats_fitting_data):
    env = PlusMazeOneHotCues2ActiveDoors(relevant_cue=CueType.ODOR, stimuli_encoding=8)

    all_simulation_data = run_simulation_sampled_brain(env=env, brain_specs=[
        ((TDBrain, QLearner, QTable),
         [5, 0.1], [1, 0.05],
         ([0.1, 10], [0.001, 0.4])),
        ((TDBrain, QLearner, OptionsTable),
         [5, 0.1], [1, 0.05],
         ([0.1, 10], [0.001, 0.4])),
        ((TDBrain, IALearner, ACFTable),
         [5, 0.1], [1, 0.05],
         ([0.1, 10], [0.001, 0.4])),
        ((TDBrain, IALearner, FixedACFTable),
         [5, 0.1], [1, 0.05],
         ([0.1, 10], [0.001, 0.4])),
        ((TDBrain, MALearner, ACFTable),
         [5, 0.1, 0.1], [1, 0.01, 0.01],
         ([0.1, 10], [0.001, 0.4], [0.001, 0.4]))],
                                                       repetitions=2)

    all_simulation_data.to_csv('test_data.csv')
    return all_simulation_data


def estimate_parameters(experiments_df):
    fitting_iterations = 50
    optimization_method = 'Hybrid'

    df = experiments_df.copy()
    df['agent'] = df['subject']
    env = PlusMazeOneHotCues2ActiveDoors(relevant_cue=CueType.ODOR, stimuli_encoding=8)
    animals = np.unique(experiments_df.subject)
    models = np.unique(experiments_df.model)

    # df = experiments_df[experiments_df['subject'].isin(range(5))]
    # # Combine 'subject' and 'model' into a new column named 'combined'
    # # Assign a unique number to each unique combination in 'combined'
    # df['combined'] = df['subject'].astype(str) + '_' + experiments_df['model'].astype(str)
    # df['agent'] = df.groupby('combined').ngroup() + 1  # +1 to start numbering from 1

    results_df = pd.DataFrame()
    for agent in df.agent.unique():
        df_sub = df[df.agent == agent]
        model_string = df_sub["model"].iloc[0]
        model, parameters_space = fitting_config_attention.map_maze_models[model_string]
        true_parameters = np.round(fitting_utils.string2list(df_sub['parameters'].iloc[0]),4)


        _, true_parameters_df = fitting_utils.run_model_on_animal_data(env, df_sub, model, true_parameters)


        aic_t, likelihood_stage_t, meanL_t, meanNLL_t = fitting_utils.analyze_fitting(true_parameters_df, 'likelihood',
                                                                                      len(true_parameters))

        print(f"---------- Agent: {agent} ----- Original Model: {utils.brain_name(model)} "
              f"---- Original parameters: {true_parameters} - AIC:{aic_t}, meanL: {meanL_t}, meanNLL: {meanNLL_t}"
              f"---- fitted Model: {utils.brain_name(model)}")


        search_result, experiment_stats, rat_data_with_likelihood = \
            MazeBayesianModelFitting(env, df_sub, model, parameters_space, fitting_iterations).optimize()

        rat_data_with_likelihood["subject"] = agent
        rat_data_with_likelihood["fitted_model"] = utils.brain_name(model)
        rat_data_with_likelihood["fitted_parameters"] = [np.round(search_result.x, 4)] * len(rat_data_with_likelihood)
        rat_data_with_likelihood["algorithm"] = f"{optimization_method}_{fitting_iterations}"


        results_df = results_df.append(rat_data_with_likelihood, ignore_index=True)

        print(f"Original Parameters: {df_sub['parameters'].iloc[0]} Recovered Parameters:{rat_data_with_likelihood['fitted_parameters'].iloc[0]}")

    results_df.to_csv(f'fitting/Results/Rats-Results/identifiability_results/recoverability_{optimization_method}_{fitting_iterations}_{utils.get_timestamp()}_.csv')
    return results_df


def compare_estimated_true_parameters_likelihoods(recovered_df):
    env = PlusMazeOneHotCues2ActiveDoors(relevant_cue=CueType.ODOR, stimuli_encoding=10)

    # run the environment against the true paramaters to get each trial likelihood.
    results_with_true_parameters_likelihood_df = pd.DataFrame()
    for agent in recovered_df.subject.unique():
        df = recovered_df[recovered_df.subject == agent]

        # in the fitted df we have the true parameters ('parameters') the fitted parameters ('fitted_parameters') and the 'fitted likelihood'
        true_parameters = fitting_utils.string2list(df.parameters.iloc[0])
        fitted_parameters = fitting_utils.string2list(df.fitted_parameters.iloc[0])

        df.rename(columns={'likelihood': 'fitted_parameters_likelihood',
                           'parameters': 'true_parameters'}, inplace=True)
        model, _ = fitting_config_attention.map_maze_models[df.model.iloc[0]]

        # here we add the likelihood of the true parameters
        experiment_stats, rat_data_with_likelihood = fitting_utils.run_model_on_animal_data(env, df, model, true_parameters)
        rat_data_with_likelihood.rename(columns={'likelihood': 'true_parameters_likelihood'},inplace=True)

        results_with_true_parameters_likelihood_df = results_with_true_parameters_likelihood_df.append(rat_data_with_likelihood, ignore_index=True)

        aic_t, likelihood_stage_t, meanL_t, meanNLL_t = fitting_utils.analyze_fitting(rat_data_with_likelihood, 'true_parameters_likelihood',
                                                                                      len(true_parameters))

        aic_f, likelihood_stage_f, meanL_f, meanNLL_f = fitting_utils.analyze_fitting(rat_data_with_likelihood, 'fitted_parameters_likelihood',
                                                                                      len(true_parameters))

        print(f'{utils.brain_name(model)} (True vs. Fitted): \nparam: {np.round(true_parameters,4)}-{fitted_parameters} \n AIC: {aic_t}-{aic_f}   meanLikelihood: {meanL_t}-{meanL_f}  meanNLL: {meanNLL_t}-{meanNLL_f}')


    return results_with_true_parameters_likelihood_df


def recoverability_correlation(recovered_df):
    df = recovered_df[['subject', 'model', 'parameters', 'fitted_parameters']]
    df = df.drop_duplicates()
    df.parameters = df.parameters.apply(lambda x: fitting_utils.string2list(x))
    df.fitted_parameters = df.fitted_parameters.apply(lambda x: fitting_utils.string2list(x))

    for model in df.model.unique():
        df_model = df[df.model==model]

        true_params = np.vstack(df_model['parameters'].tolist())
        estimated_params = np.vstack(df_model['fitted_parameters'].tolist())

        overall_corr, _ = pearsonr(true_params.flatten(), estimated_params.flatten())

        print(f"Model: {model} correlation: {overall_corr}")

        _, model_parameters = fitting_config_attention.map_maze_models[model]
        num_params = true_params.shape[1]

        parameter_names = [parameter.name for parameter in model_parameters]
        # Plotting
        fig, axes = plt.subplots(1, num_params, figsize=(15, 5))
        for i in range(num_params):
            ax = axes[i]
            ax.scatter(true_params[:, i], estimated_params[:, i], alpha=0.7)

            x = true_params[:, i]
            y = estimated_params[:, i]

            # Fit line
            coef = np.polyfit(x, y, 1)  # Linear fit
            fit_line = np.poly1d(coef)
            ax.plot(x, fit_line(x), 'r--', lw=2, label='Correlation Line')

            # Set axis limits
            min_val = min(np.min(x), np.min(y))
            max_val = max(np.max(x), np.max(y))
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)

            ax.set_xlabel(f'True Param')
            ax.set_ylabel(f'Estimated Param')
            overall_corr, _ = pearsonr(x, y)
            ax.set_title(f'{parameter_names[i]} (Pears. Corr.: {np.round(overall_corr,2)})')

        fig.suptitle(f'{fitting_config_attention.friendly_models_name_map[model]}', fontsize=16)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # all_simulation_data = run_simulation_on_models()

    all_simulation_data = pd.read_csv('fitting/Results/Rats-Results/identifiability_results/identifiability_simulation_days_25_100TPD.csv')
    estimate_parameters(all_simulation_data)

    recovered_df = pd.read_csv('fitting/Results/Rats-Results/identifiability_results/recoverability_Hybrid_50_2024_07_27_23_02_.csv')

    # recoverability_correlation(recovered_df)
    compare_estimated_true_parameters_likelihoods(recovered_df)