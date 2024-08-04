__author__ = 'gkour'

import numpy as np
from matplotlib import pyplot as plt

from environment import PlusMazeOneHotCues2ActiveDoors, StagesTransition, CueType
from fitting import fitting_utils
from fitting.MazeBayesianModelFitting import MazeBayesianModelFitting
import pandas as pd
import fitting_config_attention
import utils
from scipy.stats import pearsonr
import ast
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'font.size': 14})


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
        true_model, parameters_space = fitting_config_attention.map_maze_models[model_string]
        true_parameters = np.round(fitting_utils.string2list(df_sub['parameters'].iloc[0]),4)
        _, true_parameters_df = fitting_utils.run_model_on_animal_data(env, df_sub, true_model, true_parameters)

        aic_t, likelihood_stage_t, meanL_t, meanNLL_t = fitting_utils.analyze_fitting(true_parameters_df, 'likelihood',
                                                                                      len(true_parameters))

        candidate_models = fitting_config_attention.maze_models
        for candidate_model_arch in candidate_models:
            candidate_model, candidate_parameters_space = candidate_model_arch

            print(f"---------- Agent: {agent} ----- Original Model: {utils.brain_name(true_model)} "
                  f"---- Original parameters: {true_parameters} - AIC:{aic_t}, meanL: {meanL_t}, meanNLL: {meanNLL_t}"
                  f"---- fitted Model: {utils.brain_name(candidate_model)}")

            search_result, experiment_stats, rat_data_with_likelihood = \
                MazeBayesianModelFitting(env, df_sub, candidate_model, candidate_parameters_space, fitting_iterations).optimize()

            rat_data_with_likelihood["subject"] = agent
            rat_data_with_likelihood["fitted_model"] = utils.brain_name(candidate_model)
            rat_data_with_likelihood["fitted_parameters"] = [np.round(search_result.x, 4)] * len(rat_data_with_likelihood)
            rat_data_with_likelihood["algorithm"] = f"{optimization_method}_{fitting_iterations}"

            results_df = results_df.append(rat_data_with_likelihood, ignore_index=True)

            print(f"Original Parameters: {df_sub['parameters'].iloc[0]} Recovered Parameters:{rat_data_with_likelihood['fitted_parameters'].iloc[0]}")

    results_df.to_csv(f'fitting/Results/Rats-Results/identifiability_results/recoverability_{optimization_method}_{fitting_iterations}_{utils.get_timestamp()}_.csv', index_label=False)

    return results_df


def analyze_models_fitting(recovered_df):
    env = PlusMazeOneHotCues2ActiveDoors(relevant_cue=CueType.ODOR, stimuli_encoding=10)

    # run the environment against the true parameters to the trials likelihood.
    model_identifiability_results_df = pd.DataFrame()
    results_with_true_parameters_likelihood_df = pd.DataFrame()
    for agent in recovered_df.subject.unique():
        agent_df = recovered_df[recovered_df.subject == agent]

        # in the fitted agent_df we have the true parameters ('parameters') the fitted parameters ('fitted_parameters') and the 'fitted likelihood'
        true_parameters = fitting_utils.string2list(agent_df.parameters.iloc[0])

        agent_df.rename(columns={'likelihood': 'fitted_parameters_likelihood', 'parameters': 'true_parameters'}, inplace=True)
        true_model_name = agent_df.model.iloc[0]
        true_model, _ = fitting_config_attention.map_maze_models[true_model_name]

        # here we add the likelihood of the true parameters
        df_true = agent_df[agent_df.fitted_model == agent_df.fitted_model.iloc[0]]
        experiment_stats, rat_data_with_likelihood = fitting_utils.run_model_on_animal_data(env, df_true, true_model, true_parameters)
        rat_data_with_likelihood.rename(columns={'likelihood': 'true_parameters_likelihood'}, inplace=True)

        results_with_true_parameters_likelihood_df = results_with_true_parameters_likelihood_df.append(rat_data_with_likelihood, ignore_index=True)

        aic_t, likelihood_stage_t, meanL_t, meanNLL_t = fitting_utils.analyze_fitting(rat_data_with_likelihood, 'true_parameters_likelihood',
                                                                                      len(true_parameters))

        for fitted_model_name in agent_df.fitted_model.unique():

            df_fitted = agent_df[agent_df.fitted_model == fitted_model_name]
            fitted_parameters = fitting_utils.string2list(df_fitted.fitted_parameters.iloc[0])
            aic_f, likelihood_stage_f, meanL_f, meanNLL_f = fitting_utils.analyze_fitting(df_fitted, 'fitted_parameters_likelihood',
                                                                                      len(fitted_parameters))

            results_dict = {
                'agent': agent,
                'true_model': true_model_name,
                'true_model_parameters': true_parameters,
                'true_likelihood': {
                    'aic': aic_t,
                    'meanNLL': meanNLL_t
                },
                'fitted_model': fitted_model_name,
                'fitted_model_parameters': fitted_parameters,
                'fitted_likelihood': {
                    'aic': aic_f,
                    'meanNLL': meanNLL_f
                },
            }
            model_identifiability_results_df = model_identifiability_results_df.append(results_dict, ignore_index=True)
            print(f'{true_model_name}-{fitted_model_name} (True vs. Fitted): \nparam: {np.round(true_parameters,4)}-{fitted_parameters} \n AIC: {aic_t}-{aic_f}   meanLikelihood: {meanL_t}-{meanL_f}  meanNLL: {meanNLL_t}-{meanNLL_f}')

    model_identifiability_results_df.to_csv(f"fitting/Results/Rats-Results/identifiability_results/_results_{utils.get_timestamp()}_.csv")
    return model_identifiability_results_df


def parameters_recoverability_correlation(recovered_df):
    # df = recovered_df[['subject', 'model', 'parameters', 'fitted_model', 'fitted_parameters']]
    df = recovered_df[['agent', 'true_model', 'true_model_parameters', 'fitted_model', 'fitted_model_parameters']]

    df = df.drop_duplicates()
    df.true_model_parameters = df.true_model_parameters.apply(lambda x: fitting_utils.string2list(x))
    df.fitted_model_parameters = df.fitted_model_parameters.apply(lambda x: fitting_utils.string2list(x))

    for model in df.true_model.unique():
        df_model = df[(df.true_model == model) & (df.fitted_model == model)]

        true_params = np.vstack(df_model.true_model_parameters)
        estimated_params = np.vstack(df_model.fitted_model_parameters)

        _, model_parameters = fitting_config_attention.map_maze_models[model]
        model_short_name = fitting_config_attention.friendly_models_name_map[model]
        num_params = true_params.shape[1]

        parameter_names = [parameter.name for parameter in model_parameters]
        parameter_bounds = [parameter.bounds for parameter in model_parameters]
        # Plotting
        fig, axes = plt.subplots(1, num_params, figsize=(4*num_params, 4))
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
            rang=(max_val-min_val)*0.1
            ax.set_xlim(min_val-rang, max_val+rang)
            ax.set_ylim(min_val-rang, max_val+rang)

            # Set axis to logarithmic scale
            # ax.set_xscale('log')
            # ax.set_yscale('log')

            ax.set_xlabel(f'Fit {fitting_utils.parameters_friendly_names[parameter_names[i]]}')
            ax.set_ylabel(f'Simulated {fitting_utils.parameters_friendly_names[parameter_names[i]]}')
            overall_corr, _ = pearsonr(x, y)
            rho = np.round(overall_corr,2)
            ax.set_title(fr'$\rho$={rho}')
            fitting_utils.despine(ax)
        fig.suptitle(f'{model_short_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"fitting/Results/figures/recoverability_{model_short_name}_{utils.get_timestamp()}.pdf")
        plt.show()


def model_identifiability_confusion_matrix(df):

    # Function to extract AIC from likelihood string
    def extract_aic(likelihood_str):
        likelihood_dict = ast.literal_eval(likelihood_str)
        return likelihood_dict['aic']

    # Extract AIC values for true and fitted models
    df['true_aic'] = df['true_likelihood'].apply(extract_aic)
    df['fitted_aic'] = df['fitted_likelihood'].apply(extract_aic)

    df['true_model'] = df['true_model'].apply(lambda x: fitting_utils.friendly_models_name_map[x])
    df['fitted_model'] = df['fitted_model'].apply(lambda x: fitting_utils.friendly_models_name_map[x])

    # Extract AIC and determine the identified model with the smallest AIC for each true model instance
    df['identified_model'] = df.apply(
        lambda row: row['fitted_model'] if row['fitted_aic'] < row['true_aic'] else row['true_model'], axis=1)

    # Group by agent and select the row with the minimum true_aic for each agent
    unique_agents_df = df.loc[df.groupby('agent')['true_aic'].idxmin()]


    # Extract the true and identified models
    true_models = unique_agents_df['true_model']
    identified_models = unique_agents_df['identified_model']

    # Get the unique model names again to include identified models
    unique_models = np.unique(np.concatenate((true_models, identified_models)))

    # Create a mapping from model name to index
    model_to_index = {model: idx for idx, model in enumerate(unique_models)}

    # Map the true and identified models to their indices
    true_indices = true_models.map(model_to_index)
    identified_indices = identified_models.map(model_to_index)

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(true_indices, identified_indices, labels=list(model_to_index.values()))

    # Convert the confusion matrix to a DataFrame for better readability
    conf_matrix_df = pd.DataFrame(conf_matrix, index=unique_models, columns=unique_models)

    # Display the confusion matrix
    print(conf_matrix_df)

    # Normalize the confusion matrix by row (i.e., by the number of true instances)
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Convert the normalized confusion matrix to a DataFrame for better readability
    conf_matrix_normalized_df = pd.DataFrame(conf_matrix_normalized, index=unique_models, columns=unique_models)

    # Plot the heatmap
    ax = plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix_normalized_df, annot=True, cmap='Blues', fmt='.2f',cbar=False)
    # plt.title('Model Identifiability Confusion Matrix')
    plt.ylabel('Simulated Model')
    plt.xlabel('Fit Model')
    plt.tight_layout()
    plt.savefig(f"fitting/Results/figures/identifiability_confusion_matrix_{utils.get_timestamp()}.pdf")

    x=1


if __name__ == '__main__':
    # all_simulation_data = pd.read_csv('fitting/Results/Rats-Results/identifiability_results/simulation_25_100TPD_loguniform_large_range.csv')
    # estimate_parameters(all_simulation_data)

    # analyze_models_fitting(recovered_df)

    results_df = pd.read_csv('fitting/Results/Rats-Results/identifiability_results/original_range_loguniform/results_2024_08_04_00_03_original_range_loguniform.csv')
    parameters_recoverability_correlation(results_df)
    model_identifiability_confusion_matrix(results_df)
