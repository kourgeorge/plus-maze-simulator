__author__ = 'gkour'

import functools
import json
import os
import sys
import numpy as np
import pandas as pd
import scipy
from scipy import stats

import config
from fitting.PlusMazeExperimentFitting import PlusMazeExperimentFitting
from fitting.fitting_config_attention import friendly_models_name_map, get_parameter_names
from learners.abstractlearner import SymmetricLearner
from learners.networklearners import DQNAtt
from learners.tabularlearners import MALearner
from models.non_directional_tabularmodels import NonDirectionalFixedACFTable
from models.tabularmodels import FixedACFTable
from motivatedagent import MotivatedAgent
from environment import PlusMazeOneHotCues
from rewardtype import RewardType
import re

import pingouin
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM


def episode_rollout_on_real_data(env: PlusMazeOneHotCues, agent: MotivatedAgent, current_trial):
    total_reward = 0
    num_actions = env.num_actions()
    act_dist = np.zeros(num_actions)

    env_state = env.reset_trial()
    terminated = False
    steps = 0
    likelihood = 0
    while not terminated:
        # print("trial: {}".format(current_trial['trial']))
        # print("rewarded in real data: {}, from type: {}".format(current_trial['reward'], current_trial['reward_type']))
        steps += 1
        state = env.set_state(current_trial)
        action = int(current_trial['action']) - 1
        dec_1hot = np.zeros(num_actions)
        dec_1hot[action] = 1
        act_dist += dec_1hot
        new_state, outcome, terminated, correct_door, info = env.step(action)
        reward = agent.evaluate_outcome(outcome)

        # This validates that if the reward in the data is 0 the reward by the simulation should be RewardType.NONE.
        # and vice versa, meaning if the reward in the data is not 0 then the reward type in the simulation must not be none.
        if (outcome == RewardType.NONE and current_trial.reward != 0) or \
                (outcome != RewardType.NONE and current_trial.reward == 0):
            raise Exception(f"There is a discrepancy between data and simulation reward!!\n"
                            f"trial={current_trial.trial}, stage={current_trial.stage}, data_action={action + 1}, "
                            f"env_correct_door={correct_door + 1}, data_reward={current_trial.reward}, env_reward={reward}")

        total_reward += reward
        model_action_dist = agent.get_brain().think(np.expand_dims(state, 0), agent).squeeze().detach().numpy()
        likelihood += model_action_dist[action]

        agent.add_experience(state, dec_1hot, reward, outcome, new_state, terminated, info)

        env.set_state(current_trial)
        info.likelihood = likelihood
        info.model_action = agent.decide_greedy(state)
        _, model_action_outcome, _, _, _ = env.step(info.model_action)
        info.network_outcome = model_action_outcome

        state = new_state
    return steps, total_reward, act_dist, model_action_dist, info.model_action + 1, correct_door, likelihood, model_action_outcome


def run_model_on_animal_data(env, rat_data, model_arch, parameters, initial_motivation=RewardType.NONE, silent=True):
    if initial_motivation is None:
        initial_motivation = RewardType(rat_data.iloc[0].initial_motivation)
    (brain, learner, model) = model_arch

    resolved_params_dict = resolve_parameters(parameters, *model_arch)

    model_instance = model(encoding_size=env.stimuli_encoding_size(), num_actions=env.num_actions(), num_channels=2,
                           **resolved_params_dict)
    learner_instance = learner(model_instance, **resolved_params_dict)
    brain_instance = brain(learner_instance, **resolved_params_dict)

    nmr = 0
    agent = MotivatedAgent(brain_instance, motivation=initial_motivation,
                           motivated_reward_value=config.MOTIVATED_REWARD,
                           non_motivated_reward_value=nmr, exploration_param=0)

    if silent: blockPrint()

    env.init()

    experiment_stats, rat_data_with_likelihood = PlusMazeExperimentFitting(env, agent, dashboard=False,
                                                                           experiment_data=rat_data)
    if silent: enablePrint()

    return experiment_stats, rat_data_with_likelihood


def resolve_parameters(parameters, brain, learner, model):
    """
    Resolve and assign parameters based on the learner type and model.
    Returns:
        dict: A dictionary containing resolved parameter names and their values.
    """
    # Create an iterator for the parameters
    param_iterator = iter(parameters)

    # Initialize resolved parameters with mandatory 'beta' and 'lr' values
    resolved_params = {
        'beta': next(param_iterator),
        'lr': next(param_iterator)
    }

    # Check if the learner is not a subclass of SymmetricLearner and add 'lr_nr'
    if not issubclass(learner, SymmetricLearner):
        resolved_params['lr_nr'] = next(param_iterator)

    # Check if the learner requires 'attention_lr' parameter
    if issubclass(learner, (MALearner, DQNAtt)):
        resolved_params['attention_lr'] = next(param_iterator)

    # Check if the model is FixedACFTable and add attention parameters
    if model == FixedACFTable:
        attn_odor = next(param_iterator)
        attn_color = next(param_iterator)
        resolved_params['attn_importance'] = [attn_odor, attn_color, 1-attn_odor-attn_color]

    # Check if the model is FixedACFTable and add attention parameters
    if model == NonDirectionalFixedACFTable:
        attn_odor = next(param_iterator)
        resolved_params['attn_importance'] = [attn_odor, 1-attn_odor]

    return resolved_params


def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def get_stage_transition_days(experimental_data):
    return np.where(experimental_data['day in stage'] == 1)[0][1:]


def string2list(string):
    try:
        params = [float(x.strip()) for x in re.split(" +", string.strip(' ][()'))]
    except Exception:
        params = [float(x.strip()) for x in re.split(",", string.strip('][()'))]
    return params


def stable_unique(array):
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]


def maze_experimental_data_preprocessing(experiment_data):
    # remove trials with non-active doors selection:
    experiment_data['completed'] = experiment_data.apply(
        lambda x: False if np.isnan(x.action) or np.isnan(x.reward) else x["A{}o".format(int(x.action))] != -1,
        axis='columns')
    experiment_data_filtered = experiment_data[experiment_data.completed == True]
    experiment_data_filtered.drop('completed', axis='columns', inplace=True)

    df_sum = experiment_data.groupby(['stage', 'day in stage'], sort=False).agg(
        {'reward': 'mean', 'action': 'count'}).reset_index()

    # Take at most 7 days from the last stage.
    df = experiment_data_filtered.copy()
    df = df[~((df.stage == 3) & (df['day in stage'] > 10))]

    # criteria_days = []
    # for st in [1,2,3]:
    # 	stage_data = df_sum[df_sum.stage==st]
    # 	first_criterion_day = np.where(np.array(stage_data.reward) >= .74)
    #
    # 	criterion_day =len(stage_data.reward) if len(first_criterion_day[0]) == 0 else first_criterion_day[0][0]+1
    # 	criteria_days += [criterion_day]
    #
    # df = experiment_data.copy()
    # df = df[~((df.stage == 1) & (df['day in stage'] > criteria_days[0]))]
    # df = df[~((df.stage == 2) & (df['day in stage'] > criteria_days[1]))]
    # df = df[~((df.stage == 3) & (df['day in stage'] > np.min ([criteria_days[2],7])))]

    print("Processing behavioral data: Original:{}, removed:{}".format(len(experiment_data),
                                                                       len(experiment_data) - len(df)))

    return df


def analyze_fitting(rat_data_with_likelihood, likelihood_column_name, num_parameters):
    likelihood_column = rat_data_with_likelihood[likelihood_column_name]
    rat_data_with_likelihood['NLL'] = -np.log(likelihood_column)
    likelihood_day = rat_data_with_likelihood.groupby(['stage', 'day in stage']).mean().reset_index()
    likelihood_stage = likelihood_day.groupby('stage').mean()
    NLL = rat_data_with_likelihood.NLL.to_numpy()
    n = len(NLL)
    L = likelihood_column.to_numpy()
    meanNLL = np.nanmean(NLL)
    meanL = np.nanmean(likelihood_column)
    geomeanL = scipy.stats.mstats.gmean(likelihood_column)
    np.testing.assert_almost_equal(np.exp(-meanNLL), geomeanL)
    aic = 2 * np.sum(NLL) + 2 * num_parameters
    return round(aic, 4), round(likelihood_stage, 4), round(meanL, 4), round(meanNLL, 4)


def models_order_df(df):
    models_in_df = np.unique(df.model)
    return stable_unique([model for model in friendly_models_name_map.values() if model in models_in_df])


def models_struct_order_df(df):
    models_in_df = np.unique(df.model_struct)
    model_structs = ['m', 'B-m', 'M(B)-m', 'M(V)-m', 'M(VB)-m', 'E(V)-m', 'E(V)-m', 'E(B)-m', 'E(V)-M(B)-m',
                     'E(V)-M(VB)-m']
    return stable_unique([model_struct for model_struct in model_structs if model_struct in models_in_df])


def rename_models(model_df):
    model_df["model"] = model_df.model.map(
        lambda x: friendly_models_name_map[x] if x in friendly_models_name_map.keys() else x)
    return model_df


def cut_off_data_when_reaching_criterion(df, num_stages=3):
    df_res = pd.DataFrame()
    df_sum = df.groupby(['subject', 'stage', 'day in stage'], sort=False).agg(
        {'reward': 'mean', 'trial': 'count'}).reset_index()

    for subject in np.unique(df_sum.subject):
        df_res_subject = pd.DataFrame()
        sub_df = df_sum[df_sum.subject == subject]
        criteria_days = []
        for stage in range(num_stages):
            st = stage + 1
            stage_data = sub_df[sub_df.stage == st]
            first_criterion_day = np.where(np.array(stage_data.reward) >= .75)

            criterion_day = len(stage_data.reward) if len(first_criterion_day[0]) == 0 else first_criterion_day[0][
                                                                                                0] + 1
            relevant_trials = df[(df.subject == subject) & (df.stage == st) & (df['day in stage'] <= criterion_day)]
            df_res_subject = pd.concat([df_res_subject, relevant_trials], axis=0)
            criteria_days += [criterion_day]
        df_res = pd.concat([df_res, df_res_subject], axis=0)
    return df_res


def add_correct_door(df):
    if all(not np.isnan(value) for value in df['correct_door']):
        df['correct_door'] = df['correct_door'].astype(int)
        df['model_reward_dist'] = df.apply(lambda row: string2list(row['model_action_dist'])[row['correct_door']],
                                           axis='columns').astype(float)
        return df
    df['correct_door'] = None
    df['model_reward_dist'] = None

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Check the stage value and set the correct_door accordingly
        if row['stage'] in [1, 2]:
            for door in range(1, 5):
                if row[f'A{door}o'] == 1:
                    df.at[index, 'correct_door'] = door
        elif row['stage'] == 3:
            for door in range(1, 5):
                if row[f'A{door}c'] == 1:
                    df.at[index, 'correct_door'] = door

        correct_door = df.at[index, 'correct_door']
        model_action_dist = string2list(df.at[index, 'model_action_dist'])
        df.at[index, 'model_reward_dist'] = model_action_dist[correct_door - 1]

    df['model_reward_dist'] = df['model_reward_dist'].astype(float)
    return df


def unbox_model_variables(df_model, columns=['model_variables']):
    # format the model_variables entry

    unboxed_variable_names = []
    for column in columns:
        df_model[column] = df_model[column].apply(lambda s: s.replace("\'", "\""))
        df_model[column] = df_model[column].apply(json.loads)

        variables_names = list(df_model[column].tolist()[0].keys())
        unboxed_variable_names += variables_names
        df_variables = pd.DataFrame(df_model[column].tolist())

        df_no = df_model.drop(column, axis=1).reset_index(drop=True)
        df_model = pd.concat([df_no, df_variables], axis=1)

    # df_model = df_model.groupby(['subject', 'model', 'parameters', 'stage', 'day in stage', 'ind'],
    # 							sort=False).mean().reset_index()

    return df_model, unboxed_variable_names


def index_days(df):
    df['ind'] = df.apply(lambda x: str(x.stage) + '.' + str(x['day in stage']), axis='columns')

    def compare(x, y):
        if int(x[0]) < int(y[0]):
            return -1
        elif int(x[0]) > int(y[0]):
            return 1
        elif int(x[2:]) < int(y[2:]):
            return -1
        else:
            return 1

    order = sorted(np.unique(df['ind']), key=functools.cmp_to_key(compare))
    transition = [i for i in range(1, len(order)) if order[i][0] != order[i - 1][0]]

    return df, order, transition


def despine(axis):
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)


def dilute_xticks(axis, k=2):
    ticks = ["{}".format(int(x._text[2:])) if (int(x._text[2:]) - 1) % k == 0 else "" for ind, x in
             enumerate(axis.get_xticklabels())]
    axis.set_xticklabels(ticks)


def extract_names_from_architecture(architecture_name):
    learner, model = architecture_name.split('.')
    return learner, model


def RM_anova(df, depvar, subjects, within):
    """Repeated measures ANOVA.
    depvar: the dependant variable.
    subjects: the grouping variable.
    within: the repetitions variable for each subject. """

    # results = AnovaRM(data=df, depvar=depvar, subject=subjects, within=[within]).fit() #, aggregate_func='mean'
    aov = pingouin.rm_anova(data=df, dv=depvar, subject=subjects, within=within, detailed=True)
    aov.round(3)
    print(aov)


def one_way_anova(df, target, c):
    """target: the dependant variable
        c: the groups variable
    """
    model = ols('{} ~ C({})'.format(target, c), data=df).fit()
    result = sm.stats.anova_lm(model, type=2)
    print(result)


def two_way_anova(df, target, c1, c2):
    model = ols('{} ~ C({}) + C({}) +\
	C({}):C({})'.format(target, c1, c2, c1, c2),
                data=df).fit()
    result = sm.stats.anova_lm(model, type=2)
    print(result)


def mixed_design_anova(data, dependent_var, fixed_factor1, fixed_factor2, random_factor):
    """
    Perform mixed-design ANOVA with fixed effects and mixed random_factor.

    Args:
        data (pd.DataFrame): The dataset containing the relevant columns.
        dependent_var (str): The name of the dependent variable column.
        fixed_factor (str): The name of the fixed effects factor column.
        random_factor (str): The name of the random effects factor column.

    Returns:
        result (statsmodels.regression.mixed_linear_model.MixedLMResults): The ANOVA results.

    Example usage:
        result = mixed_design_anova(df, 'likelihood', 'stage', 'model')
    """

    # Define the model
    formula = f"{dependent_var} ~ {fixed_factor1} * {fixed_factor2}"
    groups = data[random_factor]

    # Fit the mixed-effects model
    model = MixedLM.from_formula(formula, groups=groups, data=data)
    result = model.fit()
    print(result.summary())

    return result


def calculate_fitted_parameters_stats(data):
    # Create a dictionary to store the statistics for each model
    stats_dict = {}

    # Iterate over all unique models
    for model in data['model'].unique():
        # Extract parameters for the current model
        model_data = data[data['model'] == model]
        parameters = np.array(model_data['parameters'].tolist())

        # Calculate mean and std for each parameter
        mean = np.mean(parameters, axis=0)
        std = stats.sem(parameters, axis=0)
        parameter_names = get_parameter_names(model)
        stats_dict[model] = {name: {'mean': m, 'sem': s} for name, m, s in zip(parameter_names, mean, std)}

    return stats_dict


def print_model_parameters(fitted_parameters_stats):
    # Header for the table
    print(f"{'Model':<10} {'Parameter':<15} {'Estimated Value':<20}")
    print("=" * 45)

    # Iterate over each model and its parameters
    for model, parameters in fitted_parameters_stats.items():
        first_param = True
        for param, stats in parameters.items():
            mean = stats['mean']
            std = stats['sem']
            estimated_value = f"{mean:.4g} ± {std:.4g}"
            if first_param:
                print(f"{model:<10} {param:<15} {estimated_value:<20}")
                first_param = False
            else:
                print(f"{'':<10} {param:<15} {estimated_value:<20}")


def sample_attention_distribution(dimensions=3):
    # Sample three numbers from a uniform distribution between 0 and 1
    random_numbers = np.random.rand(dimensions)
    # Normalize the numbers so that their sum equals 1
    normalized = random_numbers / np.sum(random_numbers)
    return tuple(normalized)


parameters_friendly_names = {
    'beta': r'$\beta$',
    'lr': r'$\alpha$',
    'attention_lr': r'$\alpha_{\phi}$'
}
