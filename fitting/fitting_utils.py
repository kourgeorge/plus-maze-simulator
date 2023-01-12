__author__ = 'gkour'

import os
import sys
import numpy as np

from fitting.fitting_config import friendly_models_name_map
from motivatedagent import MotivatedAgent
from environment import PlusMazeOneHotCues
from rewardtype import RewardType
import re


def episode_rollout_on_real_data(env: PlusMazeOneHotCues, agent: MotivatedAgent, current_trial):
	total_reward = 0
	num_actions = env.num_actions()
	act_dist = np.zeros(num_actions)

	env_state = env.reset()
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
		new_state, outcome, terminated, info = env.step(action)
		reward = agent.evaluate_outcome(outcome)
		if (outcome == RewardType.NONE and current_trial.reward != 0) or \
				(outcome != RewardType.NONE and current_trial.reward == 0):
			raise Exception("There is a discrepancy between data and simulation reward.")
		total_reward += reward
		model_action_dist = agent.get_brain().think(np.expand_dims(state, 0), agent).squeeze().detach().numpy()
		likelihood += model_action_dist[action]

		agent.add_experience(state, dec_1hot, reward, outcome, new_state, terminated, info)

		env.set_state(current_trial)
		info.likelihood = likelihood
		info.model_action = agent.decide(state)
		_, model_action_outcome, _, _ = env.step(info.model_action)
		info.network_outcome = model_action_outcome

		state = new_state
	return steps, total_reward, act_dist, model_action_dist, info.model_action+1, likelihood, model_action_outcome


def blockPrint():
	sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
	sys.stdout = sys.__stdout__


def get_stage_transition_days(experimental_data):
	return np.where(experimental_data['day in stage'] == 1)[0][1:]


def string2list(string):
	try:
		params= [float(x.strip()) for x in re.split(" +",string.strip(' ]['))]
	except Exception:
		params= [float(x.strip()) for x in re.split(",",string.strip(']['))]
	return params



def stable_unique(array):
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]


def maze_experimental_data_preprocessing(experiment_data):

	# remove trials with non-active doors selection:
	experiment_data['completed'] = experiment_data.apply(
		lambda x: False if np.isnan(x.action) else x["A{}o".format(int(x.action))] != -1, axis='columns')
	experiment_data = experiment_data[experiment_data.completed == True]
	experiment_data.drop('completed', axis='columns', inplace=True)

	df_sum = experiment_data.groupby(['stage', 'day in stage'], sort=False).agg({'reward': 'mean', 'action':'count'}).reset_index()

	# Take at most 7 days from the last stage.
	df = experiment_data.copy()
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
																		 len(experiment_data)-len(df)))

	return df


def models_order_df(df):
	models_in_df = np.unique(df.model)
	return [model for model in friendly_models_name_map.values() if model in models_in_df]


def rename_models(model_df):
	model_df["model"] = model_df.model.map(
		lambda x: friendly_models_name_map[x] if x in friendly_models_name_map.keys() else x)
	return model_df
