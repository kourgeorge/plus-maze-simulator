__author__ = 'gkour'

import os
import sys
import numpy as np

import config
from fitting.PlusMazeExperimentFitting import PlusMazeExperimentFitting
from fitting.fitting_config import friendly_models_name_map
from learners.networklearners import DQNAtt
from learners.tabularlearners import MALearner
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

		# This validates that if the reward in the data is 0 the reward by the simulation should be RewardType.NONE.
		# and vice versa, meaning if the reward in the data is not 0 then the reward type in the simulation must not be none.
		if (outcome == RewardType.NONE and current_trial.reward != 0) or \
				(outcome != RewardType.NONE and current_trial.reward == 0):
			raise Exception("There is a discrepancy between data and simulation reward!!\ntrial={}, stage={}, action={}"
							" reward={}".format(current_trial.trial,current_trial.stage,current_trial.action, current_trial.reward))

		total_reward += reward
		model_action_dist = agent.get_brain().think(np.expand_dims(state, 0), agent).squeeze().detach().numpy()
		likelihood += model_action_dist[action]

		agent.add_experience(state, dec_1hot, reward, outcome, new_state, terminated, info)

		env.set_state(current_trial)
		info.likelihood = likelihood
		info.model_action = agent.decide_greedy(state)
		_, model_action_outcome, _, _ = env.step(info.model_action)
		info.network_outcome = model_action_outcome

		state = new_state
	return steps, total_reward, act_dist, model_action_dist, info.model_action+1, likelihood, model_action_outcome


def run_model_on_animal_data(env, rat_data, model_arch, parameters):
	(brain, learner, model) = model_arch
	model_instance = model(env.stimuli_encoding_size(), 2, env.num_actions())

	if issubclass(learner, MALearner) or issubclass(learner, DQNAtt):
		(beta, lr, attention_lr) = parameters
		learner_instance = learner(model_instance, learning_rate=lr, alpha_phi=attention_lr)
	else:
		(beta, lr) = parameters
		learner_instance = learner(model_instance, learning_rate=lr)

	#blockPrint()
	env.init()
	agent = MotivatedAgent(brain(learner_instance, beta=beta),
						   motivation=RewardType.WATER,
						   motivated_reward_value=config.MOTIVATED_REWARD,
						   non_motivated_reward_value=config.NON_MOTIVATED_REWARD, exploration_param=0)

	experiment_stats, rat_data_with_likelihood = PlusMazeExperimentFitting(env, agent, dashboard=False,
																		   experiment_data=rat_data)
	#enablePrint()

	return experiment_stats, rat_data_with_likelihood


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
		lambda x: False if np.isnan(x.action) or np.isnan(x.reward) else x["A{}o".format(int(x.action))] != -1, axis='columns')
	experiment_data_filtered = experiment_data[experiment_data.completed == True]
	experiment_data_filtered.drop('completed', axis='columns', inplace=True)

	df_sum = experiment_data.groupby(['stage', 'day in stage'], sort=False).agg({'reward': 'mean', 'action':'count'}).reset_index()

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
																		 len(experiment_data)-len(df)))

	return df


def models_order_df(df):
	models_in_df = np.unique(df.model)
	return stable_unique([model for model in friendly_models_name_map.values() if model in models_in_df])


def rename_models(model_df):
	model_df["model"] = model_df.model.map(
		lambda x: friendly_models_name_map[x] if x in friendly_models_name_map.keys() else x)
	return model_df
