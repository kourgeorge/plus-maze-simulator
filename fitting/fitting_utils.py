__author__ = 'gkour'

import os
import sys
import numpy as np
from motivatedagent import MotivatedAgent
from environment import PlusMazeOneHotCues
from rewardtype import RewardType
import re

def get_timestamp():
	import datetime
	return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

def brain_name(architecture):
	#return "{}.{}.{}".format(architecture[0].__name__, architecture[1].__name__, architecture[2].__name__)
	return "{}.{}".format( architecture[1].__name__,architecture[2].__name__)


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
			raise Exception("There is a discripency between data and simulation reward.")
		total_reward += reward
		model_action_dist = agent.get_brain().think(np.expand_dims(state, 0), agent).squeeze().detach().numpy()
		likelihood += -1 * np.log(model_action_dist[action])

		agent.add_experience(state, dec_1hot, reward, outcome, new_state, terminated, info)

		env.set_state(current_trial)
		info.likelihood = likelihood
		info.model_action = agent.decide(state)
		_, model_action_outcome, _, _ = env.step(info.model_action)
		info.network_outcome = model_action_outcome

		state = new_state
	return steps, total_reward, act_dist, model_action_dist, likelihood, model_action_outcome


def blockPrint():
	sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
	sys.stdout = sys.__stdout__


def get_stage_transition_days(experimental_data):
	return np.where(experimental_data['day in stage'] == 1)[0][1:]


def string2list(string):
	return [float(x.strip()) for x in re.split(",",string.strip(']['))]