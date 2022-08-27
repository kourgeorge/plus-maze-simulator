__author__ = 'gkour'

import numpy as np
from motivatedagent import MotivatedAgent
from environment import PlusMazeOneHotCues


def get_timestamp():
	import datetime
	return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")


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
		total_reward += reward
		model_action_dist = agent.get_brain().think(np.expand_dims(state, 0), agent).squeeze().detach().numpy()
		likelihood += -1 * np.log(model_action_dist[action])

		agent.add_experience(state, dec_1hot, reward, outcome, new_state, terminated, info)

		state = env.set_state(current_trial)
		info.likelihood = likelihood
		info.network_action = agent.decide(state)
		_, outcome_network, _, _ = env.step(info.network_action)
		info.network_outcome = outcome_network

		state = new_state
	return steps, total_reward, act_dist, model_action_dist, likelihood