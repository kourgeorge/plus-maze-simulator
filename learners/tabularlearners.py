__author__ = 'gkour'

import numpy as np

import config
import utils
from learners.abstractlearner import AbstractLearner
from models.tabularmodels import QTable, FTable, ACFTable
from rewardtype import RewardType


class QLearner(AbstractLearner):
	def __init__(self, model: QTable, learning_rate=config.LEARNING_RATE, *args, **kwargs):
		super().__init__(model=model, optimizer={'learning_rate': learning_rate})

	def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation):
		learning_rate = self.optimizer['learning_rate']
		actions = np.argmax(action_batch, axis=1)
		all_action_values = self.model(state_batch, motivation)
		selected_action_value = all_action_values[np.arange(len(all_action_values)), actions]
		deltas = (reward_batch - selected_action_value)
		updated_q_values = selected_action_value + learning_rate * deltas

		self.model.set_actual_state_value(state_batch, actions, updated_q_values, motivation)

		return deltas


class ABQLearner(QLearner):
	def __init__(self, model: QTable, learning_rate=config.LEARNING_RATE, alpha_bias=None, *args, **kwargs):
		super().__init__(model=model, learning_rate=learning_rate,  *args, **kwargs)
		self.alpha_bias = alpha_bias if alpha_bias is not None else learning_rate

	def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation):
		deltas = super().learn(state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation)
		actions = np.argmax(action_batch, axis=1)

		for action in np.unique(actions):
			new_action_bias_value = self.model.get_bias_values(motivation)[action] + \
									self.alpha_bias * np.mean(deltas[actions == action])
			self.model.set_bias_value(motivation, action, new_action_bias_value)
			
		return deltas


class UABQLearner(QLearner):
	def __init__(self, model: QTable, learning_rate=config.LEARNING_RATE, alpha_bias=None, *args, **kwargs):
		super().__init__(model=model, learning_rate=learning_rate, alpha_bias=None, *args, **kwargs)
		self.alpha_bias = alpha_bias if alpha_bias is not None else learning_rate

	def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation):
		deltas = super().learn(state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation)
		actions = np.argmax(action_batch, axis=1)

		for action in np.unique(actions):
			self.model.action_bias[RewardType.WATER.value][action] += self.alpha_bias * np.mean(deltas[actions == action])
			self.model.action_bias[RewardType.FOOD.value][action] += self.alpha_bias * np.mean(deltas[actions == action])

		return deltas


class IALearner(AbstractLearner):
	def __init__(self, model: FTable, learning_rate=0.01, *args, **kwargs):
		super().__init__(model=model, optimizer={'learning_rate': learning_rate})

	def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation):

		learning_rate = self.optimizer['learning_rate']
		actions = np.argmax(action_batch, axis=1)

		# Calculate the Q function for all actions
		all_action_values = self.model(state_batch, motivation)

		# calculate the Q function for selected action
		selected_action_value = all_action_values[np.arange(len(all_action_values)), actions]

		if np.any(np.isinf(selected_action_value)):
			print('Warning! rat Selected inactive door!')
			return 0
		delta = (reward_batch - selected_action_value)
		selected_odors, selected_colors = self.model.get_selected_door_stimuli(state_batch, actions)

		#phi = utils.softmax(self.model.phi) if isinstance(self.model, ACFTable) else [1/3, 1/3, 1/3]
		phi = self.model.phi if isinstance(self.model, ACFTable) else [1, 1, 1]
		for odor in set(np.unique(selected_odors)).difference([self.model.encoding_size]):
			self.model.update_stimulus_value('odors', odor, motivation, learning_rate * phi[0] * np.nanmean(delta[selected_odors == odor]))
		for color in set(np.unique(selected_colors)).difference([self.model.encoding_size]):
			self.model.update_stimulus_value('colors',color,motivation, learning_rate * phi[1] * np.nanmean(delta[selected_colors == color]))
		for door in np.unique(actions):
			self.model.update_stimulus_value('spatial',door,motivation, learning_rate * phi[2] * np.nanmean(delta[actions == door]))

		return delta


class ABIALearner(IALearner):
	def __init__(self, model: FTable,  learning_rate=config.LEARNING_RATE, alpha_bias=None, *args, **kwargs):
		super().__init__(model=model, learning_rate=learning_rate, *args, **kwargs)
		self.alpha_bias = alpha_bias if alpha_bias is not None else learning_rate

	def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation):
		deltas = super().learn(state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation)
		actions = np.argmax(action_batch, axis=1)

		for action in np.unique(actions):
			new_action_bias_value = self.model.get_bias_values(motivation)[action] + \
									self.alpha_bias*np.mean(deltas[actions==action])
			self.model.set_bias_value(motivation, action, new_action_bias_value)

		return deltas

#
# class ABIALearnerG(IALearner):
# 	def __init__(self, model: FTable, learning_rate=0.01):
# 		super().__init__(model=model, learning_rate=learning_rate)
#
# 	def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation):
# 		deltas = super().learn(state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation)
# 		actions = np.argmax(action_batch, axis=1)
# 		learning_rate = self.optimizer['learning_rate']
#
# 		for action in np.unique(actions):
# 			delta = np.mean(deltas[actions == action])
# 			if (action == 0 or action == 1) and motivation.value=='food' and delta < 0:
# 				foor_actions = self.model.action_bias['food']
# 				print("a:{} - Q:{} - delta:{} - food_actions={} - food_bias={}".
# 					  format(action, self.model(state_batch, motivation)[0][action],deltas[actions == action],
# 												 foor_actions, np.sum(foor_actions[0:2]) - np.sum(foor_actions[2:4])))
#
# 			self.model.action_bias[motivation.value][action] += learning_rate*np.mean(deltas[actions==action])
#
# 		return deltas


class UABIALearner(IALearner):
	def __init__(self, model: FTable, alpha_bias=config.LEARNING_RATE, *args, **kwargs):
		super().__init__(model=model, *args, **kwargs)
		self.alpha_bias = alpha_bias

	def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation):
		deltas = super().learn(state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation)
		actions = np.argmax(action_batch, axis=1)

		for action in np.unique(actions):
			new_action_bias_value = self.model.get_bias_values(motivation)[action] + \
									self.alpha_bias * np.mean(deltas[actions == action])
			self.model.set_bias_value(RewardType.FOOD, action, new_action_bias_value)
			self.model.set_bias_value(RewardType.WATER, action, new_action_bias_value)

		return deltas


class IAAluisiLearner(AbstractLearner):
	def __init__(self, model: FTable, learning_rate=0.01):
		super().__init__(model=model, optimizer={'learning_rate': learning_rate})

	def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation):

		learning_rate = self.optimizer['learning_rate']
		actions = np.argmax(action_batch, axis=1)

		# Calculate the Q function for all actions
		all_action_values = self.model(state_batch, motivation)

		# calculate the Q function for selected action
		selected_action_value = all_action_values[np.arange(len(all_action_values)), actions]

		if np.any(np.isinf(selected_action_value)):
			print('Warning! rat Selected inactive door!')
			return 0
		delta = (reward_batch - selected_action_value)
		selected_odors, selected_colors = self.model.get_selected_door_stimuli(state_batch, actions)

		for odor in set(np.unique(selected_odors)).difference([self.model.encoding_size]):
			self.model.Q['odors'][odor] += \
				learning_rate * np.nanmean(reward_batch[selected_odors == odor] - self.model.Q['odors'][odor])
		for color in set(np.unique(selected_colors)).difference([self.model.encoding_size]):
			self.model.Q['colors'][color] += \
				learning_rate * np.nanmean(reward_batch[selected_colors == color] - self.model.Q['colors'][color])
		for door in np.unique(actions):
			self.model.Q['spatial'][door] += \
				 learning_rate * np.nanmean(reward_batch[actions == door] - self.model.Q['spatial'][door])

		return delta


class MALearner(IALearner):
	def __init__(self, model:ACFTable, alpha_phi=0.005, *args, **kwargs):
		super().__init__(model, *args, **kwargs)
		self.alpha_phi = alpha_phi

	def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation):

		super().learn(state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation)

		actions = np.argmax(action_batch, axis=1)

		all_action_values = self.model(state_batch, motivation)
		selected_action_value = all_action_values[np.arange(len(all_action_values)), actions]

		deltas = (reward_batch - selected_action_value)
		selected_odors, selected_colors = self.model.get_selected_door_stimuli(state_batch, actions)

		phi_s = utils.softmax(self.model.phi)

		optimization_data = {}
		for choice_value, selected_odor, selected_color, selected_door, delta, reward in zip(selected_action_value, selected_odors, selected_colors, actions, deltas, reward_batch):
			V = np.array([self.model.Q[motivation.value]['odors'][selected_odor],
						  self.model.Q[motivation.value]['colors'][selected_color],
						  self.model.Q[motivation.value]['spatial'][selected_door]])
			attention_regret = self.calc_attention_regret(choice_value, reward, V)

			self.model.phi += self.alpha_phi * delta * phi_s * attention_regret
			delta_phi = delta * phi_s * attention_regret

			optimization_data['delta'] = delta
			optimization_data['odor_regret'] = attention_regret[0]
			optimization_data['color_regret'] = attention_regret[1]
			optimization_data['spatial_regret'] = attention_regret[2]
			optimization_data['Q'] = selected_action_value[0]
			optimization_data['odor_V'] = V[0]
			optimization_data['color_V'] = V[1]
			optimization_data['spatial_V'] = V[2]
			optimization_data['odor_delta_phi'] = delta_phi[0]
			optimization_data['color_delta_phi'] = delta_phi[1]
			optimization_data['spatial_delta_phi'] = delta_phi[2]


		return optimization_data

	def calc_attention_regret(self, Q, reward, V):
		return V-Q


class ABMALearner(MALearner):

	def __init__(self, model: ACFTable, *args, **kwargs):
		super().__init__(model=model, *args, **kwargs)

	def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation):
		deltas = super().learn(state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation)
		actions = np.argmax(action_batch, axis=1)
		learning_rate = self.optimizer['learning_rate']

		for action in np.unique(actions):
			self.model.action_bias[motivation.value][action] += learning_rate * np.mean(deltas[actions == action])

		return deltas


class UABMALearner(MALearner):

	def __init__(self, model: ACFTable, *args, **kwargs):
		super().__init__(model=model, *args, **kwargs)

	def learn(self, state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation):
		deltas = super().learn(state_batch, action_batch, reward_batch, action_values, nextstate_batch, motivation)
		actions = np.argmax(action_batch, axis=1)
		learning_rate = self.optimizer['learning_rate']

		for action in np.unique(actions):
			self.model.action_bias[RewardType.WATER.value][action] += learning_rate * np.mean(deltas[actions == action])
			self.model.action_bias[RewardType.FOOD.value][action] += learning_rate * np.mean(deltas[actions == action])

		return deltas


class MALearnerSimple(MALearner):
	def calc_attention_regret(self, Q, reward, V):
		return V-reward


