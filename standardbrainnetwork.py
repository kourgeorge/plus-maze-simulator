__author__ = 'gkour'

import torch.nn as nn
import torch
import numpy as np
import utils
from Modules import ChannelProccessor

norm = 'fro'

class AbstractNetwork(nn.Module):

	def __str__(self):
		return self.__class__.__name__

	def get_stimuli_layer(self):
		raise NotImplementedError()

	def get_door_attention(self):
		raise NotImplementedError()

	def get_dimension_attention(self):
		raise NotImplementedError()

	def get_network_metrics(self):
		raise NotImplementedError()

	def network_diff(self, brain2):
		raise NotImplementedError()


class TabularQ():
	def __str__(self):
		return self.__class__.__name__

	def __init__(self, encoding_size, num_channels, num_actions):
		self._num_actions = num_actions
		self.Q = dict()

	def __call__(self, *args, **kwargs):
		state = args[0]
		state_actions_value = []
		for obs in np.argmax(state, axis=-1):
			if obs.tostring() not in self.Q.keys():
				self.Q[obs.tostring()] = 0.5 * np.ones(self._num_actions)
			state_actions_value.append(self.Q[obs.tostring()])
		return state_actions_value

	def set_state_action_value(self, state, action, value):
		obs = np.argmax(state, axis=-1)
		self.Q[obs.tostring()][action] = value

	def get_stimuli_layer(self):
		raise NotImplementedError()

	def get_door_attention(self):
		raise NotImplementedError()

	def get_dimension_attention(self):
		raise NotImplementedError()

	def get_network_metrics(self):
		return {'num_entries': len(self.Q.keys())}

	def network_diff(self, brain2):
		diff = [np.linalg.norm(self.Q[state] - brain2.Q[state]) for state in
				set(self.Q.keys()).intersection(brain2.Q.keys())]
		return {'table_change': np.mean(diff)*100}


class FullyConnectedNetwork(AbstractNetwork):
	def __init__(self, encoding_size, num_channels, num_actions):
		super().__init__()
		self.affine = nn.Linear(num_channels * num_actions * encoding_size, num_actions, bias=True)
		self.model = torch.nn.Sequential(
			self.affine,
			nn.Softmax(dim=-1)
		)

	def forward(self, x):
		return self.model(torch.flatten(x, start_dim=1))

	def get_stimuli_layer(self):
		return self.model[0].weight

	def get_door_attention(self):
		return torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])

	def get_dimension_attention(self):
		return torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])

	def get_network_metrics(self):
		return {'layer1_dim': utils.unsupervised_dimensionality(self.affine.weight.detach().numpy())}

	def network_diff(self, network2):
		affine1 = self.affine.weight.detach()
		affine2 = network2.affine.weight.detach()
		change = np.linalg.norm(affine1 - affine2, ord=norm)
		return {'layer1_dim_change': change}


class FullyConnectedNetwork2Layers(FullyConnectedNetwork):
	def __init__(self, encoding_size, num_channels, num_actions):
		super().__init__(encoding_size, num_channels, num_actions)
		self.controller = nn.Linear(num_actions, num_actions, bias=True)
		self.model = torch.nn.Sequential(
			self.affine,
			nn.Dropout(p=0.6),
			nn.Sigmoid(),
			self.controller,
			nn.Softmax(dim=-1)
		)

	def forward(self, x):
		return self.model(torch.flatten(x, start_dim=1))

	def get_stimuli_layer(self):
		return self.model[0].weight

	def get_door_attention(self):
		return self.model[3].weight

	def get_network_metrics(self):
		return {'layer1_dim': utils.unsupervised_dimensionality(self.affine.weight.detach()),
				'layer2_dim': utils.unsupervised_dimensionality(self.controller.weight.detach())}

	def network_diff(self, network2):
		affine1 = self.affine.weight.detach()
		affine2 = network2.affine.weight.detach()
		affine_change = np.linalg.norm(affine1 - affine2, ord=norm)

		controller1 = self.controller.weight.detach()
		controller2 = network2.controller.weight.detach()
		controller_change = np.linalg.norm(controller1 - controller2, ord=norm)
		return {'layer1_change': affine_change,
				'layer2_change': controller_change}


class EfficientNetwork(AbstractNetwork):
	"""This network handles the stimuli from each door similarly and separately. It first encodes each stimuli (channel)
	from each door (in a similar fashion) then attend the relevant stimuli and then the door. The encoding, attentions are learned"""

	def __init__(self, encoding_size, num_channels, num_actions):
		super().__init__()
		self.channels_encoding = nn.ModuleList([ChannelProccessor(encoding_size, 1) for _ in range(num_channels)])
		self.dim_attn = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=(1, num_channels))), requires_grad=True)
		self.door_attn = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=(1, num_actions))), requires_grad=True)
		self.door_attn_bias = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=(1, num_actions))), requires_grad=True)

	def forward(self, x, door_attention=None):
		channels = [torch.select(x, dim=1, index=t) for t in range(x.shape[1])]  # the +1 is for the batch dimension
		processed_channels = torch.cat([self.channels_encoding[i](channel) for i, channel in enumerate(channels)], dim=-1)
		processed_channels_t = torch.transpose(processed_channels, dim0=-1, dim1=-2)
		dimension_attended = torch.matmul(torch.softmax(self.dim_attn, dim=-1), processed_channels_t.squeeze())
		if door_attention is None:
			door_attended = torch.softmax(self.door_attn_bias + torch.mul(torch.softmax(self.door_attn, dim=-1),
																		  dimension_attended), dim=-1).squeeze()
		else:
			door_attended = torch.softmax(self.door_attn_bias +
				torch.mul(torch.softmax(torch.tensor(door_attention).float(), dim=-1), dimension_attended),
				dim=-1).squeeze()
		return door_attended

	def get_stimuli_layer(self):
		return torch.stack([channel_porc.model[0].weight.squeeze() for channel_porc in self.channels_encoding])

	def get_door_attention(self):
		return self.door_attn

	def get_dimension_attention(self):
		return self.dim_attn

	def get_network_metrics(self):
		# return {'odor_enc_norm': utils.normalized_norm(self.channels_encoding[0].model[0].weight.detach(),ord=norm),
		# 		'color_enc_norm': utils.normalized_norm(self.channels_encoding[1].model[0].weight.detach(),ord=norm),
		# 		'dim_attn_norm': utils.normalized_norm(self.dim_attn.detach(),ord=norm),
		# 		'door_attn_norm': utils.normalized_norm(self.door_attn.detach(),ord=norm)
		# 		}
		return {}

	def network_diff(self, network2):
		dim_attn1 = self.dim_attn.detach().numpy()
		dim_attn2 = network2.dim_attn.detach().numpy()
		dim_attn_change = EfficientNetwork.vector_change(dim_attn1,dim_attn2)

		door_attn1 = self.door_attn.detach().numpy()
		door_attn2 = network2.door_attn.detach().numpy()
		door_attn_change = EfficientNetwork.vector_change(door_attn1, door_attn2)

		door_attn_bias1 = self.door_attn_bias.detach().numpy()
		door_attn_bias2 = network2.door_attn_bias.detach().numpy()
		door_attn_bias_change = EfficientNetwork.vector_change(door_attn_bias1, door_attn_bias2)

		odor_proc1 = list(self.channels_encoding[0].modules())[2].weight.detach().numpy()
		odor_proc2 = list(network2.channels_encoding[0].modules())[2].weight.detach().numpy()
		odor_proc_change = EfficientNetwork.vector_change(odor_proc1, odor_proc2)


		color_proc1 = list(self.channels_encoding[1].modules())[2].weight.detach().numpy()
		color_proc2 = list(network2.channels_encoding[1].modules())[2].weight.detach().numpy()
		color_proc_change = EfficientNetwork.vector_change(color_proc1, color_proc2)

		return {'odor_enc_change': odor_proc_change,
				'color_enc_change': color_proc_change,
				'dim_attn_change': dim_attn_change,
				'door_attn_change': door_attn_change,
				'door_attn_bias_change': door_attn_bias_change}

	@staticmethod
	def vector_change(u,v):
		#return scipy.spatial.distance.correlation(u, v, centered=False)
		return utils.normalized_norm(u-v)
		#return scipy.spatial.distance.cosine(color_proc1, color_proc2)

class SeparateMotivationAreasNetwork(AbstractNetwork):

	def __init__(self, encoding_size, num_channels, num_actions):
		super().__init__()
		self.model_food = EfficientNetwork(encoding_size, num_channels, num_actions)
		self.model_water = EfficientNetwork(encoding_size, num_channels, num_actions)

	def forward(self, x, motivation):
		if motivation == 'water':
			return self.model_water(x)
		else:
			return self.model_food(x)

	def get_stimuli_layer(self):
		return torch.stack([channel_porc.model[0].weight.squeeze() for channel_porc in self.model_water.channels_encoding] + [
			channel_porc.model[0].weight.squeeze() for channel_porc in self.model_food.channels_encoding])

	def get_door_attention(self):
		return torch.cat([self.model_water.door_attn, self.model_food.door_attn])

	def get_dimension_attention(self):
		return torch.cat([self.model_water.dim_attn, self.model_food.dim_attn])

	def get_network_metrics(self):
		#return {**self.model_water.get_network_metrics(), **self.model_food.get_network_metrics()}
		return {}

	def network_diff(self, network2):
		water_diff = self.model_water.network_diff(network2.model_water)
		water_diff = {'water_{}'.format(k): v for k, v in water_diff.items()}

		food_dif = self.model_food.network_diff(network2.model_food)
		food_dif = {'food_{}'.format(k): v for k, v in food_dif.items()}

		return {**food_dif, **water_diff}