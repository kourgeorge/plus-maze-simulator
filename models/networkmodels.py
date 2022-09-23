__author__ = 'gkour'

import torch.nn as nn
import torch
import numpy as np
import scipy

import utils
from models.Modules import ChannelProccessor

norm = 'fro'


class AbstractNetworkModel(nn.Module):

	def __str__(self):
		return self.__class__.__name__

	def get_model_metrics(self):
		raise NotImplementedError()

	def get_model_diff(self, brain2):
		raise NotImplementedError()


class UniformAttentionNetwork(AbstractNetworkModel):
	def __init__(self, encoding_size, num_channels, num_actions):
		super().__init__()
		self.odors = nn.Linear(encoding_size, 1, bias=False)
		self.colors = nn.Linear(encoding_size, 1, bias=False)
		self.spatial = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=(1, num_actions))), requires_grad=True)
		self.phi = torch.ones([3, 1])/3

	def forward(self, x):
		x_odor = x[:, 0]
		x_light = x[:, 1]
		odor_val = self.odors(x_odor)
		light_val = self.colors(x_light)
		door_val = torch.unsqueeze(self.spatial.repeat(x.shape[0], 1), dim=-1)
		weighted_vals = torch.matmul(torch.cat([odor_val, light_val, door_val], dim=-1), torch.softmax(self.phi, axis=0))
		return torch.squeeze(weighted_vals, dim=2)


	def get_model_metrics(self):
		return {'colors': np.linalg.norm(self.colors.weight.detach()),
				'odors':  np.linalg.norm(self.odors.weight.detach()),
				'spatial':  np.linalg.norm(self.spatial.detach())}

	def get_model_diff(self, network2):
		return {'d(colors)': np.linalg.norm(self.colors.weight.detach() - network2.colors.weight.detach()),
				'd(odors)': np.linalg.norm(self.odors.weight.detach() - network2.odors.weight.detach()),
				'd(spatial)': np.linalg.norm(self.spatial.detach() - network2.spatial.detach())}


class AttentionAtChoiceAndLearningNetwork(UniformAttentionNetwork):
	def __init__(self, encoding_size, num_channels, num_actions):
		super().__init__(encoding_size, num_channels, num_actions)
		#self.phi = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=(3,1))), requires_grad=True)
		self.phi = nn.Parameter(torch.ones(size=(3,1)), requires_grad=True)

	def get_model_metrics(self):
		return {'odor_attn': self.phi[0].detach().numpy(),
				'color_attn': self.phi[1].detach().numpy(),
				'spatial_attn': self.phi[2].detach().numpy()}

	def get_model_diff(self, model2):
		return {'color_attn': self.phi[0].detach().numpy() - model2.phi[0].detach().numpy(),
				'odor_attn': self.phi[1].detach().numpy() - model2.phi[1].detach().numpy(),
				'spatial_attn': self.phi[2].detach().numpy() - model2.phi[2].detach().numpy()}


class FullyConnectedNetwork(AbstractNetworkModel):
	def __init__(self, encoding_size, num_channels, num_actions):
		super().__init__()
		self.affine = nn.Linear(num_channels * num_actions * encoding_size, num_actions, bias=True)
		self.model = torch.nn.Sequential(
			self.affine,
		)

	def forward(self, x):
		return self.model(torch.flatten(x, start_dim=1))

	def get_model_metrics(self):
		return {'affine': utils.normalized_norm(self.affine.weight.detach().numpy()),
				'bias': utils.normalized_norm(self.affine.bias.detach().numpy())}

	def get_model_diff(self, network2):
		affine1 = self.affine.weight.detach()
		affine2 = network2.affine.weight.detach()
		change = np.linalg.norm(affine1 - affine2, ord=norm)
		return {'layer1_dim_change': change}


class Random(AbstractNetworkModel):
	def __init__(self, encoding_size, num_channels, num_actions):
		super().__init__()
		self.affine = nn.Linear(num_channels * num_actions * encoding_size, num_actions, bias=True)

	def forward(self, x):
		return 0

	def get_stimuli_layer(self):
		return self.model[0].weight

	def get_door_attention(self):
		return torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])

	def get_dimension_attention(self):
		return torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])

	def get_model_metrics(self):
		return {'None':0}

	def get_model_diff(self, network2):
		return {'None':0}


class FullyConnectedNetwork2Layers(FullyConnectedNetwork):
	def __init__(self, encoding_size, num_channels, num_actions):
		super().__init__(encoding_size, num_channels, num_actions)
		self.controller = nn.Linear(num_actions, num_actions, bias=True)
		self.model = torch.nn.Sequential(
			self.affine,
			nn.Dropout(p=0),
			nn.Sigmoid(),
			self.controller,
		)

	def get_model_metrics(self):
		return {'affine1': utils.normalized_norm(self.affine.weight.detach().numpy()),
				'bias1': utils.normalized_norm(self.affine.bias.detach().numpy()),
				'affine2': utils.normalized_norm(self.controller.weight.detach().numpy()),
				'bias2': utils.normalized_norm(self.controller.bias.detach().numpy())
				}

	def get_model_diff(self, network2):
		affine1 = self.affine.weight.detach()
		affine2 = network2.affine.weight.detach()
		affine_change = np.linalg.norm(affine1 - affine2, ord=norm)

		controller1 = self.controller.weight.detach()
		controller2 = network2.controller.weight.detach()
		controller_change = np.linalg.norm(controller1 - controller2, ord=norm)
		return {'layer1_change': affine_change,
				'layer2_change': controller_change}


class EfficientNetwork(AbstractNetworkModel):
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
			door_attended = self.door_attn_bias + torch.mul(torch.softmax(self.door_attn, dim=-1),
																		  dimension_attended).squeeze()
		else:
			door_attended = self.door_attn_bias + \
							torch.mul(torch.softmax(torch.tensor(door_attention).float(), dim=-1), dimension_attended).squeeze()
		return door_attended

	def get_stimuli_layer(self):
		return torch.stack([channel_porc.model[0].weight.squeeze() for channel_porc in self.channels_encoding])

	def get_door_attention(self):
		return self.door_attn

	def get_dimension_attention(self):
		return self.dim_attn

	def get_model_metrics(self):
		# return {'odor_enc_norm': utils.normalized_norm(self.channels_encoding[0].model[0].weight.detach(),ord=norm),
		# 		'color_enc_norm': utils.normalized_norm(self.channels_encoding[1].model[0].weight.detach(),ord=norm),
		# 		'dim_attn_norm': utils.normalized_norm(self.dim_attn.detach(),ord=norm),
		# 		'door_attn_norm': utils.normalized_norm(self.door_attn.detach(),ord=norm)
		# 		}
		return {}

	def get_model_diff(self, network2):
		dim_attn1 = self.dim_attn.detach().numpy()
		dim_attn2 = network2.dim_attn.detach().numpy()
		dim_attn_change = EfficientNetwork.vector_change(dim_attn1, dim_attn2)

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


class SeparateMotivationAreasNetwork(AbstractNetworkModel):

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

	def get_model_metrics(self):
		#return {**self.model_water.get_network_metrics(), **self.model_food.get_network_metrics()}
		return {}

	def get_model_diff(self, network2):
		water_diff = self.model_water.get_model_diff(network2.model_water)
		water_diff = {'water_{}'.format(k): v for k, v in water_diff.items()}

		food_dif = self.model_food.get_model_diff(network2.model_food)
		food_dif = {'food_{}'.format(k): v for k, v in food_dif.items()}

		return {**food_dif, **water_diff}


class SeparateMotivationAreasFCNetwork(AbstractNetworkModel):

	def __init__(self, encoding_size, num_channels, num_actions):
		super().__init__()
		self.model_food = FullyConnectedNetwork(encoding_size, num_channels, num_actions)
		self.model_water = FullyConnectedNetwork(encoding_size, num_channels, num_actions)

	def forward(self, x, motivation):
		if motivation == 'water':
			return self.model_water(x)
		else:
			return self.model_food(x)

	def get_stimuli_layer(self):
		return self.model_food[0].weight

	def get_door_attention(self):
		return torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])

	def get_dimension_attention(self):
		return torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])

	def get_model_metrics(self):
		return {'layer1_dim': utils.unsupervised_dimensionality(self.model_food.affine.weight.detach().numpy())}

	def get_model_diff(self, network2):
		affine1 = self.model_food.affine.weight.detach()
		affine2 = network2.model_food.affine.weight.detach()
		change = np.linalg.norm(affine1 - affine2, ord=norm)
		return {'layer1_dim_change': change}