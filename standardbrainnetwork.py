__author__ = 'gkour'

import torch.nn as nn
import torch
import numpy as np
import utils
from Modules import ChannelProccessor
import scipy


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


norm = 1

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
		return {'affine_dim': utils.unsupervised_dimensionality(self.affine.weight.detach().numpy())}

	def network_diff(self, network2):
		affine1 = self.get_stimuli_layer().T.detach().numpy()
		affine2 = network2.get_stimuli_layer().T.detach().numpy()
		distance = np.linalg.norm(affine1 - affine2, ord=norm)
		return {'affine_distance': distance}


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
		return {'affine_dim':utils.unsupervised_dimensionality(self.affine.weight.detach()),
				'controller_dim': utils.unsupervised_dimensionality(self.controller.weight.detach())}

	def network_diff(self, network2):
		affine1 = self.get_stimuli_layer().detach().numpy()
		affine2 = network2.get_stimuli_layer().detach().numpy()
		affine_distance = np.linalg.norm(affine1 - affine2, ord=norm)

		controller1 = self.get_door_attention().detach().numpy()
		controller2 = network2.get_door_attention().detach().numpy()
		controller_distance = np.linalg.norm(controller1 - controller2, ord=norm)
		return {'affine_distance': affine_distance,
				'controller_distance':controller_distance}


class EfficientNetwork(AbstractNetwork):
	"""This network handles the stimuli from each door similarly and separately. It first encodes each stimuli (channel)
	from each door (in a similar fashion) then attend the relevant stimuli and then the door. The encoding, attentions are learned"""

	def __init__(self, encoding_size, num_channels, num_actions):
		super().__init__()
		self.chanells_encoding = nn.ModuleList([ChannelProccessor(encoding_size, 1) for _ in range(num_channels)])
		self.dim_attn = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=(1, num_channels))), requires_grad=True)
		self.door_attn = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=(1, num_actions))), requires_grad=True)

	def forward(self, x, door_attention=None):
		channels = [torch.select(x, dim=1, index=t) for t in range(x.shape[1])]  # the +1 is for the batch dimension
		processed_channels = torch.cat([self.chanells_encoding[i](channel) for i, channel in enumerate(channels)], dim=-1)
		processed_channels_t = torch.transpose(processed_channels, dim0=-1, dim1=-2)
		dimension_attended = torch.matmul(torch.softmax(self.dim_attn, dim=-1), processed_channels_t.squeeze())
		if door_attention is None:
			door_attended = torch.softmax(torch.mul(torch.softmax(self.door_attn, dim=-1), dimension_attended),
										  dim=-1).squeeze()
		else:
			door_attended = torch.softmax(
				torch.mul(torch.softmax(torch.tensor(door_attention).float(), dim=-1), dimension_attended),
				dim=-1).squeeze()
		return door_attended

	def get_stimuli_layer(self):
		return torch.stack([channel_porc.model[0].weight.squeeze() for channel_porc in self.chanells_encoding])

	def get_door_attention(self):
		return self.door_attn

	def get_dimension_attention(self):
		return self.dim_attn

	def get_network_metrics(self):
		return {'odor_encoding':utils.unsupervised_dimensionality(self.chanells_encoding[0].model[0].weight.detach()),
				'color_encoding': utils.unsupervised_dimensionality(self.chanells_encoding[1].model[0].weight.detach())}

	def network_diff(self, network2):
		dim_attn1 = self.dim_attn.detach().numpy()
		dim_attn2 = network2.dim_attn.detach().numpy()
		dim_attn_distance = EfficientNetwork.vector_distance(dim_attn1,dim_attn2)

		door_attn1 = self.door_attn.detach().numpy()
		door_attn2 = network2.door_attn.detach().numpy()
		door_attn_distance = EfficientNetwork.vector_distance(door_attn1, door_attn2)

		odor_proc1 = list(self.chanells_encoding[0].modules())[2].weight.detach().numpy()
		odor_proc2 = list(network2.chanells_encoding[0].modules())[2].weight.detach().numpy()
		odor_proc_distance = EfficientNetwork.vector_distance(odor_proc1, odor_proc2)


		color_proc1 = list(self.chanells_encoding[1].modules())[2].weight.detach().numpy()
		color_proc2 = list(network2.chanells_encoding[1].modules())[2].weight.detach().numpy()
		color_proc_distance = EfficientNetwork.vector_distance(color_proc1, color_proc2)

		return {'odor_proc_distance': odor_proc_distance,
				'color_proc_distance': color_proc_distance,
				'dim_attn_distance': dim_attn_distance,
				'door_attn_distance': door_attn_distance}

	@staticmethod
	def vector_distance(u,v):
		return scipy.spatial.distance.correlation(u, v, centered=False)
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
		return torch.stack([channel_porc.model[0].weight.squeeze() for channel_porc in self.model_water.chanells_encoding] + [
			channel_porc.model[0].weight.squeeze() for channel_porc in self.model_food.chanells_encoding])

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