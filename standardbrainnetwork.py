__author__ = 'gkour'

import torch.nn as nn
import torch
from Modules import ChannelProccessor


class AbstractNetwork(nn.Module):

	def __str__(self):
		return self.__class__.__name__

	def get_stimuli_layer(self):
		raise NotImplementedError()

	def get_door_attention(self):
		raise NotImplementedError()

	def get_dimension_attention(self):
		raise NotImplementedError()


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


class DoorAttentionAttention(AbstractNetwork):

	def __init__(self, encoding_size, num_channels, num_actions):
		super().__init__()
		self.models = nn.ModuleList([ChannelProccessor(encoding_size, 1) for _ in range(num_channels)])
		self.dim_attn = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=(1, num_channels))), requires_grad=True)
		self.door_attn = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=(1, num_actions))), requires_grad=True)

	def forward(self, x, door_attention=None):
		channels = [torch.select(x, dim=1, index=t) for t in range(x.shape[1])]  # the +1 is for the batch dimension
		processed_channels = torch.cat([self.models[i](channel) for i, channel in enumerate(channels)], dim=-1)
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
		return torch.stack([channel_porc.model[0].weight.squeeze() for channel_porc in self.models])

	def get_door_attention(self):
		return self.door_attn

	def get_dimension_attention(self):
		return self.dim_attn


class SeparateMotivationAreasNetwork(AbstractNetwork):

	def __init__(self, encoding_size, num_channels, num_actions):
		super().__init__()
		self.model_food = DoorAttentionAttention(encoding_size, num_channels, num_actions)
		self.model_water = DoorAttentionAttention(encoding_size, num_channels, num_actions)

	def forward(self, x, motivation):
		if motivation == 'water':
			return self.model_water(x)
		else:
			return self.model_food(x)

	def get_stimuli_layer(self):
		return torch.stack([channel_porc.model[0].weight.squeeze() for channel_porc in self.model_water.models] + [
			channel_porc.model[0].weight.squeeze() for channel_porc in self.model_food.models])

	def get_door_attention(self):
		return torch.cat([self.model_water.door_attn, self.model_food.door_attn])

	def get_dimension_attention(self):
		return torch.cat([self.model_water.dim_attn, self.model_food.dim_attn])
