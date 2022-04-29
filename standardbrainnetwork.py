__author__ = 'gkour'

import torch.nn as nn
import torch
from Modules import ChannelProccessor


class StandardBrainNetworkOrig(nn.Module):
    def __init__(self, num_channels, num_actions):
        super(StandardBrainNetworkOrig, self).__init__()
        self.affine = nn.Linear(num_channels, 16, bias=False)
        self.controller = nn.Linear(16, num_actions, bias=False)
        self.model = torch.nn.Sequential(
            self.affine,
            nn.Dropout(p=0.6),
            nn.Sigmoid(),
            self.controller,
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)


class StandardBrainNetworkAttention(nn.Module):

	def __init__(self, num_channels, num_actions):
		super(StandardBrainNetworkAttention, self).__init__()
		self.models = nn.ModuleList([ChannelProccessor(6, 1) for _ in range(num_channels)])
		self.dim_attn = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=(1,num_channels))), requires_grad=True)
		self.controller = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=(1,num_actions))), requires_grad=True)

	def forward(self, x, door_attention=None):
		channels = [torch.select(x, dim=1, index=t) for t in range(x.shape[1])] # the +1 is for the batch dimension
		processed_channels = torch.cat([self.models[i](channel) for i, channel in enumerate(channels)],dim=-1)
		processed_channels_t = torch.transpose(processed_channels,dim0=-1,dim1=-2)
		dimension_attended = torch.matmul(torch.softmax(self.dim_attn, dim=-1),processed_channels_t.squeeze())
		if door_attention is None:
			door_attended = torch.softmax(torch.mul(torch.softmax(self.controller, dim=-1),dimension_attended), dim=-1).squeeze()
		else:
			door_attended = torch.softmax(torch.mul(torch.softmax(torch.tensor(door_attention).float(), dim=-1), dimension_attended),
										  dim=-1).squeeze()
		return door_attended

	def get_stimuli_layer(self):
		return torch.stack([channel_porc.model[0].weight.squeeze() for channel_porc in self.models])

	def get_controller(self):
		return self.controller

	def get_dimension_attention(self):
		return self.dim_attn


class SeparateNetworkAttention(nn.Module):

	def __init__(self, num_channels, num_actions):
		super(SeparateNetworkAttention, self).__init__()
		self.model_food = StandardBrainNetworkAttention(num_channels, num_actions)
		self.model_water = StandardBrainNetworkAttention(num_channels, num_actions)

	def forward(self, x, motivation):
		if motivation == 'water':
			return self.model_water(x)
		else:
			return self.model_food(x)

	def get_stimuli_layer(self):
		return torch.stack([channel_porc.model[0].weight.squeeze() for channel_porc in self.model_water.models])

	def get_controller(self):
		return self.model_water.controller

	def get_dimension_attention(self):
		return self.model_water.dim_attn


