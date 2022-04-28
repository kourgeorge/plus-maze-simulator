__author__ = 'gkour'

import torch.nn as nn
import torch


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
		num_channels = 2
		super(StandardBrainNetworkAttention, self).__init__()
		self.models = nn.ModuleList([ChannelProccessor(6, 1) for _ in range(num_channels)])
		self.dim_attn = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=(1,2))), requires_grad=True)
		self.controller = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=(1,4))), requires_grad=True)

	def forward(self, x):
		channels = [torch.select(x, dim=1, index=t) for t in range(x.shape[1])] # the +1 is for the batch dimension
		processed_channels = torch.cat([self.models[i](channel) for i, channel in enumerate(channels)],dim=-1)
		processed_channels_t = torch.transpose(processed_channels,dim0=-1,dim1=-2)
		dimension_attended = torch.matmul(torch.softmax(self.dim_attn, dim=-1),processed_channels_t.squeeze())
		door_attended = torch.softmax(torch.mul(torch.softmax(self.controller, dim=-1),dimension_attended), dim=-1).squeeze()
		return door_attended

	def get_affine(self):
		return torch.stack([channel_porc.model[0].weight.squeeze() for channel_porc in self.models])


class ChannelProccessor(nn.Module):
	def __init__(self,in_features, out_features):
		super(ChannelProccessor, self).__init__()
		self.model = nn.Sequential(
			nn.Linear(in_features, 1, bias=False),
			nn.LeakyReLU()
		)

	def forward(self, input):
		res = self.model(input)
		return res