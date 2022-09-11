__author__ = 'gkour'

import torch.nn as nn
import torch
from torch.nn import functional as F


class FilterSum(nn.Module):
	def __init__(self, shape, sum_dim):
		super(FilterSum, self).__init__()
		filter_size = list(shape)
		del filter_size[sum_dim]
		self.affine = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=filter_size)))
		self.sum_dim = sum_dim

	def forward(self, input):
		res = [torch.matmul(torch.select(input, dim=self.sum_dim + 1, index=t), self.affine[:, t]) for t in
			   range(input.shape[1])]  # the +1 is for the batch dimension
		return torch.transpose(torch.stack(res, dim=1), dim0=1, dim1=2)


class FilterSum2(nn.Module):
	'''This computation layer given an n-dimensional input tensor, it multiplicates each slice in the sum_dim with a parameter'''

	def __init__(self, shape, sum_dim, encoding_size):
		super(FilterSum2, self).__init__()
		filter_size = list(shape)
		del filter_size[sum_dim]
		filter_size += [encoding_size]
		self.affine = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=filter_size)))
		self.sum_dim = sum_dim

	def forward(self, input):
		res = [torch.matmul(torch.select(input, dim=self.sum_dim + 1, index=t), self.affine[:, t]) for t in
			   range(input.shape[1])]  # the +1 is for the batch dimension
		# return torch.stack(res,dim=1)
		return torch.transpose(torch.stack(res, dim=1), dim0=1, dim1=3)


class LinearTranspose(nn.Module):
	def __init__(self, in_features, out_features):
		super(LinearTranspose, self).__init__()
		self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
		self.bias = nn.Parameter(torch.Tensor(out_features, in_features))

	def forward(self, input):
		res = torch.transpose(F.linear(input, self.weight, self.bias), dim0=-1, dim1=-2).squeeze()
		return res


class AttentionTranspose(nn.Module):
	def __init__(self, attention_size):
		super(AttentionTranspose, self).__init__()
		self.attn = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=attention_size)))

	def forward(self, input):
		res = torch.transpose(torch.matmul(input, torch.softmax(self.attn, dim=0)).squeeze(), dim0=-1, dim1=-2)
		return res


class Attention1(nn.Module):
	def __init__(self, attention_size):
		super(Attention1, self).__init__()
		self.attn = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=attention_size)))

	def forward(self, input):
		res = torch.mul(input, torch.softmax(self.attn, dim=-1))
		return res


class ChannelProccessor(nn.Module):
	def __init__(self, in_features, out_features):
		super(ChannelProccessor, self).__init__()
		self.model = nn.Sequential(
			nn.Linear(in_features, out_features, bias=False),
			nn.LeakyReLU()
		)

	def forward(self, input):
		res = self.model(input)
		return res
