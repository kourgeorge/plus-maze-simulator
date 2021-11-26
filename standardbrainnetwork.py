import torch.nn as nn
import torch

class StandardBrainNetwork(nn.Module):
    def __init__(self, num_channels, num_actions):
        super(StandardBrainNetwork, self).__init__()
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