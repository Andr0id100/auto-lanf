import torch
from torch import nn
import math


class FeedForward(nn.Module):
    def __init__(self, hidden_dim=512, ff_dim=2048):
        super().__init__()
        self.layer1 = nn.Linear(hidden_dim, ff_dim)
        self.layer2 = nn.Linear(ff_dim, hidden_dim)

    def forward(self, x):
        x = self.layer2(torch.relu(self.layer1(x)))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, max_length=1024, hidden_dim=512, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        position = torch.arange(max_length).unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2)
                             * (-math.log(10000) / hidden_dim))

        self.positional_encoding = torch.zeros(max_length, 1, hidden_dim)
        self.positional_encoding[:, 0, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, 0, 1::2] = torch.cos(position * div_term)
        self.positional_encoding = self.positional_encoding.to(device)

    def forward(self, x):
        return self.positional_encoding[:x.size(0)]
