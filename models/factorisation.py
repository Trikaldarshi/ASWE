import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
import random

# define a model with a embedding layer and list of linear layers
class LinearModel(nn.Module):
    def __init__(self, num_classes, input_dim, output_dim, hidden_dims, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.embed = nn.Embedding(num_classes, output_dim)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x
