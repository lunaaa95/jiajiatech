import torch
import torch.nn as nn
from torch.nn.functional import softmax


class PassThrough(nn.Module):
    def __init__(self, value_size):
        super().__init__()
        self.value_size = value_size

    def forward(self, value):
        return value


class LinearDecoder(nn.Module):
    def __init__(self, value_size):
        super().__init__()
        self.value_size = value_size
        self.linear = nn.Linear(value_size, value_size)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, value):
        value = self.linear(value)
        value = self.leaky_relu(value)
        value = softmax(value, dim=-1)
        # value_reading: (batch_size, value_size)
        return value
