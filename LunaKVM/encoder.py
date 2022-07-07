import torch
import torch.nn as nn
from LunaKVM.attention import MultiHeadAttention


class GRUEncoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

    def init_hidden(self, bsz, requires_grad=True):
        weight = next(self.parameters())
        return weight.new_zeros((self.num_layers, bsz, self.hidden_size), requires_grad=requires_grad)

    def forward(self, key):
        # key: (batch_size, seq_len, input_size)
        _, key_embedding = self.encoder(key, self.init_hidden(key.size(0)))
        # key_embedding: (1, batch_size, hidden_size)
        key_embedding = key_embedding.squeeze(0)
        # key_embedding: (batch_size, hidden_size)
        return key_embedding
