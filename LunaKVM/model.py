import ipdb
import torch
import torch.nn as nn
import math
from LunaKVM.attention import ScaledDotProductAttention, LinearAttention, MultiHeadAttention
from LunaKVM.encoder import GRUEncoder
from LunaKVM.decoder import PassThrough, LinearDecoder


MEM_BATCH = 1024


class KeyValueMemoryNet(nn.Module):
    """
    GRU Encoder -> KV Memory -> Scaled Dot-Product Attention -> Output
    """
    def __init__(self, input_size, value_size, hidden_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.output_size = value_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder = GRUEncoder(input_size, value_size, hidden_size, num_layers)
        self.attention = ScaledDotProductAttention(hidden_size)
        self.decoder = PassThrough(value_size)

    def forward(self, memory, query, mask):
        # query: (batch_size, seq_len, input_size)
        batch_size = query.size(0)
        key = memory[0]
        # key: (N, seq_len, input_size). N is the number of memories per query
        value = memory[1]
        # value: (N, value_size)
        # mask: (batch_size, N)
        
        '''Embedding
        query: (batch_size, seq_len, input_size)
        key: (N, seq_len, input_size)
        '''
        query_embedding = self.encoder(query)
        # query_embedding: (batch_size, hidden_size)
        key_embedding = self.encoder(key)
        # key_embedding: (N, hidden_size)
        
        '''Attention
        query_embedding: (batch_size, hidden_size)
        key_embedding: (N, hidden_size)
        value: (N, value_size)
        mask: (batch_size, N)
        '''
        query_embedding = query_embedding.view(batch_size, 1, -1)
        # query_embedding: (batch_size, 1, hidden_size)
        key_embedding = key_embedding.expand(batch_size, -1, -1)
        # key_embedding: (batch_size, N, hidden_size)
        value = value.expand(batch_size, -1, -1)
        # value: (batch_size, N, value_size)
        mask = mask.view(batch_size, 1, -1)
        # mask: (batch_size, 1, N)
        value_reading, _ = self.attention(query_embedding, key_embedding, value, mask)
        # value_reading: (batch_size, 1, value_size)
        value_reading = value_reading.view(batch_size, -1)
        # value_reading: (batch_size, value_size)
        
        '''Output
        value_reading: (batch_size, value_size)
        '''
        value_reading = self.decoder(value_reading)
        # value_reading: (batch_size, value_size)
        return value_reading

    def get_embedding(self, key):
        key_embedding = self.encoder(key)
        # key_embedding: (N, hidden_size)
        return key_embedding

    def get_attention(self, memory, query, mask):
        batch_size = query.size(0)
        key = memory[0]
        # key: (N, seq_len, input_size). N is the number of memories per query
        value = memory[1]
        # value: (N, value_size)
        # mask: (batch_size, N)

        '''Embedding
        query: (batch_size, seq_len, input_size)
        key: (N, seq_len, input_size)
        '''
        query_embedding = self.encoder(query)
        # query_embedding: (batch_size, hidden_size)
        key_embedding = self.encoder(key)
        # key_embedding: (N, hidden_size)

        '''Attention
        query_embedding: (batch_size, hidden_size)
        key_embedding: (N, hidden_size)
        value: (N, value_size)
        mask: (batch_size, N)
        '''
        query_embedding = query_embedding.view(batch_size, 1, -1)
        # query_embedding: (batch_size, 1, hidden_size)
        key_embedding = key_embedding.expand(batch_size, -1, -1)
        # key_embedding: (batch_size, N, hidden_size)
        value = value.expand(batch_size, -1, -1)
        # value: (batch_size, N, value_size)
        mask = mask.view(batch_size, 1, -1)
        # mask: (batch_size, 1, N)
        _, attn = self.attention(query_embedding, key_embedding, value, mask)
        # attn: (batch_size, (n_head,) len_q, len_k)
        attn = attn.squeeze(-2)
        # attn: (batch_size, (n_head,) len_k)
        return attn


class LinearAttentionKVMNet(KeyValueMemoryNet):
    """
    GRU Encoder -> KV Memory -> Linear Attention -> Output
    """
    def __init__(self, input_size, value_size, hidden_size, num_layers):
        super().__init__(input_size, value_size, hidden_size, num_layers)
        self.encoder = GRUEncoder(input_size, value_size, hidden_size, num_layers)
        self.attention = LinearAttention(hidden_size)
        self.decoder = PassThrough(value_size)


class MultiHeadAttentionKVMNet(KeyValueMemoryNet):
    """
    GRU Encoder -> KV Memory -> Multi-Head Attention -> Output
    """
    def __init__(self, input_size, value_size, hidden_size, num_layers):
        super().__init__(input_size, value_size, hidden_size, num_layers)
        self.encoder = GRUEncoder(input_size, value_size, hidden_size, num_layers)
        self.attention = MultiHeadAttention(
            key_size=hidden_size,
            value_size=value_size,
            n_head=4,
            d_k=8,
            d_v=2
        )
        self.decoder = PassThrough(value_size)


class FeedForwardDecoderKVMNet(KeyValueMemoryNet):
    """
    GRU Encoder -> KV Memory -> Scaled Dot-Product Attention -> Feed Forward -> Output
    """
    def __int__(self, input_size, value_size, hidden_size, num_layers):
        super().__int__(input_size, value_size, hidden_size, num_layers)
        self.encoder = GRUEncoder(input_size, value_size, hidden_size, num_layers)
        self.attention = ScaledDotProductAttention(hidden_size)
        self.decoder = LinearDecoder(value_size)
