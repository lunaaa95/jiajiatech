import torch
import torch.nn as nn
from torch.nn.functional import softmax
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, key_size):
        """
        :param key_size: dimension of key
        """
        super().__init__()
        self.key_size = key_size

    def forward(self, query, key, value, mask):
        """
        :param query: (batch_size, (n_head,) len_q, key_size)
        :param key: (batch_size, (n_head,) len_k, key_size)
        :param value: (batch_size, (n_head,) len_k, value_size)
        :param mask: (batch_size, (n_head,) len_q, len_k)
        :return: result: (batch_size, (n_head,) len_q, value_size), score: (batch_size, (n_head,) len_q, len_k)
        """
        score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.key_size)
        # score: (batch_size, (n_head,) len_q, len_k)

        '''Apply mask
        mask: (batch_size, (n_head,) len_q, len_k)
        score: (batch_size, (n_head,) len_q, len_k)
        '''
        score.masked_fill_(mask, 0.)
        score = softmax(score, dim=-1)
        # score: (batch_size, (n_head,) len_q, len_k)

        '''Result
        score: (batch_size, (n_head,) len_q, len_k)
        value: (batch_size, (n_head,) len_q, value_size)
        '''
        result = torch.matmul(score, value)
        # result: (batch_size, (n_head,) len_q, value_size)
        return result, score


class LinearAttention(nn.Module):
    def __init__(self, key_size):
        """
        :param key_size: dimension of key
        """
        super().__init__()
        self.key_size = key_size
        self.linear = nn.Linear(key_size, key_size)

    def forward(self, query, key, value, mask):
        """
        :param query: (batch_size, (n_head,) len_q, key_size)
        :param key: (batch_size, (n_head,) len_k, key_size)
        :param value: (batch_size, (n_head,) len_k, value_size)
        :param mask: (batch_size, (n_head,) len_q, len_k)
        :return: result: (batch_size, (n_head,) len_q, value_size), score: (batch_size, (n_head,) len_q, len_k)
        """
        score = self.linear(query)
        score = torch.matmul(score, key.transpose(-1, -2))
        # score: (batch_size, (n_head,) len_q, len_k)

        '''Apply mask
        mask: (batch_size, (n_head,) len_q, len_k)
        score: (batch_size, (n_head,) len_q, len_k)
        '''
        score.masked_fill_(mask, 0.)
        score = softmax(score, dim=-1)
        # score: (batch_size, (n_head,) len_q, len_k)

        '''Result
        score: (batch_size, (n_head,) len_q, len_k)
        value: (batch_size, (n_head,) len_q, value_size)
        '''
        result = torch.matmul(score, value)
        # result: (batch_size, (n_head,) len_q, value_size)
        return result, score


class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, value_size, n_head, d_k, d_v):
        """
        :param key_size: size of hidden layer
        :param value_size: size of value
        :param n_head: number of heads
        :param d_k: size of key in each head
        :param d_v: size of value in each head
        """
        super().__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(key_size, d_k * n_head)
        self.W_K = nn.Linear(key_size, d_k * n_head)
        self.W_V = nn.Linear(value_size, d_v * n_head)
        self.scaled_dot_product_attention = ScaledDotProductAttention(d_k)
        self.W_O = nn.Linear(n_head * d_v, value_size)
        self.layer_norm = nn.LayerNorm(value_size)

    def forward(self, query, key, value, mask):
        """
        :param query: (batch_size, len_q, key_size)
        :param key: (batch_size, len_k, key_size)
        :param value: (batch_size, len_k, value_size)
        :param mask: (batch_size, len_q, len_k)
        :return: output: (batch_size, len_q, value_size), score: (batch_size, n_head, len_q, len_k)
        """
        batch_size = query.size(0)

        '''
        (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        已经将memory展现给每个batch,此时再映射到多个子空间
        '''
        q_s = self.W_Q(query).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        # q_s: (batch_size, n_heads, len_q, d_k)
        k_s = self.W_K(key).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        # k_s: (batch_size, n_heads, len_k, d_k)
        v_s = self.W_V(value).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)
        # v_s: (batch_size, n_heads, len_k, d_v)
        mask = mask.unsqueeze(1).expand(-1, self.n_head, -1, -1)
        # attn_mask : (batch_size, n_heads, len_q, len_k)

        output, score = self.scaled_dot_product_attention(q_s, k_s, v_s, mask)
        # output: (batch_size, n_head, len_q, d_v)
        # score: (batch_size, n_head, len_q, len_k)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_v)
        # output: (batch_size, len_q, n_head * d_v)
        output = self.layer_norm(self.W_O(output))
        # output: (batch_size, len_q, value_size)
        return output, score
