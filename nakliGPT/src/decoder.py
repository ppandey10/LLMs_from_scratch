"""
Combining decoder components to build the nakliTransformerDecoder
- Multi-Head Attention
- Layer Normalization
- Residual Connection
- Feed Forward Network
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

#======================#

class nakliTransformerDecoder(nn.Module):
    """
    Implementation of nakliTransformerDecoder
    """
    def __init__(
        self,
        embd_dim: int, 
        num_heads: int,
        context_length: int,
        dropout: float,
        qkv_bias: bool = False,
    ): 
        super().__init__()
        self.attention = nakliMultiHeadAttention(
            in_dim=embd_dim,
            out_dim=embd_dim,
            context_length=context_length,
            dropout=dropout,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.layer_norm_1 = nn.LayerNorm(embd_dim)
        self.layer_norm_2 = nn.LayerNorm(embd_dim)
        self.feed_forward = nakliFeedForward(embd_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        temp_x = x
        #==================#
        # Multi-Head Attention
        #==================#
        x = self.layer_norm_1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = temp_x + x # residual

        #==================#
        # Feed Forward
        #==================#
        temp_x = x # change temporary x
        x = self.layer_norm_2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = temp_x + x
        
        return x

#======================#
# Multi-Head Attention
#======================#
class nakliMultiHeadAttention(nn.Module):
    """
    Implementation of efficient Multi-Head Attention
    - Number of Heads
    - Query, Key, Value: Weights and Projection
    """
    def __init__(self, in_dim, out_dim, context_length, num_heads, dropout, qkv_bias=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.qkv_bias = qkv_bias

        # head dim
        self.head_dim = out_dim // num_heads
        assert self.head_dim * num_heads == out_dim, "out_dim must be divisible by num_heads"

        # query, key, value weights
        self.W_query = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.W_key = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.W_value = nn.Linear(in_dim, out_dim, bias=qkv_bias)

        # output projection matrix
        self.W_output = nn.Linear(out_dim, out_dim, bias=qkv_bias)

        # register buffer -> makes it a part of the model's state which is not non-trainable
        self.register_buffer("mask", torch.tril(torch.ones(context_length, context_length).view(1, 1, context_length, context_length))) # so that it can be multiplied with attention scores

    def forward(self, x):
        bs, num_tokens, input_dim = x.shape # (batch_size, total_tokens_for_sentence/sequence, embd_dim)

        # query, key, value 
        # (bs, total_tokens_for_sentence, in_dim) x (in_dim, out_dim) -> (bs, total_tokens_for_sentence, out_dim)
        query = self.W_query(x) 
        key = self.W_key(x)
        value = self.W_value(x)

        # we are still missing the "multi-head" part
        # unroll the out_dim: (bs, total_tokens_for_sentence, out_dim) -> (bs, total_tokens_for_sentence, num_heads, head_dim)
        query = query.view(bs, num_tokens, self.num_heads, self.head_dim)
        key = key.view(bs, num_tokens, self.num_heads, self.head_dim)
        value = value.view(bs, num_tokens, self.num_heads, self.head_dim)

        # transpose: (bs, total_tokens_for_sentence, num_heads, head_dim) -> (bs, num_heads, total_tokens_for_sentence, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # compute attention: (bs, num_heads, total_tokens_for_sentence, head_dim) x (bs, num_heads, head_dim, total_tokens_for_sentence) -> (bs, num_heads, total_tokens_for_sentence, total_tokens_for_sentence)
        attention_scores = torch.matmul(query, key.transpose(2, 3))
        attention_scores = attention_scores / np.sqrt(key.size(-1))

        # causal mask
        causal_mask = self.mask[:, :, :num_tokens, :num_tokens]
        attention_scores = attention_scores.masked_fill(causal_mask == 0, float('-inf'))

        # apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)

        # dropout
        attention_weights = self.dropout(attention_weights)

        # context vector
        # (bs, num_heads, total_tokens_for_sentence, total_tokens_for_sentence) x (bs, num_heads, total_tokens_for_sentence, head_dim) -> (bs, num_heads, total_tokens_for_sentence, head_dim)
        context_vec = (attention_weights @ value).transpose(1, 2)

        # combine/concatenate heads
        context_vec = context_vec.contiguous().view(bs, num_tokens, self.out_dim)

        # output projection
        output = self.W_output(context_vec)

        return output

#======================#
# Feed Forward
#======================#
class nakliFeedForward(nn.Module):
    """
    Implementation of feed forward network
    - 4 times the input dimension
    - GELU activation
    """
    def __init__(self, in_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, 4 * in_dim)
        self.act = nn.GELU()
        self.W_output = nn.Linear(4 * in_dim, in_dim)

    def forward(self, x):
        return self.W_output(self.act(self.W(x)))

# test
# if __name__ == "__main__":
#     mha = nakliMultiHeadAttention(in_dim=512, out_dim=512, context_length=256, num_heads=8)
#     x = torch.randn(2, 256, 512)
#     output = mha.forward(x)
#     print(output.shape)
    