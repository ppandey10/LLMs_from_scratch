"""
Combining decoder components to build the nakliTransformerDecoder
- Multi-Head Attention
- Layer Normalization
- Residual Connection
- Feed Forward Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder_components import nakliMultiHeadAttention, nakliFeedForward

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
        x = self.layer_norm_1(temp_x)
        x = self.attention(x)
        x = self.dropout(x)
        x = temp_x + x # residual

        #==================#
        # Feed Forward
        #==================#
        temp_x = x # change temporary x
        x = self.layer_norm_2(temp_x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = temp_x + x
        
        return x
    
    