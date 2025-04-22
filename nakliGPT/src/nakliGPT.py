"""
Constructing nakliGPT
- Embedding
- Positional Encoding
- Transformer
- Logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder import nakliTransformerDecoder

class nakliGPT(nn.Module):
    """
    Implementation of nakliGPT
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config["model"]["vocab_size"], config["model"]["embd_dim"])
        self.positional_embedding = nn.Embedding(config["model"]["context_length"], config["model"]["embd_dim"])
        self.dropout_embedding = nn.Dropout(config["model"]["dropout"])

        self.transformer_block = nakliTransformerDecoder(
            embd_dim=config["model"]["embd_dim"],
            num_heads=config["model"]["num_heads"],
            context_length=config["model"]["context_length"],
            dropout=config["model"]["dropout"],
            qkv_bias=config["model"]["qkv_bias"]
        )
        self.n_transformer_blocks = nn.Sequential(
            *[
                nakliTransformerDecoder(                            # CHANGED! earlier, i reused the same nakliTransformerDecoder
                    embd_dim=config["model"]["embd_dim"],
                    num_heads=config["model"]["num_heads"],
                    context_length=config["model"]["context_length"],
                    dropout=config["model"]["dropout"],
                    qkv_bias=config["model"]["qkv_bias"]) for _ in range(config["model"]["num_blocks"])
            ]
        )

        self.final_layer_norm = nn.LayerNorm(config["model"]["embd_dim"])
        self.W_output = nn.Linear(config["model"]["embd_dim"], config["model"]["vocab_size"], bias=False)

    def forward(self, x):
        # token embeddings
        token_embd = self.token_embedding(x)

        # positional embeddings
        pos_embd = self.positional_embedding(torch.arange(x.shape[1]))

        # combined embeddings
        x = token_embd + pos_embd

        # dropout
        x = self.dropout_embedding(x)

        # transformer blocks
        x = self.n_transformer_blocks(x)
        
        # final layer norm
        x = self.final_layer_norm(x)
        
        # logits
        logits = self.W_output(x)
        return logits
        
        

    
