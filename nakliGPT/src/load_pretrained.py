#====================#
# Adapted from Sebastian Raschka     
#====================#

# HARD CODED
# Helps load the weights into the nakliGPT model
import numpy as np
import torch

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.positional_embedding.weight = assign(gpt.positional_embedding.weight, params['wpe'])
    gpt.token_embedding.weight = assign(gpt.token_embedding.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.n_transformer_blocks[b].attention.W_query.weight = assign(
            gpt.n_transformer_blocks[b].attention.W_query.weight, q_w.T)
        gpt.n_transformer_blocks[b].attention.W_key.weight = assign(
            gpt.n_transformer_blocks[b].attention.W_key.weight, k_w.T)
        gpt.n_transformer_blocks[b].attention.W_value.weight = assign(
            gpt.n_transformer_blocks[b].attention.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.n_transformer_blocks[b].attention.W_query.bias = assign(
            gpt.n_transformer_blocks[b].attention.W_query.bias, q_b)
        gpt.n_transformer_blocks[b].attention.W_key.bias = assign(
            gpt.n_transformer_blocks[b].attention.W_key.bias, k_b)
        gpt.n_transformer_blocks[b].attention.W_value.bias = assign(
            gpt.n_transformer_blocks[b].attention.W_value.bias, v_b)

        gpt.n_transformer_blocks[b].attention.W_output.weight = assign(
            gpt.n_transformer_blocks[b].attention.W_output.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.n_transformer_blocks[b].attention.W_output.bias = assign(
            gpt.n_transformer_blocks[b].attention.W_output.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.n_transformer_blocks[b].feed_forward.W.weight = assign(
            gpt.n_transformer_blocks[b].feed_forward.W.weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.n_transformer_blocks[b].feed_forward.W.bias = assign(
            gpt.n_transformer_blocks[b].feed_forward.W.bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.n_transformer_blocks[b].feed_forward.W_output.weight = assign(
            gpt.n_transformer_blocks[b].feed_forward.W_output.weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.n_transformer_blocks[b].feed_forward.W_output.bias = assign(
            gpt.n_transformer_blocks[b].feed_forward.W_output.bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.n_transformer_blocks[b].layer_norm_1.weight = assign(
            gpt.n_transformer_blocks[b].layer_norm_1.weight,
            params["blocks"][b]["ln_1"]["g"])
        gpt.n_transformer_blocks[b].layer_norm_1.bias = assign(
            gpt.n_transformer_blocks[b].layer_norm_1.bias,
            params["blocks"][b]["ln_1"]["b"])
        gpt.n_transformer_blocks[b].layer_norm_2.weight = assign(
            gpt.n_transformer_blocks[b].layer_norm_2.weight,
            params["blocks"][b]["ln_2"]["g"])
        gpt.n_transformer_blocks[b].layer_norm_2.bias = assign(
            gpt.n_transformer_blocks[b].layer_norm_2.bias,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_layer_norm.weight = assign(gpt.final_layer_norm.weight, params["g"])
    gpt.final_layer_norm.bias = assign(gpt.final_layer_norm.bias, params["b"])
    gpt.W_output.weight = assign(gpt.W_output.weight, params["wte"])
