"""
Script for running + training nakliGPT
"""

import yaml 
import tiktoken
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from text_utils import nakliGreedySampling
from nakliGPT import nakliGPT

# load config
def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# main
if __name__ == "__main__":
    config = load_config("configuration.yaml")

    # # create dataset
    # dataset = nakliDataset(
    #     txt=config["data"]["dataset_path"],
    #     tokenizer=config["data"]["tokenizer_type"],
    #     max_length_token=config["data"]["block_size"],
    #     stride=config["data"]["block_size"]
    # )
    tokenizer = tiktoken.get_encoding("gpt2")

    #=================#

    # batch = []

    input_text_1 = "He comes and chats" # line 840
    # input_text_2 = "This is about the" # line 845 

    # batch.append(torch.tensor(tokenizer.encode(input_text_1)))
    # batch.append(torch.tensor(tokenizer.encode(input_text_2)))
    # batch = torch.stack(batch, dim=0)
    # print("\n--- Input ---")
    # print(batch)
    
    # model
    nakliGPT_model = nakliGPT(config["model"])

    # # check to see if model is working
    # out = nakliGPT_model(batch)
    # print("\n--- Output Shape ---")
    # print(out.shape)
    # print("\n--- Output ---")
    # print(out)

    #=================#
    
    # generate text
    generator = nakliGreedySampling(
        model=nakliGPT_model,
        tokenizer=tokenizer,
        num_new_tokens=6,
        max_context_length=config["model"]["context_length"]
    )

    # generate text
    text = generator.generate_text(
        torch.tensor(tokenizer.encode(input_text_1)).unsqueeze(0)
    )

    print("\n--- Reference Text ---")
    print(input_text_1)
    print("\n--- Generated Text (should generate random)---")
    print(text)


    
