"""
Main script for training and testing nakliGPT
"""
import os
import yaml
import torch
import tiktoken

from torch.utils.tensorboard import SummaryWriter

from nakliGPT import nakliGPT
from trainer import nakliTrainer, load_config
from text_utils import nakliDataset, nakliGreedySampling

def main():

    # tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")

    # load config
    config_path = "/home/ge73qip/LLMs/LLMs_from_scratch/nakliGPT/configuration/configuration.yaml"
    config = load_config(config_path)

    # load dataset
    dataset = nakliDataset(
        txt_path=config["data"]["dataset_path"],
        tokenizer=tokenizer,
        max_length_token=config["model"]["context_length"],
        stride=config["data"]["stride"]
    )
    dataloader = dataset.data_loader()

    # load model
    model = nakliGPT(config)

    # load trainer
    trainer = nakliTrainer(
        config=config,
        model=model,
        dataloader=dataloader
    )

    # train
    trainer.train()

    # save model
    model_path = "nakliGPT_model.pth"
    trainer.save_model(model_path)

if __name__ == "__main__":
    main()
