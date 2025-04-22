"""
Implementing instruction finetuning
"""
import json
import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import DataLoader
from text_utils import nakliInstructionFineTuneDataset, collate_func, split_dataset
from nakliGPT import nakliGPT
from trainer import nakliTrainer
from load_pretrained import load_weights_into_gpt
from gpt_download import download_and_load_gpt2
from trainer import load_config

model_configs = {
    "gpt2-small (124M)": {"embd_dim": 768, "context_length": 1024, "num_blocks": 12, "num_heads": 12, "qkv_bias": True},
    "gpt2-medium (355M)": {"embd_dim": 1024, "context_length": 1024, "num_blocks": 24, "num_heads": 16, "qkv_bias": True},
    "gpt2-large (774M)": {"embd_dim": 1280, "context_length": 1024, "num_blocks": 36, "num_heads": 20, "qkv_bias": True},
    "gpt2-xl (1558M)": {"embd_dim": 1600, "context_length": 1024, "num_blocks": 48, "num_heads": 25, "qkv_bias": True},
}


def main():
    # load config
    config_path = "/home/ge73qip/LLMs/LLMs_from_scratch/nakliGPT/configuration/configuration.yaml"
    config = load_config(config_path)

    CHOOSE_MODEL = "gpt2-medium (355M)"
    config["model"].update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size,
        models_dir="gpt2"
    )

    # load model
    model = nakliGPT(config)
    load_weights_into_gpt(model, params)
    model.eval()

    # load dataset
    with open(config["data"]["finetune_dataset_path"], "r") as f:
        dataset = json.load(f)

    train_data, val_data, test_data = split_dataset(dataset)

    train_dataset = nakliInstructionFineTuneDataset(
        dataset=train_data,
        tokenizer=tiktoken.get_encoding("gpt2"),
    ).apalca_style()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_func
    )

    # load trainer
    trainer = nakliTrainer(
        config=config,
        model=model,
        dataloader=train_dataloader
    )

    # train
    trainer.train()

    # save model
    model_path = "nakliGPT_instructfinetuned.pth"
    trainer.save_model(model_path)

if __name__ == "__main__":
    main()

