"""
This is the trainer for the nakliGPT
"""
import os
from tqdm import tqdm
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

class nakliTrainer:
    def __init__(self, config, model, dataloader):
        self.config = config
        self.model = model

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=config["training"]["weight_decay"]
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Set up tensorboard writer
        log_dir = config["logging"]["log_dir"]
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.device = config["misc"]["device"]
        self.model.to(self.device)

        self.dataloader = dataloader
        
    def train(self):

        for epoch in range(self.config["training"]["epochs"]):
            step = 0
            self.model.train()
            total_loss = 0

            progress_bar = tqdm(
                self.dataloader, 
                desc=f"Epoch {epoch+1}/{self.config['training']['epochs']}", 
                total=len(self.dataloader), 
                dynamic_ncols=True
            )

            for batch_idx, batch in enumerate(progress_bar):
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                logits = self.model(inputs)
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["gradient_clip"])
                self.optimizer.step()

                total_loss += loss.item()
                step += 1
                
                progress_bar.set_postfix(loss=total_loss / (batch_idx + 1), refresh=True)

                # Log every N steps
                if step % self.config["logging"]["log_interval"] == 0:
                    avg_loss = total_loss / self.config["logging"]["log_interval"]
                    self.writer.add_scalar("Loss/train", avg_loss, step)
                    print(f"[Epoch {epoch+1}] Step {step}: Loss = {avg_loss:.4f}")
                    total_loss = 0

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, save_path):
        self.model.load_state_dict(torch.load(save_path))