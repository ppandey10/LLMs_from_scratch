"""
Implementation of dataset generation for the nakliGPT
"""
import tiktoken
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class nakliDataset(Dataset):
    def __init__(self, txt_path, tokenizer, max_length_token, stride):
        self.input_ids = []
        self.target_ids = []

        with open(txt_path, "r", encoding="utf-8") as f:
            txt = f.read()

        # token ids
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length_token, stride):
            input_chunk = token_ids[i:i + max_length_token]
            target_chunk = token_ids[i + 1: i + max_length_token + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def data_loader(self, batch_size=4, shuffle=True, drop_last=True, num_workers=0):

        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

class nakliGreedySampling:
    def __init__(self, model, tokenizer, num_new_tokens, max_context_length):
        self.model = model
        self.tokenizer = tokenizer
        self.num_new_tokens = num_new_tokens
        self.max_context_length = max_context_length

    def generate_next_token(self, input_ids):
        # crop current context if it exceeds the supported context size
        # input_ids: (batch_size, context_length) -> (batch_size, max_context_length)
        input_ids = input_ids[:, -self.max_context_length:]

        # get logits
        logits = self.model(input_ids)
        
        # get last token
        # logits: (batch_size, num_tokens, vocab_size) -> (batch_size, vocab_size)
        last_token = logits[:, -1, :]

        # apply softmax to get probabilities
        probs = F.softmax(last_token, dim=-1) # (batch_size, vocab_size)

        # predict next token
        next_token = torch.argmax(probs, dim=-1, keepdim=True) # (batch_size, 1)

        return next_token

    def generate_all_tokens(self, input_ids):
        for _ in range(self.num_new_tokens):
            # get next token
            next_token = self.generate_next_token(input_ids)

            # append next token to input_ids
            input_ids = torch.cat((input_ids, next_token), dim=1)

        return input_ids

    def generate_text(self, input_ids):
        # generate all tokens
        input_ids = self.generate_all_tokens(input_ids)

        # decode tokens
        text = self.tokenizer.decode(input_ids[0].tolist())
        return text






            
            
 
