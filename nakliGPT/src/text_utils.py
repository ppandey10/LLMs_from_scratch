"""
Implementation of dataset generation for the nakliGPT
"""
import tiktoken
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

#=====================#
# Instruction Dataset #
#=====================#

class nakliInstructionFineTuneDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.encoded_dataset = []

    def format_apalca_style_input(self, entry):
        instruction = (
            f"Below is an instruction that describes a task. "
            f"Write an appropriate response for it."
            f"\n\n### Instruction: \n{entry['instruction']}"
        )
        input = (
            f"\n\n### Input: \n"
            f"{entry['input']}" if entry['input'] else ""
        )
        
        combined = instruction + input

        return combined
    
    def apalca_style(self):
        for item in self.dataset:
            input_modified = self.format_apalca_style_input(item)
            response_modified = f"\n\nResponse: \n{item['output']}"
            apalca_type_format = input_modified + response_modified
            self.encoded_dataset.append(self.tokenizer.encode(apalca_type_format))

        return self.encoded_dataset
        
    def data_loaders(self, batch_size=4, shuffle=True, drop_last=True, num_workers=0):

        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )

    def __len__(self):
        return len(self.encoded_dataset)

    def __getitem__(self, idx):
        return self.encoded_dataset[idx]

#=====================#
# Text Dataset        #
#=====================#

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

#=====================#
# Greedy Sampling     #
#=====================#

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

#====================#
# Collate Function   #
#====================#

def collate_func(batch, padding_token_id=50256, allowed_max_length=None, device="cpu"):
    # since, each sample has different length, we need to pad them
    # find max length
    max_length = max(len(sample)+1 for sample in batch)

    # padded batch 
    padded_inputs = []
    padded_targets = []

    for sample in batch:
        # add padding token to make all samples same length
        padded_sample = sample + [padding_token_id] * (max_length - len(sample))

        padded_inpt = torch.tensor(padded_sample[:-1])
        padded_targ = torch.tensor(padded_sample[1:])

        # replace output padding token with -100
        mask = padded_targ == padding_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            padded_targ[indices[1:]] = -100

        if allowed_max_length is not None:
            padded_inpt = padded_inpt[:allowed_max_length]
            padded_targ = padded_targ[:allowed_max_length]

        padded_inputs.append(padded_inpt)
        padded_targets.append(padded_targ)

    # convert to tensor
    padded_inputs = torch.stack(padded_inputs).to(device)
    padded_targets = torch.stack(padded_targets).to(device)

    return padded_inputs, padded_targets

#====================#
# Dataset Splitting  #
#====================#

def split_dataset(dataset, train_ratio=0.85, val_ratio=0.1, test_ratio=0.05):
    train_portion = int(len(dataset) * train_ratio)
    val_portion = int(len(dataset) * val_ratio)
    test_portion = int(len(dataset) * test_ratio)
    return dataset[:train_portion], dataset[train_portion:train_portion + val_portion], dataset[train_portion + val_portion:train_portion + val_portion + test_portion]
