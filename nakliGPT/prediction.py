"""
Testing the saved model 
"""
import tiktoken
import torch
from trainer import nakliTrainer
from nakliGPT import nakliGPT
from text_utils import nakliGreedySampling
from trainer import load_config

if __name__ == "__main__":
    # load config
    config = load_config("configuration.yaml")

    # load model 
    nakliGPT_model = nakliGPT(config)
    nakliGPT_model.load_state_dict(torch.load("nakliGPT_model.pth"))
    nakliGPT_model.eval()
    
    # load tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # generate text
    generator = nakliGreedySampling(
        model=nakliGPT_model,
        tokenizer=tokenizer,
        num_new_tokens=10,
        max_context_length=config["model"]["context_length"]
    )

    input_text = "Where will you"
    # generate text
    text = generator.generate_text(
        torch.tensor(tokenizer.encode(input_text)).unsqueeze(0)
    )

    print("\n--- Input Text ---")
    print(input_text)
    print("\n--- Generated Text ---")
    print(text)

