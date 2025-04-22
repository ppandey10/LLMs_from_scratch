"""
Testing the saved model 
"""
import tiktoken
import torch
from trainer import nakliTrainer
from nakliGPT import nakliGPT
from text_utils import nakliGreedySampling
from trainer import load_config
from load_pretrained import load_weights_into_gpt
from gpt_download import download_and_load_gpt2

if __name__ == "__main__":
    # load config
    config = load_config("/home/ge73qip/LLMs/LLMs_from_scratch/nakliGPT/configuration/configuration.yaml")
    gpt_model_config = {
        "gpt2-medium (355M)": {"embd_dim": 1024, "context_length": 1024, "num_blocks": 24, "num_heads": 16, "qkv_bias": True}
    }
    CHOOSE_MODEL = "gpt2-medium (355M)"

    # load model 
    config["model"].update(gpt_model_config[CHOOSE_MODEL])
    nakliGPT_model = nakliGPT(config)
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size,
        models_dir="gpt2"
    )
    load_weights_into_gpt(nakliGPT_model, params)
    nakliGPT_model.eval()

    # # loading my trained model
    # nakliGPT_model.load_state_dict(torch.load("nakliGPT_model.pth"))

    
    # load tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # generate text
    generator = nakliGreedySampling(
        model=nakliGPT_model,
        tokenizer=tokenizer,
        num_new_tokens=10,
        max_context_length=config["model"]["context_length"]
    )

    input_text = "Then I suppose no one"
    # generate text
    text = generator.generate_text(
        torch.tensor(tokenizer.encode(input_text)).unsqueeze(0)
    )

    print("\n--- Input Text ---")
    print(input_text)
    print("\n---Original Text---")
    print("Then I suppose no one has ever been there!")
    print("\n--- Generated Text ---")
    print(text)