from google.colab import drive
drive.mount('/content/drive')

import os
import argparse
import torch
from tqdm import tqdm
import sys
import transformers
import pandas as pd
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AdamW, GPT2LMHeadModel
from dataloader import GPTDataLoader
from generater import generate

if sys.version_info < (3, 7):
    raise Exception("This script requires Python 3.7 or higher.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default = "skt/kogpt2-base-v2", type = str)
    parser.add_argument("--data_dir", default = "/content/data_chat.csv", type = str)
    parser.add_argument("--batch_size", default = 1, type = int)
    parser.add_argument("--epochs", default = 1, type = int)
    parser.add_argument("--lr", default = 2e-5, type = float)
    parser.add_argument("--warmup_steps", default = 200, type = int)
    args = parser.parse_args('')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({"pad_token" : "<pad>"})

    base_dir = os.getcwd()
    data_DIR = os.path.join(base_dir, args.data_dir)

    train_dataloader = GPTDataLoader(tokenizer, data_DIR, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(args.model_name).to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr = args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.warmup_steps, num_training_steps = -1)
    min_loss = int(1e-9)

    for epoch in range(args.epochs):
        print(f"Training epoch {epoch}")
        for input_text in tqdm(train_dataloader):
            input_tensor = input_text.to(device)
            outputs = model(input_tensor, labels = input_tensor)
            loss = outputs[0]
            optimizer.zero_grad()
            model.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        print(f"epoch {epoch} loss {outputs[0].item():0.2f}")


        model.save_pretrained("chatf_model.h5")

    print("Training Done!")
