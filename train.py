import argparse
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import re
from dataloader import fundchatdataset

Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = "</s>"
EOS = "</s>"
PAD = "<pad>"
MASK = "<unused0>"
SENT = "<unused1>"


def collate_batch(batch):
    data = [torch.tensor(item[0]).to(device) for item in batch]
    mask = [torch.tensor(item[1]).to(device) for item in batch]
    return torch.stack(data), torch.stack(mask)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default = 10, type = int)
parser.add_argument("--lr", default = 0.002, type = float)
parser.add_argument("--batch_size", default = 32, type = int)
parser.add_argument("--warmup_steps", default = 200, type = int)
args = parser.parse_args('')

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token = BOS, eos_token = EOS, unk_token = "<unk>", pad_token = PAD, mask_token = MASK)
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

train_data = pd.read_csv('/content/drive/MyDrive/gdsc/chat_train.csv')
test_data = pd.read_csv('/content/drive/MyDrive/gdsc/chat_test.csv')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_set = fundchatdataset(test_data, max_len = 100)
train_dataloader = DataLoader(train_set, batch_size = args.batch_size, num_workers = 0, shuffle = True, collate_fn = collate_batch,)

model.to(device)
model.train()

criterion = torch.nn.CrossEntropyLoss(reduction = "none")
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
Sneg = -1e18

print("start")
for epoch in range(args.epochs):
    print(f"Training epoch {epoch}")
    for batch_idx, samples in enumerate(train_dataloader):
        optimizer.zero_grad()
        token_ids, mask = samples
        out = model(token_ids)
        out = out.logits
        mask_3d = mask.unsqueeze(dim = 2).repeat_interleave(repeats = out.shape[2], dim = 2)
        mask_out = torch.where(mask_3d == 1, out, Sneg * torch.ones_like(out))
        loss = criterion(mask_out.transpose(1, 2), token_ids)
        avg_loss = loss.sum() / mask.sum()
        avg_loss.backward()
        optimizer.step()

    print(f"epoch {epoch} loss {avg_loss}")

    torch.save(model, f'./kogpt2_ft_model.pt')
print("end")
