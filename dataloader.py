import pandas as pd
from torch.utils.data import Dataset, DataLoader
class ChatbotDataset(Dataset):
    def __init__(self, tokenizer, file_path):
        data = pd.read_csv(file_path, encoding = "utf-8")
        concats = [label + "|" + text for label, text in zip(data['label'], data['text'])]
        self.items = [tokenizer(item, return_tensors = "pt", padding = "max_length", truncation = True, max_length = 700)["input_ids"] for item in concats]
        self.length = len(self.items)

    def __getitem__(self, i):
        return self.items[i]
    def __len__(self):
        return self.length

def GPTDataLoader(tokenizer, file_path, batch_size):
    data = ChatbotDataset(tokenizer, file_path)
    return DataLoader(data, batch_size = batch_size)
