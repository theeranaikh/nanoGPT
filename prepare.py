import torch
from pathlib import Path 
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")
vocab_size = len(tokenizer)

block_size = 128
batch_size = 16
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2

HF_DATA = "pythainlp/wisesight_sentiment"
SAVE_FILE = Path("data/tokenized_data.pt")
SAVE_FILE.parent.mkdir(parents=True, exist_ok=True)

def _encode(texts):
    sep = tokenizer.sep_token or "\n"
    ids = tokenizer.encode(sep.join(texts), add_special_tokens=False)
    return torch.tensor(ids, dtype=torch.long)

if SAVE_FILE.exists():
    print("The file is loading from the disk.")
    saved = torch.load(SAVE_FILE)
    train_data, val_data = saved["train"], saved["val"]
    print("Successfully loaded.")
else:
    print("Downloading file...")
    dataset = load_dataset(HF_DATA)
    print("encode tokenize")
    train_data = _encode(dataset["train"]["texts"])
    val_data = _encode(dataset["train"]["texts"])
    torch.save({"train": train_data, "val": val_data}, SAVE_FILE)
    print("Successfully")

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size -1 , (batch_size))
    x = torch.stack(d[i : i + block_size] for i in ix)
    y = torch.stack(d[i + 1 : i + block_size + 1 ]for i in ix)
    return x,y
