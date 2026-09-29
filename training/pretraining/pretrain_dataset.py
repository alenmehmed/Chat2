import json

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTokenDataset(Dataset):
    """Reads a flat, pre-tokenized uint16 token stream (from prepare_pretrain_data.py)
    and slices it into non-overlapping context_length+1 chunks for causal LM training.
    Uses np.memmap so the multi-GB token file never has to fit in RAM."""

    def __init__(self, tokens_path, meta_path, context_length):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.block = context_length + 1
        self.num_samples = meta["num_tokens"] // self.block
        self.tokens = np.memmap(tokens_path, dtype=np.uint16, mode="r", shape=(meta["num_tokens"],))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.block
        chunk = torch.from_numpy(self.tokens[start:start + self.block].astype(np.int64))
        return chunk[:-1], chunk[1:]
