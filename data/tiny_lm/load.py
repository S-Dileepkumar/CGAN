"""Byte-level LM data from a local text file (e.g. an enwik8 or TinyStories subset).

Download is NOT automated here. Put a file at data/tiny_lm/train.txt, e.g.:
  enwik8:       http://mattmahoney.net/dc/enwik8.zip  (unzip, take first N MB)
  TinyStories:  https://huggingface.co/datasets/roneneldan/TinyStories
Verify sources yourself before use.
"""
import numpy as np
import torch


class ByteData:
    def __init__(self, path, split=0.9):
        arr = np.fromfile(path, dtype=np.uint8)
        n = int(len(arr) * split)
        self.train, self.val = torch.from_numpy(arr[:n].astype(np.int64)), torch.from_numpy(arr[n:].astype(np.int64))
        self.vocab = 256

    def batch(self, which, batch, seq_len, generator=None, device="cpu"):
        d = self.train if which == "train" else self.val
        ix = torch.randint(0, len(d) - seq_len - 1, (batch,), generator=generator)
        x = torch.stack([d[i:i + seq_len] for i in ix])
        y = torch.stack([d[i + 1:i + seq_len + 1] for i in ix])
        return x.to(device), y.to(device)
