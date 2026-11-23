"""Dependency-free baselines that run anywhere."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _LMBase(nn.Module):
    def _loss(self, logits, targets):
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class GRULM(_LMBase):
    def __init__(self, cfg):
        super().__init__()
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rnn = nn.GRU(cfg.d_model, cfg.d_model, num_layers=cfg.n_layers, batch_first=True, dropout=cfg.dropout)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        h, _ = self.rnn(self.emb(idx))
        return self._loss(self.head(h), targets)


class TransformerLM(_LMBase):
    def __init__(self, cfg):
        super().__init__()
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, 4 * cfg.d_model, cfg.dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, cfg.n_layers)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        T = idx.size(1)
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), 1)
        return self._loss(self.head(self.enc(x, mask=mask)), targets)
