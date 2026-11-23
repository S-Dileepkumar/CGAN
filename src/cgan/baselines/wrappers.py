"""Optional wrappers for third-party baselines (GPU + extra installs required).

UNTESTED: these were written against the libraries' public APIs from memory and
have not been run. Check each library's current docs and adjust constructor args.
"""
import torch.nn as nn
import torch.nn.functional as F


class _External(nn.Module):
    def __init__(self, cfg, make_layer):
        super().__init__()
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([make_layer(i) for i in range(cfg.n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(cfg.d_model) for _ in range(cfg.n_layers)])
        self.norm_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        x = self.emb(idx)
        for n, l in zip(self.norms, self.layers):
            out = l(n(x))
            x = x + (out[0] if isinstance(out, tuple) else out)
        logits = self.head(self.norm_f(x))
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def build_external_baseline(name: str, cfg):
    try:
        if name == "mamba":
            from mamba_ssm import Mamba
            return _External(cfg, lambda i: Mamba(d_model=cfg.d_model))
        if name == "gla":
            from fla.layers import GatedLinearAttention
            return _External(cfg, lambda i: GatedLinearAttention(hidden_size=cfg.d_model, num_heads=cfg.n_heads))
        if name == "deltanet":
            from fla.layers import DeltaNet
            return _External(cfg, lambda i: DeltaNet(hidden_size=cfg.d_model, num_heads=cfg.n_heads))
        if name == "rwkv":
            from fla.layers import RWKV6Attention
            return _External(cfg, lambda i: RWKV6Attention(hidden_size=cfg.d_model, num_heads=cfg.n_heads))
    except ImportError as e:
        raise ImportError(f"Baseline '{name}' needs extra packages: pip install -e '.[baselines]' ({e})")
    raise ValueError(f"unknown baseline: {name}")
