import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CGANConfig
from .accumulator import GradientAccumulator


class SensitivityEncoder(nn.Module):
    """S_t = a_t b_t^T with a_t in R^D, b_t in R^r."""
    def __init__(self, cfg: CGANConfig):
        super().__init__()
        self.a = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.b = nn.Linear(cfg.d_model, cfg.rank, bias=False)

    def forward(self, x):                       # (B,T,D)
        a, b = self.a(x), self.b(x)
        return torch.einsum("btd,btr->btdr", a, b)


class ApproximateDecoder(nn.Module):
    """y_t = W_o((G_t q_t) * x_t)."""
    def __init__(self, cfg: CGANConfig):
        super().__init__()
        self.q = nn.Linear(cfg.d_model, cfg.rank, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, G, x):                    # G: (B,T,D,R), x: (B,T,D)
        q = self.q(x)
        g = torch.einsum("btdr,btr->btd", G, q)
        return self.out(g * x)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: CGANConfig):
        super().__init__()
        self.h = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class FFN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        return self.net(x)


class CGANBlock(nn.Module):
    def __init__(self, cfg: CGANConfig, use_attention: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.use_attention = use_attention
        if use_attention:
            self.mixer = CausalSelfAttention(cfg)
        else:
            self.sens = SensitivityEncoder(cfg)
            self.acc = GradientAccumulator(cfg.d_model, cfg.decay_init, cfg.decay_mode,
                                           cfg.scan, cfg.chunk_size)
            self.dec = ApproximateDecoder(cfg)
        self.ffn = FFN(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        h = self.norm1(x)
        if self.use_attention:
            m = self.mixer(h)
        else:
            m = self.dec(self.acc(self.sens(h)), h)
        x = x + self.drop(m)
        return x + self.drop(self.ffn(self.norm2(x)))


class CGAN(nn.Module):
    def __init__(self, cfg: CGANConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        blocks = []
        for i in range(cfg.n_layers):
            attn = cfg.hybrid_every > 0 and (i + 1) % cfg.hybrid_every == 0
            blocks.append(CGANBlock(cfg, use_attention=attn))
        self.blocks = nn.ModuleList(blocks)
        self.norm_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.norm_f(x))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
