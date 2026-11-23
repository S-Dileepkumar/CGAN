"""Decay-only accumulator: G_t = diag(decay) * G_{t-1} + S_t."""
import torch
import torch.nn as nn


def scan_loop(S: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    """Reference scan. S: (B,T,D,R), decay: (D,) in (0,1). Returns G: (B,T,D,R)."""
    B, T, D, R = S.shape
    G = S.new_zeros(B, D, R)
    out = []
    d = decay.view(1, D, 1)
    for t in range(T):
        G = d * G + S[:, t]
        out.append(G)
    return torch.stack(out, dim=1)


def scan_chunk(S: torch.Tensor, decay: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
    """Exact chunked scan. Within a chunk uses a per-channel lower-triangular
    Toeplitz matrix M[d,i,j] = decay_d^(i-j) (i>=j); across chunks carries state.
    Sequential steps: ceil(T / chunk_size)."""
    B, T, D, R = S.shape
    C = min(chunk_size, T)
    pad = (-T) % C
    if pad:
        S = torch.cat([S, S.new_zeros(B, pad, D, R)], dim=1)
    n = S.shape[1] // C
    S = S.view(B, n, C, D, R)

    # powers of decay: (D, C); work in log space for numerical safety
    idx = torch.arange(C, device=S.device)
    diff = (idx.view(C, 1) - idx.view(1, C))                    # (C,C), i-j
    mask = (diff >= 0)
    logd = torch.log(decay.clamp_min(1e-12)).view(D, 1, 1)
    M = torch.exp(logd * diff.clamp_min(0).view(1, C, C)) * mask.view(1, C, C)  # (D,C,C)
    carry_w = torch.exp(logd.view(D, 1) * (idx.view(1, C) + 1))                   # (D,C) decay^(i+1)
    last_w = torch.exp(logd.view(D, 1) * (C - 1 - idx).view(1, C))                # (D,C) decay^(C-1-j)
    full = torch.exp(logd.view(D) * C)                                           # (D,) decay^C

    M = M.to(S.dtype); carry_w = carry_w.to(S.dtype); last_w = last_w.to(S.dtype); full = full.to(S.dtype)

    local = torch.einsum("dij,bnjdr->bnidr", M, S)              # within-chunk contribution
    chunk_end = torch.einsum("dj,bnjdr->bndr", last_w, S)       # chunk's own final state

    G_prev = S.new_zeros(B, D, R)
    outs = []
    for k in range(n):
        g = local[:, k] + carry_w.t().view(1, C, D, 1) * G_prev.unsqueeze(1)
        outs.append(g)
        G_prev = full.view(1, D, 1) * G_prev + chunk_end[:, k]
    G = torch.cat(outs, dim=1)                                  # (B, n*C, D, R)
    return G[:, :T]


class GradientAccumulator(nn.Module):
    def __init__(self, d_model: int, decay_init: float = 0.9, decay_mode: str = "channel",
                 scan: str = "chunk", chunk_size: int = 64):
        super().__init__()
        n = d_model if decay_mode == "channel" else 1
        self.d_model = d_model
        self.decay_logit = nn.Parameter(torch.logit(torch.full((n,), decay_init)))
        self.scan = scan
        self.chunk_size = chunk_size

    def decay(self) -> torch.Tensor:
        d = torch.sigmoid(self.decay_logit)
        return d.expand(self.d_model) if d.numel() == 1 else d

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        d = self.decay()
        if self.scan == "loop":
            return scan_loop(S, d)
        if self.scan == "chunk":
            return scan_chunk(S, d, self.chunk_size)
        if self.scan == "triton":
            from .kernels import triton_scan
            return triton_scan(S, d)
        raise ValueError(f"unknown scan: {self.scan}")
