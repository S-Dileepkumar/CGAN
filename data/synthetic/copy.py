"""Copy task: [m_1..m_L] SEP [m_1..m_L]. Loss only on the copied half."""
import torch

PAD, SEP = 0, 1


def make_batch(batch, seq_len, vocab, generator=None, device="cpu", **_):
    L = (seq_len - 1) // 2
    m = torch.randint(2, vocab, (batch, L), generator=generator)
    s = torch.cat([m, torch.full((batch, 1), SEP), m], dim=1)   # (B, 2L+1)
    x, y = s[:, :-1].clone(), s[:, 1:].clone()
    y[:, :L] = -100
    return x.to(device), y.to(device)
