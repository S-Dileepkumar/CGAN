"""Associative recall: k1 v1 k2 v2 ... kn vn kq -> v_q. Keys unique per sample.
Vocab layout: 0 pad, 1 unused, keys in [2, 2+K), values in [2+K, vocab)."""
import torch


def make_batch(batch, seq_len, vocab, generator=None, device="cpu", n_keys=None, **_):
    n = (seq_len - 1) // 2
    K = n_keys or (vocab - 2) // 2
    assert K >= n, f"need n_keys >= n_pairs ({K} < {n}); raise vocab or lower seq_len"
    keys = 2 + torch.rand(batch, K, generator=generator).argsort(dim=1)[:, :n]       # unique
    vals = (2 + K) + torch.randint(0, vocab - 2 - K, (batch, n), generator=generator)
    q = torch.randint(0, n, (batch,), generator=generator)
    ar = torch.arange(batch)
    body = torch.stack([keys, vals], dim=2).reshape(batch, 2 * n)
    s = torch.cat([body, keys[ar, q].unsqueeze(1)], dim=1)                            # (B, 2n+1)
    pad = seq_len - s.size(1)
    if pad > 0:
        s = torch.cat([torch.zeros(batch, pad, dtype=s.dtype), s], dim=1)
    y = torch.full_like(s, -100)
    y[:, -1] = vals[ar, q]
    return s.to(device), y.to(device)
