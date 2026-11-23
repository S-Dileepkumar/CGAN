"""Induction: random sequence containing bigram (A,B); sequence ends with A, target is B."""
import torch


def make_batch(batch, seq_len, vocab, generator=None, device="cpu", **_):
    s = torch.randint(2, vocab, (batch, seq_len), generator=generator)
    p = torch.randint(0, seq_len - 3, (batch,), generator=generator)
    ar = torch.arange(batch)
    A = s[ar, p]
    s[ar, -1] = A
    # remove other occurrences of A so the successor is unambiguous
    other = (s == A.unsqueeze(1))
    other[ar, p] = False
    other[:, -1] = False
    repl = ((A - 2 + 1) % (vocab - 2)) + 2
    s = torch.where(other, repl.unsqueeze(1).expand_as(s), s)
    B = s[ar, p + 1]
    y = torch.full_like(s, -100)
    y[:, -1] = B
    return s.to(device), y.to(device)
