import torch
from cgan.accumulator import scan_loop, scan_chunk


def test_chunk_matches_loop():
    torch.manual_seed(0)
    for T, C in [(37, 8), (64, 16), (5, 64), (128, 32)]:
        S = torch.randn(2, T, 6, 3, dtype=torch.float64)
        d = torch.sigmoid(torch.randn(6, dtype=torch.float64))
        assert torch.allclose(scan_loop(S, d), scan_chunk(S, d, C), atol=1e-9), (T, C)


def test_chunk_grad():
    S = torch.randn(1, 20, 4, 2, dtype=torch.float64, requires_grad=True)
    d = torch.full((4,), 0.9, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda s, dd: scan_chunk(s, dd, 8), (S, d))
