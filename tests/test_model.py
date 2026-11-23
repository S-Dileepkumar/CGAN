import torch
from cgan import CGAN, CGANConfig


def _cfg(**kw):
    base = dict(vocab_size=50, d_model=32, rank=4, n_layers=2, dropout=0.0, chunk_size=8)
    base.update(kw)
    return CGANConfig(**base)


def test_shapes_and_loss():
    m = CGAN(_cfg())
    x = torch.randint(0, 50, (2, 20))
    logits, loss = m(x, x)
    assert logits.shape == (2, 20, 50) and torch.isfinite(loss)


def test_causality():
    for kw in [dict(), dict(hybrid_every=2)]:
        m = CGAN(_cfg(**kw)).eval()
        x = torch.randint(0, 50, (1, 24))
        x2 = x.clone(); x2[0, 15:] = (x2[0, 15:] + 1) % 50
        a, b = m(x), m(x2)
        assert torch.allclose(a[:, :15], b[:, :15], atol=1e-5), kw


def test_scan_modes_agree():
    torch.manual_seed(0)
    x = torch.randint(0, 50, (2, 30))
    m1 = CGAN(_cfg(scan="loop")).eval()
    m2 = CGAN(_cfg(scan="chunk")); m2.load_state_dict(m1.state_dict()); m2.eval()
    assert torch.allclose(m1(x), m2(x), atol=1e-4)
