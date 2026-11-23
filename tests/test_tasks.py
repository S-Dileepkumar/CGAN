import torch
from data import SYNTHETIC


def test_copy_targets():
    x, y = SYNTHETIC["copy"](4, 21, 16)
    L = 10
    assert x.shape == y.shape == (4, 2 * L)
    assert (y[:, :L] == -100).all()
    assert torch.equal(y[:, L:], x[:, :L])


def test_induction_unambiguous():
    x, y = SYNTHETIC["induction"](64, 40, 20)
    for xi, yi in zip(x, y):
        A = xi[-1]
        pos = (xi[:-1] == A).nonzero().flatten()
        assert len(pos) == 1 and xi[pos[0] + 1] == yi[-1]


def test_assoc_recall():
    x, y = SYNTHETIC["assoc_recall"](32, 33, 64)
    for xi, yi in zip(x, y):
        k = xi[-1]
        body = xi[:-1].view(-1, 2)
        i = (body[:, 0] == k).nonzero().flatten()
        assert len(i) == 1 and body[i[0], 1] == yi[-1]
