"""Fused Triton scan: STUB.

TODO: implement a kernel computing G_t = decay * G_{t-1} + S_t over T, fused with the
readout y_t = (G_t q_t) so that G_t is never materialised in HBM. Needs forward + backward.
Until then use scan="chunk" (pure PyTorch) in the config.
"""
import torch


def triton_scan(S: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError(
        "Triton kernel not implemented yet. Set scan: chunk (or loop) in the config."
    )
