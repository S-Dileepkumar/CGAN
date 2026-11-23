from .simple import GRULM, TransformerLM
from .wrappers import build_external_baseline


def build_model(name: str, cfg):
    """name: cgan | gru | transformer | mamba | rwkv | gla | deltanet"""
    if name == "cgan":
        from ..model import CGAN
        return CGAN(cfg)
    if name == "gru":
        return GRULM(cfg)
    if name == "transformer":
        return TransformerLM(cfg)
    return build_external_baseline(name, cfg)
