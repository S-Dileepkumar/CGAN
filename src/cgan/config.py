from dataclasses import dataclass, field, asdict


@dataclass
class CGANConfig:
    vocab_size: int = 32000
    d_model: int = 256
    rank: int = 8                 # width r of the accumulator (G_t is D x r)
    n_layers: int = 4
    decay_init: float = 0.9       # initial per-channel temporal decay
    decay_mode: str = "channel"   # "channel" | "scalar"
    scan: str = "chunk"           # "loop" | "chunk" | "triton"
    chunk_size: int = 64
    hybrid_every: int = 0         # 0 = pure CGAN; k>0 inserts causal attention every k layers
    n_heads: int = 4              # used only by hybrid attention blocks
    dropout: float = 0.1
    max_seq_len: int = 2048
    tie_weights: bool = True

    @classmethod
    def from_dict(cls, d):
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})

    def to_dict(self):
        return asdict(self)
