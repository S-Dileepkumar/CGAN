"""Hybrid ratio: insert causal attention every k layers (k=0 means none)."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from _sweep import sweep
sweep("experiments/configs/assoc_recall_cgan.yaml", "model.hybrid_every", [0, 4, 2], "hybrid", extra={"n_layers": 4})
