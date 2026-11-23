"""Decay init sweep (per-channel decay, input-independent). An input-gated control is TODO."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from _sweep import sweep
sweep("experiments/configs/assoc_recall_cgan.yaml", "model.decay_init", [0.5, 0.8, 0.9, 0.95, 0.99], "decay")
sweep("experiments/configs/assoc_recall_cgan.yaml", "model.decay_mode", ["scalar"], "decaymode")
