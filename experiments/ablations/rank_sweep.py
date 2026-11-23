"""Accumulator rank sweep on associative recall."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from _sweep import sweep
sweep("experiments/configs/assoc_recall_cgan.yaml", "model.rank", [1, 2, 4, 8, 16], "rank")
