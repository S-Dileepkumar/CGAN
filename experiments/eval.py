"""Evaluate a trained checkpoint, optionally at a different sequence length (length generalisation).

  python experiments/eval.py --config experiments/configs/copy_cgan.yaml --seq-len 128
"""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))

import torch
from cgan import CGANConfig
from cgan.baselines import build_model
from experiments.train import load_config, make_sampler, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--seq-len", type=int, default=None)
ap.add_argument("--batches", type=int, default=50)
a = ap.parse_args()

cfg = load_config(a.config)
device = cfg["train"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
sampler = make_sampler(cfg, device)
L = a.seq_len or cfg["task"]["seq_len"]
mcfg = CGANConfig.from_dict({**cfg["model"], "vocab_size": cfg["task"]["vocab"], "max_seq_len": max(L, cfg["task"]["seq_len"])})
model = build_model(cfg["model"].get("name", "cgan"), mcfg).to(device)
ckpt = torch.load(os.path.join(cfg["train"]["out"], "model.pt"), map_location=device)
model.load_state_dict(ckpt["model"])
res = evaluate(model, sampler, a.batches, cfg["train"]["batch_size"], seq_len=a.seq_len)
res.update({"seq_len": L, "config": a.config})
print(json.dumps(res))
with open(os.path.join(cfg["train"]["out"], "eval.jsonl"), "a") as f:
    f.write(json.dumps(res) + "\n")
