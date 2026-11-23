"""Train one model on one task from a YAML config.

  python experiments/train.py --config experiments/configs/copy_cgan.yaml [--set train.steps=500]
"""
import argparse, json, os, sys, time, copy as pycopy
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))

import torch, yaml
from cgan import CGANConfig
from cgan.baselines import build_model
from data import SYNTHETIC


def load_config(path, overrides=()):
    cfg = yaml.safe_load(open(path))
    for o in overrides:
        k, v = o.split("=", 1)
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(v)
    return cfg


def make_sampler(cfg, device):
    t = cfg["task"]
    if t["name"] == "tiny_lm":
        from data.tiny_lm.load import ByteData
        d = ByteData(t["path"])
        cfg["task"]["vocab"] = d.vocab
        return lambda which, bs, g, seq_len=None: d.batch(which, bs, seq_len or t["seq_len"], g, device)
    fn = SYNTHETIC[t["name"]]
    return lambda which, bs, g, seq_len=None: fn(bs, seq_len or t["seq_len"], t["vocab"], g, device,
                                                 **{k: v for k, v in t.items() if k in ("n_keys",)})


@torch.no_grad()
def evaluate(model, sampler, n_batches, bs, seed=1234, seq_len=None):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    tot_loss = tot_correct = tot_n = 0
    for _ in range(n_batches):
        x, y = sampler("val", bs, g, seq_len)
        logits, loss = model(x, y)
        m = y != -100
        tot_loss += loss.item() * m.sum().item()
        tot_correct += (logits.argmax(-1)[m] == y[m]).sum().item()
        tot_n += m.sum().item()
    model.train()
    return {"loss": tot_loss / tot_n, "acc": tot_correct / tot_n}


def run(cfg):
    tr = cfg["train"]
    device = tr.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(tr["seed"])
    sampler = make_sampler(cfg, device)
    mcfg = CGANConfig.from_dict({**cfg["model"], "vocab_size": cfg["task"]["vocab"],
                                 "max_seq_len": max(cfg["task"]["seq_len"], cfg["model"].get("max_seq_len", 0))})
    model = build_model(cfg["model"].get("name", "cgan"), mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr.get("weight_decay", 0.01))
    out = tr["out"]; os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "train.jsonl"), "w")
    g = torch.Generator().manual_seed(tr["seed"])
    n_params = model.num_params()
    t0 = time.time()
    last = {}
    for step in range(1, tr["steps"] + 1):
        x, y = sampler("train", tr["batch_size"], g)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % tr["eval_every"] == 0 or step == tr["steps"]:
            last = evaluate(model, sampler, tr.get("eval_batches", 20), tr["batch_size"])
            rec = {"step": step, "train_loss": loss.item(), **{f"val_{k}": v for k, v in last.items()},
                   "elapsed_s": time.time() - t0}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(rec)
    torch.save({"model": model.state_dict(), "config": cfg}, os.path.join(out, "model.pt"))
    final = {"name": os.path.basename(out.rstrip("/")), "task": cfg["task"]["name"],
             "model": cfg["model"].get("name", "cgan"), "seed": tr["seed"], "params": n_params,
             "seq_len": cfg["task"]["seq_len"], "rank": cfg["model"].get("rank"), **{f"val_{k}": v for k, v in last.items()}}
    json.dump(final, open(os.path.join(out, "final.json"), "w"), indent=2)
    return final


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides like train.steps=100")
    a = ap.parse_args()
    run(load_config(a.config, a.set))
