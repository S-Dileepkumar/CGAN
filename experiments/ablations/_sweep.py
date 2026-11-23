import os, sys, json, copy
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from experiments.train import load_config, run


def sweep(base_cfg_path, key_path, values, tag, seeds=(0, 1, 2), extra=None):
    results = []
    for v in values:
        for s in seeds:
            cfg = load_config(base_cfg_path)
            node = cfg
            ks = key_path.split(".")
            for k in ks[:-1]:
                node = node[k]
            node[ks[-1]] = v
            for k, val in (extra or {}).items():
                cfg["model"][k] = val
            cfg["train"]["seed"] = s
            cfg["train"]["out"] = f"results/logs/{tag}_{str(v).replace('.', 'p')}_s{s}"
            results.append(run(cfg))
    return results
