# CGAN: Causal Gradient Accumulation Network

**Owner:** <YOUR NAME> &nbsp;|&nbsp; **License:** MIT &nbsp;|&nbsp; **Status:** research prototype, *no empirical results yet*

CGAN is a recurrent sequence-model hypothesis: instead of a hidden state `h_t`,
keep a fixed-size, low-rank accumulator

    G_t = diag(decay) * G_{t-1} + S_t,      S_t = a_t b_t^T   (a_t in R^D, b_t in R^r)

where `decay` is a learned, **input-independent** per-channel temporal decay.
The readout is `y_t = (G_t q_t) * x_t`, then an output projection.

## Honest positioning

Mathematically, with this parameterisation CGAN is a **linear-attention-style
recurrence with per-channel, input-independent decay** (cf. RetNet, S4D/S5,
RWKV, GLA). The "gradient / sensitivity" language is a motivating *interpretation*;
nothing in the code computes a gradient or influence function. Whether that
interpretation yields any benefit is exactly what the experiments are meant to test.
Novelty claims require a real literature review; see `docs/02_prior_art.md`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                                   # unit tests (scan equivalence, shapes, causality)
python experiments/train.py --config experiments/configs/copy_cgan.yaml
python experiments/eval.py  --config experiments/configs/copy_cgan.yaml
bash scripts/run_all.sh                     # all synthetic tasks x all models
python scripts/make_report.py               # tables/figures from results/logs
```

## Layout

See `docs/03_architecture.md`. Experiments are specified in `docs/04_experiments.md`;
`docs/05_results.md` stays empty of numbers until real runs exist.

## Known limitations

* Fused Triton kernel (`src/cgan/kernels/`) is a **stub**; the chunked PyTorch scan is the working fast path.
* The readout materialises `G_t` for all `t` (O(B*T*D*r) memory). Fine for prototyping, not for long contexts.
* Third-party baselines (Mamba, RWKV, GLA, DeltaNet) are optional wrappers requiring extra installs and a GPU.
* Verbatim-recall tasks (copy) are expected to be hard for a low-rank O(1) state.
