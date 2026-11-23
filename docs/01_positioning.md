# 01 Positioning

**Claim we are willing to make today:** CGAN is a simple recurrent architecture with a low-rank,
decay-only accumulator. It is a hypothesis with code.

**Claims we are NOT making:** that it is novel, that it beats SSM/linear-attention baselines,
or that it computes influence functions/gradients.

**What would justify stronger claims:** (1) a systematic literature check, (2) wins against
parameter-matched baselines on recall and LM tasks across seeds, (3) ablations showing the
low-rank accumulator, not incidental components, drives the gain.
