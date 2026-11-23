# 04 Experiments

**Baselines (parameter-matched within ~5%):** GRU (built-in), causal Transformer (built-in),
and optional Mamba, RWKV, GLA, DeltaNet wrappers.

**Synthetic:** copy, induction heads, associative recall (vary sequence length, vocab, number of pairs).
**Tiny LM:** TinyStories or enwik8 subset; report bits-per-byte / perplexity.
**Ablations:** rank sweep {1,2,4,8,16}; decay sweep (init, scalar vs per-channel; input-gated control is TODO);
hybrid ratio {0, 1/4, 1/2}.

**Protocol:** >=3 seeds, report mean +- std, fixed token budget, same optimiser/schedule per model,
tune learning rate per model on a small grid. Pre-register the primary comparison
(CGAN vs GLA/Mamba on associative recall accuracy at length 512) before looking at results.
