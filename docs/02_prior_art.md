# 02 Prior art (TO BE VERIFIED)

Written from memory, not from a systematic search. Verify every entry and add citations
before relying on it.

| Family | Relation to CGAN | Check |
|---|---|---|
| Linear attention (Katharopoulos et al.) | Running sum of outer products `k v^T`; CGAN's `S_t = a b^T` is the same object | Compare formulations |
| RetNet | Fixed (input-independent) decay on a linear-attention state | Very close; decay is scalar per head there |
| S4/S4D/S5 | Diagonal linear recurrences with input-independent decay | Same decay structure |
| RWKV | Time-decay recurrences | Compare |
| Mamba / selective SSM | Input-dependent decay (CGAN deliberately omits this) | Baseline |
| GLA, DeltaNet | Gated / delta-rule linear attention | Baselines |
| Titans and other memory models | Test-time-updated memory | Compare read/write semantics |
| Influence functions | Motivation for the "sensitivity" analogy only | Not implemented |

**Action items:** run a real literature search (arXiv, Semantic Scholar, Google Scholar),
fill in citations, and rewrite this table with verified differences.
