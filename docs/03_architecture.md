# 03 Architecture

Per layer, input `x in R^{B x T x D}` after LayerNorm `h`:

1. **Sensitivity encoder:** `a_t = W_a h_t (D)`, `b_t = W_b h_t (r)`, `S_t = a_t b_t^T (D x r)`.
2. **Accumulator:** `G_t = diag(lambda) G_{t-1} + S_t`, `lambda = sigmoid(theta) in (0,1)^D`, input-independent.
3. **Decoder:** `q_t = W_q h_t (r)`, `y_t = W_o ( (G_t q_t) * h_t )`.
4. Residual + FFN.

Equivalent closed form: `y_t[d] = sum_{s<=t} lambda_d^{t-s} a_s[d] (b_s . q_t)`.

## Scan implementations
* `loop`: reference, O(T) Python steps.
* `chunk`: exact parallel-within-chunk scan using per-channel decay Toeplitz matrices; only T/C sequential steps.
* `triton`: fused kernel, **not implemented** (stub).

## Optional hybrid
`hybrid_every: k` inserts a causal softmax-attention block every k layers, used for the hybrid-ratio ablation.
