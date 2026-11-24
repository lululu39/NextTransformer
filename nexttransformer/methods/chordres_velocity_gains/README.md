# ChordRes with independent velocity gains

At every attention and FFN sublayer the original scalar Chord position update produces the new position q. On its tangent plane let u be the projected old velocity and d be the projected realized displacement q-h. Split d into a component parallel to u and its perpendicular remainder. The denominator floor is 1e-12, and the parallel component at u=0 is zero.

```text
v_new = beta * u + gamma_parallel * d_parallel + gamma_perpendicular * d_perpendicular
```

Beta is the original sigmoid retention. The two gammas are independent direct FP32 scalars. All three effective values start at 0.5. There are two trainable gammas at each of the 45 sites whose output velocity is consumed, adding 90 parameters. The terminal FFN's velocity coefficients are fixed at 0.5; its output position goes to the original LM head. With fixed zero initial velocity, the first site's parallel component is structurally zero.

The fused write computes the equivalent P(beta*v + gamma_perpendicular*(q-h)) + (gamma_parallel-gamma_perpendicular)*d_parallel. At initialization this reproduces the baseline write operation order, including its BF16 outputs. Reductions and geometry use FP32. See the implementation and independent forward/backward tests in this repository.
