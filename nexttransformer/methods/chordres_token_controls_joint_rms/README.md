# Token-dependent ChordRes with a joint gate RMSNorm

The gate for each token and sublayer reads the position and velocity using one shared scale. For width D and epsilon 1e-6:

```text
s = rsqrt((sum(h²) + sum(v²)) / (2D) + eps)
raw_controls = bias + s * (W_h h + W_v v)
```

This is weightless RMSNorm over the concatenated pair. The Triton read computes the sum in FP32 directly from the two BF16 states. The physical lookahead and tangent write still use the original h and v. The backward uses one radial projection over both gate inputs. Its tiled reduction uses the same inverse RMS for both weight gradients and preserves unscaled bias gradients. No concatenated activation is materialized. The 46 sites, control parameterizations and initialization match the [velocity-only control](../chordres_token_controls_vnorm/).
