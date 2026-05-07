# T4 NF4 + WikiText2 Analysis

Imported result:

- Source zip: `outputs/imported_zips/nexus_ovq_due_diligence_t4_nf4_wikitext_outputs_20260507.zip`
- Device: Tesla T4
- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- PPL source: WikiText2 subset
- Runtime label: fake-dequant runtime for VQ/OVQ methods; real bitsandbytes runtime for NF4

## Summary

| Method | Final hidden cosine | KL vs ref | WikiText2 subset PPL | Top5 overlap | Total model bpw est | Target-only bpw est |
|---|---:|---:|---:|---:|---:|---:|
| FP16 reference | 1.000 | 0.000 | 38.45 | 1.00 | 16.00 | 16.00 |
| bitsandbytes NF4 double quant | 0.899 | 0.036 | 40.96 | 0.80 | 4.50 | 4.50 |
| VQ-2bit window meta | 0.301 | 4.348 | 7497.71 | 0.20 | 5.43 | 4.00 |
| OVQ-2bit group meta | 0.552 | 1.853 | 578.09 | 0.36 | 5.54 | 4.12 |
| Mixed OVQ top4 group meta | 0.627 | 0.666 | 148.81 | 0.52 | 6.18 | 4.85 |
| Mixed OVQ top8 group meta | 0.638 | 0.573 | 123.28 | 0.60 | 6.82 | 5.58 |

## Interpretation

The critic's main objection is confirmed:

> Mixed OVQ is much better than pure 2-bit VQ/OVQ, but it is not competitive
> with a standard 4-bit NF4 baseline yet.

NF4 remains close to FP16 on this run:

- PPL: 38.45 -> 40.96
- Hidden cosine: 0.899
- KL: 0.036

Mixed OVQ top8 improves visibly over VQ-2bit and OVQ-2bit:

- VQ-2bit PPL: 7497.71
- OVQ-2bit PPL: 578.09
- Mixed top8 PPL: 123.28

However, mixed top8 is still far from NF4:

- NF4 PPL: 40.96
- Mixed top8 PPL: 123.28
- NF4 hidden cosine: 0.899
- Mixed top8 hidden cosine: 0.638

## Current Claim Boundary

Supported:

- OVQ and mixed OVQ reduce collapse compared with VQ-2bit.
- Group metadata makes the budget more defensible than per-window metadata.
- Mixed top8 produces the best OVQ-family result in this T4 run.

Not supported:

- Mixed OVQ is competitive with NF4.
- Mixed OVQ preserves LLM quality near 4-bit baselines.
- Mixed OVQ is production-ready.

## Next Technical Direction

Do not proceed to company or patent-strength claims yet. The next research
target is quality recovery:

1. Improve protected layer/module selection with measured sensitivity rather
   than the fixed EXP009 ranking.
2. Try module-level protection instead of layer-only protection.
3. Add activation-aware scaling to mixed OVQ.
4. Run top12/top16 or protected-module variants to understand the ceiling.
5. Only after quality approaches NF4 should packed runtime work become the main
   bottleneck.
