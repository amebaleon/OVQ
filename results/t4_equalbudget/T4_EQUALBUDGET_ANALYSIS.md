# T4 Equal-Budget Analysis

Imported result:

- Source zip: `outputs/imported_zips/nexus_ovq_due_diligence_t4_equalbudget_outputs_20260506.zip`
- Device: Tesla T4
- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Runtime label: fake-dequant runtime for VQ/OVQ methods

## Summary

| Method | Final hidden cosine | KL vs ref | PPL | Top5 overlap | Total model bpw est | Target-only bpw est | Metadata mode |
|---|---:|---:|---:|---:|---:|---:|---|
| FP16 reference | 1.000 | 0.000 | 71.36 | 1.00 | 16.00 | 16.00 | window |
| VQ-2bit window meta | 0.301 | 4.348 | 2668.63 | 0.20 | 5.43 | 4.00 | window |
| OVQ-2bit window meta | 0.552 | 1.853 | 375.41 | 0.36 | 8.95 | 8.00 | window |
| OVQ-2bit group meta | 0.552 | 1.853 | 375.41 | 0.36 | 5.54 | 4.12 | group |
| Mixed OVQ top2 group meta | 0.581 | 1.067 | 223.52 | 0.44 | 5.86 | 4.49 | group |
| Mixed OVQ top4 group meta | 0.627 | 0.666 | 167.57 | 0.52 | 6.18 | 4.85 | group |

## Interpretation

This run fixes the largest issue found in the previous T4 smoke test:

> With a group-shared metadata estimate, OVQ-2bit target-only bpw drops from
> about 8.00 to about 4.12 while preserving the same fake-dequant quality
> metrics.

The strongest current signal is:

> Mixed OVQ top4 group-meta reaches much better behavior preservation than
> VQ-2bit at a target-only bpw estimate below 5.0.

This is not a production deployment result. It is a compact-format quality
hypothesis because the current implementation still runs dequantized PyTorch
weights rather than a packed OVQ kernel.

## Claim Status

Supported:

- OVQ-2bit improves hidden cosine, KL, PPL, and generation collapse versus
  VQ-2bit on the T4 smoke setting.
- Group-shared metadata makes the OVQ budget much more defensible than
  per-overlap-window metadata.
- Mixed 4/2-bit OVQ improves quality further at target-only bpw estimates
  around 4.5-4.9.

Not yet supported:

- Faster runtime than 4-bit baselines.
- Better quality than NF4/GPTQ/AWQ.
- Production-ready low-bit inference.

## Next Step

Run L4 main with group-metadata methods and add an NF4 baseline adapter.
The immediate business-relevant question is whether mixed OVQ can approach
NF4/GPTQ/AWQ quality while staying below or near their effective model size.
