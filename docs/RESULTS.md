# Results

## T4 NF4 + WikiText2

Device:

- Google Colab Tesla T4
- CUDA memory reported: about 14.56 GB
- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- PPL source: WikiText2 subset

| Method | Hidden cosine | KL vs FP16 | WikiText2 subset PPL | Top5 overlap | Target bpw estimate |
|---|---:|---:|---:|---:|---:|
| FP16 reference | 1.000 | 0.000 | 38.45 | 1.00 | 16.00 |
| bitsandbytes NF4 double quant | 0.899 | 0.036 | 40.96 | 0.80 | 4.50 |
| VQ-2bit | 0.301 | 4.348 | 7497.71 | 0.20 | 4.00 |
| OVQ-2bit group metadata | 0.552 | 1.853 | 578.09 | 0.36 | 4.12 |
| Mixed OVQ top4 group metadata | 0.627 | 0.666 | 148.81 | 0.52 | 4.85 |
| Mixed OVQ top8 group metadata | 0.638 | 0.573 | 123.28 | 0.60 | 5.58 |

Conclusion:

- OVQ improves over standard 2-bit VQ.
- Mixed OVQ improves over pure OVQ.
- NF4 is still far stronger.

## T4 Equal-Budget Check

This run compared per-window metadata with group-shared metadata.

| Method | Hidden cosine | PPL | Target bpw estimate |
|---|---:|---:|---:|
| VQ-2bit window metadata | 0.301 | 2668.63 | 4.00 |
| OVQ-2bit window metadata | 0.552 | 375.41 | 8.00 |
| OVQ-2bit group metadata | 0.552 | 375.41 | 4.12 |
| Mixed OVQ top4 group metadata | 0.627 | 167.57 | 4.85 |

This shows why metadata accounting matters. Without group metadata, OVQ appears
too expensive. With group metadata, it becomes a more defensible low-bit
research direction, but it still does not match NF4 quality.

## Qualitative Generation

The `generation_samples.jsonl` files are included under `results/`.

General pattern:

- VQ-2bit often collapses into whitespace, repeated fragments, or broken tokens.
- OVQ-2bit is less broken but still unstable.
- Mixed OVQ top4/top8 often produces recognizable answers.
- NF4 produces outputs closest to FP16.
