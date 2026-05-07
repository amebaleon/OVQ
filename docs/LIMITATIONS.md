# Limitations

## NF4 Is Still Stronger

The most important result is negative:

> Nexus-OVQ does not beat NF4 in the current experiments.

NF4 remains close to FP16 on the T4 WikiText2 subset run, while mixed OVQ is
still significantly worse.

## Not a Production Runtime

The current VQ/OVQ methods are fake-dequant quality tests:

1. Quantize selected weights.
2. Dequantize them back into PyTorch tensors.
3. Run the model normally.

This does not prove packed model size, VRAM savings, or inference speed.

## Limited Hardware

My local laptop has only an integrated GPU, so I could not run the main
experiments locally. I used Google Colab T4 for the included benchmark results.

## Limited Compute Budget

I wanted to run larger-scale experiments with bigger models, more seeds, larger
WikiText2/C4 subsets, and stronger baselines such as GPTQ/AWQ. I could not do
that yet because of limited compute budget.

## Small-Scale Evaluation

The benchmark uses:

- TinyLlama 1.1B
- Small prompt sets
- WikiText2 subset PPL
- Limited generation samples

These results should not be interpreted as full production evidence.

## Engineering Limitations

The project reflects my current engineering level. I implemented a reproducible
research prototype, but not:

- packed OVQ storage,
- CUDA/Triton kernels,
- full VRAM accounting,
- real deployment infrastructure,
- large-model evaluation.

## Claim Boundary

Supported:

- OVQ reduces collapse compared with standard 2-bit VQ.
- Mixed OVQ improves over pure OVQ.
- Metadata accounting is critical for low-bit methods.

Not supported:

- OVQ beats NF4.
- OVQ is production-ready.
- OVQ is a commercial compression system.
- OVQ works at scale without further testing.
