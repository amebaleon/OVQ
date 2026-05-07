# Glossary

## Nexus-OVQ

Nexus Overlapping Vector Quantization. In this project, it means blockwise
weight quantization with overlapping reconstruction windows. Multiple restored
windows contribute to the same weight positions, and the final value is blended.

## OVQ

Overlapping Vector Quantization. The short name for the method tested here.

## VQ

Vector Quantization. In this repository, VQ usually means a simple non-overlap
blockwise quantization baseline using per-block min/max reconstruction.

## 2-bit

A payload setting where the quantized index for a value uses 2 bits. This is not
always the same as the final storage cost. Metadata, overlap, scales, and
unquantized parameters increase effective bits per weight.

## bpw

Bits per weight. This project reports estimated bpw because VQ/OVQ are evaluated
with fake-dequantized PyTorch weights rather than a packed file format.

## Target bpw

Estimated bits per weight for the targeted linear modules only. This excludes
non-target model parameters that remain in FP16 in the prototype.

## Total model bpw

Estimated bits per weight across the whole loaded model, including non-target
parameters.

## Group metadata

A storage estimate where scale/min-max metadata is shared across a larger group
instead of being stored for every overlap window. This is a compact-format
hypothesis, not a finished packed runtime.

## NF4

NormalFloat 4-bit quantization, commonly used through bitsandbytes/QLoRA-style
4-bit loading. In the benchmark, NF4 is the strong 4-bit baseline.

## Hidden cosine

Cosine similarity between FP16 hidden states and quantized-model hidden states.
It is an internal preservation proxy, not proof of final answer quality.

## KL vs FP16

KL divergence between the FP16 output distribution and the quantized output
distribution for fixed prompts.

## PPL

Perplexity. Lower is better. This project includes WikiText2 subset PPL, but not
a full large-scale benchmark.

## Fake-dequant runtime

The current VQ/OVQ implementation quantizes and then dequantizes weights back
into normal PyTorch tensors before running the model. It measures quality but
does not prove deployment speed or packed memory savings.
