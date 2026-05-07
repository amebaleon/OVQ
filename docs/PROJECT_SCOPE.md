# Project Scope

## What This Project Is

This is a student research project on extreme low-bit LLM quantization.

The central question:

> Can overlapping block reconstruction reduce the collapse caused by standard
> 2-bit VQ?

Current answer:

> Yes, it reduces collapse compared with standard 2-bit VQ. No, it does not yet
> match mature 4-bit methods such as NF4.

## What This Project Is Not

This project is not:

- a SOTA quantization method,
- a production inference engine,
- a commercial compression product,
- a replacement for NF4/GPTQ/AWQ,
- a patent or startup claim.

## Why It Is Still Useful

The project is useful because it shows:

- how badly simple 2-bit VQ can collapse,
- that overlap can partially reduce collapse,
- that mixed precision can improve recovery,
- that NF4 remains a much stronger baseline,
- that honest metadata accounting changes the interpretation of low-bit claims.

## Future Work

Possible next steps:

- NF4-style nonuniform 2-bit codebooks,
- activation-aware OVQ,
- module-level sensitivity selection,
- packed OVQ format,
- CUDA/Triton runtime,
- GPTQ/AWQ comparisons,
- larger models and more seeds.
