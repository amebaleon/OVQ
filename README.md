# Nexus-OVQ

Nexus-OVQ is a student research project exploring whether overlapping block
reconstruction can reduce collapse in extreme low-bit LLM weight quantization.

This is **not** a SOTA claim. Nexus-OVQ does **not** beat mature 4-bit baselines
such as NF4 in the current experiments. The main finding is narrower:

> Overlapping reconstruction reduces the collapse observed in standard 2-bit VQ,
> but the current method is still far from production-ready 4-bit quantization.

## Why I Built This

I wanted to understand whether a 2-bit-oriented quantization method could keep a
language model usable by reducing block boundary error. I developed Nexus-OVQ,
ran multiple reconstruction and behavior-preservation experiments, and then
tested it against a real NF4 baseline.

The result was mixed:

- Nexus-OVQ improved over standard 2-bit VQ.
- Mixed 4/2-bit Nexus-OVQ improved further.
- NF4 remained much stronger.

I am sharing the code, results, limitations, and failure cases because the
failure itself is informative.

## Key Result

T4, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, WikiText2 subset:

| Method | Hidden cosine | KL vs FP16 | WikiText2 subset PPL | Top5 overlap | Target bpw estimate |
|---|---:|---:|---:|---:|---:|
| FP16 reference | 1.000 | 0.000 | 38.45 | 1.00 | 16.00 |
| bitsandbytes NF4 double quant | 0.899 | 0.036 | 40.96 | 0.80 | 4.50 |
| VQ-2bit | 0.301 | 4.348 | 7497.71 | 0.20 | 4.00 |
| OVQ-2bit group metadata | 0.552 | 1.853 | 578.09 | 0.36 | 4.12 |
| Mixed OVQ top4 group metadata | 0.627 | 0.666 | 148.81 | 0.52 | 4.85 |
| Mixed OVQ top8 group metadata | 0.638 | 0.573 | 123.28 | 0.60 | 5.58 |

Interpretation:

- Standard 2-bit VQ collapses badly.
- OVQ reduces that collapse.
- Mixed OVQ reduces it further.
- NF4 remains much closer to FP16 and is still the stronger method.

## Repository Structure

```text
nexus-ovq/
  README.md
  benchmark/
    run_benchmark.py
    requirements.txt
    configs/
      t4_smoke.json
      t4_equalbudget.json
      t4_nf4_wikitext.json
      l4_main.json
      l4_nf4_wikitext.json
  docs/
    GLOSSARY.md
    RESULTS.md
    LIMITATIONS.md
    REPRODUCE_COLAB.md
    PROJECT_SCOPE.md
  results/
    t4_equalbudget/
    t4_nf4_wikitext/
  legacy_experiments/
    ovq_local_utils.py
    exp013_weighted_reconstruction_ablation.py
    exp014_block_size_sweep.py
    exp016_compression_runtime_tradeoff.py
```

## Reproduce

I ran the main experiments on Google Colab T4 because my local laptop has only
an integrated GPU and cannot run these experiments reliably.

```bash
cd benchmark
pip install -r requirements.txt
python run_benchmark.py --config configs/t4_nf4_wikitext.json
```

See [docs/REPRODUCE_COLAB.md](docs/REPRODUCE_COLAB.md) for a clean Colab
workflow.

## Limitations

- Nexus-OVQ does not outperform NF4.
- The VQ/OVQ benchmark uses fake-dequantized PyTorch weights, not a packed
  runtime.
- The PPL evaluation is a subset benchmark, not a full production evaluation.
- I could not run larger-scale experiments because of limited compute budget.
- My current implementation is research code, not deployment infrastructure.

See [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for the full limitation list.

## Personal Note

I am still learning. This project was built with limited compute, limited local
hardware, and a lot of trial and error. I am publishing it because the results
show both a real improvement over standard 2-bit VQ and a clear failure to match
NF4. Feedback is welcome, especially from people working on LLM quantization.

## Suggested Citation

This is not a peer-reviewed paper. If you reference it, please cite it as a
student research project and link to the repository.
