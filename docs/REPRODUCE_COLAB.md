# Reproduce on Google Colab

I used Google Colab T4 because my local laptop has only an integrated GPU.

## 1. Set Runtime

In Colab:

```text
Runtime -> Change runtime type -> GPU -> T4
```

Check:

```bash
!nvidia-smi
```

## 2. Upload Repository Zip

If using a zip, upload it:

```python
from google.colab import files
uploaded = files.upload()
```

Then unzip:

```bash
!rm -rf /content/nexus-ovq
!unzip -q nexus-ovq.zip -d /content/nexus-ovq
%cd /content/nexus-ovq/benchmark
```

If cloned from GitHub:

```bash
!git clone <repo-url> /content/nexus-ovq
%cd /content/nexus-ovq/benchmark
```

## 3. Install

```bash
!pip install -q -r requirements.txt
```

## 4. Run NF4 + WikiText2 Benchmark

```bash
!python run_benchmark.py --config configs/t4_nf4_wikitext.json
```

## 5. Zip Results

```bash
!zip -r /content/nexus_ovq_t4_nf4_wikitext_outputs.zip outputs/t4_nf4_wikitext
```

## 6. Download

```python
from google.colab import files
files.download("/content/nexus_ovq_t4_nf4_wikitext_outputs.zip")
```

## Notes

- NF4 requires bitsandbytes and a CUDA runtime.
- VQ/OVQ throughput is not production throughput because the implementation is
  fake-dequantized.
- If Colab gives a different GPU, record it in your results.
