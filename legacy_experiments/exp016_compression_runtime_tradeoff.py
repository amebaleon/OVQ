from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ovq_local_utils import iter_weight_samples, ovq_dequant, time_call, vq_dequant  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP016 Compression / Runtime Trade-off")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "experiments" / "EXP016_compression_runtime")
    parser.add_argument("--max-eval-values", type=int, default=65536)
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for layer, module, key, flat in iter_weight_samples(args.data_dir, range(args.layers), args.max_eval_values):
        usable = flat.size - (flat.size % 16)
        flat = flat[:usable]
        fp32_bytes = usable * 4
        for bits in (2, 4, 8):
            q_bytes = usable * bits / 8
            scale_bytes = (usable / 16) * 8
            vq_meta = scale_bytes
            ovq_meta = scale_bytes * 2
            _, vq_time = time_call(lambda: vq_dequant(flat, bits), repeats=3)
            _, ovq_time = time_call(lambda: ovq_dequant(flat, bits, overlap_ratio=0.5)[0], repeats=3)
            for method, meta_bytes, elapsed in [
                ("VQ", vq_meta, vq_time),
                ("Nexus-OVQ", ovq_meta, ovq_time),
            ]:
                total = q_bytes + meta_bytes
                rows.append(
                    {
                        "Layer": layer,
                        "Module": module,
                        "Bit": bits,
                        "Method": method,
                        "Values": usable,
                        "FP32 Bytes": f"{fp32_bytes:.0f}",
                        "Compressed Bytes Estimate": f"{total:.0f}",
                        "Compression Ratio vs FP32": f"{fp32_bytes / total:.6f}",
                        "Metadata Bytes Estimate": f"{meta_bytes:.0f}",
                        "Decode Time Sec": f"{elapsed:.9f}",
                        "Tensor Key": key,
                    }
                )

    with (args.output_dir / "exp016_compression_runtime.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for method in ("VQ", "Nexus-OVQ"):
        for bits in (2, 4, 8):
            vals = [r for r in rows if r["Method"] == method and r["Bit"] == bits]
            summary.append(
                {
                    "Method": method,
                    "Bit": bits,
                    "Avg Compression Ratio vs FP32": f"{np.mean([float(r['Compression Ratio vs FP32']) for r in vals]):.6f}",
                    "Avg Decode Time Sec": f"{np.mean([float(r['Decode Time Sec']) for r in vals]):.9f}",
                }
            )
    with (args.output_dir / "exp016_compression_runtime_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    plt.figure(figsize=(8, 5))
    for method in ("VQ", "Nexus-OVQ"):
        vals = [r for r in summary if r["Method"] == method]
        plt.plot([int(r["Bit"]) for r in vals], [float(r["Avg Decode Time Sec"]) for r in vals], marker="o", label=method)
    plt.xlabel("Bit")
    plt.ylabel("Avg Decode Time Sec")
    plt.title("EXP016 CPU Decode Microbenchmark")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "exp016_decode_time.png", dpi=180)
    plt.close()
    (args.output_dir / "exp016_compression_runtime_notes.md").write_text(
        "# EXP016 Compression / Runtime Trade-off\n\nPython prototype latency is not optimized; use this as relative microbenchmark and theoretical size estimate.\n",
        encoding="utf-8",
    )
    print(f"Saved EXP016 to {args.output_dir}")


if __name__ == "__main__":
    main()
