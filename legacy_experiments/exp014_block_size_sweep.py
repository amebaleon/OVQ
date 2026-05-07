from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ovq_local_utils import cosine, iter_weight_samples, mse, ovq_dequant, vq_dequant  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP014 Block Size / Group Size Sweep")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "experiments" / "EXP014_block_size_sweep")
    parser.add_argument("--max-eval-values", type=int, default=65536)
    parser.add_argument("--layers", type=int, default=16)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for layer, module, key, flat in iter_weight_samples(args.data_dir, range(args.layers), args.max_eval_values):
        print(f"EXP014 layer={layer:02d} module={module}", flush=True)
        for bits in (2, 4, 8):
            for block in (8, 16, 32, 64):
                usable = flat.size - (flat.size % block)
                sample = flat[:usable]
                base = vq_dequant(sample, bits, block=block)
                base_mse = mse(sample, base)
                for overlap in (0.0, 0.25, 0.5, 0.75):
                    restored, stride, eff = ovq_dequant(sample, bits, block=block, overlap_ratio=overlap)
                    local_mse = mse(sample, restored)
                    metadata_bits_per_value = (32 * 2) / block
                    rows.append(
                        {
                            "Layer": layer,
                            "Module": module,
                            "Bit": bits,
                            "Block Size": block,
                            "Requested Overlap": overlap,
                            "Effective Overlap": f"{eff:.6f}",
                            "Stride": stride,
                            "MSE": f"{local_mse:.12f}",
                            "Cosine": f"{cosine(sample, restored):.12f}",
                            "Improvement vs VQ (%)": f"{(base_mse - local_mse) / base_mse * 100.0:.6f}",
                            "Scale Metadata Bits/Value": f"{metadata_bits_per_value:.6f}",
                            "Tensor Key": key,
                        }
                    )

    with (args.output_dir / "exp014_block_size_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    keys = sorted({(r["Bit"], r["Block Size"], r["Requested Overlap"]) for r in rows}, key=lambda x: (int(x[0]), int(x[1]), float(x[2])))
    for bit, block, overlap in keys:
        vals = [r for r in rows if r["Bit"] == bit and r["Block Size"] == block and r["Requested Overlap"] == overlap]
        summary.append(
            {
                "Bit": bit,
                "Block Size": block,
                "Requested Overlap": overlap,
                "Avg MSE": f"{np.mean([float(r['MSE']) for r in vals]):.12f}",
                "Avg Cosine": f"{np.mean([float(r['Cosine']) for r in vals]):.12f}",
                "Avg Improvement vs VQ (%)": f"{np.mean([float(r['Improvement vs VQ (%)']) for r in vals]):.6f}",
                "Scale Metadata Bits/Value": vals[0]["Scale Metadata Bits/Value"],
            }
        )
    with (args.output_dir / "exp014_block_size_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    plt.figure(figsize=(10, 5.5))
    for bit in (2, 4, 8):
        vals = [r for r in summary if r["Bit"] == bit and abs(float(r["Requested Overlap"]) - 0.5) < 1e-9]
        vals = sorted(vals, key=lambda r: int(r["Block Size"]))
        plt.plot([int(r["Block Size"]) for r in vals], [float(r["Avg Improvement vs VQ (%)"]) for r in vals], marker="o", label=f"{bit}-bit")
    plt.xlabel("Block Size")
    plt.ylabel("Avg Improvement vs VQ (%) at 50% overlap")
    plt.title("EXP014 Block Size Sweep")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "exp014_block_size_sweep.png", dpi=180)
    plt.close()

    best = max(summary, key=lambda r: float(r["Avg Improvement vs VQ (%)"]))
    (args.output_dir / "exp014_block_size_notes.md").write_text(
        f"# EXP014 Block Size Sweep\n\nBest config: bit={best['Bit']}, block={best['Block Size']}, overlap={best['Requested Overlap']}, improvement={best['Avg Improvement vs VQ (%)']}%.\n",
        encoding="utf-8",
    )
    print(f"Saved EXP014 to {args.output_dir}")


if __name__ == "__main__":
    main()
