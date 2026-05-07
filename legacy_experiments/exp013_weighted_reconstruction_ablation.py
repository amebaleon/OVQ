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


METHODS = [
    ("uniform", "uniform", False),
    ("triangular", "triangular", False),
    ("center-heavy", "center_heavy", False),
    ("inverse-error", "triangular", True),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP013 Weighted Reconstruction Ablation")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "experiments" / "EXP013_weighted_reconstruction")
    parser.add_argument("--max-eval-values", type=int, default=65536)
    parser.add_argument("--layers", type=int, default=16)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for layer, module, key, flat in iter_weight_samples(args.data_dir, range(args.layers), args.max_eval_values):
        usable = flat.size - (flat.size % 16)
        flat = flat[:usable]
        base = vq_dequant(flat, 2)
        base_mse = mse(flat, base)
        print(f"EXP013 layer={layer:02d} module={module}", flush=True)
        for method, kind, err_weight in METHODS:
            restored = ovq_dequant(flat, 2, overlap_ratio=0.5, weight_kind=kind, error_weighted=err_weight)[0]
            local_mse = mse(flat, restored)
            rows.append(
                {
                    "Layer": layer,
                    "Module": module,
                    "Method": method,
                    "MSE": f"{local_mse:.12f}",
                    "Cosine": f"{cosine(flat, restored):.12f}",
                    "Improvement vs VQ (%)": f"{(base_mse - local_mse) / base_mse * 100.0:.6f}",
                    "Tensor Key": key,
                }
            )

    with (args.output_dir / "exp013_weighted_reconstruction.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for method in [m[0] for m in METHODS]:
        vals = [r for r in rows if r["Method"] == method]
        summary.append(
            {
                "Method": method,
                "Avg MSE": f"{np.mean([float(r['MSE']) for r in vals]):.12f}",
                "Avg Cosine": f"{np.mean([float(r['Cosine']) for r in vals]):.12f}",
                "Avg Improvement vs VQ (%)": f"{np.mean([float(r['Improvement vs VQ (%)']) for r in vals]):.6f}",
            }
        )
    with (args.output_dir / "exp013_weighted_reconstruction_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    plt.figure(figsize=(8, 5))
    plt.bar([r["Method"] for r in summary], [float(r["Avg Improvement vs VQ (%)"]) for r in summary])
    plt.ylabel("Avg Improvement vs VQ (%)")
    plt.title("EXP013 Weighted Overlap Reconstruction")
    plt.tight_layout()
    plt.savefig(args.output_dir / "exp013_weighted_reconstruction.png", dpi=180)
    plt.close()
    best = max(summary, key=lambda r: float(r["Avg Improvement vs VQ (%)"]))
    (args.output_dir / "exp013_weighted_reconstruction_notes.md").write_text(
        f"# EXP013 Weighted Reconstruction Ablation\n\nBest method: {best['Method']} ({best['Avg Improvement vs VQ (%)']}% vs VQ)\n",
        encoding="utf-8",
    )
    print(f"Saved EXP013 to {args.output_dir}")


if __name__ == "__main__":
    main()
