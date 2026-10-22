"""PSNR vs. nearest-training-view angle curve across methods (CLAUDE.md 8).

  python -m sr4d.eval.nearest_view --metrics outputs/*/387/real12/metrics_renders.json --labels ... --out results/figures/angle_real12.png
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plt.figure(figsize=(5, 3.5))
    for path, lab in zip(a.metrics, a.labels):
        per = json.loads(Path(path).read_text())
        xs, ys = [], []
        for split, r in per.items():
            for cid, c in r["per_cam"].items():
                xs.append(c["nearest_angle"])
                ys.append(c["psnr"])
        plt.scatter(xs, ys, label=lab, s=18)
    plt.xlabel("angle to nearest training camera (deg)")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(a.out, dpi=150)
    print("saved", a.out)


if __name__ == "__main__":
    main()
