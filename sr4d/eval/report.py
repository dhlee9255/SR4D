"""Markdown comparison table from results/<phase>_<setting>.csv (latest row per method/seq/split wins).

  python -m sr4d.eval.report --csv results/A_real12.csv [--out results/A_real12.md]
"""
import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    latest = {}
    for r in rows:
        latest[(r["method"], r["seq"], r["split"])] = r
    keys = ["psnr", "ssim", "lpips", "fg_psnr", "fg_ssim", "fg_lpips", "temporal", "train_min", "n_gauss"]
    lines = ["| method | seq | split | " + " | ".join(keys) + " |", "|" + "---|" * (3 + len(keys))]
    for (m, s, sp), r in sorted(latest.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        lines.append(f"| {m} | {s} | {sp} | " + " | ".join(r.get(k, "") for k in keys) + " |")
    md = "\n".join(lines)
    print(md)
    out = Path(a.out) if a.out else Path(a.csv).with_suffix(".md")
    out.write_text(md + "\n")


if __name__ == "__main__":
    main()
