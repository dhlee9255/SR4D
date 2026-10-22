"""Collect layered-run metrics (fade / plain renders, + co-visible) into rows: python tools/exp_collect.py <run> ..."""
import json
import sys


def get(run, sub):
    d = f"outputs/layered/{run}"; a = json.load(open(f"{d}/args.json")); sp = "test_extrap" if a["setting"] == "r15h35" else "test_interp"
    try:
        E = json.load(open(f"{d}/metrics_{sub}.json"))[sp]
    except Exception:
        return None
    try:
        E.update(json.load(open(f"{d}/metrics_covis_{sub}.json"))[sp])
    except Exception:
        pass
    return E


if __name__ == "__main__":
    for run in sys.argv[1:]:
        for sub in ("renders_both", "renders_fade10_25"):
            E = get(run, sub)
            if E:
                print(f"{run:34s} {sub:18s} psnr {E['psnr']:.2f} fgi {E['fgi_psnr']:.2f} lp {E['fg_lpips']:.4f} oom {E['oom_alpha']:.4f} tof {E['tof']:.2f} cv {E.get('cv_ipsnr', float('nan')):.2f}")
