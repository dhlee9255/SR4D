"""EXP-009 (D-044) table: methods x (dance, zju) x (near gen48_ho8, far r15h35), mean ± sd over generation seeds 42 / 123.
Writes results/EXP-009/metrics.csv (per seed) and summary.json (mean, sd, n). usage: python tools/exp009_table.py"""
import csv
import json
import math
from pathlib import Path

O = Path("outputs")
M = [("SR4D", lambda T: f"sr4d_final/{T}_nol4", "metrics_renders.json"),
     ("New-T", lambda T: "layered/exp008a_dance_ho8" if T == "dance_gen48_ho8" else f"layered/exp009_{T}", "metrics_renders_both.json"),
     ("DualGS", lambda T: f"dualgs/{T}", "metrics_renders.json"),
     ("FTGS++ (투명 배경)", lambda T: f"ftgspp_transparent/{T}", "metrics_renders.json"),
     ("4DGS dynerf", lambda T: f"4dgs_dynerf/{T}", "metrics_renders.json"),
     ("FTGS++ 공식 검정 (참고)", lambda T: f"ftgspp_black_official/{T}", "metrics_renders.json")]
KEYS = ["psnr", "fgi_psnr", "fg_lpips", "cv_ipsnr", "oom_alpha", "tof", "jitter"]


def one(d, f, sp, T, name):
    try:
        E = json.load(open(O / d / f))[sp]
    except Exception:
        return None
    try:
        E.update(json.load(open(O / d / f.replace("metrics_", "metrics_covis_")))[sp])
    except Exception:
        pass
    if name in ("SR4D", "New-T"):  # New-T jitter = its inner SR4D (outer layer moves on straight lines)
        try:
            E["jitter"] = json.load(open(O / f"sr4d_final/{T}_nol4/jitter.json"))["accel_rms_mm"]
        except Exception:
            pass
    return {k: E.get(k) for k in KEYS if E.get(k) is not None and E.get(k) == E.get(k)}


rows, summ = [], {}
for K in ("dance", "zju"):
    for S, V in (("gen48_ho8", "near"), ("r15h35", "far")):
        sp = "test_extrap" if S == "r15h35" else "test_interp"
        for name, dfn, f in M:
            vals = []
            for X, seed in (("", 42), ("_s123", 123)):
                T = f"{K}_{S}{X}"; E = one(dfn(T), f, sp, T, name)
                if E is None:
                    continue
                if V == "near":
                    E.pop("cv_ipsnr", None)
                rows.append({"method": name, "data": K, "view": V, "gen_seed": seed, "dir": dfn(T), **E}); vals.append(E)
            s = {}
            for k in KEYS:
                xs = [v[k] for v in vals if k in v]
                if xs:
                    mu = sum(xs) / len(xs); sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else None
                    s[k] = {"mean": mu, "sd": sd, "n": len(xs)}
            summ.setdefault(K, {}).setdefault(V, {})[name] = s
Path("results/EXP-009").mkdir(parents=True, exist_ok=True)
with open("results/EXP-009/metrics.csv", "w") as fh:
    w = csv.DictWriter(fh, ["method", "data", "view", "gen_seed", "dir"] + KEYS); w.writeheader(); w.writerows(rows)
json.dump(summ, open("results/EXP-009/summary.json", "w"), ensure_ascii=False, indent=1)
for K in summ:
    for V in summ[K]:
        print(f"== {K} {V}")
        for n, s in summ[K][V].items():
            print(f"  {n:22s}", " ".join(f"{k}={s[k]['mean']:.3f}±{s[k]['sd'] if s[k]['sd'] is not None else float('nan'):.3f}(n{s[k]['n']})" for k in s))
