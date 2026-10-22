"""EXP-009 card table HTML from results/EXP-009/summary.json (mean ± sd over generation seeds; n shown when < 2).
usage: python tools/exp009_card_table.py > results/EXP-009/card_table.html"""
import json

S = json.load(open("results/EXP-009/summary.json"))
METH = ["SR4D", "New-T", "DualGS", "FTGS++ (투명 배경)", "4DGS dynerf", "FTGS++ 공식 검정 (참고)"]
COLS = {"near": [("psnr", "전체 PSNR ↑", 2, 1), ("fgi_psnr", "fg 내부 ↑", 2, 1), ("fg_lpips", "fg LPIPS ↓", 3, -1),
                 ("oom_alpha", "마스크 밖 알파 ↓", 4, -1), ("tof", "tOF ↓", 2, -1), ("jitter", "떨림 mm ↓", 2, -1)],
        "far": [("psnr", "전체 PSNR ↑", 2, 1), ("fgi_psnr", "fg 내부 ↑", 2, 1), ("fg_lpips", "fg LPIPS ↓", 3, -1),
                ("cv_ipsnr", "co-vis fg 내부 ↑", 2, 1), ("oom_alpha", "마스크 밖 알파 ↓", 4, -1), ("tof", "tOF ↓", 2, -1), ("jitter", "떨림 mm ↓", 2, -1)]}
TIT = {("dance", "near"): "dance · 가까운 시점 (gen48_ho8 홀드아웃 8뷰)", ("dance", "far"): "dance · 먼 시점 (r15h35 35° 링 24뷰)",
       ("zju", "near"): "ZJU 387 · 가까운 시점 (gen48_ho8 홀드아웃 8뷰)", ("zju", "far"): "ZJU 387 · 먼 시점 (r15h35 35° 링 24뷰)"}
out = []
for K in ("dance", "zju"):
    for V in ("near", "far"):
        D = S[K][V]; cols = COLS[V]
        best = {}
        for c, _, _, sg in cols:
            v = [D[m][c]["mean"] for m in METH[:-1] if m in D and c in D[m]]
            best[c] = (max if sg > 0 else min)(v) if v else None
        tr = []
        for m in METH:
            if m not in D or not D[m]:
                continue
            tds = ""
            for c, _, p, _ in cols:
                x = D[m].get(c)
                if not x:
                    tds += "<td>–</td>"; continue
                s = f"{x['mean']:.{p}f}" + (f" ± {x['sd']:.{p}f}" if x["sd"] is not None else " (n1)")
                tds += f'<td{" class=best" if m in METH[:-1] and x["mean"] == best[c] else ""}>{s}</td>'
            tr.append(f'<tr{" class=ours" if m == "New-T" else ""}><th scope=row>{m}</th>{tds}</tr>')
        head = "".join(f"<th scope=col>{h}</th>" for _, h, _, _ in cols)
        out.append(f'<p class="ft"><b>{TIT[K, V]}</b></p><div class="tbl"><table><thead><tr><th scope=col>이름</th>{head}</tr></thead><tbody>{"".join(tr)}</tbody></table></div>')
print("".join(out))
