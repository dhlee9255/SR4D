"""'현재 비교 (10/15)' section at the top of the artifact page (docs/gallery/pub/sr4d_renders.html).
Tables from outputs/*/metrics_renders.json (+ jitter.json); figures from docs/gallery/pub/c1015/<k>/ (tools/cmp_figure.py).
usage: python docs/gallery/c1015_section.py  (re-run after adding rows; the section between <!--C1015--> markers is replaced)
"""
import json
import re
from pathlib import Path

S = Path("/data/AI-Graphics/Applications/SR4D/outputs"); G = Path(__file__).parent; H = G / "pub/sr4d_renders.html"
ROWS = {
    "dance": [("SR4D L4 끔 (기준선, 56k)", "sr4d_final/dance_gen48_ho8_nol4"), ("SR4D L4 끔 2배 (112k)", "sr4d_final/dance_gen48_ho8_nol4x2"),
              ("+ 디테일 층 (a)", "sr4d_final/dance_gen48_ho8_nol4_s7a"), ("+ 디테일 층 (b)", "sr4d_final/dance_gen48_ho8_nol4_s7b"),
              ("+ 디테일 층 (c) 평활 10배", "sr4d_final/dance_gen48_ho8_nol4_s7c"), ("+ 디테일 층 (d) Δ색만", "sr4d_final/dance_gen48_ho8_nol4_s7d"),
              ("DualGS", "dualgs/dance_gen48_ho8"), ("4DGS dynerf (알파 없음)", "4dgs_dynerf/dance_gen48_ho8"),
              ("FTGS++ 검정 배경", "ftgspp_black_official/dance_gen48_ho8"), ("FTGS++ 흰 배경", "ftgspp_white_official/dance_gen48_ho8")],
    "zju": [("SR4D 기준선", "sr4d_final/zju_gen48_ho8"), ("SR4D cam0", "sr4d_final/zju_gen48_ho8_cam0"), ("DualGS", "dualgs/zju_gen48_ho8"),
            ("4DGS dynerf (알파 없음)", "4dgs_dynerf/zju_gen48_ho8"), ("FTGS++ 검정 배경", "ftgspp_black_official/zju_gen48_ho8"),
            ("FTGS++ 흰 배경", "ftgspp_white_official/zju_gen48_ho8")],
}
COLS = [("psnr", "전체 PSNR ↑", 2, 1), ("fg_psnr", "fg PSNR (마스크 전체) ↑", 2, 1), ("fgi_psnr", "fg 내부 PSNR ↑", 2, 1),
        ("fg_lpips", "fg LPIPS ↓", 4, -1), ("tof", "tOF (px) ↓", 2, -1), ("jit", "떨림 (mm) ↓", 2, -1)]
NAMES = {"dance": "dance", "zju": "ZJU 생성뷰"}


def J(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def table(k):
    data = []
    for n, d in ROWS[k]:
        E = (J(S / d / "metrics_renders.json") or {}).get("test_interp")
        if E is None:
            continue
        E = dict(E); E["jit"] = (J(S / d / "jitter.json") or {}).get("accel_rms_mm"); data.append((n, E))
    best = {c: (max if sg > 0 else min)([E[c] for _, E in data if E.get(c) is not None]) for c, _, _, sg in COLS}
    tr = "".join("<tr><td>" + n + "</td>" + "".join(
        f'<td{" class=hi" if E.get(c) is not None and E[c] == best[c] else ""}>{"–" if E.get(c) is None else f"{E[c]:.{p}f}"}</td>'
        for c, _, p, _ in COLS) + "</tr>" for n, E in data)
    return "<div class=tbl><table><tr><th>방법</th>" + "".join(f"<th>{h}</th>" for _, h, _, _ in COLS) + "</tr>" + tr + "</table></div>"


def gallery(k, grp, title, cap):
    idx = J(G / f"pub/c1015/{k}/index.json") or {}
    ims = idx.get(grp, [])
    if not ims:
        return ""
    btn = "".join(f'<button type="button" data-src="c1015/{k}/{f}" aria-pressed="{str(i == 0).lower()}">{f.split("_")[1][:-4]}</button>'
                  for i, f in enumerate(ims))
    gid = f"cg_{k}_{grp}"
    return f'''<h3 style="margin:8px 0 0">{title}</h3><div class="chips" id="{gid}">{btn}</div>
<figure style="margin:0;display:flex;flex-direction:column;gap:4px"><div style="overflow-x:auto"><img id="{gid}_img" src="c1015/{k}/{ims[0]}" alt="{title}" loading="lazy" style="max-width:none;width:1600px;border-radius:4px;background:#fff"></div>
<figcaption class="num">{cap}</figcaption></figure>
<script>(()=>{{const c=document.getElementById('{gid}'),im=document.getElementById('{gid}_img');c.querySelectorAll('button').forEach(b=>b.onclick=()=>{{c.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed','false'));b.setAttribute('aria-pressed','true');im.src=b.dataset.src}})}})();</script>'''


def orbits(k):
    idx = J(G / f"pub/c1015/{k}/index.json") or {}
    v = "".join(f'<figure><figcaption>{"55° (위 링 35°보다 20° 위)" if "e55" in f else "5° (아래 링 15°보다 10° 아래)"}</figcaption>'
                f'<div style="overflow-x:auto"><video src="c1015/{k}/{f}" controls muted loop playsinline preload="metadata" style="max-width:none;width:1600px;background:#fff;border-radius:4px"></video></div></figure>'
                for f in idx.get("orbit", []))
    return f'<h3 style="margin:8px 0 0">궤도 영상 · 학습 링 밖 높이</h3><div style="display:flex;flex-direction:column;gap:10px">{v}</div>' if v else ""


CONCL = Path(G / "c1015_conclusion.html").read_text() if (G / "c1015_conclusion.html").exists() else ""
EXTRA = Path(G / "c1015_extra.html").read_text() if (G / "c1015_extra.html").exists() else ""
parts = []
for k in ("dance", "zju"):
    cap = "각 칸: 전체 이미지(절반 해상도) + 얼굴 · 머리카락 · {} 확대, 아래 줄은 |오차|×3. 칸 왼쪽 위 숫자는 그 이미지의 fg PSNR(마스크 전체).".format("점무늬" if k == "dance" else "몸통")
    parts.append(f'<h3 style="margin:14px 0 0;font-size:1.05rem">{NAMES[k]} · gen48_ho8</h3>{table(k)}'
                 + gallery(k, "held", f"{NAMES[k]} · 새 시점 (홀드아웃 8뷰 × 프레임 0·30·59)", cap)
                 + gallery(k, "seen", f"{NAMES[k]} · 학습 뷰 (15° 링 g03·g15, 35° 링 g30·g42)", cap) + orbits(k))
sec = f'''<!--C1015-->
  <section>
    <div><div class="eyebrow">현재 비교 (10/15) · gen48_ho8 · 생성 시드 42 · 학습 40뷰, 홀드아웃 8뷰 생성 이미지로 채점</div><h2>현재 비교 (10/15)</h2>
    <p class="sub">모든 수치는 홀드아웃 8뷰 × 60프레임, 흰 배경 합성 기준입니다. 4DGS dynerf는 4DGS 공식 렌더러로 흰 배경에 직접 렌더했습니다(알파 없음). 떨림은 SR4D의 가우시안 궤적에서만 잴 수 있습니다. 이미지는 가로로 스크롤됩니다.</p></div>
    {CONCL}
    {"".join(parts)}
    {EXTRA}
  </section>
<!--/C1015-->
'''
h = H.read_text()
h = re.sub(r"<!--C1015-->.*?<!--/C1015-->\n?", "", h, flags=re.S)
h = h.replace("  </header>\n", "  </header>\n" + sec, 1) if "<!--C1015-->" not in h else h
H.write_text(h)
print("ok", len(h))
