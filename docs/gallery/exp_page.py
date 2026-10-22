"""Builds the "SR4D 실험 기록" artifact page: docs/gallery/pub2/sr4d_experiments.html from docs/gallery/exp_registry.json.
Metrics come from <dir>/metrics_renders.json (+ jitter.json); figures from docs/gallery/pub2/exp/<card>/index.json
(tools/exp_figure.py). usage: python docs/gallery/exp_page.py   (re-run after editing the registry)
"""
import html
import json
from pathlib import Path

G = Path(__file__).parent; S = Path("/data/AI-Graphics/Applications/SR4D/outputs"); P = G / "pub2"
REG = json.loads((G / "exp_registry.json").read_text())
COLS = [("psnr", "전체 PSNR ↑", 2, 1), ("fgi_psnr", "fg 내부 PSNR ↑", 2, 1), ("fg_lpips", "fg LPIPS ↓", 4, -1),
        ("oom_alpha", "마스크 밖 알파 ↓", 3, -1), ("tof", "tOF (px) ↓", 2, -1), ("jit", "떨림 (mm) ↓", 2, -1)]


def J(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def metrics(row):
    f = row.get("file", "metrics_renders.json"); sp = row.get("split", "test_interp")
    E = dict((J(S / row["dir"] / f) or {}).get(sp) or {})
    if not E:
        return None
    E.update((J(S / row["dir"] / f.replace("metrics_", "metrics_covis_")) or {}).get(sp) or {})
    jit = row.get("jit", "auto")
    E["jit"] = (J(S / row["dir"] / "jitter.json") or {}).get("accel_rms_mm") if jit == "auto" else jit
    return E


def fmt(v, p):
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    return v if isinstance(v, str) else f"{v:.{p}f}"


def table(rows, note="", cols=None):
    COLS = cols or globals()["COLS"]
    data = [(r, metrics(r)) for r in rows]
    data = [(r, E) for r, E in data if E is not None]
    if not data:
        return '<p class="muted">결과 대기 중</p>'
    best = {}
    for c, _, _, sg in COLS:
        vals = [E[c] for _, E in data if isinstance(E.get(c), (int, float)) and E.get(c) == E.get(c)]
        best[c] = (max if sg > 0 else min)(vals) if vals else None
    tr = []
    for r, E in data:
        cls = ' class="ours"' if r.get("ours") else ""
        tds = "".join(f'<td{" class=best" if best[c] is not None and E.get(c) == best[c] else ""}>{fmt(E.get(c), p)}</td>' for c, _, p, _ in COLS)
        tr.append(f"<tr{cls}><th scope=row>{html.escape(r['label'])}</th>{tds}</tr>")
    head = "".join(f"<th scope=col>{h}</th>" for _, h, _, _ in COLS)
    nt = f'<p class="note">{note}</p>' if note else ""
    return f'<div class="tbl"><table><thead><tr><th scope=col>설정</th>{head}</tr></thead><tbody>{"".join(tr)}</tbody></table></div>{nt}'


def gallery(card, grp, title, cap=""):
    idx = J(P / f"exp/{card}/index.json") or {}
    ims = idx.get(grp, [])
    if not ims:
        return ""
    gid = f"g_{card}_{grp}".replace("-", "_")
    btn = "".join(f'<button type="button" data-src="exp/{card}/{f}" aria-pressed="{str(i == 0).lower()}">{f.rsplit("_", 1)[1][:-4]}</button>'
                  for i, f in enumerate(ims))
    st = idx.get("share_stats", {}).get(grp)
    sts = ("<p class=note>화면 누적 알파 중 안쪽층 비율(이 그림의 평균): " + ", ".join(f"{html.escape(k)} {v * 100:.0f}%" for k, v in st.items()) + "</p>") if st else ""
    return f'''<figure class="gal"><figcaption><span class="ft">{title}</span><span class="chips" id="{gid}">{btn}</span></figcaption>
<div class="scroll"><img id="{gid}_img" src="exp/{card}/{ims[0]}" alt="{html.escape(title)}" loading="lazy"></div>
{f'<p class="note">{cap}</p>' if cap else ''}{sts}</figure>
<script>(()=>{{const c=document.getElementById('{gid}'),im=document.getElementById('{gid}_img');c.querySelectorAll('button').forEach(b=>b.onclick=()=>{{c.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed','false'));b.setAttribute('aria-pressed','true');im.src=b.dataset.src}})}})();</script>'''


def videos(card):
    idx = J(P / f"exp/{card}/index.json") or {}
    vs = idx.get("orbit", [])
    caps = idx.get("orbit_caps", {})
    return "".join(f'<figure class="gal"><figcaption><span class="ft">궤도 · {html.escape(caps.get(f, f))}</span></figcaption><div class="scroll">'
                   f'<video src="exp/{card}/{f}" controls muted loop playsinline preload="metadata"></video></div></figure>' for f in vs)


NEAR = list(COLS)
FAR = NEAR + [("cv_ipsnr", "co-visible fg 내부 ↑", 2, 1), ("cv_lpips", "co-visible LPIPS ↓", 3, -1)]

FAR_J = [c for c in FAR if c[0] != "jit"]  # far-view card table: no jitter column, + co-visible


def top_table(rows, note=""):
    """One table, near (gen48_ho8 held-out) and far (r15h35 35-deg ring) columns side by side."""
    groups = [("near", "가까운 시점 · gen48_ho8 홀드아웃 8뷰", NEAR), ("far", "먼 시점 · r15h35 35° 링 24뷰", FAR)]
    data = [(r, {g: (metrics(r[g]) if r.get(g) else None) for g, _, _ in groups}) for r in rows]
    best = {}
    for g, _, cols in groups:
        for c, _, _, sg in cols:
            vals = [E[g][c] for _, E in data if E[g] and isinstance(E[g].get(c), (int, float)) and E[g].get(c) == E[g].get(c)]
            best[g, c] = (max if sg > 0 else min)(vals) if vals else None
    tr = []
    for r, E in data:
        tds = ""
        for g, _, cols in groups:
            for i, (c, _, p, _) in enumerate(cols):
                v = E[g].get(c) if E[g] else None
                cl = (["best"] if best[g, c] is not None and v == best[g, c] else []) + (["gs"] if i == 0 else [])
                ca = f' class="{" ".join(cl)}"' if cl else ""
                tds += f"<td{ca}>{fmt(v, p)}</td>"
        tr.append(f'<tr{" class=ours" if r.get("ours") else ""}><th scope=row>{html.escape(r["label"])}</th>{tds}</tr>')
    h1 = "".join(f'<th scope=colgroup colspan={len(cols)} class=gs>{t}</th>' for _, t, cols in groups)
    h2 = "".join(f'<th scope=col{" class=gs" if i == 0 else ""}>{h}</th>' for _, _, cols in groups for i, (_, h, _, _) in enumerate(cols))
    nt = f'<p class="note">{note}</p>' if note else ""
    return f'<div class="tbl"><table><thead><tr><th scope=col rowspan=2>이름</th>{h1}</tr><tr>{h2}</tr></thead><tbody>{"".join(tr)}</tbody></table></div>{nt}'


STATUS = {"done": ("완료", "ok"), "running": ("진행 중", "run"), "queued": ("대기", "wait"), "check": ("확인", "ok"), "blocked": ("결정 필요", "warn")}


def chip(s):
    t, c = STATUS.get(s, (s, "wait"))
    return f'<span class="st {c}">{t}</span>'


def card(e):
    body = []
    body.append(f'<dl class="qa"><dt>질문</dt><dd>{e["question"]}</dd><dt>설정</dt><dd>{e.get("setting", "")}</dd></dl>')
    if e.get("extra_html"):
        body.append(e["extra_html"])
    if e.get("rows"):
        body.append(table(e["rows"], e.get("table_note", ""), FAR_J if e.get("cols") == "far" else None))
    if e.get("nf_rows"):
        body.append(top_table(e["nf_rows"], e.get("table_note", "")))
    for g in e.get("galleries", []):
        body.append(gallery(e["card"], g["grp"], g["title"], g.get("cap", "")))
    body.append(videos(e["card"]))
    if e.get("conclusion_long"):
        body.append(f'<div class="concl"><h4>결론</h4>{e["conclusion_long"]}</div>')
    return f'''<article class="card" id="{e["id"].lower()}"><header><span class="eid">{e["id"]}</span><h3>{e["title"]}</h3>{chip(e["status"])}<span class="date">{e["date"]}</span></header>{"".join(body)}</article>'''


def main():
    exps = REG["experiments"]
    lst = "".join(f'<tr><th scope=row><a href="#{e["id"].lower()}">{e["id"]}</a></th><td>{e["date"]}</td><td>{e["question"]}</td><td>{e.get("conclusion", "–")}</td><td>{chip(e["status"])}</td></tr>' for e in exps)
    names = "".join(f"<tr><th scope=row>{html.escape(n)}</th><td>{html.escape(d)}</td></tr>" for n, d in REG["names"])
    T = REG["top"]
    top = (f'<figure class="gal"><figcaption><span class="ft">{T["fig_title"]}</span></figcaption><div class="scroll"><img src="{T["figure"]}" alt="{html.escape(T["fig_title"])}"></div>'
           f'<p class="note">{T["fig_cap"]}</p></figure>' + top_table(T["rows"], T.get("note", "")))
    base = "".join(f'<h4 class="sub">{html.escape(k)}</h4>' + table(v) for k, v in REG["baselines"].items())
    page = f'''<title>SR4D 실험 기록</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: lab notebook — summary block (best setting, experiment index, fixed baselines) then one card per experiment, in order */
:root {{ --bg:#f5f6f4; --paper:#ffffff; --ink:#1d2421; --muted:#5d6a64; --line:#d9dfdb; --accent:#1f6f5c; --accent-soft:#e2efe9;
  --ok:#2c7a4b; --run:#2f5fb3; --warn:#b4661a; --wait:#7b8580; --best:#e9f4ee; --ours:#f3f7fd;
  --sans:"IBM Plex Sans KR", "Apple SD Gothic Neo", "Malgun Gothic", system-ui, sans-serif; --mono:"IBM Plex Mono", ui-monospace, Menlo, monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#121614; --paper:#1a201d; --ink:#e3e8e5; --muted:#9aa7a1; --line:#2e3833; --accent:#6cc3a8; --accent-soft:#1f3029;
  --ok:#6cc58f; --run:#86a9ee; --warn:#e3a35e; --wait:#8e9a94; --best:#1f3328; --ours:#1b2433; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --bg:#121614; --paper:#1a201d; --ink:#e3e8e5; --muted:#9aa7a1; --line:#2e3833; --accent:#6cc3a8; --accent-soft:#1f3029;
  --ok:#6cc58f; --run:#86a9ee; --warn:#e3a35e; --wait:#8e9a94; --best:#1f3328; --ours:#1b2433; color-scheme:dark; }}
body {{ background:var(--bg); color:var(--ink); font:15px/1.6 var(--sans); }}
.wrap {{ max-width:1180px; margin:0 auto; padding-block:28px 64px; padding-left:max(16px, env(safe-area-inset-left, 0px)); padding-right:max(16px, env(safe-area-inset-right, 0px)); display:flex; flex-direction:column; gap:28px; }}
h1 {{ font-size:1.6rem; margin:0; font-weight:600; letter-spacing:-.01em; text-wrap:balance; }}
h2 {{ font-size:1.05rem; margin:0 0 10px; font-weight:600; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }}
h3 {{ font-size:1.12rem; margin:0; font-weight:600; text-wrap:balance; flex:1 1 260px; min-width:0; }}
h4 {{ margin:6px 0 4px; font-size:.95rem; }} h4.sub {{ color:var(--muted); font-weight:500; }}
.lede {{ color:var(--muted); margin:6px 0 0; max-width:70ch; }}
.best-line {{ background:var(--accent-soft); border-left:4px solid var(--accent); padding:12px 16px; border-radius:4px; }}
.best-line b {{ color:var(--accent); }}
.tbl {{ overflow-x:auto; border:1px solid var(--line); border-radius:6px; background:var(--paper); }}
table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; font-size:.88rem; }}
th, td {{ padding:7px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
thead th {{ font-weight:500; color:var(--muted); background:var(--bg); font-size:.8rem; }}
tbody th, thead th:first-child {{ text-align:left; font-weight:500; white-space:normal; min-width:150px; }}
td.best {{ background:var(--best); font-weight:600; }} tr.ours th {{ color:var(--accent); }} tr.ours {{ background:var(--ours); }}
th.gs, td.gs {{ border-left:2px solid var(--line); }} thead th.gs[scope=colgroup] {{ text-align:center; color:var(--ink); }}
.names td {{ text-align:left; white-space:normal; }} .names th {{ font-family:var(--mono); color:var(--accent); width:1%; white-space:nowrap; }}
.idx td:nth-child(3), .idx td:nth-child(4) {{ text-align:left; white-space:normal; min-width:220px; }}
.idx a {{ color:var(--accent); font-family:var(--mono); text-decoration:none; }} .idx a:hover, .idx a:focus-visible {{ text-decoration:underline; }}
.st {{ display:inline-block; font-size:.75rem; padding:1px 8px; border-radius:99px; border:1px solid currentColor; font-weight:500; }}
.st.ok {{ color:var(--ok); }} .st.run {{ color:var(--run); }} .st.warn {{ color:var(--warn); }} .st.wait {{ color:var(--wait); }}
.card {{ background:var(--paper); border:1px solid var(--line); border-radius:8px; padding:18px; display:flex; flex-direction:column; gap:14px; min-width:0; }}
.card > header {{ display:flex; flex-wrap:wrap; align-items:center; gap:10px; }}
.eid {{ font-family:var(--mono); font-size:.85rem; color:var(--paper); background:var(--accent); padding:2px 8px; border-radius:4px; }}
.date {{ font-family:var(--mono); font-size:.8rem; color:var(--muted); }}
.qa {{ display:grid; grid-template-columns:max-content 1fr; gap:4px 14px; margin:0; }} .qa dt {{ color:var(--muted); font-size:.85rem; padding-top:2px; }} .qa dd {{ margin:0; min-width:0; }}
.note, .muted {{ color:var(--muted); font-size:.82rem; margin:4px 0 0; }}
.gal {{ margin:0; display:flex; flex-direction:column; gap:6px; min-width:0; }}
.gal figcaption {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; }} .ft {{ font-weight:500; font-size:.9rem; }}
.chips {{ display:flex; flex-wrap:wrap; gap:4px; }}
.chips button {{ font:500 .78rem var(--mono); border:1px solid var(--line); background:var(--bg); color:var(--ink); border-radius:4px; padding:2px 8px; cursor:pointer; }}
.chips button[aria-pressed="true"] {{ background:var(--accent); color:var(--paper); border-color:var(--accent); }}
.chips button:focus-visible, .idx a:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
.scroll {{ overflow-x:auto; border-radius:4px; }} .scroll img, .scroll video {{ max-width:none; display:block; background:#fff; }}
.concl {{ border-top:1px dashed var(--line); padding-top:10px; }} .concl p {{ margin:4px 0; max-width:80ch; }}
code {{ font-family:var(--mono); font-size:.85em; }}
@media (max-width:560px) {{ .qa {{ grid-template-columns:1fr; }} }}
</style>
<div class="wrap">
<header><h1>SR4D 실험 기록</h1><p class="lede">2층 구조(바깥 = FTGS식 자유 가우시안, 안쪽 = SR4D 밑칠) 실험부터 번호를 매겨 기록합니다. 모든 수치는 생성 이미지로 채점합니다(gen48_ho8: 홀드아웃 8뷰 × 60프레임, r15h35: 35° 링 24뷰). 배경은 흰색으로 합성합니다. 설계 메모: docs/DECISIONS.md D-040·D-041.</p></header>
<section><h2>이름표</h2><div class="tbl names"><table><tbody>{names}</tbody></table></div></section>
<section style="display:flex;flex-direction:column;gap:12px"><h2>한눈에 보기</h2>{top}</section>
<section><h2>현재 최선 설정</h2><div class="best-line">{REG["best"]}</div></section>
<section><h2>실험 목록</h2><div class="tbl idx"><table><thead><tr><th scope=col>번호</th><th scope=col>날짜</th><th scope=col>질문</th><th scope=col>결론</th><th scope=col>상태</th></tr></thead><tbody>{lst}</tbody></table></div></section>
<section><h2>고정 비교 기준 · gen48_ho8</h2>{base}<p class="note">4DGS dynerf는 공식 렌더러가 알파를 내지 않아 마스크 밖 알파가 없습니다. 떨림은 가우시안 궤적이 있는 SR4D에서만 잽니다.</p></section>
<section style="display:flex;flex-direction:column;gap:18px"><h2>실험 카드</h2>{"".join(card(e) for e in exps)}</section>
</div>
'''
    (P / "sr4d_experiments.html").write_text(page)
    print("ok", len(page))


if __name__ == "__main__":
    main()
