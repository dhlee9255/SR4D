"""Interactive 4D viewer (user step 6, plan A): viser + nerfview in the browser, port 8080.

Methods are render adapters:
  SR4D   : checkpoint -> model.deform(t) (replay, L4 discarded = mean code, as the evaluation protocol) -> SR4D renderer
  DualGS : per-frame Inria PLYs (outputs/dualgs/<run>/model/ckt/point_cloud_{rel}.ply)
  4DGS   : per-frame PLYs exported by scripts/baselines/fourdgs_export_frames.py into outputs/viewer_cache/4dgs/<run>/
  FTGS++ : not yet (needs its own environment) - shown as unavailable
GUI: time slider + play, method, dataset, SR4D diagnostic toggles (L2 dx, L3), training-camera frustums, background.
SR4D frames for the default toggles are deformed once and cached on CPU (moved to the GPU per frame); other toggle
combinations are deformed on the fly.

usage: CUDA_VISIBLE_DEVICES=<free gpu> PYTHONPATH=. python tools/viewer4d.py [--port 8080] [--datasets zju dance]
then on your machine: ssh -L 8080:localhost:8080 <server>  and open http://localhost:8080   (see docs/VIEWER.md)
"""
import argparse
import threading
import time
from functools import lru_cache
from pathlib import Path

import gsplat
import nerfview
import numpy as np
import torch
import viser
from omegaconf import OmegaConf

from sr4d.data import io
from sr4d.data.dataset import MultiViewSequence
from sr4d.model.renderer import composite, render
from sr4d.train.trainer import load_model

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {  # label -> (data root, setting, run tag)
    "ZJU gen48_all (s42)": ("data_dev/zju_mocap_gen/387_gen48_2ring", "gen48_all", "zju_gen48_all"),
    "dance gen48_all (s42)": ("data_dev/fouranyone_gen/dance5885633_gen48_2ring", "gen48_all", "dance_gen48_all"),
    "ZJU gen24_all": ("data_dev/zju_mocap_gen/387_gen24_genonly", "gen24", "zju_gen24_all"),
    "dance gen24_all": ("data_dev/fouranyone_gen/dance5885633_gen24", "gen24", "dance_gen24_all"),
}


class SR4DAdapter:
    def __init__(self, ckpt: Path):
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = OmegaConf.create(st["cfg"])
        self.data = MultiViewSequence(str(ROOT / cfg.data.root) if not Path(cfg.data.root).is_absolute() else cfg.data.root,
                                      cfg.data.setting, list(range(*cfg.data.frames)), load_images=False)
        self.m = load_model(ckpt, self.data, cfg).eval()
        self.m.body_adhere = False
        self.T = self.data.num_frames
        self.cache = {}

    @torch.no_grad()
    def _deform(self, t, l2dx, l3):
        m = self.m
        m.use_l2, m.use_l3, m.use_l4 = m.l2 is not None, (m.l3 is not None) and l3, m.l4 is not None
        m.l2_no_dx = not l2dx
        g = m.deform(t, replay=l3, need_aux=False)
        aff = []
        if g["l3"] is not None:
            aff.append(tuple(x.detach() for x in (g["l3"]["color_A"], g["l3"]["color_b"])))
        if m.l4 is not None:
            aff.append(tuple(x.detach() for x in m.l4.affine(None)))
        keep = {k: g[k].detach() for k in ("means", "covars", "opacities", "shs", "normals")}
        return keep, aff

    def frame(self, t, l2dx=True, l3=True):
        key = (t, l2dx, l3)
        if (l2dx, l3) == (True, True):
            if key not in self.cache:
                g, aff = self._deform(t, l2dx, l3)
                self.cache[key] = ({k: v.cpu() for k, v in g.items()}, [(a.cpu(), b.cpu()) for a, b in aff])
            g, aff = self.cache[key]
            return {k: v.cuda(non_blocking=True) for k, v in g.items()}, [(a.cuda(), b.cuda()) for a, b in aff]
        return self._deform(t, l2dx, l3)

    @torch.no_grad()
    def render(self, t, K, w2c, W, H, bg, l2dx=True, l3=True):
        g, aff = self.frame(t, l2dx, l3)
        r = render(g, K, w2c, W, H, self.m.sh_degree, self.m.filter3d, with_normal=False)
        return composite(r["rgb_p"], r["alpha"], bg, aff).clamp(0, 1)


class PLYAdapter:
    def __init__(self, pattern: str, T: int):
        self.pattern, self.T = pattern, T

    @lru_cache(maxsize=8)
    def _load(self, t):
        from scripts.baselines.render_ply import load_ply
        return load_ply(self.pattern.format(rel=t))

    @torch.no_grad()
    def render(self, t, K, w2c, W, H, bg, **_):
        means, quats, scales, op, sh, deg = self._load(t)
        img, _, _ = gsplat.rasterization(means, quats, scales, op, sh, w2c[None], K[None], W, H, sh_degree=deg,
                                         backgrounds=bg[None], packed=False)
        return img[0].clamp(0, 1)


def adapters_for(tag: str, T: int):
    out = {}
    ck = ROOT / f"outputs/sr4d_final/{tag}/ckpt_final.pt"
    if ck.exists():
        out["SR4D"] = lambda: SR4DAdapter(ck)
    dg = ROOT / f"outputs/dualgs/{tag}/model/ckt/point_cloud_0.ply"
    if dg.exists():
        out["DualGS"] = lambda: PLYAdapter(str(dg).replace("point_cloud_0.ply", "point_cloud_{rel}.ply"), T)
    f4 = ROOT / f"outputs/viewer_cache/4dgs/{tag}/time_00000.ply"
    if f4.exists():
        out["4DGS"] = lambda: PLYAdapter(str(f4).replace("time_00000.ply", "time_{rel:05d}.ply"), T)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = reachable from other PCs on the network")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--datasets", nargs="+", default=["zju", "dance"])
    a = ap.parse_args()
    dsets = {k: v for k, v in DATASETS.items() if any(k.lower().startswith(d) for d in a.datasets)
             and (ROOT / v[0]).exists()}
    state = {"t": 0, "playing": False}
    loaded = {}

    def get_adapter(ds, meth):
        key = (ds, meth)
        if key not in loaded:
            root, setting, tag = dsets[ds]
            T = len(io.load_split(ROOT / root, setting)["frames"])
            fac = adapters_for(tag, T).get(meth)
            loaded[key] = fac() if fac else None
        return loaded[key]

    # CUDA init + kernel JIT + default adapter load happen HERE, not in nerfview's render thread: nerfview interrupts
    # renders with a trace hook, and an interrupted lazy CUDA init killed the first viewer process.
    torch.zeros(1, device="cuda")
    first = next(iter(dsets))
    for meth in ("SR4D", "DualGS"):
        ad = get_adapter(first, meth)
        if ad is not None:
            K = torch.tensor([[400.0, 0, 176], [0, 400.0, 320], [0, 0, 1]], device="cuda")
            w2c = torch.eye(4, device="cuda"); w2c[2, 3] = 3.0
            ad.render(0, K, w2c, 352, 640, torch.ones(3, device="cuda"))
            print(f"warm-up done: {meth} / {first}", flush=True)
    server = viser.ViserServer(host=a.host, port=a.port, verbose=False)
    with server.gui.add_folder("4D"):
        ds_dd = server.gui.add_dropdown("데이터", tuple(dsets))
        meth_dd = server.gui.add_dropdown("방법", ("SR4D", "DualGS", "4DGS", "FTGS++"))
        t_sl = server.gui.add_slider("시간 (프레임)", min=0, max=59, step=1, initial_value=0)
        play = server.gui.add_checkbox("재생", False)
        fps = server.gui.add_slider("재생 fps", min=1, max=30, step=1, initial_value=12)
        bg_dd = server.gui.add_dropdown("배경", ("흰색", "검정"))
        frus = server.gui.add_checkbox("학습 카메라 frustum", False)
        status = server.gui.add_markdown("")
    with server.gui.add_folder("SR4D 진단용 (렌더 시 끄기)"):
        l2_cb = server.gui.add_checkbox("L2 Δx", True)
        l3_cb = server.gui.add_checkbox("L3 (replay)", True)

    frustum_handles = []

    def update_frustums(_=None):
        for h in frustum_handles:
            h.remove()
        frustum_handles.clear()
        if not frus.value:
            return
        root, setting, _ = dsets[ds_dd.value]
        cams = io.load_cameras(ROOT / root)
        for c in io.load_split(ROOT / root, setting)["train_cams"]:
            R, t = np.array(cams[c]["R"]), np.array(cams[c]["t"]).reshape(3)
            c2w = np.eye(4); c2w[:3, :3] = R.T; c2w[:3, 3] = -R.T @ t
            fov = 2 * np.arctan(cams[c]["H"] / 2 / cams[c]["K"][1][1])
            frustum_handles.append(server.scene.add_camera_frustum(
                f"/cams/{c}", fov=fov, aspect=cams[c]["W"] / cams[c]["H"], scale=0.12,
                wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz, position=c2w[:3, 3], color=(255, 160, 40)))

    def render_fn(camera_state: nerfview.CameraState, render_tab_state: nerfview.RenderTabState):
        W, H = render_tab_state.viewer_width, render_tab_state.viewer_height
        try:
            ad = get_adapter(ds_dd.value, meth_dd.value)
        except Exception as e:  # never let one adapter failure kill the server
            status.content = f"**{meth_dd.value}** 로드 실패: {type(e).__name__}"
            return np.full((H, W, 3), 128, np.uint8)
        if ad is None:
            status.content = f"**{meth_dd.value}**: 이 데이터용 렌더 어댑터/결과가 아직 없습니다."
            return np.full((H, W, 3), 128, np.uint8)
        t = min(int(t_sl.value), ad.T - 1)
        K = torch.tensor(camera_state.get_K((W, H)), dtype=torch.float32, device="cuda")
        w2c = torch.linalg.inv(torch.tensor(camera_state.c2w, dtype=torch.float32, device="cuda"))
        bg = torch.ones(3, device="cuda") if bg_dd.value == "흰색" else torch.zeros(3, device="cuda")
        img = ad.render(t, K, w2c, W, H, bg, l2dx=l2_cb.value, l3=l3_cb.value)
        status.content = f"{meth_dd.value} · {ds_dd.value} · t={t}" + ("" if meth_dd.value == "SR4D" else " (SR4D 토글은 적용 안 됨)")
        return (img.cpu().numpy() * 255).astype(np.uint8)

    viewer = nerfview.Viewer(server=server, render_fn=render_fn, mode="rendering")
    for h in (ds_dd, meth_dd, t_sl, bg_dd, l2_cb, l3_cb):
        h.on_update(lambda _: viewer.rerender(_))
    frus.on_update(update_frustums); ds_dd.on_update(update_frustums)

    def player():
        while True:
            if play.value:
                t_sl.value = (int(t_sl.value) + 1) % 60
            time.sleep(1.0 / max(1, fps.value))
    threading.Thread(target=player, daemon=True).start()
    print(f"viewer on {a.host}:{a.port}  (same network: http://<server-ip>:{a.port}; or ssh -L {a.port}:localhost:{a.port} <server>)", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
