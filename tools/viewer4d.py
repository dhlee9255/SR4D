"""Interactive 4D viewer v2 (user 2026-10-13): viser only, own render loop (nerfview v1 kept as viewer4d_nerfview_v1.py).

Why v2: nerfview lowered the resolution (rays/s budget) and JPEG quality (40) for every "rerender", i.e. for every playback
frame -> splats rendered small and upscaled = "dust" while playing; full quality only after the camera rested. Here every
frame is rendered at the browser canvas resolution (capped by "최대 해상도") with JPEG 90; if rendering is slower than the
playback fps, frames are SKIPPED (time advances with the wall clock), never degraded.

Features
  - methods: SR4D (checkpoint, replay, L4 discarded), DualGS / 4DGS (per-frame PLY); side-by-side SR4D | DualGS
  - camera: viser orbit controls (left drag = rotate about the person, right drag = pan, wheel = zoom), world up = +y,
    rotation centre = pelvis of the current frame
  - generated-camera orbit mode: sphere / cylinder through the 48 generated cameras (centre = mean pelvis over frames,
    radius = mean camera distance), camera always looks at the centre, focal interpolated from the nearest cameras;
    azimuth 0-360, elevation -30..+80 (cylinder: height), step 1 / 5 deg; drag is snapped back onto the surface;
    shows azimuth / elevation, angle to the nearest TRAINING camera, interpolation vs extrapolation
  - jump to any of the 48 cameras (dropdown or click its frustum); training cams orange, ho8 held-out cams blue
  - person rotation (yaw / pitch / roll about the pelvis, exact: applied as the inverse camera motion), reset; locked
    and reset while orbit mode is on
usage: CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. python tools/viewer4d.py [--host 0.0.0.0] [--port 8080]
"""
import argparse
import math
import threading
import time
from functools import lru_cache
from pathlib import Path

import gsplat
import numpy as np
import torch
import viser
import viser.transforms as vtf
from omegaconf import OmegaConf

from sr4d.data import io
from sr4d.data.dataset import MultiViewSequence
from sr4d.model.body_model import BodyModel, params_to_tensors
from sr4d.model.renderer import composite, render
from sr4d.train.trainer import load_model

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {  # label -> (data root, setting, run tag, split used to colour train / held-out cams)
    "ZJU gen48_all (s42)": ("data_dev/zju_mocap_gen/387_gen48_2ring", "gen48_all", "zju_gen48_all", "gen48_ho8"),
    "dance gen48_all (s42)": ("data_dev/fouranyone_gen/dance5885633_gen48_2ring", "gen48_all", "dance_gen48_all", "gen48_ho8"),
    "ZJU gen24_all": ("data_dev/zju_mocap_gen/387_gen24_genonly", "gen24", "zju_gen24_all", "gen24"),
    "dance gen24_all": ("data_dev/fouranyone_gen/dance5885633_gen24", "gen24", "dance_gen24_all", "gen24"),
}
GPU_LOCK = threading.Lock()


# ---------------------------------------------------------------- render adapters
class SR4DAdapter:
    def __init__(self, ckpt: Path):
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = OmegaConf.create(st["cfg"])
        root = cfg.data.root if Path(cfg.data.root).is_absolute() else str(ROOT / cfg.data.root)
        self.data = MultiViewSequence(root, cfg.data.setting, list(range(*cfg.data.frames)), load_images=False)
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
        return {k: g[k].detach() for k in ("means", "covars", "opacities", "shs", "normals")}, aff

    def frame(self, t, l2dx=True, l3=True):  # one complete frame (all attributes from the same t)
        if (l2dx, l3) == (True, True):
            if t not in self.cache:
                g, aff = self._deform(t, True, True)
                self.cache[t] = ({k: v.cpu() for k, v in g.items()}, [(a.cpu(), b.cpu()) for a, b in aff])
            g, aff = self.cache[t]
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

    @lru_cache(maxsize=12)
    def _load(self, t):
        from scripts.baselines.render_ply import load_ply
        return load_ply(self.pattern.format(rel=t))

    @torch.no_grad()
    def render(self, t, K, w2c, W, H, bg, **_):
        means, quats, scales, op, sh, deg = self._load(t)
        img, _, _ = gsplat.rasterization(means, quats, scales, op, sh, w2c[None], K[None], W, H, sh_degree=deg,
                                         backgrounds=bg[None], packed=False)
        return img[0].clamp(0, 1)


def adapter_factories(tag: str, T: int):
    out = {}
    for name, sub in (("SR4D", "sr4d_final"),):
        for cand in (f"{tag}_cam0", tag):  # prefer the cam0 model once it exists (D-031 baseline)
            ck = ROOT / f"outputs/{sub}/{cand}/ckpt_final.pt"
            if ck.exists():
                out[name] = (lambda ck=ck: SR4DAdapter(ck)); out[name + "_src"] = cand; break
    nl = ROOT / f"outputs/sr4d_final/{tag.replace('gen48_all', 'gen48_ho8')}_nol4/ckpt_final.pt"  # D-038/L4-off (40-view ho8 model)
    if "gen48_all" in tag and nl.exists():
        out["SR4D L4 끔"] = (lambda nl=nl: SR4DAdapter(nl))
        for v in ("a", "b", "c", "d"):  # D-037/D-039 per-frame detail layer on top of the L4-off model
            dk = nl.parent.parent / f"{nl.parent.name}_s7{v}/ckpt_final.pt"
            if dk.exists():
                out[f"SR4D L4 끔 + 디테일 층 ({v})"] = (lambda dk=dk: SR4DAdapter(dk))
    dg = ROOT / f"outputs/dualgs/{tag}/model/ckt/point_cloud_0.ply"
    if dg.exists():
        out["DualGS"] = lambda: PLYAdapter(str(dg).replace("point_cloud_0.ply", "point_cloud_{rel}.ply"), T)
    f4 = ROOT / f"outputs/viewer_cache/4dgs/{tag}/time_00000.ply"
    if f4.exists():
        out["4DGS"] = lambda: PLYAdapter(str(f4).replace("time_00000.ply", "time_{rel:05d}.ply"), T)
    return out


# ---------------------------------------------------------------- dataset geometry (cameras, pelvis, orbit surface)
class Geometry:
    def __init__(self, root: Path, setting: str, cam_split: str):
        self.cams = io.load_cameras(root)
        sp = io.load_split(root, setting)
        self.frames = sp["frames"]
        try:
            cs = io.load_split(root, cam_split)
            self.train = set(cs["train_cams"]); self.held = set(cs.get("test_interp_cams", []) + cs.get("test_extrap_cams", [])) - self.train
        except Exception:
            self.train, self.held = set(sp["train_cams"]), set()
        meta = io.load_meta(root)
        bm = BodyModel(meta["body_model"], meta.get("gender", "neutral"))
        pel = []
        for f in self.frames:
            b, p, t, _ = params_to_tensors([io.load_body(root, f)])
            with torch.no_grad():
                pel.append(bm(b, p, t)[1][0, 0].numpy())
        self.pelvis = np.array(pel)                       # (T,3), world y up
        self.center = self.pelvis.mean(0)
        self.ids = sorted(self.cams)
        C = []; self.fy_norm = []
        for c in self.ids:
            R, t = np.array(self.cams[c]["R"]), np.array(self.cams[c]["t"]).reshape(3)
            C.append(-R.T @ t); self.fy_norm.append(self.cams[c]["K"][1][1] / self.cams[c]["H"])
        self.C = np.array(C); d = self.C - self.center
        self.radius = float(np.linalg.norm(d, axis=1).mean())
        self.r_h = float(np.linalg.norm(d[:, [0, 2]], axis=1).mean())
        self.az = np.degrees(np.arctan2(d[:, 0], d[:, 2])) % 360
        self.el = np.degrees(np.arcsin(d[:, 1] / np.linalg.norm(d, axis=1)))
        self.el_rng = (float(self.el.min()), float(self.el.max()))
        self.h_rng = (float(d[:, 1].min()), float(d[:, 1].max()))

    def pose_on(self, shape, az, el_or_h):
        a = math.radians(az)
        if shape == "구":
            e = math.radians(el_or_h)
            off = self.radius * np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])
        else:  # 원기둥: el_or_h = height above the centre [m]
            off = np.array([self.r_h * math.sin(a), el_or_h, self.r_h * math.cos(a)])
        return self.center + off

    def dir_angle(self, pos, idx):
        u = pos - self.center; v = self.C[idx] - self.center
        return float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))

    def nearest(self, pos, only_train=False):
        cand = [i for i, c in enumerate(self.ids) if (c in self.train or not only_train)]
        angs = [(self.dir_angle(pos, i), i) for i in cand]
        return min(angs)

    def focal_norm(self, pos):  # fy / H, inverse-angle weighted over the 2 nearest cameras
        angs = sorted((self.dir_angle(pos, i), i) for i in range(len(self.ids)))[:2]
        w = np.array([1 / max(a, 1e-3) for a, _ in angs])
        return float(np.dot(w, [self.fy_norm[i] for _, i in angs]) / w.sum())


def look_at_wxyz(pos, target, up=np.array([0.0, 1.0, 0.0])):
    """OpenCV camera (x right, y down, z forward) looking at target -> viser wxyz."""
    z = target - pos; z /= np.linalg.norm(z)
    x = np.cross(z, up); x /= max(np.linalg.norm(x), 1e-9)    # right
    y = np.cross(z, x)                                         # down
    return vtf.SO3.from_matrix(np.stack([x, y, z], 1)).wxyz


def rot_about(center, yaw, pitch, roll):
    R = vtf.SO3.from_y_radians(math.radians(yaw)) @ vtf.SO3.from_x_radians(math.radians(pitch)) @ vtf.SO3.from_z_radians(math.radians(roll))
    M = np.eye(4); M[:3, :3] = R.as_matrix(); M[:3, 3] = center - R.as_matrix() @ center
    return M


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--datasets", nargs="+", default=["zju", "dance"])
    a = ap.parse_args()
    dsets = {k: v for k, v in DATASETS.items() if any(k.lower().startswith(d) for d in a.datasets) and (ROOT / v[0]).exists()}
    geos, loaded = {}, {}

    def geo(ds):
        if ds not in geos:
            root, setting, _, cs = dsets[ds]; geos[ds] = Geometry(ROOT / root, setting, cs)
        return geos[ds]

    def adapter(ds, meth):
        key = (ds, meth)
        if key not in loaded:
            root, setting, tag, _ = dsets[ds]
            fac = adapter_factories(tag, len(geo(ds).frames)).get(meth)
            with GPU_LOCK:
                loaded[key] = fac() if fac else None
        return loaded[key]

    # warm-up outside any request: CUDA init, kernel JIT, default adapters
    torch.zeros(1, device="cuda")
    first = next(iter(dsets))
    for meth in ("SR4D", "DualGS"):
        ad = adapter(first, meth)
        if ad is not None:
            K = torch.tensor([[400.0, 0, 176], [0, 400.0, 320], [0, 0, 1]], device="cuda"); w2c = torch.eye(4, device="cuda"); w2c[2, 3] = 3
            with GPU_LOCK:
                ad.render(0, K, w2c, 352, 640, torch.ones(3, device="cuda"))
            print(f"warm-up: {meth} / {first}", flush=True)

    server = viser.ViserServer(host=a.host, port=a.port, verbose=False)
    server.scene.set_up_direction("+y")
    S = {"t": 0, "t0": time.time(), "playing": False, "dirty": 0, "snap": False}

    with server.gui.add_folder("4D"):
        ds_dd = server.gui.add_dropdown("데이터", tuple(dsets))
        meth_dd = server.gui.add_dropdown("방법", ("SR4D", "SR4D L4 끔", "SR4D L4 끔 + 디테일 층 (a)", "SR4D L4 끔 + 디테일 층 (b)", "SR4D L4 끔 + 디테일 층 (c)", "SR4D L4 끔 + 디테일 층 (d)", "DualGS", "4DGS", "SR4D | DualGS 나란히", "SR4D L4 끔 | DualGS 나란히"))
        t_sl = server.gui.add_slider("시간 (프레임)", min=0, max=59, step=1, initial_value=0)
        play = server.gui.add_checkbox("재생", False)
        fps = server.gui.add_slider("재생 fps", min=1, max=30, step=1, initial_value=12)
        bg_dd = server.gui.add_dropdown("배경", ("흰색", "검정"))
        res = server.gui.add_slider("최대 해상도 (긴 변 px)", min=480, max=2048, step=16, initial_value=1280)
        status = server.gui.add_markdown("")
    with server.gui.add_folder("카메라"):
        cam_dd = server.gui.add_dropdown("생성 카메라로 이동", ("—",))
        frus = server.gui.add_checkbox("카메라 frustum (주황 = 학습, 파랑 = 홀드아웃)", True)
        orbit = server.gui.add_checkbox("생성 카메라 궤도 모드", False)
        shape = server.gui.add_dropdown("궤도 면", ("구", "원기둥"))
        step = server.gui.add_dropdown("스텝", ("5°", "1°"))
        az_sl = server.gui.add_slider("방위각 (°)", min=0, max=359, step=1, initial_value=0)
        el_sl = server.gui.add_slider("고도 (°) / 원기둥: 높이 (cm)", min=-30, max=80, step=1, initial_value=15)
        bgrp = server.gui.add_button_group("이동", ("◀", "▶", "▲", "▼"))
        orbit_info = server.gui.add_markdown("")
    with server.gui.add_folder("사람 회전 (골반 중심)"):
        yaw = server.gui.add_number("yaw (°)", 0.0, step=5.0); pitch = server.gui.add_number("pitch (°)", 0.0, step=5.0)
        roll = server.gui.add_number("roll (°)", 0.0, step=5.0); rgrp = server.gui.add_button_group("회전", ("yaw −", "yaw +", "pitch −", "pitch +", "roll −", "roll +"))
        reset = server.gui.add_button("원래대로")
    with server.gui.add_folder("SR4D 진단용 (렌더 시 끄기)"):
        l2_cb = server.gui.add_checkbox("L2 Δx", True); l3_cb = server.gui.add_checkbox("L3 (replay)", True)

    handles = []

    def refresh_scene(_=None):
        g = geo(ds_dd.value)
        cam_dd.options = tuple(f"{c} ({'학습' if c in g.train else '홀드아웃' if c in g.held else '-'})" for c in g.ids)
        for h in handles:
            h.remove()
        handles.clear()
        if not frus.value:
            return
        for c in g.ids:
            R, t = np.array(g.cams[c]["R"]), np.array(g.cams[c]["t"]).reshape(3)
            fov = 2 * math.atan(g.cams[c]["H"] / 2 / g.cams[c]["K"][1][1])
            col = (255, 150, 30) if c in g.train else (40, 120, 255)
            h = server.scene.add_camera_frustum(f"/cams/{c}", fov=fov, aspect=g.cams[c]["W"] / g.cams[c]["H"], scale=0.10,
                                                color=col, wxyz=vtf.SO3.from_matrix(R.T).wxyz, position=-R.T @ t)
            h.on_click(lambda ev, c=c: goto_cam(c))
            handles.append(h)
        S["dirty"] += 1

    def set_client_cam(client, pos, target, fy_norm=None):
        with client.atomic():
            S["snap"] = True
            client.camera.position = pos; client.camera.look_at = target; client.camera.up_direction = (0.0, 1.0, 0.0)
            client.camera.wxyz = look_at_wxyz(np.asarray(pos, float), np.asarray(target, float))
            if fy_norm is not None:
                client.camera.fov = 2 * math.atan(0.5 / fy_norm)
            S["snap"] = False

    def orbit_pos():
        g = geo(ds_dd.value)
        v = el_sl.value if shape.value == "구" else el_sl.value / 100.0
        return g.pose_on(shape.value, az_sl.value, v)

    def apply_orbit(_=None):
        if not orbit.value:
            return
        g = geo(ds_dd.value); pos = orbit_pos()
        for cl in server.get_clients().values():
            set_client_cam(cl, pos, g.center, g.focal_norm(pos))
        update_orbit_info(pos); S["dirty"] += 1

    def update_orbit_info(pos):
        g = geo(ds_dd.value)
        ang, i = g.nearest(pos, only_train=True)
        d = pos - g.center; el = math.degrees(math.asin(d[1] / np.linalg.norm(d))); az = math.degrees(math.atan2(d[0], d[2])) % 360
        inside = g.el_rng[0] - 2 <= el <= g.el_rng[1] + 2
        orbit_info.content = (f"방위각 **{az:.0f}°**, 고도 **{el:.1f}°** · 가장 가까운 학습 카메라 **{g.ids[i]}**와 **{ang:.1f}°** · "
                              f"{'**보간** (카메라 범위 안)' if inside else '**외삽** (범위 밖)'} — 카메라 고도 범위 {g.el_rng[0]:.0f}–{g.el_rng[1]:.0f}°")

    def goto_cam(c):
        g = geo(ds_dd.value); i = g.ids.index(c)
        if orbit.value:
            az_sl.value = int(round(g.az[i])) % 360
            el_sl.value = int(round(g.el[i])) if shape.value == "구" else int(round((g.C[i] - g.center)[1] * 100))
            apply_orbit(); return
        R, t = np.array(g.cams[c]["R"]), np.array(g.cams[c]["t"]).reshape(3)
        for cl in server.get_clients().values():
            with cl.atomic():
                S["snap"] = True
                cl.camera.position = -R.T @ t; cl.camera.wxyz = vtf.SO3.from_matrix(R.T).wxyz
                cl.camera.fov = 2 * math.atan(g.cams[c]["H"] / 2 / g.cams[c]["K"][1][1]); cl.camera.look_at = g.pelvis[min(int(t_sl.value), len(g.pelvis) - 1)]
                S["snap"] = False
        S["dirty"] += 1

    @cam_dd.on_update
    def _(_):
        if cam_dd.value != "—":
            goto_cam(cam_dd.value.split()[0])

    @orbit.on_update
    def _(_):
        locked = orbit.value
        for h in (yaw, pitch, roll, rgrp, reset):
            h.disabled = locked
        if locked:
            yaw.value = pitch.value = roll.value = 0.0  # person rotation reset + locked in orbit mode
            el_sl.min, el_sl.max = (-30, 80) if shape.value == "구" else (-80, 250)
            apply_orbit()
        S["dirty"] += 1

    @shape.on_update
    def _(_):
        el_sl.min, el_sl.max = (-30, 80) if shape.value == "구" else (-80, 250)
        el_sl.value = 15 if shape.value == "구" else int(round((geo(ds_dd.value).h_rng[0]) * 100))
        apply_orbit()

    @bgrp.on_click
    def _(ev):
        st = 5 if step.value == "5°" else 1
        if ev.target.value == "◀":
            az_sl.value = (az_sl.value - st) % 360
        elif ev.target.value == "▶":
            az_sl.value = (az_sl.value + st) % 360
        else:
            d = st if ev.target.value == "▲" else -st
            el_sl.value = int(np.clip(el_sl.value + (d if shape.value == "구" else 5 * d), el_sl.min, el_sl.max))
        if not orbit.value:
            orbit.value = True
        apply_orbit()

    @rgrp.on_click
    def _(ev):
        k, s = ev.target.value.split()
        h = {"yaw": yaw, "pitch": pitch, "roll": roll}[k]
        h.value = float(h.value) + (5.0 if s == "+" else -5.0); S["dirty"] += 1

    @reset.on_click
    def _(_):
        yaw.value = pitch.value = roll.value = 0.0; S["dirty"] += 1

    for h in (az_sl, el_sl):
        h.on_update(apply_orbit)
    for h in (meth_dd, bg_dd, res, l2_cb, l3_cb, yaw, pitch, roll):
        h.on_update(lambda _: S.__setitem__("dirty", S["dirty"] + 1))
    t_sl.on_update(lambda _: S.__setitem__("dirty", S["dirty"] + 1) if not S["playing"] else None)
    ds_dd.on_update(refresh_scene); frus.on_update(refresh_scene)

    @play.on_update
    def _(_):
        S["playing"] = play.value; S["t0"] = time.time() - int(t_sl.value) / max(1, fps.value)

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        g = geo(ds_dd.value)
        c0 = g.ids[0]; R, t = np.array(g.cams[c0]["R"]), np.array(g.cams[c0]["t"]).reshape(3)
        set_client_cam(client, -R.T @ t, g.center, g.fy_norm[0])

        @client.camera.on_update
        def _(_):
            if S["snap"]:
                return
            if orbit.value:  # drag in orbit mode: project the new position back onto the surface
                gg = geo(ds_dd.value); d = np.asarray(client.camera.position) - gg.center
                az_new = int(round(math.degrees(math.atan2(d[0], d[2])))) % 360
                if shape.value == "구":
                    el_new = int(np.clip(round(math.degrees(math.asin(d[1] / np.linalg.norm(d)))), -30, 80))
                else:
                    el_new = int(np.clip(round(d[1] * 100), -80, 250))
                if (az_new, el_new) != (az_sl.value, el_sl.value):
                    az_sl.value, el_sl.value = az_new, el_new  # triggers apply_orbit
                elif np.linalg.norm(np.asarray(client.camera.position) - orbit_pos()) > 0.01:
                    apply_orbit()  # only if it drifted off the surface (avoids an echo loop)
            S["dirty"] += 1

        threading.Thread(target=render_loop, args=(client,), daemon=True).start()

    def frame_now():
        if S["playing"]:
            T = len(geo(ds_dd.value).frames)
            t = int((time.time() - S["t0"]) * fps.value) % T   # wall-clock time: slow renders skip frames
            if t != int(t_sl.value):
                t_sl.value = t
            return t
        return int(t_sl.value)

    def render_loop(client):
        last = None
        while True:
            try:
                t = frame_now()
                key = (S["dirty"], t, ds_dd.value, meth_dd.value)
                if key == last:
                    time.sleep(0.005); continue
                cam = client.camera
                W0, H0 = int(cam.image_width or 1280), int(cam.image_height or 720)
                sc = min(1.0, res.value / max(W0, H0)); W, H = max(16, int(W0 * sc)), max(16, int(H0 * sc))
                fy = H / 2 / math.tan(cam.fov / 2)
                c2w = np.eye(4); c2w[:3, :3] = vtf.SO3(np.asarray(cam.wxyz)).as_matrix(); c2w[:3, 3] = cam.position
                g = geo(ds_dd.value)
                M = rot_about(g.pelvis[min(t, len(g.pelvis) - 1)], yaw.value, pitch.value, roll.value)
                w2c = torch.tensor(np.linalg.inv(c2w) @ M, dtype=torch.float32, device="cuda")  # person rotation = inverse camera motion
                bg = torch.ones(3, device="cuda") if bg_dd.value == "흰색" else torch.zeros(3, device="cuda")
                meths = [meth_dd.value.split(" | ")[0], "DualGS"] if " | " in meth_dd.value else [meth_dd.value]
                Wm = W // len(meths); imgs = []; notes = []
                for mname in meths:
                    ad = adapter(ds_dd.value, mname)
                    if ad is None:
                        imgs.append(np.full((H, Wm, 3), 128, np.uint8)); notes.append(f"{mname}: 결과 없음"); continue
                    K = torch.tensor([[fy, 0, Wm / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32, device="cuda")
                    with GPU_LOCK:
                        im = ad.render(min(t, ad.T - 1), K, w2c, Wm, H, bg, l2dx=l2_cb.value, l3=l3_cb.value)
                    imgs.append((im.cpu().numpy() * 255).astype(np.uint8))
                img = np.concatenate(imgs, 1)
                client.scene.set_background_image(img, format="jpeg", jpeg_quality=90)
                src = adapter_factories(dsets[ds_dd.value][2], 60).get("SR4D_src", "-")
                status.content = f"t = {t} · {W}×{H} · SR4D 모델 `{src}`" + (" · " + ", ".join(notes) if notes else "")
                if orbit.value:
                    update_orbit_info(np.asarray(cam.position))
                last = key
            except Exception as e:  # never kill the server
                status.content = f"렌더 오류: {type(e).__name__}: {e}"
                time.sleep(0.5)

    refresh_scene()
    print(f"viewer on {a.host}:{a.port}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
