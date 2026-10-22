"""Model construction: body surfel sampling, free-Gaussian init by space carving + inverse LBS (CLAUDE.md 7.2 L1-b),
free bones, L2/L3/L4 modules.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from sr4d.data.dataset import MultiViewSequence
from sr4d.model.body_model import BodyModel
from sr4d.model.canonical import init_body_set, init_free_set
from sr4d.model.deform import SR4DModel
from sr4d.model.free_bones import FreeBones
from sr4d.model.instant_residual import InstantResidual
from sr4d.model.pose_nonrigid import PoseNonRigid
from sr4d.model.scaffold import Scaffold, dense_surface_points, nearest_vertex, sample_surface, scaffold_kw
from sr4d.model.view_absorber import ViewAbsorber


@torch.no_grad()
def carve_frame(data: MultiViewSequence, t: int, bmin: torch.Tensor, bmax: torch.Tensor, voxel: float, dilate_px: int = 2):
    """Visual hull of frame t from the training cameras' masks; returns surface voxel centres (N,3)."""
    dev = bmin.device
    n = ((bmax - bmin) / voxel).ceil().long()
    axes = [torch.arange(int(k), device=dev, dtype=torch.float32) * voxel + bmin[i] + voxel / 2 for i, k in enumerate(n)]
    X = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    occ = torch.ones(X.shape[0], dtype=torch.bool, device=dev)
    seen = torch.zeros(X.shape[0], dtype=torch.int32, device=dev)
    for cid in data.train_ids:
        cam = data.cameras[cid]
        m = (data.masks[cid][t] > 127).float()[None, None]
        if dilate_px > 0:
            m = F.max_pool2d(m, 2 * dilate_px + 1, 1, dilate_px)
        m = m[0, 0] > 0
        pc = X @ cam.w2c[:3, :3].T + cam.w2c[:3, 3]
        z = pc[:, 2]
        u = (pc[:, 0] / z.clamp_min(1e-6) * cam.K[0, 0] + cam.K[0, 2]).floor().long()
        v = (pc[:, 1] / z.clamp_min(1e-6) * cam.K[1, 1] + cam.K[1, 2]).floor().long()
        inside = (z > 0.05) & (u >= 0) & (u < cam.W) & (v >= 0) & (v < cam.H)
        hit = torch.zeros_like(occ)
        hit[inside] = m[v[inside], u[inside]]
        occ &= hit | ~inside
        seen += inside.int()
    occ &= seen >= 2
    grid = occ.reshape(*[int(k) for k in n]).float()[None, None]
    eroded = -F.max_pool3d(-grid, 3, 1, 1)
    surf = (grid > 0) & (eroded < 1)
    return X[surf.reshape(-1)]


@torch.no_grad()
def init_free_points(cfg, data: MultiViewSequence, sc: Scaffold, no_scaffold: bool = False):
    fc = cfg.model.free
    dev = sc.pose_init.device
    v_c, _, _, _ = sc.canonical()
    surf_c = dense_surface_points(v_c, sc.faces)
    frames = list(range(0, data.num_frames, max(1, int(fc.carve_every)))) if not no_scaffold else [0]
    pts = []
    for t in frames:
        G = sc.frame_transforms(t)
        v_t = sc.posed_vertices(G, v_c)[0]
        pad = 0.25
        P = carve_frame(data, t, v_t.min(0).values - pad, v_t.max(0).values + pad, fc.carve_voxel_m)
        if P.shape[0] == 0:
            continue
        if no_scaffold:
            pts.append(P)
            continue
        vi, _ = nearest_vertex(P, v_t)
        w = sc.lbs_weights[vi]
        T = torch.einsum("nj,jxy->nxy", w, G[0])
        Pc = (torch.linalg.inv(T) @ torch.cat([P, torch.ones_like(P[:, :1])], -1)[..., None])[:, :3, 0]
        _, dist = nearest_vertex(Pc, surf_c)
        pts.append(Pc[dist > fc.d_free_m])
    P = torch.cat(pts, 0) if pts else torch.zeros(0, 3, device=dev)
    if P.shape[0] > 0:
        key = torch.floor(P / fc.voxel_m).long()
        _, inv = torch.unique(key, dim=0, return_inverse=True)
        Pm = torch.zeros(int(inv.max()) + 1, 3, device=dev).index_add_(0, inv, P)
        cnt = torch.zeros(int(inv.max()) + 1, device=dev).index_add_(0, inv, torch.ones_like(P[:, 0]))
        P = Pm / cnt[:, None]
        if P.shape[0] > fc.max_init:
            P = P[torch.randperm(P.shape[0], device=dev)[: fc.max_init]]
    return P


@torch.no_grad()
def project_colors(points_w: torch.Tensor, normals_w: torch.Tensor | None, data: MultiViewSequence, t: int = 0):
    """Colour of each point from the training camera that faces it most (no occlusion test)."""
    best = torch.full((points_w.shape[0],), -2.0, device=points_w.device)
    col = torch.full((points_w.shape[0], 3), 0.5, device=points_w.device)
    center = points_w.mean(0)
    nrm = normals_w if normals_w is not None else F.normalize(points_w - center, dim=-1)
    for cid in data.train_ids:
        cam = data.cameras[cid]
        d = F.normalize(cam.center - points_w, dim=-1)
        score = (d * nrm).sum(-1)
        pc = points_w @ cam.w2c[:3, :3].T + cam.w2c[:3, 3]
        z = pc[:, 2].clamp_min(1e-6)
        u = (pc[:, 0] / z * cam.K[0, 0] + cam.K[0, 2]).floor().long()
        v = (pc[:, 1] / z * cam.K[1, 1] + cam.K[1, 2]).floor().long()
        ok = (u >= 0) & (u < cam.W) & (v >= 0) & (v < cam.H) & (score > best)
        if ok.any():
            m = data.masks[cid][t][v[ok], u[ok]] > 127
            idx = ok.nonzero()[:, 0][m]
            col[idx] = data.images[cid][t][v[idx], u[idx]].float() / 255.0
            best[idx] = score[idx]
    return col


def build_model(cfg, data: MultiViewSequence, scaffold_state: dict | None = None) -> SR4DModel:
    dev = data.device
    meta = data.meta
    no_scaffold = bool(cfg.model.get("no_scaffold", False))
    bm = BodyModel(meta["body_model"], meta.get("gender", "neutral")).to(dev)
    sc = Scaffold(bm, data.body_params, cfg.model.canonical_pose, **scaffold_kw(cfg)).to(dev)
    if scaffold_state is not None:
        sc.load_state_dict(scaffold_state, strict=False)
    sc.betas.requires_grad_(False)
    sc.invalidate()
    v_c, J_c, _, _ = sc.canonical()
    sh_max = cfg.model.body_surfels.max_sh_degree
    g = torch.Generator(device=dev).manual_seed(cfg.seed)

    body = None
    nb = int(cfg.model.body_surfels.n_init)
    if nb > 0 and not no_scaffold:
        tri, bary = sample_surface(v_c, sc.faces, nb, g)
        G0 = sc.frame_transforms(0)
        v0 = sc.posed_vertices(G0, v_c)[0]
        p0 = (v0[sc.faces[tri]] * bary[..., None]).sum(1)
        a, b_, c = v0[sc.faces[tri, 0]], v0[sc.faces[tri, 1]], v0[sc.faces[tri, 2]]
        n0 = F.normalize(torch.cross(b_ - a, c - a, dim=-1), dim=-1)
        col = project_colors(p0, n0, data, 0)
        body = init_body_set(nb, tri, bary, sh_max, cfg.model.body_surfels.init_scale, col)

    free, bones = None, None
    if cfg.model.free.enabled:
        P = init_free_points(cfg, data, sc, no_scaffold)
        if P.shape[0] < 1000:  # tight clothing: keep a small free layer anyway (seeded near the surface)
            extra = dense_surface_points(v_c, sc.faces, 5000, cfg.seed + 1)
            extra = extra + F.normalize(torch.randn_like(extra), dim=-1) * cfg.model.free.d_free_m
            P = torch.cat([P, extra], 0)
        vi, _ = nearest_vertex(P, v_c)
        free = init_free_set(P, vi, sh_max, cfg.model.free.init_scale)
        if int(cfg.model.free.n_bones) > 0:  # n_bones 0 = V0: free Gaussians follow the body LBS only (D-023)
            bones = FreeBones(P, int(cfg.model.free.n_bones), data.num_frames, None if no_scaffold else J_c,
                              mode=cfg.model.free.get("bone_mode", "free"), pose_dim=sc.pose_feature(0).shape[-1],
                              spline_step=int(cfg.model.free.get("spline_step", 4)))

    l2 = None
    if cfg.model.l2.enabled:
        pts = [v_c] + ([free["means"].detach()] if free is not None else [])
        allp = torch.cat(pts, 0)
        pad = 0.3
        pose_in = sc.pose_feature(0).shape[-1]
        if cfg.model.l2.get("type", "mlp") == "cnn":  # D-035: optional 2D-CNN L2 (default stays MLP)
            from sr4d.model.pose_nonrigid_cnn import PoseNonRigidCNN
            l2 = PoseNonRigidCNN(allp.min(0).values - pad, allp.max(0).values + pad, pose_in, cfg.model.l2).to(dev)
        else:
            l2 = PoseNonRigid(allp.min(0).values - pad, allp.max(0).values + pad, pose_in, cfg.model.l2).to(dev)
    l3 = None
    if cfg.model.l3.enabled:
        l3 = InstantResidual(data.num_frames, sc.num_joints, bones.K if bones is not None else 0, cfg.model.l3).to(dev)
    l4 = None
    if cfg.model.l4.enabled:
        l4 = ViewAbsorber(len(data.cam_index), cfg.model.l4).to(dev)
    model = SR4DModel(cfg, sc, body, free, bones, l2, l3, l4).to(dev)
    model.sh_degree = 0
    return model
