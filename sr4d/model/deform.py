"""SR4D model: composes L0..L4 and deforms canonical Gaussians into frame t (CLAUDE.md 7.3).

Order:
  1. X <- X + Delta_L2(X, p_t)                 (position / rotation / scale / colour)
  2. free only: X <- sum_k w_k(X) B_{k,t} X     (free bones, canonical space = relative to parent joint)
  3. all: X <- LBS(X; G_t)                       (body surfels: posed-triangle frames, exact surface tracking)
  4. replay only: L3 residuals (applied as joint / bone SE(3) residuals inside 2-3, colour affine after render)
Covariances are carried as full 3x3 matrices (Sigma' = M Sigma M^T) and handed to gsplat via `covars`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sr4d.model.canonical import C0, GaussianSet, body_local_rotation
from sr4d.model.free_bones import FreeBones
from sr4d.model.instant_residual import InstantResidual
from sr4d.model.pose_nonrigid import PoseNonRigid
from sr4d.model.scaffold import Scaffold
from sr4d.model.view_absorber import ViewAbsorber
from sr4d.utils.geometry import quat_to_rotmat, triangle_frames


class SR4DModel(nn.Module):
    def __init__(self, cfg, scaffold: Scaffold, body: GaussianSet | None, free: GaussianSet | None,
                 bones: FreeBones | None, l2: PoseNonRigid | None, l3: InstantResidual | None, l4: ViewAbsorber | None):
        super().__init__()
        self.cfg = cfg
        self.scaffold = scaffold
        self.body = body
        self.free = free
        self.bones = bones
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        self.no_scaffold = bool(cfg.model.get("no_scaffold", False))
        self.eps_scale = cfg.model.body_surfels.eps_scale
        self.use_l2 = False
        self.l2_no_dx = bool(cfg.model.l2.get("no_dx", False))  # ablation / diagnostic: L2 keeps dq, dlog_s, dcolor but not dx
        self.use_l3 = False
        self.use_l4 = False
        self.body_adhere = True  # S2 first iters: surfels pinned to the mesh (only barycentrics learn)
        self.sh_degree = 0
        self.register_buffer("filter3d", torch.zeros(0))

    @property
    def n_body(self):
        return len(self.body) if self.body is not None else 0

    @property
    def n_free(self):
        return len(self.free) if self.free is not None else 0

    def instant(self, t: int, replay: bool):
        if replay and self.use_l3 and self.l3 is not None:
            return self.l3(t)
        return None

    def frame_G(self, t: int, l3o: dict | None):
        J = self.scaffold.num_joints
        if self.no_scaffold:
            return torch.eye(4, device=self.scaffold.pose_init.device).expand(J, 4, 4)
        pd = l3o["joint_rot"][None] if l3o is not None else None
        td = l3o["transl"][None] if l3o is not None else None
        return self.scaffold.frame_transforms(t, pd, td)[0]

    def deform(self, t: int, replay: bool = True, l3o: dict | None = None, need_aux: bool = True):
        sc = self.scaffold
        if l3o is None:
            l3o = self.instant(t, replay)
        G = self.frame_G(t, l3o)
        v_c, J_c, _, _ = sc.canonical()
        faces = sc.faces
        pose_feat = sc.pose_feature(t)[0].detach()
        means, covs, opac, shs, normals = [], [], [], [], []
        aux = {}

        # ---- canonical attributes before L2
        if self.n_body > 0:
            b = self.body
            tri = b.tri
            bary = torch.softmax(b["bary_logits"], -1)
            Fc_all, _ = triangle_frames(v_c, faces)
            Fc = Fc_all[tri]
            base_c = (v_c[faces[tri]] * bary[..., None]).sum(1)
            offset = b["offset"] if not self.body_adhere else b["offset"].detach() * 0
            xb_c = base_c + (Fc @ offset[..., None])[..., 0]
        if self.n_free > 0:
            xf_c = self.free["means"]

        # ---- L2 (one call for both sets)
        d2 = None
        if self.use_l2 and self.l2 is not None:
            xs = []
            if self.n_body > 0:
                xs.append(xb_c)
            if self.n_free > 0:
                xs.append(xf_c)
            if getattr(self.l2, "H", None) is not None:  # D-035 CNN L2: needs current-pose LBS positions of every Gaussian
                vid = []
                if self.n_body > 0:
                    vid.append(faces[self.body.tri][:, 0])
                if self.n_free > 0:
                    vid.append(self.free.lbs_vidx)
                Tl = torch.einsum("nj,jxy->nxy", sc.lbs_weights[torch.cat(vid, 0)], G)
                xc = torch.cat(xs, 0)
                xp = (Tl[:, :3, :3] @ xc[..., None])[..., 0] + Tl[:, :3, 3]
                d2 = self.l2(xc, pose_feat, x_posed=xp.detach(), G0=G[0].detach())
            else:
                d2 = self.l2(torch.cat(xs, 0), pose_feat)
            if self.l2_no_dx:
                d2 = {**d2, "dx": torch.zeros_like(d2["dx"])}
            if need_aux:
                aux["l2"] = d2

        def l2_slice(lo, hi):
            return None if d2 is None else {k: v[lo:hi] for k, v in d2.items()}

        # ---- body surfels: exact tracking of posed triangles
        if self.n_body > 0:
            b = self.body
            d = l2_slice(0, self.n_body)
            q_loc = body_local_rotation(b["rot_theta"], b["rot_res"] if not self.body_adhere else b["rot_res"].detach())
            R_c = Fc @ quat_to_rotmat(q_loc)
            s = torch.cat([b["log_scales"].exp(), torch.full_like(b["log_scales"][:, :1], self.eps_scale)], -1)
            sh0 = b["sh0"]
            off = offset
            if d is not None:
                off = off + (Fc.transpose(-1, -2) @ d["dx"][..., None])[..., 0]
                R_c = quat_to_rotmat(d["dq"]) @ R_c
                s = torch.cat([s[:, :2] * d["dlog_s"][:, :2].exp(), s[:, 2:]], -1)
                sh0 = sh0 + d["dcolor"][:, None, :] / C0
            if self.no_scaffold:
                v_t = v_c
            else:
                v_t = sc.posed_vertices(G[None], v_c)[0]
            Ft_all, _ = triangle_frames(v_t, faces)
            Ft = Ft_all[tri]
            base_t = (v_t[faces[tri]] * bary[..., None]).sum(1)
            x_t = base_t + (Ft @ off[..., None])[..., 0]
            R_t = Ft @ Fc.transpose(-1, -2) @ R_c
            cov = R_t @ torch.diag_embed(s * s) @ R_t.transpose(-1, -2)
            means.append(x_t)
            covs.append(cov)
            opac.append(torch.sigmoid(b["opacity"][:, 0]))
            shs.append(torch.cat([sh0, b["shN"]], 1))
            normals.append(R_t[..., 2])
            if need_aux:
                aux["body_offset"] = off
                aux["body_normal_dot"] = (R_c[..., 2] * Fc[..., 2]).sum(-1)
                aux["body_scales"] = s

        # ---- free Gaussians: L2 -> bones -> LBS
        if self.n_free > 0:
            f = self.free
            d = l2_slice(self.n_body, self.n_body + self.n_free)
            R = quat_to_rotmat(f["quats"])
            s = f["log_scales"].exp()
            x = xf_c
            sh0 = f["sh0"]
            if d is not None:
                x = x + d["dx"]
                R = quat_to_rotmat(d["dq"]) @ R
                s = s * d["dlog_s"].exp()
                sh0 = sh0 + d["dcolor"][:, None, :] / C0
            M = R
            if self.bones is not None:
                wb = self.bones.weights(xf_c.detach())
                B = self.bones.transforms(t, l3o["bone_rot"] if l3o else None, l3o["bone_trans"] if l3o else None, scaffold=sc)
                Tb = torch.einsum("nk,kxy->nxy", wb, B)
                x = (Tb[:, :3, :3] @ x[..., None])[..., 0] + Tb[:, :3, 3]
                M = Tb[:, :3, :3] @ M
            if not self.no_scaffold:
                w = sc.lbs_weights[f.lbs_vidx]
                Tl = torch.einsum("nj,jxy->nxy", w, G)
                x = (Tl[:, :3, :3] @ x[..., None])[..., 0] + Tl[:, :3, 3]
                M = Tl[:, :3, :3] @ M
            cov = M @ torch.diag_embed(s * s) @ M.transpose(-1, -2)
            amin = s.argmin(-1)
            n = torch.gather(M, 2, amin[:, None, None].expand(-1, 3, 1))[..., 0]
            means.append(x)
            covs.append(cov)
            opac.append(torch.sigmoid(f["opacity"][:, 0]))
            shs.append(torch.cat([sh0, f["shN"]], 1))
            normals.append(F.normalize(n, dim=-1))
            if need_aux:
                aux["free_scales"] = s

        means_all, shs_all = torch.cat(means, 0), torch.cat(shs, 0)
        det = getattr(self, "detail", None)
        if det is not None and getattr(self, "use_detail", True):  # D-037 per-frame detail (s7 and later renders)
            means_all = means_all + det.dx[t]
            shs_all = torch.cat([shs_all[:, :1] + det.dc[t][:, None] / C0, shs_all[:, 1:]], 1)
        out = {
            "means": means_all,
            "covars": torch.cat(covs, 0),
            "opacities": torch.cat(opac, 0),
            "shs": shs_all,
            "normals": torch.cat(normals, 0),
            "l3": l3o,
            "aux": aux,
        }
        return out
