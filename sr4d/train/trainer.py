"""SR4D trainer: S1 scaffold fit, then staged S2-S4 optimisation with monitoring and stage checkpoints."""
from __future__ import annotations

import json
import time
from pathlib import Path

import lpips
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from sr4d.data.dataset import MultiViewSequence
from sr4d.model.build import build_model
from sr4d.model.canonical import GaussianSet
from sr4d.model.deform import SR4DModel
from sr4d.model.renderer import composite, compute_filter3d, render
from sr4d.model.scaffold import Scaffold, nearest_vertex, sample_surface, scaffold_kw
from sr4d.train import densify, losses
from sr4d.train.stages import layer_flags, stage_at, stage_table


def psnr(a, b, mask=None):
    if mask is None:
        mse = ((a - b) ** 2).mean()
    else:
        mse = (((a - b) ** 2).mean(-1) * mask).sum() / mask.sum().clamp_min(1)
    return float(-10 * torch.log10(mse.clamp_min(1e-10)))


# ---------------------------------------------------------------- S1
def fit_scaffold(cfg, data: MultiViewSequence, sc: Scaffold, iters: int, log=print):
    """Small multi-view silhouette refinement of the dataset SMPL(-X) (pose, transl, betas).
    Asymmetric: the body leaving the mask is penalised more than the mask not being covered (clothing)."""
    if iters <= 0:
        return
    dev = data.device
    L = cfg.train.loss
    lr = cfg.train.lrs
    sc.betas.requires_grad_(True)
    opt = torch.optim.Adam([g for g in ({"params": sc.pose_params(), "lr": lr.s1_pose}, {"params": sc.transl_params(), "lr": lr.s1_transl},
                            {"params": [sc.betas], "lr": lr.s1_betas}) if g["params"]])
    g = torch.Generator(device=dev).manual_seed(cfg.seed)
    v_c, _, _, _ = sc.canonical()
    tri, bary = sample_surface(v_c.detach(), sc.faces, 30000, g)
    gcpu = torch.Generator().manual_seed(cfg.seed)

    def sil_iou():
        ious = []
        with torch.no_grad():
            for t in range(0, data.num_frames, max(1, data.num_frames // 5)):
                for cid in data.train_ids[::3]:
                    a = silhouette(t, cid) > 0.5
                    m = data.masks[cid][t] > 127
                    ious.append(float((a & m).sum() / (a | m).sum().clamp_min(1)))
        return sum(ious) / len(ious)

    def silhouette(t, cid):
        cam = data.cameras[cid]
        v_c, _, _, _ = sc.canonical()
        v_t = sc.posed_vertices(sc.frame_transforms(t), v_c)[0]
        p = (v_t[sc.faces[tri]] * bary[..., None]).sum(1)
        n = p.shape[0]
        _, alpha, _ = __import__("gsplat").rasterization(
            p, torch.tensor([1.0, 0, 0, 0], device=dev).expand(n, 4), torch.full((n, 3), 0.008, device=dev),
            torch.full((n,), 0.9, device=dev), torch.ones(n, 1, device=dev), cam.w2c[None], cam.K[None], cam.W, cam.H, packed=False)
        return alpha[0, ..., 0]

    iou0 = sil_iou()
    backup = {k: v.detach().clone() for k, v in sc.state_dict().items()}
    for it in range(iters):
        loss = 0
        for t, cid in data.sample(4, gcpu):
            a = silhouette(t, cid)
            m = (data.masks[cid][t] > 127).float()
            loss = loss + L.s1_outside * (a * (1 - m)).mean() * 10 + L.s1_inside * ((1 - a) * m).mean() * 10
        loss = loss / 4 + L.s1_pose_prior * sc.prior_loss()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sc.invalidate()
    iou1 = sil_iou()
    log(f"[S1] scaffold silhouette IoU {iou0:.4f} -> {iou1:.4f} ({iters} iters)")
    if iou1 < iou0:
        sc.load_state_dict(backup)
        log("[S1] IoU did not improve -> keeping the dataset scaffold")
    sc.betas.requires_grad_(False)
    sc.invalidate()
    return {"iou_before": iou0, "iou_after": iou1}


# ---------------------------------------------------------------- trainer
class Trainer:
    def __init__(self, cfg, out_dir: str | Path, device="cuda"):
        self.cfg = cfg
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
        self.dev = device
        torch.manual_seed(cfg.seed)
        self.data = MultiViewSequence(cfg.data.root, cfg.data.setting, list(range(*cfg.data.frames)) if cfg.data.frames else None,
                                      device=device)
        self.tb = SummaryWriter(str(self.out / "tb"))
        self.logf = open(self.out / "log.jsonl", "a")
        self.table = stage_table(cfg)
        self.total = self.table[-1].end if self.table else 0
        self.lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
        for p in self.lpips_fn.parameters():
            p.requires_grad_(False)
        self.gen = torch.Generator().manual_seed(cfg.seed)
        self.model: SR4DModel | None = None
        self.it = 0
        self.t0 = time.time()
        self.train_time = 0.0

    def log(self, msg):
        print(msg, flush=True)
        self.logf.write(json.dumps({"it": self.it, "msg": msg}) + "\n")
        self.logf.flush()

    # ---------------- setup
    def setup(self, resume_stage: str | None = None):
        cfg = self.cfg
        if resume_stage:
            self.load(self.out / f"ckpt_{resume_stage}.pt")
            return
        from sr4d.model.body_model import BodyModel
        bm = BodyModel(self.data.meta["body_model"], self.data.meta.get("gender", "neutral")).to(self.dev)
        sc = Scaffold(bm, self.data.body_params, cfg.model.canonical_pose, **scaffold_kw(cfg)).to(self.dev)
        n1 = int(cfg.train.stages.get("s1", 0) or 0)
        t = time.time()
        s1 = fit_scaffold(cfg, self.data, sc, n1, self.log) if not cfg.model.get("no_scaffold", False) else None
        self.train_time += time.time() - t
        self.model = build_model(cfg, self.data, sc.state_dict())
        self.s1_stats = s1
        self.log(f"[init] body surfels {self.model.n_body}, free {self.model.n_free}, bones {self.model.bones.K if self.model.bones else 0}")
        self._setup_optim()
        self._refresh_filter()

    def _setup_optim(self):
        cfg, m = self.cfg, self.model
        lr = cfg.train.lrs
        if m.body is not None:
            m.body.setup_optimizers({"bary_logits": lr.bary, "offset": lr.offset, "rot_theta": lr.rot, "rot_res": lr.rot,
                                     "log_scales": lr.scales, "opacity": lr.opacities, "sh0": lr.sh0, "shN": lr.shN})
        if m.free is not None:
            m.free.setup_optimizers({"means": lr.means, "quats": lr.rot, "log_scales": lr.scales, "opacity": lr.opacities,
                                     "sh0": lr.sh0, "shN": lr.shN})
        groups = [{"params": m.scaffold.pose_params() + m.scaffold.transl_params(), "lr": cfg.train.scaffold_pose_lr, "name": "scaffold"}]
        if m.bones is not None:
            groups += [{"params": m.bones.motion_parameters(), "lr": lr.bones, "name": "bones"},
                       {"params": [m.bones.log_scale], "lr": lr.bone_scales, "name": "bone_scales"}]
        if m.l2 is not None and self.cfg.model.l2.get("type", "mlp") == "cnn":  # D-035
            groups += [{"params": list(m.l2.parameters()), "lr": lr.l2, "name": "l2"}]
        elif m.l2 is not None:
            groups += [{"params": m.l2.grid.table, "lr": lr.l2_hash, "name": "l2_hash", "eps": 1e-15},
                       {"params": list(m.l2.pose_mlp.parameters()) + list(m.l2.mlp.parameters()), "lr": lr.l2, "name": "l2"}]
        if m.l3 is not None:
            groups += [{"params": [m.l3.z], "lr": lr.l3_z, "name": "l3_z"}, {"params": list(m.l3.dec.parameters()), "lr": lr.l3, "name": "l3"}]
        if m.l4 is not None:
            groups += [{"params": [m.l4.codes] + list(m.l4.mlp.parameters()), "lr": lr.l4, "name": "l4"}]
            if m.l4.camera_refine:
                groups += [{"params": [m.l4.cam_delta], "lr": lr.l4_cam, "name": "l4_cam"}]
            if m.l4.confidence:
                groups += [{"params": [m.l4.conf_logits], "lr": lr.l4_conf, "name": "l4_conf"}]
        self.opt = torch.optim.Adam(groups, eps=1e-15)

    def _active_groups(self, flags):
        act = {"scaffold", "bones", "bone_scales"}
        if flags["l2"]:
            act |= {"l2", "l2_hash"}
        if flags["l3"]:
            act |= {"l3", "l3_z"}
        if flags["l4"]:
            act |= {"l4", "l4_cam", "l4_conf"}
        return act

    @torch.no_grad()
    def _refresh_filter(self):
        m = self.model
        if not self.cfg.model.mip_filter:
            m.filter3d = torch.zeros(0, device=self.dev)
            return
        g = m.deform(self.data.num_frames // 2, replay=False, need_aux=False)
        m.filter3d = compute_filter3d(g["means"], [self.data.cameras[c] for c in self.data.train_ids], self.cfg.model.mip_kernel)

    # ---------------- one render
    def render_view(self, t: int, cid: str, replay: bool, use_l4_code: bool | None = None, l4_mean: bool = False, with_normal=True, bg=None):
        m, data = self.model, self.data
        cam = data.cameras[cid]
        vi = data.cam_index[cid]
        g = m.deform(t, replay=replay, need_aux=self.model.training)
        w2c = cam.w2c
        if m.l4 is not None and m.use_l4 and cam.source == "generated" and not l4_mean:
            w2c = m.l4.refine_w2c(vi, w2c)
        r = render(g, cam.K, w2c, cam.W, cam.H, m.sh_degree, m.filter3d, with_normal=with_normal)
        aff = []
        if g["l3"] is not None:
            aff.append((g["l3"]["color_A"], g["l3"]["color_b"]))
        if m.l4 is not None and m.use_l4 and (use_l4_code is None or use_l4_code):
            aff.append(m.l4.affine(None if l4_mean else vi))
        r["rgb"] = composite(r["rgb_p"], r["alpha"], data.background if bg is None else bg, aff)
        r["g"] = g
        return r

    # ---------------- training
    def train(self):
        cfg, m, data = self.cfg, self.model, self.data
        L, T = cfg.train.loss, cfg.train
        n_s2 = next((s.end - s.start for s in self.table if s.name == "s2"), 0)
        s2_start = next((s.start for s in self.table if s.name == "s2"), 0)
        base_total = next((x.start for x in self.table if x.name == "s7"), self.total)  # D-037: s7 does not shift s2-s4
        reloc_stop = int(base_total * T.relocate_stop_frac)
        cons_mode = cfg.model.l4.get("conf_mode", "learned") == "consistency"  # D-036
        cap = int(cfg.model.cap_total)
        cur_stage = None
        for self.it in range(self.it, self.total):
            it = self.it
            stage = stage_at(self.table, it)
            if stage != cur_stage:
                self.save(self.out / f"ckpt_{stage}.pt")
                cur_stage = stage
                self.log(f"=== stage {stage} at it {it}")
                if stage == "s7" and getattr(m, "detail", None) is None:  # D-037: freeze the shared model, add per-frame detail
                    from sr4d.model.detail import FrameDetail
                    m.detail = FrameDetail(data.num_frames, m.n_body + m.n_free).to(self.dev)
                    dps = [m.detail.dc]
                    if bool(T.get("detail_dx", True)):
                        dps.append(m.detail.dx)
                    else:  # D-039 (d): colour-only detail, dx stays 0
                        m.detail.dx.requires_grad_(False)
                    self.opt_detail = torch.optim.Adam(dps, lr=float(T.get("detail_lr", 1e-4)))
                    self.log(f"detail layer: {data.num_frames} x {m.n_body + m.n_free} Gaussians, dx {'on' if len(dps) == 2 else 'off'}")
            if cons_mode and (it == int(T.get("cons_start", 3000)) or (it > int(T.get("cons_start", 3000)) and it % int(T.get("cons_every", 5000)) == 0)
                              or (stage == "s7" and not getattr(self, "_cons_s7", False))):
                from sr4d.train import consistency
                m.eval(); self.cons_maps = consistency.compute(m, data, floor=float(T.get("cons_floor", 0.1))); m.train()
                self._cons_s7 = stage == "s7"
                self.log(f"consistency confidence refreshed at it {it}")
            flags = layer_flags(cfg, stage)
            m.use_l2, m.use_l3, m.use_l4 = flags["l2"], flags["l3"], flags["l4"]
            m.body_adhere = (it - s2_start) < T.body_adhere_iters
            m.sh_degree = min(cfg.model.body_surfels.max_sh_degree, cfg.model.body_surfels.sh_degree + it // int(T.get("sh_every", 6000))) if it >= 1000 else 0
            frac = min(it / max(base_total, 1), 1.0)
            m.use_detail = flags.get("detail", False)
            means_lr = T.lrs.means * (T.means_lr_final_mult ** frac)
            if m.free is not None:
                m.free.set_lr("means", means_lr)
            if m.body is not None:
                m.body.set_lr("offset", T.lrs.offset * (T.means_lr_final_mult ** frac))
            tic = time.time()
            m.train()

            loss_tot = 0.0
            stats = {}
            batch = data.sample(T.batch_size, self.gen)
            for t, cid in batch:
                replay = m.use_l3 and (torch.rand(1, generator=self.gen).item() > cfg.model.l3.dropout)
                if T.get("random_bg", False):  # D-031: semi-transparent edges can no longer hide in the fixed white
                    bg = torch.rand(3, generator=self.gen).to(self.dev)
                    r = self.render_view(t, cid, replay, bg=bg)
                    _, mask, raw = data.get(cid, t)
                    gt = raw * mask[..., None] + bg * (1 - mask[..., None])
                else:
                    r = self.render_view(t, cid, replay)
                    gt, mask, _ = data.get(cid, t)
                cam = data.cameras[cid]
                conf = None
                if m.l4 is not None and m.use_l4 and cam.source == "generated":
                    conf = m.l4.confidence_map(data.cam_index[cid], cam.H, cam.W)
                wc = None
                if cons_mode and getattr(self, "cons_maps", None) is not None and (cid, t) in self.cons_maps:  # D-036
                    wc = torch.nn.functional.interpolate(self.cons_maps[(cid, t)].to(self.dev)[None, None], size=(cam.H, cam.W), mode="nearest")[0, 0]
                    conf = None  # consistency replaces the learned confidence
                if wc is not None:
                    from sr4d.train import consistency
                    lp = consistency.weighted_photometric(r["rgb"], gt, L.ssim_lambda, wc, consistency.ssim_map)
                else:
                    lp = losses.photometric(r["rgb"], gt, L.ssim_lambda, conf.detach() if conf is not None else None)
                if cam.source == "source":
                    lp = lp * L.source_view_mult
                loss = lp
                if L.lpips > 0 and it >= s2_start + L.lpips_start_frac * n_s2:
                    if wc is not None:
                        if not hasattr(self, "lpips_spatial"):
                            import lpips as _lp
                            self.lpips_spatial = _lp.LPIPS(net="vgg", spatial=True, verbose=False).to(self.dev).eval()
                            for q in self.lpips_spatial.parameters():
                                q.requires_grad_(False)
                        from sr4d.train import consistency
                        loss = loss + L.lpips * consistency.weighted_lpips(self.lpips_spatial, r["rgb"], gt, wc, losses.mask_bbox(mask))
                    else:
                        loss = loss + L.lpips * losses.lpips_loss(self.lpips_fn, r["rgb"], gt, mask)
                loss = loss + L.mask * losses.mask_loss(r["alpha"], mask)
                if L.normal > 0 and "normal" in r:
                    loss = loss + L.normal * losses.normal_consistency(r["normal"], r["depth_normal"], r["alpha"])
                if conf is not None and L.conf > 0:
                    res = (r["rgb"] - gt).abs().mean(-1).detach()
                    loss = loss + L.conf * losses.confidence_loss(res, conf)
                # per-sample priority regularisers
                if m.bones is not None and L.bone_reg > 0:
                    loss = loss + L.bone_reg * m.bones.reg_loss(t, m.scaffold)
                if m.bones is not None and L.get("bone_res_reg", 0) > 0:
                    loss = loss + L.bone_res_reg * m.bones.residual_loss(t, m.scaffold)
                if L.get("pose_smooth", 0) > 0 and not m.no_scaffold:
                    loss = loss + L.pose_smooth * m.scaffold.smooth_loss(t)
                if m.bones is not None and L.get("bone_smooth", 0) > 0:
                    loss = loss + L.bone_smooth * m.bones.smooth_loss(t)
                if r["g"]["l3"] is not None:
                    loss = loss + L.l3_mag * m.l3.mag_loss(r["g"]["l3"]) + L.l3_smooth * m.l3.smooth_loss(t)
                aux = r["g"]["aux"]
                if "l2" in aux and L.l2_mag > 0:
                    loss = loss + L.l2_mag * losses.l2_magnitude(aux["l2"])
                loss = loss / len(batch)
                loss.backward(retain_graph=False)
                loss_tot += float(loss)
                stats["psnr"] = stats.get("psnr", 0) + psnr(r["rgb"].detach(), gt) / len(batch)

            # global regularisers (once per iteration)
            reg = 0.0
            if m.body is not None:
                bs = m.body["log_scales"].exp()
                reg = reg + L.aniso * losses.anisotropy(bs, L.aniso_ratio) + L.opacity_entropy * losses.opacity_entropy(torch.sigmoid(m.body["opacity"]))
                if not m.body_adhere:
                    pos, nrm = losses.body_alignment(m.body["offset"], self._body_normal_dot())
                    reg = reg + L.body_align_pos * pos * 1e4 + L.body_align_normal * nrm  # pos in (cm)^2 units
            if m.free is not None:
                fs = m.free["log_scales"].exp()
                reg = reg + L.aniso * losses.anisotropy(fs, L.aniso_ratio) + L.opacity_entropy * losses.opacity_entropy(torch.sigmoid(m.free["opacity"]))
            if m.use_l2 and L.l2_smooth > 0:
                xs = m.free["means"] if m.free is not None else m.scaffold.canonical()[0]
                reg = reg + L.l2_smooth * losses.l2_smoothness(m.l2, xs, m.scaffold.pose_feature(batch[0][0])[0].detach())
            if m.l4 is not None and m.use_l4:
                reg = reg + L.l4_reg * m.l4.reg_loss()
            if m.use_detail and getattr(m, "detail", None) is not None:  # D-037
                mag, sm = m.detail.reg(sorted({t for t, _ in batch}))
                reg = reg + float(L.get("detail_mag", 0.01)) * mag + float(L.get("detail_smooth", 0.01)) * sm
            if torch.is_tensor(reg):
                reg.backward()

            # steps
            if m.use_detail:  # D-037 s7: only the per-frame detail layer learns
                self.opt_detail.step(); self.opt_detail.zero_grad(set_to_none=True)
                for st_ in (m.body, m.free):
                    if st_ is not None:
                        st_.zero_grad()
                self.opt.zero_grad(set_to_none=True)
            elif m.body is not None:
                keys = None if not m.body_adhere else {"bary_logits", "rot_theta", "log_scales", "opacity", "sh0", "shN"}
                m.body.step(keys)
            if m.free is not None and not m.use_detail:
                m.free.step()
            act = self._active_groups(flags) if not m.use_detail else set()
            for gr in self.opt.param_groups:
                if gr["name"] not in act:
                    for p in gr["params"]:
                        p.grad = None
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            m.scaffold.invalidate()
            if m.l4 is not None and getattr(m.l4, "camera_refine", False) and cfg.model.l4.get("cam_mean_zero", False):
                with torch.no_grad():  # D-031: refinements may not move the whole scene off the original camera frame
                    rows = torch.tensor([data.cam_index[c] for c in data.train_ids if data.cameras[c].source == "generated"], device=self.dev)
                    if len(rows):
                        m.l4.cam_delta.data[rows] -= m.l4.cam_delta.data[rows].mean(0, keepdim=True)

            # densification (MCMC)
            if T.densify == "mcmc":
                if m.free is not None and not m.use_detail:
                    densify.inject_noise_free(m.free, T.mcmc_noise_lr * means_lr)
                if it >= s2_start + 500 and it < reloc_stop and it % T.relocate_every == 0:
                    self._densify(cap)
            self.train_time += time.time() - tic

            if it % 100 == 0:
                self.tb.add_scalar("train/loss", loss_tot, it)
                self.tb.add_scalar("train/psnr", stats["psnr"], it)
            if it % 500 == 0:
                self.log(f"it {it} [{stage}] loss {loss_tot:.4f} psnr {stats['psnr']:.2f} nb {m.n_body} nf {m.n_free} "
                         f"sh {m.sh_degree} {self.train_time / 60:.1f}min" + (f" unseen {getattr(self, 'last_unseen', 0)}" if T.get("prune_unseen", False) else ""))
            if (it % cfg.eval.every == 0 and it > 0) or it == self.total - 1:
                self.monitor()
                self.save(self.out / "ckpt_latest.pt")
        self.it = self.total
        self.save(self.out / "ckpt_final.pt")
        self.log(f"done. train time {self.train_time / 60:.1f} min, peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")

    def _body_normal_dot(self):
        m = self.model
        from sr4d.model.canonical import body_local_rotation
        from sr4d.utils.geometry import quat_to_rotmat
        q = body_local_rotation(m.body["rot_theta"], m.body["rot_res"])
        return quat_to_rotmat(q)[..., 2, 2]  # local frame: z = triangle normal

    @torch.no_grad()
    def _densify(self, cap: int):
        m, T = self.model, self.cfg.train
        eps = self.cfg.model.body_surfels.eps_scale
        nrel = 0
        sets = [s for s in (m.body, m.free) if s is not None]
        if T.get("prune_unseen", False) and m.free is not None and m.n_free > 0:
            # D-029: free Gaussians outside every training image in every frame -> dead, reused by relocate() below
            nk = densify.kill_rows(m.free, densify.unseen_free_mask(m, self.data, replay=m.use_l3), T.opacity_dead)
            self.n_unseen_killed = getattr(self, "n_unseen_killed", 0) + nk
            self.last_unseen = nk
        for s in sets:
            nrel += densify.relocate(s, T.opacity_dead, eps)
        total = sum(len(s) for s in sets)
        room = cap - total
        if room > 0:
            for s in sets:
                n_add = min(int(len(s) * T.grow_rate), int(room * len(s) / total))
                densify.grow(s, n_add, T.opacity_dead, eps)
        if m.free is not None:
            vi, _ = nearest_vertex(m.free["means"].detach(), m.scaffold.canonical()[0].detach())
            m.free.lbs_vidx = vi
        self._refresh_filter()

    # ---------------- monitoring (CLAUDE.md 7.6)
    @torch.no_grad()
    def monitor(self):
        m, data, cfg = self.model, self.data, self.cfg
        m.eval()
        T = data.num_frames
        frames = [int(i * (T - 1) / max(cfg.eval.monitor_frames - 1, 1)) for i in range(cfg.eval.monitor_frames)]
        res = {}
        for split, ids in (("interp", data.interp_ids), ("extrap", data.extrap_ids)):
            ids = ids[: cfg.eval.monitor_cams]
            if not ids:  # generated-only data has no extrapolation cameras (D-024)
                continue
            for mode in ("replay", "l3off"):
                vals = []
                for t in frames:
                    for cid in ids:
                        r = self.render_view(t, cid, replay=(mode == "replay"), l4_mean=True, with_normal=False)
                        gt, mask, _ = data.get(cid, t)
                        vals.append(psnr(r["rgb"], gt))
                res[f"val_{split}_{mode}"] = sum(vals) / len(vals)
        if "val_interp_replay" in res:
            res["gap_L3"] = res["val_interp_replay"] - res["val_interp_l3off"]
        # absorber effect on training views: own code vs discarded (mean code)
        tr = data.train_ids[:: max(1, len(data.train_ids) // cfg.eval.monitor_cams)][: cfg.eval.monitor_cams]
        for name, kw in (("train_l4on", {}), ("train_l4off", {"l4_mean": True})):
            vals = []
            for t in frames:
                for cid in tr:
                    r = self.render_view(t, cid, replay=True, with_normal=False, **kw)
                    vals.append(psnr(r["rgb"], data.get(cid, t)[0]))
            res[name] = sum(vals) / len(vals)
        if m.body is not None:
            res["n_body"] = m.n_body
            res["opac_sum_body"] = float(torch.sigmoid(m.body["opacity"]).sum())
        if m.free is not None:
            res["n_free"] = m.n_free
            res["opac_sum_free"] = float(torch.sigmoid(m.free["opacity"]).sum())
        for k, v in res.items():
            self.tb.add_scalar(f"monitor/{k}", v, self.it)
        self.log("[monitor] " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()))
        with open(self.out / "monitor.jsonl", "a") as f:
            f.write(json.dumps({"it": self.it, **res}) + "\n")
        m.train()
        return res

    # ---------------- checkpoints
    def save(self, path: Path):
        m = self.model
        st = {
            "it": self.it,
            "cfg": OmegaConf.to_container(self.cfg),
            "scaffold": m.scaffold.state_dict(),
            "body": m.body.state_dict_full() if m.body is not None else None,
            "free": m.free.state_dict_full() if m.free is not None else None,
            "bones": m.bones.state_dict() if m.bones is not None else None,
            "l2": m.l2.state_dict() if m.l2 is not None else None,
            "l3": m.l3.state_dict() if m.l3 is not None else None,
            "l4": m.l4.state_dict() if m.l4 is not None else None,
            "detail": m.detail.state_dict() if getattr(m, "detail", None) is not None else None,
            "filter3d": m.filter3d,
            "sh_degree": m.sh_degree,
            "train_time": self.train_time,
        }
        torch.save(st, path)

    def load(self, path: Path):
        self.model = load_model(path, self.data, self.cfg)
        st = torch.load(path, map_location=self.dev, weights_only=False)
        self.it = st["it"]
        self.train_time = st.get("train_time", 0.0)
        self._setup_optim()


def load_model(path: Path, data: MultiViewSequence, cfg=None) -> SR4DModel:
    st = torch.load(path, map_location=data.device, weights_only=False)
    cfg = cfg or OmegaConf.create(st["cfg"])
    model = build_model_skeleton(cfg, data, st)
    model.filter3d = st["filter3d"]
    model.sh_degree = st["sh_degree"]
    return model


def build_model_skeleton(cfg, data, st) -> SR4DModel:
    """Rebuild modules with the checkpoint's sizes (no carving) and load weights."""
    from sr4d.model.body_model import BodyModel
    from sr4d.model.free_bones import FreeBones
    from sr4d.model.instant_residual import InstantResidual
    from sr4d.model.pose_nonrigid import PoseNonRigid
    from sr4d.model.view_absorber import ViewAbsorber

    dev = data.device
    bm = BodyModel(data.meta["body_model"], data.meta.get("gender", "neutral")).to(dev)
    sc = Scaffold(bm, data.body_params, cfg.model.canonical_pose, **scaffold_kw(cfg)).to(dev)
    sc.load_state_dict(st["scaffold"])
    sc.betas.requires_grad_(False)
    body = GaussianSet.from_state("body", st["body"]) if st["body"] is not None else None
    free = GaussianSet.from_state("free", st["free"]) if st["free"] is not None else None
    bones = None
    if st["bones"] is not None:
        K = st["bones"]["centers"].shape[0]
        bones = FreeBones(torch.randn(K * 50, 3, device=dev), K, data.num_frames, None,
                          mode=cfg.model.free.get("bone_mode", "free"), pose_dim=sc.pose_feature(0).shape[-1],
                          spline_step=int(cfg.model.free.get("spline_step", 4)))
        bones.load_state_dict(st["bones"])
    l2 = None
    if st["l2"] is not None:
        if cfg.model.l2.get("type", "mlp") == "cnn":  # D-035
            from sr4d.model.pose_nonrigid_cnn import PoseNonRigidCNN
            l2 = PoseNonRigidCNN(st["l2"]["bmin"], st["l2"]["bmax"], sc.pose_feature(0).shape[-1], cfg.model.l2).to(dev)
        else:
            l2 = PoseNonRigid([0, 0, 0], [1, 1, 1], sc.pose_feature(0).shape[-1], cfg.model.l2).to(dev)
        l2.load_state_dict(st["l2"])
    l3 = None
    if st["l3"] is not None:
        l3 = InstantResidual(data.num_frames, sc.num_joints, bones.K if bones is not None else 0, cfg.model.l3).to(dev)
        l3.load_state_dict(st["l3"])
    l4 = None
    if st["l4"] is not None:
        l4 = ViewAbsorber(len(data.cam_index), cfg.model.l4).to(dev)
        l4.load_state_dict(st["l4"])
    m = SR4DModel(cfg, sc, body, free, bones, l2, l3, l4).to(dev)
    if st.get("detail") is not None:  # D-037
        from sr4d.model.detail import FrameDetail
        m.detail = FrameDetail(*st["detail"]["dx"].shape[:2]).to(dev)
        m.detail.load_state_dict(st["detail"])
    return m

