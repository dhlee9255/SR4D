"""M1 environment check: GPU, gsplat (3DGS + 2DGS), SMPL-X forward, LPIPS.

Usage: conda activate sr4d && python scripts/check_env.py
"""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def check_gpu():
    assert torch.cuda.is_available(), "CUDA not available"
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    print(f"[gpu] torch {torch.__version__} cuda {torch.version.cuda}, {n} GPUs: {names}")


def check_gsplat():
    import gsplat

    dev = "cuda"
    N = 1000
    means = torch.randn(N, 3, device=dev) * 0.3
    quats = torch.nn.functional.normalize(torch.randn(N, 4, device=dev), dim=-1)
    scales = torch.full((N, 3), 0.02, device=dev)
    opac = torch.full((N,), 0.8, device=dev)
    colors = torch.rand(N, 3, device=dev)
    viewmat = torch.eye(4, device=dev)[None]
    viewmat[0, 2, 3] = 3.0
    K = torch.tensor([[300.0, 0, 128], [0, 300.0, 128], [0, 0, 1]], device=dev)[None]
    t0 = time.time()
    img, alpha, _ = gsplat.rasterization(means, quats, scales, opac, colors, viewmat, K, 256, 256)
    torch.cuda.synchronize()
    print(f"[gsplat] {gsplat.__version__} 3DGS render ok {tuple(img.shape)} alpha_max={alpha.max():.3f} ({time.time()-t0:.1f}s incl. JIT)")
    has_2dgs = hasattr(gsplat, "rasterization_2dgs")
    if has_2dgs:
        out = gsplat.rasterization_2dgs(means, quats, scales, opac, colors[None], viewmat, K, 256, 256)
        print(f"[gsplat] 2DGS render ok, outputs={len(out)} (render_colors {tuple(out[0].shape)}, normals {tuple(out[2].shape)})")
    else:
        print("[gsplat] rasterization_2dgs NOT available")
    return has_2dgs


def check_smplx():
    import smplx

    for mt, sub in [("smplx", "smplx"), ("smpl", "smpl")]:
        p = ROOT / "assets/body_models" / sub
        if not any(p.iterdir()):
            print(f"[{mt}] no model files in {p}, skipped")
            continue
        kw = dict(use_pca=False, flat_hand_mean=True, num_betas=10) if mt == "smplx" else {}
        m = smplx.create(str(ROOT / "assets/body_models"), model_type=mt, gender="neutral", ext="npz" if mt == "smplx" else "pkl", **kw)
        out = m()
        print(f"[{mt}] forward ok verts={tuple(out.vertices.shape)} joints={tuple(out.joints.shape)}")


def check_lpips():
    import lpips

    fn = lpips.LPIPS(net="vgg", verbose=False).cuda()
    a = torch.rand(1, 3, 64, 64, device="cuda") * 2 - 1
    d = fn(a, a).item()
    print(f"[lpips] vgg ok, self-distance={d:.4f}")


if __name__ == "__main__":
    check_gpu()
    check_gsplat()
    check_smplx()
    check_lpips()
    print("ALL CHECKS PASSED")
    sys.exit(0)
