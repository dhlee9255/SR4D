# PROGRESS

## 2026-10-08 (session 2) — previous-project checklist applied, new results

### New A-stage results (ZJU-387 real12, common evaluation)
| method | interp PSNR / LPIPS / fg / fgi | extrap PSNR / LPIPS / fg / fgi | tOF@5 (i/e) |
|---|---|---|---|
| **SR4D full** | **25.01** / 0.038 / 15.51 / **19.90** | 23.51 / 0.051 / 14.35 / **18.39** | 1.24 |
| SR4D short_s4 (S4 6k) | 25.03 / 0.038 / 15.52 / 19.79 | **23.57** / 0.051 / 14.52 / 18.65 | 1.37 / 1.25 |
| SR4D posesmooth (pose lr 1e-4 + D-017) | 24.97 / 0.037 / 15.43 / 19.91 | 23.56 / 0.051 / 14.35 / 18.47 | 1.33 / 1.33 |
| (d) no_l4 | 24.92 / **0.036** / 15.56 / 20.20 | 23.27 / **0.048** / 14.23 / 17.96 | 1.30 / 1.27 |
| 4DGS (prune unblocked, D-018) | 24.23 / 0.041 / 14.64 / 19.35 | 22.63 / 0.051 / 13.69 / 17.31 | 1.29 / **1.11** |
| DualGS | 24.04 / 0.046 / 14.46 / 17.84 | 21.81 / 0.061 / 12.34 / 14.24 | 1.40 / 1.32 |
| FTGS++ black bg (official training) | 17.44 / 0.271 / **16.46** / 19.37 | 16.71 / 0.282 / **15.05** / 17.51 | 1.49 / 1.26 |
| FTGS++ raw bg (official training) | (n/a: bg) / – / 16.26 / 16.43 | (n/a) / – / 14.51 / 14.52 | 1.41 / 1.18 |
- 4DGS with pruning unblocked: essentially unchanged (+0.03/+0.07 dB) — the prune gate was not the reason it trails SR4D.
- DualGS (81-cam design) is the weakest on extrap in this 12-view setting.
- **FTGS++ black/raw**: on the mask-pixel `fg PSNR`, FTGS++ black beats SR4D (+0.95/+0.70 dB); on `fg-interior` (mask eroded 3 px) SR4D wins (+0.53/+0.88), and on fg LPIPS (0.41 vs 0.15), fg SSIM (0.59 vs 0.81) and full-image metrics SR4D wins by a large margin (FTGS++ leaves floaters outside the mask, composited by its own alpha). Likely reason for the fg-PSNR flip: ZJU masks are loose, so the pixels just inside the mask are dark background; a model trained on black predicts dark there, a white-trained one predicts white. → **The main-table FTGS++ protocol is a user decision** (below).
- L4 on real data: removing it improves LPIPS slightly but costs −0.24 dB extrap PSNR / −0.43 dB extrap fgi → keep.
- Shorter S4 (6k) is as good as 18k (extrap +0.06) → candidate default for dev runs.

### Checklist from the previous project (Recon Ceiling Live)
0. Legacy scripts copied to `tools/legacy/` (`build_ftgspp_dynerf.py`, `render_orbit2.py`, `register_source_camera.py`, `zju_387_camB5_camera_registration.json`). `fg_verify.py` was not found anywhere under /data/AI-Graphics/Applications.
1. **FTGS++ export** (`tools/check_ftgspp_export.py`): our poses_bounds == legacy builder (pose columns diff 0, both round-trip through FTGS++'s own parser to 9e-16; only near 0.1 vs 0.3). The camera drop is **not** a pose flip — it is the cheirality formula `w2c[:3,2]·(X−w2c[:3,3])` (column instead of row). Official formula on the 4DAnyone gen24 ring: 13/24 cameras pass, 11 fully dropped → the ring still trains because the remaining half covers 360°, consistent with "FTGS++ trained fine before". On ZJU: 0/12 pass → the fix (D-012) is required there. Background variants black/raw now run with the **official** `ftgspp.train` (D-021). Init stays RoMa.
2. **render.py FOV bug** confirmed (`render.py:344`, camera-0 fx). Our numbers are **not** affected: `ftgspp_render.py` has always used each camera's own K. The training loader uses per-image K (`multiview.py:310` → `train.py:122`). Undistortion happens at conversion (dist = 0 in cameras.json; ZJU fx 513–555 differs per camera).
3. **4DGS export**: init colours are in [0,1] (constant 0.5 grey; Panoptic reader takes them as-is); **pruning was gated by `>200000` and never ran** (model ~52–75k) → fixed via env (D-018), re-run above; masked background compositing: yes (white).
4. **Init frame**: `tools/check_init_in_masks.py` — FTGS++ RoMa 90%, 4DGS/DualGS hull+SMPL 97% of points inside all 12 training masks (3 px tolerance). We don't call `build_sparse_point_cloud`, so there is no Z-up issue.
5. **B-stage registration** — found a real bug (D-019): `BodyModel.forward` returned wrong joint positions, so the joint-Umeyama registration was wrong. Fixed → `387_gen24_v2` (no evaluation data used): scale 1.008, joint residual 4.1 cm, **real-eval-camera IoU 0.827**, RTMW-x keypoint error 6–10 px. The previous "re-registered" data (`387_gen24_reg`) was a source-camera MSE-silhouette 7DoF fit on the buggy init → superseded. New `tools/register_real_cams.py` (R_init·exp(so3), Dice + RTMW-x keypoint Huber, no MSE): source-only similarity over-fits (eval IoU 0.72, not adopted); per-camera pose 0.850 (uses eval data → sensitivity only). See D-020.
6. **sdc1** is another user's external NTFS drive (label "새 볼륨", owner seokhyunjeong), already mounted rw at `/mnt/ext_ssd` by someone else, 97% full. Not touched, not used. Large FTGS++ memmaps are deleted after each run; root NVMe has ~113 GB free.

### Running now
- GPU0 SR4D real8, GPU1 SR4D real_all (ZJU-387; ~2.5 h). Baselines for real8/real_all to follow.

### B-stage (ZJU-387 gen24 → real eval cams), re-evaluated on v2 registration (same trained models, g* cams unchanged)
Eval cams `387_gen24_v2` (joint Umeyama with the D-019 fix, no evaluation data; real-cam IoU 0.827). Results in `results/B_v2_gen24.csv`.
| method | interp PSNR / LPIPS / fg / fgi | extrap PSNR / LPIPS / fg / fgi |
|---|---|---|
| SR4D (trained earlier, unchanged) | **20.37** / 0.057 / 10.76 / 13.45 | **20.52** / 0.058 / 12.17 / 15.56 |
| 4DGS | 20.36 / 0.057 / 10.87 / 13.29 | 20.45 / 0.057 / 12.38 / 15.30 |
| FTGS++ white bg + init fix | 20.24 / **0.054** / 10.74 / 13.20 | 18.50 / 0.065 / 9.11 / 11.12 |
| **FTGS++ legacy setting** (black bg, official init) | 19.52 / 0.138 / **11.51** / **13.73** | 19.27 / 0.134 / **13.07** / **15.65** |
- **Registration check passed**: the legacy-setting FTGS++ reaches fg PSNR 11.5/13.1 (mean ≈ 12.3), matching the previous project's expected ≈ 12.5 at 24 views. (Old registration: every method had fg ≈ 6–8.)
- With per-camera pose alignment (`387_gen24_v2pc`, `results/B_v2pc_gen24.csv`) the ordering is the same: SR4D 10.99/11.62, 4DGS 11.24/12.09, FTGS++ legacy 11.99/13.05 fg PSNR.
- **Honest summary of B-stage on ZJU**: SR4D ≈ 4DGS; the legacy-setting FTGS++ is best on fg PSNR / fgi (+0.3–0.9 dB) but worst on LPIPS (fg LPIPS 0.34 vs 0.22) and full-image PSNR. SR4D has not been tuned for B-stage yet (L4 camera refine + confidence on; trained before the registration fix — training does not depend on the real cameras, so the model itself is still valid). **SR4D does not yet win the B-stage → I will not change the design before discussing it (CLAUDE.md §11).**

## 2026-10-08 (session 1)

### Status
| ID | Item | Status |
|---|---|---|
| M0 | Environment survey report | ✅ `docs/ENV_REPORT.md`. **Storage paths need user confirmation** (see below) |
| M1 | `sr4d` env | ✅ `scripts/check_env.py` passes (GPU, gsplat 3DGS+2DGS, SMPL-X/SMPL forward, LPIPS). `environment.yml`, `requirements.txt` saved |
| M2 | DNA-Rendering download + conversion | ⛔ **No data** (application + license needed → user). Converter `tools/convert_dna_rendering.py` is written (format per the official reader code, untested). **Substitute: ZJU-MoCap 387** converted, verified and split |
| M3 | Baseline reproduction | 🔄 FTGS++ / DualGS / 4DGS export, run, render and common-eval pipelines all built; running on ZJU-387 `real12` |
| M4 | SR4D S1–S2 | 🔄 Full implementation done (S1–S4, L0–L4), full run (54k it) on ZJU-387 `real12` in progress |
| M5–M7 | | not started |

### Data (substitute dev set)
- `data_dev/zju_mocap/387`: 23 cams × 60 frames, 512×512 (1/2), undistortion + principal-point centering (D-005).
- `verify_dataset.py`: SMPL-vs-mask mean IoU **0.785** (min 0.722); the overlay alignment is correct (the hoodie makes the mask larger than SMPL).
- splits: `real_all`(17) / `real12` / `real8`; test_interp {01,09,16} (nearest training camera ~13°), test_extrap {06,07,08} (~25°).

### Implementation summary (SR4D)
- `sr4d/model`: `body_model` (own LBS, SMPL/SMPL-X), `scaffold` (L0), `canonical` (L1 GaussianSet: body surfels = triangle+barycentric+local offset, free Gaussians), `free_bones` (k-means K=25, Mahalanobis softmax, per-frame SE(3)), `pose_nonrigid` (L2, dense grid + pose MLP), `instant_residual` (L3, z_t∈R^24 → joint/bone SE(3) + color affine, dropout 0.3), `view_absorber` (L4, per-view code → color affine, camera refine / confidence options), `deform` (L2 → bones → LBS → L3 order, covariance Σ'=MΣMᵀ), `renderer` (gsplat single pass, RGB+normal+depth, Mip 3D filter), `build` (space carving → inverse LBS → d_free filter → 5 mm voxel downsampling)
- `sr4d/train`: `trainer` (S1 silhouette refit → S2/S3/S4, stage checkpoints, `--resume_stage`, monitoring of gap_L3 / PSNR(L4 off) / per-set count and opacity sum every 2k it), `losses` (full table from CLAUDE.md 7.6), `densify` (per-set MCMC)
- `sr4d/eval`: `render_cams`, `metrics` (full/fg PSNR·SSIM·LPIPS, temporal, CSV), `nearest_view` (angle-vs-PSNR plot), `report` (md table)
- Speed (4090, 512², ~400k Gaussians): S2 ~8 it/s, S3/S4 ~6 it/s, peak 13.5 GB.

### Problems found and fixed
- **FTGS++ init cheirality bug** (D-012): first FTGS run on ZJU → only 411 Gaussians → stopped at 600 it. Fixed by an in-memory patch wrapper. **The same bug drops points from ~half the cameras on the 4DAnyone ring too.**
- **Black-background training is not equivalent to white** (D-011): FTGS++ trained on black produced many dark floaters (PSNR 17.0/16.3) → white background injected in memory for both FTGS++ and 4DGS, then retrained.
- 4DGS `export_perframe_3DGS.py` crashes on Panoptic cameras → own exporter.

### ★ SR4D vs baselines (ZJU-387, real12 = 12 training cameras, 60 frames, 512², white-background protocol, common evaluation)
| method | split | PSNR | SSIM | LPIPS | fg PSNR | fg-interior PSNR | temporal↓ | train(min) | #G |
|---|---|---|---|---|---|---|---|---|---|
| **SR4D (S1–S4)** | interp | **25.01** | **0.959** | **0.038** | **15.51** | **19.90** | 0.0550 | 161 | 400k |
| SR4D S2-only (abl. a) | interp | 24.79 | 0.957 | 0.040 | 15.36 | 19.33 | 0.0592 | 65 | 400k |
| 4DGS | interp | 24.20 | 0.958 | 0.041 | 14.60 | 19.20 | **0.0441** | ~17 | 75k |
| **SR4D (S1–S4)** | extrap | 23.51 | 0.942 | **0.051** | 14.35 | 18.39 | 0.0588 | | |
| SR4D S2-only | extrap | **23.55** | 0.942 | 0.055 | **14.73** | **19.08** | 0.0603 | | |
| 4DGS | extrap | 22.56 | **0.946** | 0.052 | 13.64 | 17.07 | **0.0436** | | |
- Per-camera PSNR: SR4D beats 4DGS on all 6 evaluation cameras (interp 01/09/16: +0.6/+0.9/+1.0; extrap 06/07/08: +0.9/+1.2/+0.8). Plot: `results/figures/angle_387_real12.png`.
- **First decision point (CLAUDE.md §9-2): S2 alone gives test_extrap 23.55 vs 4DGS 22.56 (+1.0 dB), and FTGS++ collapses under this protocol** → passed.
- L2+L3 (S3–S4): interp +0.22 dB / LPIPS improved, but extrap interior PSNR drops 19.08 → 18.39 (late overfitting; the monitor showed extrap falling from 23.6 to 23.2 during S4). gap_L3 ≈ 0 (L3 does not eat the canonical). The no_l3 ablation will separate the cause.
- **Temporal (D-016)**: on the CSV `temporal`, SR4D 0.055 vs 4DGS 0.044, but **this metric favors still/blurry renders** (static render = 0.041; 4DGS moves only half as much as GT). On the newly added tOF@5 (RAFT, static = 2.25 px): SR4D 1.24 vs 4DGS 1.14 px → 4DGS's motion fidelity is slightly better (suspected cause: SR4D is pinned to the ZJU SMPL poses). `bone_smooth` (D-015) had no effect.

### Ablations (M5, ZJU-387 real12; ablation (d) no_l4 / (e) joint in progress)
| variant | interp PSNR / LPIPS / fgi | extrap PSNR / LPIPS / fgi | temporal (i/e) | it |
|---|---|---|---|---|
| (a) S2 only | 24.79 / 0.040 / 19.33 | 23.55 / 0.055 / 19.08 | 0.059 / 0.060 | 24k |
| (b) +L2 (no_l3) | **25.08** / **0.038** / **19.91** | **23.76** / **0.051** / 19.04 | 0.056 / 0.058 | 36k |
| (c) full (+L3, +L4) | 25.01 / 0.038 / 19.90 | 23.51 / 0.051 / 18.39 | 0.055 / 0.059 | 54k |
| (e) joint (all layers from S2 start) | **25.14** / 0.039 / 19.74 | 23.33 / 0.052 / **17.41** | 0.048 / 0.050 (tOF@5 1.21/1.15) | 54k |
| (b') L2 only, 54k (no_l3_long) | 24.97 / 0.037 / 19.75 | 23.57 / **0.050** / 18.19 | 0.056 / 0.059 (tOF@5 1.33/1.23) | 54k |
| (f) no scaffold (free bones only, 64 bones) | 24.40 / 0.040 / 19.44 | 23.15 / 0.055 / 18.69 | 0.071 / 0.072 | 54k |
| 4DGS (ref.) | 24.20 / 0.041 / 19.20 | 22.56 / 0.052 / 17.07 | 0.044 / 0.044 | 20k |
- L2 helps on both splits (+0.3/+0.2 dB). **The extrap drop comes mostly from training length** (no_l3 36k → no_l3_long 54k: −0.19 dB; adding L3: −0.06 dB). Testing a shorter S4 (6k) → `short_s4` running.
- **Staged vs joint training (e)**: joint wins interp (+0.13) but is worst on extrap (interior PSNR 17.41 vs staged 18.39) → **the CLAUDE.md principle of "fit the widely shared layers first" is supported by this result.**
- The scaffold accounts for +0.6/+0.4 dB and much of the temporal stability (0.071 → 0.055).

### Baseline details (same conditions)
| method | split | PSNR | SSIM | LPIPS | fg PSNR | fg-interior PSNR | temporal | train(min) | #G |
|---|---|---|---|---|---|---|---|---|---|
| 4DGS | interp | 24.20 | 0.958 | 0.041 | 14.60 | 19.20 | 0.0441 | ~17 | 75k |
| 4DGS | extrap | 22.56 | 0.946 | 0.052 | 13.64 | 17.07 | 0.0436 | | |
| FTGS++ (white bg) | interp | 16.07 | 0.928 | 0.077 | 4.63 | 5.21 | 0.0473 | 11.5 | 500k |
| FTGS++ (white bg) | extrap | 13.75 | 0.917 | 0.106 | 2.40 | 2.61 | 0.0407 | | |
| FTGS++ (black bg) | interp | 17.00 | 0.858 | 0.273 | 16.54 | – | 0.0460 | | |
| FTGS++ (black bg) | extrap | 16.28 | 0.849 | 0.280 | 15.17 | – | 0.0474 | | |
- **FTGS++ failure mode**: training views reach 36 dB, but test views (12-view sparse setting) are covered by white (or dark, for black bg) floaters → the body disappears. FTGS++ has no alpha/mask supervision, so on a fixed-color background the floaters cost nothing in the training views. In its native 4DAnyone setting (raw background) the background constrains them. → **Protocol question for the user** (below).
- ZJU masks have inaccurate boundaries, so 98% of SR4D's training-view error sits on mask-edge pixels (3%) (interior PSNR 34.9). This is why `fg-interior PSNR` (mask eroded by 3 px) was added to the JSON as a diagnostic.
- DualGS: about 8 min/frame (README default 15000 it/frame) → long run in progress.

### Waiting for user decisions (updated session 2)
0. **sdc1 = another user's external NTFS drive** (mounted at /mnt/ext_ssd by someone else, 97% full) → not usable; please name a work-data location if root NVMe (113 GB free) is not enough.
1. **Storage**: root NVMe has 15 GB free; `/mnt/ssd_nas` and `/mnt/Dataset2` are full. Can `sdc1` (3.6T, unmounted) be mounted as the SSD work drive? Is `/mnt/nas` (2.1T free) OK for raw DNA-Rendering data?
2. **DNA-Rendering**: please obtain the data (application at dna-rendering.github.io). First 10 test sequences, `main` + `annotations` smc.
3. (FYI) The FTGS++ cheirality bug may affect the existing 4DAnyone FTGS++ results.
4. **FTGS++ protocol**: under the masked + fixed-background protocol (CLAUDE.md §6) FTGS++ collapses from floaters. (a) keep it as is (report the failure), (b) give FTGS++ alone a random-background augmentation (needs a patch to its training, deviates from the protocol), or (c) also report FTGS++ trained on raw backgrounds (its 4DAnyone default) with a foreground-only evaluation — which one?

### Next
- Compare the ZJU-387 real12 results (SR4D vs 3 baselines) → write them up here.
- Tune SR4D (pose lr, L3 effect), run ablations a–f (M5).
- Run real8 / real_all.
- Once DNA-Rendering arrives: verify the converter (Color_Calibration shape, SMPLx scale, IoU) → move to the A-stage main experiments.

### 2026-10-08 evening: B-stage generated-view GT, L4 check, orbits
- B-stage self-eval (GT = the 24 generated training views, `data_dev/zju_mocap_gen/387_gen24_gv`, results/B_genview_gen24.csv):
  4DGS 24.88 / fg 21.85, FTGS++ legacy 19.08 / fg 22.53, SR4D (L4 mean) 22.47 / fg 18.97.
- L4 at render: per-view colour code +0.01 dB (no effect); + per-view camera refinement -> 26.58 / fg 23.22 / fgi 28.77, LPIPS 0.074.
  Learned deltas are tiny (mean 0.15 deg / 0.4 cm, max 0.35 deg / 0.9 cm) but absorb the per-view misplacement of the generated person.
  Only usable on training views; NVS renders use the mean (no refinement).
- Orbit NVS: `tools/make_orbit.py` (circle fitted to train cams, 120 views, time ping-pong) + `--orbit` in render_cams / render_ply / ftgspp_render,
  wrapper `scripts/render_orbit_models.sh`. gen24 orbits in outputs/orbits/387_gen24/renders/*.mp4.
- A-stage rerun on mask_cihp (D-022) in progress, phase Acihp.
- Jitter diagnosis (`tools/diag_jitter.py`, RMS 2nd difference of deformed Gaussian centres):
  gen24 full 12.3 mm (vel 11.8) | L3 off 12.3 | L2 off 12.3 | **free bones = identity 2.2 mm** | input SMPL poses 12.6.
  real12 (old masks) 10.1 -> bones identity 1.5 mm. bone_smooth=1.0 variant: 8.0 mm (still jittery).
  => the shaking comes from per-frame independent free-bone SE(3) (60x25 free params), not from SMPL poses / L2 / L3.
- L4 camera refinement shifts the image by 1.55 px on average (0.3-2.4 px): invisible by eye, but edge errors dominate PSNR.

## 2026-10-08 — bone 떨림 수정 V0–V2 결과 (ZJU real12, mask_cihp, D-023)
| 모델 | 떨림 accel RMS | tOF int/ext | fgi int/ext | fgLPIPS int/ext | 판정 |
|---|---|---|---|---|---|
| 현재 (free) | 10.05 mm | 1.19 / 1.13 | 21.14 / 19.13 | 0.1301 / 0.1653 | 기준 |
| V0 bone 없음 | 1.40 | 1.10 / 1.02 | 21.03 / 18.95 | 0.1294 / 0.1721 | 탈락 (fgLPIPS ext +0.0068) |
| V1 pose MLP | 1.42 | 1.06 / 1.00 | 20.97 / 18.84 | 0.1283 / 0.1670 | 탈락 (fgi ext −0.29) |
| V2 spline | 1.42 | 1.07 / 0.93 | 20.83 / 18.93 | 0.1289 / 0.1660 | 탈락 (fgi int −0.31, ext −0.20) |
- 떨림은 세 변형 모두 해결(≈1.4 mm, 입력 SMPL 포즈로 돌리면 ≈2.0 mm). 화질 기준은 모두 근소하게 넘지 못함(0.2 dB / 0.005 경계). 단일 시드라 시드 편차 크기 미확인.
- V0에서 S4부터 loss가 nan으로 로그됨: L3 mag_loss가 K=0 bone 텐서의 빈 평균. 빈 텐서라 기울기는 0이고 PSNR 정상 → 학습 결과 유효. `instant_residual._mean0`로 수정(dance V0부터 적용).
- `tools/diag_jitter.py` bone-identity를 모든 bone mode에서 동작하게 수정. `scripts/post_bones.sh`(진단+궤도) 추가.
- V3 (real12): 떨림 1.45 mm, tOF 1.09/0.98, fgi 20.92/19.16, fgLPIPS 0.1276/0.1632 → 탈락(fgi int −0.22). V0–V3 모두 경계에서 근소 탈락.
- 사용자 지적(떨림은 생성뷰24에서 관찰됨) → ZJU gen24(generated-only, zju_gen_dev)에서 V0/V1/V2 + 같은 설정 free 재학습 (`scripts/queue_gen_bones.sh`, outputs/sr4d_bones/387_gen24). dance SR4D는 이후 같은 GPU에서 (`scripts/queue_dance2.sh`: v0/v1/v2/free). dance 4DGS·FTGS++ 완료(self-eval fgi 25.64 / 26.92, tOF 2.69 / 2.43, oom 0.0116 / 0.0279), 궤도 `outputs/orbits/dance_gen24`.
- ZJU gen24 (generated-only, self-eval, L4 off render): free(재학습) 떨림 11.37 mm, tOF 1.56, fgi 25.32, fgLPIPS 0.1126 / V0 2.18, 1.34, 25.87, 0.1023 / V1 2.23, 1.30, 25.92, 0.1016 / V2 2.26, 1.31, 25.83, 0.1009 → V0–V2 모두 기준 통과(4DGS tOF 1.79). 규칙상 V0 후보, dance 결과 보고 확정.
- ZJU gen24 V3: 떨림 2.22 mm, tOF 1.31, fgi 25.94, fgLPIPS 0.1011 → 통과. V0–V3 모두 통과, 차이는 노이즈 수준.
- dance gen24 (self-eval): free 떨림 13.01 mm, tOF 2.39, fgi 23.65, fgLPIPS 0.2271 / V0 4.02, 2.18, 24.08, 0.1992 / V2 3.91, 2.20, 24.23, 0.2002 (4DGS tOF 2.69, fgi 25.64, fgLPIPS 0.2807; FTGS++ 2.43, 26.92, 0.2750). V0·V2 떨림 기준(3 mm) 탈락. 분해: 입력 SMPL 자세로 바꾸면 2.14/2.24 mm → 남은 떨림은 프레임별 자세 보정(0.04 rad, 15 mm) 때문. 자세 보정 시간 평활 제안(사용자 확인 대기).
- dance V1 3.85 mm, tOF 2.19, fgi 24.17, fgLPIPS 0.2014 / V3 3.88, 2.15, 24.33, 0.2009 → 모두 떨림 기준 탈락, 입력 자세면 2.21 mm. V0–V3 차이는 노이즈 수준.

## 2026-10-09 — 항목 4: 높이가 다른 홀드아웃 링 (4DAnyone 15°+35°, 같은 GVHMR·시드 42)
- `scripts/gen_4danyone_2ring.sh` → `outputs/fouranyone_runs/{zju,dance}_2ring` (각 ~65 min, 1 GPU).
- 공동 생성이 기존 15° 링을 바꾸는가 (`tools/compare_gen_runs.py`, 원본 24뷰 vs 새 15° 링, 전 프레임 PSNR):
  - ZJU: 카메라 동일, 그래도 영상이 달라짐 — 평균 29.7 dB, 최저 22.7 dB.
  - dance: 카메라도 바뀜(framing이 35° 링까지 고려 → 카메라 높이 +2.4 cm, 초점 +0.26%) — 평균 21.6 dB.
  - → 기존 gen24 학습 데이터로 35° 링을 GT로 쓸 수 없음. 새 15° 링으로 재학습 후 35° 링으로 채점.
- `scripts/prep_2ring.sh`: 변환 + split `r15h35`(train g00–g23 = 15°, test_extrap g24–g47 = 35°). `data_dev/zju_mocap_gen/387_gen48_2ring`, `data_dev/fouranyone_gen/dance5885633_gen48_2ring`.
- 학습: `scripts/run_r15h35.sh` — SR4D V0, 4DGS, FTGS++(legacy) × {zju, dance}.
- r15h35 ZJU 생성뷰 (35° held-out): SR4D V0 fgi 20.37 / fgLPIPS 0.1414 / PSNR 22.55 / LPIPS 0.0716 / tOF 1.57 / oom 0.0003; 4DGS 19.28 / 0.1728 / 22.20 / 0.0875 / 1.78 / 0.0055; FTGS++ 21.00 / 0.3073 / 19.43 / 0.2387 / 1.73 / 0.0140.
- 사고: dance r15h35 작업이 ~16 h 시작 안 됨 — 대기 루프의 `pgrep -f "[p]rep_2ring.sh"`가 같은 문자열을 명령줄에 가진 내 Bash 셸과 매칭. 이후 PID 파일로 대기.
- dance V0 층 진단(재학습 없음, 생성 뷰 채점): 전체 fgi 24.08 / L3 끔 24.04 / L2+L3 끔 21.20 (tOF 2.18 → 3.19) → 옷·머리 움직임은 L2 담당, L3는 기여 없음.
- 고스팅 크롭 `tools/ghost_crops.py` → `outputs/ghost/{dance,zju}_gen24`: 생성 GT에는 고스팅 없음(마스크 가장자리 계단). SR4D free 얼굴 이중상, V0에서 사라짐. 4DGS 고스팅 가장 심함. FTGS++ 질감 선명·floater.
- DualGS r15h35 (zju, dance), 자세 (a)/(b) 학습 중.

## 2026-10-10 — 자세 보정 (a)/(b) 판정 (r15h35, 35° held-out 생성 GT, bones V0)
| | 떨림 mm | tOF | fgi | fgLPIPS | PSNR | LPIPS | oom |
|---|---|---|---|---|---|---|---|
| ZJU V0 free 자세 | 2.17 | 1.568 | 20.37 | 0.1414 | 22.55 | 0.0716 | 0.0003 |
| ZJU (a) global | 2.26 | 1.581 | 20.51 | 0.1393 | 22.60 | 0.0705 | 0.0003 |
| ZJU (b) spline | 2.21 | 1.569 | 20.54 | 0.1410 | 22.59 | 0.0712 | 0.0002 |
| dance V0 free 자세 | 3.79 | 2.450 | 17.70 | 0.3375 | 18.10 | 0.2090 | 0.0384 |
| dance (a) global | 1.61 | 2.415 | 18.18 | 0.3420 | 17.24 | 0.2144 | 0.0448 |
| dance (b) spline | 1.67 | 2.517 | 17.32 | 0.3401 | 17.71 | 0.2122 | 0.0419 |
| dance 4DGS | – | 2.531 | 15.23 | 0.3478 | 18.09 | 0.2129 | 0.0118 |
| dance FTGS++ | – | 2.946 | 15.58 | 0.3633 | 15.09 | 0.3103 | 0.0449 |
- ZJU: (a)≈(b) (차이 ≤0.03 dB, 0.002, 0.1 mm) → 동률. dance: (a)가 떨림·tOF·fgi에서 우세, fgLPIPS만 (b)가 0.002 낮음 → (a). 규칙(같으면 a)에 따라 **(a) 채택** (D-025).
- dance V0 프레임별 자세의 떨림 3.79 mm → (a) 1.61 mm로 3 mm 기준 통과. ZJU는 입력 자세가 원래 2.80 mm라 (a)가 2.26 mm.
- 주의: dance 전체 이미지 PSNR/LPIPS는 (a)가 V0보다 나쁨(17.24 vs 18.10), oom 0.045 — 마스크 밖 알파 큼. 확인 필요.
- 단일 시드. 항목 5: 시드 123으로 두 링 재생성 시작(`gen_4danyone_2ring.sh ... 123` → `outputs/fouranyone_runs/{zju,dance}_2ring_s123`).
- dance r15h35 oom이 큰 원인: 15° 학습 링 24뷰 전부 무릎 아래가 프레임 밖(마스크가 아래 가장자리에 닿음), 35° 링에서만 종아리·발이 보임 → 학습에서 한 번도 안 보인 영역이 채점에 들어감. 렌더는 무릎 아래에 바닥처럼 퍼진 floater(마스크 밖 알파의 대부분). ZJU는 15° 링에서 전신이 보여 해당 없음(0/24). 모든 방법에 같은 조건이나, dance r15h35 숫자 해석 시 주의. 진단 이미지 scratchpad oomchk.png.
- DualGS r15h35 (35° held-out): ZJU fgi 20.24 / fgLPIPS 0.1688 / PSNR 23.46 / LPIPS 0.0891 / tOF 1.62 / oom 0.0160; dance 12.99 / 0.2962 / 17.56 / 0.1711 / 2.50 / 0.0262. dance는 학습 뷰에서 안 보인 무릎 아래를 비워 둠(흰색) → 다리가 있는 곳의 fgi는 매우 낮지만 floater가 없어 전체 PSNR·LPIPS·fgLPIPS는 가장 좋음. ZJU는 전체 PSNR 최고(23.46 vs SR4D (a) 22.60), fg 지표는 SR4D가 앞섬(fgi 20.51, fgLPIPS 0.139). DualGS 렌더는 손·얼굴 윤곽이 번짐.
- 항목 5 (생성 실행 편차): 시드 123으로 두 링 재생성(`gen_4danyone_2ring.sh ... 123`, 카메라는 시드 42와 동일) → `data_dev/.../*_gen48_2ring_s123`. 영상 자체의 시드 간 차이(전 프레임 PSNR, `cmp_s42.json`): ZJU 평균 22.5 dB(15° 22.2 / 35° 22.9, 최저 15.9), dance 24.0 dB(최저 21.2). `scripts/chain_s123.sh` → SR4D (a) → 4DGS → FTGS++ 재학습 중 (`SFX=_s123 run_r15h35.sh`, 출력 `*_r15h35_s123`).
- 항목 5 결과 (35° held-out, 시드 42 → 123; fgi / fgLPIPS / PSNR / tOF / oom):
  - ZJU SR4D (a) 20.51→20.02 / 0.1393→0.1418 / 22.60→22.41 / 1.58→1.53 / 0.0003→0.0002, 떨림 2.26→2.21 mm
  - ZJU 4DGS 19.28→18.33 / 0.1728→0.1722 / 22.20→21.95 / 1.78→1.73; FTGS++ 21.00→20.82 / 0.3073→0.3042 / 19.43→19.58 / 1.73→1.65
  - dance SR4D (a) 18.17→18.00 / 0.3420→0.3424 / 17.24→17.21 / 2.41→2.33 / 0.0448→0.0441, 떨림 1.61→1.70 mm
  - dance 4DGS 15.23→15.28 / 0.3478→0.3446 / 18.09→18.36; FTGS++ 15.58→15.81 / 0.3633→0.3269 / 15.09→17.45 / oom 0.0449→0.0159
  - 편차(생성+학습 합산): fgi 0.05–0.95 dB, fgLPIPS ≤0.004 (FTGS++ dance 0.036). 방법 간 순위는 두 시드에서 동일(ZJU fgi FTGS++>SR4D>4DGS, fgLPIPS·tOF·oom SR4D 최고; dance fgi·tOF SR4D 최고). 자세 (a)/(b) 차이(ZJU 0.03 dB)와 D-023 판정 경계(0.2 dB / 0.005)는 이 편차보다 작거나 비슷 → 근소 차 판정은 단일 시드로 확정 불가.

## 2026-10-12 — 사용자 결정 반영 (D-026 bone V0 확정·판정 규칙, D-027 co-visible 지표)
- 정정: 2026-10-09의 "옷·머리 움직임은 L2 담당"은 근거 부족. `--no_l2`는 L2의 dx·dq·dlog_s·dcolor를 모두 끄므로 L2+L3 끔의 하락은 학습 때 없던 조합으로 렌더한 결과일 뿐(원인 결론 불가). 결론 가능한 것: L3를 렌더 시 꺼도 변화 없음.
- 진단 1 (렌더 시 L2 dx만 제거, `--no_l2dx`, 학습 뷰 vs 생성 GT, 영역=입력 SMPL 관절 투영, `tools/diag_l2dx_regions.py`): 몸통 대비 상대 하락만 보고.
  - dance: 머리(머리카락 포함) +4.07 dB, 치마 +0.09 dB (몸통 하락 2.06 dB 기준). 그림: 머리카락 위치가 어긋나 잔상, 치마 윤곽은 거의 그대로.
  - ZJU: 머리 +0.21, 허벅지(치마 영역) +1.51 dB (몸통 3.83 dB 기준).
- 진단 2 (L2 dx 크기, 영역 = 스키닝 꼭짓점의 argmax 관절, `tools/diag_l2dx_mag.py`): 시간 변동(RMS, 움직이는 몫) dance 자유 가우시안 머리 13.2 mm / 엉덩이·허벅지 4.8 / 몸통 12.2; ZJU 6.1 / 7.9 / 9.3. 몸 surfel 몸통도 큼(ZJU 15.2 mm) — 자세 (a)에서 입력 자세 오차를 L2가 흡수하는 것으로 보임(추정).
  - 해석: dance 치마는 L2 dx 기여가 몸통보다 작음 → 대부분 LBS(허벅지)로 따라감. 머리카락은 L2 dx가 상대적으로 큼.
- 항목 5 dance SR4D 마스크 밖 알파 위치 (`outputs/sr4d_bones/dance_r15h35/oom_location.json`, `oom_gaussians.json`): 무릎선 아래 비율 SR4D (a) 0.915, 프레임별 자세 0.925, 4DGS 0.524, FTGS++ 0.149, DualGS 0.114 (FTGS++·DualGS는 머리 위 floater). SR4D 가우시안 ~40만 중 ~9.5만(24%)이 35° 뷰에서 마스크 밖, 그중 자유 가우시안 90%, 72%는 15° 학습 카메라 어디에도 화면 안에 들지 않음(학습 손실이 닿지 않는 공간). → 무릎 아래 잘림과 직접 관련. 초기화 carving은 seen≥2를 요구하므로, 학습 중 MCMC 재배치/이동으로 들어간 것으로 추정(미확인).
- 사고: 진단 렌더를 GPU2(ZJU 시드 7 생성 중)에 올려 생성이 OOM으로 종료 → 재시작. 이후 GPU 배치 확인 후 실행.
- co-visible 지표 (D-027, 35° held-out, 시드 평균; cv 내부 PSNR / cv LPIPS(마스크 안 LPIPS 지도 평균) / cv tOF):
  - ZJU: SR4D (a) 20.34 / 0.268 / 1.56 (n=2), 자세 프레임별 20.36 / 0.271, (b) 20.51 / 0.272, 4DGS 19.97 / 0.323 / 1.76 (n=2), FTGS++ 20.89 / 0.296 / 1.70 (n=2), DualGS 20.41 / 0.316 / 1.63.
  - dance: SR4D (a) 20.19 / 0.422 / 2.26 (n=2), 프레임별 20.48 / 0.419, (b) 20.37 / 0.417, 4DGS 20.31 / 0.474 / 2.33 (n=2), FTGS++ 20.37 / 0.401 / 2.68 (n=2), DualGS 16.93 / 0.437 / 2.32.
  - 해석: dance에서 전체 기준 SR4D의 내부 PSNR 우위(18.1 vs 15.3–15.7)는 학습에서 안 보인 다리 영역 몫이었음. co-visible에서는 4DGS·FTGS++와 동률 수준(±0.2 dB, 시드 σ 0.17–0.33 내), LPIPS는 FTGS++가 가장 낮고 SR4D 2위, tOF는 SR4D 최저. ZJU co-visible: 내부 PSNR FTGS++ > DualGS ≈ SR4D > 4DGS, LPIPS·tOF는 SR4D 최저.
  - 자세 (a)/(b) co-visible 차이(ZJU 0.17 dB, dance 0.18 dB)는 시드 σ 이내 → D-026 규칙상 동률, (a) 유지.
- ablation (처음부터 학습, r15h35 시드 42, 기준 = SR4D V0 + 자세 (a); 비교 척도 = 기준의 시드 42/123 차이, D-026):
  - ZJU no-L3: cv 내부 PSNR +0.01 (동률), cv LPIPS +0.0107 (2.8×), cv tOF +0.16 (2.8×), 떨림 2.26→2.82 mm (11×) → 나빠짐.
  - dance no-L3: cv PSNR −0.21 (1.7×, 시드 추가), cv LPIPS +0.0116 (1.2×), cv tOF +0.23 (2.7×), 떨림 1.61→2.12 mm (5.7×) → 나빠짐.
  - → no-L3는 동률 아님. **L3 유지.** 렌더 시 L3 제거는 변화 없음(위)과 대조 — L3는 학습 중 프레임별 잡음을 흡수해 canonical/L2를 매끄럽게 하는 역할로 보임(추정). 확정용 시드 123 ablation 대기열.
  - no-L2Δx: ZJU cv PSNR −0.58 (2.3×), cv LPIPS +0.048 (12.5×), tOF +0.18 (3.1×); dance −0.40 (3.3×), +0.039 (4.2×), +0.22 (2.6×) → L2 Δx 필요.
- 생성 시드 7: 4DGS·FTGS++ 완료(ZJU cv 19.98/0.329, 21.25/0.298; dance 20.29/0.482, 19.84/0.416). DualGS ZJU s7은 첫 skin 단계에서 CUDA "invalid configuration argument" → 재시도 중(GPU2). 대기열: SR4D (a) s7 → no-L3 s123 → no-L2Δx s123 (GPU2 ZJU, GPU3 dance).
- ablation 시드 123 (기준 대비 차이, 시드 42 / 123): ZJU no-L3 cv PSNR +0.005/+0.005, cv LPIPS +0.011/+0.013, cv tOF +0.17/+0.18, 떨림 +0.56/+0.64 mm; dance no-L3 −0.21/−0.19, +0.012/+0.012, +0.23/+0.23, +0.51/+0.41 mm. no-L2Δx ZJU −0.58/−0.56, +0.048/+0.045, +0.18/+0.21; dance −0.40/−0.68, +0.039/+0.033, +0.22/+0.22. → 두 시드 순위·크기 일치, D-026 규칙상 **확정: L3 유지, L2 Δx 유지.**
- 3시드 요약 `results/B_r15h35_seeds.csv` (생성 시드 42/123/7, co-visible 주 지표, 평균±sd):
  - ZJU: SR4D (a) cv 내부 PSNR 20.36±0.13 / cv LPIPS 0.269±0.002 / cv tOF 1.57±0.03; 4DGS 19.97±0.32 / 0.325 / 1.77; FTGS++ 21.01±0.23 / 0.297 / 1.69; DualGS(2시드) 19.18±1.74 / 0.318 / 1.63.
  - dance: SR4D (a) 20.17±0.07 / 0.427±0.010 / 2.24±0.06; 4DGS 20.30±0.15 / 0.476 / 2.32; FTGS++ 20.20±0.31 / 0.406 / 2.69; DualGS 16.93±0.30 / 0.441 / 2.32.
  - DualGS ZJU 시드 7: 두 번 모두 1번 프레임 skin 단계 backward에서 "CUDA error: invalid configuration argument"(재현됨). CUDA_LAUNCH_BLOCKING=1로 원인 확인 중(`outputs/dualgs/zju_r15h35_s7_dbg`).

## 2026-10-12 (오후) — 사용자 판단 반영 (D-029)
- 미관측 자유 가우시안 사후 확인(dance): 6.4만/24만 미관측, 제거 시 35° 마스크 밖 알파 0.0448→0.0311(무릎 아래에서 감소), co-visible 거의 불변. 학습 중 규칙으로 SR4D dance·ZJU 시드 42 학습 중, 4DGS·FTGS++ 부록 학습 중(훅 동작 확인: SR4D relocation마다 ~260개, 4DGS coarse 단계 수십 개).
- DualGS ZJU 시드 123 프레임별 co-visible 내부 PSNR: 0번 프레임부터 18.0 dB(시드 42 19.9 dB), 이후 17.3 근처로 완만히 하락 → 추적 실패(급락) 아님, 첫 프레임 정적 복원부터 낮음. `outputs/dualgs/zju_r15h35_s123/per_frame_covis.json`.
- 몸 surfel 렌더: 2DGS 래스터화가 아니라 평탄화 3DGS(D-004; 세 번째 축 0.5 mm, 법선 일관성 손실). SH 차수는 몸·자유 공통, 현재 최대 2 → SH 3 실험은 ZJU 미관측 학습 뒤 GPU0.
- 실수: 실행 중인 run_r15h35.sh를 또 수정 → 즉시 원상 복구, 부록 모드는 `scripts/run_r15h35_unseen.sh`로 분리.
- D-029 결과 (r15h35, 시드 42; 기준 → 미관측 제거; co-vis 내부 PSNR / co-vis LPIPS / co-vis tOF | 전체 fgi / fgLPIPS / PSNR / 마스크 밖 알파):
  - SR4D dance: 20.25→20.00 / 0.4177→0.4160 / 2.30→2.25 | 18.17→17.77 / 0.342→0.322 / 17.24→18.36 / 0.0448→0.0256 (−43%).
  - SR4D ZJU: 20.47→20.50 / 0.2661→0.2673 / 1.59→1.59 | 0.0003→0.0003 (미관측 0개라 사실상 동일, 기대대로).
  - 4DGS dance: 20.46→20.14 / 0.4712→0.4703 / 2.43→2.42 | oom 0.0118→0.0097; 4DGS ZJU: 20.29→20.56 / 0.3232→0.3224 / 1.79→1.83 | fgi 19.28→20.47, oom 0.0055→0.0033.
  - FTGS++ 부록: 래퍼 버그(ftgspp.train 패키지의 `train` 속성이 함수라 모듈을 잘못 잡음)로 시작 직후 실패 → importlib로 수정, 재실행 중(GPU3).
  - 판정 메모: SR4D dance 마스크 밖 알파 −43%·전체 PSNR +1.1 dB는 큼. co-visible 내부 PSNR −0.25 dB는 기준 시드 차이(0.12)의 약 2배 — "변화 없어야 함"과 어긋남. r15h35 동결(D-030)로 시드 123 확인은 하지 않음.

## 2026-10-13 — 주 프로토콜 시드 42 결과 (D-030)
- gen48_ho8 (40 학습 / 8 홀드아웃 생성 뷰, 최근접 14.7°; fg 내부 PSNR / fg LPIPS / tOF / 마스크 밖 알파 | co-vis 내부 PSNR·LPIPS 보조):
  - ZJU: SR4D 24.36 / **0.114** / 1.51 / **0.0001** | 24.28 / **0.211**; 4DGS 24.93 / 0.154 / 1.87 / 0.0022; FTGS++ 25.16 / 0.256 / 1.59 / 0.0198; DualGS **26.18** / 0.123 / **1.31** / 0.0178 | 26.12 / 0.234.
  - dance: SR4D 22.97 / **0.221** / 2.29 / **0.0024** | 23.40 / **0.340**; 4DGS 23.44 / 0.281 / 2.69 / 0.0155; FTGS++ **23.60** / 0.358 / 2.42 / 0.0506; DualGS **23.60** / 0.237 / **1.97** / 0.0054.
  - 해석: 보간 영역(밀한 48뷰)에서 SR4D는 fg 내부 PSNR 최하(ZJU −1.8 dB vs DualGS), LPIPS·마스크 밖 알파는 최고, tOF는 DualGS가 최고. SR4D 떨림 1.82 / 1.51 mm.
  - SR4D 경계: 실루엣 IoU는 최고(0.948)인데 윤곽선을 따라 얇은 흰 테두리 — 경계 3 px 띠 평균 절대오차 44.8 (DualGS 29.5, 4DGS 35.8). 흰 배경 합성 학습에서 가장자리 반투명이 흰색과 섞이는 것으로 추정. fg PSNR(18.9)과 fgi(24.4) 차이가 큰 이유.
- A단계 3시드 (ZJU-387 real12, 학습 시드 0/1/2, `results/A_real12_seeds.csv`; fg 내부 PSNR / fg LPIPS / tOF / 마스크 밖 알파):
  - interp: SR4D V0 **20.95±0.08** / **0.129** / **1.11** / **0.0000**; 4DGS 19.96±0.39 / 0.158 / 1.24 / 0.0065; FTGS++ 19.57±0.25 / 0.416 / 1.31 / 0.065; DualGS 18.84±0.10 / 0.165 / 1.34 / 0.0004.
  - extrap: SR4D **19.00±0.08** / 0.169 / **1.01** / **0.0002**; 4DGS 18.14±0.56 / **0.167** / 1.02 / 0.0082; FTGS++ 17.94±0.11 / 0.442 / 1.21 / 0.066; DualGS 14.83±0.06 / 0.201 / 1.17 / 0.0006.
  - 시드 0 기준 방법은 alpha 재평가본(metrics_renders_all.json) 사용.
- gen24_all 4방법 × 2데이터 학습 중.
- 2026-10-13 단계 1: SH 최대 2→3 (gen48_ho8 ZJU 시드 42) 학습 중(GPU0). 단계 2: 흰 테두리 진단 → D-031. 단계 3: rbg / cam0 / rbg+cam0 세 변형 학습 중(GPU1–3).
- SH 최대 2→3 (ZJU gen48_ho8 시드 42): fgi 24.36→24.42, fg LPIPS 0.1142→0.1140, 경계 PSNR 10.42→10.63, IoU 0.948 동일, tOF 1.51→1.49, 떨림 1.82→1.81 mm → 기준(0.3 dB/0.005) 미만, 동률 → SH 최대 2 유지, 시드 123 생략.
- 점무늬 1단계 보강: 크롭 LPIPS 학습 뷰 SR4D 0.409 / DualGS 0.297 / FTGS++ 0.462, 홀드아웃 0.454 / 0.461 / 0.493 (선명도는 노이즈도 셈). 학습 예산(이미지당 평균 노출): SR4D 45, 4DGS 9.6, FTGS++ 12.5, DualGS 763(프레임별 joint+skin). 마지막 10%는 S4 내부(단계 경계 없음); "2 dB"는 양 끝 3점 비교의 잡음 — 구간 평균 +0.4 dB.
- 뷰어: tools/viewer4d.py 동작 확인(SR4D·DualGS 어댑터, L2Δx/L3 토글; 첫 렌더 110 s 커널 컴파일, 이후 0.07 s/프레임), GPU0에서 8080 포트 실행 중.
- GPU0 대기열: 단일 프레임(30) → 스케줄 2배(112k). GPU3: confidence 끔(rbgcam0·DualGS ZJU gen24 뒤).
- 배경 무작위화 rbg (ZJU gen48_ho8 시드 42): fgi 24.36→24.31, fg LPIPS 0.1142→0.1170, 경계 PSNR 10.42→10.51, 경계 MAE 38.5→39.2, IoU 0.948→0.950, tOF 1.51→1.46 → 테두리 개선 없음. GT 마스크 가장자리(계단 모양) 영향 가능성 — cam0·rbgcam0 결과 후 판단.
- 뷰어를 0.0.0.0:8080으로 재시작(GPU0, 약 1 GB). 첫 실행은 렌더 스레드 안의 지연 CUDA 초기화가 nerfview 인터럽트에 끊겨 죽었음 → 시작 시 미리 초기화·워밍업. ufw: ufw.conf ENABLED=no, user.rules에 8080 없음(root 아님, 변경 없음).
- **cam0 (L4 보정 평균 0, ZJU gen48_ho8 시드 42): fgi 24.36→25.79 (+1.43), fg LPIPS 0.1142→0.1147, 경계 PSNR 10.42→12.08, 경계 MAE 38.5→32.0, IoU 0.948→0.949, 전역 이동 0.38→0.23 px, 떨림 1.86 mm.** 기준 이상 → 확인 대기열: dance gen48_ho8 cam0 (GPU1, 머리·목 뒤), ZJU 시드 123 기준+cam0 (GPU2, 머리·목 all 뒤).
- rbg+cam0: fgi 25.85, fg LPIPS 0.1167, 경계 PSNR 11.38, MAE 35.8, IoU 0.955, tOF 1.47, 떨림 1.96 mm → cam0 단독(25.79 / 0.1147 / 12.08 / 32.0)과 fgi 동률, 경계·LPIPS는 cam0가 나음 → 배경 무작위화 제외, cam0 단독 후보(dance·시드 123 확인 대기).
- D-035 L2 CNN 선택지 구현·2000 it 동작 확인 완료(outputs/sr4d_tests/l2cnn_smoke).
- 점무늬 2단계: 단일 프레임(30) SR4D (L3 끔, cam0 없음 — cam0 기준선 전 시작), 30번 프레임 크롭(학습 뷰 g01·g11 / 홀드아웃 g00·g12): PSNR·LPIPS·선명도 — 학습 뷰 60프레임 SR4D 27.47/0.428/356, 단일 프레임 27.62/0.308/1016, DualGS 31.24/0.306/2387; 홀드아웃 26.84/0.460/325, 26.03/0.449/1025, 25.46/0.470/2917. → 판정: "2가 선명함 → 시간 공유 중 정렬 문제"(부분): 단일 프레임은 선명도 3배, 학습 뷰 LPIPS가 DualGS 수준. 다만 학습 뷰 PSNR은 DualGS보다 3.6 dB 낮고 점무늬도 여전히 덜 선명 → 남은 차이는 예산/표현 쪽. 단일 프레임 가우시안: 몸 surfel 36만 + 자유 3.9만.
- 머리·목 B-spline 보정 (dance gen48_ho8, cam0 없음): fgi 22.97→22.88, fg LPIPS 0.2205→0.2231, 얼굴 PSNR/LPIPS 17.93/0.252→17.94/0.249, 점무늬 홀드아웃 LPIPS 0.454→0.464, 떨림 1.57 mm → 효과 없음(동률) → 자세 (a) 유지(D-033 종결).
- **confidence 끔** (dance gen48_ho8, cam0 없음): fgi 22.97→22.38 (−0.59), **fg LPIPS 0.2205→0.1916 (−0.029)**, 얼굴 LPIPS 0.252→0.229 (DualGS 0.244보다 낮음), 점무늬 홀드아웃 LPIPS 0.454→0.369·선명도 352→753, tOF 2.29→2.25, 떨림 1.61 mm → 판정 "3이 선명함 → confidence 문제"(사용자 기준). PSNR은 −0.59 dB로 반대 방향. 기준 이상 → cam0 기준선에서 재확인 필요(cam0+confidence 끔, 이후 시드 123).
- DualGS ZJU r15h35 시드 7: CUDA_LAUNCH_BLOCKING=1 재실행 결과를 정식 폴더로 이동·B_r15h35로 재채점(노트 표시). 이전 실패 시도 폴더 삭제. results/A_r15h35.csv의 디버그 줄 정리(그 두 줄뿐이라 파일 삭제).
- 뷰어 v2 (viser 직접, `tools/viewer4d.py`; v1 = viewer4d_nerfview_v1.py): 재생 중 "먼지" 원인 = nerfview가 rerender(재생 프레임마다)를 저해상도(초당 광선 예산) + JPEG 40으로 그림 → v2는 매 프레임 캔버스 해상도·JPEG 90, 느리면 프레임 건너뜀. 궤도 모드(구/원기둥, 중심 평균 골반, 보간/외삽 표시), 48 카메라 이동(학습/홀드아웃 색), 골반 중심 사람 회전(카메라 역변환으로 정확), SR4D|DualGS 나란히. nerfview v1은 렌더 예외 시 os._exit(1)로 서버를 종료하는 구조였음.
- L2 CNN 중간 떨림(32,000 it, S3 — L2는 24k부터 켜짐): 2.22 mm (L2 끄면 2.09) → 3 mm 이하.
- **2배 스케줄+cam0 (dance gen48_ho8, 112k it)**: fgi 22.97→20.03, fg LPIPS 0.2205→0.2459, tOF 2.29→2.26, 떨림 1.47 mm; 크롭(학습 뷰 얼굴 PSNR 17.98→14.30, 점무늬 27.65→25.91). 원인: 학습 뷰 PSNR L4 켬/끔 차이 2.6 dB(1배) → 5.1 dB(2배), 뷰별 카메라 이동 평균 7.6→9.4 mm(최대 17 mm) — 모델이 뷰별로 보정된 카메라에 맞춰지고, 보정 없이 렌더하는 홀드아웃·자유시점이 나빠짐(ZJU cam0는 차이 1.1 dB, 4.5 mm). l4_cam 학습률은 스케줄과 무관하게 상수라 2배에서 더 표류. → 반복 수 효과는 판정 불가(L4 카메라 보정과 교란). L4 카메라 보정이 dance에서 해로울 가능성 큼 — L4 완전히 끔 실험(GPU3 큐 끝, ZJU)을 dance에도 권장.
- **dance cam0 1배 (확인 실행)**: fgi 22.97→22.07 (−0.90), fg LPIPS 0.2205→0.2322, 경계 PSNR 11.22→10.61, 얼굴 17.93/0.252→17.31/0.264, 점무늬 홀드아웃 PSNR/LPIPS 26.76/0.454→26.17/0.468, tOF 2.29 동일, 떨림 1.48 mm; 학습 뷰 L4 켬/끔 23.72/20.71(차이 3.0 dB), 카메라 이동 평균 7.3 mm. → ZJU(+1.43 dB)와 반대 방향 — cam0는 일반 개선이 아님(D-026: 두 데이터 순위 불일치 → 확정 불가). dance에서는 뷰별 카메라 보정 자체(평균 0이어도 뷰별 7 mm)가 문제로 보임.
- **L2 CNN 본 학습 (dance gen48_ho8 시드 42, cam0, 56k)** vs cam0 1배 (같은 설정, MLP): fgi 22.07→21.94, fg LPIPS 0.2322→0.2382, 경계 PSNR 10.61→10.66, 얼굴 17.31/0.264→17.14/0.281, 점무늬 학습/홀드아웃 LPIPS 0.445/0.468→0.433/0.474, tOF 2.29→2.40, 떨림 1.48→1.70 mm(최종; 중간 2.22) → 개선 없음(LPIPS·tOF·얼굴은 약간 나쁨). 학습 뷰 L4 켬/끔 23.43/20.94(차이 2.5 dB) — 같은 L4 카메라 보정 문제를 공유하므로, L4 끈 상태에서 다시 비교해야 공정.
- B 코드 준비: D-036 일치율 confidence, D-037 프레임별 디테일 층(s7) 구현·2000 it 동작 확인(outputs/sr4d_tests/cons_detail_smoke).
- 공정성 점검 D-038 기록.
- **dance L4 끔 (카메라 보정·confidence 끔, gen48_ho8 시드 42)**: fgi 23.22 (기준 22.97, conf끔 22.38, cam0 22.07), fg LPIPS **0.1895** (0.2205 / 0.1916 / 0.2322), 경계 PSNR **11.95** (11.22), 얼굴 17.91/**0.226** (DualGS 18.75/0.244), 점무늬 학습 뷰 PSNR/LPIPS 28.50/0.320 (기준 27.65/0.418), tOF 2.25, 떨림 1.62 mm; 학습 뷰 L4 켬=끔 22.44(카메라 보정 없음). → 전 항목에서 기준선 이상, PSNR·LPIPS 동시 개선. D-026: fgi +0.25(<0.3, 동률 구간), LPIPS −0.031(기준 이상) → 시드 123 확인 필요. 뷰어에 "SR4D L4 끔"(ho8 40뷰 모델) 추가.
- **L4 끔 + CNN** vs L4 끔(MLP), dance gen48_ho8 시드 42: fgi 23.22→23.09, fg LPIPS 0.1895→0.1979, 경계 11.95→11.84, 얼굴 17.91/0.226→17.94/0.234, 점무늬 홀드아웃 PSNR/LPIPS 26.75/0.391→26.26/0.409, tOF 2.25→2.36, 떨림 1.62→1.80 mm → 카메라 문제를 뺀 공정 비교에서도 CNN 개선 없음(LPIPS·tOF 약간 나쁨) → L2는 MLP 유지.
- L2 규제 점검(CPU, dance L4 끔 체크포인트, 10프레임): 가중치 l2_mag 0.01, l2_smooth 0.01, lr l2 1e-3 / 해시 1e-2. 크기 손실(가중 전) 평균 0.0159 → 가중 후 0.00016(사진 손실 ~0.15의 약 0.1 %) → 규제에 눌려 있지 않음. 분포 p50/p90/p99: 몸 surfel |dx| 6.6/17.0/35.9 mm, 회전 2.5/5.5/9.8°, |dlog_s| 0.006/0.007/0.009, |dcolor| 0.005/0.012/0.017; 자유 |dx| 3.4/18.4/50.3 mm, 회전 1.7/5.2/12.9°, dlog_s 0.006/0.007/0.010, dcolor 0.004/0.010/0.016. 위치·회전은 크게 쓰지만 크기·색 변화는 거의 0 — 규제가 아니라 출력 스케일(×0.1)과 0 초기화 쪽 제약으로 보임(자세 따른 음영·주름 색은 L2가 거의 안 냄).
- **사고 (2026-10-14 01:15–17:00)**: 4DGS dynerf 프리셋(dataloader=True, 작업자 16개)이 "data loading done" 직후 교착 — PanopticSports 로더가 카메라 생성 시 CUDA 텐서를 만들어 fork된 작업자에서 CUDA 오류. 완료 대기만 걸어 두어 ~15시간 GPU2·3 유휴, 후속 FTGS++ 흰 배경·s7 미시작. GPU0(DualGS 05:01 완료)·GPU1(L4 끔+2배 03:44 완료)도 이후 유휴. 조치: 멈춘 프로세스 종료, `configs/fourdgs/dynerf_default_nodl.py`(공식 dynerf에서 dataloader만 False, 배치 4 유지 — 비로더 경로도 매 반복 무작위 4카메라)로 GPU0·1에서 재실행, 로그 멈춤 감시(`scripts/stall_watch.sh`) 추가.
- **L4 끔 + 2배 (dance gen48_ho8, 112k it, 304 min)** vs L4 끔 1배: fgi 23.22→23.38, fg LPIPS 0.1895→0.1893, 경계 11.95→12.18, 얼굴 17.91/0.226→18.11/0.221, 점무늬 학습 뷰 PSNR/LPIPS 28.50/0.320→28.96/0.300, 홀드아웃 26.75/0.391→26.56/0.387, tOF 2.25→2.15, 떨림 1.60 mm → 카메라 표류가 없으면 2배 학습은 소폭 개선(대부분 0.3 dB/0.005 미만; tOF −0.10). 학습량은 큰 병목이 아님.
- **gen24_all (원래 24뷰 전부 학습; 학습 뷰 자체 적합 — 새 시점 일반화 아님)**: fgi / fg LPIPS / tOF / 마스크 밖 알파 — ZJU: SR4D 26.02/0.103/1.40/0.0003, 4DGS 27.24/0.144/1.78/0.0111, FTGS++ 28.10/0.274/1.41/0.0190, DualGS 35.46/0.071/0.82/0.0185; dance: SR4D 24.49/0.202/2.34/0.0061, 4DGS 25.60/0.279/2.71/0.0131, FTGS++ 26.98/0.278/2.41/0.0249, DualGS 32.01/0.126/1.35/0.0556. DualGS가 자체 적합에서 압도(이미지당 노출 ~760회 → 학습 뷰 과적합). SR4D 떨림 2.24/2.62 mm.
- CLAUDE.md §12 운영 규칙(로그 멈춤·GPU 큐 비움 알림) 추가, `scripts/queue_watch.sh`.
- **공정성 재학습 완료 (2026-10-14, 아티팩트 v45)**: 4DGS 공식 dynerf(공식 렌더러로 채점; PLY 내보내기 경로는 SH·불투명도 변형 때문에 색이 어긋나 무효 처리, CSV에 `_INVALID_plyexport` 표기) ZJU fgi 25.26 / fg LPIPS 0.152, dance 23.76 / 0.280 — dnerf(24.93/0.154, 23.44/0.281)보다 소폭 우세 → 4DGS 대표 = dynerf. FTGS++ 흰 배경: ZJU 24.35/0.110, dance 21.76/0.229 (검정 25.16/0.256, 23.60/0.358) → PSNR 검정·LPIPS 흰 우세, 흰 배경은 배경을 흰 가우시안으로 채움(마스크 밖 알파 0.86). 주 표에 두 줄 모두 유지.
- **S7 디테일 층 (D-037)**, L4 끔 최종 56k 체크포인트 + 12k: (a) fgi 23.75 / fg LPIPS 0.206 / 얼굴 18.83/0.230 / tOF 2.28 / 떨림 4.04 mm / +549 MB; (b, consistency 1회) 23.63 / 0.205 / 18.62/0.228 / 2.28 / 4.03. 대조군 L4 끔 2배 23.38 / 0.189 / 18.11/0.221 / 2.15 / 1.60. PSNR↑ 대신 LPIPS·선명도·떨림 악화(점무늬 LPIPS 0.30→0.41) → 채택하지 않을 것을 권고(사용자 결정 대기). 뷰어에 "SR4D L4 끔 + 디테일 층 (a)/(b)" 추가.
- GPU 0–3 대기(뷰어만 GPU0). 다음 후보(사용자 결정 대기): L2 출력 스케일 실험, dance L4 끔 시드 123 확인.
- **2026-10-15**: 서버 재부팅 후 재개(뷰어 재시작 GPU2, tmux 세션 `sr4djobs`). 디스크 여유 ~8 GB(100%). 아티팩트 v46에 "현재 비교 (10/15)" 섹션(표 2개, 홀드아웃 8뷰·학습 뷰 4개 × 프레임 0/30/59 크롭+오차 지도, 55°·5° 궤도 나란히 영상) 추가. 도구: `tools/cmp_figure.py`, `tools/make_camlist.py`, `make_orbit.py --abs_elev_deg`, `fourdgs_render_native.py --orbit`, `scripts/render_cmp1015.sh`. FTGS++ 흰 배경 dance g12 프레임 57–59는 빈 화면(흰 가우시안이 카메라를 가림).
- s7 (c) 평활 10배 GPU0, (d) Δ색만 GPU1 실행 중(각 ~3.5 h).
- **s7 (c)·(d) 결과 (아티팩트 v47)**: (c) 평활 10배: 전체 23.17 / fg 20.11 / fgi 23.80 / fg LPIPS 0.198 / tOF 2.20 / 떨림 2.32 / 얼굴 18.78·0.224 / 점무늬 LPIPS 학습 0.376·홀드아웃 0.424 / 549 MB. (d) Δ색만: 22.90 / 19.46 / 23.33 / 0.188 / 2.18 / 1.62 / 17.97·0.223 / 0.316·0.387 / 275 MB. (c)는 (a)보다 전부 낫지만 기준선 대비 LPIPS·점무늬·떨림이 나쁨, (d)는 기준선과 동률(기준 미만). → PSNR 이득·흐림·떨림 모두 Δ위치에서 나옴. 뷰어에 (c)·(d) 추가. 디스크 여유 ~6 GB.

## 2026-10-17 — 2층 구조 실험 (D-040)
- 코드: `third_party/ftgs_outer/`(FTGS++ 복사·수정), `sr4d/layered/{core,train,render}.py`, `scripts/{run_layered,eval_layered,render_layered_figs}.sh`, `tools/{exp_figure,exp_orbit,share_frames}.py`. 새 아티팩트 "SR4D 실험 기록" https://claude.ai/artifact/PwcwmMgmZwUhmCKgn84C2a (빌더 `docs/gallery/exp_page.py`, 등록부 `docs/gallery/exp_registry.json`).
- 동작 확인(2k): 지시대로면 바깥층이 죽음(17% 생존, 안쪽 99.99%) → 안쪽 dropout 0.5 제안(`--inner_drop`, 미승인 변형으로 병행 실행).
- EXP-001 바깥층만: 23.33 / fgi 24.15 / fgLPIPS 0.181 / oom 0.004 / tOF 2.51 (FTGS++ 흰 22.49/21.76/0.229/0.87/2.61).
- EXP-002 지시대로: 22.87/23.28/0.189/tOF 2.25 = 안쪽층만(바깥층 0.3% 생존). EXP-002b dropout: 22.98/24.25/0.187/tOF 2.15, fg PSNR 20.88. 안쪽층 알파 비율 80%(바깥층은 주로 머리카락).
- 다음: EXP-003(r15h35) 바깥층만 / 둘 다 / 둘 다+dropout 학습 중. 결정 대기: dropout 채택 여부, 바깥층 비중 키우기(dropout 0.8 또는 안쪽 켠 반복에서 불투명도 규제 끔).
- EXP-003 (r15h35, 35° 링, test_extrap): 바깥층만 17.52/fgi 13.32/fgLPIPS 0.283/tOF 2.61; 둘 다(지시대로) 17.24/17.99/0.345/2.46 (= 안쪽층만 17.24/18.18/0.342/2.42); 둘 다+dropout 16.37/15.42/0.345/2.46. 안쪽층은 바깥층의 빈 곳(다리)은 채우나, 바깥층이 틀리게 덮은 정수리는 가려져 못 고침. 결과 `results/EXP-00{1,2,3}/`, 아티팩트 v4.
- 결정 대기: (1) dropout 채택 여부 (2) 바깥층이 학습 뷰에서 덜 보인 곳에서 물러나게 하는 장치(관측 횟수 기반 불투명도 감쇠 등) 실험 여부 (3) EXP-001/002b 시드 확인.

## 2026-10-17 (계속) — 이름 통일 + EXP-004 New-fade (D-041)
- 이름: SR4D / FTGS-only / New / New-drop / New-fade. 층 기여 색은 바깥 = 주황, 안쪽 = 파랑. 카드 그림·궤도 영상을 모두 다시 만들었다(`scripts/rebuild_exp_figs.sh`). 맨 위에 그림 한 장(`tools/exp_top.py`)과 가까운/먼 통합 표를 두었다. 아티팩트 v5.
- EXP-004: 재학습 없이 렌더 규칙만 더했다(`sr4d/layered/fade.py`, `render.py --fade`, `scripts/run_exp004.sh`, `scripts/run_exp004_extra.sh`). 결과는 `results/EXP-004/metrics.csv`.

| θ1, θ2 | 가까운 fg 내부 | 먼 fg 내부 | 먼 co-visible | 안쪽층 비율 (가까운/먼) |
|---|---|---|---|---|
| New-drop | 24.25 | 15.42 | 17.89 | 80% / 65% |
| 5°, 15° (추가) | 23.42 | 18.18 | 20.25 | 97% / 99.9% |
| 10°, 25° | 24.22 | 17.85 | 19.99 | 82% / 89% |
| 15°, 30° | 24.25 | 16.21 | 18.64 | 80% / 74% |
| 20°, 40° | 24.25 | 15.54 | 18.03 | 80% / 67% |

- 기준: 가까운 시점 SR4D 23.22, 먼 시점 SR4D 18.18.
- 두 목표(가까운 ≥ 24.25, 먼 ≥ 18.18)를 함께 넘는 θ는 없다. (5°, 15°)의 먼 시점은 바깥층을 거의 끈 결과라 SR4D와 같다.
- 결정 대기:
  - θ를 데이터별로 다르게 쓸지;
  - 바깥층 학습을 바꿀지(덜 관측된 곳을 만들지 않게);
  - 시드 확인은 그 뒤에.
- 정정 (2026-10-17): "dance는 생성 시드 123 데이터가 없다"는 보고는 틀렸다. `data_dev/fouranyone_gen/dance5885633_gen48_2ring_s123`(48뷰 × 60프레임, gen48_ho8·r15h35 split)이 있다. 시드 확인 때 생성 시드 123을 쓸 수 있다. 다만 SR4D L4 끔 s123 안쪽층 체크포인트가 먼저 필요하다.
- 인수인계 문서 `README.md`를 작성했다.

## 2026-10-17 (밤) — D-042: New-fade (10°,25°) 고정, EXP-005 결과, EXP-006·007 실행
- 1번: 맨 위 그림 열을 GT / SR4D / New-drop / New-fade / DualGS / 4DGS / FTGS++로 바꾸고 표에 떨림 열을 추가했다(New 계열은 안쪽층 값). 아티팩트 v7.
- EXP-005 (`train.py --cov_min_views`, `scripts/run_exp005.sh`, `results/EXP-005/metrics.csv`), dance, 시드 0:

| | 가까운 fgi | 가까운 fgLPIPS | 먼 fgi | 먼 co-vis | 먼 fgLPIPS |
|---|---|---|---|---|---|
| New-fade (10°,25°) | 24.22 | 0.1869 | 17.85 | 19.99 | 0.3420 |
| N3 + fade | 24.21 | 0.1865 | 17.52 | 19.95 | 0.3426 |
| N6 + fade | 24.18 | 0.1867 | 17.63 | 19.97 | 0.3429 |
| N3 fade 없음 | 24.25 | 0.1870 | 14.87 | 17.62 | 0.3468 |
| N6 fade 없음 | 24.21 | 0.1872 | 14.81 | 17.83 | 0.3472 |

- 결론: 먼 시점이 나아지지 않았다(오히려 −0.22~−0.33 dB). 먼 시점 정수리 확대 칸 바깥층 기여의 92%가 학습 카메라 12대 이상에 보인 가우시안이라 뷰 수로는 걸러지지 않는다. 규칙상 N = 3.
- EXP-006·007 큐 실행 중 (`scripts/queue_exp006_007.sh 3`, tmux sr4djobs:exp006_007, 감시 logs/watch_exp006.out): ZJU ho8/r15h35 × {N3, New-drop}, dance 바깥층 시드 1·2 × {N3, New-drop} × {ho8, r15h35}. 모든 run에서 fade 렌더와 fade 없는 렌더를 함께 채점.
- EXP-006 (ZJU, 시드 0, `results/EXP-006/`, 카드 EXP-006):
  - New-fade: 가까운 fgi 26.11, fgLPIPS 0.103 / 먼 fgi 20.70, co-vis 20.67, LPIPS 0.1385.
  - SR4D: 가까운 24.36, 0.114 / 먼 20.51, co-vis 20.47, LPIPS 0.139.
  - N3 + fade는 New-fade와 동률(26.13 / 20.70)이다.
  - 기준선: DualGS 가까운 26.18, FTGS++ 먼 21.00(LPIPS 0.307).
- EXP-007 (dance 바깥층 시드 0·1·2, `results/EXP-007/metrics.csv`, `summary.json`):
  - New-fade: 가까운 24.22 ± 0.03 / 먼 17.86 ± 0.15 / co-vis 20.03 ± 0.16.
  - N3 + fade: 24.22 ± 0.02 / 16.99 ± 0.51 / 19.46 ± 0.45 → N3 기각.
  - New-fade − SR4D: 가까운 +1.00, 먼 −0.31, co-vis −0.22.
- 아티팩트 v9. 실행 중인 작업 없음.

## 2026-10-18 — 흰 줄 진단(DIAG-1), EXP-008 실행, 안쪽층 L4 끔 재학습 (D-043)
- 진단 결과(`results/DIAG-streak/`, 카드 DIAG-1): 흰 줄은 바깥층에서 나온다. 흰색(휘도 1.00), 불투명도 0.87–0.89, 축 비율 중앙값 60–12,000이고, 65–75%가 visual hull 밖에 있으며, 98%가 학습 카메라 12대 이상에 보였다. 정수리 오차의 73–80%를 차지한다.
  - 배경색 줄 가우시안: New-fade 모델(먼) 흰색 2,102개, N3 3,834–5,633개, FTGS++ 검정판 검정 879개(흰 0), FTGS++ 흰판(가까운) 흰 690개, 회색 배경 렌더에서는 배경 전체가 흰색이다.
  - 가설이 맞다.
- FTGS++ 코드에는 배경 옵션, 알파 손실, 배경 무작위화, 모양 규제, opacity reset이 모두 없다. FreeTimeGS 논문(전체 장면 데이터)과 같다. 이 저장소는 FreeTimeGS++ 논문의 공식 코드다.
- 실행 중:
  - EXP-008 (`scripts/queue_exp008.sh 2 3`, tmux sr4djobs:exp008, 감시 logs/watch_exp008.out).
  - 안쪽층 재학습 (`scripts/queue_inner_nol4.sh 0 1`, tmux sr4djobs:inner_nol4, 감시 logs/watch_inner.out). run당 약 2.3시간, 7개.

## 2026-10-18 02:00 — EXP-008 완료 (D-043 결과)
- (a) 투명 배경(무작위 배경 + 알파 L1 0.05)을 채택했다.
  - 흰 줄: 2,102개 → 1개.
  - 먼 시점 fg 내부 17.85 → 18.80으로, SR4D 18.17을 넘었다. co-vis는 19.99 → 20.97이다.
  - 가까운 시점은 24.22 → 24.39로 올랐지만, LPIPS가 0.1869 → 0.1930으로 나빠졌다.
- (b)는 줄이 0개지만 먼 시점이 18.51로 (a)보다 낮다.
- FTGS++ (투명): 가까운 23.99, 먼 16.33, 먼 LPIPS 0.275.
- 투명 학습 뒤에는 fade가 손해다(fade 없음: 먼 19.15). 사용자 결정 대기.
- 그림: `docs/gallery/pub2/exp/exp008/`. 아티팩트 v11.
- 안쪽층 L4 끔 재학습: 시드 42 dance·ZJU r15h35가 약 39k/56k까지 진행됐다.
- 다음: EXP-009(새 안쪽층 위 바깥층 재학습)는 사용자 확인 후 시작한다.

## 2026-10-18 11:50 — D-044: New-T 채택, 디스크 정리, EXP-009 시작
- 안쪽층 L4 끔 재학습 7개가 모두 끝났다(09:51).
- 디스크 정리(A+B, `logs/cleanup_20261018.txt`): 14 GB를 지웠다. 여유 공간은 31 GB에서 45 GB가 됐다. 최종 체크포인트는 모두 남겼다.
- EXP-009 큐 시작(tmux `sr4djobs:exp009`, 감시 `logs/watch_exp009.out`).
  - 작업은 21개다: New-T 7, DualGS 2, FTGS++ 투명 6, 4DGS dynerf 6.
  - DualGS는 개당 약 6–7시간이 걸린다.

### 2026-10-18 17:35 — EXP-009 4DGS dynerf 렌더 실패 복구
- 재큐한 dynerf 6개는 학습은 끝났으나 `fourdgs_render_native.py`가 `a.orbit` AttributeError로 6개 모두 렌더 실패(4DGS `get_combined_args`가 None 인자를 버림). `getattr`로 수정.
- 큐 로그 `rc=$?`가 같은 echo 안의 `$(date)`에 덮여 rc=0으로 찍혀 실패를 놓침 → 새 큐(`scripts/queue_exp009_dynscore.sh`)는 rc를 먼저 저장.
- 대기 루프의 `grep QUEUEDONE`이 `DYNQUEUEDONE`에도 걸려 일찍 끝남 → `^QUEUEDONE`으로 수정.
- 재학습 없이 렌더+채점만(`scripts/baselines/score_4dgs_dynerf.sh`), GPU 2·3, 17:32 시작. 끝나면 GPU2에서 궤도 영상 재생성.

### 2026-10-18 18:10 — EXP-009 완료
- 전 작업 완료(New-T 7 + 재사용 1, DualGS 2, FTGS++ 투명 6, 4DGS dynerf 6). 표 `results/EXP-009/`, 그림·궤도 영상 `docs/gallery/pub2/exp/exp009/`, 카드 EXP-009.
- 요약: dance는 New-T가 fg 내부 1위(24.56 / 19.01). ZJU는 가까운 비김, 먼 시점 FTGS++ 투명이 +0.78 dB. 먼 LPIPS는 FTGS++ 투명이 일관되게 낫다. 상세는 DECISIONS D-044 끝.
- 상단 표: SR4D·New-T 먼 시점을 L4 끔 안쪽층 기준으로, 4DGS 먼 시점을 dynerf로 교체. 상단 그림은 EXP-009 dance 그림으로 교체.
- 다음: 사용자 결정 대기.
