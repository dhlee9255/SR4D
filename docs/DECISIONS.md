# DECISIONS

Design changes and deviations from CLAUDE.md, with reasons. (2026-10-08~)

## D-001 New `sr4d` env: torch 2.4.1+cu118, gsplat 1.5.3
- The system nvcc is 11.8, so cu118 is required. gsplat 1.5.3 = the same version FTGS++ was validated with. No prebuilt wheel exists (the docs.gsplat.studio index has no cu118 wheel for pt24), so it was installed from PyPI source and JIT-compiled on first use (~75 s, cached afterwards).
- chumpy (needed to load the SMPL .pkl) is installed with `--no-build-isolation`.

## D-002 SMPL-X models are copied into the SR4D repo
- `4Danyone/models/body_models` only has NEUTRAL. DNA-Rendering needs gender-specific models, so NEUTRAL/MALE/FEMALE npz files were extracted from `4Danyone/models_smplx_v1_1.zip` into `SR4D/assets/body_models/smplx/`. SMPL (for ZJU) is a symlink to GauHuman's `SMPL_NEUTRAL.pkl`.

## D-003 Develop on ZJU-MoCap 387 until DNA-Rendering is available  ← **user confirmation needed**
- DNA-Rendering is not available locally and needs an application plus license consent (CLAUDE.md rule 3). Development uses ZJU-MoCap-refine `my_387` (23 real cameras, a single ring, SMPL), which is already on the machine.
- Because it has only 23 cameras, the settings are `real_all`(17) / `real12` / `real8` (not 48/24/16). Evaluation cameras: interp {01, 09, 16}, extrap {06, 07, 08} (contiguous block, ~25° from the nearest training camera). It is a single ring, so there are no elevation-extrapolation cameras.
- The SR4D scaffold supports both SMPL and SMPL-X (`body_model.py`).

## D-004 Body surfels are rendered as flattened 3DGS
- gsplat's `rasterization_2dgs` works, but it cannot be composited with 3DGS (free Gaussians) in one pass. So, as CLAUDE.md 7.4 allows, all Gaussians are rendered in one 3DGS pass, with the body surfels' third axis fixed to `eps_scale` = 0.5 mm. The normal is composited as an extra channel and a normal-consistency loss against the depth-derived normal is applied.
- Covariances are passed as full 3×3 matrices via gsplat `covars` (Σ' = MΣMᵀ for non-orthogonal LBS blend matrices).

## D-005 Unified data: undistortion + principal-point centering in a single remap
- FTGS++ (SIMPLE_PINHOLE, cx=W/2), DualGS and 4DGaussians (Inria FoV cameras) all **ignore** the principal point. So the converter combines undistortion and remapping to a square-pixel, centered principal point in one `initUndistortRectifyMap` (Lanczos) pass. Every method then uses the same pixels and the same K, and no method needs extra resampling. Verified by an identical SMPL-mask IoU (0.786 → 0.785).
- K is stored in the OpenCV pixel-center convention. gsplat puts pixel centers at +0.5, so cx,cy are shifted by +0.5 when passed to gsplat (`dataset.Camera`, baseline wrappers). COLMAP cameras.txt also gets +0.5.

## D-006 L2 spatial encoding = multi-resolution dense grid (grid_sample)
- tinycudann is not installed in sr4d (build cost). The pure-PyTorch hash grid made S3/S4 about 5× slower (1.7 it/s), so the default is now `F.grid_sample` dense grids at 32³/64³/128³ × 8 channels (~6 it/s). The hash grid remains available via `model.l2.encoding=hash`.

## D-007 S1 = asymmetric silhouette loss + accept only if IoU improves
- There is no differentiable mesh rasterizer (pytorch3d/nvdiffrast). SMPL surface samples (30k, 8 mm isotropic) are rendered with gsplat to get a soft silhouette. The body leaving the mask costs ×1, the mask not being covered costs ×0.3 (clothing), plus a prior on pose/transl/betas.
- If IoU decreases after S1, the dataset SMPL is restored (lr 3e-3 lowered IoU 0.845 → 0.827; lr 5e-4 raised it to 0.845 → 0.854).
- The keypoint loss is not implemented yet (ZJU uses body25 2D keypoints; to be added along with the DNA keypoint format).

## D-008 L3 joint SE(3) residual = a pose residual inside forward kinematics
- Instead of applying "SE(3) per joint" directly after LBS, the residual is added to the joint axis-angle (plus global translation) and passed through FK. The effect is the same, but the skeleton chain stays intact. Free-bone residuals are added to each bone's per-frame SE(3). The color affine is applied after rendering to the premultiplied foreground, so the white background is not altered.

## D-009 Densification = per-set MCMC (gsplat relocation kernel)
- Body and free sets are relocated/grown separately. Body copies inherit the source triangle with jittered barycentrics; free Gaussians get MCMC position noise. The free set's LBS vertex index is recomputed at every relocation.

## D-010 Not implemented yet: S5 (L3→L2 distillation), S6 (fixer virtual supervision), L3 transient Gaussians, L4 initial confidence from cross-view reprojection consistency
- All are off by default in CLAUDE.md. L4 confidence starts from a constant 1 (logit 0) and is learned with the w·res − log w loss.

## D-011 Baseline background handling
- First attempt: FTGS++ / 4DGS trained on black composites, then composited onto white with the rendered alpha. → **Failed**: on a black background, alpha is unconstrained wherever the color is black, so dark floaters cost nothing in training (FTGS++ PSNR 17.0/16.3, full of floaters). Discarded.
- Final: every baseline is **trained on white-composited images with a white rasterizer background**.
  - FTGS++: `scripts/baselines/ftgspp_train_whitebg.py` monkeypatches `gsplat.rasterization` in memory → for RGB calls (sh_degree set), the result is `img + (1−α)`. The call itself (packed mode) is unchanged, because FTGS's densification state needs the packed outputs (gaussian_ids). (`backgrounds=` + `packed=False` was tried and broke FTGS's state; with packed=True, gsplat 1.5.3 mis-shapes the backgrounds tensor.)
  - 4DGS: `scripts/baselines/fourdgs_train_whitebg.py` replaces the camera `bg` from Panoptic `setup_camera` with white (`_replace`).
  - DualGS: its default `white_background=True`.
  - Rendering: FTGS++ uses its own wrapper (rendered alpha → white); DualGS/4DGS use per-frame PLYs + gsplat (`backgrounds=white, packed=False`).
- No baseline is evaluated with its own script: renders at the evaluation cameras go through the SR4D common `sr4d/eval/metrics.py`.
- FTGS++ / 4DGS are exported with **training cameras only** (FTGS's RoMa init matches every camera in the memmap → including test cameras would leak them).

## D-012 FTGS++ init cheirality bug — fixed in memory via a wrapper
- `ftgspp/init/points.py:matches_to_points` tests cheirality as `w2c[2,:3]·(xyz − w2c[:3,3])` (the correct value is `w2c[2,:3]·xyz + w2c[2,3]`).
  - ZJU: only 0–9% of points survive per camera → 411 Gaussians → all Gaussians die by iteration 600 (first run, failed).
  - **4DAnyone rig: all points from ~half of the 24 cameras (yaw 90–270°) are dropped** (measured: 0% pass on 11 of 24 cameras). This likely affects existing 4DAnyone FTGS++ results too → worth reporting to the user.
- `scripts/baselines/ftgspp_points_fixed.py` monkeypatches only that function (identical except for the depth test) and runs `ftgspp.init.points.main`. No external file is modified. After the fix: ~700k points per keyframe.

## D-013 4DGaussians hyperparameters
- `mmcv` is not in `hust4dgs`, so `--configs` cannot be used, and the existing env is not modified. The `arguments/dnerf/dnerf_default.py` values are passed as CLI flags; only `multires` (a list) cannot go through argparse, so the default [1,2,4,8] is kept.
- The per-frame PLY export uses `scripts/baselines/fourdgs_export_frames.py` (the original `export_perframe_3DGS.py` crashes on Panoptic dict cameras).

## D-014 B-stage data (4DAnyone generated) — coordinate frame and registration
- Coordinate frame = **4DAnyone `canonical_human_world`**. The generated rig and GVHMR SMPL-X live there natively, so the SMPL-X never needs scaling. The real cameras (evaluation GT) are mapped into it with a similarity transform S.
- S = Umeyama (real SMPL(-X) joints 0..21 → GVHMR joints, all frames). The GVHMR → canonical world transform is recomputed with 4DAnyone's own `_body_geometry` (via `tools/source_camera/geometry.recompute_body_geometry`; this older run's outputs don't store it).
- ZJU-387 test (`data_dev/zju_mocap_gen/387_gen24`): scale 1.153, mean joint residual 8.1 cm (p90 16.6) = GVHMR monocular error. The center differs by 2.6 m from 4DAnyone's own source-camera registration (`source_camera.json`; focal 2013 px, distance 6.2 m), but that registration lies along the depth/focal ambiguity, so it is not used as a reference.
- GVHMR SMPL-X projected into the real cameras has IoU ~0.5 (generated cameras ~0.8+). This is the "generated views inherit HMR error" from CLAUDE.md 5.4-4 → S1 should only correct β and small errors.
- Hands: GVHMR (supermotion) uses `flat_hand_mean=False` with zero PCA → absolute hand pose = hand mean.
- 4DAnyone can only specify target cameras through ring parameters → for DNA-Rendering B-stage, "generating at the same positions as real cameras" needs either injecting cameras via the internal API or "ring generation + real cameras as GT". Left for later.

## D-015 (experimental, default off) temporal smoothness of free bones `train.loss.bone_smooth`
- On ZJU-387 real12, SR4D's temporal error (0.055) is worse than 4DGS (0.044), and S2-only (no L3) is even worse (0.059) → suspect the free-bone per-frame SE(3) (which is only pulled toward identity). Adds Σ‖B_t − B_{t±1}‖² as an option. Default 0 keeps the CLAUDE.md design; adopting it is up to the user.

## D-016 Interpreting the temporal metric + tOF added
- The CSV `temporal` = mean|(R_t−R_{t−1})−(G_t−G_{t−1})| (CLAUDE.md §8 as written). On ZJU-387: a **static render would score exactly mean|ΔG| = 0.041**, while SR4D scores 0.055 and 4DGS 0.044. 4DGS's render change between frames (|ΔR| = 0.021) is only half the GT's (0.041), i.e. it under-moves / is temporally blurred, which flatters it on this metric. → The metric favors blur and stillness.
- Plain warping error also favors blur (4DGS goes negative relative to GT). → Added **tOF** (RAFT-large flow on the render sequence vs on the GT sequence, endpoint error, foreground) to the JSON. ZJU-387 at 50 fps moves only ~0.45 px/frame, so stride 1 is noise-dominated → default `--tof_stride 5`.
- ZJU-387 real12, tOF@5 (static render = 2.25 px): SR4D 1.24 / no_l3 1.29 / **4DGS 1.14**. 4DGS's motion fidelity is slightly better. Likely cause: SR4D follows ZJU's SMPL poses (S2+ pose lr 1e-5 ≈ frozen). Raising the pose lr to 1e-4 lowered S2 val PSNR (24.73 → 24.10 @8k), so another approach is needed, e.g. a per-frame temporal smoothing prior on the pose plus a medium lr (not tried yet).
- `bone_smooth=1.0` (D-015): temporal 0.0550 → 0.0555, PSNR −0.15 dB → no effect, not adopted.

## D-017 (experimental, default off) pose temporal prior `train.loss.pose_smooth`
- `scaffold.smooth_loss`: 2nd-order (acceleration) penalty on the per-frame pose/transl, so a higher pose lr (1e-4) can follow the images without jitter. ZJU-387 real12 (`sr4d_posesmooth`): interp 24.97 / extrap 23.56, tOF@5 1.33/1.33 (full SR4D 1.24) → no gain, not adopted.

## D-018 4DGaussians: prune unblocked via `FOURDGS_PRUNE_MIN` (third_party copy edited)
- Upstream `train.py:274` prunes only when the model has >200 000 Gaussians. On ZJU-387 the model stays at ~50–75k, so **pruning never ran** in the earlier 4DGS results (flagged by the user from a previous project).
- `third_party/4DGaussians` is our own clone (not one of the read-only repos): line 274 now reads the threshold from env `FOURDGS_PRUNE_MIN` (default 200000 = upstream). Run `4dgs_prune` uses 0.
- Init colours: `init_pt_cld.npz` rgb is a constant 0.5 in [0,1] (the Panoptic reader uses it as-is, no /255 issue). Masked background compositing: yes (white, D-011).

## D-019 Bug fix: `BodyModel.forward` / `Scaffold.canonical` joint positions
- Both returned `A[..., :3, 3]` (translation of the rest→posed relative transform) as the joint position. Correct: `A[:3,:3]·J_rest + A[:3,3]`. Verified: now matches J_regressor(posed verts) within 1 cm.
- Impact: (1) **B-stage joint-Umeyama registration** (`convert_4danyone.py`) was fitted on wrong joints → scale 1.153 / residual 8.1 cm; fixed → scale 1.008 / residual 4.1 cm, real-camera IoU 0.53 → 0.83 (see D-020). (2) Free-bone parent assignment used wrong canonical joints, but `FreeBones.parent` is not used by the deformation (bones act in canonical space, then LBS) → **A-stage SR4D numbers are unaffected**.

## D-020 B-stage registration v2
- `387_gen24_v2`: joint Umeyama with the D-019 fix (no evaluation data used). Real evaluation cams: SMPL-X/mask IoU **0.827** (r01..r16: 0.805–0.850), RTMW-x keypoint error 6–10 px (512²). Matches the previous project's expected IoU ≈ 0.83.
- `tools/register_real_cams.py`: R = exp([w]×)R_init, t = t_init+dt, loss = soft-silhouette **Dice** + Huber(RTMW-x 2D keypoints, δ 5 px) (+ tiny priors). No MSE (degenerate off-screen solution). Keypoints: `tools/detect_keypoints_rtmw.py` (pose_wb env, rtmlib performance = RTMW-x).
  - `similarity` fitted on the source camera only (`387_gen24_v2sim`): source IoU 0.841→0.868 but eval cams 0.827→0.719 (over-fits the monocular view) → **not adopted**. The earlier `387_gen24_reg` (MSE silhouette on wrong-joint init) is superseded.
  - `per_camera` (`387_gen24_v2pc`): eval IoU 0.850, kp 5.6–8.3 px, but it uses evaluation masks/keypoints for pose → sensitivity check only.
- **Main B-stage evaluation cameras = `387_gen24_v2`.** Mask-interior photometric L1 (suggested) is not used: it needs a reconstruction to render, which would bias the registration toward that method.

## D-021 FTGS++ protocol variants (from the previous project's notes)
- `tools/check_ftgspp_export.py`: our `export_ftgspp.py` poses_bounds is **identical** to legacy `build_ftgspp_dynerf.py` (pose columns diff 0; both round-trip through `ftgspp.data.sfm.parse_poses_bounds_npy` to 9e-16). Only near differs (0.1 vs 0.3). → The camera drop is **not** a pose-direction flip: it is the cheirality formula (D-012). Measured with the official formula: ZJU real12 0/12 cameras pass, 4DAnyone gen24 ring 13/24 pass (11 fully dropped). The ring still trains because the surviving half covers 360°.
- FTGS++ `render.py` FOV bug (fov from `dataset.intrinsic[0]` only) confirmed; **we never used render.py** — `ftgspp_render.py` passes each camera's own K. Training uses per-image K (`multiview.py:310` → `train.py:122`). Undistortion is applied at conversion (dist = 0).
- `run_ftgspp.sh` now takes `BG=white|black|raw` (black/raw = official `ftgspp.train`, no patch) and `INIT_FIX=0|1`. Legacy-recommended variants run: `ftgspp_black`, `ftgspp_raw`.
- Init frame check (`tools/check_init_in_masks.py`): FTGS++ RoMa points 90%, 4DGS/DualGS hull+SMPL 97% inside all 12 training masks (3 px tolerance) → no Z-up/Y-up mix-up. (We don't use 4DAnyone's `build_sparse_point_cloud`.)

## D-022 (2026-10-08) ZJU mask source: `mask/` -> `mask_cihp/` (user-spotted GT issue)
- User noticed GT looked ragged/flickery in the render viewer. Cause: `convert_zju_mocap.py` read ZJU `mask/` (per-frame matting), which tears on the dark trousers/shoes against the dark background and attaches floor shadow at the feet. `mask_cihp/` (CIHP parsing, used by GauHuman `scene/dataset_readers.py:1112`) is clean.
- Consecutive-frame mask IoU (cams 01/07/09): 0.977 -> 0.983. SMPL-projection IoU 0.785 -> 0.789 (bounded by naked SMPL).
- Added `--mask_dir` to the converter; new dataset `data_dev/zju_mocap/387_cihp` (cameras identical, old `387` kept). All A-stage numbers so far use the old masks -> to be rerun on 387_cihp.
- User also directed: B-stage GT should be generated views (not real cams). Pending: holdout layout decision.

## D-023 (2026-10-08) Free-bone parametrisation variants (user-approved design change)
- Diagnosis (tools/diag_jitter.py): per-frame independent free-bone SE(3) causes the shaking (accel RMS 12.3 mm -> 2.2 mm with bones = identity; SMPL poses / L2 / L3 not the cause).
- `model.free.bone_mode`: free (old) | pose_mlp (V1: MLP(pose_t, pose_t-pose_{t-1}, pose_t-pose_{t-2})) | spline (V2: cubic B-spline, control point / 4 frames) | mlp_spline (V3: V1 + V2 residual, `train.loss.bone_res_reg`=1.0). `n_bones: 0` = V0 (free Gaussians follow body LBS only).
  Configs: configs/ablations/bones_v{0..3}.yaml.
- Pre-registered decision rule (user): accel RMS <= 3 mm and tOF <= 4DGS; fg-interior PSNR / LPIPS not worse than current by > 0.2 dB / 0.005; pick the simplest passing (V0 < V2 < V1 < V3). Run on ZJU real12 (mask_cihp) and on the 4DAnyone dance clip (loose clothes / hair); winner goes to B-stage.

## D-024 (2026-10-08) B-stage is generated-only (user rule)
- Models trained on generated views are validated and scored ONLY on generated images. No real-camera registration, scoring or in-training validation.
  Supersedes CLAUDE.md 5.3/9 "gen24 evaluated on real hold-outs" and D-014/D-020 for evaluation purposes (registration code kept for reference only).
- `tools/convert_4danyone.py` without `--real_root` = generated-only dataset (split gen24: train = test_interp = all g*). ZJU: data_dev/zju_mocap_gen/387_gen24_genonly (SMPL-X identical to the old root, checked), dance: data_dev/fouranyone_gen/dance5885633_gen24.
- L4 in B: camera refinement on, colour affine off (`model.l4.color: false`); colour affine only in A. B-stage main metrics: fg-interior PSNR and LPIPS of L4-off renders.

## D-025 — B단계 자세 보정 모드 비교 (사용자 승인 2026-10-10)
- 배경: dance gen24에서 V0–V3 모두 떨림 ≈4 mm, 입력 GVHMR 자세로 바꾸면 ≈2.2 mm → 남은 떨림은 프레임별 SMPL 자세 보정(S1 + S2~ 공동 최적화).
- `model.scaffold_pose`: `free`(기존 프레임별) / `global`(안 a: 자세는 입력 고정, β와 시퀀스 공유 이동 오프셋만) / `spline`(안 b: 입력 + 시간 cubic B-spline 보정, 제어점 4프레임 간격, `scaffold_spline_step`).
- 비교: r15h35(ZJU 생성뷰, dance) × bones V0. 판정: 떨림(가속도 RMS), tOF, r15h35 fg 내부 PSNR·LPIPS. 같으면 (a).
- 구현: `sr4d/model/scaffold.py` (pose/transl를 모드별 property로, 기존 체크포인트의 pose/transl 키는 pose_p/transl_p로 자동 변환), `configs/ablations/bones_v0_pose_{a,b}.yaml`.
- **판정 (2026-10-10): (a) 채택.** ZJU 동률, dance에서 (a)가 떨림 1.61 vs 1.67 mm, tOF 2.42 vs 2.52, fgi 18.18 vs 17.32 dB로 우세(fgLPIPS만 (b) 0.0019 낮음). PROGRESS 2026-10-10 표.

## D-026 — bone V0 확정, 판정 규칙 교체 (사용자 결정 2026-10-12)
- 자유 bone: **V0(bone 없음) 확정.** real12 재판정은 하지 않음.
- D-023의 고정 기준(fgi −0.2 dB / fgLPIPS +0.005)은 폐기. 새 규칙 (비교마다 같은 조건의 시드 간 편차 σ 기준):
  - 차이 ≤ σ → 동률, 단순한 쪽 채택.
  - σ < 차이 ≤ 2σ → 시드 추가.
  - 차이 > 2σ 이고 두 시드 순위 같음 → 확정.
- σ 참고값(r15h35, 생성+학습 시드 42 vs 123): SR4D (a) fgi 0.49 / 0.17 dB (ZJU / dance), fgLPIPS 0.0025 / 0.0004.

## D-027 — r15h35 co-visible 주 지표 (사용자 결정 2026-10-12)
- dance 15° 링에서 무릎 아래가 프레임 밖 → 35° 채점에 학습에서 안 보인 부위 포함. Dycheck masked metric과 같은 개념의 co-visible 지표를 주 지표로, 전체 지표를 보조로 둔다.
- 판정은 방법과 무관: 생성 마스크(48뷰 전부, 마스크 5 px 팽창, 화면 밖 투영은 깎지 않음, 2뷰까지 불일치 허용)로 만든 visual hull의 첫 교차 깊이로 35° 픽셀의 3D 점을 구하고, 15° 학습 카메라 ≥2대에서 화면 안·마스크 안이면 채점. `tools/covis_masks.py`, `sr4d/eval/covis_metrics.py`.
- co-visible 비율: dance 0.869(ray miss 0.006), ZJU 0.971. 빠지는 곳: dance 종아리·발, 두 데이터 모두 정수리 일부.
- 4DAnyone 구도: CLI 옵션 없음. `fdanyone/geometry/framing.py`가 입력 영상에서 보이는 몸 하단(visible_body_bottom)을 분석해 생성 카메라 구도를 맞춤(dance 입력 = half_body → f/H 1.769, 무릎 근처에서 잘림; ZJU = full_body, f/H 1.300). 전신 구도로 바꾸려면 저장소 복사본(third_party)에서 `input_min_confidence`를 1 초과로 두는 등 적응 구도를 꺼야 하며, 그러면 입력에 없는 종아리·발을 생성기가 지어냄 → 사용자 확인 필요.

## D-028 — 시드 3개 보고 (사용자 결정 2026-10-12, 항목 6)
- B단계: 4DAnyone 생성 시드 42 / 123 / 7 (두 링을 통째로 재생성 → 생성+학습 편차). 모든 방법 r15h35에서 3시드, 평균 ± 표준편차.
- A단계 (ZJU-387 mask_cihp real12): 학습 시드 0(기존) / 1 / 2. `scripts/run_A_seed.sh`.
  - SR4D: config `seed`. FTGS++: 자체 `--seed-offset`. 4DGS(6666 고정)·DualGS(0 고정): 외부 코드는 그대로 두고 `scripts/baselines/seed_offset.py`가 메모리에서 random/np/torch 시드 호출에 TRAIN_SEED를 더함(4DGS는 `fourdgs_train_whitebg.py`에서, DualGS는 `run_dualgs_seed.sh`가 런처로 사용). TRAIN_SEED 미설정이면 기존과 동일.
- 최종 SR4D 구성: V0 bone(D-026), B단계 자세 (a)(D-025), L2 Δx·L3 유지(ablation, PROGRESS 2026-10-12).

## D-029 — 미관측 자유 가우시안 제거 (사용자 결정 2026-10-12)
- 벤치마크는 co-visible 주 지표로 유지, dance 전신 재생성은 보류. 최종 데모/제품 학습에는 15°+35° 두 링 모두 사용(하체 관측).
- 규칙: 자유 가우시안만(몸 surfel 제외). 모든 프레임·모든 학습 카메라에서 변형 후 중심이 화면 밖이면 제거, 한 번이라도 들어오면 유지. 학습 중 densification(relocation, 100 it)마다 적용. SR4D는 MCMC(개수 고정)라 "제거" = 불투명도를 dead 임계 아래로 → 같은 relocate()에서 관측되는 곳으로 재배치. `train.prune_unseen`, `sr4d/train/densify.py: unseen_free_mask, kill_rows`, `configs/ablations/prune_unseen.yaml`. relocation이 멈추는 마지막 20%에는 적용 안 됨.
- 사전 확인(학습된 dance 모델, 사후 제거): 자유 가우시안 6.4만/24만(26.5%)이 미관측, ZJU 0. 제거 시 35° 마스크 밖 알파 0.0448→0.0311, 줄어든 곳은 무릎 아래, co-visible 내부 PSNR −0.13 dB / LPIPS −0.0006.
- 부록: 같은 규칙을 4DGS(densify 직후 prune_points, 시간별 변형 위치)·FTGS++(relocation 직전 dead 표시, means_t, 학습 렌더에서 기록한 카메라·시각만 사용, 평가 렌더 제외)에 메모리 패치로 적용. `scripts/run_r15h35_unseen.sh`.
- SH 실험(사용자 항목 4): SH 차수는 몸·자유 가우시안 공통(한 패스). 현재 1에서 시작해 6000 it마다 +1, 최대 2 → 실험은 최대 2→3 (`scripts/run_sh3.sh`).

## D-030 — 주 프로토콜 변경, r15h35 부록 동결 (사용자 결정 2026-10-12)
- gen48_all: r15h35용 두 링 48뷰(생성 시드 42/123/7)를 전부 학습. 궤도·육안·인터랙티브 뷰어는 이 모델. split `gen48_all`(=기존 gen48; test=같은 뷰, 자체 적합만).
- gen48_ho8 (측정): 링마다 6뷰 간격 4대씩 8뷰 홀드아웃(15° g00/06/12/18, 35° g27/33/39/45 — 방위각이 겹치지 않게 3칸 엇갈림), 40뷰 학습. 최근접 학습 뷰 14.7°. 지표: fg 내부 PSNR, LPIPS, tOF, 마스크 밖 알파 (co-visible 보조). `tools/make_g48_splits.py`.
- gen24_all: 원래 24뷰 생성 전부 학습, GT 없는 지표 + 육안.
- r15h35: 외삽 부록으로 동결. 하체 잘림 대응에 추가 투자 없음(진행 중 D-029 실행만 마무리). r15h35용 SH3 대기열 취소 → SH 실험은 gen48_ho8로 옮겨 제안.
- 순서: 시드 42로 4방법 × gen48_all/gen48_ho8 먼저(`scripts/run_g48.sh`, PHASE B_g48), 최종 표만 시드 3개.
- DualGS: 비동기 CUDA 실패 시에만 CUDA_LAUNCH_BLOCKING=1로 재시도(노트에 표시). `tools/export_dualgs.py`는 학습 뷰와 겹치는 test 뷰를 eval_views에서 제외(자체 적합 설정용).

## D-031 — 흰 테두리 진단과 수정안 (사용자 단계 2–3, 2026-10-13; 수정 두 가지는 사전 승인)
- 진단 (gen48_ho8 ZJU 시드 42, 홀드아웃 8뷰 × 12프레임, `tools/diag_edge.py`, `outputs/sr4d_final/zju_gen48_ho8/edge_diag/`):
  - a) 경계 6 px 띠 오차는 좌·우(윤곽 법선이 수평) 쪽이 크고 위쪽이 작음 — 네 방법 공통 패턴. SR4D의 초과분은 좌 +10.6, 우 +13.7 (DualGS 대비)로 양쪽 대칭 → 한쪽 치우침 아님.
  - b) L4 카메라 보정: 회전 평균 0.16°(최대 0.37°), 카메라 중심 이동 |평균| 4.7 mm, 평균 벡터의 크기 1.2 mm → 장면 전체 이동은 작음(≈0.4 px).
  - c) 홀드아웃 전역 이동(위상 상관): SR4D |평균| 0.38 px(dx −0.35), 4DGS 0.18, FTGS++ 0.13, DualGS 0.13 → SR4D만 약간 크지만 서브픽셀.
  - d) GT 마스크 안쪽 3 px 띠의 렌더 알파: SR4D 0.849(최저), 4DGS 0.875, FTGS++ 0.863, DualGS 0.928; 내부는 모두 ≈0.999.
- 판단: 주원인은 "대칭 테두리 + 경계 알파 < 1"(흰 배경과 섞임) → 배경 무작위화. SR4D 전용 전역 이동도 약하게 있어(타 방법의 2–3배, 0.35 px) 규칙(둘 다 해당) 대로 세 가지 모두 학습.
- 구현: `train.random_bg`(샘플마다 무작위 배경, GT도 마스크로 같은 색에 합성), `model.l4.cam_mean_zero`(매 step 후 학습 뷰 cam_delta 평균을 0으로). `configs/ablations/{random_bg,cam_mean_zero}.yaml`, `scripts/run_g48_var.sh`.

## D-032 — 결과 표 고정 열 (사용자 단계 4, 2026-10-13)
- 모든 결과 표에 SSIM(전체·fg)과 경계/내부 분리 지표를 고정 열로: band = GT 윤곽 ±3 px 띠, interior = 3 px 침식 마스크(=fgi). `sr4d/eval/metrics.py`가 band_psnr 추가 계산, 기존 결과는 `tools/rescore_band.py`(`scripts/rescore_band_all.sh`)로 재채점 → `metrics_band_renders.json`(band/interior PSNR·MAE, 마스크 IoU).
- 논문 표: LPIPS와 마스크 밖 알파를 PSNR과 같은 비중(같은 줄, 같은 강조 규칙)으로 둔다. 사용자 육안 판단(gen48: SR4D > DualGS > 나머지)과 PSNR 순위가 다를 수 있음을 본문에 명시.

## D-033 — dance 얼굴 뭉개짐: 진단과 머리·목 자세 보정 (사용자 지시 2026-10-13)
- 사용자 육안 정정: dance gen48_all에서 DualGS가 더 선명(노이즈 약간), SR4D는 일부 프레임 얼굴 뭉개짐. ZJU는 SR4D가 나음.
- 얼굴 지표 `tools/face_metrics.py` (SMPL-X 머리 관절 중심 정사각 크롭, 크기 2.6×목–머리 거리; 마스크 가중 PSNR + 128 px LPIPS): dance gen48_all(학습 뷰 자체 적합) SR4D 19.06 dB / 0.235, DualGS 22.22 / 0.160 → 60프레임 전부 SR4D가 나쁨, 0–5번·40–50번이 최악.
- 머리 자세 확인 `tools/face_landmarks.py` + `tools/face_kp_check.py` (ViTPose-H, GVHMR 사본을 파일 경로로 로드, 4danyone 환경, CPU): GT 얼굴 키포인트 대 SMPL-X 얼굴 꼭짓점 0.26±0.03 머리 크기(프레임 간 거의 일정 → 주로 정의 차이), 얼굴 흐림과 상관 0.22. 렌더 대 GT: SR4D 0.07–0.12, DualGS 0.02–0.06 → SR4D가 GT 얼굴의 프레임별 움직임을 덜 따라감(뒤 프레임에서 증가). 경계선 판정 → 지시대로 머리·목만 보정 실험.
- 구현: `model.scaffold_pose: global_headspline` — 몸은 (a)처럼 입력 자세 고정 + 전역 이동, 목(12)·머리(15)만 시간 cubic B-spline(4프레임 간격). `configs/ablations/bones_v0_pose_headspline.yaml`. 떨림 3 mm 이하 유지 확인 필요.
- 앞으로 모든 결과 표에 얼굴 지표(face PSNR, face LPIPS) 추가.

## D-034 — 옷 점무늬 흐림: 원인 판별 (사용자 지시 2026-10-13)
- 1단계(CPU, 기존 결과, `tools/dots_diag.py`, dance 점무늬 고정 크롭 72 px = SMPL-X spine1 중심, 5프레임 × 학습 뷰 g01·g11(gen48_all 모델) + 홀드아웃 g00·g12(gen48_ho8 모델)):
  - 크롭 PSNR / 라플라시안 분산: 학습 뷰 SR4D 28.00/342, DualGS 31.51/2444, FTGS++ 28.48/769, GT 6107; 홀드아웃 SR4D 26.76/352, DualGS 25.19/3079, FTGS++ 26.37/1149, GT 6477 → 학습 뷰에서도 흐림.
  - 몸통+엉덩이 가우시안(불투명도>0.1): SR4D 몸 surfel 107,536 + 자유 62,586, DualGS 30,188(30번 프레임), FTGS++ 29,591(t=30/30 s, 시간 불투명도 포함) → 개수 부족 아님.
  - 마지막 10% train loss 기울기 −0.0062 / 1k it, train PSNR 23.5→25.5 dB → 아직 내려가는 중 → 규칙: 기존 체크포인트에서 2배 이어서 학습(GPU0, 단일 프레임 실험 뒤).
  - L4 confidence: 학습 뷰 거의 전 픽셀이 상한 e³≈20.1(점무늬 크롭 평균 20.09) → 국소 억제 없음, 대신 사진 손실 전체가 ~20배.
  - 주의: 사용자 지시의 SH3/rbg/cam0/rbgcam0는 ZJU 학습이라 점무늬 크롭에 넣을 수 없음.
- 2단계(GPU0): 30번 프레임만으로 SR4D 학습(L3 off, 나머지 기준선과 동일). 3단계: confidence off 전체 학습(dance gen48_ho8, 56000 it).
- 수정(사용자 2026-10-13): 2배 학습은 체크포인트 이어 학습이 아니라 스케줄 전체 2배 새 학습(112,000 it; S1 4k/S2 48k/S3 24k/S4 36k, body_adhere 6k, relocate_every 200, sh_every 12k — `train.sh_every` 신설, 기본 6000) — 학습 끝 학습률이 ~0이라 이어 학습은 반복 수 효과를 과소평가. `configs/ablations/schedule_x2.yaml`, dance gen48_ho8 시드 42, GPU0(단일 프레임 뒤). 학습 뷰·홀드아웃 뷰 모두 렌더해 같은 크롭 비교.
- **문제 기록: L4 confidence 포화.** 학습 뷰 거의 모든 픽셀이 상한 exp(3)≈20.1에 붙음 → 사실상 사진 손실 ×20, 불확실성 구분 기능 없음(신뢰도 손실 w·res − log w에서 res가 작으면 w가 상한까지 커짐). 수정 방향(끄기 / 이미지별 평균 1로 정규화)은 3단계(confidence off) 결과를 보고 결정.

## D-031 확정 / 새 기준선 (사용자 2026-10-13)
- cam0(L4 카메라 보정 평균 0)을 새 기준선으로. 이후 새로 시작하는 실행은 모두 cam0 켬(진행 중인 머리·목 보정, confidence 끔은 그대로). dance·시드 123 확인 실행은 계속.
- GPU 큐(우선순위 B1–B5): GPU0 단일 프레임 → 2배+cam0(x2cam0; 비교 기준 dance cam0 1배) | GPU1 머리·목 → dance cam0 → ZJU gen48_all cam0 | GPU2 머리·목 all → ZJU s123 기준·cam0 → confidence 정규화+cam0(dance, confidence 끔 뒤) | GPU3 confidence 끔 → dance gen48_all cam0 → L4 완전히 끔(ZJU ho8, cam0 비교).
- `model.l4.conf_norm`(confidence 이미지별 평균 1) 신설, `configs/ablations/{conf_norm,no_l4_all}.yaml`.

## D-035 — L2를 2D CNN으로 바꿀 수 있는 선택지 (사용자 지시 2026-10-13; 추가 선택지만, 기본값 MLP 유지)
- 동기: MLP L2(해시/격자 위치 인코딩 + 자세 코드)는 자세 정보가 공간적으로 정렬되지 않음. Animatable Gaussians(CVPR 2024)처럼 자세를 2D 맵 위에 그리고 CNN으로 풀면 국소 디테일(옷 주름·머리카락)을 공간적으로 일관되게 낼 수 있음.
- 맵: SMPL-X UV는 자유 가우시안(머리카락·치마)을 못 덮으므로, canonical 가우시안 전체(몸 surfel + 자유)를 canonical z축으로 정사영(u = x, v = −y, 고정 bbox; 256 × 비율). 픽셀마다 가장 가까운/먼 가우시안 깊이의 중간에서 앞/뒤 두 층으로 나눔.
- 입력(8채널): 각 가우시안의 현재 자세 LBS 위치(골반 기준 좌표, 전역 회전·이동 제거)를 자기 픽셀에 평균으로 그림 — 앞층 xyz + 점유, 뒷층 xyz + 점유. LBS 가중치는 몸 surfel = 삼각형 첫 꼭짓점, 자유 = lbs_vidx (기존 변형과 동일).
- 네트워크: 작은 U-Net(32-64-128-128, GroupNorm, 마지막 층 0 초기화 → 시작 시 출력 0), 출력 2 × 13채널. 각 가우시안은 자기 층·픽셀을 쌍선형으로 읽음. 출력 dict와 스케일(dx×0.05, dlog_s×0.1, dcolor×0.1, dq 잔차)은 MLP와 같아 이후 변형·손실 코드는 그대로.
- 정규화: L2 크기 손실은 그대로 적용. 공간 평활 손실(MLP용)은 CNN에서는 사실상 꺼짐(맵 보간이라 원래 매끄러움; 정규화 호출은 앞 샘플의 맵을 그래프 없이 읽음 — 그래프 재사용 오류 수정).
- 구현: `sr4d/model/pose_nonrigid_cnn.py`; 분기 추가만(`build.py`, `trainer.py` 최적화 그룹·로드, `deform.py` 호출) — MLP 코드·기본 경로 불변. 설정 `model.l2.type: mlp | cnn`(기본 mlp), `cnn_res`, `cnn_ch`; `configs/ablations/l2_cnn.yaml`.
- 동작 확인(본 학습 아님): dance gen48_ho8, cam0 켬, 가우시안 상한 12만으로 줄여 2000 it(S1 200 / S2 600 / S3 1200, L2는 S3부터). 오류 없이 완료 5.7 min, 최대 메모리 14.9 GB; 학습 후 |dx| 평균 6.9 mm(최대 31 mm), 체크포인트 재로드 정상, U-Net 1.21 M 파라미터. 화질 수치는 짧은 학습이라 의미 없음.
- 본 학습은 단일 프레임·2배 학습 결과를 보고 사용자가 지시.
- (사용자 2026-10-13 18:52) L2 CNN 본 학습 승인: dance gen48_ho8 시드 42, cam0, l2.type cnn, 56,000 it, 가우시안 상한 기준선과 동일(40만). GPU3에서 confidence 끔 직후. 비교 기준 dance cam0 1배. 중간(≥30k it) 떨림 측정, 3 mm 넘으면 즉시 보고. GPU3 기존 큐(dance gen48_all cam0 → L4 끔)는 CNN 뒤로.
- DualGS dance gen24_all: GPU2에서 SR4D와 공유로 프레임당 ~50 min(9/60) → 중단·부분 결과 삭제, GPU0에서 2배 학습 뒤 같은 설정으로 처음부터 재실행.

## D-031 철회 / L4 끔 실험 (사용자 2026-10-13 22시)
- cam0 미채택(ZJU +1.43, dance −0.90 dB로 방향 반대). 화질을 올린 변경은 confidence 끔뿐; cam0·2배·머리목·CNN은 개선 없음 → 공통 원인 후보 L4 카메라 보정.
- "L4 끔" = 카메라 보정 끔 + confidence 끔(색 보정은 원래 꺼짐) = `model.l4.enabled: false`. dance만.
- 큐: GPU3 dance L4 끔(56k) | GPU1 dance L4 끔 + 2배(112k; 표류 없는 반복 수 효과) | GPU2 dance L4 끔 + CNN(공정 재비교) | GPU0 DualGS dance gen24_all(3번째 시도; 2번째·재시도 모두 frame 1 skin "illegal memory access").
- 취소(중단, 부분 출력은 ckpt_final 없이 남음): dance gen48_all cam0, ZJU gen48_all cam0, ZJU 시드 123 cam0, confidence 정규화+cam0, ZJU L4 끔. ZJU 시드 123 기준선은 완료.
- 뷰어용 gen48_all 재학습은 L4 결론 후 최종 설정으로.

## D-036 — 일치율 confidence (사용자 지시 2026-10-13; 설정 `model.l4.conf_mode: consistency`, 기본 learned 유지)
- 목적: 포화된 학습형 confidence(×20, 구분 기능 없음) 대신, 생성 뷰 간 실제 일치도로 픽셀 가중. 여러 뷰가 일치하는 내용은 높게, 한 뷰에만 있는 무늬는 낮게.
- 계산(`sr4d/train/consistency.py`): 프레임 t의 각 학습 뷰를 현재 모델로 1/4 해상도 깊이 렌더(gsplat ED = 이미 depth/alpha) → GT 전경 픽셀 역투영 → 카메라 중심 각도 기준 가까운 학습 뷰 4장에 재투영, 가림 검사(|z − 깊이| < 2 cm + 2 %, 상대 마스크 안) → GT 색 차 exp(−‖Δc‖²/2σ²), σ = 0.1, 보이는 이웃 평균(없으면 1). 전경 평균 1로 정규화, 하한 0.1, 배경 1. 잔차로 갱신하지 않음: `train.cons_start`(기본 3000)에 처음, 이후 `cons_every`(5000)마다, s7 시작 시 재계산. 40뷰 × 60프레임 맵을 CPU에 보관.
- 손실: L1, SSIM(가중 SSIM 지도), LPIPS(공간 LPIPS 지도 가중 평균) 모두 같은 w. 학습형 confidence는 이 모드에서 무시.
- 동작 확인(2000 it 축소판, L4 끔 + s7 포함): 오류 없음; 맵 2400장, 전경 평균 1.002, 0.5 미만 8.3 %, 범위 0.10–1.51. 낮은 값이 윤곽과 점무늬에 몰림 — 짧은 학습의 거친 깊이에서는 일관된 고주파 무늬도 불일치로 잡힐 수 있어, 본 학습에서는 cons_start를 형태가 자리 잡은 뒤(예: S3 시작)로 늦추는 것을 권장.

## D-037 — 프레임별 디테일 층, 2단계 학습 (사용자 지시 2026-10-13; `train.stages.s7`, 기본 0 = 꺼짐)
- 1단계 = 기존 SR4D(공유 coarse). 2단계 s7: 공유 모델 전부 고정(가우시안·L2·L3·스캐폴드 최적화 단계 없음, MCMC 노이즈·densify 없음), 프레임마다 모든 가우시안의 Δ위치 dx[t]·Δ색 dc[t](SH0에 /C0로 더함)만 학습(`sr4d/model/detail.py`, 0 초기화, T × N × 6, N = s7 시작 시 개수 고정; 40만 × 60 ≈ 0.58 GB + Adam).
- 규제: detail_mag(dx는 cm², dc 제곱) + detail_smooth(이웃 프레임과 차의 제곱), 기본 0.01 / 0.01, lr 1e-4. 손실 가중은 D-036 confidence(설정 시).
- s7을 켜도 s2–s4는 기준선과 동일: 위치 학습률 일정과 relocation 종료(80 %)를 s7 이전 반복 수로 계산.
- 렌더·평가·뷰어: 체크포인트에 detail이 있으면 자동 적용(load_model).
- 동작 확인(위와 같은 실행, s7 900 it): s7 동안 공유 파라미터 변화 정확히 0, detail |dx| 평균 1.5 mm(최대 30 mm), 재로드 정상, 5.9 min, 최대 14.9 GB.
- "그 프레임에만 있는 소수의 가우시안" 변형은 아직 구현 안 함(오프셋 방식 먼저).

## D-038 — 공정성 점검 (사용자 지시 2026-10-13 밤; 읽기만, 수정·재실행 없음)
- 프레임: 4DAnyone 생성 121프레임(25 fps, 704×1280; ZJU·dance, 시드 42/123/7, gen24 원본 동일). 실험 = 앞 60프레임 0–59 연속(`tools/fouranyone_prep.py --frames 0 60`, `scripts/prep_2ring.sh`, configs `data.frames: [0, 60]`). 모든 split·SR4D 설정·4DGS/FTGS++/DualGS export_meta·평가(split frames) 동일 확인.
- 시간값: 4DGS t = rel/T (0…59/60, 학습·렌더 동일), FTGS++ t = frame/fps, fps = 30 고정(4DAnyone 설정; dance 실제 25 fps — 시간축 배율만 다름), DualGS·SR4D는 프레임 인덱스. ZJU meta fps 50은 생성 영상(25 fps)과 불일치하나 어느 방법도 사용 안 함.
- 해상도: 생성 704×1280 → 전 방법 학습·채점 352×640(1/2). DualGS `-r 1`(입력이 이미 352×640), FTGS++ poses_bounds 640×352, 4DGS train_meta 352×640, 렌더 4종 352×640.
- 옵션 차이와 공정성 위험: (1) 4DGS가 D-NeRF(단안 합성) 프리셋 사용(D-013: mmcv 없어 CLI 전달) — 공식 다중 뷰 dynerf(14k it, 배치 4, width 128, plane_tv 2e-4, time_smooth 1e-3)·multipleview(15k, 배치 1, width 128)와 다름, multires [1,2,4,8](dnerf [1,2] 전달 불가); (2) 학습 예산 방법별 기본값 그대로(이미지당 노출 SR4D 45 / 4DGS 9.6 / FTGS++ 12.5 / DualGS 763); (3) DualGS 초기화 = visual hull + SMPL-X 꼭짓점(공식 = 데이터셋 점군) — 몸 사전정보가 DualGS에도 들어감 → **유지, 논문에 명시** (사용자 2026-10-13); (4) FTGS++ legacy 검정 배경 학습 + 알파→흰색 합성 채점(다른 방법은 흰 배경 학습) — 검정에서 floater 무비용; (5) 1/2 해상도는 공식(원본 해상도)과 다르나 전 방법 동일.
- DualGS 오류: 같은 데이터·같은 옵션에서 성공/실패가 갈림(dance gen24: 1·2차 frame 1 실패, 3차 통과 9/60; ZJU s7: 비동기 2회 실패, blocking 성공) → 옵션 차이보다 비결정적 수치 문제(skin 단계 래스터라이저)로 보임.

### D-038 조치 (사용자 2026-10-13)
1. 4DGS 재학습: 공식 다중 뷰 프리셋 `arguments/dynerf/default.py`(14k it + coarse 3k, 배치 4, width 128, multires [1,2], k-planes [64,64,64,150], plane_tv 2e-4, time_smooth 1e-3) — `fourdgs_train_whitebg.py`의 메모리 내 mmcv.Config 대체(외부 저장소·conda 환경 불변), `scripts/baselines/run_4dgs_preset.sh`, 결과 `outputs/4dgs_dynerf/`. 병합값 확인 완료.
2. FTGS++ 흰 배경 학습(`BG=white`, D-011 래퍼, 공식 초기화) `outputs/ftgspp_white_official/` — 검정 배경과 나란히, 주 표는 홀드아웃에서 나은 쪽.
3. DualGS SMPL-X 초기화 유지, 논문에 명시.
4. 학습량 표(이미지당 노출·학습 시간)를 결과 아티팩트에 상시 포함.
- 대상: gen48_ho8 시드 42, ZJU·dance. 큐: GPU3(L4 끔 뒤) ZJU 4DGS→FTGS++ 흰 배경, GPU2(L4 끔+CNN 뒤) dance 4DGS→FTGS++ 흰 배경 (`scripts/run_fair.sh`).
- D-038 조치 결과(2026-10-14): 4DGS dynerf 프리셋 첫 실행은 DataLoader 작업자 교착(15 h 유휴) → dataloader만 끈 복사본 `configs/fourdgs/dynerf_default_nodl.py`로 재학습. 프레임 PLY 내보내기는 (1) cfg_args에 프리셋 병합값이 없어 폭 64로 재구성 → `--configs` 추가, (2) 4DGS `get_state_at_time`이 변형 불투명도를 버림 → 직접 `_deformation` 호출, (3) 그래도 SH·불투명도 변형 프리셋에서 색이 보라색으로 어긋남(4DGS 자체 렌더는 정상) → **4DGS 공식 렌더러로 평가 카메라를 직접 렌더**(`scripts/baselines/fourdgs_render_native.py`, 흰 배경, t = rel/T)해 채점. PLY 경로 결과는 `*_plyexport_INVALID`로 보관, results CSV의 첫 4dgs_dynerf 줄(fgi 19.08/19.50)은 무효. dnerf 프리셋(변형 불투명도·SH 꺼짐)은 PLY 경로 영향 없음.
- 공식 dynerf(공식 렌더러) vs dnerf(PLY): ZJU fgi 25.26 vs 24.93, fg LPIPS 0.152 vs 0.154; dance 23.76 vs 23.44, 0.280 vs 0.281 → 공식 프리셋이 약간 나음 → 주 표 4DGS = dynerf. 공식 렌더러는 알파를 안 내 마스크 밖 알파 없음.
- FTGS++ 흰 vs 검정(홀드아웃): ZJU fgi 24.35 vs 25.16, fg LPIPS 0.110 vs 0.256; dance 21.76 vs 23.60, 0.229 vs 0.358 → PSNR은 검정, LPIPS는 흰 배경이 우세(혼합). 흰 배경 모델의 마스크 밖 알파 0.86은 실제 값: 같은 렌더러로 낸 알파이며, 마스크 손실 없이 흰 배경을 학습해 배경을 흰 가우시안으로 채운 것(궤도·다른 배경에서 흰 안개로 보임). 이 때문에 주 표의 FTGS++는 검정 배경을 유지.
- D-039 (2026-10-15): 디테일 층 수정 실험. 확인 결과 (a)·(b)도 s7에서 LPIPS 손실이 **켜져 있었음**(가중치 0.05, 기준선과 동일; it ≥ S2 중반 조건 충족. (b)는 consistency 가중 공간 LPIPS). 따라서 (c)는 (a)와 LPIPS 동일, 차이는 이웃 프레임 평활 10배(detail_smooth 0.01→0.1)뿐. (d)는 Δ위치를 0으로 고정하고 Δ색만 학습(`train.detail_dx: false`, 저장 용량 절반). 둘 다 L4 끔 최종 56k 체크포인트 + 12,000회.
- 서버 재부팅(2026-10-15)으로 /tmp 스크래치패드의 갤러리 원본이 사라짐 → 아티팩트 v45를 읽어 `docs/gallery/pub/`로 복구, 이후 갤러리는 이 폴더에서 관리.

## D-040 — 2층 구조 실험: 바깥 = 자유 4DGS(주인공), 안쪽 = SR4D canonical(밑칠) (사용자 지시 2026-10-17)
- 동기(사용자): 지금 SR4D의 자유 가우시안은 LBS·관절에 묶여 있어 머리카락·치마가 출렁이지 못한다. 디테일 층(D-037)은 canonical에 자유 가우시안을 조금 더 붙인 것에 그쳤다. 원래 의도는 이렇다:
  - 겉은 멀티뷰 4DGS처럼 자유롭게 움직인다.
  - 학습 시점에서 먼 새 시점에 생기는 빈 곳은 안쪽의 시간 누적 canonical이 함께 렌더되어 칠한다(디테일하지 않아도 됨).
- 구조:
  - 안쪽층 = 기존 SR4D 체크포인트, **완전 고정**. 매 프레임 `model.deform(t)`로 변형한 가우시안을 미리 계산해 GPU에 캐시한다.
  - 바깥층 = FTGS++식 시간 가우시안. 뼈대·LBS와 무관하다.
  - 두 층을 **한 번의 gsplat 래스터화**로 합성한다. 안쪽층에는 L3 프레임 색 affine을 가우시안 색에 직접 적용한다. 프리멀티플라이 합성과 수학적으로 같다.
  - 4번째 채널(안쪽=1, 바깥=0)을 같이 래스터화해 층별 누적 알파 지도를 얻는다.
- 코드: 바깥층은 `third_party/ftgs_outer/`에 둔다. FTGS++(`4Danyone/4dgs_recon/FreeTimeGSPlusPlus`, 커밋 없음, 2026-10-17 사본)에서 다음을 복사했다:
  - `models/gaussians.py`(시간 가우시안·시간 불투명도·marginal gate);
  - `train/relocation.py`(3d_mcmc 재배치);
  - `train/state.py`;
  - `init/__init__.py:init_dynamic`(키프레임 초기화·KNN 속도);
  - `train/train.py:make_optimizers`(학습률 규칙).
  - 학습 루프는 새 파일 `sr4d/layered/train.py`에 있다. 기존 SR4D 코드는 수정하지 않았다(기존 실행 경로·기본값 그대로).
  - 사용자가 준 경로 `/data/AI-Graphics/Applications/FreeTimeGSPlusPlus`는 존재하지 않아 CLAUDE.md의 경로를 썼다.
- FTGS++에서 그대로 가져온 것:
  - 표현: 위치·등장 시각 μ_t·지속 시간(σ_t = max_duration/6·sigmoid(d), max_duration 1 s)·명시적 속도 v, 위치 x(t) = x + (t − μ_t)·v;
  - 시간 불투명도: o·[g + (1−g)·exp(−½((t−μ_t)/σ_t)²)], marginal gate 켬;
  - 시간 단위: 초(frame/30);
  - 키프레임 간격 10(0,10,…,50,59)과 KNN(k=3) 속도 초기화. 초기 불투명도 0.5, 크기 0.1×이웃 간격, 지속 시간 0.1;
  - 바깥층 가우시안 50만 개(FTGS++ num_gaussians);
  - 3d_mcmc 재배치: 100회마다, 500~25,000회, dead 기준 불투명도 0.005, 점수 = 불투명도 + 2D 기울기 정규화 합;
  - 학습률: means 1.6e-4×장면 크기(→0.01배 지수 감쇠), velocity 1.6e-3×장면 크기(→0.01배), quats 1e-3, scales 5e-3, opacities 0.05, sh_0 2.5e-3, sh_n 1.25e-4, durations 5e-3, times 1.6e-4, gate 1e-3. Adam, batch 1;
  - SH 차수 3, 1,000회마다 1씩 올림; 30,000회;
  - 규제: 불투명도 희소 L1 0.01·mean(o·시간불투명도), 크기 0.01·mean(scale), gate 0.001.
- **FTGS++와 다르게 한 것**:
  1. 초기화 점군: RoMa 삼각측량 대신 **키프레임마다 안쪽층을 그 자세로 옮긴 가우시안 중심 + visual hull 표면 점**(학습 카메라 마스크, 1 cm 복셀, `sr4d.model.build.carve_frame`)을 반씩 쓴다. 색은 안쪽층 색 / visual hull 점은 학습 뷰 투영 색의 중앙값.
  2. 배경: 흰색 합성(img + (1−α)), GT도 마스크로 흰 배경 합성. FTGS++ 흰 기준선 래퍼(D-011)와 같다.
  3. 사진 손실: SR4D 기준선과 같은 0.8·L1 + 0.2·D-SSIM(pytorch_msssim)에 LPIPS VGG 0.05(마스크 bbox, 1,000회부터)를 더한다. FTGS++ 원본은 fused-ssim에 LPIPS alex 0.01, 전체 이미지다.
  4. 카메라별 색 보정(color correctors) 끔. SR4D L4 끔 기준선과 맞춘 것이고, 홀드아웃 렌더에는 어차피 쓰이지 않는다.
  5. 래스터화: gsplat `antialiased` 모드, packed=False, 공분산 직접 입력. 안쪽층이 이 모드와 Mip 3D 필터로 학습됐기 때문이다. FTGS++는 classic이다. 바깥층에는 3D 필터를 쓰지 않는다.
  6. 추가 규제(사용자 지시):
     - 지속 시간 하한: 0.01·mean(relu(σ_min − σ_t)/σ_min). σ_min은 반치폭 3프레임(σ_min = 3/30/2.355 s ≈ 0.042 s).
     - 속도 크기: 0.01·mean(‖v‖) (m/s).
  7. 마스크 손실 없음(사용자 지시대로 사진+LPIPS만). 마스크 밖 알파는 지표로만 본다. 문제가 되면 플래그(`mask_w`)로 켤 수 있게 둔다.
  8. 떨림(가우시안 궤적 가속도)은 바깥층이 정의상 등속 직선 운동이라 0이다. 그래서 바깥층에는 보고하지 않고 안쪽층 값과 tOF로 본다.
- 실험 번호 규칙: EXP-001부터. 실험 하나에 질문 하나. 결과는 `results/EXP-00N/`.
- 판정은 D-026 규칙을 그대로 따른다.

## D-041 (2026-10-17) — 이름 통일 + EXP-004 New-fade (사용자 지시)
- 이름(아티팩트·표·그림 공통, "이름 (설명)"):
  - SR4D: 기존 SMPL-X 골격에 묶인 아바타. dance gen48_ho8은 L4 끔 체크포인트, r15h35는 v0_posea 체크포인트.
  - FTGS-only: FTGS식 가우시안만 사용, 뼈대 없음, 우리 구현. EXP-001, EXP-003 바깥층만.
  - New: 안쪽 SR4D + 바깥 FTGS식 가우시안 함께 렌더. EXP-002, EXP-003 지시대로.
  - New-drop: New + 학습 때 절반은 SR4D 층을 가림. EXP-002b, EXP-003b.
  - New-fade: New-drop + 학습 카메라가 못 본 방향에서 바깥층 투명화. EXP-004.
  - 공식 기준선은 DualGS, FTGS++ (공식 코드, 검정/흰 배경), 4DGS로 따로 표기.
  - 층 기여 지도 색: 바깥층 = 주황, 안쪽층 = 파랑.
- EXP-004 New-fade (`sr4d/layered/fade.py`, `render.py --fade TH1 TH2`, `scripts/run_exp004.sh`):
  - New-drop 모델을 그대로 쓴다. 재학습 없이 렌더 규칙만 더한다.
  - "보였던 방향" 기록:
    - 학습이 끝난 바깥층으로 모든 학습 (카메라, 프레임)을 다시 렌더한다. 바깥층만 렌더하며, 이는 dropout 반복에서 바깥층이 감독받던 조건이다.
    - 가우시안 i, 카메라 c마다 기여량 Σ_픽셀 T·α를 누적한다. 렌더 특징에 대한 기울기로 정확히 얻는다.
    - 기여량으로 가중 평균한 방향(가우시안 → 카메라 중심)을 함께 기록한다.
    - 기여량 합이 1픽셀 이상인 카메라를 "보였던 방향"으로 친다. 이 문턱으로 버려지는 기여량은 전체의 0.6%다(dance ho8 0.62%, r15h35 0.58%).
    - "학습 중"의 궤적을 저장해 둔 것이 없어서, 최종 파라미터로 다시 렌더한 값으로 근사했다.
  - 렌더 규칙:
    - θ는 현재 시점 방향(현재 프레임 위치의 가우시안 → 카메라)과 가장 가까운 "보였던 방향" 사이의 각도다.
    - 불투명도에 1 − smoothstep((θ−θ1)/(θ2−θ1))을 곱한다. 어느 학습 카메라에서도 안 보였던 가우시안은 0이다.
  - (θ1, θ2) = (10°, 25°), (15°, 30°), (20°, 40°)를 dance gen48_ho8과 r15h35 둘 다에서 평가한다.
  - 목표: 가까운 시점 fg 내부 ≥ 24.25(New-drop), 먼 시점 fg 내부 ≥ 18.18(SR4D).
  - 시드 확인은 EXP-004 결과로 설계가 정해진 뒤에 한다.

## D-042 (2026-10-17) — New-fade (10°, 25°) 고정 + EXP-005~007 (사용자 지시)
- New-fade를 주 후보로 정한다. θ = (10°, 25°)로 고정하고, 이후 어떤 데이터·시드에서도 다시 고르지 않는다.
- 떨림 열: SR4D 구조에만 정의된다(가우시안 궤적 가속도 RMS). New-drop·New-fade는 안쪽층(같은 SR4D 체크포인트) 값을 적는다. 바깥층은 등속 직선 운동이라 0이고, 나타났다 사라지는 깜빡임은 tOF로 본다. 기준선은 "–".
- EXP-005 바깥층 관측 뷰 수 제한 (`sr4d/layered/train.py --cov_min_views N`, `third_party/ftgs_outer/outer.py relocate_mcmc(src_ok, rows)`, `scripts/run_exp005.sh`):
  - New-drop 학습(inner_drop 0.5, 30k회) 위에 더해 새로 학습한다.
  - 뷰 수 세기:
    - 매 반복 렌더에 값이 0인 5번째 채널을 더하고, 손실에 그 출력 합(= 0)을 더한다. 손실과 다른 기울기는 바뀌지 않는다(확인: 최대 차 7e-10). 이 채널의 기울기가 그 반복에서 가우시안별 기여량 Σ_픽셀 T·α다.
    - 가우시안 × 학습 카메라별로 기여량과 카메라별 반복 수를 누적한다. 100회마다 0.97배로 줄여서 최근 약 3천 회를 반영한다.
    - 추정 기여량 = 누적 기여량 / 카메라 반복 수 × 프레임 수(전 프레임을 한 번씩 렌더했을 때의 기여량 근사). 1픽셀 이상이면 그 카메라를 "보였다"로 친다(EXP-004와 같은 문턱).
  - 3,000회부터:
    - relocation(100회마다)에서 보인 뷰 수가 N 미만인 가우시안은 복제 원본이 될 수 없다.
    - 1,000회마다 그런 가우시안을 dead로 표시한다. 개수가 고정된 MCMC라서, 같은 relocation에서 잘 보이는 가우시안의 복제로 다시 쓰인다(D-029와 같은 방식).
    - relocation이 끝나는 25,000회 뒤에는 적용하지 않는다.
    - 복제본은 원본의 뷰 통계를 물려받는다.
  - N = 3, 6. 렌더는 New-fade (10°, 25°)를 함께 쓴다. 보였던 방향은 EXP-004와 같이 최종 모델 재렌더로 기록한다. fade 없는 렌더도 같이 채점해서 학습 효과만 따로 본다.
  - 나은 N 고르기(결과 보기 전에 정함): fade 렌더 기준으로 가까운 시점 fg 내부 ≥ 24.2를 만족하는 것 중 먼 시점 fg 내부가 높은 쪽을 고른다.
    - 먼 시점 차이가 0.2 dB 미만이고 fg LPIPS 차이가 0.005 미만이면 동률로 보고 N = 3(약한 개입)을 쓴다.
    - 둘 다 가까운 시점 조건을 못 넘으면 가까운 시점이 높은 쪽을 쓴다.
- EXP-006: 고른 N으로 ZJU 가까운 시점(gen48_ho8, 안쪽층 `sr4d_final/zju_gen48_ho8` = ZJU 기본 SR4D)과 먼 시점(r15h35, 안쪽층 `sr4d_bones/zju_r15h35/v0_posea`)을 돌린다. 참고용으로 New-drop도 같이 학습한다.
- EXP-007: dance에서 바깥층 학습 시드 1, 2를 더한다. 안쪽 SR4D 체크포인트는 시드 42 그대로 고정한다. 고른 N과 New-drop을 둘 다, 가까운 시점과 먼 시점에서 돌린다. 시드 편차를 함께 보고 D-026 규칙으로 판정한다.
- 결과 (2026-10-17 밤):
  - EXP-005: 규칙대로 N = 3을 골랐다. 가까운 시점 24.2를 넘은 것은 N3뿐이었다.
  - EXP-007: 바깥층 시드 3개 기준 먼 시점 N3 + fade 16.99 ± 0.51, New-fade 17.86 ± 0.15였다. 차이가 시드 편차보다 크므로 **N3(관측 뷰 수 제한)를 기각**한다. 코드는 플래그(기본 꺼짐)로 남긴다.
  - New-fade (10°, 25°)를 주 후보로 유지한다.

## D-043 (2026-10-18) — 흰 줄 진단, EXP-008 투명 배경 학습, 안쪽층 L4 끔 재학습 (사용자 지시)
- New-fade (10°, 25°)는 잠정 주 후보다. 이번 흰 줄 문제를 고친 뒤 확정한다. 위쪽 가상 감독은 보류한다.
- 진단 도구:
  - `tools/diag_streak.py`: 흰 줄 픽셀을 고르고 층별로 렌더하며 흰 줄 가우시안의 속성을 잰다.
  - `tools/diag_streak_fig.py`: 위 결과의 비교 그림.
  - `tools/count_bg_streaks.py`: 배경색 줄 가우시안을 센다.
  - `scripts/baselines/ftgspp_dump_state.py`: FTGS++ 결과를 같은 형식으로 덤프한다.
- 배경색 줄 가우시안 정의(EXP-008 전에 고정):
  - 시각 중심에서 학습 뷰 visual hull(마스크 15 px 팽창) 밖에 있다.
  - 최대 불투명도가 0.1 이상이다.
  - 축 비율(가장 긴 축 / 가장 짧은 축)이 10 이상이다.
  - DC 색 휘도가 0.85 초과면 흰색, 0.15 미만이면 검정으로 센다.
- EXP-008 (`train.py --rand_bg --mask_w --max_ratio --vh_prune`, `scripts/run_layered_fade.sh`, `scripts/queue_exp008.sh`):
  - 기반: New-drop(inner_drop 0.5, 시드 0, 30k회) + New-fade (10°, 25°) 렌더. 안쪽층은 기존과 같은 체크포인트(가까운 dance_gen48_ho8_nol4, 먼 v0_posea)를 써서 기존 New-fade와 바로 비교한다.
  - (a) 투명 배경:
    - 매 반복 배경색을 무작위로 뽑는다(균등 RGB). GT = 원본·마스크 + 배경·(1−마스크), 렌더 = 누적 색 + 배경·(1−알파).
    - 알파와 마스크의 L1 손실, 가중치 0.05(SR4D 마스크 손실과 같은 값).
  - (b) (a)에 다음을 더한다:
    - 매 반복 바깥층 축 비율 ≤ 10으로 투영한다(가장 짧은 축을 가장 긴 축 / 10 이상으로 올림).
    - 100회마다 시각 중심이 visual hull 밖(15 px 팽창)인 바깥층 가우시안을 dead로 표시한다. relocation 구간에서는 재배치되고, 25k회 뒤에는 꺼진 채로 남는다.
  - FTGS++ (투명 배경):
    - 공식 파이프라인(검정 마스크 export, 공식 init, 공식 config)에 (a)와 같은 무작위 배경과 알파 L1 0.05를 더한다.
    - 학습 루프만 `third_party/ftgspp_transparent/train.py`로 복사해 고쳤다. 원본은 `FreeTimeGSPlusPlus/ftgspp/train/train.py`(commit 48ebe10).
    - 메모리에서 함수를 바꿔 실행한다(`scripts/baselines/ftgspp_train_transparent.py`, `run_ftgspp_transparent.sh`).
    - 색 보정(color corrector)은 공식대로 켠다. 보정은 렌더 색에 적용한 뒤 배경을 합성한다.
  - 나은 설정 고르기(결과 보기 전에 정함):
    - New-fade 렌더 기준으로 가까운 시점 fg 내부 ≥ 24.2를 만족하는 것 중 먼 시점 fg 내부가 높은 쪽을 고른다.
    - 먼 시점 차이가 0.2 dB 미만이고 LPIPS 차이가 0.005 미만이면 (a)(더 단순한 쪽)를 쓴다.
- 안쪽층 재학습 (`scripts/queue_inner_nol4.sh`, GPU 0·1):
  - 확인: 기존 r15h35 안쪽층(dance·ZJU v0_posea)은 L4 켬이었다. ZJU gen48_ho8 안쪽층(sr4d_final/zju_gen48_ho8)도 L4 켬이었다.
  - 사용자 지시대로 최종 설정(L4 끔, `no_l4_all.yaml`)으로 다시 학습한다:
    - 생성 시드 42: dance r15h35, ZJU r15h35, ZJU gen48_ho8. dance gen48_ho8은 기존 nol4 체크포인트가 있다.
    - 생성 시드 123: dance·ZJU × gen48_ho8·r15h35.
  - 출력은 `outputs/sr4d_final/<k>_<split><sfx>_nol4`이다.
- EXP-009(새 안쪽층 위에서 바깥층 재학습, 시드 표)는 사용자 확인 후 시작한다.
- **EXP-008 결과 (2026-10-18 02:00)**, `results/EXP-008/metrics.csv`, 카드 EXP-008:
  - 가까운 시점 fg 내부 / LPIPS (New-fade 렌더): 기존 24.22 / 0.1869, (a) 24.39 / 0.1930, (b) 24.33 / 0.1939.
  - 먼 시점 fg 내부 / co-vis: 기존 17.85 / 19.99, (a) 18.80 / 20.97, (b) 18.51 / 20.70. SR4D는 18.17 / 20.25.
  - 먼 시점 흰색 줄 가우시안: 기존 2,102개, (a) 1개, (b) 0개.
  - 사전 규칙을 적용했다. 둘 다 가까운 ≥ 24.2를 만족하고, 먼 시점 차이가 0.29 dB ≥ 0.2이므로 **(a) 투명 배경을 채택**한다.
  - 기록만 하고 결정하지 않은 것: 투명 학습 뒤에는 fade 없음 렌더가 더 높다(먼 19.15 / co-vis 21.28, 가까운 24.45). θ는 바꾸지 않았다. fade를 유지할지는 사용자 결정 사항이다.
  - FTGS++ (투명): 가까운 23.99, 먼 16.33, 먼 LPIPS 0.275. 공식 검정판은 23.60 / 15.58 / 0.363이다.

## D-044 (2026-10-18) — fade 제거, New-T 채택, EXP-009 (사용자 결정)
- fade를 제거한다. 새 주 후보는 **New-T**다. New-T = New-drop(inner_drop 0.5) + 투명 배경 학습(`--rand_bg --mask_w 0.05`)이고, fade 없음(renders_both)이다.
- FTGS++의 대표 행은 "FTGS++ (투명 배경)"으로 바꾼다. 공식 설정(검정/흰) 행은 참고용으로 남긴다.
- 정수리 붉은 번짐과 FTGS++ 보라 번짐은 EXP-009 이미지에서 관찰만 하고, 따로 실험하지 않는다.
- 디스크 정리 (사용자 승인 범주 A+B, 목록은 `logs/cleanup_20261018.txt`):
  - 지운 것: 끝난 run의 중간 체크포인트(SR4D `ckpt_s*`/`latest`/`mid` 6.6 GB, 바깥층 `outer_0*` 3.0 GB), 기각된 run(N3·N6, drop 없는 EXP-001~003)의 렌더 4.4 GB.
  - 남긴 것: 최종 체크포인트(`ckpt_final`, `outer_final`, `vis_outer_final`, `ckpt_base`)와 metrics json은 모두 남겼다.
  - 여유 공간은 정리 전 31 GB에서 정리 후 45 GB가 됐다.
- EXP-009 (`scripts/queue_exp009.sh`, `scripts/run_newt.sh`, GPU 0–3 작업 풀):
  - New-T 대상: (dance, ZJU) × (gen48_ho8, r15h35) × 생성 시드 (42, 123). 바깥층 학습 시드는 0이다.
  - 안쪽층은 `outputs/sr4d_final/<k>_<split><sfx>_nol4`(L4 끔)를 쓴다.
  - dance gen48_ho8 시드 42는 기존 exp008a_dance_ho8을 쓴다. 안쪽층(dance_gen48_ho8_nol4)과 학습 설정이 같고, renders_both가 New-T 렌더다.
  - 추가 베이스라인:
    - DualGS gen48_ho8 시드 123 (dance, ZJU);
    - FTGS++ 투명 6개(dance 시드 123, ZJU 전부);
    - 4DGS dynerf 6개(r15h35 전부, gen48_ho8 시드 123).
  - 4DGS dynerf의 r15h35는 처음 돌린다. 기존 먼 시점 4DGS는 다시점 기본 설정이 아닌 run이었다.
  - 정리 방식: 생성 시드 2개의 평균 ± 편차(n = 2, 편차는 |차이|/√2 = 표본 표준편차).
- **EXP-009 결과 (2026-10-18 18:10)**, `results/EXP-009/{metrics.csv,summary.json}`, 카드 EXP-009. fg 내부 PSNR, 생성 시드 2개 평균 ± 편차:
  - dance 가까운: New-T 24.56 ± 0.14, SR4D 23.47 ± 0.34, FTGS++ 투명 24.05 ± 0.09, 4DGS dynerf 23.82, DualGS 23.76.
  - dance 먼: New-T 19.01 ± 0.16, SR4D 18.10 ± 0.07, FTGS++ 투명 16.65 ± 0.44, 4DGS dynerf 15.46, DualGS 12.76. co-vis는 New-T 21.36 = FTGS++ 21.35(비김). 먼 fg LPIPS·전체 PSNR은 FTGS++ 투명이 낫다(0.276 vs 0.325, 19.29 vs 18.07).
  - ZJU 가까운: FTGS++ 투명 26.56 ± 0.01, New-T 26.41 ± 0.21 → 비김(D-026). LPIPS는 New-T 0.103 < 0.119.
  - ZJU 먼: FTGS++ 투명 21.31 ± 0.13 > New-T 20.53 ± 0.20(+0.78), co-vis 21.26 vs 20.56. New-T는 SR4D 20.44와 비김. 전체 PSNR은 New-T 22.86 < SR4D 23.59.
  - tOF는 New-T가 4곳 모두 SR4D보다 낮고, fg LPIPS는 4곳 모두 0.003~0.005 나쁘다.
  - 번짐: New-T 붉은 번짐과 FTGS++ 투명 보라 번짐은 dance 먼 시점 머리에서만 보인다(관찰만).
  - 4DGS dynerf 6개는 렌더 스크립트 오류(`a.orbit`)로 처음에 점수가 비었고, 학습된 모델로 렌더·채점만 다시 했다(`scripts/baselines/score_4dgs_dynerf.sh`).
  - 다음 방향(ZJU 먼 시점 열세 대응 여부 등)은 사용자 결정을 기다린다.
