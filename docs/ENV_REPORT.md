# ENV_REPORT (M0) — 2026-10-08

Everything below was confirmed by reading code or running commands. Nothing is guessed.

## 1. Hardware / driver

| Item | Value |
|---|---|
| GPU | 4 × RTX 4090 24GB (driver 580.65.06, driver CUDA 13.0) |
| CUDA toolkit | `/usr/local/cuda-11.8` (nvcc 11.8.89) |
| RAM / CPU | 188 GB / 20 cores |
| OS | Ubuntu 22.04, kernel 6.8 |

## 2. Storage (needs user confirmation — CLAUDE.md rule 3)

| Mount | Kind | Size | Free | Notes |
|---|---|---|---|---|
| `/` (nvme0n1p2, WD SN850X) | local NVMe | 3.6T | **~17G (100%)** | holds /data, /home, conda envs. **Almost no room left** |
| `/mnt/nas` (192.168.0.3:/volume2) | NFS4, Synology | 21T | 2.1T | candidate for **raw data** |
| `/mnt/ssd_nas` (192.168.0.4:/volume2/Workspace2) | NFS3 | 7.3T | **17G (100%)** | supposed to be the SSD NAS, but full |
| `/mnt/Dataset2` (sda1) | local HDD exFAT | 16.4T | **0** | full; holds TaoGS/HiFi4G data and DualGS results |
| `sdb1` | local HDD 16.4T | – | ? | **not mounted** |
| `sdc1` ("M.2 NVME", ROTA=1 → probably a USB enclosure) | 3.6T | – | ? | **not mounted**. Best candidate for SSD work data once mounted |

→ No SSD has room for work data right now. **Question for the user**: (a) can `sdc1` be mounted and used as the SSD work drive? (b) raw data: is `/mnt/nas/Dataset/...` OK?
Until then, development uses a small set (ZJU-MoCap 1 seq × 60 frames × 1/2 resolution, a few hundred MB) under `SR4D/data_dev/`, which is inside the project.

## 3. Conda environments (relevant ones)

| Env | Use | Check |
|---|---|---|
| `sr4d` (**new**) | SR4D | torch 2.4.1+cu118, gsplat 1.5.3 (JIT build with nvcc 11.8), smplx 0.1.28, lpips. `scripts/check_env.py` passes |
| FTGS++ `.venv` (uv) | `4Danyone/4dgs_recon/FreeTimeGSPlusPlus/.venv` | torch 2.6.0+cu124, gsplat 1.5.3 |
| `dualgs`, `dualgs_4danyone` | DualGS | torch 2.1.2, diff_gaussian_rasterization / simple_knn / fused_ssim import OK |
| `hust4dgs` | 4DGaussians (HUST) | torch 2.1.2+cu118, diff_gaussian_rasterization / simple_knn OK |
| `4danyone` | 4DAnyone inference, GVHMR | – |

## 4. FreeTimeGS++ (`4Danyone/4dgs_recon/FreeTimeGSPlusPlus`)

- Entry point: `./run <dataset> <configs_dir> <out> --scenes <s>`, or per-module `uv run -m ftgspp.{data.extract, data.sfm, data.prep, init.points, init, train, eval, render}` (`scripts/run_dynerf`).
- Config: TOML (`msgspec`, `extends` supported). `ftgspp/utils/config.py`: `[model] [data] [init] [init.roma] [train] [train.relocation] [train.lrs] [train.lr_schedules]`.
- **Input layout (dynerf dataset type)**: `video_path/` holds either `*.mp4`, or `images/<cam_idx>/<frame>.png` (`extract.py:symlink_image_dataset`) + `calibration_path=poses_bounds.npy` (LLFF, one row per camera, static cameras). **It is not EasyVolcap `intri/extri.yml`.**
  The 4DAnyone project already wrote an LLFF converter (`4dgs_recon/scripts/build_ftgspp_dynerf.py`). It reconstructs `SIMPLE_PINHOLE` intrinsics, so it **requires fx≈fy and a principal point at the center** (only one f is stored).
- Camera selection: `data.train_cameras`, `data.eval_cameras` (list or interval). Frames: `data.frames={start,stop}`, `fps`, `scale` (resolution scale).
- **Masks: not used.** Rasterization calls `gsplat.rasterization(...)` with no `backgrounds` argument, so the **background is black**.
- Initialization: `ftgspp.data.sfm` (builds COLMAP from known poses) → `ftgspp.init.points` (RoMa matching + triangulation, `keyframe_stride=10`, `num_points_per_frame=700k`) → `ftgspp.init` (500k Gaussians, velocity from KNN). Space carving is not used.
- Default hyperparameters (4DAnyone `ftgspp` preset): 30k iter, LPIPS loss on, relocation (3d_mcmc, every 100 iter, start 500 / stop 25k), `color_correction=true` (start 3000), `max_duration=1`, `marginal_gating=true`, SH 3.
- Known pitfalls (4Danyone/CLAUDE.md): `render.py --camera N` uses camera 0's FOV, and time is `t=frame/fps`.
- → SR4D side: **we write our own render wrapper** (import ftgspp, load the checkpoint, render the evaluation cameras with their own K and the white background). The external code is not modified.

## 5. DualGS (`DualGS`)

- Input: `<root>/<frame>/<view>.png` (masked, white background) + `<root>/colmap/sparse/0` (`cameras.bin/txt`, `images.bin/txt`, PINHOLE / SIMPLE_PINHOLE only, so images must be undistorted) + `points3d.ply` (if missing, a random 100k cube is used).
  `scene/dataset_readers.py:readColmapSceneInfo`: the frame folder name `str(load_frame_id)` is used as `images`. The image name inside COLMAP must equal the view filename.
- Training: `python train.py -s <root> -m <out> --frame_st 0 --frame_ed N --iterations 30000 --subseq_iters 15000 --training_mode 0 --parallel_load -r 2`. The first frame is JointGS → SkinGS; later frames are warped from frame 0 (default) or the previous frame (`--seq`).
- Held-out views: `_parse_eval_views(eval_views)` (an `eval_views` option that the local copy added).
- Render / eval: `render.py -m <model> -st -e --views`, `scripts/evaluation.py -g -r`. The local `HANDOFF.md` mentions TaoGS jobs, but no GPU process is running right now.

## 6. 4DAnyone (`4Danyone/4DAnyone`)

- Inference: `python inference.py --video_path ... --views_per_layer N --layer_pitches '[..]' --start_yaw --yaw_span` (env `4danyone`).
- **Target cameras**: set **only by ring parameters** (yaw/pitch/layer). `fdanyone/geometry/cameras.py:camera_ring/camera_grid` decides the radius and focal automatically (`framing.method=sequence_input_profile_static_radius_target_focal`). **There is no CLI to pass arbitrary camera poses directly.** Making views at the same positions as DNA-Rendering's real cameras (stage B, CLAUDE.md 5.4) needs either a wrapper that calls the internal API (`ViewPlan`/camera list injection) or a "ring approximation + use the actual output cameras as GT" approach. Handle this in stage B.
- Output: `data/fdanyone/<clip>/{cameras.json, metadata.json, skeletons/NN.mp4, videos/dense/NN.mp4}`, 121 frames, 704×1280.
  `cameras.json`: OpenCV camera, `camera_to_world` 4×4, `K` (fx≈fy, cx≈W/2), world=`canonical_human_world` (y-up, the subject faces +z, origin=initial root projected onto the floor, meters).
- **GVHMR result**: `data/gvhmr/results/<clip>/motion.safetensors` — `smpl_params_global.{betas(121,10), body_pose(121,63), global_orient, transl}`, `smpl_params_incam.*`, `K_fullimg`, `observed_keypoints_2d(121,17,3)`. `motion.json`: fps, `motion_world=gvhmr_gravity_aligned_y_up`. Body-only (21 joints) SMPL-X parameters.
- SMPL-X models: `4Danyone/models/body_models/smplx/SMPLX_NEUTRAL.npz`; `4Danyone/models_smplx_v1_1.zip` has NEUTRAL/MALE/FEMALE. → copied to `SR4D/assets/body_models/smplx/` (npz, 3 files, ~320MB).

## 7. 4DGS (Wu et al.)

- `third_party/4DGaussians` (hustvl/4DGaussians @ 843d5ac, clone). The existing `hust4dgs` env is used.
- Multi-view input: the N3V (DyNeRF) format `poses_bounds.npy` + `cam00/images/*.png` folders, or the `dnerf` (Blender `transforms_train/test.json`, with time) format. Dynamic human multi-view → we plan to use the **dnerf format** (per-frame time, any K) (export_4dgs.py).

## 8. Data

- **DNA-Rendering**: no `.smc` found locally or on the NAS (`find` on the candidate paths, depth limit 5, timed out after 120 s; `_annots.smc` not found in the scanned range). Official download needs an application (link by email) plus license agreement → **user action needed.**
  The format was checked from the official code (`third_party/DNA-Rendering/scripts/3DGS/SMCReader.py`, `dataset_readers.py`):
  - `Camera_5mp` = 48 cams with ids '0'..'47' (2K, 2448×2048), `Camera_12mp` = ids '48'..'59' (4K)
  - annots `Camera_Parameter/<id:02d>/{K, D(k1,k2,p1,p2,k3), RT(=**c2w**, OpenCV), Color_Calibration}`
  - masks `Mask/<id>/mask/<frame>`, color `Camera_5mp/<id>/color/<frame>` (BGR JPEG bytes)
  - `SMPLx/{betas, expression, fullpose(N,55,3), transl, scale}`, gender-specific model (`actor_info.gender`), `flat_hand_mean=False`
  - The official code undistorts with `cv2.undistort(img,K,D)` and keeps K.
- **Substitute dev data (available locally)**: ZJU-MoCap refine `my_377/387/393` (`/data/AI-Graphics/Applications/3DGS_lee/GauHuman/data/zju_mocap_refine/`): 23 real cameras (single ring), `intri.yml/extri.yml` (OpenCV w2c, meters), `mask/` ({0,1}), SMPL params (`Rh, Th, poses(72), shapes`). Also TaoGS / HiFi4G on `/mnt/Dataset2/HiFiHuman`.
  → While DNA-Rendering is missing, **we build and validate the pipeline on ZJU-MoCap 387** (SMPL body model; SR4D's scaffold supports both SMPL and SMPL-X).

## 9. Input formats per baseline and conversion plan

| Target | Input format | Masks / background | Conversion script |
|---|---|---|---|
| SR4D | unified format (CLAUDE.md 5.2) | uses masks, white background | `tools/convert_{dna_rendering,zju_mocap,4danyone}.py` |
| FTGS++ | `images/<idx>/<frame>.png` + LLFF `poses_bounds.npy` + TOML | composite onto white with the mask; FTGS's internal background is black, so we train/render with **our own wrapper that passes a white background** | `tools/export_ftgspp.py`, `scripts/baselines/ftgspp_wrapper.py` |
| DualGS | `<frame>/<view>.png` (white composite, undistorted) + `colmap/sparse/0/*.txt` + `points3d.ply` (SMPL-X vertices or visual hull) | white | `tools/export_dualgs.py` |
| 4DGS | dnerf format `transforms_{train,test}.json` (+time) + `points3D` init | white (`--white_background`) | `tools/export_4dgs.py` |

Shared preprocessing: undistort + resize to the same resolution with K scaled to match, so every method gets the same pixels. FTGS++ / DualGS need PINHOLE (cx,cy may be off-center → DualGS PINHOLE works; for FTGS's SIMPLE_PINHOLE, the export **crops/pads to a centered principal point** — must be checked).
