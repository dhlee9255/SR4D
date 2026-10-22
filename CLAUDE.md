# SR4D — Claude Code 작업 지시서

> 이 파일은 `/data/AI-Graphics/Applications/SR4D/CLAUDE.md`에 둔다. Claude Code는 이 문서를 기준으로 환경 구성 → 데이터 준비 → 베이스라인 재현 → SR4D 구현 → 실험까지 순서대로 진행한다.
> 사용자가 "구현해줘" / "다음 단계 진행해줘"라고만 해도 아래 **10. 마일스톤**의 다음 미완료 단계부터 이어서 한다.

---

## 0. 한 줄 요약

**SR4D (Scaffold–Residual 4D)** 는 다시점 인간 영상(실측 리그 또는 4DAnyone 생성 24/48뷰)을 입력으로, SMPL-X를 비계(scaffold)로 삼은 **하나의 canonical 가우시안**에 모든 프레임 × 모든 뷰의 관측을 모은 뒤, 옷·머리카락 자유층 → 자세 비강체 → 순간 잔차를 **단계적으로** 얹고, 생성 뷰 불일치는 렌더 때 버리는 **뷰 흡수기**가 맡는 4DGS 파이프라인이다. 출력은 입력 영상을 자유시점으로 재생하는 4DGS 시퀀스다.

**1차 목표 (A단계)**: DNA-Rendering 실측 다시점에서 **FreeTimeGS++ / 4DGS / DualGS를 이긴다** (특히 24·16뷰의 궤도 밖 시점).
**최종 목표 (B단계)**: 같은 시퀀스를 4DAnyone으로 생성한 24/48뷰 입력에서도 이긴다.

---

## 1. 절대 규칙

1. **외부 저장소는 읽기 전용이다.** 아래 경로의 코드는 수정하지 않는다. 필요하면 SR4D 쪽에 래퍼 스크립트를 만들거나, 수정이 꼭 필요하면 `SR4D/third_party/<name>`으로 복사한 뒤 복사본만 고치고 `docs/DECISIONS.md`에 이유를 남긴다.
   - `/data/AI-Graphics/Applications/4Danyone`
   - `/data/AI-Graphics/Applications/DualGS`
   - `/data/AI-Graphics/Applications/4Danyone/4dgs_recon/FreeTimeGSPlusPlus` (이 프로젝트에서 "FTGS"는 이 FreeTimeGS++를 뜻한다)
2. **기존 conda 환경을 건드리지 않는다.** SR4D는 새 환경(`sr4d`)을 만든다. 베이스라인은 각자 기존 환경으로 돌린다.
3. **사용자 확인이 필요한 일**: 대용량 다운로드(수십 GB 이상), 데이터 삭제·이동, 저장소(NAS/SSD) 경로 결정, 라이선스 동의가 필요한 파일(SMPL-X 모델, DNA-Rendering) 확보. 이런 일은 멈추고 물어본다.
4. **추측으로 포맷을 가정하지 않는다.** 외부 코드의 데이터 포맷·실행 방법은 README와 코드를 직접 읽어 확인하고, 확인한 내용을 `docs/ENV_REPORT.md`에 적은 뒤 변환기를 만든다.
5. **모든 비교는 같은 조건**: 같은 시퀀스, 같은 프레임 구간, 같은 학습/평가 카메라, 같은 해상도, 같은 마스크·배경 처리.
6. 진행 상황은 `docs/PROGRESS.md`, 설계 변경은 `docs/DECISIONS.md`, 실험 결과는 `results/*.csv`에 계속 기록한다.

---

## 2. 경로와 저장소

```
/data/AI-Graphics/Applications/
├── SR4D/                     # 이 프로젝트 (CLAUDE.md 위치)
├── 4Danyone/                 # 4DAnyone (생성기 + GVHMR 등 전처리)
│   └── 4dgs_recon/FreeTimeGSPlusPlus/   # 베이스라인: FTGS(++)
└── DualGS/                   # 베이스라인: DualGS
```

- **NAS / SSD**: 별도 저장소가 있다. 경로는 아직 미정이다. `df -h`, `lsblk`, `mount`로 후보를 찾아 용량·속도와 함께 `docs/ENV_REPORT.md`에 정리하고, **사용자에게 확인받은 뒤** `configs/paths.yaml`에 기록한다. 권장 배치:
  - 원본 데이터(DNA-Rendering `.smc`): NAS
  - 변환된 작업 데이터, 체크포인트, 렌더 결과: SSD
- 4DGS(Wu et al.)는 로컬에 없다. `third_party/4DGaussians`로 공식 저장소(`hustvl/4DGaussians`)를 clone한다.

`configs/paths.yaml` 예시 (값은 확인 후 채운다):

```yaml
root: /data/AI-Graphics/Applications/SR4D
repos:
  fouranyone: /data/AI-Graphics/Applications/4Danyone
  ftgspp: /data/AI-Graphics/Applications/4Danyone/4dgs_recon/FreeTimeGSPlusPlus
  dualgs: /data/AI-Graphics/Applications/DualGS
  fourdgs: /data/AI-Graphics/Applications/SR4D/third_party/4DGaussians
storage:
  raw_data: <NAS 경로 — 사용자 확인>
  work_data: <SSD 경로 — 사용자 확인>
  outputs: <SSD 경로 — 사용자 확인>
assets:
  smplx_models: <SMPL-X 모델 폴더 — 4Danyone 안에 있으면 재사용>
```

---

## 3. 첫 작업: 환경 탐색 리포트 (M0)

코드를 쓰기 전에 아래를 조사해 `docs/ENV_REPORT.md`로 정리한다.

1. **하드웨어·드라이버**: `nvidia-smi`(GPU 종류·수·메모리), CUDA 드라이버/툴킷 버전, 디스크 여유 공간, NAS/SSD 마운트 후보.
2. **conda 환경 목록**: `conda env list`. 각 외부 저장소가 어떤 환경을 쓰는지 찾는다.
3. **FreeTimeGS++** (`4Danyone/4dgs_recon/FreeTimeGSPlusPlus`):
   - 입력 데이터 레이아웃(이미지·카메라 파일 형식. EasyVolcap식 `images/<cam>/<frame>.jpg` + `intri.yml`/`extri.yml`인지 확인), 마스크 사용 여부, 초기화 방식(space carving 등), 학습·렌더·평가 명령, 주요 하이퍼파라미터.
4. **DualGS** (`DualGS`):
   - README상 HiFi4G/DualGS 데이터셋 기준이고 카메라는 `colmap/sparse/0` (Gaussian Splatting 형식)을 쓴다고 되어 있다. 프레임별 폴더 구조, 첫 프레임 초기화 방식, joint/skin 단계 명령, 평가 스크립트 유무를 확인한다.
5. **4DAnyone** (`4Danyone`):
   - 추론 진입점, **타깃 카메라를 지정하는 방식**(파일 형식·좌표계), 출력 비디오 저장 위치·해상도·프레임 수.
   - **GVHMR로 추정한 SMPL-X 시퀀스가 어디에 어떤 형식으로 저장되는지**, 스켈레톤 렌더링에 쓰는 카메라·좌표계가 무엇인지. (논문 부록: GVHMR로 지면 정렬 SMPL-X 시퀀스를 추정 → 70개 Goliath 키포인트 회귀 → 타깃 뷰마다 깊이 버퍼 스켈레톤 렌더.)
   - SMPL-X 모델 파일이 저장소 안에 있는지(있으면 SR4D가 재사용).
6. **4DGS**: clone 후 요구 환경과 다시점 데이터 입력 형식 확인.

리포트 끝에 "각 베이스라인에 넣을 데이터 형식"과 "SR4D 통합 포맷 → 각 형식 변환 계획"을 표로 적는다.

---

## 4. 환경 구성 (`sr4d` conda 환경)

- Python 3.10 또는 3.11, M0에서 확인한 CUDA에 맞는 PyTorch.
- 필수 패키지: `gsplat`(래스터라이저, MCMC densification), `smplx`, `lpips`, `torchmetrics` 또는 `pytorch-msssim`, `opencv-python`, `imageio[ffmpeg]`, `h5py`(DNA-Rendering SMC 읽기), `trimesh`, `pyyaml`/`omegaconf`, `tensorboard` 또는 `wandb`, `tqdm`, `rich`.
- 선택: `tinycudann`(해시 인코딩, 설치가 어려우면 순수 PyTorch 해시 그리드로 대체), RAFT 계열 optical flow(시간 지표용).
- **2DGS 래스터화**: 설치한 gsplat 버전에 2DGS 래스터화 함수가 있는지 확인한다. 있으면 몸 surfel에 쓰고, 없으면 3DGS + 평탄화(최소 축 크기 고정) + 법선 손실로 대체하고 `DECISIONS.md`에 기록한다.
- 설치 후 `scripts/check_env.py`로 GPU 인식, gsplat 렌더 1회, SMPL-X forward 1회, LPIPS 1회를 확인한다.
- 재현용으로 `environment.yml`과 `requirements.txt`를 남긴다.

---

## 5. 데이터

### 5.1 DNA-Rendering (현재 없음)

- 먼저 NAS/SSD에 이미 받아둔 사본이 있는지 찾는다: `find <후보 경로> -name "*_annots.smc" 2>/dev/null | head`.
- 없으면 **멈추고 사용자에게 요청**한다. 공식 다운로드는 신청 후 링크를 받는 방식이고, 데이터는 `{actorID}_{performanceID}.smc`(본 데이터)와 `{actorID}_{performanceID}_annots.smc`(주석) H5 파일로 제공된다. 주석에는 카메라 보정, 색 보정, 마스크, 키포인트, SMPL-X가 들어 있다.
- 읽기 도구는 공식 저장소(`github.com/DNA-Rendering/DNA-Rendering`)의 처리 코드를 참고해 `sr4d/data/dna_rendering.py`에 구현한다. 2K(48대)와 4K(12대) 카메라 그룹의 구분·번호 체계는 SMC 파일 구조를 직접 확인해 정한다.
- **우선 확보할 시퀀스** (4DAnyone 논문의 DNA-Rendering 테스트 세트, 4DAnyone 학습에서 제외됨):
  `0012_09, 0019_06, 0025_11, 0034_04, 0094_02, 0124_03, 0152_01, 0165_08, 0188_02, 0219_07`
- **헐렁한 옷·소품 세트** (DressRecon이 고른 시퀀스, 2차): `0008_01, 0047_01, 0047_12, 0102_02, 0113_06, 0121_02, 0123_02, 0128_04, 0133_07, 0152_01, 0166_04, 0188_02, 0206_04, 0239_01`
- 개발 초기에는 **1개 시퀀스, 60프레임, 1/2 해상도**로 시작한다.

### 5.2 SR4D 통합 데이터 포맷

모든 데이터(실측, 4DAnyone 생성)를 하나의 포맷으로 바꾸고, 베이스라인 입력은 이 포맷에서 내보낸다.

```
<work_data>/<dataset>/<seq>/
├── images/<cam_id>/<frame:06d>.jpg
├── masks/<cam_id>/<frame:06d>.png        # 전경 마스크 (0/255)
├── cameras.json                          # cam_id → K(3x3), R(3x3), t(3), W, H, (dist), source: "real"|"generated"
├── smplx/<frame:06d>.npz                 # betas, global_orient, body_pose, jaw/eye, hand poses, transl, expression (전역 좌표)
├── splits/<setting>.json                 # train_cams, test_interp_cams, test_extrap_cams, frames
└── meta.json                             # 원본 경로, 해상도, fps, 색 보정 적용 여부, 변환 버전
```

- 실측 데이터는 SMC의 색 보정을 적용해 저장하고 적용 여부를 `meta.json`에 기록한다.
- 변환기: `tools/convert_dna_rendering.py`, `tools/convert_4danyone.py`
- 내보내기: `tools/export_ftgspp.py`, `tools/export_dualgs.py`, `tools/export_4dgs.py`
- 변환 검증 `tools/verify_dataset.py`: SMPL-X 메시를 각 카메라에 투영해 마스크와의 IoU를 계산하고, 대표 프레임 오버레이 이미지를 저장한다. IoU가 낮으면 카메라·좌표계 변환 오류로 보고 멈춘다.

### 5.3 카메라 split

`tools/make_splits.py`가 `cameras.json`에서 카메라의 방위각·고도를 계산해 split을 만든다.

| setting | 학습 카메라 | 평가 |
| --- | --- | --- |
| `real48` | 2K 48대 중 홀드아웃 제외 전부 | 홀드아웃 |
| `real24` | 48대에서 방위각이 고르게 퍼진 24대 | 홀드아웃 |
| `real16` | 48대에서 고르게 퍼진 16대 | 홀드아웃 |
| `gen24` / `gen48` (B단계) | 4DAnyone 생성 뷰 (실측 카메라와 같은 위치) | 실측 홀드아웃 |

- **평가 카메라는 두 묶음**으로 나눈다.
  - `test_interp`: 학습 카메라 사이에 끼어 있는 카메라 (궤도 위 보간)
  - `test_extrap`: 학습 카메라와 고도가 다르거나, 연속된 방위각 구간을 통째로 뺀 카메라 (궤도 밖)
- 모든 setting에서 평가 카메라 집합은 동일하게 고정한다(학습 카메라 수만 바꾼다).
- 각 평가 카메라의 **최근접 학습 카메라 각도**를 split 파일에 같이 저장한다.
- 4DAnyone 입력용 소스 카메라는 **22번**(4DAnyone 논문과 동일)으로 하고, 학습·평가에서 제외하거나 별도 표기한다.

### 5.4 4DAnyone 생성 데이터 (B단계에서만)

1. 같은 DNA-Rendering 시퀀스에서 22번 카메라 영상을 4DAnyone 입력으로 쓴다.
2. 타깃 카메라를 `real24`/`real48`의 학습 카메라와 **같은 위치**로 지정해 생성한다(그래야 나머지 실측 카메라가 GT가 된다). 타깃 카메라 지정 방식은 M0 조사 결과를 따른다.
3. **SMPL-X는 4DAnyone 전처리에서 나온 GVHMR 결과를 재사용**한다. 생성 뷰가 그 스켈레톤을 따르므로 좌표·자세가 처음부터 맞는다. DNA-Rendering 좌표계로 옮길 때 필요한 정합(회전·이동·스케일)을 구해 `cameras.json`과 같은 좌표계로 저장한다.
4. 주의: HMR 오류는 생성 뷰도 물려받으므로, 생성 뷰로 SMPL-X를 재정합해도 자세 오류는 고쳐지지 않는다. B단계 S1에서는 체형(β)과 작은 오차만 보정한다.

---

## 6. 베이스라인 (A단계부터 동일 조건으로)

| 이름 | 위치 | 비고 |
| --- | --- | --- |
| FTGS (FreeTimeGS++) | `4Danyone/4dgs_recon/FreeTimeGSPlusPlus` | 4DAnyone 기본 복원기. 4DAnyone 설정을 기본값으로 쓴다 |
| DualGS | `DualGS` | 첫 프레임 초기화 → joint/skin 추적. 81캠 전제 설계라 희소 뷰에서의 동작을 같이 기록 |
| 4DGS (Wu et al., CVPR 2024) | `third_party/4DGaussians` (clone) | canonical + HexPlane 변형장 |

- 실행 래퍼: `scripts/baselines/run_{ftgspp,dualgs,4dgs}.sh <seq> <setting>` → 통합 포맷에서 export → 학습 → 평가 카메라 렌더 → `outputs/<method>/<seq>/<setting>/renders/<cam>/<frame>.png`.
- 평가는 **SR4D의 공통 평가 스크립트**로 한다(각 저장소 자체 평가 스크립트 숫자와 섞지 않는다).
- 배경 처리: 주 프로토콜은 **마스크로 전경만 남기고 고정 배경색(흰색)** 으로 모든 방법을 학습·평가한다. 전체 이미지 평가는 선택 사항으로 둔다.
- 학습 시간, 가우시안 수, 프레임당 용량도 기록한다.

---

## 7. SR4D 구현 명세

### 7.1 디렉터리 구조

```
SR4D/
├── CLAUDE.md
├── configs/
│   ├── paths.yaml
│   ├── base.yaml                 # 공통 기본값
│   ├── dna_real.yaml             # A단계
│   ├── gen_4danyone.yaml         # B단계
│   └── ablations/{s2_only,no_l2,no_l3,no_l4,joint_training,no_scaffold}.yaml
├── sr4d/
│   ├── data/      dataset.py, dna_rendering.py, fouranyone.py, samplers.py
│   ├── model/     scaffold.py (L0), canonical.py (L1), free_bones.py, pose_nonrigid.py (L2),
│   │              instant_residual.py (L3), view_absorber.py (L4), deform.py, renderer.py
│   ├── train/     stages.py, losses.py, densify.py, trainer.py
│   └── eval/      metrics.py, render_cams.py, temporal.py, nearest_view.py, report.py
├── tools/         convert_*.py, export_*.py, make_splits.py, verify_dataset.py
├── scripts/       check_env.py, run_sr4d.sh, run_ablation.sh, baselines/
├── third_party/   4DGaussians/ (clone), 기타 복사본
├── docs/          ENV_REPORT.md, PROGRESS.md, DECISIONS.md
└── results/       *.csv, figures/
```

### 7.2 표현: 다섯 층

핵심 원칙: **공유 범위가 넓은 층이 먼저, 최대한 많이 설명한다.** 아래 층일수록 공유 범위가 좁고 용량이 작다.

**L0 — SMPL-X 비계 (`scaffold.py`)**
- 파라미터: 시퀀스 공유 `betas`, 프레임별 `global_orient, body_pose, hand_pose, jaw/expression, transl`.
- 제공: canonical(T-pose 또는 A-pose) 메시, 꼭짓점 skinning weight, 프레임별 관절 변환 `G_t`.
- 렌더하지 않는다. 학습 중 자세는 작은 학습률로 공동 최적화(S1 이후 기본 1e-5, config로 조절).

**L1 — canonical 가우시안 (`canonical.py`)**
- (a) **몸 surfel**: SMPL-X 메시 표면에서 면적 비례 샘플링(초기 약 10만 개). 각 가우시안은 (삼각형 id, 무게중심 좌표)를 가진다. 초기에는 평평하게(최소 축 크기 ε) 하고 법선을 삼각형 법선에 맞춘다. 속성: 색(SH 차수 0~1로 시작, 최대 2), opacity, 2축 크기, 면내 회전.
- (b) **자유 가우시안** (옷·머리카락·소품): 초기화 절차
  1. 프레임마다 학습 카메라 마스크로 space carving → 관측 공간 점군.
  2. 각 점을 가장 가까운 SMPL-X 꼭짓점의 skinning weight로 역 LBS → canonical 점.
  3. canonical SMPL-X 표면과의 거리가 `d_free`(기본 2cm)보다 큰 점만 남김.
  4. 전 프레임을 합쳐 복셀(기본 5mm) 다운샘플, 최대 15만 개.
- 개수 상한: 합계 30만~50만(config). densification은 gsplat MCMC 전략(또는 상한이 있는 기본 전략).

**자유 bone (`free_bones.py`)**
- 자유 가우시안 canonical 점을 k-means로 K개(기본 25) 군집 → 각 군집이 하나의 Gaussian bone(중심, 방향, 크기).
- 각 bone은 가장 가까운 SMPL-X 관절을 부모로 가진다. 프레임별로 부모 관절 변환에 대한 **상대 SE(3)** 를 학습한다(초기값 항등).
- 자유 가우시안의 bone skinning weight: canonical bone까지의 마하라노비스 거리 softmax(음수) + 선택적으로 작은 MLP 보정.

**L2 — 자세 비강체 (`pose_nonrigid.py`)**
- 입력: canonical 위치 x (해시 인코딩), 자세 코드 p_t (body_pose 63차원 → 2층 MLP → 32차원).
- 출력: Δx(3), Δrotation(4, quaternion 잔차), Δscale(3), Δcolor(3).
- 전 시간 공유(자세의 함수). 몸 surfel과 자유 가우시안 모두에 적용.

**L3 — 순간 잔차 (`instant_residual.py`)**
- 프레임별 latent z_t ∈ R^24 (초기값 0) + 공유 디코더.
- 출력: (i) 자유 bone별·SMPL-X 관절별 작은 SE(3) 잔차, (ii) 프레임별 색 affine(3×3 + 3).
- 선택: FreeTimeGS식 **일시 가우시안**(시간 중심 μ_t, 폭 σ_t의 가우시안 시간 opacity). 기본 꺼짐, 켜면 전체 개수의 5% 이하로 제한.
- 렌더 모드: `replay`(켬) / `animate`(끔).

**L4 — 뷰 흡수기 (`view_absorber.py`)**
- 뷰별 외형 코드 a_v ∈ R^16 → 작은 MLP → 색 affine(3×3 + 3). 선택: (뷰, 10프레임 구간)별 코드.
- 생성 뷰 전용: 뷰별 카메라 미세보정 6DoF(크기 제한).
- 픽셀 신뢰도 w: 초기값은 뷰 간 재투영 일관성으로 계산, 학습 중에는 뷰별 저해상도(32×32) logit 맵을 업샘플해 미세 조정.
- **렌더 시 버린다**(평균 코드 사용).
- A단계(실측)에서는 카메라별 색 affine만 켜고 카메라 보정·신뢰도는 끈다. 실측에서 L4 자체의 효과도 ablation으로 본다.

### 7.3 변형 순서 (`deform.py`)

canonical 가우시안 X → 프레임 t의 관측 공간:

1. `X ← X + Δ_L2(X, p_t)` (위치·회전·크기·색)
2. 자유 가우시안만: `X ← Σ_k w_k(X) · B_{k,t} · X` (자유 bone 변환, 부모 관절 기준 상대 변환)
3. 전체: `X ← LBS(X; G_t)` (SMPL-X, 몸 surfel은 삼각형 기준 위치·법선을 같이 변환)
4. `X ← X ⊕ Δ_L3(z_t)` (`replay` 모드에서만)

공분산(회전·크기)도 같은 변환으로 함께 옮긴다.

### 7.4 렌더러 (`renderer.py`)

- gsplat 기반. 몸 surfel은 2DGS 래스터화(가능한 경우), 자유 가우시안은 3DGS. 두 집합을 한 번에 합성할 수 없으면 둘 다 3DGS로 렌더하고 몸 surfel에 평탄화·법선 정규화를 건다(결정을 `DECISIONS.md`에 기록).
- 출력: RGB, alpha, depth, normal(가능한 경우).
- Mip 3D smoothing filter 옵션(기본 켬): 학습 카메라 샘플링률 기반 최소 크기.

### 7.5 학습 단계 (`stages.py`)

| 단계 | 켜는 층 | 하는 일 | 반복 (개발 기본값) |
| --- | --- | --- | --- |
| S0 | – | 데이터·마스크·SMPL-X·split 로드, 초기 신뢰도 계산 | – |
| S1 | L0 | 실측: 데이터셋 SMPL-X를 다시점 실루엣·키포인트로 소폭 보정. 4DAnyone: β와 작은 오차만 | 2k |
| S2 | L1 + 자유 bone (+ L4 색 affine) | 처음 3k 반복은 몸 surfel을 SMPL-X 표면에 고정(무게중심 좌표만 학습), 이후 떼어내고 정렬 손실로 묶음 | 24k |
| S3 | + L2 | 자세 비강체 학습 | 12k |
| S4 | + L3 | 순간 잔차 학습, 잔차 dropout p=0.3 | 18k |
| S5 (선택) | L2 ← L3 | 자세가 비슷한 프레임들의 평균 L3 잔차를 L2로 증류 후 L3 리셋 | 주기적 |
| S6 (선택) | 전체 | 궤도 밖 가상 렌더 → fixer → 학습 카메라가 못 보는 픽셀에만 손실 | 6k |

- 각 단계 시작 시 체크포인트 저장. `--resume_stage`로 단계 단위 재개.
- 프레임 샘플링: 매 반복 (프레임, 카메라) 쌍을 무작위로 뽑되, 한 배치에 여러 프레임이 섞이게 한다.

### 7.6 손실 (`losses.py`) — 초기 가중치, 튜닝 대상

| 묶음 | 항 | 초기 가중치 | 단계 |
| --- | --- | --- | --- |
| 사진 | (1−λ)·L1 + λ·D-SSIM, λ=0.2, 픽셀 신뢰도 w로 가중 | 1.0 | S2~ |
| 사진 | 실측 소스 뷰 가중 배수 (B단계에서 선택 사용) | ×3 | S2~ |
| 사진 | LPIPS (VGG) | 0.05 (S2 중반부터) | S2~ |
| 사진 | 마스크(alpha vs mask) | 0.05 | S2~ |
| 표면 | 법선 일관성(렌더 법선 vs depth 법선) | 0.05 | S2~ |
| 표면 | depth distortion (2DGS) | 2DGS 기본값에서 시작 | S2~ |
| 표면 | 이방성 상한 max/min 크기 비 > r(=4) 페널티 | 0.01 | S2~ |
| 표면 | 몸 surfel–메시 정렬 (위치 + 법선, SAGA식) | 1.0 / 0.1 | S2 후반~ |
| 표면 | opacity 이진화(entropy) | 0.001 | S2~ |
| 우선순위 | 자유 bone 변형 크기 ‖B_t − I‖ (DressRecon의 옷 변형 최소화 항과 같은 역할) | 0.1 | S2~ |
| 우선순위 | L2 출력 크기 + 공간 평활 | 0.01 | S3~ |
| 우선순위 | L3 출력 크기 L2 + 인접 프레임 평활 | 0.01 / 0.01 | S4~ |
| 우선순위 | L4 코드 → 0, 카메라 보정 크기 제한 | 1e-3 | S2~ |
| 신뢰도 | w·residual − log w (불확실성 손실, 생성 뷰만) | 0.01 | S2~ (B단계) |

**학습 중 모니터링 (필수)**: 2k 반복마다 검증 카메라에서
- `gap_L3 = PSNR(replay) − PSNR(L3 off)` — 크면 잔차가 canonical을 잠식한 것 → L3 정규화 강화
- `PSNR(L4 off)` — 흡수기를 버렸을 때의 품질
- 몸 surfel / 자유 가우시안별 개수·opacity 합 — 자유층이 몸을 먹는지 확인

### 7.7 config 예시 (`configs/dna_real.yaml`)

```yaml
data:
  dataset: dna_rendering
  seq: "0012_09"
  setting: real24
  frames: [0, 60]
  downscale: 2
  background: white
model:
  body_surfels: {n_init: 100000, sh_degree: 1}
  free: {d_free_m: 0.02, voxel_m: 0.005, max_init: 150000, n_bones: 25}
  cap_total: 400000
  l2: {enabled: true, pose_dim: 32}
  l3: {enabled: true, z_dim: 24, transient: false, transient_budget: 0.05, dropout: 0.3}
  l4: {enabled: true, code_dim: 16, camera_refine: false, confidence: false}
train:
  stages: {s1: 2000, s2: 24000, s3: 12000, s4: 18000, s5: false, s6: false}
  body_adhere_iters: 3000
  densify: mcmc
eval:
  splits: [test_interp, test_extrap]
  every: 2000
```

---

## 8. 평가 (`sr4d/eval/`)

- **화질**: PSNR / SSIM / LPIPS(VGG), 전체 이미지와 전경(마스크) 각각. `test_interp`와 `test_extrap`을 **따로** 보고한다.
- **최근접 학습 뷰 각도별 곡선**: 평가 카메라마다 (각도, PSNR) 점을 찍고 방법별로 비교.
- **시간 일관성**: 전경 영역에서 `mean | (R_t − R_{t−1}) − (G_t − G_{t−1}) |` (R=렌더, G=GT). 가능하면 optical flow 기반 warping 오차도 추가.
- **SR4D 진단**: `gap_L3`, `PSNR(L4 off)`.
- **(B단계)** 생성 뷰 간 일관성: 4DAnyone 논문식 12→4 홀드아웃 PSNR, 가능하면 MEt3R.
- **비용**: 학습 시간, 피크 GPU 메모리, 가우시안 수, 프레임당 용량.
- `sr4d/eval/report.py`가 `results/<phase>_<setting>.csv`와 비교 표(markdown)를 만든다.

결과 CSV 열: `method, seq, setting, split, psnr, ssim, lpips, fg_psnr, fg_ssim, fg_lpips, temporal, train_min, n_gauss, notes`

---

## 9. 실험 계획

**A단계 — 실측 다시점 (주 실험, 파이프라인 개발)**
1. 1개 시퀀스(`0012_09`), 60프레임, 1/2 해상도, `real24`에서 FTGS++ / 4DGS / DualGS 재현.
2. SR4D S2만으로 FTGS++ 대비 `test_extrap` 우위를 확인한다(**첫 번째 판단 지점**).
3. L2, L3를 차례로 켜고 `test_interp`·시간 지표 개선을 확인한다.
4. `real48 / real24 / real16` 전체 비교 → 10개 테스트 시퀀스로 확대 → 헐렁한 옷 세트.
5. **승리 조건**: `real48`에서 동등 이상, `real24`·`real16`의 `test_extrap`에서 PSNR·LPIPS 확실한 우위, 시간 지표 개선.
6. 확장 검증(시간 여유가 있을 때): PKU-DyMVHumans(60캠), DualGS·TaoGS 81캠 데이터셋, VVC26 희소 뷰 트랙(6캠). 모두 데이터 확보는 사용자 확인 후.

**B단계 — 4DAnyone 생성 입력 (최종 목적)**
7. A단계 시퀀스를 22번 카메라에서 `gen24`/`gen48`로 생성, 같은 실측 카메라로 평가.
8. A단계 실측 결과와의 차이 = 생성 불일치의 몫. 여기서 L4(카메라 보정·신뢰도 포함)와 신뢰도 가중의 효과를 측정한다.

**Ablation (A·B 공통)**
- (a) S2만 (b) +L2 (c) +L3 (d) +L4 (e) 단계 학습 vs 모든 층 동시 학습 (f) SMPL-X 비계 없이 자유 bone만(비교 대상들이 SMPL-X를 쓰지 않으므로 공정성 확인용).

---

## 10. 마일스톤과 완료 기준

| ID | 내용 | 완료 기준 |
| --- | --- | --- |
| M0 | 환경 탐색 리포트 | `docs/ENV_REPORT.md` 작성, 저장소 경로 사용자 확인 완료 |
| M1 | `sr4d` 환경 | `scripts/check_env.py` 통과, `environment.yml` 저장 |
| M2 | DNA-Rendering 확보·변환 | 1개 시퀀스 통합 포맷 변환, `verify_dataset.py` 마스크 IoU 확인, split 생성 |
| M3 | 베이스라인 재현 | FTGS++/4DGS/DualGS가 `real24`에서 렌더·평가 완료, CSV 기록 |
| M4 | SR4D S1–S2 | canonical만으로 학습·렌더, FTGS++와 `test_extrap` 비교 |
| M5 | SR4D 전체(L2, L3, L4) | ablation (a)–(f) 결과 CSV |
| M6 | A단계 확대 | 10개 시퀀스 × real48/24/16 결과 표 |
| M7 | B단계 | 4DAnyone 생성 데이터 변환, gen24/gen48 결과 표 |

각 마일스톤이 끝나면 `docs/PROGRESS.md`에 결과 요약(숫자 포함)과 다음 할 일을 적고, 사용자에게 보고한 뒤 다음으로 넘어간다.

---

## 11. 막혔을 때

- 외부 저장소 실행이 안 되면: 오류 로그, 시도한 것, 추정 원인을 `PROGRESS.md`에 적고 사용자에게 묻는다. 외부 코드를 몰래 고치지 않는다.
- 좌표계가 안 맞으면(마스크 IoU 낮음): 카메라 규약(world→cam vs cam→world, OpenCV vs OpenGL), 단위(mm/m), 색 채널(BGR/RGB)부터 확인한다.
- 메모리가 부족하면: 해상도 1/4, 프레임 수 축소, 가우시안 상한 축소 순으로 줄이고 기록한다.
- 결과가 기대와 다르면: 숫자를 그대로 보고한다. 설계를 바꾸기 전에 사용자와 상의한다.

---

## 부록: 참고 자료

- 4DAnyone — arXiv 2608.20335 (생성기, GVHMR SMPL-X, FreeTimeGS 복원, DNA-Rendering 평가 프로토콜)
- FreeTimeGS — CVPR 2025, arXiv 2506.05348
- DualGS — SIGGRAPH Asia 2024 (TOG), arXiv 2409.08353, 코드 `HiFi-Human/DualGS`
- 4DGS (Wu et al.) — CVPR 2024, 코드 `hustvl/4DGaussians`
- DNA-Rendering — ICCV 2023, `dna-rendering.github.io`, 코드 `DNA-Rendering/DNA-Rendering`
- SAGA (몸 surfel 붙였다 떼기, 정렬 손실) — arXiv 2412.00845
- DressRecon (몸/옷 2층 bone, 옷 변형 크기 정규화) — 3DV 2025, arXiv 2409.20563
- AniGS (생성 뷰 불일치를 4DGS로 흡수) — CVPR 2025
- Mind the Gap (보간 vs 외삽 평가 격차) — arXiv 2607.01556
- VVC 2026 희소 뷰 트랙 3위 보고서 (커버리지 마스크 가상 감독, FTGS++ 기반) — `github.com/Doreen1113/SIGGRAPH-Asia-2026-VVC-SparseView`

---

## 12. 운영 규칙 (학습 큐 감시) — 2026-10-14 추가

- GPU 작업을 큐에 걸면 **완료 대기와 함께 감시를 반드시 같이 건다** (`scripts/queue_watch.sh <로그 멈춤 한도 s> <GPU 유휴 한도 s> pidfile:logfile ...`):
  1. **로그 멈춤 알림**: 살아 있는 작업의 로그가 정해진 시간(기본 30분) 동안 바뀌지 않으면 즉시 알리고 원인을 확인한다.
  2. **GPU 큐 비움 알림**: 어떤 GPU에 (뷰어 외) 학습 프로세스가 없는 상태가 정해진 시간(기본 10분) 이어지면 즉시 알린다.
- 감시는 세션이 다시 시작되면 함께 멈추므로, 세션 시작 시 큐 상태를 확인하고 감시를 다시 건다.
- 실행 중인 스크립트는 수정하지 않는다(새 파일로 복사). GPU 하나에 SR4D 학습은 하나만 올린다.
