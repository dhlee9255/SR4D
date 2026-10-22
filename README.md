# SR4D — 인수인계 문서 (2026-10-17 기준)

다시점 인간 영상으로 자유시점 4DGS 시퀀스를 만드는 프로젝트입니다. 영상은 실측 리그 영상이거나 4DAnyone이 생성한 24/48뷰 영상입니다.
- 원래 설계와 작업 규칙: `CLAUDE.md`
- 설계 결정 기록: `docs/DECISIONS.md` (D-001 ~ D-042)
- 날짜별 진행 기록: `docs/PROGRESS.md`
- 이 문서는 그 셋을 처음 보는 사람용으로 요약한 것입니다.

결과는 아티팩트 두 개에서 볼 수 있습니다(비공개, 소유자 계정).
- **SR4D 실험 기록** (2층 구조 EXP-001~007, 최신): https://claude.ai/artifact/PwcwmMgmZwUhmCKgn84C2a
  - 페이지 원본: `docs/gallery/pub2/sr4d_experiments.html`
  - 생성 스크립트: `docs/gallery/exp_page.py`
  - 등록부: `docs/gallery/exp_registry.json`
- 이전 결과 뷰어 (v47, 10/15까지): https://claude.ai/artifact/3AASzMca8m9eJntAvwsH8M

---

## 1. 지금 상태 한눈에

| 항목 | 상태 |
|---|---|
| 데이터 | DNA-Rendering 없음. ZJU-MoCap 387과 4DAnyone 생성 데이터로 개발함(D-003, 사용자 확인 대기 항목). |
| A단계 (실측) | ZJU-387 real12, 시드 3개. SR4D가 interp·extrap 모두 fg 내부 PSNR 1위. |
| B단계 (생성 뷰) | 주 프로토콜은 gen48_ho8. 보간 영역에서 SR4D는 PSNR 최하, LPIPS와 마스크 밖 알파는 1위. 채점은 생성 이미지로만 함. |
| 현재 SR4D 최선 | dance는 **SR4D L4 끔**(`no_l4_all`), ZJU는 기본 설정. |
| 2층 구조 실험 | EXP-001~008 완료. 주 후보 = **New-T** (New-drop + 투명 배경 학습, fade 없음; D-044). New-fade(D-042)와 N3(EXP-005)는 이전 후보·기각. EXP-009(생성 시드 확인) 완료: dance는 New-T 1위, ZJU는 가까운 비김·먼 시점 FTGS++ 투명 우위. |
| 돌고 있는 작업 | 없음. GPU 0~3 유휴(뷰어 제외). |

---

## 2. SR4D 본체 — 무엇을 만들었나

### 2.1 구조 (`sr4d/model/`)
SMPL-X를 비계로 삼은 **하나의 canonical 가우시안**에 모든 프레임·모든 뷰를 모으고, 그 위에 층을 단계적으로 얹습니다.

| 층 | 파일 | 하는 일 |
|---|---|---|
| L0 비계 | `scaffold.py`, `body_model.py` | SMPL-X β(공유)와 프레임별 자세. 작은 학습률로 함께 보정한다. |
| L1 canonical | `canonical.py`, `free_bones.py` | 몸 surfel(평탄 3DGS, D-004)과 자유 가우시안(옷·머리카락). 자유 가우시안은 자유 bone 25개에 묶인다(bone V0, D-023/D-026). |
| L2 자세 비강체 | `pose_nonrigid.py` (CNN판 `pose_nonrigid_cnn.py`, D-035) | 위치·자세 코드 → Δx, Δrot, Δscale, Δcolor. |
| L3 순간 잔차 | `instant_residual.py` | 프레임 latent → 관절·bone SE(3) 잔차와 색 affine. |
| L4 뷰 흡수기 | `view_absorber.py` | 뷰별 색 affine, 카메라 미세보정, 신뢰도. 렌더할 때는 버린다. |
| S7 디테일 층 | `detail.py` | 프레임별 디테일(D-037). 기본은 꺼짐이고 채택하지 않았다. |

- 변형 순서는 `deform.py`에 있습니다: L2 → 자유 bone → LBS → L3.
- 렌더러는 `renderer.py`입니다. gsplat 1.5.3, antialiased 모드, Mip 3D 필터를 씁니다.
- 학습 단계는 S1(자세 보정) → S2(canonical) → S3(+L2) → S4(+L3)이고, `sr4d/train/stages.py`와 `trainer.py`에 있습니다. densify는 집합별 MCMC입니다(D-009).

### 2.2 데이터 (`data_dev/`, 통합 포맷은 CLAUDE.md §5.2)

| 경로 | 내용 |
|---|---|
| `data_dev/zju_mocap/387_cihp*` | ZJU-387 실측. 마스크는 `mask_cihp`(D-022). |
| `data_dev/zju_mocap_gen/387_gen48_2ring[_s123,_s7]` | ZJU 4DAnyone 생성 48뷰(15°·35° 두 링). 생성 시드 42/123/7. |
| `data_dev/fouranyone_gen/dance5885633_gen48_2ring[_s123,_s7]` | dance 생성 48뷰. 생성 시드 42/123/7이 모두 있고, 각각 gen48_ho8·r15h35 split이 있다. |
| `.../dance5885633_gen24`, `387_gen24_genonly` | 원래의 24뷰 생성 데이터. |

split 정의:
- `gen48_ho8`: 학습 40뷰, 홀드아웃 8뷰. 홀드아웃의 최근접 학습 뷰까지 각도는 14.7°입니다. → **"가까운 시점"**
- `r15h35`: 15° 링 24뷰로 학습하고 35° 링 24뷰로 채점합니다. → **"먼 시점"** (D-030에서 부록으로 동결)
- `gen48_all`, `gen24_all`: 모든 뷰로 학습합니다. 시각 비교와 뷰어용입니다.

### 2.3 베이스라인 (`scripts/baselines/`)
- **FTGS++**: `run_ftgspp.sh`. `BG=black|white`로 배경을 고릅니다. 외부 venv를 쓰고 저장소는 읽기 전용입니다. init 버그는 래퍼로 메모리에서 고칩니다(D-012).
- **DualGS**: `run_dualgs.sh` (conda env `dualgs`).
- **4DGS**: `run_4dgs.sh`는 dnerf 설정, `run_4dgs_preset.sh`는 공식 dynerf 설정입니다. 비교 대표는 dynerf입니다(D-038). conda env는 `hust4dgs`이고 코드는 `third_party/4DGaussians`에 있습니다.
- 모든 방법을 SR4D 공통 평가 스크립트 `sr4d/eval/metrics.py`로 채점합니다. 배경은 흰색으로 합성합니다.

### 2.4 평가 지표 (`sr4d/eval/`)
| 지표 | 정의 |
|---|---|
| 전체 PSNR | 이미지 전체 |
| **fg 내부 PSNR (fgi)** | 마스크를 3 px 깎은 안쪽. 주 지표다. |
| fg LPIPS | VGG, 마스크 안 |
| 마스크 밖 알파 | 배경 쪽에 그려진 불투명도. floater를 잡는다. |
| tOF | optical flow 기반 시간 일관성(px) |
| 떨림 | 가우시안 궤적 가속도 RMS(mm). SR4D에만 있다. |
| co-visible | 학습 카메라 2대 이상이 본 픽셀만 본다. r15h35 보조 지표다(`covis_metrics.py`, D-027). |

**판정 규칙 (D-026)**
- 시드 편차 이내의 차이는 동률로 보고, 더 단순한 쪽을 택합니다.
- 0.3 dB 또는 0.005 LPIPS 이상의 효과는 시드 123으로 확인합니다.

---

## 3. SR4D 본체 주요 수치

### 3.1 A단계 — ZJU-387 real12 실측, 학습 시드 3개 (`results/A_real12_seeds.csv`)
| | interp fgi | interp fg LPIPS | extrap fgi | extrap fg LPIPS | tOF (interp) |
|---|---|---|---|---|---|
| **SR4D** | **20.95±0.08** | **0.129** | **19.00±0.08** | 0.169 | **1.11** |
| 4DGS | 19.96±0.39 | 0.158 | 18.14±0.56 | **0.167** | 1.24 |
| FTGS++ | 19.57±0.25 | 0.416 | 17.94±0.11 | 0.442 | 1.31 |
| DualGS | 18.84±0.10 | 0.165 | 14.83±0.06 | 0.201 | 1.34 |

### 3.2 B단계 주 프로토콜 — gen48_ho8, 시드 42 (`results/B_g48_gen48_ho8.csv`)
fgi / fg LPIPS / tOF / 마스크 밖 알파

| | ZJU | dance |
|---|---|---|
| SR4D 기본 | 24.36 / **0.114** / 1.51 / **0.0001** | 22.97 / 0.221 / 2.29 / **0.0024** |
| **SR4D L4 끔** (dance 현재 최선) | – | 23.22 / **0.190** / 2.25 / 0.003 |
| DualGS | **26.18** / 0.123 / **1.31** / 0.0178 | 23.60 / 0.237 / **1.97** / 0.0054 |
| FTGS++ (공식, 검정 배경) | 25.16 / 0.256 / 1.59 / 0.0198 | 23.60 / 0.358 / 2.42 / 0.0506 |
| FTGS++ (공식, 흰 배경) | 24.35 / 0.110 / – / 0.86 | 21.76 / 0.229 / 2.61 / 0.868 |
| 4DGS (dynerf) | 25.26 / 0.152 / – / – | **23.76** / 0.280 / 2.58 / – |

해석: 촘촘한 48뷰 보간에서 SR4D는 PSNR이 가장 낮고, LPIPS·floater·몸 윤곽 IoU는 가장 좋습니다. 윤곽선에는 얇은 흰 테두리가 남습니다(D-031).

### 3.3 그 밖에 시도하고 결론이 난 것 (자세한 내용은 PROGRESS 2026-10-13 ~ 10-15)
| 시도 | 결과 |
|---|---|
| SH 최대 차수 2→3 | 동률이라 2를 유지 |
| 배경 무작위화 | 흰 테두리 개선 없음 |
| L4 카메라 보정 평균 0 (cam0) | ZJU는 +1.43 dB, dance는 −0.90 dB로 방향이 반대 → 확정 불가 |
| L4 confidence 끔 | dance LPIPS −0.029, PSNR −0.59 |
| **L4 전부 끔** | dance에서 PSNR·LPIPS가 함께 개선 → dance 현재 최선. 시드 123 확인은 미완 |
| 2배 학습(L4 끔) | 소폭 개선. 학습량은 병목이 아님 |
| L2 CNN | 개선 없음 → MLP 유지 |
| 머리·목 B-spline 자세 | 효과 없음 |
| S7 디테일 층 (a)–(d) | PSNR은 오르지만 LPIPS·떨림이 나빠짐 → 채택 비권고 |

---

## 4. 최근 작업 — 2층 구조 (D-040 ~ D-042, 2026-10-17)

### 4.1 동기와 구조
사용자가 지적한 문제: SR4D의 자유 가우시안은 결국 관절과 LBS에 묶여 있어서 머리카락과 치마가 자유롭게 출렁이지 못합니다.

그래서 다음 구조를 시험했습니다.
- **바깥층**: FreeTimeGS++식 시간 가우시안. 뼈대 없이 자유롭게 움직이는 주인공입니다.
- **안쪽층**: 학습을 마친 SR4D를 고정해 둔 밑칠입니다. 바깥층이 비는 먼 시점의 빈 곳을 칠하는 역할입니다.
- 두 층은 gsplat 호출 한 번으로 함께 래스터화합니다.

| 이름 | 설명 |
|---|---|
| SR4D | 기존 SMPL-X 골격에 묶인 아바타 |
| FTGS-only | FTGS식 가우시안만 쓰고 뼈대 없이 자유롭게 움직임. 우리 구현 |
| New | 안쪽 SR4D와 바깥 FTGS식 가우시안을 함께 렌더 |
| New-drop | New에 더해, 학습 때 절반은 SR4D 층을 가림(`--inner_drop 0.5`) |
| New-fade | New-drop에 더해, 학습 카메라가 못 본 방향에서 바깥층을 투명하게 함. 렌더 규칙만 바꾸고 재학습 없음 |

### 4.2 코드
| 파일 | 내용 |
|---|---|
| `third_party/ftgs_outer/outer.py` | FTGS++ 시간 가우시안, 옵티마이저, 3d_mcmc 재배치를 복사해 수정한 것. 원본 사본은 `orig/`에 있고, 차이 목록은 D-040에 있다. |
| `sr4d/layered/core.py` | `InnerCache`(SR4D를 프레임별로 미리 변형해 둠), `render2`(두 층 합동 렌더, 4번째 채널 = 안쪽층 비율), `init_outer`(키프레임 SR4D 위치 + visual hull로 초기화) |
| `sr4d/layered/train.py` | 바깥층 학습. 30k회, 사진 손실 + LPIPS + 규제(불투명도, 지속 시간 하한, 속도) |
| `sr4d/layered/render.py` | `--mode both\|outer\|inner`, `--orbit`, `--fade TH1 TH2` |
| `sr4d/layered/fade.py` | EXP-004에서 바깥층 가우시안별로 "보였던 학습 카메라 방향"을 기록하고 감쇠 계수를 계산 |
| `tools/exp_figure.py`, `exp_top.py`, `exp_orbit.py`, `share_frames.py` | 카드 그림, 맨 위 그림, 나란히 궤도 영상, 층 기여 지도(주황 = 바깥, 파랑 = 안쪽) |

### 4.3 결과 — 가까운 시점(gen48_ho8 홀드아웃 8뷰)과 먼 시점(r15h35 35° 링 24뷰), dance 시드 42
| | 가까운 시점 전체 PSNR | 가까운 시점 fgi | 가까운 시점 fg LPIPS | 가까운 시점 tOF | 먼 시점 전체 PSNR | 먼 시점 fgi | 먼 시점 fg LPIPS | 먼 시점 co-vis fgi | 안쪽층 비율 (가까운/먼) |
|---|---|---|---|---|---|---|---|---|---|
| SR4D | 22.85 | 23.22 | 0.190 | 2.25 | 17.24 | **18.18** | 0.342 | 20.25 | 100% |
| FTGS-only | 23.33 | 24.15 | **0.181** | 2.51 | 17.52 | 13.32 | **0.283** | 17.48 | 0% |
| New | 22.87 | 23.28 | 0.189 | 2.25 | 17.24 | 17.99 | 0.345 | 20.09 | 99.5% / 96% |
| New-drop | 22.98 | **24.25** | 0.187 | 2.15 | 16.37 | 15.42 | 0.345 | 17.89 | 80% / 65% |
| New-fade (10°,25°) | 23.04 | 24.22 | 0.187 | **2.14** | 17.22 | 17.85 | 0.342 | 19.99 | 82% / 89% |
| New-fade (15°,30°) | 22.99 | 24.25 | 0.188 | 2.15 | 16.72 | 16.21 | 0.342 | 18.64 | 80% / 74% |
| New-fade (20°,40°) | 22.99 | 24.25 | 0.188 | 2.15 | 16.45 | 15.54 | 0.344 | 18.03 | 80% / 67% |
| New-fade (5°,15°) · 추가 | 22.95 | 23.42 | 0.188 | 2.24 | 17.24 | 18.18 | 0.342 | 20.25 | 97% / 99.9% |
| DualGS | **23.83** | 23.60 | 0.237 | 1.97 | 17.56 | 12.99 | 0.296 | 16.93 | – |
| FTGS++ (공식, 검정 배경) | 16.52 | 23.60 | 0.358 | 2.42 | 15.09 | 15.58 | 0.363 | 20.32 | – |
| 4DGS | 22.83 | 23.76 | 0.280 | 2.58 | **18.09** | 15.23 | 0.348 | **20.46** | – |

결과 파일은 `results/EXP-001/` ~ `results/EXP-008/`에 있습니다.

### 4.4 결론
1. **EXP-001 FTGS-only**: 공식 FTGS++보다 좋습니다. 흰 배경판보다는 모든 항목이 낫고, 검정 배경판보다는 tOF만 약간 나쁩니다. 다만 초기화에 SR4D 정보가 들어가므로 순수한 재현은 아닙니다.
2. **New(지시대로)**: 고정된 SR4D가 이미 학습 뷰를 거의 설명하기 때문에, 불투명도 규제가 바깥층을 지웁니다(생존 0.3%). 결과는 SR4D와 같습니다.
3. **New-drop**:
   - 가까운 시점에서는 지금까지 중 가장 좋습니다(fgi 24.25, tOF 2.15). 바깥층이 맡는 것은 주로 머리카락입니다.
   - 먼 시점에서는 바깥층이 정수리를 틀린 머리카락으로 덮어 SR4D보다 2.76 dB 나쁩니다. 바깥층이 빈 곳(다리)은 SR4D가 채웁니다.
4. **New-fade (EXP-004)**:
   - 두 목표(가까운 ≥ 24.25, 먼 ≥ 18.18)를 함께 넘는 θ는 없었습니다.
   - (10°,25°)가 가장 균형이 좋습니다. 가까운 시점은 New-drop과 비기고, 먼 시점은 SR4D보다 −0.33 dB입니다.
   - (5°,15°)의 먼 시점 결과는 바깥층이 거의 다 꺼져서 SR4D와 같아진 것입니다.
   - 즉 각도로 바깥층을 줄이는 방식은 먼 시점을 SR4D 수준까지 되돌릴 수 있을 뿐, 그 이상은 못 갑니다.

### 4.5 EXP-005~007 (D-042, 2026-10-17 밤)
New-fade를 주 후보로 정하고 θ = (10°, 25°)로 고정했습니다(사용자 결정). 이후 데이터·시드마다 다시 고르지 않습니다.

- **EXP-005 관측 뷰 수 제한** (`train.py --cov_min_views N`): 학습 중에 바깥층 가우시안이 학습 카메라 몇 대에 보였는지 정확히 셉니다. N대 미만에서만 보인 가우시안은 복제하지 않고 1,000회마다 가지치기합니다.
  - N = 3, 6을 시험했고 렌더에는 New-fade를 함께 썼습니다.
  - 결과: 가까운 시점은 같았지만, 먼 시점은 New-fade 17.85보다 낮았습니다(N3 17.52, N6 17.63).
  - 원인: 먼 시점의 정수리 머리카락을 그리는 바깥층 기여 중 92%가, 학습 카메라 24대 중 12대 이상에 보였던 가우시안입니다. 그래서 뷰 수로는 걸러지지 않습니다.
- **EXP-006 ZJU** (시드 0, 가까운 / 먼):

| | 가까운 fg 내부 | 가까운 fg LPIPS | 먼 fg 내부 | 먼 co-vis | 먼 fg LPIPS |
|---|---|---|---|---|---|
| SR4D | 24.36 | 0.114 | 20.51 | 20.47 | 0.139 |
| **New-fade (10°,25°)** | 26.11 | **0.103** | 20.70 | 20.67 | **0.1385** |
| N3 + fade | 26.13 | 0.103 | 20.70 | 20.67 | 0.138 |
| DualGS | **26.18** | 0.123 | 20.24 | 20.41 | 0.169 |
| FTGS++ (공식, 검정) | 25.16 | 0.256 | **21.00** | **20.98** | 0.307 |

- **EXP-007 dance 바깥층 시드 0·1·2** (안쪽 SR4D는 시드 42로 고정, 평균 ± 표준편차):

| | 가까운 fg 내부 | 먼 fg 내부 | 먼 co-vis |
|---|---|---|---|
| SR4D (시드 하나) | 23.22 | **18.17** | **20.25** |
| **New-fade (10°,25°)** | **24.22 ± 0.03** | 17.86 ± 0.15 | 20.03 ± 0.16 |
| N3 + fade | 24.22 ± 0.02 | 16.99 ± 0.51 | 19.46 ± 0.45 |
| New-drop | 24.26 ± 0.03 | 15.50 ± 0.11 | 17.96 ± 0.24 |

  → N3는 기각했습니다. New-fade는 SR4D보다 가까운 시점에서 1.00 dB 높고, 먼 시점에서 0.31 dB 낮습니다.
- 실행: `scripts/run_exp005.sh <gpu> <name> <data> <split> <SR4D ckpt> <N, 0 = New-drop> <seed>`는 학습 → 보였던 방향 기록 → fade 렌더·채점까지 한 번에 합니다. `PLAIN=1`을 주면 fade 없는 렌더도 같이 만듭니다. 묶음 실행은 `scripts/queue_exp006_007.sh <N>`. 정수리 확대 그림은 `tools/crown_crop.py`로 만듭니다.

- **DIAG-1 / EXP-008: 흰 줄 원인과 투명 배경 학습** (D-043, `results/DIAG-streak/`, `results/EXP-008/`):
  - 원인: 고정 흰 배경에 알파 감독 없이 학습해서 배경색과 같은 floater가 생겼습니다. FTGS++ 검정판에서는 같은 원리로 검은 floater가 생깁니다.
  - 처방 (a): 매 반복 배경색을 무작위로 바꾸고, 알파를 마스크로 L1 감독합니다(가중치 0.05). 플래그는 `train.py --rand_bg --mask_w 0.05`입니다. (b)는 여기에 `--max_ratio 10 --vh_prune`을 더합니다.

| dance, 바깥층 시드 0 | 가까운 fg 내부 | 가까운 LPIPS | 먼 fg 내부 | 먼 co-vis | 먼 흰 줄 |
|---|---|---|---|---|---|
| SR4D | 23.22 | 0.190 | 18.17 | 20.25 | — |
| New-fade 기존 | 24.22 | **0.187** | 17.85 | 19.99 | 2,102 |
| **New-fade 투명 (a)** | **24.39** | 0.193 | **18.80** | **20.97** | 1 |
| (b) 투명 + 모양 규제 | 24.33 | 0.194 | 18.51 | 20.70 | 0 |
| FTGS++ (투명) | 23.99 | 0.237 | 16.33 | 21.23 | 0 |

  → 사전 규칙에 따라 (a)를 채택했습니다. 투명 학습 뒤에는 fade 없음이 먼 시점 19.15로 더 높아서, fade를 빼고 **New-T**(fade 없음)를 주 후보로 정했습니다(D-044). 실행은 `scripts/run_newt.sh`, 묶음 실행은 `scripts/queue_exp009.sh`입니다.

### 4.6b EXP-009 (D-044, 2026-10-18) — New-T 생성 시드 확인
fg 내부 PSNR, 생성 시드 42·123 평균 ± 편차 (안쪽층 = L4 끔 SR4D, 바깥층 시드 0). 전체 표는 `results/EXP-009/`.

| | dance 가까운 | dance 먼 | ZJU 가까운 | ZJU 먼 |
|---|---|---|---|---|
| SR4D | 23.47 ± 0.34 | 18.10 ± 0.07 | 25.90 ± 0.33 | 20.44 ± 0.21 |
| **New-T** | **24.56 ± 0.14** | **19.01 ± 0.16** | 26.41 ± 0.21 | 20.53 ± 0.20 |
| DualGS | 23.76 ± 0.22 | 12.76 ± 0.32 | 26.25 ± 0.09 | 18.31 ± 2.72 |
| FTGS++ (투명) | 24.05 ± 0.09 | 16.65 ± 0.44 | **26.56 ± 0.01** | **21.31 ± 0.13** |
| 4DGS dynerf | 23.82 ± 0.08 | 15.46 ± 0.13 | 25.30 ± 0.07 | 17.68 ± 0.10 |

  → dance는 New-T가 두 시점 모두 1위입니다. ZJU는 가까운 시점이 비기고(편차 안), 먼 시점은 FTGS++ 투명이 +0.78 dB 앞섭니다. 먼 시점 fg LPIPS는 FTGS++ 투명이 일관되게 낫습니다(dance 0.276 vs 0.325).

### 4.6 사용자 결정 대기
- EXP-009 이후 방향. ZJU 먼 시점에서 FTGS++ 투명에 지고, 먼 시점 LPIPS가 안쪽층 수준에 머뭅니다.
- 예전부터 남은 것: dance L4 끔 시드 123 확인, L2 출력 스케일 실험, 저장소(NAS/SSD) 경로 확정, DNA-Rendering 확보.

---

## 5. 실행 방법 (대략)

### 환경
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate sr4d     # torch 2.4.1+cu118, gsplat 1.5.3
export PATH=/usr/local/cuda-11.8/bin:$PATH CUDA_HOME=/usr/local/cuda-11.8 PYTHONPATH=.
python scripts/check_env.py
```
- 베이스라인 환경: FTGS++는 `4Danyone/4dgs_recon/FreeTimeGSPlusPlus/.venv`, DualGS는 `dualgs`, 4DGS는 `hust4dgs`입니다.
- 기존 conda 환경과 외부 저장소는 수정하지 않습니다.

### SR4D 학습·렌더·채점
```bash
# 일반형: config + ablation들 + key=value 덮어쓰기 → <out>/ckpt_final.pt, renders/, metrics_renders.json
scripts/run_sr4d.sh configs/dance_r15h35.yaml outputs/sr4d_final/dance_gen48_ho8 \
    configs/ablations/bones_v0_pose_a.yaml -- data.setting=gen48_ho8
# dance 현재 최선 (L4 끔) = 위 명령에 configs/ablations/no_l4_all.yaml 추가
# 주 프로토콜 묶음 (SR4D/4DGS/FTGS++/DualGS): scripts/run_g48.sh <zju|dance> <gen48_ho8|gen48_all|gen24_all> <sr4d|4dgs|ftgspp|dualgs> <gpu>
```
- 학습 시간은 56k회에 약 2.5시간, GPU 메모리는 약 17 GB입니다. **GPU 하나에 SR4D 하나만** 올립니다.
- 채점만 다시 할 때:
  ```bash
  python -m sr4d.eval.metrics --data_root <data> --setting <split> --renders <dir> --method <name> --phase B --no_csv
  ```
  r15h35에는 `sr4d.eval.covis_metrics`를 추가로 돌립니다.

### 2층 구조
```bash
# 바깥층 학습 (tmux 안에서): run_layered.sh <gpu> <name> <data> <split> <SR4D ckpt> [train.py 인자]
scripts/run_layered.sh 1 exp002b_dance_ho8_both_drop05 data_dev/fouranyone_gen/dance5885633_gen48_2ring gen48_ho8 \
    outputs/sr4d_final/dance_gen48_ho8_nol4/ckpt_final.pt --seed 42 --inner_drop 0.5
#   FTGS-only = --no_render_inner, New = 인자 없음, New-drop = --inner_drop 0.5   (30k회 약 20–30분, 7~8 GB)
scripts/eval_layered.sh <gpu> <name> both          # 홀드아웃 렌더 + 채점
scripts/render_layered_figs.sh <gpu> <name> both "g03 g15" outputs/orbits/exp003/orbit_e35   # 학습 뷰·궤도 렌더
# New-fade: 보였던 방향을 기록한 뒤 3가지 θ로 렌더·채점 (재학습 없음, run당 수 분)
scripts/run_exp004.sh <gpu> <name> [covis]
python -m sr4d.layered.render --run outputs/layered/<name> --mode both --fade 10 25 --out <dir>
```

### 그림·아티팩트
```bash
python tools/exp_figure.py ...        # 카드 그림 (사용 예: scripts/rebuild_exp_figs.sh)
python tools/exp_top.py ...           # 맨 위 그림
python docs/gallery/exp_page.py       # exp_registry.json → sr4d_experiments.html
```
그다음 Claude Code에서 Artifact publish를 실행합니다. 페이지는 `docs/gallery/pub2/sr4d_experiments.html`이고, 이미지 파일은 `exp/...`를 `files`로 함께 올립니다.

### 뷰어
```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. python tools/viewer4d.py --host 0.0.0.0 --port 8080   # 사용법: docs/VIEWER.md
```

---

## 6. 운영 규칙과 주의사항
- 학습은 tmux(세션 `sr4djobs`) 안에서 돌리고 로그는 `logs/`에 남깁니다. 큐를 걸 때마다 `scripts/queue_watch.sh <로그 멈춤 s> <GPU 유휴 s> pid:log ...`를 같이 겁니다(CLAUDE.md §12). 4DGS 교착으로 GPU가 15시간 논 적이 있습니다.
- 돌고 있는 bash 스크립트는 수정하지 않습니다. bash가 파일을 오프셋으로 읽기 때문에, 고치려면 새 파일로 복사합니다.
- 다음 일은 사용자에게 먼저 묻습니다: 데이터 삭제·이동, 대용량 다운로드, 돌고 있는 작업 중지, 승인되지 않은 설계 변경.
- B단계는 생성 이미지로만 채점·검증합니다. 실측 카메라는 쓰지 않습니다(D-024).
- 디스크 여유가 적습니다(약 79 GB). 일부 베이스라인 폴더(예: `outputs/4dgs/*/data/ims`)는 공간 때문에 학습 이미지를 지웠습니다. 다시 렌더하려면 `scripts/fix_exp003_4dgs.sh` 방식으로 임시로 다시 내보내야 합니다.
- 저장소 경로(`configs/paths.yaml`의 work_data/outputs)는 아직 임시이며 사용자 확인이 필요합니다.
