# 디지털 초파리 연구 지형도

FlyGym 하나만 보면 이 분야의 절반도 안 보입니다. 지금 같은 초파리를 대상으로
**서로 다른 세 그룹이 각각 다른 목적의 모델**을 만들고 있고, 그 옆에
**전뇌 커넥톰(connectome)** 데이터가 완성되면서 "뇌 배선도 → 몸 → 행동"을
끝까지 연결하려는 시도가 폭증하는 중입니다.

이 문서는 2026년 9월 기준으로, 이 프로젝트에서 실제로 연구를 시작할 수 있도록
생태계 전체를 정리한 것입니다. 저장소 정보는 GitHub API로, 논문 내용은 원문
PDF로 직접 확인했습니다. 실행해 보지 않고 문서만 읽은 항목은 그렇게 표시했습니다.

---

## 1. 디지털 초파리 모델 세 가지

| | **NeuroMechFly v2** (FlyGym) | **flybody** | **FlyMimic** (근골격) |
|---|---|---|---|
| 개발 | EPFL Ramdya Lab | Google DeepMind + HHMI Janelia | EPFL Ramdya + Ijspeert |
| 저장소 | [NeLy-EPFL/flygym](https://github.com/NeLy-EPFL/flygym) (337★) | [TuragaLab/flybody](https://github.com/TuragaLab/flybody) (876★) | flygym 2.1.0에 통합됨 |
| 논문 | Nature Methods 2024 [s41592-024-02497-y](https://www.nature.com/articles/s41592-024-02497-y) | Nature 643:1312 (2025) [s41586-025-09029-4](https://www.nature.com/articles/s41586-025-09029-4) | [arXiv:2509.06426](https://arxiv.org/abs/2509.06426) |
| 구동 방식 | 관절 위치 제어 + 발 부착(adhesion) | 관절 + 날개 + 유체력 + 부착 | **Hill-type 근육** |
| 비행 | ✗ | **✓** (타원체 유체 모델) | ✗ |
| 보행 | ✓ (평지/틈/블록/혼합 지형) | ✓ (모방학습) | ✓ (모방학습) |
| 시각 | ✓ 겹눈 721 옴마티디아 × 2 | ✓ (vision-guided flight) | ✓ (근사) |
| 후각 | 버전에 따라 다름 (§2) | ✗ | ✗ |
| RL 스택 | Gymnasium + Stable-Baselines3 | Acme + TF + Ray (DMPO) | Gymnasium + SB3 |
| 강점 | **감각-운동 폐루프, 지형, 접근성** | **비행, 대규모 분산 RL** | **근육 수준 생체역학** |

세 모델은 경쟁 관계가 아니라 **해상도 계층**입니다. flybody는 날 수 있고,
NeuroMechFly는 보고 냄새 맡으며, FlyMimic은 근육을 씁니다. 그리고 FlyGym 2.1.0은
**세 모델을 모두 로드할 수 있습니다.** 이 PC에서 직접 확인한 결과입니다.

```
NeuroMechFly        nq=73  nv=72  nu=48   ngeom=70   (관절 위치 제어 42 + 부착 6)
FlyBody             body segments = 67              (DeepMind 모델이 flygym으로 로드됨)
MusculoskeletalFly  nq=14  nv=14  nu=15   → MUSCLE 액추에이터 15개
```

```python
from flygym.compose import FlyBody, MusculoskeletalFly, build_musculoskeletal_simulation
sim, fly = build_musculoskeletal_simulation(name="msk")   # Hill-type 근육 15개
```

> 주의 1: FlyMimic은 **다리 하나**의 근골격 모델입니다(14 DoF). 전신 근육 모델이
> 아닙니다. 논문 제목이 "limb movement biomechanics"인 이유입니다.
>
> 주의 2: 이 모델들을 처음 쓸 때 메시 애셋을 **S3에서 일회성 다운로드**합니다
> (여기서는 140 MB + 14 MB, 약 10분 소요, `~/.cache/flygym_assets`에 캐시).
> 첫 실행이 멈춘 것처럼 보여도 정상입니다.

---

## 2. ⚠️ FlyGym 버전 갈래 — 연구 시작 전에 반드시 확인할 것

같은 이름의 `flygym`이 **세 갈래**로 존재하고, 기능이 서로 다릅니다.
여기 설치된 PyPI 2.1.0에는 **후각이 없습니다.** 설치본 전체를 `grep`한 결과
`odor`/`olfact` 관련 API가 하나도 없음을 직접 확인했습니다.

| | **A. PyPI `flygym` 2.1.0** (설치됨) | **B. cobar-2026** (EPFL 수업) | **C. PyPI `flygym-gymnasium` 1.3.2** (설치됨) |
|---|---|---|---|
| 설치 | `pip install flygym` | [NeLy-EPFL/cobar-2026](https://github.com/NeLy-EPFL/cobar-2026) 클론 + `uv sync` | `pip install flygym-gymnasium` |
| Python | ≥3.12, <3.15 | ≥3.12, <3.13 | ≥3.10, **<3.13** |
| 내부 버전 | 2.1.0 | 2.0.0-rc.0 (벤더링) | 1.3.2 |
| MuJoCo | 3.9 | 3.5 | 구버전 |
| 시각 | ✓ | ✓ | ✓ |
| **후각** | **✗** | **✓** `OdorMixin` / `sim.get_olfaction()` | **✓** |
| **바람** | ✗ | **✓** `sim.set_wind(magnitude, angle_deg)` | ✗ |
| 더듬이 기계감각 | ✗ | ✓ `sim.get_antenna_data()` | ✓ |
| 경로적분 / 머리 안정화 | ✗ (예제 없음) | ✓ (week6) | ✓ (예제 모듈) |
| 커넥톰 기반 시각 처리 | ✗ | ✗ | ✓ `advanced_vision.ipynb` |
| 냄새 기둥(plume) 추적 | ✗ | ✗ | ✓ `advanced_olfaction.ipynb` |
| GPU (MJWarp) | ✓ | ✓ | ✗ |
| 근골격 / flybody 모델 | ✓ | ✗ | ✗ |
| 브라우저 WASM | ✓ | ✗ | ✗ |

**실무 지침**

- 지형 보행, 실시간 뷰어, GPU, 근육 모델 → **A** (이미 설치됨)
- 바람, 포식자 회피 → **B**를 별도 폴더에 클론
- 커넥톰 시각 회로, 냄새 기둥, 경로적분 재현 → **C**를 별도 venv에 (설치돼 있음)
- 세 개를 한 venv에 넣지 마세요. 전부 `flygym`이라는 이름을 씁니다.

> **후각은 이 프로젝트에서 A로 이식했습니다.** [flyplay/odor.py](flyplay/odor.py)가
> 1.x와 **같은 센서 배치**(하악수염 2 + 더듬이 2, `config.yaml`의 좌표 그대로)와
> **같은 거리 역제곱 모델**을 2.x API 위에 구현합니다. 따라서 위 표에서 A의 후각
> 항목은 이 프로젝트 한정으로 ✓입니다. 검증은 `scripts/10_sense_check.py` 7항목.

이건 제가 임의로 나눈 게 아니라 현재 커뮤니티의 실제 상태입니다. FlyGym
[Discussions](https://github.com/NeLy-EPFL/flygym/discussions)의 가장 최근 글
(2026년 4월)이 *"1.x.x 튜토리얼이 2.0.0으로 이식될 예정인가요?"*이고 아직
답이 달리지 않았습니다. 2.x는 성능과 모델 통합을 먼저 가져가면서 감각 관련
예제를 아직 되돌려놓지 못한 상태입니다.

### cobar-2026의 후각 API (직접 확인한 시그니처)

```python
from flygym.compose.world import FlatGroundWorld, OdorMixin

class WorldWithOdor(OdorMixin, FlatGroundWorld): pass

world = WorldWithOdor()
world.add_odor_source(pos=[24, 0, 1.5], peak_intensity=[1, 0])   # 유인
world.add_odor_source(pos=[8, -4, 1.5], peak_intensity=[0, 1])   # 기피

intensities = sim.get_olfaction(fly.name)   # (4 센서, n_냄새차원)
```

센서 4개는 좌/우 더듬이(antenna)와 좌/우 하악수염(maxillary palp)이고
(`src/flygym/assets/model/olfaction.yaml`), 냄새는 **거리 역제곱**으로 감쇠합니다.
냄새 "차원"은 물리적 발생원 수와 별개여서, 발생원 3개를 2차원(유인/기피)
공간에 배치할 수 있습니다.

---

## 3. NeuroMechFly v2 논문이 실제로 한 실험 (재현 대상)

논문 PDF에서 직접 추출한 그림 목록입니다. 각각이 곧 연구 주제입니다.

| 그림 | 내용 | 재현 코드 |
|---|---|---|
| Fig 2 | 3차원 보행: **수직 벽 오르기**(부착력 40 mN), 미끄러지는 임계 경사각, 험지 보행 | `nmf2-paper/leg_adhesion`, `complex_terrain` |
| Fig 3 | 시각·후각 폐루프 제어 (색 기둥 추적, 좌우 냄새 차이 기반 주행성) | `nmf2-paper/visual_inputs`, `odor_inputs` |
| Fig 4 | **상행 운동 피드백**으로 경로적분 + 머리 안정화 | `flygym-gymnasium` 예제 |
| Fig 5 | **RL 계층 제어기**로 다중감각 과제 (냄새 접근 + 시각 장애물 회피 + 험지) | `nmf2-paper/integrated_task/preprint_trial/` |
| Fig 6 | 복잡한 냄새 기둥 추적(PhiFlow), **커넥톰 제약 시각망**으로 초파리-초파리 추종 | `flygym-gymnasium/notebooks/advanced_*.ipynb` |
| ED 1–7 | 생체역학 개선, 실측 스텝 운동학, 시각 캘리브레이션, 경로적분 궤적, 머리 안정화 효능, 육각 합성곱 시각 모델, LC9/LC10 상류 뉴런 추종 성능 | — |

### 논문의 RL 설정 (PDF에서 확인)

- 알고리즘: **Soft Actor-Critic (SAC)**, MLP 정책
- 정책 출력: **하행 선회 명령(descending turning commands)** — 관절각이 아님
- 하위 제어: CPG + 감각 피드백 + 발 부착의 하이브리드 제어기
- 시각 전처리: **육각 합성곱 CNN**이 원시 옴마티디아에서 물체 방향·거리·시야
  내 여부·양안 방위각·망막상 크기를 추출
- 후각 전처리: 좌우 강도 비대칭 $\Delta I_o = (I_L - I_R)/\overline{I}$

> **이 프로젝트와의 관계.** [flyplay/env.py](flyplay/env.py)에 구현한 구조는
> 논문과 **같은 계층 구조**입니다 — 정책이 하행 신호 2차원을 내고 하이브리드
> CPG가 다리를 움직입니다. 차이는 두 가지뿐입니다: (1) SAC 대신 PPO,
> (2) 시각·후각 대신 목표 방위를 직접 관측으로 줌. 즉 **관측을
> `sim.get_ommatidia_readouts()` / `sim.get_olfaction()`으로 바꾸면 논문
> Figure 5를 그대로 재현하는 경로**가 됩니다.

동영상 14편: [NeLy-EPFL/nmf2-videos](https://github.com/NeLy-EPFL/nmf2-videos)

---

## 4. EPFL 수업 BIOENG-456 (CoBAR) — 가장 좋은 학습 경로

[NeLy-EPFL/cobar-2026](https://github.com/NeLy-EPFL/cobar-2026)은 FlyGym 저자들이
직접 만든 **정답이 포함된** 실습 커리큘럼입니다. 튜토리얼보다 체계적입니다.

| 주차 | 노트북 | 내용 |
|---|---|---|
| 1 | `kinematic_replay.ipynb` | 실측 관절 운동학 재생 |
| 2 | `cpg_controller.ipynb` + `cpg_network.py` | CPG 진동자망 |
| 3 | `turning.ipynb` | 하행 신호 기반 선회 |
| 4 | `vision.ipynb`, `olfaction.ipynb` | 겹눈, 냄새 주행성 |
| 5 | `neural_network.ipynb` + `collect_data.py` | 데이터 수집 → 신경망 학습 |
| 6 | `path_integration.ipynb` | 고유수용 신호로 경로 적분 |

각 주차마다 `exercises.ipynb`(문제)와 `solutions.ipynb`(정답)가 쌍으로 있습니다.

**미니프로젝트** (`miniproject/`)는 5단계 레벨이 있는 완성된 과제입니다.

- `arena/banana.py` — 유인 냄새원(바나나 조각)
- `arena/dragonfly.py` — **포식자 잠자리** (위치·자세·색 제어 가능)
- `arena/grass.py`, `sky.py`, `terrain.py` — 환경
- `notebooks/odor.ipynb`, `notebooks/wind.ipynb` — 냄새와 **바람** 단서
- `run_interactive.py --level N --render-fly-vision` — WASD 키보드 조종 + 겹눈 시점

즉 **"포식자를 피하면서 바람과 냄새를 단서로 먹이를 찾는 초파리"**가 이미
만들어져 있습니다. 학기 프로젝트 규모의 연구를 바로 시작할 수 있습니다.

---

## 5. flybody — 비행을 연구하려면

```bash
git clone https://github.com/TuragaLab/flybody   # Python 3.10 + conda 필요
```

포함된 RL 태스크 (`flybody/tasks/`):

| 파일 | 과제 |
|---|---|
| `walk_imitation.py` | 실측 보행 궤적 모방 (행동 공간 59차원) |
| `flight_imitation.py` | 비행 궤적 모방 |
| `vision_flight.py` | **시각 유도 비행** (장애물 회피) |
| `walk_on_ball.py` | 구슬 위 고정 보행 (실험 셋업 재현) |
| `template_task.py` | 새 과제를 만들 때의 출발점 |

- 유체력: `ellipsoid_fluid_model.py` (날갯짓 공기역학 근사)
- 학습: `agents/agent_dmpo.py` + `ray_distributed_dmpo.py` (Acme/TF/Ray 분산 DMPO)
- 데이터: [Janelia Figshare](https://janelia.figshare.com/articles/dataset/25309105) (모캡 궤적)
- 모델만 필요하면 [mujoco_menagerie/flybody](https://github.com/google-deepmind/mujoco_menagerie/blob/main/flybody/README.md)

> 미실행. 문서 기준 정리입니다. Python 3.10 + TensorFlow라서 여기 3.12 환경과
> 별도 venv가 필요합니다.

---

## 6. 커넥톰 — 뇌 배선도를 몸에 연결하기

2024–2026년에 **초파리 신경계 전체의 배선도**가 공개되면서 판이 바뀌었습니다.

| 데이터셋 | 범위 | 규모 | 접근 |
|---|---|---|---|
| **MaleCNS** v1.0 (2026-06) | 수컷 **뇌 + 배신경삭 전체** | 166,691 뉴런 / 11,691 세포형, 신경전달물질 예측 포함 | [male-cns.janelia.org](https://male-cns.janelia.org/) — 인증 없이 다운로드 |
| **FlyWire** (FAFB v783) | 암컷 뇌 전체 | ~139,000 뉴런 | [flywire.ai](https://flywire.ai/), [Codex](https://github.com/murthylab/codex) |
| **MANC** | 수컷 배신경삭 | — | [Janelia MANC](https://www.janelia.org/project-team/flyem/manc-connectome) |
| **BANC** | 암컷 뇌+신경삭 | — | [htem/BANC-project](https://github.com/htem/BANC-project) |
| **Hemibrain** | 뇌 중앙부 | ~25,000 뉴런 | neuPrint |

**Python 접근 도구**

```bash
pip install neuprint-python caveclient navis fafbseg connectome-interpreter
```

- [`neuprint-python`](https://github.com/connectome-neuprint/neuprint-python) — neuPrint 질의
- [`CAVEclient`](https://github.com/CAVEconnectome/CAVEclient) — FlyWire 세그멘테이션
- [`navis`](https://github.com/navis-org/navis) — 뉴런 형태 분석·시각화
- [`connectome_interpreter`](https://github.com/YijieYin/connectome_interpreter) — 유효 연결성, 회로 조작
- [`Connecto`](https://github.com/schlegelp/connecto) — CAVE/neuPrint 질의 통합 인터페이스

**표준 시뮬레이션 모델**

- [`philshiu/Drosophila_brain_model`](https://github.com/philshiu/Drosophila_brain_model) (286★) —
  Shiu et al.의 leaky integrate-and-fire 전뇌 모델. 이 분야의 사실상 기준 구현.
- [`TuragaLab/flyvis`](https://github.com/TuragaLab/flyvis) (165★) —
  커넥톰 제약 시각계 심층 기계론적 모델(PyTorch). 65개 원주형 세포형, 45,669 세포.
  Lappalainen et al., Nature 2024. **사전학습 모델 제공.** NeuroMechFly의
  겹눈 출력을 여기에 바로 먹일 수 있습니다.

### 커넥톰 ↔ 신체 결합 오픈소스

| 저장소 | 조합 |
|---|---|
| [erojasoficial-byte/fly-brain](https://github.com/erojasoficial-byte/fly-brain) (58★) | FlyWire v783 전뇌 138,639 뉴런 LIF → 생체역학 신체 |
| [Lulzx/fly-brain](https://github.com/Lulzx/fly-brain) | 165k 뉴런 LIF(WASM) + flybody + flyvis, 브라우저 |
| [abgnydn/webgpu-fly](https://github.com/abgnydn/webgpu-fly) | FlyWire 뇌 + MANC 신경삭 + flybody, WebGPU 실시간 |
| [seven-monarchs/NeuroFly](https://github.com/seven-monarchs/NeuroFly) | FlyWire 활동 → NeuroMechFly 신체 폐루프 |
| [caparison1234/chimera](https://github.com/caparison1234/chimera) | 유충 커넥톰 1,373 뉴런 → MuJoCo 신체 |

전체 색인: [cobanov/awesome-fly](https://github.com/cobanov/awesome-fly) — 92개 항목
(게임, 데스크톱 초파리, 분석 라이브러리, 데이터셋까지 망라).

> 이 목록의 상당수는 MaleCNS 공개 직후 만들어진 개인 프로젝트이고 검증 수준이
> 제각각입니다. 연구 목적이면 `philshiu/Drosophila_brain_model`,
> `flyvis` 같은 논문 기반 구현부터 보는 편이 안전합니다.

### 학술적으로 가장 앞선 시도

> Zehao Jin, Yaoye Zhu, Chen Zhang, Yanan Sui,
> **"Whole-Brain Connectomic Graph Model Enables Whole-Body Locomotion Control
> in Fruit Fly"** (2026), [arXiv:2602.17997](https://arxiv.org/abs/2602.17997)

성체 초파리 전뇌 커넥톰을 **그래프 구조 신경망 그 자체**로 구현해
(뉴런=노드, 시냅스=엣지, 메시지 패싱으로 정보 전파), 원심성 상태를 운동 명령으로
디코딩해 생체역학 초파리를 제어합니다. 그래프/비그래프 기준선보다 **표본 효율이
좋다**고 보고합니다. 즉 "커넥톰 배선 구조가 그 자체로 유용한 귀납 편향"이라는
주장입니다. 이 프로젝트의 확장 주제 8번과 직결됩니다.

### 유충(larva) 모델 — 성체보다 신경계가 훨씬 작습니다

성체 뇌가 14만 뉴런인 반면 **1령 유충 커넥톰은 약 3,000 뉴런**(Winding et al.)이라
전뇌를 통째로 시뮬레이션하기가 현실적입니다. 커넥톰↔행동 폐루프 연구를
소규모로 시작하려면 유충이 더 나은 출발점일 수 있습니다.

- [nawrotlab/larvaworld](https://github.com/nawrotlab/larvaworld) — 유충 행동 분석·시뮬레이션 플랫폼
  ([eLife reviewed preprint](https://elifesciences.org/reviewed-preprints/104262))
- [ChenYvhang/cyber-larva](https://github.com/ChenYvhang/cyber-larva) — Winding et al. L1 커넥톰 제약
  LIF 신경망 + 11분절 MuJoCo 신체 + 변형 가능한 Three.js 렌더링, 1 ms 스텝

---

## 7. 실험 데이터 · 포즈 추정 (EPFL 도구 체인)

시뮬레이션을 실제 초파리와 맞추려면 필요합니다.

| 저장소 | 용도 |
|---|---|
| [DeepFly3D](https://github.com/NeLy-EPFL/DeepFly3D) (101★) | 고정된 초파리의 무표지 3D 포즈 추정 (7카메라) |
| [LiftPose3D](https://github.com/NeLy-EPFL/LiftPose3D) (49★) | 2D 포즈 → 3D 리프팅 |
| [sequential-inverse-kinematics](https://github.com/NeLy-EPFL/sequential-inverse-kinematics) (16★) | 3D 포즈 → 관절각 (시뮬레이터 입력용) |
| [df3dPostProcessing](https://github.com/NeLy-EPFL/df3dPostProcessing) | 정렬 및 다리 관절각 계산 |
| [poseforge](https://github.com/NeLy-EPFL/poseforge) | **시뮬레이션 기반 무라벨 포즈 추정** |
| [spotlight](https://github.com/NeLy-EPFL/spotlight) / [spotlight-hardware](https://github.com/NeLy-EPFL/spotlight-hardware) | 신규 기록 플랫폼 (소프트웨어 + 하드웨어 설계) |

FlyGym에 내장된 실측 데이터는 `flygym_demo/spotlight_data/`에 있고,
`scripts/replay_behavior_cpu.py` / `replay_behavior_gpu.py`로 재생됩니다.

---

## 8. 브라우저에서 바로 — 설치 없이

FlyGym 2.1.0은 MuJoCo를 WebAssembly로 컴파일해 브라우저에서 돌립니다.
둘 다 접속 확인했습니다 (HTTP 200).

- **NeuroMechFly Live (슬라럼 게임)** — <https://neuromechfly.org/wasm/game/game.html>
  세 가지 추상화 수준으로 초파리를 조종합니다: 레벨 1은 CPG가 알아서 걷고
  방향만 지시, 레벨 2는 삼각보행 그룹 단위, 레벨 3은 **다리 6개를 개별 조작**.
  운동 제어 계층을 몸으로 이해하게 만드는 훌륭한 교보재입니다.
- **대화형 자세 뷰어** — <https://neuromechfly.org/wasm/viewer/viewer.html>
  액추에이터별 슬라이더, 접촉/힘/관절 오버레이.

소스는 `flygym-src/wasm/`에 있으니 직접 수정해 자기 실험을 브라우저로 배포할 수도 있습니다.

---

## 9. 여기서 바로 시작할 수 있는 연구 주제

이 PC에 이미 구축된 것(실시간 뷰어, 지형 4종, PPO 파이프라인, 분석 스크립트)
위에서 난이도 순으로 정리했습니다.

**난이도 하 — 며칠**

1. **지형별 조향 법칙 비교.** `05_train.py`를 4개 지형에 각각 돌리고
   `07_analyze_policy.py --run nav_flat nav_blocks ...`로 학습된 조향 곡선을
   겹쳐 그립니다. 가설: 험지일수록 이득(gain)이 낮고 포화가 빠를 것.
   *이미 만들어져 있습니다. 실행만 하면 됩니다.*
2. **보행 패턴(gait) 비교.** `get_cpg_biases`가 tripod/tetrapod/wave를 지원합니다.
   지형별로 어느 보행이 유리한지 — 속도, 안정성, 에너지(액추에이터 힘 적분).
3. **부착력(adhesion) 절제 실험.** `make_locomotion_fly(add_adhesion=False)`.
   논문 Fig 2의 임계 경사각을 직접 재측정.

**난이도 중 — 1~2주**

4. **시각 기반 목표 탐색.** `build(vision=True)` 후 관측의 목표 방위 2차원을
   `get_ommatidia_readouts()`로 교체. 721×2 옴마티디아를 그대로 넣으면 학습이
   어려우니, 논문처럼 CNN으로 (방향, 거리, 시야 내 여부)를 먼저 뽑아내는
   2단계 구조를 권합니다. → **논문 Figure 5의 시각 부분 재현**
5. **후각 주행성 + RL.** ✅ **구현 완료** — [flyplay/odor.py](flyplay/odor.py),
   [flyplay/env_multimodal.py](flyplay/env_multimodal.py), `scripts/10~12`.
   손으로 만든 주화성 규칙이 장애물 없는 평지에서 8/8 도달하고, 기둥 2개를 넣으면
   3/8로 무너지며(충돌 134스텝), 시각을 더하면 8/8·충돌 3.1로 회복됩니다.
   → **논문 Figure 3/5에 해당**
6. **경로 적분.** 다리 고유수용 신호만으로 출발점 방향을 추정하는 회귀망 학습.
   cobar-2026 week6이 출발점. → **논문 Figure 4 재현**
7. **CPG 자체를 학습시키기.** 지금 정책은 하행 신호만 냅니다. 행동 공간을
   42개 관절각으로 바꾸면 수백만 스텝이 필요하지만, **GPU 배치 시뮬레이션**으로
   가능합니다. 규모를 제대로 잡는 게 관건입니다 — §9.1 참고.

### 9.1 GPU를 제대로 쓰는 법 (공식 튜토리얼 3에서 확인)

```python
from flygym.warp import GPUSimulation
from flygym.warp.utils import check_gpu

check_gpu()
sim = GPUSimulation(world, n_worlds=2048)   # Simulation과 같은 인터페이스 + 월드 차원
```

- **월드 수가 전부입니다.** 수백 개로는 CPU를 못 이깁니다. FlyGym 자체 스케일링
  테스트 기준 **최대 처리량은 2,000~17,000 월드**에서 나옵니다.
  RTX 3080 Ti에서 **~30× 실시간**, L40S/H100에서 **~60× 실시간**.
- 모든 월드가 **같은 모델**을 공유해야 합니다(데이터는 달라도 됨).
- `set_renderer(..., use_gpu_batch_rendering=True)`로 렌더링도 GPU에서 일괄 처리.
  단 텍스처가 꺼지고 조명이 달라집니다.
- **CPU-GPU 동기화가 진짜 병목입니다.** 파이썬 루프에서 매 스텝 제어 입력을
  올리면 그 오버헤드가 대부분을 차지합니다. `wp.ScopedCapture()`로 내부 루프를
  **CUDA 그래프로 캡처**하고 제어 입력을 Warp 커널이 GPU 상에서 직접 쓰게 해야
  광고된 성능이 나옵니다.
  ⚠️ 캡처 블록 안의 파이썬(CPU) 코드는 캡처 시점에 **한 번만** 실행되고 이후
  재생에서는 건너뜁니다. `step_with_profile()`을 그 안에 쓰면 안 됩니다.

[scripts/08_gpu_benchmark.py](scripts/08_gpu_benchmark.py)가 단순 루프와
그래프 캡처 루프를 512/2048/8192 월드로 비교합니다. 이 PC의 RTX 5060(8 GiB,
sm_120, CUDA 12.9)에서 Warp가 GPU를 인식하는 것은 확인했고, 처리량 측정은
CPU가 한가할 때 돌려야 의미가 있습니다.

**이 PC에서 잰 값 (2026-09-17, 샌드박스 배경 실험을 GPU로 옮길지 판단하려고).** 모든 파리를
합친 속도(시뮬레이션 초 ÷ 실제 초)입니다.

| 방식 | 64 | 256 | 1024 | 4096 |
|---|---:|---:|---:|---:|
| GPU, 가만히 선 자세 (08_gpu_benchmark) | 6.8× | 12.1× | 14.1× | 14.5× |
| GPU, 걷는 관절 목표 재생 (flygym_demo 벤치마크 방식) | — | 8.5× | 11.5× | 12.8× |
| 위 + 500 Hz마다 CPU와 관절각·목표 주고받기 | — | 8.5× | 11.1× | 12.6× |

| 방식 | 1 | 8 | 14 |
|---|---:|---:|---:|
| CPU 프로세스, 샌드박스 파리 전체(물리·걷기 제어기·겹눈·버섯체) | 0.72× | 4.5× | 6.5× |

샌드박스 파리 자체 모델(방 물건 풀, 다리-벽 충돌 쌍 756개, 주둥이)로 같은 걷기 재생을 다시 잰 값입니다(같은 날 저녁).

| 방식 | 64 | 256 | 1024 | 2048 |
|---|---:|---:|---:|---:|
| GPU, 샌드박스 모델 걷기 재생 | 5.0× | 9.2× | 12.5× | 끝나지 않음 |
| 위 + 500 Hz마다 CPU와 주고받기 | 4.7× | 8.8× | 12.2× | — |

- 충돌 쌍이 많아도 FlyGym 기본 파리와 속도가 거의 같습니다.
- 2048마리는 노트북 GPU 메모리 8 GB 중 7.9 GB를 채우고 16분 동안 0.2초를 못 끝내서 멈췄습니다. 이 PC의 상한은 약 1024마리이고,
  파리 100마리 아래에서는 CPU 14프로세스(6.5×)가 더 빠릅니다.
- MJWarp는 MULTICCD가 켜진 모델에서 충돌 쌍의 margin(0.001)을 거부합니다. GPU용 사본에서는 0으로 둬야 합니다.
- FlyGym `GPUSimulation`에는 접촉 힘, 월드 하나만 초기화, 겹눈 읽기가 없습니다. CPU용 메서드는 GPU가 갱신하지 않는
  `mj_data`를 읽어 오래된 값을 돌려줍니다.

- GPU 상한은 CPU의 약 2배지만 **물리만**의 값입니다. 버섯체·행동 규칙·겹눈을 일괄 계산으로 다시 짜서
  더해야 하고, MuJoCo-Warp는 CPU 모델이 쓰는 noslip 반복을 끕니다(경고로 확인). 걷기부터 다시 검증해야
  합니다. 파리 수십 마리 규모에서는 CPU 병렬이 낫다고 판단했습니다.
- **처음 잰 CPU 값(15개에 3.75×)은 틀렸습니다.** 다른 뷰어 프로세스 하나가 코어 14.7개를 쓰고 있었습니다
  — FlyGym 망막의 numba 병렬 루프(OpenMP 계층)와 numpy OpenBLAS의 쉬는 스레드가 잠들지 않고 돌았습니다.
  `OMP_WAIT_POLICY=PASSIVE`, `KMP_BLOCKTIME=0`, `OPENBLAS_NUM_THREADS=1`로 파리 한 마리가 같은 속도
  (13.7 ms/스텝)에 CPU 4.7코어 → 1.1코어가 됐습니다. 멀티프로세스 측정 전에는 다른 파이썬 프로세스의
  CPU 사용량부터 확인해야 합니다.

8. **후각 전단(front-end) 추가** ✅ **구현 완료** —
   [flyplay/olfactory.py](flyplay/olfactory.py), 검증은
   [scripts/13_mb_check.py](scripts/13_mb_check.py)

   [flyplay/odor.py](flyplay/odor.py)는 센서당 **스칼라 하나**를 줍니다. 그것으로는
   냄새의 *정체*를 표현할 수 없고, 정체가 없으면 버섯체가 벌과 짝지을 대상도
   없습니다. 실제 초파리의 후각 경로를 구조만 옮겨 넣었습니다.

   | 단계 | 구현 | 측정으로 확인한 역할 |
   |---|---|---|
   | ORN 50종 | 냄새 차원마다 수용체의 **30%만** 구동하는 희소 친화도 행렬 + 힐 포화 | 밀집 행렬이면 모든 항이 양수라 두 냄새의 PN 패턴 상관이 0.7대가 되어 분리에 실패 |
   | 촉각엽(AL) | `r_i / (σ + Σ_j r_j)` | **KC 부호에는 아무 영향이 없음** — 아래 참고 |
   | KC 2000개 | KC당 PN 6개, **시냅스 세기가 제각각**인 희소 투사 → top-k (5%) | 세기가 균일하면 활성 사구체가 같은 KC끼리 구동값이 정확히 같아져 임계에서 동수가 생김 |
   | MBON | KC 가중합 | 10번에서 가소성을 붙임 |

   **측정값** (배선 시드 8개, 초파리가 실제로 걷는 농도 범위 0.0016→0.11, 69배):

   | 항목 | 값 | 기준 |
   |---|---:|---|
   | 같은 냄새, 농도 69배 변화 | **0.989** (최악 0.978) | > 0.9 |
   | 냄새 A 대 B | **0.051** (최악 0.180) | < 0.2 |
   | A와 B가 공유하는 KC | 8.2% | — |

   **처음 적었던 가설 하나가 틀렸습니다.** "분할 정규화가 농도 불변성을
   만든다"고 봤는데, 이 형태의 분모는 스칼라라 PN 벡터 전체를 같은 비율로
   줄일 뿐이고 top-k는 크기에 불변입니다. σ를 0에서 1.0까지 쓸어도 KC 부호가
   **하나도 바뀌지 않습니다.** 불변성은 ORN 힐 포화가 작동 범위에서 거의
   선형이기 때문에 생깁니다. 사구체별로 작동하는 Olsen–Wilson 형태
   (`r^1.5 / (r^1.5 + σ^1.5 + (m Σr)^1.5)`)도 시험했는데 불변성이 0.989 →
   **0.822로 오히려 나빠졌습니다.** 실제 촉각엽에서는 이 모델이 다루지 않는 훨씬
   넓은 농도 범위와 ORN 적응이 함께 있어야 의미가 생기는 연산으로 보입니다.

   **작동 범위를 넘으면 무너집니다.** 0.1→1.0 구간(초파리가 발생원에 머리를 박은
   상태)에서는 불변성이 0.87로 떨어집니다. 그래서 10번 과제는 냄새 세기를
   `min_distance=3.0` mm에서 잘라 상한을 정확히 0.111에 맞춥니다. 안 그러면
   버섯체가 "멀리서 맡은 A"와 "가까이서 맡은 A"를 다른 냄새로 배웁니다.

   **남은 질문.** 혼합물 — 50:50으로 섞으면 KC 부호가 순수 A와 평균 0.46, 순수
   B와 0.63으로 둘 사이에 놓이지만, 배선 시드에 따라 **0.20에서 0.91까지** 크게
   흔들립니다(시드 8개 측정). 어느 냄새의 수용체 친화도가 그 배선에서 우연히 더
   셌느냐가 혼합물의 정체를 정한다는 뜻이고, 개체마다 혼합물을 다르게 지각할
   것이라는 예측이 됩니다. 혼합물 일반화나 농도 판별 과제는 아직 해보지
   않았습니다.

**난이도 상 — 한 학기 이상**

9. **커넥톰 제약 시각망 연결.** flyvis 사전학습 모델을 NeuroMechFly 겹눈 뒤에
   붙여, 실제 T4/T5 운동 검출 회로가 만들어내는 광류(optic flow)로 조향.
   → **논문 Figure 6의 초파리 추종 재현**

   실현 가능성 확인: `pip install flyvis`가 이 Python 3.12 환경에서 **의존성
   충돌 없이 해결됩니다**(dry-run 확인). 다만 torch/torchvision, scikit-learn,
   umap, xarray 등 30개 남짓을 추가로 끌어오므로 **별도 venv**에 두고 넘파이
   배열로 주고받는 편이 안전합니다. flyvis 튜토리얼 7개 중 *Custom Stimuli*가
   외부 입력을 넣는 방법을 다루므로 거기서 시작하세요. 접점은 명확합니다 —
   FlyGym의 `sim.get_ommatidia_readouts()`가 (2, 721, 2) 육각 격자 값을 주고,
   flyvis도 육각 수용체 격자를 입력으로 받습니다.

10. **버섯체 연합학습** ✅ **구현 완료 (역전은 소거로 해결)** —
   [flyplay/mushroom_body.py](flyplay/mushroom_body.py),
   [flyplay/conditioning.py](flyplay/conditioning.py), `scripts/13~15`,
   `09_web_viewer.py --conditioning`. 결과는 이 항목 끝의 **구현 결과**에 있습니다.

   구현 전의 상태: 이 프로젝트의 "뇌"는 `Linear(40→128)→Tanh→Linear(128→128)→Tanh→
   Linear(128→2)` MLP 하나였습니다. 버섯체도 케니언 세포도 도파민 뉴런도
   없었고, FlyGym도 제공하지 않습니다(NMF2 논문이 쓴 커넥톰 신경망은
   **시엽**이지 버섯체가 아닙니다).

   **핵심은 과제부터 바꿔야 한다는 것입니다.** 버섯체는 연속 조향 제어기가
   아니라 **연합학습** 회로입니다. 지금 먹이찾기 과제에 그대로 끼워 넣으면
   동작하지 않습니다. 버섯체가 실제로 푸는 과제로 바꿔야 합니다 —

   > 냄새 A와 냄새 B를 제시하고, **A 뒤에만 처벌(또는 보상)이 따라옵니다.**
   > 초파리는 시행을 거치며 A를 회피하고 B로 접근하게 되어야 합니다.
   > 이후 역전(reversal) 시행으로 재학습 능력까지 볼 수 있습니다.

   구조 (8번 후각 전단 위에 얹습니다):

   ```
   KC(2000, 희소) ──┬─→ MBON_approach ─┐
                    │                   ├─→ 접근/회피 → 하행 신호
                    └─→ MBON_avoid ────┘
          ↑
      DAN (도파민 뉴런): 처벌/보상 시점에 발화
      학습 규칙: Δw(KC→MBON) = −η · KC활성 · DAN활성      (반헤브 억압)
   ```

   실제 초파리의 특징을 살리려면 두 가지를 꼭 넣으세요.
   - **구획(compartment)별 독립성** — 한 DAN은 자기 구획의 KC→MBON 시냅스만
     바꿉니다. 지금의 스칼라 δ 하나와 가장 크게 다른 지점입니다.
   - **억압(depression) 방향** — 도파민은 활성 KC에서 MBON으로 가는 연결을
     **약화**시킵니다. 강화가 아닙니다. 그래서 "처벌과 짝지어진 냄새는 접근
     MBON을 덜 구동한다"가 됩니다.

   난이도 **상**, 한 학기 규모. 참고 문헌:
   - [MB 커넥톰 원 논문](http://lk.zuckermaninstitute.columbia.edu/pdf/li_connectome_2020.pdf) — Li et al. 2020, hemibrain 기반 버섯체 배선도
   - [프로그래밍 가능한 MB 모델](https://www.biorxiv.org/content/10.1101/2022.09.10.506218v1.full) — 커넥톰 기반, PN→KC 연결과 APL 피드백이 냄새 표상에 미치는 영향
   - [MBON-α3 결정 모듈](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10069864/) — EM 형태 + 948개 KC 시냅스 + 패치클램프 막특성
   - [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) (286★) — FlyWire 전뇌 LIF. 버섯체 서브그래프를 뽑아 쓸 수 있습니다

   이것이 이 프로젝트에서 **"진짜 몸 + 진짜 감각기 + 가짜 뇌"의 마지막 칸을
   채우는** 작업입니다.

   **구현 결과**

   *회로* (`13_mb_check.py`, 합성 시행, 12항목 전부 통과): CS+ 20시행 후
   MBON_approach가 짝지어진 냄새에 대해 −69%, 짝지어지지 않은 냄새는 −1.1%, 다른
   구획은 0.0%. 도파민 없이 120초 두면 0.987로 복원. **전역 도파민 하나로는
   가치 분리가 정확히 0.000** — 스칼라 신호가 두 구획에 같은 입력을 주므로
   가중치가 갈라질 수 없는 항등식입니다.

   *행동* (`14_run_conditioning.py`, 실제 몸, 조건당 5마리 × 58시행, 8.6분):

   | 조건 | 순진 | A에 벌 준 뒤 | B로 역전한 뒤 | 가치 분리 |
   |---|---:|---:|---:|---:|
   | 구획 2개 (본 실험) | −0.05 ± 0.07 | **+0.49 ± 0.02** | **−0.30 ± 0.01** | +0.392 |
   | 전역 도파민 1개 | −0.05 ± 0.07 | +0.01 ± 0.13 | −0.04 ± 0.05 | **0.000** |
   | 가소성 없음 | −0.05 ± 0.07 | +0.01 ± 0.13 | −0.04 ± 0.05 | **0.000** |

   선호도는 +1이 B, −1이 A 쪽. 오차는 개체 간 표준오차(n=5). 전역 도파민과
   가소성 없음 조건은 같은 시드에서 **궤적까지 비트 단위로 같습니다**(확인함).
   둘 다 가치가 정확히 0이라 조향 입력이 같기 때문입니다. 다만 같은 회로는
   아닙니다 — 전역 도파민은 시냅스를 기준 조건보다 **더 깊게** 누르는데(접근 MBON
   최저 0.02~0.03, 기준 조건 0.45~0.52), 회피를 못 배워 계속 A에 들어가 벌을 더
   맞기 때문이고, 두 구획이 똑같이 눌려서 차이가 생기지 않을 뿐입니다.

   훈련 시행당 도파민을 세면 이것이 행동으로 보입니다. 기준 조건은 첫 5시행
   0.76초에서 마지막 5시행 0.21초로 줄어 **회피를 배운 만큼 벌을 덜 맞고**, 전역
   도파민 조건은 2.25초 → 2.45초로 줄지 않습니다(평균 2.30초, 기준 조건 0.33초의
   7배). 선호도는 시험 시행만 채점하므로, 이 지표는 훈련 중의 학습을 따로
   보여줍니다.

   계획서의 기준과 대조: 획득 후 > +0.4 **통과**, 전역 도파민 학습 불가
   **통과**, 역전 후 < −0.4 **미달**(5마리 모두 부호는 뒤집힘).

   **역전이 약한 이유 — 이것이 이 모델의 가장 흥미로운 결과입니다.** 이 회로가
   기억을 지우는 방법은 도파민과 무관한 수동적 복원(시정수 100초) 하나뿐입니다.
   배우는 속도는 도파민이 정하고 잊는 속도는 시정수가 정하므로 둘이 맞을 이유가
   없습니다. 실제로 역전 시험이 시작될 때 A에 대한 approach 반응은 0.85까지만
   돌아와 있어서, 초파리는 새로 벌받은 B뿐 아니라 A도 여전히 조금 싫어합니다.
   수치도 맞아떨어집니다 — 두 냄새 사이 접근 구동 차이가 획득 때 0.45, 역전 때
   0.30으로 비율 0.67이고, 선호도 비율은 0.49 대 0.30으로 0.61입니다.

   **고칠 수 있는 파라미터가 있지만 고치지 않았습니다.** 복원을 빠르게 하면
   역전은 좋아지겠지만 시험 블록 동안 기억도 그만큼 빨리 날아갑니다. 시정수
   30초에서는 시험 6회 동안 A의 가치가 −0.31에서 −0.12로, 눌림의 63%가
   사라졌습니다. 다만 그 실행은 조향 방식을 고치기 전의 것이라 **행동 수치는
   비교할 수 없고**, 고친 조향으로 다시 재지는 않았습니다. 무엇보다 기준을
   맞추려고 조정하면 위의 메커니즘이 가려집니다.

   **후속 실험: 능동적 소거(extinction)** ✅ — 역전이 약한 원인을 고치는 실험.

   실제 초파리는 옛 기억이 저절로 흐려지기를 기다리지 않습니다. Felsenberg 등
   (2018)에 따르면 혐오 기억의 소거에는 특정 도파민 뉴런이 필요하고, 그 뉴런들은
   **벌이 오지 않은 것을 긍정적 경험으로 기억**하게 합니다. 이렇게 생긴 새 기억이
   원래 혐오 기억과 나란히 존재하다가 회피를 지시하는 MBON에서 합쳐져 원래 기억을
   상쇄합니다.

   *구현.* MBON → DAN 되먹임입니다(`MushroomBody.step(feedback=)`). 벌을 예상한
   냄새(가치 < 0)를 맡았는데 그 시행에 벌이 없으면, 예상한 크기 `max(0, −가치)`만큼
   도파민이 지정한 구획으로 갑니다. 이득 1.0은 "가치 −1이면 실제 벌과 같은 양"이라는
   기준으로 정했고 결과에 맞춘 값이 아닙니다. 모델에 시간 개념(적격 흔적)이 없어서
   예상은 시행 단위로 판정합니다 — 벌이 오는 시행에서는 신호를 끕니다. 2구획판
   근사이고, 논문의 회로도와 같지는 않습니다.

   *프로토콜.* 원래 과제는 역전 훈련 중 A를 아예 보여주지 않아서 소거가 작동할
   기회가 없습니다. 그래서 세 조건 모두 **차등 조건화**(벌 있는 시행과 벌 없는
   노출을 번갈아, 블록 길이는 그대로)로 바꿨습니다. 결과를 보기 전에 예측을
   적어 두고 실행했습니다.

   | 조건 | 획득 후 | 역전 후 | 대칭성 \|역전\|/\|획득\| | 예측 |
   |---|---:|---:|---:|---|
   | 차등, 소거 없음 | +0.49 ± 0.01 | −0.30 ± 0.00 | 0.61 | 원래와 같음 ✅ |
   | **차등 + 소거 → avoid** | **+0.48 ± 0.01** | **−0.50 ± 0.02** | **1.04** | 역전 강해짐 ✅ |
   | 차등 + 소거 → approach | −0.01 ± 0.04 | +0.06 ± 0.01 | — | 역전 실패 (실제로는 더 나쁨) |

   조건당 5마리 × 58시행, 7.5분(`14_run_conditioning.py --experiment extinction`).
   회로 검사 `13_mb_check.py`에 소거 항목 4개가 추가돼 16/16 통과합니다.

   **첫째, 역전이 획득만큼 강해졌습니다.** 5마리 모두 −0.42~−0.56이고 대칭성이
   0.61에서 **1.04**로 올랐습니다. 프로토콜만 바꾼 대조군은 원래와 똑같이 −0.30이라,
   차이는 소거 신호에서 옵니다.

   **둘째, 원래 기억은 지워지지 않았습니다.** 역전 시험 직전 A에 대한 approach
   반응이 소거 조건과 대조군에서 **0.872로 같고**, 소거 조건에서만 avoid가 1.000 →
   0.873으로 내려와 가치가 −0.001이 됐습니다. 두 기억이 나란히 있고 서로 상쇄하는
   것입니다. 몸 없는 검사에서는 approach가 수동 망각만 한 조건과 소수점 9자리까지
   같았습니다.

   **셋째, 예상과 달리 소거가 획득 시험을 해치지 않았습니다.** 시험도 벌이 없으니
   획득 수치가 약해질 거라고 예측했지만 +0.48로 그대로였고, 시험 6회 동안 선호도도
   +0.39 → +0.54로 줄지 않았습니다. 시험 중 소거 신호가 시행당 0.12 도파민·초로
   작았기 때문입니다(측정). A를 피하는 초파리가 B 쪽에 머물러 B 위주의 혼합물을
   맡기 때문으로 보입니다(해석).

   **넷째, 잘못 연결하면 공포가 번집니다 — 가장 뜻밖의 결과.** 소거 신호를 approach로
   보내면 역전만 망가질 거라고 봤는데 **획득부터** 무너졌습니다. 원인을 추적하니 이렇습니다.
   B는 A와 케니언 세포 8%를 공유해서, 첫 A 벌 직후 가치가 **−0.0042**로 아주
   조금 음수가 됩니다(두 조건에서 같은 값). 올바른 배선에서는 이 작은 혐오가
   노출 중에 상쇄되어 −0.002 근처에 머뭅니다. 잘못된 배선에서는 그 작은 혐오가
   신호를 만들고 신호가 혐오를 더 키우는 **양성 되먹임**이 되어, B 노출 10회 동안
   −0.004 → −0.011 → −0.022 → −0.039 → −0.074 → −0.137 → −0.253 → −0.423 →
   −0.566 → **−0.650**으로 거의 매번 두 배가 됐습니다. 한 번도 벌받지 않은 B가
   벌받은 A(−0.44)보다 더 무서워졌습니다. 같은 신호라도 **어느 구획에 들어가느냐가
   소거와 공포 일반화를 가릅니다.**

   **남은 질문.**
   - 소거된 기억의 **자발적 회복**(시간이 지나면 옛 공포가 돌아옴)은 이 모델에 없습니다.
     두 구획의 복원 시정수가 같아서 상쇄가 계속 유지되기 때문입니다. 구획마다
     시정수를 다르게 두면 생기는지 시험할 수 있습니다.
   - 시험 중 두 냄새를 함께 맡을 때 B의 소거가 **공유 케니언 세포를 통해 A로 번져**
     A의 가치가 0을 넘어 양수(+0.03)가 되는 것을 한 마리에서 봤습니다. 안도감의
     일반화로 보이며, 따로 측정하지 않았습니다.
   - [Felsenberg et al. 2017, Nature 544:240](https://www.nature.com/articles/nature21716) — 기억을 인출할 때 예측이 맞았는지에 따라 MBON이 서로 다른 도파민 뉴런 조합을 구동해 (보상) 기억을 소거하거나 재공고화함
   - [Felsenberg et al. 2018, Cell 175:709](https://doi.org/10.1016/j.cell.2018.08.021) — 혐오 기억의 소거는 서로 맞서는 두 기억이 MBON에서 합쳐지는 것

   **기억 라이브러리.** 학습된 초파리는 블록마다 시냅스가 `.npz`로 저장되고,
   `scripts/16_memory.py`로 이름을 붙여 `memories/`에 보관합니다(실험을 다시 돌려도
   덮어써지지 않음). 기억 파일에는 가중치와 배선 시드, **초파리 설정**이 함께
   들어갑니다 — 같은 시냅스라도 소거 설정이 다르면 다음 시행부터 다르게 행동하기
   때문입니다. 지금 보관된 것: `naive`, `a-punished`, `reversed-passive`,
   `reversed-extinction`, `fear-generalized`. `09_web_viewer.py --memory 이름`으로
   불러와 행동을 봅니다.

   **설계 중 뒤집힌 판단 세 가지** (코드 주석에 근거와 함께 남김):
   - 조향 이득은 12가 아니라 **80**. 정면 발생원의 좌우차는 0.0006이라 어떤
     이득으로도 조향이 안 되므로, 훈련 발생원도 ±35° 안의 무작위 방위에 둡니다.
   - 가치는 **냄새별로** 곱해야 합니다. 좌우차를 더한 뒤 혼합물의 가치 하나를
     곱하면, 시험 시행에서 좌우의 두 기울기가 상쇄돼 초파리가 배운 것을 행동으로
     옮길 방법이 없어집니다.
   - 도파민 지연은 시행 시작이 아니라 **냄새를 처음 맡은 순간**부터 잽니다.
     초파리는 0.7~2.2초 사이에 냄새 구역을 지나가 버려서, 시행 시계로 3초를
     기다리면 도파민이 한 번도 안 들어갑니다(측정).

11. **다개체 상호작용.** `world.add_fly()`를 여러 번 호출하면 됩니다. 확인 결과
   3마리에서 `nq=219, nu=144`, 5마리에서 `nq=365, nu=240`로 정상 구성됩니다.
   구애 추종(논문 Fig 6), 무리 행동, 포식자 회피에 쓸 수 있습니다.

   ⚠️ **단, `add_ground_contact_sensors=False`가 필요합니다.** 기본값(`True`)으로
   2마리 이상을 넣으면 다리별 접촉 센서 이름이 충돌해
   `ValueError: repeated name 'ground_contact_lf_leg' in sensor`로 컴파일이
   실패합니다. flygym 2.1.0의 버그로 보이며, 보고할 가치가 있습니다. 접촉력이
   필요하면 `Simulation.get_bodysegment_contact_forces()`로 대체할 수 있습니다
   (이 프로젝트의 [flyplay/build.py](flyplay/build.py)가 그렇게 합니다).
12. **근육 수준 제어.** `MusculoskeletalFly` + `flygym_demo.muscle_imitation`.
    Hill-type 근육 활성 0~1을 직접 학습. 수동 관절 특성이 학습 속도에 미치는
    영향은 Özdil et al.이 이미 보고했으니, 그 위에서 확장하면 됩니다.
13. **Sim-to-real 검증.** DeepFly3D/poseforge로 실제 초파리를 촬영·추정하고,
    같은 지형에서 시뮬레이션 보행과 통계를 비교.

---

## 10. 이 PC에서 직접 확인한 수치 (참고용 상수)

연구 설계할 때 필요한 값들입니다. 전부 여기서 실행해 얻은 결과입니다.

| 항목 | 값 |
|---|---|
| 물리 시간 간격 | 1e-4 s (10 kHz) |
| NeuroMechFly 모델 | nq=73, nv=72, nu=48 (위치 액추에이터 42 + 부착 6) |
| 겹눈 | 눈당 **721 옴마티디아** (pale 216 / yellow 505), 망막 격자 512×450 |
| `get_ommatidia_readouts()` 반환 | (2, 721, 2) — (좌/우 눈, 옴마티디아, pale/yellow 채널) |
| 후각 센서 (cobar 버전) | 4개 — 좌/우 더듬이, 좌/우 하악수염 |
| 순수 물리 처리량 (CPU 1월드) | 18,483 steps/s = **1.85× 실시간** |
| 하이브리드 제어기 20스텝마다 | 10,855 steps/s = **1.09× 실시간** (보행 동일) |
| 보행 속도 (하행 신호 1.0/1.0) | 14.1 mm/s |
| 선회 속도 범위 | −253 ~ +216 °/s (신호 0.4~1.6 구간) |
| RL 환경 처리량 | 130 env-steps/s (단일), 340~370 (6병렬) |
| 애셋 최초 다운로드 | ~154 MB, 약 10분 (S3 → `~/.cache/flygym_assets`) |

자세한 설정 근거와 지형별 특성은 [README.md](README.md)에 있고, 이 수치들을
얻은 실행 가능한 코드는 [flyplay/](flyplay/), [scripts/](scripts/)에 있습니다.

---

## 11. 인용

```bibtex
@article{wangchen2024neuromechfly2,
  title   = {NeuroMechFly v2: simulating embodied sensorimotor control in adult Drosophila},
  author  = {Wang-Chen, Sibo and Stimpfling, Victor Alfred and Lam, Thomas Ka Chung
             and {\"O}zdil, Pembe Gizem and Genoud, Louise and Hurtak, Femke and Ramdya, Pavan},
  journal = {Nature Methods}, year = {2024}, doi = {10.1038/s41592-024-02497-y}
}

@article{vaxenburg2025flybody,
  title   = {Whole-body physics simulation of fruit fly locomotion},
  author  = {Vaxenburg, Roman and Siwanowicz, Igor and Merel, Josh and others},
  journal = {Nature}, volume = {643}, pages = {1312--1320}, year = {2025},
  doi     = {10.1038/s41586-025-09029-4}
}

@article{lappalainen2024flyvis,
  title   = {Connectome-constrained networks predict neural activity across the fly visual system},
  author  = {Lappalainen, Janne K. and others},
  journal = {Nature}, year = {2024}, doi = {10.1038/s41586-024-07939-3}
}
```

원조 NeuroMechFly v1: Lobato-Rios et al., *Nature Methods* 19 (2022),
[10.1038/s41592-022-01466-7](https://doi.org/10.1038/s41592-022-01466-7).

근골격 모델: Özdil, Ning, Phelps, Wang-Chen, Elisha, Blanke, Ijspeert, Ramdya,
*Musculoskeletal simulation of limb movement biomechanics in Drosophila
melanogaster*, [arXiv:2509.06426](https://arxiv.org/abs/2509.06426).

---

## 12. 분야 전체를 한 번에 조망하려면

FlyGym 저자들이 직접 쓴 리뷰가 가장 좋은 출발점입니다.

> Sibo Wang-Chen & Pavan Ramdya, **"The embodied brain: Bridging the brain, body,
> and behavior with biorealistic neuromechanical models"**,
> *Current Opinion in Neurobiology* (2026).
> [arXiv:2601.08056](https://arxiv.org/abs/2601.08056)

핵심 주장은 이렇습니다. 신경계만, 또는 몸만 따로 모델링하면 근본적으로 불완전하며,
생체현실적 신경역학 모델은 (1) 실험으로 측정하기 어려운 생물리 변수를 계산적
섭동(perturbation)으로 추론하게 해 주고, (2) 신경과학·로보틱스·기계학습 사이의
교환을 촉진하며, (3) 실험 연구와 그 "신경역학 대리물(surrogate)"을 짝지어
능동적으로 탐침하는 것이 분야를 가속할 방법이라는 것입니다.

이 문서에서 정리한 모든 도구가 그 주장의 구현체입니다.

인접 리뷰: Ijspeert & Daley, *The neuromechanics of animal locomotion: From
biology to robotics and back*, [Science Robotics](https://www.science.org/doi/10.1126/scirobotics.adg0279).

---

## 13. 다음 방향 문헌 검토 — 시각 연결과 밀폐된 방 (2026-09-17)

사용자가 정한 다음 순서는 "① 버섯체에 시각 연결 → ② 냄새 나는 설탕물과 장애물이 있는
밀폐된 방에서 길 찾기"입니다. 구현 전에 별도 에이전트 3개로 문헌을 조사해 방향을
검토했습니다. 인용은 검색으로 확인된 것만 남겼고, 에이전트 보고 외에 직접 다시 확인한
것은 따로 표시했습니다.

### 13.1 버섯체의 시각 정보는 무엇에 쓰이는가

- **해부 — 근거 강함.** 버섯체 케니언 세포의 약 **8%**가 주로 시각 입력을 받습니다.
  시각 케니언 세포는 여러 시각 채널을 무작위로 드물게 조합해 받고, 입력 뉴런 다수는
  **수용장이 넓습니다** — 냄새처럼 희소·분산·조합 부호입니다
  ([Ganguly et al. 2024, Nat Commun](https://www.nature.com/articles/s41467-024-49616-z), 직접 확인).
  시각에만 반응하는 γd 세포는 복측 부속 꽃받침을 이루고, 색 기억과 밝기 기억은 서로
  다른 시각 투사 뉴런을 씁니다([Vogt et al. 2016, eLife](https://elifesciences.org/articles/14009), 직접 확인).
  hemibrain 기준 시각 케니언 세포는 γd 99개 + α/βp 60개입니다
  ([Li et al. 2020, eLife](https://doi.org/10.7554/eLife.62576)).
- **행동 — 근거 보통(논쟁 있음).** 걷는 초파리의 파랑/초록 색 학습에 버섯체 γ엽 출력과
  냄새 학습과 **같은** 보상·처벌 도파민 뉴런이 필요합니다
  ([Vogt et al. 2014, eLife](https://elifesciences.org/articles/02395), 직접 확인). 반면 버섯체가 없는
  초파리가 시각 과제 9개를 정상 수행했다는 결과도 있습니다
  ([Wolf et al. 1998, Learn Mem](https://doi.org/10.1101/lm.5.1.166)). 시각 기억은 냄새 기억보다 약합니다.
- **버섯체가 하지 않는 것 — 근거 강함.** 초파리의 시각 **장소 학습은 타원체(중심복합체)**가
  필요하고 버섯체는 필요 없습니다([Ofstad, Zuker & Reiser 2011, Nature](https://doi.org/10.1038/nature10131)).
  비행 시뮬레이터의 무늬 기억은 부채꼴체입니다([Liu et al. 2006, Nature](https://doi.org/10.1038/nature04381)).
  개미에서는 버섯체가 시각 경로 기억을 맡지만, 초파리에서 같은 증거는 찾지 못했습니다.
- **FlyGym 겹눈의 타당성.** 파랑 대 초록은 Vogt 2014가 쓴 바로 그 색 쌍이라 과제로
  적합합니다. 다만 UV(R7)와 광대역 R1–6 채널이 없어서 **색과 밝기가 섞이고**, 빨강을
  검정으로 보는 것은 실제 초파리와 다릅니다(에이전트 보고: 실제로는 빨강을 보지만 초록과
  구별하지 못함). 색 실험에는 밝기를 무작위로 바꾸는 대조군이 반드시 필요합니다.

### 13.2 밀폐된 방에서 냄새로 먹이 찾기

- **냄새 물리.** 확산만으로는 느립니다 — 에틸아세테이트 D ≈ 0.073 cm²/s
  ([Gorur-Shandilya et al. 2019, J Exp Biol](https://doi.org/10.1242/jeb.207787))로 8 cm에 몇 분이
  걸립니다(추정). 연속 발생원의 정상 상태 농도는 3차원에서 1/r, 얇은 2차원에서 로그로
  줄고 **1/r²는 정상 상태 해가 아닙니다.** 벽이 모두 막힌 방에 흡수원이 없으면 정상 상태가
  없습니다. 라플라스 장은 내부 극값이 없어 로봇 경로 계획에 쓰입니다
  ([Connolly et al. 1990, ICRA](https://doi.org/10.1109/ROBOT.1990.126315)).
- **지금 모델의 문제.** `OdorField`는 NeuroMechFly v2와 같은 I/d² 정적 장이고 벽을
  통과합니다([Wang-Chen et al. 2024, Nat Methods](https://doi.org/10.1038/s41592-024-02497-y)도 벽 없이
  사용). 장애물 우회 거리(측지 거리) 장은 길을 미리 알려주는 셈이라 **기준선(오라클)으로만**
  써야 합니다.
- **실제 초파리 전략.** 걷는 초파리는 냄새를 맡으면 **바람을 거슬러** 가속하고, 잃으면
  감속·회전을 늘려 국소 탐색합니다([Álvarez-Salvado et al. 2018, eLife](https://doi.org/10.7554/eLife.37815)).
  난류 속에서는 냄새를 만나는 빈도로 치우친 무작위 회전을 합니다
  ([Demir et al. 2020, eLife](https://doi.org/10.7554/eLife.57524)). 좌우 비교는 존재하지만 주된 단서가
  아닙니다. 먹이를 찾은 뒤에는 자기 운동으로 경로를 적분하며 국소 탐색합니다
  ([Kim & Dickinson 2017, Curr Biol](https://doi.org/10.1016/j.cub.2017.06.026)).
- **설탕물은 냄새가 없습니다.** 식초·에탄올 같은 휘발성 물질과 짝지어야 합니다.
- **걷는 초파리가 실제 장애물을 돌아 냄새를 따라가는 연구는 찾지 못했습니다.** 결과는
  모델의 예측으로 다뤄야 합니다.

### 13.3 항법 구조 — 가치와 공간이 만나는 곳

- **머리 방향과 목표는 중심복합체입니다 — 근거 강함.** 타원체의 활동 덩어리가 시각
  지표와 자기 운동으로 머리 방향을 추적하고
  ([Seelig & Jayaraman 2015, Nature](https://doi.org/10.1038/nature14446)), 부채꼴체 FC2 뉴런이 목표
  방향을, PFL3 뉴런이 "현재 방향 대 목표"를 좌우 회전 명령으로 바꿉니다
  ([Mussells Pires et al. 2024, Nature](https://doi.org/10.1038/s41586-023-07006-3)).
- **버섯체 출력은 부채꼴체와 전운동 영역으로 갑니다 — 해부 보통, 기능 추측.** 학습된
  가치가 목표를 직접 쓰는 규칙은 아직 밝혀지지 않았습니다.
- **밀폐된 방에 적용하면.** 바람이 없으니 부채꼴체의 바람 기반 목표 경로에 방향 정보가
  없고, 장애물 뒤에서는 냄새 좌우차가 사라집니다. 둘 다 **시각 나침반 + 목표 기억**이
  있어야 해결됩니다.

### 13.4 결론

1. **"버섯체에 시각 연결"은 색·밝기의 의미를 배우는 데만 근거가 있습니다.** 설계는
   문헌 그대로 — 케니언 세포의 약 8%를 시각용으로 두고, 각 세포가 넓은 수용장의 색 채널
   몇 개를 무작위로 받게 하고, 냄새와 **같은** 접근·회피 구획과 도파민을 공유합니다.
   첫 실험은 Vogt 2014 재현(밝기 무작위 대조군, 시각 케니언 세포 차단 대조군 포함).
2. **방 실험은 버섯체 시각 연결 없이도 성립하고, 그것만으로는 성립하지 않습니다.** 필요한
   것은 ① 장애물을 막힌 경계로 두고 흡수원이 있는 이류–확산 냄새장(바람 조건이 행동 근거가
   가장 강함), ② 바람 거슬러 가기·국소 탐색·벽 따라가기 같은 실제 전략, ③ 장애물 뒤를
   돌아가기 위한 시각 나침반과 목표 기억, ④ 장애물 반사입니다. 버섯체 시각은 "이 색 표지가
   먹이를 뜻한다"를 과제에 넣을 때만 방 실험에 들어갑니다.

---

## 14. 실험 과제용 자극과 반복 프로토콜 — 문헌 조사 (2026-09-17)

사용자 요청: "실험 과제를 위해 다른 자극들이 있는지 파악하고, 다른 사람들이 어떻게 했는지 확인. 실험은
여러 번 자동으로 반복하면서 행동이 어떻게 바뀌는지 보고서를 쓸 수 있어야 함." 에이전트 두 개가 초록·본문을
열어 확인한 것만 남겼습니다(원문 사본: 세션 scratchpad `lit/`). 이 조사를 바탕으로 샌드박스에 A/B 실험
세트, 배경 반복 실행기, 다시 보기, 탐구보고서 생성기를 만들었습니다(PLAN.md E절).

### 14.1 지금 샌드박스에 없는 자극

| 자극 | 실제 초파리 | 학습 강화물로 | 반복하면 보이는 것 | 이 모델에 넣기 |
|---|---|---|---|---|
| **쓴맛** (퀴닌·카페인·DEET) | 피함. 설탕에 섞으면 먹기와 단맛 신경 반응이 줄어듦 ([Jeong 2013](https://doi.org/10.1016/j.neuron.2013.06.025)) | 됨. 1 M 설탕 + 0.6% DEET와 짝지은 냄새는 **직후엔 피하고 30분 뒤엔 다가감** — 짧은 혐오 기억과 긴 보상 기억이 함께 생김 ([Das 2014](https://doi.org/10.1016/j.cub.2014.05.078)). 0.4% DEET ≈ 70 V 전기 | 시간에 따라 선호가 뒤집힘 | 쉬움: 설탕 방울의 첨가물 하나 + 처벌 도파민. 배고프면 쓴맛에 둔해짐([Inagaki 2014](https://doi.org/10.1016/j.neuron.2014.09.032)) |
| **소금** | 1–100 mM 좋아함(50 mM 최고), 200 mM 이상 거부 ([Zhang 2013](https://doi.org/10.1126/science.1234133)) | 확인 못 함 | 농도-반응 곡선 | 쉬움 |
| **물 (목마름)** | 목마르면 습한 쪽으로, 물을 먹었으면 피함 | 됨. 16시간 물 없이 둔 뒤 냄새 + 물, 기억 24시간 이상 ([Lin 2014](https://doi.org/10.1038/nn.3827)) | 목마름 상태에 따라 행동이 뒤집힘 | 보통: 배고픔과 같은 상태 변수 하나 더 |
| **온도** | 24–27 °C 선호; 25 °C 대 시험 온도 두 칸 선택 지수 ([Gallio 2011](https://doi.org/10.1016/j.cell.2011.01.028), [Hamada 2008](https://doi.org/10.1038/nature07001)) | 됨. 뜨거운 반쪽(40 °C) 피하기(heat box) — 4분 훈련 PI 0.60/시험 0.35, 12분 0.85/0.56 ([Putz & Heisenberg 2002](https://doi.org/10.1101/lm.50402)). 뜨거운 바닥의 시원한 칸 찾기는 10회 시행에 걸쳐 찾는 시간이 거의 절반 ([Ofstad 2011](https://doi.org/10.1038/nature10131)), 바닥 단서로도 3회 안에 줄어듦 ([Foucaud 2010](https://doi.org/10.1371/journal.pone.0015231)) | **시행마다 찾는 시간이 줄어듦 — 학생 그래프로 가장 좋음** | 보통: 전기 구역과 비슷한 온도 구역. 단 heat box·장소 기억은 실제로 버섯체가 필요 없음(보고서에 적어야 함) |
| **CO2** | 0.1% 이상 피함 ([Suh 2004](https://doi.org/10.1038/nature02980)) | 확인 못 함 | 굶거나 식초가 있으면 회피가 줄고 버섯체가 필요해짐 ([Bräcker 2013](https://doi.org/10.1016/j.cub.2013.05.029), [Lewis 2015](https://doi.org/10.1016/j.cub.2015.07.015)); 오래 맡으면 습관화 ([Das 2011](https://doi.org/10.1073/pnas.1106411108)) | 보통: 냄새 차원 추가(케니언 세포 부호가 바뀌므로 기존 방 재검증) |
| **빛** | UV 쪽으로 가는 힘이 초록보다 약 10배 ([Gao 2008](https://doi.org/10.1016/j.neuron.2008.08.010)); **날지 못하면 어두운 쪽을 고름** ([Gorostiza 2016](https://doi.org/10.1098/rsob.160229)) | 색은 단서로 됨(아래) | — | 어려움: 겹눈에 UV 채널이 없음 |
| **그림자·다가오는 물체** | 지나가는 그림자에 빨라지거나 멈춤; 1초 간격 2–10번이면 반응이 커지고 10–20초 간격 약 20번이면 습관화; 먹이로 돌아가는 시간이 늦어짐 ([Gibson 2015](https://doi.org/10.1016/j.cub.2015.03.058), [Zacarias 2018](https://doi.org/10.1038/s41467-018-05875-1)) | 확인 못 함 | **반복 횟수·간격에 따라 커지거나 줄어듦** | 보통: "각성" 변수 하나와 멈춤/달리기 반응. 방 크기(100 mm)가 실험과 같음 |
| **바람 + 냄새** | 냄새가 오면 바람을 거슬러 달리고(최고 4.4초), 끊기면 수십 초 국소 탐색 ([Álvarez-Salvado 2018](https://doi.org/10.7554/eLife.37815)); Flywalk는 90초마다 냄새를 8시간 반복 ([Steck 2012](https://doi.org/10.1038/srep00361)) | 확인 못 함 | 처음 몇 번 뒤로 안정 — **학습 없는 대조군으로 좋음** | 어려움: 방향 있는 냄새장(PLAN.md B-1) 필요 |
| **습도** | 약 70% 선호, 목마름에 따라 달라짐 ([Enjin 2016](https://doi.org/10.1016/j.cub.2016.03.049), [Knecht 2016](https://doi.org/10.7554/eLife.17879)) | 확인 못 함 | — | 보통 |
| **빛으로 켜는 가짜 보상·벌 구역** | 쓴맛·CO2 신경을 켜면 피하고, 단맛 신경을 켜면 그 자리를 맴돔 ([Aso 2014](https://doi.org/10.7554/eLife.04580), [Corfas 2019](https://doi.org/10.1016/j.cub.2019.03.004)) | 됨. 도파민 뉴런을 켜며 냄새를 주면 기억이 생김 ([Aso & Rubin 2016](https://doi.org/10.7554/eLife.16135), [Claridge-Chang 2009](https://doi.org/10.1016/j.cell.2009.08.034)) | — | 쉬움: 모델의 도파민 입력에 바로 대응 |
| **에탄올 증기** | 취함 | 됨. **훈련 30분 뒤엔 피하고 24시간 뒤엔 좋아함**(12–15시간에 바뀜) ([Kaun 2011](https://doi.org/10.1038/nn.2805)) | 시간에 따라 뒤집힘 | 어려움 |
| **진동** | 쉬는 파리를 깨움; 시간대와 수면 부족에 따라 반응이 달라짐 ([Faville 2015](https://doi.org/10.1038/srep08454)) | 아님 | — | 보통 |
| **자기장** — **논쟁 중** | 회피와 설탕 학습 보고 ([Gegear 2008](https://doi.org/10.1038/nature07183)) | — | — | **넣지 않음**: 미로 파리 97,658마리로 재현 실패 ([Bassetto 2023](https://doi.org/10.1038/s41586-023-06397-7)) |

에이전트가 요청문의 인용 네 곳을 바로잡았습니다. Masek & Scott 2010은 쓴맛이 아니라 적외선 레이저 열로
벌을 줬습니다. Das 2014의 쓴맛은 DEET입니다. 음식 냄새가 버섯체를 통해 CO2 회피를 **줄인다**는 것은
Bräcker 2013이 아니라 Lewis 2015입니다. Fenckova 2019는 Biol Psychiatry입니다.

### 14.2 반복하면 행동이 바뀌는 프로토콜

| 프로토콜 | 설계 | 바뀌는 것 | 근거 |
|---|---|---|---|
| 냄새 + 전기 T자 미로 | CS+ 60초 동안 60 V 1.25초 전기 12번, CS− 60초, 120초 선택 | 약 95%가 CS+를 피함. 전기 횟수가 늘면 커지다가 한 번의 훈련 안에서 포화. 7시간에 걸쳐 줄고 24시간 뒤에도 남음 | [Tully & Quinn 1985](https://doi.org/10.1007/BF01350033) |
| 한 번씩 따로 주는 훈련 | 냄새 10초 + 8초에 전기 한 번 | 점수(×100) 1회 33, 붙여서 2회 34, **15분 간격 2회 51** | [Beck 2000](https://doi.org/10.1523/JNEUROSCI.20-08-02944.2000) |
| 나눠서 대 몰아서 | 10회, 15분 쉬기 대 쉬지 않기 | 몰아서: 4일 안에 사라짐. 나눠서: 단백질 합성이 필요한 기억이 7일간 줄지 않음 | [Tully 1994](https://doi.org/10.1016/0092-8674(94)90398-0) |
| 냄새 + 설탕 | 16–20시간 굶김, 2분씩 | 한 번으로 1–36시간 거의 줄지 않음. 먹이면 행동에서 사라지고 다시 굶기면 돌아옴. 아라비노스는 직후만, 아라비노스+소르비톨 ≈ 자당 | [Krashes & Waddell 2008](https://doi.org/10.1523/JNEUROSCI.5333-07.2008), [Krashes 2009](https://doi.org/10.1016/j.cell.2009.08.035), [Burke & Waddell 2011](https://doi.org/10.1016/j.cub.2011.03.032) |
| 소거 | CS+/CS− 반복을 전기 없이 | 5번은 효과 없음, 15–30번에 회피가 약 절반 | [Qin & Dubnau 2010](https://doi.org/10.1111/j.1601-183X.2009.00548.x), [Felsenberg 2018](https://doi.org/10.1016/j.cell.2018.08.021) |
| 색 학습 | 파랑·초록 60초씩, 설탕 또는 전기 | 설탕: 1·2·4·8번 모두 기억, 늘수록 커짐. 훈련 안 한 파리는 초록을 약간 선호. 전기: 15 V부터, 30–120 V 비슷, 한 번이면 충분. 한 마리 점수 ≈ 무리 점수 | [Schnaitmann 2010](https://doi.org/10.3389/fnbeh.2010.00010), [Vogt 2014](https://doi.org/10.7554/eLife.02395) |
| 습관화·민감화 | 발에 설탕 반복(PER), 냄새 점프 반사, 새 냄새 반복 | PER이 10분 이상 억제, 입에 설탕을 주면 다시 민감해짐. 버섯체 α′3 출력은 2번째에 −50%, 3–7번째 −80%, 1시간에 회복 | [Duerr & Quinn 1982](https://doi.org/10.1073/pnas.79.11.3646), [Asztalos 2007](https://doi.org/10.1080/01677060701247508), [Hattori 2017](https://doi.org/10.1016/j.cell.2017.04.028) |
| 한 시간 안의 포만 | 단맛 대 영양 선택, 효모 대 설탕 | 처음엔 맛으로 고르다가 몇 시간에 걸쳐 영양 쪽으로; 아라비노스 한 모금은 국소 탐색을 일으키고 소르비톨은 아님 | [Dus 2011](https://doi.org/10.1073/pnas.1017096108), [Stafford 2012](https://doi.org/10.1523/JNEUROSCI.1887-12.2012), [Murata 2017](https://doi.org/10.1242/jeb.161646) |

### 14.3 학교와 교육용 소프트웨어는 어떻게 했나

- **학교 실험.** AP Biology 12번 실험은 양 끝에 물질을 둔 선택 상자에 초파리 20–40마리를 넣고 5분 뒤
  마릿수를 셉니다. 먼저 양쪽에 물만 둬서 치우침을 확인하고, 물질 위치를 바꿔 다시 합니다
  ([College Board](https://secure-media.collegeboard.org/digitalServices/pdf/ap/bio-manual/Bio_Lab12-FruitFlyBehavior.pdf)).
  초보자용 유충 학습 실습서는 짝지은 냄새를 바꾼 두 무리의 차이를 둘로 나눠 타고난 선호를 지웁니다
  ([Michels 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5395560/)). 한국 학생의 초파리 **행동** 탐구 사례는
  찾지 못했습니다.
- **평가 기준.** KOSAC의 2022 교육과정 탐구 평가 틀은 변인 명시, 효과의 방향("A가 ~할수록 B가 ~하다"),
  통제 증거, 단위가 있는 표, 축 이름과 단위가 있는 그래프(조작 변인이 x축), 데이터와 맞는 결론, 가설이
  틀렸으면 오차 원인·맞았으면 새 질문을 봅니다. 중학교 기준은 **반복 측정의 평균**, 평균과 비율, 이상값
  지적이고 표준편차와 유의성은 고등학교입니다
  ([KOSAC 2026](https://cdn.kosac.re.kr/files/cms/attach/202604/577b15647756492693359ef184506ad5_1775634971221.pdf)).
  전국과학전람회는 도구·생성형 AI 사용을 밝히게 합니다
  ([72회 요강](https://www.kess64.net/bbs/notice/1220/download/924)).
- **소프트웨어.** NetLogo BehaviorSpace는 변수 조합마다 반복 횟수를 정하고 시드를 실행 번호에 묶어 표로
  냅니다. larvaworld는 유충 가상 실험실로 프리셋·반복 실행·선호 지수를 제공합니다. heat box는 30초 사전
  시험 → 훈련 → 3분 시험을 15개 방에서 동시에 돌리고, 한 칸 길이도 안 움직인 파리는 제외합니다
  ([Putz & Heisenberg 2002](https://pmc.ncbi.nlm.nih.gov/articles/PMC187128/)).

### 14.4 이 조사가 샌드박스 설계에 준 것

1. **실험은 A/B 한 쌍**으로 묶었습니다 — 한 가지만 다르게 한 두 조건, 같은 번호의 파리는 같은 시드
   (거울 대조와 같은 목적: 차이가 조작 때문인지 가림).
2. **판정은 훈련 뒤 시험 시행에서** 합니다(heat box·T자 미로처럼 강화 없는 시험).
3. 보고서는 KOSAC 틀을 따르고, 가설 판정·그래프·표는 자동으로, 동기·이유·결론은 **학생이 쓰는 칸**으로
   둡니다. 통계는 평균·범위·쌍의 방향 개수까지만, 부호 검정은 "동전 던지기 확률"로 풀어 씁니다.
4. 다음에 넣을 만한 자극(추천 순): **온도 구역**(시원한 칸 찾기 — 시행마다 줄어드는 그래프),
   **쓴맛 첨가물**(시간에 따라 뒤집히는 선호), **반복 그림자**(각성의 누적과 습관화), **빛으로 켜는
   도파민 구역**. 자기장은 넣지 않습니다.
