# two-robot-transport-twin

**MPC 로 푼 두 대 협력 운송 궤적을, Isaac Sim 안에서 학습된 RL 정책으로 검증하는 리그.**

실기로 바로 시험하면 시행착오 비용이 크다. 그래서 관제 PC 가 실제로 쓰는 MPC 플래너와
로봇에 올릴 정책을 **같은 코드 그대로** 시뮬로 옮겨, 추종 오차·통행 가능성·결합 구속을
정량으로 먼저 재는 것이 이 저장소의 목적이다.

```
   [1] MPC 플래너                    [2] 목표 변환             [3] Isaac Sim 리그
   two_robot_nlp                     plan_to_target            runner/episode
   A* 시드 -> IPOPT NLP              z / pitch / yaw 보정      로봇 2 대 + 압출한 실제 맵
   물체·차체·EE 궤적                 속도 피드포워드           30 Hz 정책 추론
        |  CSV (t,x,y,yaw)                 |                        |
        +---------------------------------->------------------------+
                                                                    v
                                                    EE 오차 / 차체 편차 / 클리어런스
```

두 로봇은 **같은 정책**을 쓴다. 정책은 관측이 전부 차체 프레임이라 로봇을 구분하지 않는다.

---

## 1. 리포지토리 구성

| 경로 | 내용 |
|---|---|
| [`planner/`](planner/) | MPC 플래너 ROS 2 패키지 (`mobile_manipulator_trajectory`). A\* 시드 + CasADi/IPOPT NLP, RViz 관제 노드 |
| [`twin/`](twin/) | Isaac Sim 검증 리그. 씬 구성, 계획 보간, 목표 변환, 정책 재생, 정합성 게이트 |
| [`source/ee_track_ppo/`](source/ee_track_ppo/) | Isaac Lab task 패키지. 리그가 액션 항·가감속 램프·액션 지연을 **학습과 같은 코드로** 받기 위해 필요 |
| [`handoff/traj_v29/`](handoff/traj_v29/) | 채택 정책 배포본 (`policy.pt`, `policy.onnx`, `ee_obs.py`) + 판정 기록 |
| `handoff/traj_v28/`, `traj_v12/` | 비교용 정책 (차체 편차 ↔ EE 오차 교환관계를 볼 때) |
| [`handoff/map_munji_3f_2025_wide/`](handoff/map_munji_3f_2025_wide/) | 2D SLAM 맵 (`.pgm` / `.yaml`). 플래너·`map_server`·압출이 모두 이 파일을 쓴다 |
| `data/plans.npz` | 실기 MPC 주행 122 개. 뒤 24 개가 학습 held-out — 리그 종단 검증에 쓴다 |

### 문서

| 문서 | 누구를 위해 |
|---|---|
| **[`docs/STUDENT_GUIDE_KR.md`](docs/STUDENT_GUIDE_KR.md)** | **처음 쓰는 사람.** 설치 → MPC 경로 생성 → Isaac 재생 → 결과 읽기, 커맨드 전부 |
| [`docs/BASE_TRACKING_KR.md`](docs/BASE_TRACKING_KR.md) | 차체 추종을 보상에 넣으면 왜 무너지는가 — 여섯 판의 실측, 원인, 다음 시도 |
| [`twin/README_KR.md`](twin/README_KR.md) | 개발 기록. 무엇을 왜 그렇게 했는지, 잡은 버그, 미해결 항목 |

**처음이면 [학생용 실행 가이드](docs/STUDENT_GUIDE_KR.md) 부터 읽으면 된다.**
아래 §3~§5 는 이미 환경이 갖춰진 사람을 위한 요약이다.

---

## 2. ⚠ 이 저장소에 **없는** 것

| 없는 것 | 왜 | 어떻게 |
|---|---|---|
| **로봇 USD / 메시 에셋** (약 420 MB) | 사내 로봇 모델이고 GitHub 파일 한도를 넘는다 | 별도 제공. `source/ee_track_ppo/ee_track_ppo/assets/data/` 에 풀어 넣는다 |
| 압출한 3D 맵 USD | 생성물 | `twin/world/map_extrude.py` 로 만든다 (§5.1) |
| MPC 계획 CSV | 생성물 | `twin/tools/solve_plan.py` 로 만든다 (§5.2) |
| 학습 체크포인트 (`model_*.pt`), 학습 로그 | 재개용이라 리그에 불필요 | 배포본 `policy.pt` / `policy.onnx` 만 포함 |

로봇 에셋이 없으면 씬이 뜨지 않는다. `scene_builder` 가 `FileNotFoundError: 페이로드 USD 가 없다`
로 **명시적으로** 죽으므로 조용히 틀리지는 않는다.

`twin/configs/policy_*.yaml` 의 `checkpoint.run:` 은 학습 런 경로라 이 저장소에 없다.
리그는 `observation.module` 과 `checkpoint.jit` 만 읽으므로 문제되지 않는다.

---

## 3. 사전 준비

개발·검증 환경:

```
Isaac Sim 5.1.0  +  Isaac Lab 2.3.0
ROS 2 Jazzy  (시스템 python 3.12)
NVIDIA GPU (RTX)
```

### 파이썬이 세 개다 — 이 리그에서 제일 자주 넘어지는 지점

| 용도 | 인터프리터 | 진입점 |
|---|---|---|
| Isaac 계열 (재생·게이트·맵압출) | conda env, python 3.11 | **`twin/run.sh <스크립트>`** |
| MPC 풀기 (헤드리스) | casadi 가 있는 아무 python | 직접 호출 |
| ROS 2 노드 (플래너·다리) | 시스템 python 3.12 (rclpy) | `twin/ros/*.sh` |

Isaac Lab conda env 는 python 3.11, ROS 2 Jazzy 는 3.12 라 **`rclpy` 를 Isaac 프로세스
안으로 import 할 수 없다.** 그래서 라이브 모드는 공유메모리 다리(`twin/ros/twin_bridge.py`)로
두 프로세스를 잇는다.

기계마다 다른 값은 환경변수로 덮는다:

```bash
export TWIN_CONDA_ROOT=$HOME/miniconda3   # conda 설치 위치
export TWIN_CONDA_ENV=ppo_afl             # Isaac Lab conda env 이름
export TWIN_ROS_DISTRO=jazzy              # ROS 2 배포판
export TWIN_ROS_WS=$HOME/ros2_ws          # colcon 워크스페이스 (플래너 빌드용)
```

---

## 4. 설치

```bash
git clone <this-repo> two-robot-transport-twin
cd two-robot-transport-twin

# 1) 로봇 에셋을 풀어 넣는다 (별도 제공)
#    -> source/ee_track_ppo/ee_track_ppo/assets/data/amr_fr3_payload/usd/... 등

# 2) Isaac Lab task 패키지 설치 (Isaac Lab conda env 에서)
twin/run.sh -m pip install -e source/ee_track_ppo

# 3) 플래너를 ROS 2 워크스페이스에 빌드 (라이브 모드에만 필요)
ln -s "$PWD/planner" "$TWIN_ROS_WS/src/mobile_manipulator_trajectory"
cd "$TWIN_ROS_WS" && colcon build --packages-select mobile_manipulator_trajectory
```

헤드리스로 계획만 풀 때는 **3번이 필요 없다.** `solve_plan.py` 가 `planner/` 를
직접 import 한다 (`solve_transport` 는 순수 CasADi/numpy).

> **Kit 이 stdout 을 가로챈다.** 판정이 필요한 스크립트는 전부 `--report <파일>` 을 받는다.
> 안 붙이면 결과가 Isaac 로그에 묻힌다.

---

## 5. 사용법

### 5.1 맵 압출 (최초 1회)

2D SLAM 맵을 높이 2 m 로 세워 정적 콜라이더가 있는 USD 로 만든다. 동시에
**margin 별 통행 가능성**을 분석한다.

```bash
twin/run.sh twin/world/map_extrude.py --report /tmp/map.txt      # 분석 + USD
twin/run.sh twin/world/map_extrude.py --no-usd --report /tmp/map.txt   # 분석만
```

산출: `twin/world/assets/munji_3f_2025_wide.usd`

### 5.2 MPC 경로 만들기

#### A. 헤드리스 (재현성·배치용, 권장)

```bash
SOLVER_PY=<casadi 있는 python>     # 개발 기계에서는 ros2 conda env

$SOLVER_PY twin/tools/solve_plan.py \
    --obj-len 1.50 --margins 0.18 --obj-margin 0.05 \
    --target-dist 6.0 --tag mytest \
    --report /tmp/solve.txt
```

- 출력: `twin/data/plans/L{obj_len×100}_base{margin×100}_obj{obj_margin×100}_{tag}/`
  → 위 예시는 `L150_base018_obj005_mytest/`
- 파일 5개: `object.csv`, `base_A.csv`, `base_B.csv`, `ee_A.csv`, `ee_B.csv` (`t,x,y,yaw`)
- 시작·목표 직접 지정: `--start X Y YAW --goal X Y YAW` (없으면 `--target-dist` 길이로 자동)
- margin 비교: `--margins 0.05 0.18 0.30`

**실패율을 재려면** (한 경로만 보고 판단하지 않기 위해):

```bash
$SOLVER_PY twin/tools/solve_batch.py --n 6 --margin 0.18 --obj-len 1.5 --dist 6.0 \
    --report /tmp/batch.txt
```
성립 / 미수렴 / 불연속(순간이동) / 충돌 을 센다.

#### B. RViz 에서 직접 찍기

```bash
twin/ros/run_planner.sh
```

1. **2D Pose Estimate** → 물체 시작 자세
2. **2D Goal Pose** → 물체 목표 자세 (여기서 풀기 시작, 수십 초 ~ 수 분)
3. 애니메이션으로 계획 확인
4. **INITIAL** → 시작 자세 유지 / **EXECUTE** → 재생

계획 CSV 는 `twin/data/plans/live/` 에 떨어진다. 바꾸려면:

```bash
CSV=$PWD/twin/data/plans/mycase twin/ros/run_planner.sh
MAP=/path/to/other.yaml         twin/ros/run_planner.sh
```

### 5.3 정책 재생 — 오프라인 (주력)

**먼저 리그 종단 검증부터.** held-out 궤적이라 EE 중앙 **9.5 mm 근처**가 나와야 정상이다.
안 나오면 씬·관측·액션·보간 중 하나가 깨진 것이니 그것부터 잡는다.

```bash
twin/run.sh twin/runner/run_episode.py \
    --track-a npz:98 --track-b npz:110 \
    --out twin/data/runs/pair_98_110 \
    --report /tmp/ep.txt
```

MPC 계획으로 돌리기:

```bash
twin/run.sh twin/runner/run_episode.py \
    --track-a csv:twin/data/plans/L150_base018_obj005_mytest:a \
    --track-b csv:twin/data/plans/L150_base018_obj005_mytest:b \
    --out twin/data/runs/mytest_v29 --report /tmp/ep.txt
```

눈으로 보기:

```bash
twin/run.sh twin/runner/run_episode.py --track-a npz:98 --track-b npz:110 \
    --gui --real-time --speed 2 --max-sep 8 \
    --out twin/data/runs/pair_gui --report /tmp/ep.txt
```

궤적 지정 문법:

```
npz:<i>            data/plans.npz 의 i 번 (뒤 24 개가 held-out)
npz:<path>:<i>     다른 npz
csv:<dir>:<a|b>    MPC 계획 디렉토리에서 한쪽
```

자주 쓰는 옵션:

| 옵션 | 용도 |
|---|---|
| `--policy-cfg twin/configs/policy_v28.yaml` | 정책 교체 (v12 / v28 / v29) |
| `--localization ground_truth` \| `amcl` | 위치를 참값으로 / 학습과 같은 AMCL 모델로 |
| `--no-map` | 빈 평지. 맵이 원인인지 가를 때 |
| `--ff-mode zero` | 속도 피드포워드 채널 끄기 |
| `--caster-damping 0.5` | 캐스터 자전 가설 검증 |
| `--bar 0` | 두 EE 사이 시각용 막대 끄기 |
| `--hold-only` | 시작 자세 유지만. 수렴값만 볼 때 |
| `--spawn-a "x,y,yaw"` | 배치 자리 직접 지정 (기본은 맵에서 자동 선택) |

### 5.4 정책 재생 — 라이브 (RViz 로 찍으면 Isaac 이 바로 따라감)

터미널 3개:

```bash
twin/ros/run_planner.sh                      # T1  MPC + map_server + RViz
twin/ros/run_bridge.sh                       # T2  ROS ↔ Isaac 공유메모리 다리
twin/run.sh twin/runner/live_sim.py --gui    # T3  Isaac Sim
```

RViz 에서 **2D Pose Estimate → 2D Goal Pose → INITIAL → EXECUTE**.
`INITIAL` 에서 Isaac 이 로봇 두 대를 계획 t=0 자세로 스폰하고, 팔이 첫 파지점으로 붙는다.

상태는 `park`(맵 밖 대기) → `hold`(시작 자세 유지) → `run`. SETUP 전에 두 대를 원점에
올리면 서로 겹쳐 밀어내므로 스폰 시점을 사람이 버튼으로 정한다.

### 5.5 정합성 게이트

정책이나 관측을 바꿨으면 **재생 전에 이걸 먼저** 돌린다.

```bash
# G1  상태 -> 관측 (ee_obs.py vs 학습 ObsTerm)          통과 기준 <= 1e-6
twin/run.sh twin/tools/check_obs_parity.py --amcl on \
    --policy handoff/traj_v29/policy.pt --report /tmp/g1.txt

# G2  계획 보간 (twin vs 플래너 노드 _sample_step)      통과 기준 오차 0
python3 twin/tools/check_plan_sampler.py

# G3  계획 -> EE 목표 (twin vs 학습 커맨드 term)        pose 2e-6 / rot6d 5e-7
twin/run.sh twin/tools/check_plan_target.py --report /tmp/g3.txt

# 씬 스모크 (스폰·낙하·자전)
twin/run.sh twin/tools/check_scene.py --report /tmp/scene.txt
```

G2 는 Isaac 이 필요 없어서 시스템 python 으로 돈다.

### 5.6 결과

```
twin/data/runs/<이름>/
  log.npz     30 Hz 전 채널 (ee / base / twist / jpos / jvel / act / tgt / base_ref / ee_dist / robot_dist ...)
  meta.json   정책 md5, 관측 md5, yaw_offset, 궤적, 맵, localization, step_dt ...
```

지표 표(EE 중앙/95%, 자세, 차체 편차)는 실행할 때 `--report` 파일에 나온다.
`log.npz` 는 사후 분석용 원시 로그다.

---

## 6. 계약 — 여기서 어긋나면 조용히 틀린다

정책 v29 기준. 값의 유일한 출처는 [`twin/configs/policy_v29.yaml`](twin/configs/policy_v29.yaml) 이다.

| 항목 | 값 |
|---|---|
| 관측 | **52 차원**, 전부 차체 프레임 (v12~v28 은 58 차원) |
| 액션 | 8 = 팔 6 (`scale 0.5`, default offset) + 차체 (v, ω) |
| 목표 요 보정 | **−π/2** |
| EE 목표 높이 / 피치 | 0.55 m (학습 0.50~0.60) / 정확히 π |
| 제어 주기 | 30 Hz (sim 1/120, decimation 4) |

기대치 (held-out 24, 리그가 이 근처를 재현해야 한다):

```
EE 중앙 9.52 mm | EE 95% 17.05 mm | 자세 중앙 1.72 deg | 차체편차 중앙 0.501 m | 3cm 통과 88%
```

> ⚠ **`PLAN_YAW_OFFSET` 을 task 패키지에서 import 하지 말 것.** 전역 상수는 `+π/2`(v30용)
> 이고 v29 는 `−π/2` 로 학습됐다. import 하면 에러 없이 180° 틀어진 목표를 준다.
> 값은 config 에서만 읽는다.

> ⚠ **정책 출력을 clamp 하지 말 것.** 학습의 `JointPositionAction` 에는 clamp 가 없고
> 관측의 `last_action` 은 raw 값이다. `clamp(-1,1)` 을 넣었다가 EE 오차가 275 mm 로
> 발산한 적이 있다 (팔 액션의 12~20 % 가 |a| > 0.98).

---

## 7. 알려진 제약

- **`base_margin ≥ 0.70` 은 이 맵에서 성립하지 않는다.** v29 의 차체 편차(중앙 0.50 m)를
  플래너 여유로 흡수하려면 0.70 이 필요한데, 그러면 주행 가능 영역이 21 조각으로 끊겨
  방 사이를 갈 수 없다. IPOPT 로도 확인됐다 (0.50 / 0.70 미수렴). 실제로 푸는 값은 **0.18**.
- **한 판에 갈 수 있는 거리는 8.3 m.** v29 권장 속도(`obj_v_max 0.12`)와 기본
  `max_steps 700` 의 곱이다. 건물을 가로지르려면 계획을 구간으로 쪼개야 한다.
- **협력 운송 편대는 v29 학습 분포 밖이다.** 한 대가 주행의 86 % 를 후진해야 하는데
  v29 는 후진 벌점을 걸고 학습됐다. 지금 리그의 기본 구성은 **두 대가 각자 자기 궤적을
  추종**하는 것이고, 편대는 재학습 항목이다. 상세는 [`twin/README_KR.md`](twin/README_KR.md).
- **명령 0 인데 차체가 초당 3.6° 자전한다.** 캐스터가 `damping 0` 이라 그렇고,
  학습 env 에서도 같은 폭으로 돈다 (리그 버그 아님). `--caster-damping` 으로 시험할 수 있다.
- **`set -u` 를 켜지 말 것.** `setup_conda_env.sh` 가 `$ZSH_VERSION` 을 언바운드 참조해
  그 자리에서 죽는다.

---

## 8. 진행 상태

```
Phase 0   완료      정합성 게이트 G1 / G2 / G3 통과
Phase 1   완료      두 대 동시 재생, 리그 종단 검증 통과 (EE 중앙 12.5~13.7 mm)
Phase 2   대기      막대 결합 B1(스프링댐퍼) -> B2(강체)
Phase 3   대기      로컬라이제이션 현실화 (실기 AMCL 로그 기반)
Phase 4   진행 중   ROS 2 통합 (라이브 모드)
Phase 5   대기      스윕 / 회귀
```
