# 학생용 실행 가이드

MPC 로 경로를 만드는 것부터, Isaac Sim 에 로봇 두 대를 띄워 학습된 정책으로
추종시키고 결과를 읽는 것까지 — 커맨드 전부.

처음이면 **§0 → §1 → §2 → §3 → §4 순서대로** 따라가면 된다. §3 까지만 해도
"MPC 가 경로를 푼다" 는 확인이 되고, §4 부터 Isaac 이 필요하다.

---

## 0. 이 프로젝트는 무엇인가

로봇 두 대가 긴 물체(막대)를 같이 옮기는 상황을 다룬다. 세 층이다.

```
 [1] MPC 플래너                  물체를 A 에서 B 로 옮기려면
     planner/                    두 로봇의 차체와 손이 각각 어디에 있어야 하는가
     A* 시드 -> IPOPT NLP        -> 궤적 CSV (t, x, y, yaw)
          |
          v
 [2] 목표 변환                   플래너의 2D 셋포인트를 정책이 먹는 형태로
     twin/plan/                  z / pitch / yaw 보정 + 속도 피드포워드
          |
          v
 [3] RL 정책 + Isaac Sim         손이 그 목표를 실제로 따라갈 수 있는가
     twin/runner/                30 Hz 추론 -> 팔 관절목표 + 차체 (v, ω)
```

**왜 시뮬인가.** 실기로 바로 시험하면 한 번 돌릴 때마다 로봇 두 대를 세팅하고
사람이 붙어야 한다. 시뮬에서 먼저 "이 계획은 애초에 못 따라간다" 를 걸러내면
실기 시간을 아낄 수 있다.

**두 로봇은 같은 정책을 쓴다.** 정책의 관측이 전부 차체 프레임이라 A 인지 B 인지
구분하지 않는다.

---

## 1. 준비 — 파이썬이 세 개다

이 프로젝트에서 **가장 많이 넘어지는 지점**이다. 처음부터 이해하고 가자.

| 무엇을 | 어떤 파이썬 | 왜 | 어떻게 부르나 |
|---|---|---|---|
| Isaac Sim 돌리기 | conda env, **3.11** | Isaac Sim 5.1 이 3.11 로 빌드됨 | `twin/run.sh <스크립트>` |
| MPC 풀기 | casadi 있는 아무 python | CasADi 만 있으면 됨. Isaac 무관 | 직접 호출 |
| ROS 2 노드 | 시스템 python **3.12** | Jazzy 가 3.12 로 빌드됨 | `twin/ros/*.sh` |

**3.11 과 3.12 는 서로 import 가 안 된다.** 그래서 Isaac 프로세스 안에서 `rclpy` 를
쓸 수 없고, 라이브 모드는 공유메모리로 두 프로세스를 잇는다.

### 왜 `python` 을 직접 치면 안 되나

```bash
python twin/runner/run_episode.py     # ← 이렇게 하면 십중팔구 실패
```

`.bashrc` 가 다른 venv 를 활성화해 두면 그쪽 python 이 먼저 잡힌다. 비대화형 셸에서는
`deactivate` 가 함수로 안 잡혀 해제도 안 된다. 그래서 **래퍼를 쓴다.**

에러 메시지로 무엇을 빠뜨렸는지 구분할 수 있다:

```
ModuleNotFoundError: No module named 'isaacsim'   -> conda activate 가 안 됨
ModuleNotFoundError: No module named 'rsl_rl'     -> PATH 를 안 잡음 (다른 python)
```

### 기계마다 다르면 환경변수로 덮는다

```bash
export TWIN_CONDA_ROOT=$HOME/miniconda3   # conda 설치 위치
export TWIN_CONDA_ENV=ppo_afl             # Isaac Lab conda env 이름
export TWIN_ROS_DISTRO=jazzy
export TWIN_ROS_WS=$HOME/ros2_ws
```

### `--report` 를 항상 붙일 것

Isaac 의 Kit 앱이 stdout 을 가로챈다. 그래서 판정이 필요한 스크립트는 전부
`--report <파일>` 을 받는다. **안 붙이면 결과가 Isaac 로그 수천 줄에 묻힌다.**

---

## 2. 설치

```bash
git clone <repo> two-robot-transport-twin
cd two-robot-transport-twin
```

### 2-1. 로봇 에셋 넣기 ★ 필수

로봇 USD·메시(약 420 MB)는 저장소에 없다. 별도로 받아서 여기에 푼다:

```
source/ee_track_ppo/ee_track_ppo/assets/data/
├── amr_fr3_payload/usd/amr_fr3_payload.usd    ← 이게 주로 쓰인다
├── amr_fr3/...
└── ...
```

없으면 씬이 `FileNotFoundError: 페이로드 USD 가 없다` 로 **명시적으로** 죽는다.
조용히 이상하게 도는 일은 없으니 안심해도 된다.

### 2-2. task 패키지 설치

```bash
twin/run.sh -m pip install -e source/ee_track_ppo
```

이게 왜 필요한가: 리그가 액션 항·가감속 램프·액션 지연을 **학습과 똑같은 코드로**
받기 위해서다. 같은 로직을 두 벌 만들면 언젠가 갈라지고, 그러면 에러 없이 조용히
다른 결과가 나온다.

### 2-3. 설치 확인 — Isaac 없이 도는 게이트부터

```bash
python3 twin/tools/check_plan_sampler.py
```

`G2 통과` 가 나오면 계획 보간 로직이 플래너와 일치한다는 뜻이다. Isaac 이 필요 없어서
설치 직후 바로 돌려볼 수 있다.

### 2-4. (라이브 모드에만) 플래너를 ROS 2 에 빌드

```bash
ln -s "$PWD/planner" "$TWIN_ROS_WS/src/mobile_manipulator_trajectory"
cd "$TWIN_ROS_WS" && colcon build --packages-select mobile_manipulator_trajectory
```

**§3 (헤드리스로 계획 풀기) 만 할 거면 이건 필요 없다.**

---

## 3. MPC 로 경로 만들기

### 3-0. 먼저 맵을 3D 로 세운다 (최초 1회)

2D SLAM 맵(`.pgm`)을 높이 2 m 로 압출해서 Isaac 이 쓸 USD 를 만든다.

```bash
twin/run.sh twin/world/map_extrude.py --report /tmp/map.txt
```

분석만 하려면 (USD 안 만들고, Isaac 도 거의 안 씀):

```bash
twin/run.sh twin/world/map_extrude.py --no-usd --report /tmp/map.txt
```

리포트에서 봐야 할 것:

```
주행 가능 영역   264.7 m^2,  여유거리 중앙 0.57 m

margin 별 통행 가능성 — 필요 여유 = 디스크반경 + margin
  base  0.05   0.356 m   68.4%   177.7 m^2( 2)   연결 양호
  base  0.30   0.606 m   46.8%   102.7 m^2( 5)   분절 주의
  base  0.70   1.006 m   19.8%    15.3 m^2(21)   분절 심함
```

**중요한 개념:** 필요한 여유는 `margin` 이 아니라 **`디스크반경 + margin`** 이다.
플래너는 사각형 차체를 원판으로 덮어서 SDF 와 비교하는데, 그 원판 반경이 0.306 m 다.
`margin` 만 보고 판단하면 30 cm 를 통째로 빠뜨린다.

그리고 문제는 면적이 아니라 **연결성**이다. margin 0.70 이면 면적은 20% 남지만
21 조각으로 끊겨서 방 사이를 갈 수가 없다.

### 3-1. 방법 A — 헤드리스로 풀기 (권장, 재현 가능)

```bash
SOLVER_PY=<casadi 있는 python>      # 예: ~/miniconda3/envs/ros2/bin/python

$SOLVER_PY twin/tools/solve_plan.py \
    --obj-len 1.50 \
    --margins 0.18 \
    --obj-margin 0.05 \
    --target-dist 6.0 \
    --tag mytest \
    --report /tmp/solve.txt
```

**나오는 것:**

```
twin/data/plans/L150_base018_obj005_mytest/
├── object.csv     물체 궤적
├── base_A.csv     로봇 A 차체
├── base_B.csv     로봇 B 차체
├── ee_A.csv       로봇 A 손 (EE)
└── ee_B.csv       로봇 B 손
```

각 CSV 는 `t,x,y,yaw` 네 열이다. 좌표는 **map 프레임**, yaw 는 라디안,
`t` 는 초 (플래너 `dt` 기본 0.15 s 간격).

디렉토리 이름 규칙: `L{obj_len×100}_base{margin×100}_obj{obj_margin×100}_{tag}`

**자주 쓰는 옵션:**

| 옵션 | 뜻 |
|---|---|
| `--start X Y YAW --goal X Y YAW` | 시작·목표를 직접 지정 (없으면 자동 선택) |
| `--target-dist 6.0` | 자동 선택할 때 경로 길이 [m] |
| `--margins 0.05 0.18 0.30` | 여러 margin 을 한 번에 풀어 비교 |
| `--obj-len 1.50` | 막대 길이 [m] |
| `--max-iter 3000` | IPOPT 반복 상한 |
| `--dt 0.15` | 계획 시간 간격 [s] |

**리포트에서 봐야 할 것:**

```
margin   IPOPT s   최소여유   목표오차   동역학잔차   판정
 0.18      197.4   +0.049     0.000     3.2e-05    성립
```

- `목표오차` 가 0 이 아니면 → 목표에 못 닿았다. **해가 아니다.**
- `동역학잔차` 가 크면(1e-2 이상) → 유니사이클로 주행 불가능. **해가 아니다.**
- IPOPT 가 미수렴이면 마지막 반복치를 뱉는데, 그건 계획이 아니라 그냥 숫자다.

### 3-2. 한 판만 보고 판단하지 말 것

경로 하나가 잘 풀렸다고 그 설정이 좋은 게 아니다. 실패율을 재려면:

```bash
$SOLVER_PY twin/tools/solve_batch.py \
    --n 6 --margin 0.18 --obj-len 1.5 --dist 6.0 \
    --report /tmp/batch.txt
```

성립 / 미수렴 / 불연속(순간이동) / 충돌 을 센다.

> 이 저장소는 "표본 하나로 판정" 하는 실수를 이미 두 번 했다. 궤적 하나로 재학습
> 결론을 냈다가 뒤집혔다. **n ≥ 6 으로 돌리는 습관을 들일 것.**

### 3-3. 방법 B — RViz 에서 마우스로 찍기

눈으로 보면서 계획을 만들고 싶을 때.

```bash
twin/ros/run_planner.sh
```

RViz 가 뜨면:

1. **2D Pose Estimate** 버튼 → 지도 위를 드래그 → 물체 **시작** 자세
2. **2D Goal Pose** 버튼 → 드래그 → 물체 **목표** 자세
   → 여기서 풀기 시작한다 (수십 초 ~ 수 분. 터미널에 IPOPT 진행이 뜬다)
3. 다 풀리면 애니메이션으로 계획이 재생된다. 마음에 안 들면 다시 찍으면 된다
4. **INITIAL** 버튼 → 시작 자세를 계속 발행 (로봇이 그 자리로 갈 시간을 준다)
5. **EXECUTE** 버튼 → 계획대로 셋포인트가 흐른다

계획 CSV 는 `twin/data/plans/live/` 에 자동으로 떨어진다.

바꾸고 싶으면:

```bash
CSV=$PWD/twin/data/plans/mycase twin/ros/run_planner.sh
MAP=/path/to/other_map.yaml     twin/ros/run_planner.sh
```

> **플래너 기본값을 그대로 쓰면 안 된다.** `run_planner.sh` 는 이미 우리 로봇에
> 맞춘 프로파일로 실행한다 (`obj_v_max 0.12`, `ee_v_max 0.20` 등). 노드 기본값
> (`obj_v_max 0.60`)을 쓰면 EE 속력이 1.2 m/s 가 되는데, 우리 정책은 0.25 m/s 에서
> 무너진다.
>
> ```
> EE 속력 ≈ obj_v_max + (obj_len / 2) × obj_w_max
> ```

---

## 4. Isaac Sim 에서 정책 돌리기 — 오프라인

### 4-1. ★ 제일 먼저: 리그가 맞는지부터 확인

**MPC 계획으로 바로 가지 말 것.** 먼저 "학습 때 쓴 것과 같은 궤적" 으로 돌려서
리그 자체가 정상인지 확인한다.

```bash
twin/run.sh twin/runner/run_episode.py \
    --track-a npz:98 --track-b npz:110 \
    --out twin/data/runs/pair_98_110 \
    --report /tmp/ep.txt
```

`data/plans.npz` 의 뒤 24 개는 **학습에서 뺀 held-out** 이고, 정책 v29 가 거기서
EE 오차 중앙 **9.5 mm** 를 냈다. 그러니 여기서 **9.5 mm 근처가 나와야 한다.**

- 나오면 → 씬·관측·액션·보간·목표변환 전 경로가 맞다는 뜻. 다음 단계로.
- 안 나오면 → **먼저 이것부터 고친다.** 이 상태로 MPC 계획을 돌리면 나쁜 결과가
  나와도 정책 탓인지 리그 탓인지 알 수 없다.

### 4-2. 눈으로 보기

```bash
twin/run.sh twin/runner/run_episode.py \
    --track-a npz:98 --track-b npz:110 \
    --gui --real-time --speed 2 --max-sep 8 \
    --out twin/data/runs/pair_gui --report /tmp/ep.txt
```

| 옵션 | 뜻 |
|---|---|
| `--gui` | Isaac Sim 창을 띄우고 계획 경로·목표·오차선을 그린다 |
| `--real-time` | 벽시계에 맞춰 재생 (기본은 최대 속도로 돌린다) |
| `--speed 2` | 2 배속 |
| `--max-sep 8` | 두 로봇을 8 m 안쪽에 배치 (한 화면에 담기) |

### 4-3. 내가 만든 MPC 계획으로 돌리기

```bash
twin/run.sh twin/runner/run_episode.py \
    --track-a csv:twin/data/plans/L150_base018_obj005_mytest:a \
    --track-b csv:twin/data/plans/L150_base018_obj005_mytest:b \
    --out twin/data/runs/mytest_v29 \
    --report /tmp/ep.txt
```

### 궤적 지정 문법

```
npz:<i>              data/plans.npz 의 i 번   (0~121, 뒤 24 개가 held-out)
npz:<path>:<i>       다른 npz 파일에서
csv:<dir>:<a|b>      MPC 계획 디렉토리에서 A 쪽 / B 쪽
```

### 4-4. 옵션 사전

| 옵션 | 언제 쓰나 |
|---|---|
| `--policy-cfg twin/configs/policy_v28.yaml` | 정책 교체. v12 / v28 / v29 가 있다 |
| `--localization ground_truth` | 위치를 참값으로. 정책만 보고 싶을 때 |
| `--localization amcl` | 학습과 같은 AMCL 오차 모델 (기본) |
| `--no-map` | 빈 평지에서. "맵이 원인인가?" 를 가를 때 |
| `--ff-mode zero` | 속도 피드포워드 끄기 |
| `--caster-damping 0.5` | 캐스터 감쇠 바꾸기 |
| `--bar 0` | 두 EE 를 잇는 시각용 막대 끄기 |
| `--hold-only` | 시작 자세만 유지. 수렴값만 볼 때 |
| `--spawn-a "x,y,yaw"` | 배치 자리 직접 지정 |
| `--settle 300` / `--hold 300` | 안정화 / 유지 스텝 수 |

**실험할 때는 한 번에 하나만 바꾼다.** 두 개를 같이 바꾸면 어느 쪽이 원인인지
알 수 없다. (이 저장소도 v29 에서 다섯 개를 한 번에 바꿨고, 그래서 원인 분리를
포기했다고 문서에 적어 두었다.)

---

## 4-5. 고전 제어기 기준선 — **정책 실험 전에 반드시**

### 왜 이걸 먼저 하나

"MPC 계획을 로봇이 따라갈 수 있는가" 와 "정책이 잘 학습됐는가" 는 **다른 질문**이다.
정책으로만 재생하면 두 질문이 섞여서, 추종이 나쁠 때 계획 탓인지 정책 탓인지 알 수 없다.

그래서 정책 없이 **무난한 고전 제어기**로 같은 계획을 돌려 본다.

```
차체   base_ref 를 유니사이클 궤적추종기로     (Kanayama 피드포워드 + 오차 피드백)
팔     ee 목표를 가중 감쇠최소자승 미분 IK 로  (PhysX 자코비안)
```

플랜트(씬·액션 항·액션지연·가감속 램프·로컬라이제이션)는 정책 경로와 **완전히 같고
제어기만 다르다.** 그래야 두 결과를 나란히 놓을 수 있다.

### 실행

```bash
twin/run.sh twin/runner/run_classic.py --raw-coords \
    --track-a csv:twin/data/plans/L150_base018_obj005_wa:a \
    --track-b csv:twin/data/plans/L150_base018_obj005_wa:b \
    --out twin/data/runs/classic_wa --report /tmp/classic.txt
```

> **`--raw-coords` 를 잊지 말 것.** MPC 계획 CSV 는 이미 map 절대좌표다. 빼면
> 리그가 맵 자유공간에서 자리를 다시 골라 버린다 (그건 `plans.npz` 궤적용 기능이다).

### 결과 읽는 법

| 고전 | 정책 | 해석 |
|---|---|---|
| O | O | 계획도 정책도 문제 없음 |
| O | X | **정책 문제.** 학습을 손볼 차례 |
| X | X | **계획 문제.** 정책을 아무리 학습해도 안 된다 |
| X | O | 정책이 IK 보다 낫다 — 대개 정책이 **차체를 옮겨서** 푼 경우다 |

리포트의 `제어기 진단` 절이 원인까지 갈라 준다:

```
IK 위치잔차 크고 리치 > reach_max(0.60)  -> 계획이 팔 밖을 요구한다
차체오차 크고 속도지령 포화가 잦다        -> 계획이 차체 한계보다 빠르다
둘 다 작은데 EE 오차가 크다               -> 제어기 이득 문제 (--kx/--ky/--kth)
```

### 실측 참고값 (MPC 계획 `L150_base018_obj005_wa`, 맵 켬, 30 s)

```
              hold 수렴      EE 중앙    EE 95%    차체편차 중앙
고전 제어기    6 / 9 mm      44 / 69    173/127     0.019 / 0.043 m
정책 v29       (참고) ~9.5 mm 급                    ~0.50 m
```

**hold 가 6~9 mm 로 수렴한다는 것이 핵심이다** — MPC 계획이 기구학적으로 성립한다는
뜻이다. 남은 44~69 mm 는 움직이는 목표에 대한 추종 지연이지 도달 불가가 아니다.

### 옵션

| 옵션 | 뜻 | 기본 |
|---|---|---|
| `--kx` / `--ky` / `--kth` | 차체 전진 / 횡 / 방위 오차 이득 | 1.5 / 6.0 / 2.5 |
| `--ik-lam` | DLS 감쇠. 크면 안정하고 느리다 | 0.05 |
| `--ik-w-rot` | 자세 오차 가중치 | 0.35 |
| `--ik-max-step` | 제어주기당 관절 변화 상한 [rad] | 0.15 |
| `--lead` | 목표 선행보상 [s] | 0.10 |

나머지(`--no-map`, `--gui`, `--localization`, `--hold-only` …)는 `run_episode.py` 와 같다.

### ⚠ `--ik-w-rot` 이 왜 있나 — 단위가 섞인다

Isaac Lab 기본 IK 는 오차를 `[위치(m), 자세(rad)]` 로 쌓아 **같은 가중치**로 푼다.
그런데 1 rad = 57° 이고 1 m 는 이 팔에서 거의 전체 리치다. 즉 **자세 60° 가 위치 1 m 와
맞먹는다.**

실제로 이것 때문에 팔이 자세를 맞추려다 **일자로 뻗은 채 위치 28 cm 를 포기**하는
현상을 봤다 (j3 +0.5°, 특이점, 관절 한계 접촉은 없음). `w_rot` 이 그 저울이다.

### 알아 둘 것 — `plans.npz` 궤적으로는 이 기준선이 실패한다

`--track-a npz:98` 같은 학습 데이터셋 궤적으로 돌리면 팔이 못 따라간다. 계획 CSV 와
달리 `plans.npz` 는 **base 와 EE 의 정합성이 보장되지 않기** 때문이다 — MPC 계획은
`reach_min ≤ |ee − base| ≤ reach_max` 를 하드 제약으로 풀지만 데이터셋은 아니다.

실측 (`npz:110`, 차체를 참조에 4 cm 로 붙였을 때):

```
고전 제어기   hold 에서 262 mm / 61 deg 에 갇힘 (팔이 일자로 뻗음)
정책 v29      EE 8.5 mm.  단 차체를 참조에서 0.61 m 벗어남
```

**정책은 차체를 옮겨서 푼 것이다.** 이 대비가
[`BASE_TRACKING_KR.md`](BASE_TRACKING_KR.md) 가 말하는 교환관계의 직접 증거다.

---

## 5. Isaac Sim 에서 정책 돌리기 — 라이브

RViz 에서 마우스로 찍으면 Isaac 안의 로봇이 바로 따라간다. 터미널 **3 개**가 필요하다.

```bash
# 터미널 1 — MPC 플래너 + map_server + RViz
twin/ros/run_planner.sh

# 터미널 2 — ROS 2 ↔ Isaac 공유메모리 다리
twin/ros/run_bridge.sh

# 터미널 3 — Isaac Sim
twin/run.sh twin/runner/live_sim.py --gui
```

그다음 RViz 에서 **2D Pose Estimate → 2D Goal Pose → INITIAL → EXECUTE**.

로봇의 상태는 셋이다:

```
park   대기.  맵 밖 먼 곳에 떨어뜨려 두고 아무것도 안 한다
hold   INITIAL 을 받아 시작 자세로 옮긴 뒤 t=0 목표를 붙잡고 있다
run    EXECUTE.  계획대로 재생
```

**왜 park 상태가 있나:** 계획이 오기 전에 두 대를 원점에 올리면 서로 겹쳐서 밀어내느라
버벅인다. 그래서 스폰 시점을 사람이 버튼으로 정하게 했다.

`live_sim.py` 옵션: `--no-map`, `--localization ground_truth`, `--bar 0`,
`--park 60 60`, `--wait 120`

---

## 6. 정합성 게이트 — 정책이나 관측을 건드렸으면 먼저 이것

```bash
# G1  상태 -> 관측 조립이 학습과 같은가        통과 기준 ≤ 1e-6
twin/run.sh twin/tools/check_obs_parity.py --amcl on \
    --policy handoff/traj_v29/policy.pt --report /tmp/g1.txt

# G2  계획 보간이 플래너 노드와 같은가          통과 기준 오차 0   (Isaac 불필요)
python3 twin/tools/check_plan_sampler.py

# G3  계획 -> EE 목표 변환이 학습과 같은가      pose 2e-6 / rot6d 5e-7
twin/run.sh twin/tools/check_plan_target.py --report /tmp/g3.txt

# 씬 스모크 — 스폰·낙하·자전
twin/run.sh twin/tools/check_scene.py --report /tmp/scene.txt
```

**왜 이런 게이트가 있나.** 관측 벡터를 손으로 조립하는 것이 이 배포에서 가장 위험한
부분이다. 한 칸만 어긋나도 **에러 없이 조용히 이상한 값**을 내고, 그게 제일 찾기
어려운 버그다. 실제로 이 프로젝트에서 팔 정책(42차원)과 통합 정책(42차원)이
**차원은 같은데 내용도 순서도 전혀 다른** 상태가 있었다. 차원 검사로는 안 걸린다.

G2 에서 `[0] 플래너 노드 md5` 가 실패하면: 플래너 노드 파일이 바뀐 것이다.
`_sample_step` 함수를 직접 대조해 보고, 같으면 `twin/plan/plan_sampler.py` 의
`PLANNER_NODE_MD5` 를 갱신한다.

---

## 7. 결과 읽기

```
twin/data/runs/<이름>/
├── log.npz     30 Hz 전 채널 원시 로그
└── meta.json   정책 md5, 관측 md5, yaw_offset, 궤적, 맵, localization ...
```

지표 표는 실행할 때 `--report` 파일에 나온다.

### 기준선 (정책 v29)

```
EE 중앙  9.52 mm | EE 95%  17.05 mm | 자세 중앙  1.72 deg
차체 편차 중앙  0.501 m | 3 cm 통과  88 %
```

### 로그를 직접 뜯어보기

```python
import numpy as np, json
d = np.load("twin/data/runs/mytest_v29/log.npz")
print(list(d.keys()))
print(json.load(open("twin/data/runs/mytest_v29/meta.json"))["columns"])

err = np.linalg.norm(d["ee_a"] - d["tgt_a"], axis=1)     # A 의 EE 오차 [m]
print("중앙 %.1f mm, 95%% %.1f mm" % (np.median(err)*1000, np.percentile(err,95)*1000))
```

주요 채널: `t`, `plan_t`, `ee_a/b`, `ee_quat_a/b`, `base_a/b`, `base_yaw_a/b`,
`twist_a/b`, `jpos_a/b`, `jvel_a/b`, `act_a/b`, `tgt_a/b`, `tgt_yaw_a/b`,
`base_ref_a/b`, `ee_dist`, `robot_dist`

### 무엇을 봐야 하나

| 지표 | 의미 | 나쁘면 |
|---|---|---|
| EE 위치 오차 | 손이 목표를 따라가는가 | 정책 성능 또는 목표가 분포 밖 |
| **EE 자세 오차** | 손 방향이 맞는가 | **먼저 무너지는 쪽이다. 반드시 따로 볼 것** |
| 차체 편차 | 계획 경로에서 얼마나 벗어났나 | 플래너 여유를 넘으면 실기에서 충돌 |
| `ee_dist` | 두 EE 사이 거리 | 막대 길이에서 벗어나면 강체 결합이 불가능 |
| `robot_dist` | 두 로봇 사이 거리 | `robot_robot_min 0.90` 아래면 충돌 위험 |

---

## 8. 문제 해결

| 증상 | 원인 | 조치 |
|---|---|---|
| `ModuleNotFoundError: isaacsim` | conda env 미활성 | `twin/run.sh` 로 실행 |
| `ModuleNotFoundError: rsl_rl` | 다른 python 이 잡힘 | 〃 |
| `ZSH_VERSION: unbound variable` | 스크립트에 `set -u` | **`set -u` 금지** |
| `FileNotFoundError: 페이로드 USD 가 없다` | 로봇 에셋 없음 | §2-1 |
| `ModuleNotFoundError: casadi` | Isaac python 으로 solve_plan 실행 | casadi 있는 python 으로 |
| 리포트가 안 보인다 | Kit 이 stdout 을 먹음 | `--report <파일>` |
| IPOPT 가 안 끝난다 | margin 이 과하거나 경로가 김 | margin 낮추기, `--target-dist` 줄이기 |
| 목표오차 ≠ 0 인데 CSV 가 나옴 | IPOPT 미수렴 = 해가 아님 | 그 계획은 쓰지 말 것 |
| EE 오차가 200 mm 넘게 발산 | 액션을 clamp 했거나 목표가 분포 밖 | §9 참고 |
| 두 로봇이 겹쳐서 밀어낸다 | 배치 자리 문제 | `--spawn-a/b`, `--min-sep` |

---

## 9. 절대 하면 안 되는 것

### 정책 출력을 clamp 하지 말 것

```python
action = policy(obs)
action = action.clamp(-1, 1)     # ← 이것 때문에 EE 오차가 275 mm 로 발산했다
```

학습의 `JointPositionAction` 에는 clamp 가 없고, 관측의 `last_action` 은 **raw** 값이다.
실측으로 팔 액션의 12~20 % 가 `|a| > 0.98` 이다. ±1 로 자르면 팔이 뻗을 수 있는
범위가 잘리고, 그 잘린 값이 관측으로 되먹여진다.

차체(`DiffDriveAction`)는 자기 안에서 속도 한계로 자르므로 그쪽은 무관하다.

### `PLAN_YAW_OFFSET` 을 import 하지 말 것

```python
from ee_track_ppo...commands import PLAN_YAW_OFFSET     # ← 하지 말 것
```

전역 상수는 현재 `+π/2` (v30 용) 인데 v29 는 `−π/2` 로 학습됐다. import 하면
**에러 없이 180° 틀어진 목표**를 준다. 값은 `twin/configs/policy_v29.yaml` 에서만 읽는다.

### 관측 차원이 같다고 안심하지 말 것

v12~v28 은 58 차원, v29 는 52 차원이다. 그런데 과거에 **차원은 같은데 내용이 다른**
사고가 있었다. 정책을 바꿨으면 반드시 G1 게이트를 돌린다.

---

## 10. 알아 두면 좋은 제약

- **한 판에 갈 수 있는 거리는 약 8.3 m.** 지평선 예산이
  `(max_steps − n_settle) × dt × v_nom / horizon_slack` 이고 기본값으로 8.3 m 다.
  건물을 가로지르려면 계획을 구간으로 쪼개야 한다.
- **`base_margin ≥ 0.70` 은 이 맵에서 안 된다.** 주행 영역이 21 조각으로 끊긴다.
  IPOPT 로도 확인됐다 (0.50 / 0.70 미수렴). 실제로 푸는 값은 **0.18**.
- **2 m 막대가 어떤 방향으로든 들어가는 자리는 주행 영역의 67 % 뿐이다.**
  시작·목표 자세는 하드 제약이라 나머지 33 % 에는 애초에 물체를 놓을 수 없다.
- **명령이 0 인데 차체가 초당 3.6° 자전한다.** 캐스터 감쇠가 0 이라 그렇고,
  학습 env 에서도 같은 폭으로 돈다 (리그 버그가 아니다).
- **협력 운송 편대는 현재 정책의 학습 분포 밖이다.** 한 대가 주행의 86 % 를
  후진해야 하는데 v29 는 후진 벌점을 걸고 학습됐다. 지금 리그의 기본 구성은
  **두 대가 각자 자기 궤적을 추종**하는 것이다.

---

## 11. 연습 과제

1. **리그 재현** — `npz:98 / npz:110` 으로 돌려 EE 중앙 9.5 mm 근처를 확인.
   안 나오면 왜인지 추적.
2. **맵의 영향** — 같은 궤적을 `--no-map` 과 비교. 차이가 있나? 없다면 왜?
3. **로컬라이제이션의 영향** — `--localization ground_truth` vs `amcl`.
   차이가 EE 오차 예산의 몇 %인가?
4. **정책 비교** — v12 / v28 / v29 를 같은 궤적에 돌려 EE 오차와 차체 편차를 표로.
   **교환관계가 보이는가?** (→ [`BASE_TRACKING_KR.md`](BASE_TRACKING_KR.md))
5. **margin 스윕** — `solve_batch.py --n 6` 으로 margin 0.05 / 0.18 / 0.30 의
   성립률을 재고, 각 계획을 리그에 돌려 **실제로 소진된 여유**를 측정.
   계획이 보장한 여유와 실제 여유의 차이가 곧 안전 여유의 근거다.
6. **피드포워드가 필요한가** — `--ff-mode planner` vs `zero`. EE 오차가 갈리면
   그 채널은 실재하고, 안 갈리면 지워도 되는 채널이다. (아직 미검증 항목이다)
