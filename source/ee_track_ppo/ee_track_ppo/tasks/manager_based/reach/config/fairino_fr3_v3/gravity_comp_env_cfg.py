# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fairino FR3 reach **v3 (중력보상 정합 + 작업공간 확장)**.

지시서: ``handoff/RETRAIN_V3.md``

v2f 가 남긴 것
--------------
안정성은 해결됐다. 실기 루프이득이 **-1.187(발산) -> -0.113(수축)** 으로 바뀌었고,
``follow_gain`` 을 7배 올리고 속도를 3배 푼 조건에서 진동이 사라졌다.

남은 결함은 **중력보상 불일치 하나**다. 실기 5곳 스캔 결과::

    목표          위치오차   성분 [x, y, z] mm      자세오차
    기준(홈)      12.7 mm   [3.3, -0.5, 12.2]     1.96 deg
    z -5cm        11.6 mm   [2.5, -0.5, 11.3]     1.88 deg
    x -5cm        12.8 mm   [1.9, -0.3, 12.7]     2.47 deg
    y +5cm        13.0 mm   [4.4, -0.8, 12.2]     1.84 deg

**오차가 사실상 +z 상수 하나다.** 상수 12.1 mm 를 빼면 잔여가 3.17 mm 로,
시뮬 성적 3.13 mm 와 일치한다. 즉 **정책의 추종 능력은 그대로 전이됐고**
그 위에 상수 편향만 얹혀 있다.

원인
----
학습 시 팔에 중력보상이 없다고 가정했다. 시뮬 관절 PD 는 중력을 버티느라
``q_target - q = tau_g / kp`` 만큼 처지고, 정책은 그 처짐을 예상해 **위로 더 명령**하도록
학습한다. 실기 FAIRINO 컨트롤러는 자체 중력보상이 있어 명령을 그대로 실행하므로
정책이 얹은 보정분만큼 **위로 초과**한다.

학습 PC 실측 (``scratchpad/gravity_sag.py``, 기본자세, 목표=기본자세)::

    강성       j2 처짐    j3 처짐    EE 변위
    0.4배      116.9 mrad 100.7 mrad  z -77.0 mm
    1.0배       46.1       43.7       z -31.5 mm      <- 학습 기준
    2.5배       19.4       19.0       z -13.4 mm

    kp x dq 가 세 조건에서 18.5 / 18.7 / 19.4 N*m 로 일정 -> dq = tau_g/kp 성립.
    기본자세 중력토크: j2 18.9, j3 17.5, j4 2.1 N*m (j1/j5/j6 은 ~0)

실기 편향 12.1 mm 는 전체 처짐 31.5 mm 의 38% 다. 100% 가 아닌 것이 오히려 가설을
강화한다 - 정책이 ``ee_pose_error`` 를 관측하므로 되먹임으로 상당 부분을 되돌리지만,
메모리 없는 MLP 라 적분 작용이 없어 일부가 남는다.

로봇 PC 에서 **팔을 뻗은 거리를 바꿔가며 중력 토크를 조절해 편차를 확인**했고,
편향이 토크에 따라 변하는 것이 확인되어 중력 원인이 확정됐다.
(RETRAIN_V3 §6 이 "일정하면 중력 외 요인" 이라고 적어둔 갈림길에서 중력 쪽으로 판정)

v3 에서 바뀌는 것 (딱 두 가지)
------------------------------
  [2-A] 중력 제거      : 실기 컨트롤러가 중력을 보상하므로, **정책 입장의 플랜트는
                         중력이 없는 것처럼 거동하는 것이 맞다.** 가장 충실한 모델이다.
  [2-C] 작업공간 확장  : DLS-IK 도달률 검사를 통과한 상자로만 넓힌다.

바꾸지 않는 것
--------------
도메인 랜덤화(지연 33~117 ms, 강성 0.4~2.5배 등)는 **전부 유지**한다. 정확도 문제의
원인은 중력이지 DR 이 아니고, DR 은 v2f 의 안정성을 만들어낸 바로 그 요소다.
중력을 제거하면 처짐 자체가 없어지므로 **DR 이 편향을 만들 이유도 함께 사라진다.**

보상(v2f 스윕에서 고른 파레토 최적점), 손목 속도 페널티 -0.005, 6D 회전 표현(v4 로 미룸),
관측 규약 38차원도 그대로다. -> **배포 쪽 코드 변경 없음.**
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import ee_track_ppo.tasks.manager_based.reach.mdp as ee_mdp

from ee_track_ppo.assets.fairino_fr3 import FAIRINO_FR3_EE_BODY  # isort: skip
from ee_track_ppo.tasks.manager_based.reach.config.fairino_fr3_v2.sweep_env_cfg import (  # isort: skip
    FR3SweepFEnvCfg,
)


##
# [2-C] 확장된 목표 상자 (base frame)
##
GOAL_X = (-0.50, -0.20)
GOAL_Y = (-0.25, 0.25)
GOAL_Z = (0.10, 0.50)
"""``scripts/check_reach_box.py`` 로 고른 상자 (전방향 +5 cm).

DLS-IK 위치+자세 동시 도달률 (300 샘플, 재시작 3회)::

    현재 상자 x(-0.45,-0.25) y(+-0.20) z(0.15,0.45)   95.7%   24.0 L
    확장 A  x +-5cm                                   95.7%   36.0 L
    확장 B  y +-5cm                                   95.3%   30.0 L
    확장 C  z +-5cm                                   97.0%   32.0 L
 -> 확장 D  전방향 +5cm                                95.7%   60.0 L   <- 채택
    확장 F  x 넓게                                    92.0%   66.5 L
    확장 E  전방향 +10cm                               84.3%  120.0 L   <- 탈락
    (대조군) Franka 상자 x(0.35,0.65)                  30.7%   42.0 L

**도달률을 현재와 같은 수준(95.7%)으로 유지하면서 부피를 2.5배**로 키운 지점이다.
+10 cm 는 84.3% 로 떨어져 쓰지 않는다 - 도달 불가능한 목표가 섞이면 위치 보상이
원리적으로 포화하지 못하고 학습 신호가 소멸한다 (franka v1 에서 실제로 겪은 함정).

자세(pitch) 자유화는 이번에 **하지 않는다.** 위치 확장과 동시에 하면 도달률이 급락하고
실패 시 어느 쪽 탓인지 구분이 안 된다.
"""


@configclass
class FairinoFR3V3EnvCfg(FR3SweepFEnvCfg):
    """v2f 의 보상·액추에이터 모델을 그대로 두고 중력만 제거 + 상자 확장."""

    def __post_init__(self):
        super().__post_init__()

        # [2-A] 중력 제거.
        #
        #   실기 FAIRINO 컨트롤러가 중력을 보상하므로 정책이 보는 플랜트에는 중력이 없다.
        #   시뮬에도 없애야 "정책이 학습한 플랜트 = 실기 플랜트" 가 된다.
        #
        #   대안으로 중력을 켠 채 액추에이터에 보상 토크를 더하는 방법(중력 피드포워드)도
        #   있다. 관성·코리올리까지 살릴 수 있어 더 정확하지만 구현이 더 들고,
        #   이번 사이클에서 닫으려는 것은 **정상상태 편향**이라 이쪽으로 충분하다.
        #
        #   한계: 실기 중력보상도 완벽하지 않다. 툴/페이로드가 붙으면 잔차가 생기므로
        #   그리퍼를 달면 재확인이 필요하다. 지금은 툴이 없어 문제되지 않는다.
        self.sim.gravity = (0.0, 0.0, 0.0)

        # [2-C] 작업공간 확장
        self.commands.ee_pose.ranges.pos_x = GOAL_X
        self.commands.ee_pose.ranges.pos_y = GOAL_Y
        self.commands.ee_pose.ranges.pos_z = GOAL_Z


@configclass
class FairinoFR3V3bEnvCfg(FairinoFR3V3EnvCfg):
    """v3b: v3 가 드러낸 **보상 불균형**을 잡는다. v3 대비 단일 주제(자세/위치 균형).

    v3 결과 (3000 it)::

        위치 2.31 mm  (목표 5 mm 이하 -> 크게 통과)
        자세 0.2297 rad = 13.16 deg  (목표 2.5 deg -> 5배 미달)

    "상자를 키우면서 자세를 자유화 못 해서" 라는 가설은 **기각됐다.**
    ``scratchpad/ik_residual.py`` 로 잰 기구학적 여력::

                            IK 잔여 자세오차 평균   5도 초과 비율
        구 상자 24 L              0.33 deg              1.5%
        신 상자 60 L              0.37 deg              1.3%
          껍질(새 영역만)          0.46 deg              1.7%
        기본자세 출발만 (분기 제한): 구 1.03 / 신 1.35 deg

    상자를 2.5배로 키워도 기구학적 자세 여력은 거의 안 나빠진다. ``pitch = pi`` 를
    고정한 채로도 신 상자의 **98.7% 가 자세오차 0 도로 도달 가능**하다.
    자세오차 상위 10% 목표의 base 거리도 0.491 m 로 전체 평균 0.492 m 와 같아,
    상자 가장자리에 몰려 있지도 않다.

    즉 **13.16 도는 기구학이 만든 게 아니다.** 낼 수 있는 최선이 0.37 도인데 정책이
    13.16 도를 낸다 - 35배 차이다. 도달 가능한 자세를 그냥 안 맞추고 있다.

    원인: 기울기 비 162배
    ----------------------
    v3 최종 시점(위치 2.31 mm / 자세 0.2297 rad)의 항별 기울기::

        위치                                자세
          std 0.100 w0.50 ->  5.00           std 0.30 w0.10 -> 0.180
          std 0.010 w0.15 -> 14.06           std 0.05 w0.08 -> 0.000  (구간 밖)
          std 0.004 w0.15 -> 27.49           L2       w0.20 -> 0.092
          합계               46.55           합계               0.287

    정책 입장에서 자세를 1도 줄이는 이득이 위치를 0.01 mm 줄이는 이득보다 작다.
    v2f 의 위치 칸 두 개는 **자세가 이미 0.05 rad 로 풀린 뒤에** 얹은 것이라,
    상자가 커져 자세가 어려워지자 균형이 무너졌다.

    상자 확장은 원인이 아니라 **방아쇠**다 - 이 불균형을 드러냈을 뿐이다.

첫 시도(B3) 는 실패했다 - 기록
    -----------------------------
    ``logs/rsl_rl/fr3_reach_v3b/2026-07-29_11-22-55_ABORTED_B3``  (it1000 에서 중단)

    처방: 자세 fine 0.1 -> **0.35**, 자세 mid std 0.15 w **0.35**, 위치 ultra **제거**.
    결과: 자세는 의도대로 개선(it1000 에서 9.11 deg vs v3 17.86)됐으나
    **위치가 61.32 mm 로 무너졌다** (같은 시점 v3 는 3.05 mm).

    사후 분석에서 원인이 처음 생각과 달랐다.

    (a) **위치 ultra 제거는 무죄였다.** std 0.004 칸은 오차가 10 mm 아래로 내려와야
        살아난다. 무너진 87 mm 구간의 위치 기울기는 v3 와 **소수점까지 동일**했다::

            pos 100 mm : 차이 0.0%      pos 10 mm : 차이 8.1%
            pos  50 mm : 차이 0.0%      pos  4 mm : 차이 46.9%

    (b) 진짜 원인은 **성형보상 총 가중치(= 정책이 얻을 수 있는 상금)의 역전** 이었다::

            v3    위치 0.80  자세 0.18   -> 위치가 4.44배 큰 상금
            B3    위치 0.65  자세 0.78   -> 위치가 0.83배   <- 뒤집힘

        PPO 는 총 리턴을 최대화한다. B3 에서 자세가 **더 큰 상금**이 되었고 동시에
        **더 싸게** 얻을 수 있었다(팔을 옮기지 않고 손목만 돌리면 된다).
        그래서 정책이 자세부터 챙기고 위치를 방치했다.

    교훈: 기울기 비만 보지 말고 **가중치 총합의 대소**를 함께 볼 것. 그리고 기울기는
    한 점이 아니라 **학습 경로 전 구간**에서 확인할 것.

    v3b 의 처방 (B3' - 채택)
    ------------------------
    위치는 **전혀 건드리지 않고**(ultra 포함 그대로) 자세만 v3 대비 약 2배로 올린다.
    위치가 여전히 더 큰 상금이도록 유지하는 것이 핵심이다.

    ================================  100mm/.55  50mm/.40  20mm/.30  10mm/.20  5mm/.10  2mm/.05  상금비
    v3   자세 0.10                        8.5      16.4      22.6      40.5     60.4     45.9    4.44
    B3   자세 0.35+0.35 (실패)             6.3       8.1       7.6       7.9      6.1      4.9    0.83
    B3'  자세 0.18+0.18 (채택)             7.6      12.0      12.9      15.9     18.4     19.8    1.82

    (표의 숫자는 학습 경로를 따라간 위치/자세 기울기 비)

    ``end_effector_orientation_tracking_precise`` (w 0.08 / std 0.05) 는 지시대로
    **가중치도 std 도 건드리지 않는다.** 자세 오차가 0.05 rad 근처로 내려오면
    그때 이 항이 알아서 살아난다.
    """

    def __post_init__(self):
        super().__post_init__()

        ee = SceneEntityCfg("robot", body_names=[FAIRINO_FR3_EE_BODY])

        # [1] 자세 fine(std 0.3) 가중치 상향. 큰 오차 대역에서 유일하게 살아 있는 칸이다.
        #     0.35 는 과했다(위 실패 기록). 0.18 이면 자세 총 가중치가 0.44 로
        #     위치 0.80 보다 여전히 작다.
        self.rewards.end_effector_orientation_tracking_fine_grained.weight = 0.18

        # [2] 자세 중간 칸 추가. std 0.3 과 0.05 사이가 6배 벌어져 있어
        #     v3 가 정체한 0.23 rad 대역에 칸이 없다. tanh 기울기는 (오차 ~ std) 에서 최대다.
        self.rewards.end_effector_orientation_tracking_mid = RewTerm(
            func=ee_mdp.orientation_command_error_tanh,
            weight=0.18,
            params={"asset_cfg": ee, "std": 0.15, "command_name": "ee_pose"},
        )

        # [3] 위치는 건드리지 않는다. v3 의 2.31 mm 수렴 능력을 그대로 유지한다.
        #     (B3 에서 ultra 를 뺐지만 무죄로 판명됐고, 빼면 종점 정밀도만 잃는다)


##
# [v3c] pitch 자유화 범위
##
PITCH_RANGE = (math.pi - 0.5, math.pi + 0.5)
"""툴이 수직 아래에서 **+-0.5 rad (약 +-29도)** 이내로 기운다.

왜 이 범위인가 - 반구 규약 때문이다
-----------------------------------
양쪽(시뮬/배포)이 ``dot(q, [0,0,1,0])`` 의 부호로 반구를 정한다. roll = 0 일 때
이 값은 정확히 ``sin(p/2) * cos(psi/2)`` 다 (검산 완료). pitch 가 0 쪽으로 가면
``sin(p/2)`` 가 0 으로 수렴하면서 부호가 부동소수점 잡음에 묻힌다.

**다만 실제 병목은 pitch 가 아니라 yaw 다.** ``cos(psi/2)`` 가 psi = +-pi 에서 0 으로
가기 때문에, pitch = pi 고정인 v3b 에서도 최소값이 이미 6.1e-17 이었다::

    pitch pi      (sin=1.000)  dot 최소 6.1e-17
    pitch pi+-0.5 (sin=0.969)  dot 최소 5.9e-17     <- 3% 감소일 뿐
    pitch pi/2    (sin=0.707)  dot 최소 4.3e-17

그래서 pitch 를 좁히는 것보다 **yaw 를 좁히는 것이 실질적 방어**다 (:data:`YAW_RANGE`).

도달률은 문제가 아니다 (실측)
-----------------------------
``scripts/check_reach_box.py`` (400 샘플, 재시작 4회, 60 L 상자)::

    pitch 고정        yaw +-3.14   96.0%
    pitch 고정        yaw +-3.00   95.0%
    pitch pi+-0.5     yaw +-3.00   96.0%    <- 하락 없음
    pitch pi+-0.3     yaw +-3.00   96.5%
    pitch pi/2~3pi/2  yaw +-3.00   93.0%    <- +-90도까지 풀어도 90% 이상

"pitch 를 풀면 pose 집합이 5D -> 6D 라 달성률이 떨어진다" 는 예상과 달랐다.
FR3 는 6DOF 라 pitch 고정이든 아니든 **어차피 6D pose 를 정확히 맞춰야 하므로**
여유자유도가 남아돌지 않는다. 살짝 기울이는 편이 오히려 유리한 목표도 생긴다.

즉 이 범위를 정하는 것은 도달률이 아니라 **반구 규약과 실기 충돌 위험**이다.
데이터상으로는 +-90도까지 여유가 있으므로, 실기에서 충돌 모델이 준비되면
v4 에서 더 넓힐 수 있다.
"""

YAW_RANGE = (-3.0, 3.0)
"""+-3.14 -> +-3.0 으로 좁힌다. 반구 판정의 실질적 안전 마진을 만드는 것이 목적이다::

    pitch pi+-0.5, yaw +-3.14  ->  dot 최소 7.7e-4
    pitch pi+-0.5, yaw +-3.00  ->  dot 최소 6.9e-2     <- 90배 안전

float32 epsilon 이 1.2e-7 이므로 +-3.0 이면 완전히 안전 구간이다.
잃는 것은 yaw 양 끝 0.14 rad (8도) 씩이고 도달률 손실은 1%p 다.
"""


@configclass
class FairinoFR3V3cEnvCfg(FairinoFR3V3bEnvCfg):
    """v3c: v3b 에 **pitch 자유화**를 더한다. v3b 대비 단일 주제.

    사다리의 마지막 단이다::

        v3   중력제거 + 박스확장                  (자세 13.16 deg 로 실패)
        v3b  + 보상 재균형                        2.82 mm / 2.48 deg   <- 이관 완료
        v3c  + pitch 자유화 (pi +- 0.5)           이 클래스

    바뀌는 것은 커맨드의 자세 범위 두 줄뿐이다. 보상·중력·상자·DR·지연·관측 규약은
    v3b 와 완전히 동일하므로 **배포 쪽 변경도 없다** (``OBS_DIM = 38`` 유지).

    지켜볼 것
    ---------
    pitch 를 풀면 자세 난이도가 올라간다. v3 에서 박스를 넓혔을 때 자세가 굶었던 것과
    **같은 함정**이 재현될 수 있다. 다만 v3b 의 자세 사다리는 칸이 3개(std 0.3/0.15/0.05)에
    가중치 총합 0.44 로, v3 당시(칸 2개 / 0.18)보다 2.4배 강하다.

    it1000 에서 자세가 0.20 rad 를 넘으면 사다리에 칸을 하나 더 넣어야 한다는 신호다.
    (v3 는 같은 시점 0.31 rad, v3b 는 0.145 rad 였다)
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.ee_pose.ranges.pitch = PITCH_RANGE
        self.commands.ee_pose.ranges.yaw = YAW_RANGE


##
# [v3d] 실제 관절 한계(j3 +-150) 에서 다시 잡은 상자
##
GOAL_X_V3D = (-0.48, -0.24)
GOAL_Y_V3D = (-0.28, 0.28)
GOAL_Z_V3D = (0.18, 0.46)
"""37.6 L. 평균 도달률 95.8%, 최악 구석 93.2%.

왜 60 L 에서 줄었나
-------------------
60 L 은 URDF 의 **틀린 j3 한계(+-162 deg)** 로 계산된 값이었다. 실기 컨트롤러의
``GetJointSoftLimitDeg`` 실측은 **+-150 deg** 이고, 그 차이 때문에 상자 구석
(x -0.22 / z 0.18 부근, 도달률 20%)에서 ``ServoJ`` 가 joint overrun 폴트를 냈다.

즉 되돌아간 것이 아니라 **처음부터 쓸 수 없던 영역을 뺀 것**이다.
구 상자 24 L 대비로는 여전히 1.6 배이고, 여기에 pitch +-29 도 자유화가 더해진다.

어떻게 골랐나
-------------
``scripts/find_max_box.py`` 로 작업공간 도달률 지도를 만들고, 그 위에서 후보 상자를
2500 샘플씩 검증했다. 판정은 두 가지를 동시에 건다::

    평균 도달률 >= 95%
    8 개 구석(팔분면)별 도달률 >= 90%     <- 죽은 구석 금지

**두 번째가 핵심이다.** v3c 상자는 평균 93.7% 로 나쁘지 않았는데 한 구석이 72% 라
거기서 실기가 폴트를 냈다. 평균만 보면 죽은 구석이 통과한다.

후보 비교 (j3 +-150, pitch pi+-0.5, yaw +-3.0)::

    D   x(-.48,-.26) y+-.25 z(.18,.48)   96.5% / 93.8%   33.0 L
    D1  x(-.48,-.24) y+-.25 z(.18,.48)   96.0% / 93.6%   36.0 L
 -> D5  x(-.48,-.24) y+-.28 z(.18,.46)   95.8% / 93.2%   37.6 L   <- 채택
    Q   x(-.52,-.24) y+-.25 z(.20,.50)   94.3% / 79.5%   42.0 L
    U   x(-.50,-.24) y+-.30 z(.20,.50)   94.9% / 81.5%   46.8 L
    T   x(-.54,-.26) y+-.28 z(.20,.52)   89.0% / 55.0%   50.2 L

**37.6 L 이 축정렬 박스의 한계다.** 더 넓히면 예외 없이 `멀고·옆으로·높은` 구석에서
무너진다. 그 방향이 특별히 나빠서가 아니라 **박스 대각선이 도달 껍질을 뚫기 때문**이다.

다음에 더 넓히려면 - 원통 껍질
------------------------------
팔의 도달 영역은 base 중심 구각이고, y=0 평면 지도를 반경으로 읽으면 수평거리
0.26~0.50 구간이 z 0.18~0.54 전 구간에서 90~100% 인 **띠(annulus)** 다.
원통 껍질로 바꾸면 같은 도달률에서 부피가 1.6~2 배가 될 것으로 추정된다::

    rho 0.26~0.50, z 0.18~0.50, azimuth +-60deg  ->  약 61 L
    azimuth +-75deg                              ->  약 76 L

(추정치다. 껍질 도달률 실측은 중단했다 - 운송 작업에 그만한 범위가 필요하지 않다고
판단해서다. 필요해지면 그때 재고 구현하면 된다)

``UniformPoseCommand`` 가 박스 균등 샘플링이라 **커스텀 커맨드 term 이 필요**하다
(``commands.py`` 의 ``UniformPoseWorldCommand`` 를 본떠 만들면 된다).
배포 쪽 관측 규약에는 영향이 없고 ``TRAIN_POS_RANGE`` 가 (rho, z, azimuth) 로 바뀔 뿐이다.
"""


@configclass
class FairinoFR3V3dEnvCfg(FairinoFR3V3cEnvCfg):
    """v3d: v3c 에 **실제 관절 한계(j3 +-150)** 를 반영하고 상자를 다시 잡는다.

    에셋 쪽 변경(URDF j3 +-162 -> +-150, USD 재생성)은 이 클래스와 무관하게
    **모든 태스크에 적용**된다. 여기서는 그 한계에 맞는 상자만 지정한다.

    실기 검증에서 드러난 네 번째 sim-to-real 간극이다::

        1. 액추에이터 모델 (alpha)  -> v2  지연 주입 + DR
        2. 지연 72 ms               -> v2  액션 링버퍼
        3. 중력보상                 -> v3b 시뮬 중력 제거      (실기 검증 완료)
        4. 관절 한계 검증 방식       -> v3d 이번

    4 번의 성질: Isaac 의 ``JointPositionAction`` 은 액션을 PD **설정점**으로 넣으므로
    한계를 넘는 값을 줘도 관절이 한계에서 멈출 뿐 무해하다. 그래서 시뮬에서는
    j3 가 150~162 구간을 요구하는 목표도 "대충 도달" 한 것처럼 보인다.
    반면 실기의 ``ServoJ`` 는 위치 명령을 **검증하고 거부**한다.
    **시뮬에서 정상인 명령이 실기에서 폴트를 낸다.**
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.ee_pose.ranges.pos_x = GOAL_X_V3D
        self.commands.ee_pose.ranges.pos_y = GOAL_Y_V3D
        self.commands.ee_pose.ranges.pos_z = GOAL_Z_V3D


@configclass
class FairinoFR3V3EnvCfg_PLAY(FairinoFR3V3EnvCfg):
    """학습 결과 확인 / export 용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FairinoFR3V3bEnvCfg_PLAY(FairinoFR3V3bEnvCfg):
    """학습 결과 확인 / export 용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FairinoFR3V3cEnvCfg_PLAY(FairinoFR3V3cEnvCfg):
    """학습 결과 확인 / export 용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FairinoFR3V3dEnvCfg_PLAY(FairinoFR3V3dEnvCfg):
    """학습 결과 확인 / export 용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
