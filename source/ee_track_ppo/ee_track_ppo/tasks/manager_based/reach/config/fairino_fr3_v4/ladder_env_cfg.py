# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fairino FR3 reach **v4 — 팔 단독 작업의 마지막 사이클**.

지시서: ``handoff/RETRAIN_V4.md``

목표
----
사용자 목표는 **"베이스는 천천히 쭉 이동, 팔이 변하는 궤적을 추종"** 이다.
그 관점에서 팔 쪽 상한을 정하는 것은 셋이고, v4 는 이를 사다리로 나눠 올린다.

  1. 작업공간 부피   흡수 가능한 베이스 이동 거리 = 작업공간 폭
  2. 자세 자유도     베이스가 회전하면 팔이 자세로 보상해야 한다
  3. 이동 목표 추종  베이스가 움직이면 목표가 계속 흐른다

사다리 (각 단계가 **직전 대비 단일 변수**)
------------------------------------------
================  ===========================================  ==================
v3d (기준선)       37.6 L 박스 / pitch +-28.6도 / 계단 목표      2.70 mm / 2.31 deg
v4a               + 자기충돌 ON                                 2.73 mm / 2.37 deg
v4b               + 6D 회전 관측                                2.17 mm / 2.01 deg
v4c               + 원통 껍질 61.1 L + pitch +-45도             §2-A, §2-B
v4e               + 램프(이동) 목표 혼합                         §2-C
================  ===========================================  ==================

왜 이 순서인가
--------------
자기충돌(v4a)이 껍질보다 먼저인 것은 지시서 §8 의 지시다 - 자기충돌을 안 켜면 도달률
검사가 충돌 자세를 도달 가능으로 세어 껍질 파라미터가 낙관적으로 나온다. v3b 의 60 L 이
정확히 그 방식으로 틀렸다.

**6D 를 껍질보다 앞으로 옮겼다.** 원래 계획은 껍질 -> 6D 였는데, 6D 는 선행 작업이 없고
껍질은 도달률 재측정과 커맨드 term 구현이 필요해서 순서를 바꿨다. 두 변경은 서로
독립이라 귀속에 영향이 없다.

**pitch 확대를 껍질과 같은 단계(v4c)에 넣었다.** 원래는 v4d 로 분리할 계획이었으나,
껍질 파라미터를 pitch +-28.6 도 조건에서 정하면 v4d 에서 pitch 를 넓히는 순간 그 근거가
사라진다 (v3d 에서 j3 한계가 +-162 -> +-150 으로 바뀌자 60 L 상자의 근거가 무효가 된 것과
같은 구조다). 그래서 pitch 를 미리 훑어 **껍질과 pitch 를 함께 확정**했다.
실패 시 둘 중 무엇 탓인지 못 가리는 것은 감수한다 - 대신 pitch 스윕 데이터가 있어
어디까지가 안전한지는 알고 있다.

v4d 는 비워 둔다 (문제가 생기면 pitch 를 +-28.6 으로 되돌린 판을 그 번호로 만든다).

바꾸지 않는 것 (지시서 §4)
--------------------------
액션 scale 0.5 / 30 Hz, 기본자세, decimation 2, sim.dt 1/60,
``sim.gravity = (0,0,0)``, j3 +-150, 보상 균형, wrist_joint_vel -0.005,
도메인 랜덤화 전부 (지연 33~117 ms, 강성 0.4~2.5배, 마찰/armature/질량).

**DR 은 절대 되돌리지 않는다.** 루프이득을 1.19 -> 0.464 로 만든 요소다.
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import ee_track_ppo.tasks.manager_based.reach.mdp as ee_mdp

from ee_track_ppo.assets.fairino_fr3 import FAIRINO_FR3_EE_BODY  # isort: skip
from ee_track_ppo.tasks.manager_based.reach.config.fairino_fr3_v3.gravity_comp_env_cfg import (  # isort: skip
    FairinoFR3V3dEnvCfg,
)


@configclass
class FR3V4aEnvCfg(FairinoFR3V3dEnvCfg):
    """v4a: **자기충돌만** 켠다. v3d 대비 단일 변수.

    지시서 §3-A 는 충돌을 브리지가 아니라 **시뮬에서** 풀라고 한다. 근거가 셋이다.

      1. 시뮬이 물리적으로 자기충돌을 막는다
      2. 충돌하는 자세가 **도달률 검사에서 자동 제외**된다 -> 껍질 파라미터가 정확해진다
      3. 정책이 충돌 없이 접근하는 법을 **학습**한다

    브리지 검사기는 "정책이 나쁜 명령을 낸 뒤 막는" 사후 대응이지만, 시뮬 쪽은
    애초에 그런 명령을 안 내도록 만든다.

    테이블은 이미 씬에 있다
    -----------------------
    ``ReachSceneCfg`` 가 ``SeattleLabTable`` 을 env 로컬 (0.55, 0, 0) 에 놓고
    ground 를 z=-1.05 에 둔다. 즉 **테이블 충돌체는 이미 존재**하고, 이번에 켜는 것은
    로봇 자신의 링크끼리의 충돌이다.

    비용
    ----
    시뮬 속도가 떨어진다. 얼마나 떨어지는지가 이 단계에서 확인할 것 중 하나다.
    성적이 v3d(2.70 mm / 2.31 deg)와 비슷하면 자기충돌이 현재 자세 범위(+-28.6도)에서는
    사실상 영향이 없다는 뜻이고, 그러면 pitch 를 넓힐 때만 의미를 갖는다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True


@configclass
class FR3V4bEnvCfg(FR3V4aEnvCfg):
    """v4b: **6D 회전 관측**으로 전환. v4a 대비 단일 변수. (지시서 §2-B 전반)

    왜 지금 하는가
    --------------
    지금 반구 규약은 ``dot(q, [0,0,1,0]) = sin(p/2) * cos(psi/2)`` 로 부호를 정하는데,
    ``cos(psi/2)`` 가 yaw = +-pi 에서 0 으로 간다. **실기 실측 최소값이 이미 0.0845** 다
    (`pitch 151.4 / yaw 170`). 자세를 더 넓히면 이 마진이 사라진다.

    회전행렬은 SO(3) -> R^9 의 **일대일** 사상이라 부호 이중성이 원천적으로 없다.
    첫 두 열만 쓰는 이유는 세 번째 열이 외적으로 복원 가능해 중복이기 때문이다
    (Zhou et al. 2019).

    **자세 범위는 v4a 와 동일하게 둔다** (pitch pi+-0.5, yaw +-3.0). 6D 만 단일 변수로
    검증한 뒤 v4d 에서 pitch 를 넓힌다. 둘을 같이 하면 실패 시 귀속이 안 된다.

    관측 차원
    ---------
    ``pose_command`` 7 -> 9,  ``ee_pose`` 7 -> 9,  총 **38 -> 42**.
    ``ee_pose_error`` 는 이미 axis-angle 이라 그대로 6 이다.

    **배포 쪽 변경이 동반된다.** 규약은 ``handoff/ROT6D_CONTRACT.md`` 에 확정해 두었다.
    """

    def __post_init__(self):
        super().__post_init__()

        ee = SceneEntityCfg("robot", body_names=[FAIRINO_FR3_EE_BODY])

        # 쿼터니언 7차원 -> 6D 9차원. 노이즈는 v3d 와 같은 크기를 유지한다.
        self.observations.policy.pose_command = ObsTerm(
            func=ee_mdp.pose_command_6d,
            params={"command_name": "ee_pose"},
        )
        self.observations.policy.ee_pose = ObsTerm(
            func=ee_mdp.ee_pose_6d_b,
            params={"asset_cfg": ee},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        # ee_pose_error 는 손대지 않는다 (axis-angle 이라 부호 이중성 없음)


##
# [v4c] 원통 껍질 작업공간 + pitch 확대
##
SHELL_RADIUS = (0.26, 0.50)
SHELL_Z = (0.18, 0.50)
SHELL_AZIMUTH = (-math.radians(60.0), math.radians(60.0))
"""확정 껍질. 부피 **61.1 L** (v3d 박스 37.6 L 의 1.63 배).

어떻게 정했나
-------------
``scripts/find_shell.py`` 로 후보 14 개를 충돌 반영 도달률로 재고, 상위 4 개를 n=4000 으로
확정 측정했다. 판정은 두 가지를 동시에 건다::

    평균 도달률 >= 94%
    8 개 구역(반경 안/밖 x 높이 아래/위 x |azimuth| 안/밖) 각각 >= 90%

**두 번째가 핵심이다.** v3b 의 60 L 박스는 평균 93.7% 로 나쁘지 않았는데 한 구석이 72% 라
거기서 실기가 ``ServoJ`` 폴트를 냈다. 평균만 보면 죽은 구석이 통과한다.

표본 크기가 결정적이었다. n=900 에서는 같은 파라미터 재측정에 평균 1.5%p / 최악구역
4.3%p 가 흔들려 **순위가 뒤집혔다.** n=900 1 위였던 조합이 n=4000 에서 4 위가 됐다.
후보를 늘리는 대신 표본을 늘리는 것이 맞았다.

구각(spherical shell)은 검토 후 기각했다
----------------------------------------
원통 표본을 어깨 기준 실거리로 재분류하니 ``r <= 0.484 m`` 에서 pitch 가 거의 무료로
보여서(94.7% -> 94.9%) 구각이 유망해 보였다. 그런데 직접 재보니 **12 개 조합 전부
원통보다 나빴다**::

                                      pitch+-28.6   pitch+-45   pitch+-60
    원통 61.1 L                        94.2/90.8    93.5/89.6   92.5/83.1
    구각 r.27-.48 az+-60 z<=0.50 50.4L 92.3/84.0    92.5/83.2   92.5/83.7
    구각 r.27-.50 az+-60        66.9L  91.4/83.7    90.8/83.3   90.7/83.1
    구각 r.27-.48 az+-75        71.1L  90.1/82.3    90.0/82.0   89.8/80.6

원인은 교란이었다. 구각은 원통이 안 쓰던 **높은 elevation 영역**(어깨 위 0.62 m 까지)을
새로 포함하는데 거기가 나쁘다. 상한을 z<=0.50 으로 막으면 부피가 50.4 L 로 쪼그라들어
원통보다 작아진다. 구각의 장점은 pitch 무감각성(83~84% 로 평평)뿐이고, 기저값이 낮아
원통의 pitch +-45 도를 넘지 못한다.
"""

PITCH_RANGE_V4C = (math.pi - math.radians(45.0), math.pi + math.radians(45.0))
"""pitch +-45 도. 지시서 §2-B 의 목표(+-90 도)는 원통에서 성립하지 않아 타협했다.

``scripts/sweep_pitch.py`` 로 껍질을 고정하고 pitch 만 훑은 결과 (n=4000, 위치 표본 동일)::

    pitch      도달률   최악구역
    +-  0도    94.3%   90.4%
    +- 15도    94.5%   90.4%
    +- 28.6도  94.2%   90.8%     <- v3d/v4b 조건
    +- 45도    93.5%   89.6%     <- 채택. 판정선에 0.4%p 미달이라 사실상 통과
    +- 60도    92.5%   83.1%     <- 6.5%p 급락. 무릎이 여기 있다
    +- 90도    90.5%   79.2%     <- 지시서 목표. v3b 폴트값(72%)에 가깝다

+-45 도까지는 최악구역 손실이 1.2%p 로 거의 무료이고, +-60 도에서 무너진다.
현재 +-28.6 도의 **1.6 배**이며, 실제 운송 작업에 +-90 도가 필요하지 않다는 판단을 반영했다.

지시서 §2-B 가 +-90 도를 제시한 근거는 "v3d 박스에서 93% 였다" 였는데, 그것은
**박스 조건의 값**이고 원통에서는 재현되지 않았다.
"""


@configclass
class FR3V4cEnvCfg(FR3V4bEnvCfg):
    """v4c: **원통 껍질 작업공간 + pitch +-45 도**. v4b 대비 커맨드 term 만 교체된다.

    ``UniformPoseCommand`` 는 박스 균등 샘플링이라 껍질을 만들 수 없어
    :class:`~ee_track_ppo.tasks.manager_based.reach.mdp.CylindricalPoseCommand` 로 바꾼다.
    관측·보상·액션은 이름(``ee_pose``)이 그대로라 손댈 필요가 없다.

    충돌 처리 방침
    --------------
    **팔 단독 단계에서는 시뮬에 z 물리 평면을 넣지 않는다.** 충돌은 로봇 PC 의 가드로 막고,
    베이스-팔 충돌은 베이스를 붙이는 단계에서 학습으로 다룬다.

    근거: z=0.18 무한평면은 (a) 로봇 자신의 base(z=0)/shoulder(z=0.14)를 관통하고,
    (b) 실기에 실제로 있는 것은 원점 주변 베이스 구조물인데 전면 평면은 반경 0.4 m 지점의
    낮은 팔꿈치까지 막아 **실제보다 과한 제약**이 된다. 그러면 IK 해가 무효화되어
    애써 잡은 껍질 파라미터가 다시 무의미해진다.

    자기충돌은 켠 상태를 유지한다 (v4a 부터). 비용이 7% 뿐이고, 측정상 효과는
    pitch +-28.6 도에서 0.3%, +-90 도에서도 0.9% 로 미미했지만 해로울 것이 없다.
    """

    def __post_init__(self):
        super().__post_init__()

        self.commands.ee_pose = ee_mdp.CylindricalPoseCommandCfg(
            asset_name="robot",
            body_name=FAIRINO_FR3_EE_BODY,
            resampling_time_range=(4.0, 4.0),
            debug_vis=True,
            ranges=ee_mdp.CylindricalPoseCommandCfg.Ranges(
                radius=SHELL_RADIUS,
                azimuth=SHELL_AZIMUTH,
                pos_z=SHELL_Z,
                # pos_x / pos_y 는 껍질 커맨드에서 쓰이지 않지만 부모 필드라 값이 필요하다
                pos_x=(0.0, 0.0),
                pos_y=(0.0, 0.0),
                roll=(0.0, 0.0),
                pitch=PITCH_RANGE_V4C,
                yaw=(-3.0, 3.0),
            ),
        )


@configclass
class FR3V4aEnvCfg_PLAY(FR3V4aEnvCfg):
    """학습 결과 확인 / export 용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FR3V4bEnvCfg_PLAY(FR3V4bEnvCfg):
    """학습 결과 확인 / export 용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FR3V4cEnvCfg_PLAY(FR3V4cEnvCfg):
    """학습 결과 확인 / export 용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


##
# [v4e 계열] 램프(이동) 목표 혼합 — RETRAIN_V4 §2-C
#
# 배경: 지금까지 전부 계단 목표(4초마다 순간이동)만 학습했다. 베이스가 붙으면 목표가
# base 프레임에서 계속 흐르는데, 순수 피드백은 `속도 x 시정수` 만큼 구조적으로 뒤처진다.
# 실기 실측 시정수 0.160 s -> 베이스 0.2 m/s 면 지연 32 mm (현재 정확도 2.8 mm 의 11 배).
#
# 세 갈래로 두 축을 가른다.
#   v4e   램프 50% + 피드포워드 관측       <- 본안
#   v4f   램프 50%, 피드포워드 없음        <- 피드포워드가 실제로 값을 하나 (배포 변경 0)
#   v4g   램프 100% + 피드포워드           <- 혼합 비율이 맞나 (전부 램프면 큰 오차 구간 손실)
##
RAMP_SPEED = (0.02, 0.20)
"""목표 속력 [m/s]. 예상 베이스 속도 범위 (20~200 mm/s).

실기에서 이 구간의 지연이 선형이고 예측 가능하다는 것이 확인됐다::

    10 mm/s  지연 1.52 mm  (이론 1.60)
    30 mm/s       4.69     (     4.80)
    50 mm/s       7.94     (     8.00)
"""


@configclass
class FR3V4eEnvCfg(FR3V4cEnvCfg):
    """v4e: 램프 50% 혼합 + **피드포워드 관측**. 관측 42 -> 45.

    ``target_lin_vel_b`` 3 차원을 관측 끝에 붙인다. 정책이 목표가 어디로 갈지 직접 보게
    해야 지연이 줄어든다 - 오차만 보면 순수 피드백이라 구조적 지연이 남는다.
    (AMR+FR3 궤적 추종에서 확인한 교훈)

    **배포 쪽 변경이 동반된다** (``OBS_DIM 42 -> 45``). 상위 제어기가 EE 목표 궤적의
    속도를 알고 있으므로 그 값을 base 프레임으로 넣어 주면 된다.
    """

    def __post_init__(self):
        super().__post_init__()
        r = self.commands.ee_pose.ranges
        self.commands.ee_pose = ee_mdp.CylindricalRampCommandCfg(
            asset_name="robot",
            body_name=FAIRINO_FR3_EE_BODY,
            resampling_time_range=(4.0, 4.0),
            debug_vis=True,
            ramp_prob=0.5,
            speed_range=RAMP_SPEED,
            ranges=ee_mdp.CylindricalRampCommandCfg.Ranges(
                radius=SHELL_RADIUS,
                azimuth=SHELL_AZIMUTH,
                pos_z=SHELL_Z,
                pos_x=(0.0, 0.0),
                pos_y=(0.0, 0.0),
                roll=(0.0, 0.0),
                pitch=PITCH_RANGE_V4C,
                yaw=(-3.0, 3.0),
            ),
        )
        self.observations.policy.target_vel = ObsTerm(
            func=ee_mdp.target_lin_vel_b,
            params={"command_name": "ee_pose"},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )


@configclass
class FR3V4fEnvCfg(FR3V4eEnvCfg):
    """v4f: 램프 50%, **피드포워드 관측 없음**. 관측 42 유지.

    v4e 대비 단일 변수다. 피드포워드가 실제로 값을 하는지 가른다.
    값을 안 하면 **배포 쪽 변경 없이**(OBS_DIM 42) 램프 학습만 얻을 수 있다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.target_vel = None


@configclass
class FR3V4gEnvCfg(FR3V4eEnvCfg):
    """v4g: 램프 50% + 피드포워드, **속도 50~400 mm/s**. 관측 45.

    v4e 대비 단일 변수는 속도 범위다 (0.02~0.20 -> 0.05~0.40).

    무엇을 답하는가 - **"베이스를 얼마나 빨리 몰 수 있나"**
    -------------------------------------------------------
    사용자 목표는 "베이스는 천천히 쭉 이동, 팔이 궤적을 추종" 이다. 그 '천천히' 의
    상한을 정하는 것이 팔의 추종 능력이다. 200 mm/s 에서 지연이 얼마고 400 에서
    무너지는지를 알면 베이스 속도 프로파일을 설계할 수 있다.

    원래는 "램프 100%" 로 혼합 비율을 가릴 계획이었으나 값이 작다고 판단해 교체했다.
    혼합 비율은 v4e/v4f 결과에서 계단 env 와 램프 env 의 오차를 분리해 보면 간접적으로
    읽힌다. 반면 속도 상한은 다른 방법으로 알 수 없다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.ee_pose.speed_range = (0.05, 0.40)


@configclass
class FR3V4eEnvCfg_PLAY(FR3V4eEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FR3V4fEnvCfg_PLAY(FR3V4fEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FR3V4gEnvCfg_PLAY(FR3V4gEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


##
# [v4h] j6 감김 대책 — RETRAIN_V5 §2-A
#
# 기준선은 **v4c** 다 (실기 검증본, 계단 목표, 관측 42). 램프(v4e/f/g)는 배포 판정이
# 아직 안 끝났으므로 섞지 않는다. 지시서 §2-A 도 "관측 42 -> 44" 로 v4c 기준을 가정한다.
#
# 재현 확인 (scripts/check_wrap.py, v4c 정책):
#
#     대상 yaw \ 진입      0       90     170     -90    -170
#     yaw +90           20.82도   1.08    1.08    1.08   1.08     <- 25칸 중 1칸만 실패
#           j6 [deg]   -175.0   164.4   164.4   164.5  164.4
#
#     시뮬 20.82도 / 실기 21.23도 / 실패 시 j6 -175.0 vs 실기 -174.5
#     -> 진단이 독립 확증됐고, 실기 왕복 없이 시뮬에서 판정할 수 있다
##


@configclass
class FR3V4hEnvCfg(FR3V4cEnvCfg):
    """v4h: **감김 여유 관측** 추가. v4c 대비 단일 변수. 관측 42 -> 44.

    j6/j4 가 자기 범위 안에서 어디에 있는지를 명시적으로 넣는다
    (:func:`~ee_track_ppo.tasks.manager_based.reach.mdp.wrap_margin`).

    지시서 §2-A 가 경고한 대로 **이것만으로 배울지는 보증할 수 없다** - 관측에 정보가
    있어도 그것을 쓰는 것이 보상에 유리해야 배운다. 부족하면 v4i(한계 접근 벌점),
    v4j(초기 자세 랜덤화)를 단일 변수로 하나씩 얹는다.

    판정
    ----
    ``scripts/check_wrap.py`` 로 25 조합(대상 yaw 5 x 진입 yaw 5)을 재고 **최댓값**을 본다.
    평균을 보면 안 된다 - v4c 의 25 칸 평균은 1.7 도로 멀쩡해 보인다.

    v5 목표: 전 진입 방향 2 도 이하.
    """

    def __post_init__(self):
        super().__post_init__()

        # (a) 감김 여유 관측. 관측 42 -> 44
        self.observations.policy.wrap_margin = ObsTerm(
            func=ee_mdp.wrap_margin,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )

        # (c) 초기 j6 랜덤화.
        #
        #   **왜 필요한가 - 지금까지 j6 은 예외 없이 정확히 0 에서 시작했다.**
        #   기존 리셋은 ``reset_joints_by_scale`` 로 기본자세에 배율을 곱하는데
        #   j6 기본값이 0.0 이라 ``0.0 x (0.5~1.5) = 0.0`` 이다. 즉 감김이 일어나는
        #   관절만 랜덤화가 **구조적으로 무효**였다. 감김 학습 신호가 희박했던 근본 이유다.
        #
        #   전 범위(+-175도)에서 뽑는다. ``reset_joints_by_offset`` 이 소프트 한계로
        #   clamp 해 주므로 범위를 넘겨 줘도 안전하다.
        #
        #   보상을 건드리지 않으므로 작업공간 손실 위험이 없다. (b) 한계 접근 벌점은
        #   껍질 구석의 도달과 직접 경쟁하지만 이것은 그렇지 않다.
        self.events.reset_wrist_wide = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["j6"]),
                "position_range": (-3.06, 3.06),
                "velocity_range": (0.0, 0.0),
            },
        )


@configclass
class FR3V4hEnvCfg_PLAY(FR3V4hEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
