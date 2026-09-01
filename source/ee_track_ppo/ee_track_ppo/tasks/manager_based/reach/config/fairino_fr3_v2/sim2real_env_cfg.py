# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fairino FR3 reach **v2 (sim-to-real)**.

v1 정책을 실기에 올린 결과와, 거기서 나온 진단을 반영한 재학습 설정이다.

실기 결과 (v1 정책)
-------------------
    위치 오차 7.9 mm / 자세 오차 0.91 deg / 진동 없음
    (시뮬 4.98 mm / 1.57 deg 대비 위치는 1.6배 나쁘고 자세는 오히려 실기가 낫다)

단, 이 성적은 브리지에서 ``follow_gain = 0.005`` 로 인위적 감쇠를 넣고 속도를 10 deg/s
로 묶어야만 나온다. 그 목발을 빼는 것이 v2 의 목적이다.

원인은 **액추에이터 모델 불일치** 하나로 정리됐다
------------------------------------------------
정책의 액션은 Isaac 관절 PD 의 설정점이다. 시뮬에서 관절은 정책 한 주기(1/30 s) 동안
설정점까지 다 가지 못하고 일부(alpha)만 간다. 그 "못 따라감" 이 폐루프의 감쇠로 작동해
안정성을 만들어 주고 있었다. 반면 실기 ServoJ 는 명령을 **99.4%** 추종한다.
감쇠가 사라지자 정책 자체의 루프이득이 그대로 드러났고, 실측 자기 되먹임 이득
``d(cmd j4)/d(j4) = -1.187`` 이라 발산했다.
(정책을 떼고 고정 목표만 넣은 개루프 시험은 0.04 mm 로 완벽히 정지 -> 하드웨어/브리지 무죄)

방침: 시뮬을 실기에 맞추지 않고 **실기를 시뮬에 맞춘다**
--------------------------------------------------------
``scripts/../step_response.py`` 로 시뮬 alpha 를 실측했다 (kp 400 / kd 40, 정책 30 Hz)::

    계단 0.1 rad : j1..j6 = 0.209 0.169 0.199 0.264 0.278 0.281   평균 0.233
    계단 0.4 rad : j1..j6 = 0.202 0.135 0.147 0.257 0.263 0.263   평균 0.211

강성을 실기 수준으로 올려 alpha 를 0.95 로 만드는 길도 검토했으나 **물리적으로 불가능**하다.
kp 를 10 배(400->4000) 올려도 alpha 는 2.7 배(0.233->0.620) 밖에 안 오른다. 첫 물리 스텝에서
항상 토크 상한(어깨 150 / 손목 28 N*m)에 걸리기 때문이다. 게다가 실제 명령 크기인 0.4 rad
에서는 kp 4000 이어도 alpha 가 0.232 로 되돌아온다.

그래서 반대 방향을 택한다. 브리지의 ``follow_gain`` 은 없애야 할 목발이 아니라
**액추에이터 모델 그 자체**이고, 제거 대상이 아니라 교정 대상이다.
서보 200 Hz / 정책 30 Hz = 6.67 substep 에서 역산하면::

    alpha = 1 - (1 - g)^6.67      ->      g = 1 - (1 - alpha)^(1/6.67)

    alpha 0.211 (전체 평균)  ->  g = 0.0349
    관절별 : j1 0.0333  j2 0.0215  j3 0.0235  j4 0.0436  j5 0.0448  j6 0.0448

즉 **시뮬은 강성·감쇠·보상을 전부 그대로 두고**, 실기 쪽에서 시뮬의 액추에이터를 재현한다.

그러면 v2 에서 바뀌는 것은 **딱 두 가지**다
-------------------------------------------
  [1-B] 지연 주입    : 필수. alpha 만 맞추는 걸로는 부족하다는 게 실패로 증명됐다.
                       실기에서 follow_gain 0.05 (alpha 0.29) 로 올렸을 때, 시뮬의 0.23 과
                       사실상 같은 영역인데도 **108 mm 로 발산**했다. (alpha, 지연) 쌍이
                       맞아야 한다.
  [1-E] 도메인 랜덤화: 필수. 브리지의 비례 필터는 1차 지연이고 시뮬 PD 는 2차 응답이라
                       alpha 를 맞춰도 **응답 형상**은 다르다. 이 차이는 이론으로 맞추기보다
                       랜덤화로 덮는 게 실용적이다.

관측/액션 차원은 그대로다. 배포 쪽(``fr3_rl/policy_runner.py``) 은 **변경 없음**,
``OBS_DIM = 38`` 유지.

[1-C] 6D 회전 표현은 v2 에서 **뺐다** (-> v3 단독 사이클)
--------------------------------------------------------
처음에는 v2 에 같이 넣으려 했으나 두 가지 이유로 미룬다.

  1. 변수 통제. v2 의 목적은 "지연·액추에이터 모델 수정이 통하는가" 하나를 검증하는
     것이다. 회전 표현까지 같이 바꾸면 실패했을 때 원인을 못 가른다.
     4096 x 2000 을 돌려놓고 귀속이 안 되는 게 가장 나쁜 결과다.
  2. 6D 표현은 "행이냐 열이냐 / 평탄화 순서가 뭐냐" 로 값이 갈린다. 두 저장소·두 PC 에
     걸쳐 규약을 맞춰야 하는데, 이건 이번에 하루를 태운 쿼터니언 부호 사고와
     **정확히 같은 계열의 함정**이다. 본 학습과 동시에 할 일이 아니다.

부호 사고 자체는 이미 배포 쪽에서 고치고 실기 검증까지 끝냈다 (성분 우선순위 대신
학습 분포 기준 자세와의 내적으로 반구 결정. 명령 편차 52.98 deg -> 4.31 deg).
시뮬 쪽에 남아 있던 진짜 구멍은 ``ee_pose_b()`` 가 쿼터니언을 **전혀 정규화하지 않고**
``subtract_frame_transforms`` 의 raw 값을 쓰던 것이었다. 배포는 정규화하므로 원리적으로
어긋나 있었고(PhysX 가 일관된 반구를 내놔서 실측상 안 터졌을 뿐), 이번에 배포와 동일한
반구 정규화를 넣어 규약을 맞췄다. 관측 차원도 배포 코드도 그대로다.
(``mdp/observations.py`` 의 :data:`QUAT_HEMISPHERE_REF`)

바꾸지 않는 것 (결과 귀속을 위해)
---------------------------------
보상 가중치, action scale 0.5, use_default_offset, decimation 2, sim.dt 1/60,
기본 관절자세, 커리큘럼, wrist_joint_vel -0.005, 그리고 **작업공간 범위**.
현재 목표 상자는 DLS-IK 도달률 96.7% 로 고른 것이라 넓히면 도달 불가능한 목표가 섞여
위치 보상이 포화하지 못한다. 확장은 다음 사이클에서 따로 한다.

합격 기준
---------
실기에서 ``|d(cmd j4)/d(j4)| < 1``. 그 밖에 제자리 유지 진폭, 위치 오차 7.9 mm 이하,
``follow_gain`` 을 위 표의 값으로 올렸을 때 진동이 없을 것.
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp

import ee_track_ppo.tasks.manager_based.reach.mdp as ee_mdp

from ee_track_ppo.tasks.manager_based.reach.config.fairino_fr3.joint_pos_env_cfg import (  # isort: skip
    FairinoFR3ReachEnvCfg,
)


##
# [1-B] 지연 [물리 스텝]. sim dt = 1/60 s -> 1 스텝 = 16.67 ms.
##
MIN_DELAY_STEPS = 2
"""2 스텝 = 33 ms."""

MAX_DELAY_STEPS = 7
"""7 스텝 = 117 ms.

목표 범위는 **40 ~ 120 ms** 다. 실측치는 72 ms 한 점뿐인데 부하·네트워크·컨트롤러
상태에 따라 변할 수 있고, **지연이 발산의 지배 인자**였으므로 DR 예산을 여기에 쓴다.

지연 단위가 물리 스텝이라 16.67 ms 로 양자화된다. 2~7 스텝 = 33 ~ 117 ms 로,
목표 구간을 **안쪽에서 자르지 않고 바깥에서 감싸는** 쪽을 택했다
(3~7 로 하면 하한이 50 ms 가 되어 40 ms 를 못 덮는다).
"""


##
# [1-E] alpha 랜덤화용 강성 배율
##
STIFFNESS_SCALE = (0.4, 2.5)
"""alpha 를 약 0.15 ~ 0.35 로 흔드는 강성 배율.

주의: **+-30% 로는 부족하다.** 실측한 alpha-kp 관계는 토크 포화 때문에 강하게
sublinear 하다 (kp 400 -> 0.233, 1600 -> 0.436, 4000 -> 0.620, 즉 alpha ~ kp^0.43).
+-30% 를 넣으면 alpha 가 0.20~0.26 밖에 안 흔들려 실기가 낼 수 있는 범위를 못 감싼다.

감쇠는 **독립으로 뽑지 않고** sqrt(배율) 로 함께 걸어 감쇠비를 유지한다.
독립으로 뽑으면 하한 쪽에서 부족감쇠가 되어 시뮬 관절 자체가 링잉하는데,
그건 실기에 없는 현상이라 정책만 과보수적으로 만든다.
(:func:`~ee_track_ppo.tasks.manager_based.reach.mdp.randomize_gains_iso_damping`)
"""


@configclass
class FairinoFR3Sim2RealEnvCfg(FairinoFR3ReachEnvCfg):
    """v1 의 보상/커리큘럼을 그대로 두고 액추에이터 현실성만 올린 설정."""

    def __post_init__(self):
        # v1 전체를 그대로 상속 (보상, 목표 범위, 커리큘럼, 관측 노이즈 ...)
        super().__post_init__()

        # ------------------------------------------------------------------
        # [1-B] 액션 지연 주입
        #   scale / use_default_offset / joint_names 은 v1 과 **완전히 동일**하게 둔다.
        #   바뀌는 것은 설정점이 액추에이터에 도달하는 시각뿐이다.
        # ------------------------------------------------------------------
        self.actions.arm_action = ee_mdp.DelayedJointPositionActionCfg(
            asset_name="robot",
            joint_names=["j[1-6]"],
            scale=0.5,
            use_default_offset=True,
            min_delay=MIN_DELAY_STEPS,
            max_delay=MAX_DELAY_STEPS,
        )

        # 관측은 v1 과 **완전히 동일**하다 (38 차원). 배포 쪽 변경 없음::
        #
        #   [ 0: 6]  joint_pos_rel    (j1..j6, 기본자세 기준 상대각)
        #   [ 6:12]  joint_vel_rel    (j1..j6)
        #   [12:19]  pose_command     (목표 xyz 3 + quat 4, root 프레임)
        #   [19:25]  last_action      (6)
        #   [25:32]  ee_pose          (현재 EE xyz 3 + quat 4, root 프레임)
        #   [32:38]  ee_pose_error    (위치오차 3 + axis-angle 3)
        #
        # ee_pose 의 쿼터니언에 반구 정규화가 들어간 것만 v1 과 다르다. 이건 값의
        # 규약을 배포와 맞춘 것이지 차원이나 의미가 바뀐 게 아니다.
        # (mdp/observations.py 의 QUAT_HEMISPHERE_REF)

        # ------------------------------------------------------------------
        # [1-E] 도메인 랜덤화
        #
        #   mode="startup" 인 이유: 암시적(implicit) 액추에이터의 gain 을 바꾸는 것은
        #   CPU 텐서를 거치므로 매 리셋마다 하면 매우 느리다. env 마다 고정된 gain 을
        #   한 번 뽑아 두면, 4096 개 env 가 곧 4096 개 샘플이라 커버리지는 충분하다.
        #   (지연 쪽은 액션 term 안에서 리셋마다 재추첨되므로 시간축 다양성도 확보된다)
        # ------------------------------------------------------------------
        self.events.randomize_actuator_gains = EventTerm(
            func=ee_mdp.randomize_gains_iso_damping,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["j[1-6]"]),
                "stiffness_scale_range": STIFFNESS_SCALE,
                "distribution": "log_uniform",
            },
        )

        # 관절 마찰. 실기 FR3 의 하모닉 드라이브 마찰은 URDF 에 없고, 이것도 alpha 의
        # 유효값을 낮추는 방향으로 작용한다.
        #   operation="abs" 인 이유: 기본 마찰이 0 이라 "scale" 을 쓰면 결과가 항상 0 이다.
        self.events.randomize_joint_friction = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["j[1-6]"]),
                "friction_distribution_params": (0.0, 0.02),
                "operation": "abs",
                "distribution": "uniform",
            },
        )

        # armature(반사 관성). 감속기 관성이 URDF 에 반영돼 있지 않으면 응답이 실제보다 빠르다.
        self.events.randomize_joint_armature = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["j[1-6]"]),
                "armature_distribution_params": (0.8, 1.25),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        # 링크 질량 +-10%. 실기 툴 플랜지에 붙은 어댑터/케이블 무게가 URDF 에 없다.
        self.events.randomize_link_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[".*"]),
                "mass_distribution_params": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        # 목표 pitch 는 v1 과 동일하게 pi 고정 (작업공간을 안 건드리는 방침).
        # 6D 표현으로 바꿨으므로 이제 특이면이어도 문제가 없다.
        assert self.commands.ee_pose.ranges.pitch == (math.pi, math.pi)


@configclass
class FairinoFR3Sim2RealEnvCfg_PLAY(FairinoFR3Sim2RealEnvCfg):
    """학습 결과 확인/내보내기용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
