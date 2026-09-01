# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fairino FR3 (모델명 ``fairino3_v6``) 6축 협동로봇 설정.

출처
----
공식 ROS2 저장소 https://github.com/FAIR-INNOVATION/frcobot_ros2
  ``fairino_description/urdf/fairino3_v6.urdf`` + ``meshes/fairino3_v6/*.STL``

원본 URDF 는 ``data/fairino_fr3/urdf/fairino3_v6.urdf`` 에 그대로 보관하고,
``package://`` 경로만 상대경로로 바꾼 것이 ``fairino3_v6_isaac.urdf`` 다.
USD 는 아래 명령으로 한 번만 생성한다 (이미 생성되어 있으면 다시 할 필요 없음)::

    D=source/ee_track_ppo/ee_track_ppo/assets/data/fairino_fr3
    python ~/IsaacLab/scripts/tools/convert_urdf.py \
        $D/urdf/fairino3_v6_isaac.urdf $D/usd/fairino3_v6.usd \
        --fix-base --joint-stiffness 400.0 --joint-damping 40.0 --headless

로봇 제원 (URDF 실측)
--------------------
* DOF        : 6 (``j1`` ~ ``j6``, 전부 revolute)
* 링크       : base_link - shoulder_link - upperarm_link - forearm_link
               - wrist1_link - wrist2_link - **wrist3_link (EE 플랜지)**
* 링크 길이  : j2 z+0.14 / j3 x-0.28 / j4 x-0.24 / j5 z+0.102 / j6 z+0.102
* 질량       : 0.784 / 2.123 / 4.805 / 2.231 / 1.596 / 1.596 / 0.539 kg
* effort     : j1~j3 = 150 N*m,  j4~j6 = 28 N*m
* velocity   : j1~j3 = 3.15 rad/s, j4~j6 = 3.2 rad/s
* 관절 범위  : j1/j5/j6 = +-3.0543,  j3 = +-2.8274,  j2/j4 = -4.6251 ~ 1.4835

주의: 원본 URDF 의 ``<dynamics damping="0" friction="0"/>`` 라서
관절 감쇠는 여기 actuator 설정에서 직접 줘야 한다.

Franka 와 다른 점 (태스크 설정 시 반드시 반영)
---------------------------------------------
1. **6 DOF** (Franka 는 7 DOF). 여유자유도가 없어 IK 해가 이산적이다.
2. **팔의 기본 방향이 base 기준 -x** 다. j3/j4 의 origin 이 x 음수라서
   ``q=0`` 일 때 EE 가 base frame (-0.520, -0.102, 0.038) 에 있다 (FK 확인).
   그래서 base frame 기준 목표 범위는 x 가 음수다.

   .. note::
      예전 주석에는 "``j1`` 범위가 +-3.0543 rad 이라 base +x 는 약 5도 폭의 사각지대다"
      라고 적혀 있었으나 **이는 틀린 내용이었다.** FK 40만 샘플로 실측한 결과
      수평거리 0.10 / 0.30 / 0.50 m 어느 기준에서도 방위각에 빈 구간이 없다
      (24개 빈 x 15도, 정면 +-20도를 2도로 확대해도 동일).
      j1 만 보면 반대편 10도가 비는 것처럼 보이지만 j2/j4 범위가 -265~+85도,
      j3 가 +-150도로 넓어 팔을 접어 반대편으로 뻗을 수 있기 때문이다.
      실제로 있는 것은 사각지대가 아니라 밀도 저하다 (리치 한계에서 방위각
      +10~+18도 구간의 IK 해 밀도가 다른 방위의 60% 수준).

3. **base 회전을 주지 않는다** (``rot`` = 항등).
   실물 FR3 를 AMR 에 올렸을 때 매니퓰레이터 x축과 AMR x축이 같은 방향이고,
   FR3 웹앱으로도 그 배치를 확인했다. 시뮬과 실물의 j1 기준을 일치시켜야
   관절값을 그대로 비교/이식할 수 있으므로 여기서도 회전을 주지 않는다.
   (한때 z축 180도를 주었는데, 그 근거였던 위 "사각지대" 주장이 틀렸다.)

   태스크의 목표/관측/보상이 모두 **base 프레임** 기준이라 이 회전은 학습에 영향이 없다.
   world 상 팔이 뻗는 방향만 바뀌므로, 테이블 위치를 함께 옮겨 맞춘다
   (``config/fairino_fr3/joint_pos_env_cfg.py`` 참고).
3. 그리퍼가 없다. ``wrist3_link`` 가 곧 EE(툴 플랜지)다.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from . import ASSETS_DATA_DIR

FAIRINO_FR3_USD_PATH = os.path.join(ASSETS_DATA_DIR, "fairino_fr3", "usd", "fairino3_v6.usd")
"""변환된 FR3 USD 경로."""

FAIRINO_FR3_EE_BODY = "wrist3_link"
"""엔드이펙터로 쓸 링크 이름 (툴 플랜지). 그리퍼를 달면 여기를 바꾼다."""


FAIRINO_FR3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=FAIRINO_FR3_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        # 회전 없음 (w, x, y, z) = 항등.
        #   실물에서 매니퓰레이터 x축과 AMR x축이 같은 방향이라 시뮬도 그에 맞춘다.
        #   즉 j1=0 이 실물의 j1=0 과 같은 방향을 가리킨다.
        #   팔은 base -x 로 뻗으므로 테이블도 world -x 로 옮겨 놓았다.
        #   (예전에는 (0,0,0,1) = z축 180도였다. 자세한 경위는 모듈 설명 3번 참고)
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # UR 계열 "ready" 자세. FK 검산: EE = base(-0.342, -0.102, 0.420),
            # EE z축 = (0, 0, -1) 즉 아래를 향한다 -> 목표 pitch=pi 와 일치.
            "j1": 0.0,
            "j2": -1.5708,
            "j3": 1.5708,
            "j4": -1.5708,
            "j5": -1.5708,
            "j6": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["j[1-6]"],
            # URDF <limit effort/velocity> 값을 그대로 사용
            effort_limit_sim={"j[1-3]": 150.0, "j[4-6]": 28.0},
            velocity_limit_sim={"j[1-3]": 3.15, "j[4-6]": 3.2},
            stiffness=400.0,
            damping=40.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Fairino FR3 (fairino3_v6) 설정 - 그리퍼 없음, base 고정."""
