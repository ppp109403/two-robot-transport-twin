# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMR 베이스 + Fairino FR3 팔 = 모바일 매니퓰레이터 설정.

구성 (260730 검증 모델과 동일 좌표)
-----------------------------------
* 베이스 : 자체 설계 차동구동 AMR 개정판 (매니퓰레이터 고정 프레임 포함)
* 팔     : Fairino FR3 (fairino3_v6), 6축
* 그리퍼 : 파이퍼 2핑거 (prismatic 2개, 편측 행정 0.035 m -> 총 개구 70 mm)
* 마운트 : 고정 프레임 상단 **x=+0.205, z=+0.385, 회전 없음**
* EE     : ``tool0`` = 두 손가락 사이 TCP

전부 Isaac Sim 에서 형상을 눈으로 확인하고 확정한 값이다
(``scripts/tools/view_amr_fr3_260730.py``).

URDF 는 손으로 합치지 않고 스크립트로 생성한다. 베이스/팔/그리퍼 좌표가 바뀌면
다시 돌려야 반영된다::

    python scripts/tools/build_amr_fr3_urdf.py

    D=source/ee_track_ppo/ee_track_ppo/assets/data/amr_fr3
    python ~/IsaacLab/scripts/tools/convert_urdf.py \
        $D/urdf/amr_fr3.urdf $D/usd/amr_fr3.usd \
        --joint-target-type position \
        --joint-stiffness 0.0 --joint-damping 0.0 --headless

``--fix-base`` 를 **주지 않는다** (주행해야 하므로 루트가 자유로워야 한다).
``--merge-joints`` 는 **쓰지 않는다.** 고정 조인트를 합치면 ``arm_base_link`` /
``gripper_body_link`` / ``tool0`` 이 부모에 흡수되어 사라지는데, 태스크가 ``tool0``(TCP)를
EE body 로 찾아야 한다.
``--joint-stiffness/--joint-damping`` 은 0 으로 두고 아래 액추에이터 설정이 전부 결정한다.

이름 규칙
---------
팔 링크에는 ``arm_`` 접두사가 붙는다 (두 URDF 모두 ``base_link`` 를 갖고 있었다).
조인트 이름 ``j1``~``j6`` 은 충돌하지 않아 그대로여서, FR3 단독 태스크와 같은
정규식 ``j[1-6]`` 을 쓸 수 있다.

팔 마운트 회전이 없는 이유
--------------------------
실물에서 매니퓰레이터 x축과 AMR x축이 같은 방향이고 FR3 웹앱으로도 확인했다.
시뮬과 실물의 j1 기준을 일치시켜야 관절값을 그대로 비교/이식할 수 있다.

한때 z축 180도를 주었고 근거는 "j1 범위 때문에 base +x 가 5도 사각지대"였는데,
**그 주장은 틀렸다** (FK 40만 샘플 실측 결과 방위각에 빈 구간 없음.
:mod:`~ee_track_ppo.assets.fairino_fr3` 설명 참고).

질량
----
베이스 75 kg (차체 70 + 휠/캐스터 5) + 팔 13.7 kg + 그리퍼 1.2 kg (몸통 1.0 + 손가락 0.1 x2)
= 약 90 kg. 차체/그리퍼 질량은 지정값이고 팔은 FR3 URDF 원본 값이다.
팔과 그리퍼가 앞쪽 위에 붙어 질량중심이 전방·상방으로 이동하므로 정지 시
**전방 캐스터**로 지지된다.

.. note::
   그리퍼 1.2 kg 은 FR3 페이로드(3 kg)의 40% 다. 여기에 XJC 힘센서와 마운트 프레임,
   카메라가 더해지면 실제 집을 수 있는 무게가 크게 줄어든다. 툴 어셈블리 질량이
   확정되면 손목 처짐과 페이로드 여유를 다시 확인해야 한다.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from . import ASSETS_DATA_DIR
from .amr_diff_drive import (
    AMR_CASTER_JOINTS,
    AMR_LEFT_WHEEL_JOINT,
    AMR_RIGHT_WHEEL_JOINT,
    INIT_HEIGHT,
    WHEEL_BASE,
    WHEEL_RADIUS,
)

__all__ = [
    "AMR_FR3_CFG",
    "AMR_FR3_USD_PATH",
    "AMR_FR3_BASE_BODY",
    "AMR_FR3_EE_BODY",
    "AMR_FR3_FLANGE_BODY",
    "AMR_FR3_GRIPPER_JOINTS",
    "AMR_FR3_ARM_JOINTS",
    "AMR_FR3_WRIST_JOINTS",
    "AMR_FR3_LEFT_WHEEL_JOINT",
    "AMR_FR3_RIGHT_WHEEL_JOINT",
    "AMR_FR3_WHEEL_RADIUS",
    "AMR_FR3_WHEEL_BASE",
    "AMR_FR3_INIT_HEIGHT",
    "ARM_MOUNT_XYZ",
    "ARM_MOUNT_RPY",
]

AMR_FR3_USD_PATH = os.path.join(ASSETS_DATA_DIR, "amr_fr3", "usd", "amr_fr3.usd")
"""변환된 모바일 매니퓰레이터 USD 경로."""

AMR_FR3_BASE_BODY = "base_link"
"""베이스 링크. 주행/로컬라이제이션의 기준 프레임."""

AMR_FR3_EE_BODY = "tool0"
"""엔드이펙터로 쓸 링크 = **TCP** (파이퍼 그리퍼 두 손가락 사이 파지 기준점).

그리퍼를 달기 전에는 ``arm_wrist3_link``(툴 플랜지)를 EE 로 썼다. 그대로 두면
플랜지 기준으로 학습해서 실제 파지점이 266 mm 어긋난다.
  wrist3_link 원점(j6 축) -> 플랜지 면 100 mm -> 그리퍼 마운트에서 TCP 166 mm
"""

AMR_FR3_FLANGE_BODY = "arm_wrist3_link"
"""FR3 툴 플랜지 링크. 링크 원점은 플랜지 면이 아니라 j6 회전축이다 (100 mm 뒤)."""

AMR_FR3_GRIPPER_JOINTS = ["gripper_finger_.*"]
"""파이퍼 그리퍼 손가락 조인트 (prismatic 2개, 편측 행정 0.035 m).

이 태스크는 EE pose 추종이라 정책이 그리퍼를 제어하지 않는다.
액추에이터만 붙여 기본 자세(닫힘)를 유지시킨다.
"""

AMR_FR3_ARM_JOINTS = ["j[1-6]"]
"""팔 조인트 정규식."""

AMR_FR3_WRIST_JOINTS = ["j[5-6]"]
"""손목 조인트. 도달 후 떨림 억제 페널티를 따로 거는 대상이다.

FR3 는 6축이라 손목이 j5/j6 이다 (Franka 7축의 panda_joint[5-7] 에 대응).
"""

# 베이스 제원은 AMR 단독 에셋과 같은 값을 쓴다 (같은 URDF 에서 왔다)
AMR_FR3_LEFT_WHEEL_JOINT = AMR_LEFT_WHEEL_JOINT
AMR_FR3_RIGHT_WHEEL_JOINT = AMR_RIGHT_WHEEL_JOINT
AMR_FR3_WHEEL_RADIUS = WHEEL_RADIUS
AMR_FR3_WHEEL_BASE = WHEEL_BASE
AMR_FR3_INIT_HEIGHT = INIT_HEIGHT

ARM_MOUNT_XYZ = (0.205, 0.0, 0.385)
"""AMR base_link 기준 팔 마운트 위치 [m].

260730 개정 차체의 매니퓰레이터 고정 프레임 상단이다. 실측 지정값이며
``build_amr_fr3_260730_urdf.py`` 의 ``ARM_MOUNT_XYZ`` 와 같아야 한다
(결합 URDF 생성기가 그 값을 import 한다).
"""

ARM_MOUNT_RPY = (0.0, 0.0, 0.0)
"""팔 마운트 자세. 회전 없음 = 매니퓰레이터 x축과 AMR x축이 같은 방향 (실물 확인)."""


AMR_FR3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=AMR_FR3_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            max_angular_velocity=400.0,
            max_linear_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, AMR_FR3_INIT_HEIGHT),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # 구동륜 / 캐스터
            "wheel_.*": 0.0,
            "caster_.*": 0.0,
            # 팔: UR 계열 "ready" 자세 (FR3 단독 태스크와 동일)
            "j1": 0.0,
            "j2": -1.5708,
            "j3": 1.5708,
            "j4": -1.5708,
            "j5": -1.5708,
            "j6": 0.0,
            # 그리퍼: 닫힘 (정책이 제어하지 않고 이 자세를 유지한다)
            "gripper_finger_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # 구동륜: 속도 제어 (stiffness=0 -> D 제어기)
        #   2026-08-03 에 damping 400 / effort 40 을 시도했다가 **되돌렸다.**
        #   "도달 후 차체와 팔이 함께 떤다" 는 관찰에 대해 역구동 저항이 부족한 것으로
        #   가정한 조치였는데, 진짜 원인은 **캐스터가 2 mm 떠 있어 차체가 구동륜 축을
        #   중심으로 기우는 것**이었다 (build_amr_fr3_260730_urdf.py 의 CASTER_MOUNT_Z 참고).
        #   바퀴 damping 은 바퀴 회전을 막지 차체 기울기를 막지 않으므로 무관한 변경이었다.
        #
        #   실물 감속기의 역구동 저항을 반영하는 것 자체는 여전히 검토할 가치가 있으나,
        #   **실물 모터 제원(감속비, 정지 토크) 없이 추정으로 바꾸지 않는다.**
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[AMR_FR3_LEFT_WHEEL_JOINT, AMR_FR3_RIGHT_WHEEL_JOINT],
            effort_limit_sim=15.0,
            velocity_limit_sim=15.0,
            stiffness=0.0,
            damping=100.0,
        ),
        # 캐스터: 수동. 자유회전
        "casters": ImplicitActuatorCfg(
            joint_names_expr=[AMR_CASTER_JOINTS],
            effort_limit_sim=0.5,
            velocity_limit_sim=200.0,
            stiffness=0.0,
            damping=0.0,
        ),
        # 팔: 위치 제어.
        #   effort/velocity 는 FR3 URDF <limit> 값 그대로.
        #
        #   damping 은 **40** 이다. 한때 80 을 썼는데(ridgeback 태스크의 도달 후 떨림을
        #   잡으려던 값), 2026-07-31 계단응답 측정에서 **응답성을 40% 깎고 있다**는 것이
        #   드러나 되돌렸다::
        #
        #       조건                        평균 alpha   follow_gain
        #       그리퍼X, kd 40              0.221        0.0370
        #       그리퍼O, kd 40              0.225        0.0377      <- 그리퍼 영향 +1.8%
        #       그리퍼O, kd 80              0.134        0.0213      <- 감쇠가 40% 를 깎는다
        #
        #   손목이 토크가 아니라 **속도 상한**(3.2 rad/s)에 걸려 있어서다. kd 80 이면
        #   kd*q̇ = 256 N*m 의 감쇠토크가 effort 상한 28 N*m 를 압도해 속도 상한에
        #   도달하지 못한다.
        #
        #   kd 40 은 실기로 검증된 값이다 — v4c 정책(팔 단독 에셋, kd 40)의 실측 시정수
        #   0.155~0.160 s 가 시뮬 예측 0.16 s 와 유효숫자 3 자리로 일치했다.
        #   kd 80 은 FR3 실기로 검증된 적이 없다.
        #
        #   떨림이 재발하면 감쇠가 아니라 **행동 변화율 벌점**으로 잡을 것.
        #   그쪽은 실기 응답 모델을 건드리지 않는다.
        "arm": ImplicitActuatorCfg(
            joint_names_expr=AMR_FR3_ARM_JOINTS,
            effort_limit_sim={"j[1-3]": 150.0, "j[4-6]": 28.0},
            velocity_limit_sim={"j[1-3]": 3.15, "j[4-6]": 3.2},
            stiffness=400.0,
            damping=40.0,
        ),
        # 그리퍼: 위치 유지용.
        #   이 태스크는 EE pose 추종이라 정책이 그리퍼를 열고 닫지 않는다.
        #   액추에이터가 없으면 손가락이 중력/관성으로 흘러내려 TCP 가 흔들린다.
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=AMR_FR3_GRIPPER_JOINTS,
            effort_limit_sim=60.0,
            velocity_limit_sim=0.1,
            stiffness=800.0,
            damping=40.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""AMR + FR3 모바일 매니퓰레이터 설정."""
