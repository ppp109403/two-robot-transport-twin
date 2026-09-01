# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""자체 설계 차동구동 AMR 설정.

원본
----
``~/amr_urdf/amr_nmpc.urdf`` + 메쉬는 ``~/amr_urdf/mesh_260730/`` (개정 차체).
원본은 ``data/amr_diff_drive/urdf/amr_nmpc_original.urdf`` 에 그대로 보관하고,
Isaac 용으로 손본 것이 ``amr_isaac.urdf`` 다. 무엇을 왜 바꿨는지는 그 파일 상단
주석에 정리해 두었다 (요약: base_footprint 제거 / collision box 실측 교체 /
질량·관성 실화 / 캐스터 4개 추가 / continuous 조인트 limit 추가).

USD 생성 (한 번만)::

    D=source/ee_track_ppo/ee_track_ppo/assets/data/amr_diff_drive
    python ~/IsaacLab/scripts/tools/convert_urdf.py \
        $D/urdf/amr_isaac.urdf $D/usd/amr_isaac.usd \
        --joint-target-type velocity --joint-stiffness 0.0 --joint-damping 100.0 --headless

``--fix-base`` 를 **주지 않는다** (주행해야 하므로 루트가 자유로워야 한다).

로봇 제원 (URDF/메쉬 실측)
--------------------------
* 구동         : 차동구동 2륜 (``wheel_left_joint`` / ``wheel_right_joint``)
* 휠 반경      : 0.085 m
* 윤거         : 0.29 m (±0.145)
* 차체         : **260730 개정판** ``amr_base.stl`` (매니퓰레이터 고정 프레임 포함)
                 bbox x -0.317~+0.3375 / y ±0.201 / z -0.00005~+0.38495 m
                 상단 z=0.385 가 FR3 마운트 면이다
* base_link    : 지면에서 0.010 m (원본의 base_footprint -> base_link 오프셋)
* 질량         : 차체 70 kg + 구동륜 1.5 kg x2 + 캐스터 0.5 kg x4 = 75 kg (차체는 지정값)
* 캐스터       : 전방 2 + 후방 2, 반경 0.025 m, 선회축 + 구름축 수동 2자유도

차체 콜리전은 프리미티브 박스가 아니라 **STL 메쉬**다 (형상 검증 모델과 동일).
평지 주행 태스크에서는 차체 콜리전이 실제로 쓰이지 않지만(접지는 휠/캐스터,
self-collision 은 off) 검증한 모델과 형상을 일치시키기 위해 그대로 둔다.

가정한 값 (실측치를 알면 고칠 것)
---------------------------------
* 캐스터 x/y 배치 — 메쉬에 정보가 없어 차체 안쪽 모서리로 가정했다.
  전방 x=+0.25 / 후방 x=-0.26, y=±0.15. ``amr_isaac.urdf`` 의 swivel 조인트
  ``origin xyz`` 만 고치면 된다.
* 차체 질량중심 z=0.140 (하부 편중 가정). 균일 박스 중심은 0.192 다.
  질량 70 kg 자체는 지정값이다.
* 구동륜 1.5 kg — 원본 값 0.0285 kg 은 TurtleBot3 잔여값이라 회전관성이
  100 배 가까이 작았다. 그대로 두면 가속 응답이 실제와 전혀 달라진다.

액추에이터 설정 근거
--------------------
구동륜은 속도 제어다. 임플리시트 액추에이터에서 ``stiffness=0`` 이면
PD 제어기가 D 항만 남아 속도 추종기가 된다 (``damping`` 이 속도 이득).
:class:`~ee_track_ppo.tasks.manager_based.nav.mdp.DiffDriveAction` 이
(v, ω) 를 좌/우 휠 각속도 목표로 변환해 이 액추에이터에 넣는다.

effort 15 N*m -> 휠당 176 N, 총 353 N (75 kg 에서 4.7 m/s^2).
실제로는 접지 마찰이 먼저 한계이므로(구동륜 하중 x μ) 슬립이 물리적으로 재현된다.

캐스터는 수동이다. ``stiffness=0, damping=0`` + 작은 effort 한계로 자유회전시킨다.
(0 이 아니면 캐스터가 스스로 돌아 로봇을 밀어버린다)
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from . import ASSETS_DATA_DIR

##
# 로봇 제원 — 로봇을 바꿀 때 여기만 고친다
##

AMR_USD_PATH = os.path.join(ASSETS_DATA_DIR, "amr_diff_drive", "usd", "amr_isaac.usd")
"""변환된 AMR USD 경로."""

AMR_BASE_BODY = "base_link"
"""베이스 링크 이름. 목표 pose 추종의 기준 프레임이자 AMCL 이 추정하는 프레임."""

AMR_LEFT_WHEEL_JOINT = "wheel_left_joint"
AMR_RIGHT_WHEEL_JOINT = "wheel_right_joint"
"""구동륜 조인트 이름. 한쪽에 바퀴가 여러 개면(스키드 스티어) 정규식으로 여러 개를 잡아도 된다."""

AMR_CASTER_JOINTS = "caster_.*"
"""수동 캐스터 조인트 정규식 (선회축 + 구름축)."""

WHEEL_RADIUS = 0.085
"""구동륜 반경 [m]. URDF collision cylinder 실측."""

WHEEL_BASE = 0.29
"""좌우 구동륜 간격(윤거) [m]. URDF 조인트 origin ±0.145."""

INIT_HEIGHT = 0.010
"""바퀴가 지면에 닿을 때의 base_link 높이 [m].

원본 URDF 의 ``base_footprint -> base_link`` 오프셋과 같다.
휠 중심이 base_link 기준 z=0.075 이고 반경이 0.085 이므로 0.085 - 0.075 = 0.010.
틀리면 매 리셋마다 로봇이 땅에 박히거나 낙하한다.
"""

FOOTPRINT_RADIUS = 0.393
"""외접원 반경 [m]. sqrt(0.3375^2 + 0.201^2) = 0.393 (260730 개정 차체 메쉬 bbox 기준)."""


AMR_DIFF_DRIVE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=AMR_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            # 캐스터 바퀴가 가벼워(0.2 kg) 각속도가 튈 수 있어 상한을 둔다
            max_angular_velocity=400.0,
            max_linear_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, INIT_HEIGHT),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        # 구동륜: 속도 제어 (stiffness=0 -> D 제어기)
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[AMR_LEFT_WHEEL_JOINT, AMR_RIGHT_WHEEL_JOINT],
            effort_limit_sim=15.0,
            velocity_limit_sim=15.0,
            stiffness=0.0,
            damping=100.0,
        ),
        # 캐스터: 수동. 자유회전시킨다
        "casters": ImplicitActuatorCfg(
            joint_names_expr=[AMR_CASTER_JOINTS],
            effort_limit_sim=0.5,
            velocity_limit_sim=200.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""자체 설계 차동구동 AMR 설정."""
