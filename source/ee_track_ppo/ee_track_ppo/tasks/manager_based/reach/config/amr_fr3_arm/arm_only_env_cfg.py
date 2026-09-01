# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""**차체 정지** 상태에서 통합 로봇의 팔만 학습한다 (2단계 격리 검증).

왜 이 태스크가 필요한가
-----------------------
통합 학습(``Reach-AMR-FR3-PPO-v0``)의 위치오차가 33 mm 인데, 팔 단독 v4c 는 실기에서
**1.0 mm** 를 냈다. 33 mm 안에 두 종류의 오차가 섞여 있어 무엇이 무엇인지 모른다::

    팔 쪽 격차    tool0 로 EE 가 266 mm 나감, 그리퍼 1.2 kg, 중력보상 방식 변경
    차체 쪽 격차  주행 위치오차, 기울기, 선회 지연

이 태스크는 **차체를 고정**해 뒤쪽을 0 으로 만든다. 남는 차이가 곧 그리퍼가 만든 격차다.

v4c 와 무엇이 같고 다른가
-------------------------
학습 설정(보상 사다리, 액션 지연, DR, 6D 관측)은 v4c 를 그대로 상속한다. 바뀌는 것만::

    로봇       FAIRINO_FR3_CFG  ->  AMR_FR3_CFG (fix_root_link=True)
               차체·그리퍼가 있는 실제 형상. 루트를 고정해 주행만 뺀다
    EE         wrist3_link      ->  tool0
    껍질       v4c 값           ->  1단계에서 tool0 기준으로 재확정한 값
    목표 yaw   base 프레임 고정  ->  **반경방향 기준 ±45도** (j6 감김 해결)

이 태스크가 실기에 올릴 수 있는 첫 후보다 — 차체가 안 움직이므로 전복 위험이 없고,
v4c 배포 절차를 거의 그대로 쓴다.

.. warning::
   **부모 ReachEnvCfg 는 관측·리셋·페널티에서 로봇의 모든 관절을 고른다.**
   AMR 에셋에는 구동륜/캐스터가 있고 그것들은 ``continuous`` 라 관절각이 무한정
   누적된다. 그대로 두면 관측이 발산하고 정규화도 의미를 잃는다.
   아래 ``__post_init__`` 에서 전부 팔 관절로 좁힌다.
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import ee_track_ppo.tasks.manager_based.reach.mdp as ee_mdp  # isort: skip
from ee_track_ppo.assets.amr_fr3 import AMR_FR3_CFG, AMR_FR3_EE_BODY  # isort: skip
from ee_track_ppo.tasks.manager_based.reach.config.fairino_fr3_v4.ladder_env_cfg import (  # isort: skip
    FR3V4cEnvCfg,
)

ARM_JOINTS = ["j[1-6]"]
WRIST_JOINTS = ["j[4-6]"]

##
# 1단계에서 확정한 껍질 (handoff/BASELINE_INTEGRATED.md §1)
##

SHELL_RADIUS = (0.20, 0.50)
SHELL_Z = (0.05, 0.22)
"""**팔 base 축 기준**. 지면 기준 TCP 높이로는 0.435 ~ 0.605 m 다."""

SHELL_AZIMUTH = (-math.radians(120.0), math.radians(120.0))
"""부채꼴 반폭. **±135 에서 ±120 으로 좁혔다** (2026-08-04).

±135 는 1단계에서 **EE=wrist3, pitch±45** 조건으로 정한 값이다. tool0 + 반경방향 yaw
조건에서는 ``+120~+135`` x ``rho 0.20~0.30`` 한 구석에서 j1 이 한계에 눌린다::

    +120~+135도, rho 0.20~0.30   도달률 100%, j1 여유 하위10% 14.8도, 한계15도이내 10.65%

도달은 되는데 한계 코앞이라 정책이 그 구석에서 정밀도를 잃는다. 실측
(arm_j1reset 10000 it, scripts/diagnose_arm.py)::

    az -135~-90  10.04 mm    az   0~+45   4.96 mm
    az  -90~-45   6.75 mm    az  +45~+90  6.87 mm
    az  -45~  0   7.87 mm    az  +90~+135 33.65 mm   <- 여기만 4배

안쪽 반경일수록 y 오프셋 보정 ``arcsin(0.102/rho)`` 이 커져 j1 이 더 밀린다
(rho 0.20 에서 30.7도, rho 0.50 에서 11.8도). 그래서 **안쪽 + 먼 방위각** 조합이 최악이다.

±120 으로 좁히면 부피가 84.1 -> 74.8 L (-11%) 지만 그 구석이 사라진다.
1단계에서 "벽이 절벽형이라 여유를 남긴다" 고 한 것과 같은 판단이다.
"""
PITCH_RANGE = (math.pi - math.radians(10.0), math.pi + math.radians(10.0))

YAW_OFFSET = (-math.radians(45.0), math.radians(45.0))
"""**반경방향 기준** yaw 오프셋. j6 감김을 없앤다.

base 프레임에 yaw 를 고정하면 방위각이 넓을 때 ``j1`` 이 목표를 따라 돌고 ``j6`` 가
그만큼 되돌려야 해서 한계까지 밀린다. 반경방향으로 정의하면 그 보상이 사라진다::

    고정 ±172      j6 여유 하위10% 16.6도,  한계15도이내 9.15%
    반경방향 ±45   j6 여유 하위10% 66.1도,  한계15도이내 0.00%   도달률도 상승
"""

ARM_MOUNT_XYZ = (0.205, 0.0, 0.385)
"""껍질 축을 로봇 root(base_link) 프레임의 어디에 둘지.

팔 단독은 root 가 곧 팔 base 라 (0,0,0) 이지만, 통합 모델은 팔이 차체 위에 있다.
빠뜨리면 껍질이 차체 중심을 축으로 잡혀 팔이 닿지 않는 영역이 섞인다.
"""


@configclass
class AmrFr3ArmEnvCfg(FR3V4cEnvCfg):
    """차체 고정 + 팔만 제어. v4c 설정 위에 에셋/EE/껍질만 교체한다."""

    def __post_init__(self):
        super().__post_init__()

        # --- 로봇 교체: 차체·그리퍼 포함, 루트 고정 -------------------------
        self.scene.robot = AMR_FR3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.fix_root_link = True
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        # 차체가 지면에 닿을 필요가 없다 (루트 고정). 테이블도 치운다 — 차체가 그 자리다.
        self.scene.table = None

        # --- EE 를 tool0 로 ---------------------------------------------------
        #   보상뿐 아니라 **관측·종료 항도** 같이 바꿔야 한다. 하나라도 빠뜨리면
        #   "Not all regular expressions are matched" 로 죽는다 (wrist3_link 가 없으므로).
        ee = AMR_FR3_EE_BODY  # "tool0"

        def _retarget_ee(group):
            for name in dir(group):
                if name.startswith("_"):
                    continue
                term = getattr(group, name, None)
                params = getattr(term, "params", None) if term is not None else None
                if isinstance(params, dict) and isinstance(params.get("asset_cfg"), SceneEntityCfg):
                    if params["asset_cfg"].body_names is not None:
                        params["asset_cfg"].body_names = [ee]

        _retarget_ee(self.rewards)
        _retarget_ee(self.observations.policy)
        _retarget_ee(self.terminations)

        # --- 관절 선택을 팔로 좁힌다 (위 warning 참고) -----------------------
        #   부모는 asset_cfg 없이 전체 관절을 고른다. 구동륜은 continuous 라 발산한다.
        self.observations.policy.joint_pos.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)
        }
        self.observations.policy.joint_vel.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)
        }
        # --- 리셋 랜덤화: by_scale -> by_offset, 그리고 j1 을 넓게 -------------
        #
        #   **부모(IsaacLab ReachEnvCfg)는 ``reset_joints_by_scale`` 에 (0.5, 1.5) 를 쓴다.**
        #   기본값에 배율을 곱하므로 기본값이 0 인 관절은 0 x [0.5,1.5] = 0 이 되어
        #   **영원히 랜덤화되지 않는다.** 기본자세는 j1=0, j6=0 이라 정확히 그 둘이다.
        #
        #   그 결과 정책은 매 에피소드를 j1=0 에서 시작한다. 그런데 확정 껍질의
        #   방위각 ±135도는 j1 을 -166 ~ +105도까지 쓴다. 특히 az +90~+135 대역은
        #   j1 중앙값이 **-130도**라, 정책은 그 자세에서 시작해 본 적이 한 번도 없다.
        #
        #   실측 (arm_tool0 10000 it, scripts/diagnose_arm.py) — 그 대역만 무너진다::
        #
        #       az -135~-90    15.38 mm      az   0~+45    20.31 mm
        #       az  -90~-45     9.56 mm      az  +45~+90   20.44 mm
        #       az  -45~  0    10.48 mm      az  +90~+135 196.59 mm   <- 10배
        #
        #   기구학 문제가 아니다 — 같은 분포에서 IK 도달률 99.42%, j1 여유 하위10%
        #   26.9도, 자기충돌 0.04% 다. 순수하게 **그 자세를 연습한 적이 없어서**다.
        #
        #   v4c 는 부채꼴이 ±60도라 j1 범위가 좁아 이 결함이 덜 드러났다.
        self.events.reset_robot_joints.func = ee_mdp.reset_joints_by_offset
        #   j2/j3 는 ±0.25 로 좁게 준다. ±0.45 면 어깨·팔꿈치가 차체와 겹친 자세로
        #   놓여 자기충돌이 3.47% 발생한다 (리셋 자세 충돌 검사, 2026-08-04).
        #   차체 고정이라 전복은 안 나지만 리셋 직후 관절이 튕기는 것은 같다.
        self.events.reset_robot_joints.params = {
            "position_range": (-0.25, 0.25),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["j[2-3]"]),
        }
        self.events.reset_arm_rest = EventTerm(
            func=ee_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.60, 0.60),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=["j[4-5]"]),
            },
        )
        # j1 은 방위각 전 범위를 덮도록 넓게. 한계(±175도)로 자동 클램프된다.
        self.events.reset_j1_wide = EventTerm(
            func=ee_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-2.9, 1.9),   # 약 -166 ~ +109도. 필요한 j1 범위
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=["j1"]),
            },
        )
        # j6 도 기본값이 0 이라 같은 이유로 랜덤화되지 않았다. 반경방향 yaw 로
        # 감김은 해결됐지만(여유 중앙 106.6도), 초기 산포는 여전히 있어야 한다.
        self.events.reset_j6_wide = EventTerm(
            func=ee_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-1.5, 1.5),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=["j6"]),
            },
        )
        for name in ("joint_vel", "wrist_joint_vel", "arm_joint_acc"):
            term = getattr(self.rewards, name, None)
            if term is not None and isinstance(getattr(term, "params", None), dict):
                jn = WRIST_JOINTS if name == "wrist_joint_vel" else ARM_JOINTS
                term.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=jn)

        # --- 커맨드: tool0 껍질 + 반경방향 yaw --------------------------------
        self.commands.ee_pose = ee_mdp.CylindricalPoseCommandCfg(
            asset_name="robot",
            body_name=ee,
            resampling_time_range=(4.0, 4.0),
            debug_vis=True,
            origin_offset=ARM_MOUNT_XYZ,
            yaw_relative_to_radial=True,
            ranges=ee_mdp.CylindricalPoseCommandCfg.Ranges(
                radius=SHELL_RADIUS,
                azimuth=SHELL_AZIMUTH,
                pos_z=SHELL_Z,
                pos_x=(0.0, 0.0),   # 껍질 커맨드에서 안 쓰이지만 부모 필드라 값이 필요
                pos_y=(0.0, 0.0),
                roll=(0.0, 0.0),
                pitch=PITCH_RANGE,
                yaw=YAW_OFFSET,
            ),
        )


@configclass
class AmrFr3ArmEnvCfg_PLAY(AmrFr3ArmEnvCfg):
    """학습 결과 확인용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
