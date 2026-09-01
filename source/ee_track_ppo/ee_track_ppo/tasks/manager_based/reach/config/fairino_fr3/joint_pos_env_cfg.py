# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""고정형 매니퓰레이터(Fairino FR3) 의 EE pose 추종 태스크.

``config/franka`` 의 보상 구조를 가져와 FR3 에 맞게 튜닝한다.
버전 번호는 franka 와 분리해서 관리한다 (FR3 v0 = franka v5 와 동일 보상 기준선).
Franka 쪽 파일은 건드리지 않았으므로 두 태스크는 독립적으로 학습/비교 가능하다.

Franka(7DOF) -> FR3(6DOF) 로 오면서 값이 달라진 곳
--------------------------------------------------
  [SWAP 1] 로봇        FRANKA_PANDA_CFG      -> FAIRINO_FR3_CFG
  [SWAP 2] EE 링크     panda_hand            -> wrist3_link
  [SWAP 3] 액션 관절   panda_joint.*         -> j[1-6]
  [SWAP 4] 목표 범위   x (+0.35, +0.65)      -> x (-0.45, -0.25)   <- 부호가 뒤집힌다
  [SWAP 5] 손목 페널티 panda_joint[5-7]      -> j[4-6]

[SWAP 4] 가 왜 음수인가 (중요)
------------------------------
FR3 는 URDF 상 ``q=0`` 에서 팔이 base 기준 **-x** 를 향한다
(FK: EE = base frame (-0.520, -0.102, 0.038)).
``UniformPoseCommand`` 는 **base(root) 프레임** 기준으로 목표를 뽑으므로
범위를 x 음수로 준다.

수치 근거 (URDF FK + damped-least-squares IK 로 300 샘플씩 검증):
  Franka 와 동일한 x(0.35,0.65) 박스 : 위치+자세 동시 달성 IK 성공률 **46.0%**
  x(-0.45,-0.25) z(0.15,0.45) 박스   : 성공률 **96.7%**   <- 채택
도달 불가능한 목표가 섞이면 위치 보상이 원리적으로 포화되지 못한다.
(franka v1 에서 tanh std 를 잘못 잡아 학습 신호가 소멸했던 것과 같은 종류의 함정)

base 회전과 테이블 위치
-----------------------
예전에는 로봇 base 를 z축 180도 돌려서 "base -x" 가 world +x(테이블 쪽)를 향하게 했다.
그 근거로 적혀 있던 "j1 범위 때문에 base +x 가 5도 사각지대"는 **틀린 주장이었고**
(FK 40만 샘플 실측 결과 방위각에 빈 구간 없음. assets/fairino_fr3.py 설명 참고),
실물에서는 매니퓰레이터 x축과 AMR x축이 같은 방향이다.

그래서 base 회전을 없애고(항등) **테이블을 world -x 로 옮겼다.**
팔이 뻗는 방향(base -x)이 그대로 테이블 위가 되므로 작업 배치는 이전과 같다.

★ 이 변경은 학습에 영향이 없다.
  관측(joint_pos/vel, pose_command, ee_pose, ee_pose_error)과 보상이 전부
  **base 프레임 또는 관절 공간**이라 base 의 world 회전이 어디에도 나타나지 않는다.
  따라서 기존 체크포인트(``fr3_reach``)를 그대로 재생/재개할 수 있고 재학습이 필요 없다.
  바뀌는 것은 world 에서 로봇이 어느 쪽을 보고 서 있느냐뿐이다.
"""

import math

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

# 이 프로젝트의 커스텀 보상/관측
import ee_track_ppo.tasks.manager_based.reach.mdp as ee_mdp

##
# 사전 정의된 로봇 설정
##
from ee_track_ppo.assets.fairino_fr3 import FAIRINO_FR3_CFG, FAIRINO_FR3_EE_BODY  # isort: skip


@configclass
class FairinoFR3ReachEnvCfg(ReachEnvCfg):
    """Fairino FR3 로 EE pose 를 추종하는 환경 설정 (보상 튜닝 이력은 __post_init__ 주석 참고)."""

    def __post_init__(self):
        # 부모(ReachEnvCfg)의 기본값 먼저 적용
        super().__post_init__()

        # [SWAP 1] 로봇 교체 -------------------------------------------------
        self.scene.robot = FAIRINO_FR3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # [SWAP 1b] 테이블을 반대편으로 --------------------------------------
        #   FAIRINO_FR3_CFG 가 base 회전을 주지 않게 되면서 팔이 world -x 로 뻗는다.
        #   부모 ReachEnvCfg 의 테이블은 world +x(0.55)에 있으므로 그대로 두면
        #   팔이 테이블 반대편 허공을 향한다. x 부호만 뒤집어 작업 배치를 유지한다.
        #   (학습에는 영향 없음 — 목표/보상이 모두 base 프레임이다. 위 docstring 참고)
        self.scene.table.init_state.pos = (-0.55, 0.0, 0.0)

        # [SWAP 2] EE(엔드이펙터) 링크 이름 -----------------------------------
        # FR3 는 그리퍼가 없으므로 툴 플랜지(wrist3_link)가 곧 EE 다.
        ee_body = FAIRINO_FR3_EE_BODY  # "wrist3_link"
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [ee_body]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [ee_body]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [ee_body]

        # [SWAP 3] 액션: 팔 관절 위치 제어 -------------------------------------
        # scale/use_default_offset 는 franka 와 동일하게 유지 (변수 통제).
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["j[1-6]"],
            scale=0.5,
            use_default_offset=True,
        )

        # [SWAP 4] 목표 pose 커맨드 -------------------------------------------
        self.commands.ee_pose.body_name = ee_body
        # base frame 기준. 위 docstring 의 IK 검증으로 고른 범위다.
        self.commands.ee_pose.ranges.pos_x = (-0.45, -0.25)
        self.commands.ee_pose.ranges.pos_y = (-0.20, 0.20)
        self.commands.ee_pose.ranges.pos_z = (0.15, 0.45)
        # FR3 도 ready 자세에서 wrist3_link 의 z축이 (0,0,-1) 로 아래를 향한다.
        # 따라서 Franka 와 같은 pitch=pi 규약을 그대로 쓸 수 있다. (FK 로 확인함)
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)

        # ---------------------------------------------------------------
        # FR3 보상 튜닝 이력 (franka 와 버전 번호를 분리해서 관리한다)
        #
        # [v0] fr3_reach/2026-07-27_13-59-25_v0_base (1000 iter)
        #      franka v5 와 완전히 동일한 보상으로 시작 (로봇만 교체한 A/B 기준선)
        #        position_error    0.157 -> 0.0574 (최고 0.0450 @it895)
        #        orientation_error 0.530 -> 0.0157 (최고 0.0127 @it989)  <- 프로젝트 전체 최고
        #      실현 보상: 위치 0.1237 / 자세 0.0878+0.1013=0.1891  -> 비율 0.65 : 1 (자세 우세)
        #      같은 보상인데도 franka v5(위치 0.064 / 자세 0.048)보다 자세가 3배 좋다.
        #      FR3 가 자세를 잘 맞추는 만큼 자세 tanh 항이 많이 지급되어 위치가 밀린 구조.
        #
        #      참고: 학습 전 "6DOF 라 여유자유도가 없으니 손목 속도 페널티가 자세를 방해할 것"
        #      이라고 우려했으나, 실측은 반대였다 (wrist_joint_vel -0.0077 로 franka v5 의
        #      -0.0101 보다 작고 자세는 3배 좋음). 그래서 손목 페널티는 그대로 유지한다.
        #
        # [v1] 현재: 위치 오차(5.7cm)를 잡기 위해 franka v5->v6 에서 검증된 조치를 이식.
        #      franka 에서는 이 두 값 변경으로 위치가 0.064 -> 0.004 (16배) 개선되었고
        #      v7 로 재현까지 확인했다.
        #        위치 fine   weight 0.2  -> 0.35
        #        자세 precise weight 0.15 -> 0.08
        #      예상 비율 약 2.1 : 1 (franka v6 의 3.3:1 보다는 약함. FR3 는 자세 보상을
        #      원래 30% 더 많이 벌기 때문에 같은 가중치로도 위치 우세가 덜 만들어진다).
        #      부족하면 다음 수는 위치 weight 를 0.5 로 더 올리는 것.
        #      (franka v7 실험에서 이 영역의 실질적 손잡이는 자세가 아니라 위치 가중치임을 확인)
        # ---------------------------------------------------------------

        # (a) L2 페널티
        self.rewards.end_effector_position_tracking.weight = -0.2
        self.rewards.end_effector_orientation_tracking.weight = -0.2

        # (b) 개별 성형 보상 (접근 단계 담당)
        robot_ee = SceneEntityCfg("robot", body_names=[ee_body])

        self.rewards.end_effector_position_tracking_fine_grained.params["std"] = 0.1
        self.rewards.end_effector_position_tracking_fine_grained.weight = 0.35  # [v1] v0: 0.2

        self.rewards.end_effector_orientation_tracking_fine_grained = RewTerm(
            func=ee_mdp.orientation_command_error_tanh,
            weight=0.1,
            params={"asset_cfg": robot_ee, "std": 0.3, "command_name": "ee_pose"},
        )

        # (c) 관측 추가: 현재 EE pose + 목표까지의 오차 벡터
        #     Franka 는 32 -> 45 차원, FR3 는 25 -> 38 차원이 된다 (실측 확인).
        #       기본  : joint_pos 6 + joint_vel 6 + pose_command 7 + actions 6 = 25
        #       추가  : ee_pose 7 + ee_pose_error 6 = 13
        #     (Franka 가 32 인 것은 손가락 관절 2개까지 포함해 관절이 9개이기 때문)
        self.observations.policy.ee_pose = ObsTerm(
            func=ee_mdp.ee_pose_b,
            params={"asset_cfg": robot_ee},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        self.observations.policy.ee_pose_error = ObsTerm(
            func=ee_mdp.ee_pose_error_b,
            params={"asset_cfg": robot_ee, "command_name": "ee_pose"},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )

        # (d) 자세 정밀 tanh (franka v5 의 e-1)
        #     [v1] weight 0.15 -> 0.08. v0 에서 이 항이 0.1013 을 벌어
        #     위치:자세 비율을 0.65:1 로 만든 최대 원인이었다.
        self.rewards.end_effector_orientation_tracking_precise = RewTerm(
            func=ee_mdp.orientation_command_error_tanh,
            weight=0.08,
            params={"asset_cfg": robot_ee, "std": 0.05, "command_name": "ee_pose"},
        )

        # [SWAP 5] 손목 관절 속도 페널티 (franka v5 의 e-2)
        #   Franka 의 panda_joint[5-7] 에 대응하는 FR3 의 약한 손목 관절은 j[4-6] 이다
        #   (effort 28 N*m vs 어깨 150 N*m -> 구조적으로 떨기 쉬운 쪽).
        #
        #   [주의 / 재진단 필요]
        #   franka v5 의 이 항은 "7DOF 여유자유도(null space)에서 손목이 배회한다"는
        #   진단에 근거했다. FR3 는 6DOF 라 여유자유도가 0 이므로 그 진단은 성립하지 않는다.
        #   FR3 에서 떨림이 나온다면 원인은 null-space drift 가 아니라
        #   IK 해 분기(elbow up/down, wrist flip) 사이의 점프일 가능성이 높다.
        #   일단 franka 와 같은 값으로 두고, baseline 로그를 본 뒤 다시 판단한다.
        self.rewards.wrist_joint_vel = RewTerm(
            func=mdp.joint_vel_l2,
            weight=-0.005,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["j[4-6]"])},
        )

        # (f) 커리큘럼: franka v5 와 동일
        self.curriculum.action_rate.params["weight"] = -0.003
        self.curriculum.joint_vel.params["weight"] = -0.001
        self.curriculum.action_rate.params["num_steps"] = 12000
        self.curriculum.joint_vel.params["num_steps"] = 12000


@configclass
class FairinoFR3ReachEnvCfg_PLAY(FairinoFR3ReachEnvCfg):
    """학습된 정책 시각화(play)용 - 환경 수를 줄이고 관측 노이즈를 끈다."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
