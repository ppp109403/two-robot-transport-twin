# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""모바일 매니퓰레이터(Clearpath Ridgeback + Franka Panda)의 EE pose 추종 태스크.

Isaac Lab 에는 이 로봇의 RL 태스크가 없다(에셋 ``RIDGEBACK_FRANKA_PANDA_CFG`` 와
데모 스크립트 ``source/isaaclab/test/assets/check_ridgeback_franka.py`` 만 존재).
그래서 내장 ``ReachEnvCfg`` 의 구조를 그대로 따라가되, 모바일 베이스에 맞게 새로 정의한다.

고정팔(Franka) 버전과 다른 점 4가지
-----------------------------------
1. 씬     : 테이블 제거, 지면을 z=0 으로 (고정팔 태스크는 테이블 위 설치라 지면이 z=-1.05)
2. 액션   : 팔(위치 제어) + 베이스(속도 제어) 2개로 분리
3. 커맨드 : 목표 범위를 팔 작업공간보다 넓게 -> 베이스가 움직이지 않으면 도달 불가
4. 보상   : 거리 스케일이 크므로 coarse/fine tanh 2단계 + 베이스 과잉 주행 페널티

베이스 모델링에 대한 중요한 사실 (실측 확인함)
----------------------------------------------
Ridgeback 은 ``dummy_base_prismatic_x/y_joint`` + ``dummy_base_revolute_z_joint``
(홀로노믹 planar 3-DOF)로 주행을 표현한다. 그런데 실제로 띄워서 확인해 보면
``is_fixed_base = False`` 이고, 베이스 조인트를 구동하면 ``root_pos_w`` 가 함께 이동한다.

따라서 Isaac Lab 기본 ``UniformPoseCommand`` (목표를 **로봇 root 프레임**에서 샘플링)를 쓰면
목표가 로봇에 붙어 따라다녀서 **베이스가 주행할 이유가 사라진다.**
그래서 이 태스크는 env 원점 기준으로 목표를 고정하는 커스텀 커맨드
:class:`~ee_track_ppo.tasks.manager_based.reach.mdp.UniformPoseWorldCommand` 를 쓴다.
(보상도 월드 좌표 목표를 쓰는 ``*_w`` 버전을 사용)

나중에 직접 만든 AMR + FR3 로 바꿀 때
-------------------------------------
- 베이스가 차동구동(논홀로노믹)이면 ``base_action`` 을 좌/우 휠 속도 2개로 바꾸거나
  ``mdp.NonHolonomicActionCfg`` 를 쓴다.
- 커맨드/보상은 이미 월드 고정 방식이라 그대로 재사용 가능하다.
"""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# 이 프로젝트의 mdp 모듈 (= isaaclab.envs.mdp 전체 + 월드 고정 목표 커맨드/보상)
import ee_track_ppo.tasks.manager_based.reach.mdp as mdp

##
# 사전 정의된 로봇 설정
##
from isaaclab_assets.robots.ridgeback_franka import RIDGEBACK_FRANKA_PANDA_CFG  # isort: skip
from ee_track_ppo.assets import ASSETS_DATA_DIR  # isort: skip


# EE 링크 이름 / 관절 정규식 — 로봇 교체 시 여기만 바꾸면 된다
EE_BODY = "panda_hand"
ARM_JOINTS = ["panda_joint.*"]
BASE_JOINTS = ["dummy_base_.*"]


##
# 씬 정의
##


@configclass
class MobileReachSceneCfg(InteractiveSceneCfg):
    """모바일 매니퓰레이터 + 평지 씬."""

    # 지면 (모바일이므로 z=0)
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # 로봇
    #
    # [v6] init_state.pos 를 실제 지면 안착 자세로 지정한다. 이게 없으면 리셋이 깨진다.
    #
    #   RIDGEBACK_FRANKA_PANDA_CFG 는 init_state.pos 를 지정하지 않아 기본값이 (0,0,0) 이다.
    #   USD 로 스폰할 때는 문제가 없지만, v5 에서 추가한 reset_root_state_uniform 은
    #   **default_root_state(=init_state) 기준으로 루트를 배치**하므로
    #   매 리셋마다 로봇을 지면 아래 0.61 m 에 꽂아 넣게 된다.
    #
    #   실측 (v5 진단):
    #     step  0 : root z = 0.000  (정상 안착 높이는 0.609)
    #     step 10 : 루트 속도 6.70 m/s 로 튕겨 오름
    #     step 30 : 원점에서 3.6 m 날아감
    #     step300 : 4 m 밖에 기울어진 채 정착 (quat 이 수직이 아님)
    #   -> 정책이 "목표 추종"이 아니라 "발사된 뒤 기어서 돌아오기"를 학습하고 있었다.
    #      GUI 로 봐야만 보이는 종류의 버그다 (수치상으로는 위치 오차가 좀 큰 정도로만 보인다).
    #
    #   [v7] ★ 결정적 사실: 이 articulation 의 **루트 링크는 base 가 아니라 `panda_link2`**(팔 링크)다.
    #        (실측: body_names[0] == "panda_link2" 이고 root_pos_w == body_pos_w[:,0])
    #        따라서 reset_root_state_uniform 은 "팔 중간 링크"를 지정 pose 로 순간이동시키고
    #        나머지 몸통이 딸려가게 만든다. rot 를 항등(1,0,0,0)으로 두면 로봇이 통째로 뒤틀려 눕는다.
    #        v6 에서 "안착 높이 0.395" 로 본 것은 **누운 상태**의 값이었다.
    #
    #        아래 값은 루트 리셋을 끄고 기본 관절자세로 안정화시켜 실측한 것이다.
    #          root(panda_link2) = (-0.0520, 0.0168, 0.6092), quat (-0.6726, 0.6855, 0.1734, 0.2182)
    #          base_link         = (-0.3514, 0.0366, -0.0038)   <- 지면에 접지
    #        base_link 가 env 원점에 오도록 루트를 평행이동한 값을 init_state.pos 로 쓴다.
    #   [v10] ★ 근본 해결: `world` 링크를 월드에 고정한 USD 사본을 쓴다.
    #
    #     원인 확정 — 원본 USD 는 ArticulationRootAPI 가 최상위 Xform(/panda_mobile)에 있어
    #     PhysX 가 루트 바디로 `panda_link2`(팔 링크)를 골랐다. 그 결과 운동학 트리가
    #     팔에서 시작해 베이스로 내려가는 형태가 되어, planar 조인트를 구동해도
    #     질량 없는 말단 더미 링크만 움직이고 **로봇은 제자리였다** (실측 0.000 m/s).
    #
    #     `world` 링크(RigidBodyAPI 보유)를 fixed joint 로 월드에 고정하면
    #     루트가 `world` 가 되고 planar 조인트가 로봇을 실제로 밀어낸다.
    #     실측: 명령 1.0 -> 관절속도 1.000 m/s (명령값 그대로), 관절 한계 +-600 m.
    #
    #     이 사본은 tools/ 없이 pxr 로 생성했다 (원본을 reference 하고 fixed joint 만 추가).
    #     재생성이 필요하면 assets/data/ridgeback_franka/ 의 파일을 지우고
    #     scratchpad 의 patch_ridgeback.py 와 같은 방식으로 다시 만들면 된다.
    #
    #     루트가 고정되었으므로 v6~v7 에서 넣었던 init_state.pos/rot 실측 보정과
    #     reset_root_state_uniform 이벤트는 모두 불필요해졌다.
    robot: ArticulationCfg = RIDGEBACK_FRANKA_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=RIDGEBACK_FRANKA_PANDA_CFG.spawn.replace(
            usd_path=os.path.join(ASSETS_DATA_DIR, "ridgeback_franka", "ridgeback_franka_anchored.usd"),
        ),
    )

    # 조명
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP 설정
##


@configclass
class CommandsCfg:
    """목표 EE pose 생성기."""

    # ★ 월드(env 원점) 고정 목표. ranges 는 로봇 기준이 아니라 env 원점 기준이다.
    ee_pose = mdp.UniformPoseWorldCommandCfg(
        asset_name="robot",
        body_name=EE_BODY,
        # 한 에피소드(15초) 안에서 목표를 두 번 바꾼다
        resampling_time_range=(7.0, 7.0),
        debug_vis=True,  # 목표 pose 마커를 GUI 에 표시
        ranges=mdp.UniformPoseWorldCommandCfg.Ranges(
            # 팔 리치(~0.85 m)를 훨씬 넘는 범위 -> 베이스 주행이 필수
            pos_x=(-1.5, 1.5),
            pos_y=(-1.5, 1.5),
            pos_z=(0.4, 0.9),
            roll=(0.0, 0.0),
            pitch=(math.pi, math.pi),  # Franka 는 EE 축이 z 방향
            yaw=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """액션: 팔(위치) + 베이스(속도) 분리."""

    # 팔: 관절 위치 제어. 기본자세 기준 상대 오프셋
    arm_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        scale=0.5,
        use_default_offset=True,
    )

    # 베이스: 관절 속도 제어.
    # RIDGEBACK_FRANKA_PANDA_CFG 에서 base actuator 가 stiffness=0, damping=1e5 로
    # 설정되어 있어 사실상 속도 제어기다. scale=1.0 -> 정책 출력이 m/s, rad/s.
    base_action: ActionTerm = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=BASE_JOINTS,
        scale=1.0,
    )


@configclass
class ObservationsCfg:
    """관측 정의."""

    @configclass
    class PolicyCfg(ObsGroup):
        # 순서가 그대로 관측 벡터 순서가 된다
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # 목표 pose (로봇 root 프레임 기준 7차원: xyz + quat)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        # [v1] 현재 EE pose + 목표까지의 오차 벡터.
        #   고정형 팔 실험에서 가장 큰 효과를 낸 항목이다 (franka v3->v4 에서
        #   위치 오차 0.220 -> 0.011 로 20배 개선). 정책이 자기 EE 위치를 직접 보지 못하면
        #   순기구학을 신경망 안에서 암묵 학습해야 하고 그것이 정밀도의 병목이 된다.
        #   커맨드가 root 프레임 기준이라 모바일에도 그대로 쓸 수 있다.
        ee_pose = ObsTerm(
            func=mdp.ee_pose_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[EE_BODY])},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        ee_pose_error = ObsTerm(
            func=mdp.ee_pose_error_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[EE_BODY]), "command_name": "ee_pose"},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """리셋 이벤트."""

    # 팔 관절 리셋.
    #   [v7] (0.5, 1.5) -> (1.0, 1.0), [v10] 다시 (0.5, 1.5) 로 복원.
    #   v7 의 고정은 "루트가 팔 링크라 팔을 흔들면 베이스 위치가 달라진다"는
    #   제약 때문이었는데, v10 에서 루트가 world 로 바뀌어 그 제약이 사라졌다.
    #   이유: 이 로봇의 articulation 루트가 팔 링크(panda_link2)라서,
    #   articulation 상태 = (루트 pose + 관절값) 으로 모든 링크 위치가 결정된다.
    #   루트를 고정 pose 에 놓고 팔 관절만 무작위로 바꾸면 **베이스가 어디에 놓일지 달라져서**
    #   땅에 박히거나 공중에 뜬다. 팔 초기자세 다양성은 포기하고 정합성을 택한다.
    #   (태스크 다양성은 목표 pose 무작위화가 담당한다)
    reset_arm_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS),
        },
    )

    # 베이스는 위치/속도 모두 0 으로 리셋
    reset_base_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=BASE_JOINTS),
        },
    )

    # ---------------------------------------------------------------
    # [v5] ★ 루트(root) 상태 리셋 - 이게 빠져 있던 것이 v1~v4 실패의 진짜 원인이었다.
    #
    #   Ridgeback 은 is_fixed_base=False 라 로봇 전체가 떠다니는 자유 루트를 가진다.
    #   그런데 상속한 IsaacLab reach 태스크의 EventCfg 는 **고정형 팔 전용**이라
    #   관절만 리셋하고 루트 pose 를 되돌리는 이벤트가 없다.
    #   그 결과 에피소드가 끝나도 로봇이 표류한 자리에 그대로 남고, 표류가 누적된다.
    #
    #   실측 (v4 진단):
    #     관절값 prismatic_x 는 2.5 를 넘어 종료가 정상적으로 걸리고 0 으로 리셋되지만
    #     root-원점 거리는 0.35 -> 2.80 m 로 계속 누적되었다.
    #     본 학습에서는 이것이 쌓여 position_error 가 18 m 까지 갔다.
    #     (자세 오차는 베이스 위치와 무관하므로 0.019 rad 까지 잘 내려갔다)
    #
    #   locomotion 계열 태스크에는 반드시 들어가는 항목이다.
    # ---------------------------------------------------------------
    #   [v10] 루트 리셋 이벤트 삭제.
    #   `world` 고정으로 articulation 이 fixed-base 가 되었으므로
    #   루트 pose 를 되돌릴 필요가 없다 (관절 리셋만으로 완전히 초기화된다).


@configclass
class RewardsCfg:
    """보상 정의."""

    # --- 태스크 보상 (모두 월드 고정 목표 기준: *_w) -------------------------
    # [v4] 거리 L2 페널티 제거 (weight 0.0).
    #   상한이 없는 유일한 항이었고 v1 폭주(value_loss 1843 -> NaN)의 직접 원인이었다.
    #   아래 tanh 사다리만으로 전 구간 신호를 만들 수 있으므로 굳이 남길 이유가 없다.
    #   (0.0 이면 RewardManager 가 계산 자체를 건너뛴다)
    ee_position_tracking = RewTerm(
        func=mdp.position_command_error_w,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[EE_BODY]), "command_name": "ee_pose"},
    )
    # coarse: 먼 거리 담당. [v2] std 0.8 -> 1.5, [v4] std 1.5 -> 2.5 / weight 0.3 -> 0.5
    #   v3 실측에서 이 항이 4 m 지점에 0.0028(pre-dt) 밖에 주지 못해
    #   생존 보너스(0.0333/스텝)에 300 배 차이로 묻혔다. 그 결과 정책이
    #   "살아만 있고 목표는 무시"하는 상태가 되어 위치 오차가 1.0 -> 3.8 m 로 증가했다.
    #   std 2.5 / w 0.5 로 키우면 거리별 값(pre-dt)이
    #     4 m -> 0.039,  2 m -> 0.168,  1 m -> 0.31,  0.1 m -> 0.48
    #   로 전 구간에서 "다가갈수록 커지는" 단조 신호가 된다.
    ee_position_tracking_coarse = RewTerm(
        func=mdp.position_command_error_tanh_w,
        weight=0.4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[EE_BODY]), "std": 1.0, "command_name": "ee_pose"},
    )
    # [v9] ★ 접근 속도 보상 - 베이스를 쓰게 만드는 핵심 항.
    #   v8 은 it599 에서 position_error 0.56 m 로 수렴한 뒤 900 iteration 동안 정체했다.
    #   그 값은 "베이스를 전혀 안 쓰고 팔만 뻗었을 때"의 예상 오차와 일치한다
    #   (팔 리치 0.85 m, 목표 평균거리 1.2 m).
    #   기존 보상이 전부 정적("가까이 있으면 보상")이라 베이스 주행에 즉각적 이득이 없었다.
    #   이 항은 거리의 시간미분이라 potential-based shaping 이고 최적 정책을 바꾸지 않는다.
    #   weight 0.5: 접근속도 0.5 m/s 면 0.25 로 coarse tanh 와 비슷한 크기가 된다.
    ee_position_progress = RewTerm(
        func=mdp.position_command_progress_w,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[EE_BODY]), "command_name": "ee_pose"},
    )
    # fine: 마지막 수 cm 를 맞추는 보상. [v1] 0.2 -> 0.35 (고정형 팔에서 검증된 값)
    ee_position_tracking_fine = RewTerm(
        func=mdp.position_command_error_tanh_w,
        weight=0.35,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[EE_BODY]), "std": 0.1, "command_name": "ee_pose"},
    )
    # 자세(회전) 오차 L2. [v1] -0.05 -> -0.2, [v4] -0.2 -> -0.05
    #   자세 오차는 최대 pi 로 유계라 폭주 위험은 없지만,
    #   -0.2 면 초기(오차 1.5 rad)에 -0.3 pre-dt 로 위치 신호(0.039)의 8 배가 되어
    #   "스텝 보상을 양수로 유지한다"는 v4 설계를 깨뜨린다. 그래서 낮춘다.
    ee_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error_w,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[EE_BODY]), "command_name": "ee_pose"},
    )
    # [v1] 자세 성형 보상 2단 추가.
    #   기존에는 자세에 L2 페널티만 있어 "붙잡을 이유"가 없었다.
    #   franka v0 가 정확히 이 상태로 자세 0.196 rad 에서 정체했다.
    # [v11] 0.1 -> 0.2. v10 실현 보상이 위치 0.773 vs 자세 0.095 = 8.1:1 로
    #   극단적 위치 우세였다. 위치가 5 cm 로 여유가 크니 자세에 나눠준다.
    ee_orientation_tracking_fine = RewTerm(
        func=mdp.orientation_command_error_tanh_w,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[EE_BODY]), "std": 0.3, "command_name": "ee_pose"},
    )
    # [v11] 0.08 -> 0.15
    ee_orientation_tracking_precise = RewTerm(
        func=mdp.orientation_command_error_tanh_w,
        weight=0.15,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[EE_BODY]), "std": 0.05, "command_name": "ee_pose"},
    )
    # [v1] 손목 관절 속도 페널티 (franka v5 에서 도달 후 떨림을 잡은 항)
    #   [v4] -0.005 -> -0.002.
    #   실측에서 이 항이 -0.0551 로, 최대 태스크 보상(coarse +0.1386)의 40% 였다.
    #   franka 에서는 -0.0101 vs 태스크 +0.4 로 비율이 훨씬 작았다.
    #   모바일은 아직 접근 단계라 팔을 크게 움직여야 하는데 그걸 과하게 억제하고 있다.
    #   떨림은 도달 후의 문제이므로 학습이 붙은 뒤 다시 올리면 된다.
    # [v12] 관절 가속도 페널티 - 미세 고주파 떨림 전용.
    #   action_rate_l2 는 1차 차분 |a_t - a_{t-1}|^2 이라 큰 진동은 잡지만
    #   작은 진폭으로 매 스텝 부호가 바뀌는 고주파 진동은 차분값이 작아 거의 안 걸린다.
    #   가속도는 2차 미분이라 진폭 A / 주파수 f 진동에서 A*f^2 에 비례 -> 고주파에 제곱으로 민감.
    #   v11 실측: 진폭은 줄었으나 미세 떨림이 남는다는 GUI 관찰에 대한 처방.
    arm_joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
    )

    # [v11] -0.002 -> -0.005 (franka v5 에서 도달 후 떨림을 해소한 값으로 복원).
    #   v4 에서 낮춘 이유는 "접근 단계에서 팔을 크게 움직여야 한다"였는데,
    #   v10 에서 접근은 이미 해결됐으므로(위치 5 cm) 이제 떨림 억제를 우선한다.
    wrist_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint[5-7]"])},
    )

    # ---------------------------------------------------------------
    # [v3] 자살 병리(deliberate early termination) 차단
    #
    #   v2 실측: 이탈 종료 비율이 it6 0.003 -> it12 0.618 -> it18 1.000 으로
    #   단조 증가하고 에피소드 길이가 450 -> 16 으로 줄었다.
    #   즉 정책이 "일부러 나가는 것"을 학습했다.
    #
    #   원인은 산수였다. RewardManager 는 보상 = func * weight * dt (dt=1/30) 로 계산한다.
    #     - 종료로 아끼는 것 : 남은 434 스텝 * 0.031 = 약 13.5
    #     - v2 의 페널티     : -10 * (1/30) = -0.333
    #   종료가 40 배 이득이었다.
    #
    #   해법은 두 가지를 같이 쓴다:
    #     (1) alive 보너스로 **스텝 보상 자체를 양수로** 만든다. 그러면 오래 살수록 이득이라
    #         종료 유인이 원리적으로 사라진다. 이게 근본 대책이다.
    #     (2) 종료 페널티도 제대로 된 크기로 올린다 (보조 안전장치).
    # ---------------------------------------------------------------

    # (1) 생존 보너스.
    #     [v3] 1.0 -> [v4] 0.2.
    #     v3 에서 1.0 은 자살 병리를 확실히 막았지만(이탈 종료 0.19 -> 0.03),
    #     태스크 신호를 300 배 차이로 압도해 "살아만 있고 목표는 무시"를 유발했다.
    #     v4 는 태스크 보상 자체를 전 구간 양수/단조로 만들었으므로
    #     생존 보너스는 보조 역할만 하면 된다.
    alive = RewTerm(func=mdp.is_alive, weight=0.2)

    # (2) 이탈 종료 페널티. -50 * (1/30) = -1.67 (1회성).
    #     생존 보너스 상실분(434 * 0.0333 = 14.5)과 합쳐 종료 비용이 약 16 이 되어,
    #     종료로 아끼는 13.5 보다 확실히 크다.
    out_of_bounds_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)

    # --- 정규화(페널티) 항 --------------------------------------------------
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    arm_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
    )
    # [v8] 베이스 속도 페널티 제거 (weight 0.0).
    #   원래 "베이스만 흔들어 보상을 먹는 것"을 막으려고 넣었는데,
    #   그 현상 자체가 스폰/루트 리셋 버그의 부산물이었다.
    #   지금은 베이스 주행이 태스크의 핵심인데 그걸 직접 억제하고 있었다.
    #   [v9] weight 를 0 이 아니라 -1e-6 으로 둔다. RewardManager 는 weight==0 인 항의
    #   계산을 건너뛰어 로그에도 안 남는데, 이 값은 **베이스 사용량의 유일한 지표**라
    #   관측을 유지해야 한다. -1e-6 이면 보상에 미치는 영향은 사실상 0 이다.
    # [v11] -1e-6 -> -0.0005. 도달 후 베이스가 목표 주변에서 진동하는 것을 억제한다.
    #   v8 에서 0 으로 뺐던 이유(베이스 주행 억제 우려)는 당시 베이스가 아예
    #   작동하지 않던 상황의 오진이었다. 지금은 1 m/s 로 잘 달리므로 소폭 페널티는 안전하다.
    base_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=BASE_JOINTS)},
    )


@configclass
class TerminationsCfg:
    """종료 조건."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # [v2] 베이스가 작업영역을 벗어나면 종료.
    #   v1 은 초기 탐색(액션 노이즈 std ~1.2)의 랜덤워크로 로봇이 4 m 넘게 배회했고,
    #   env_spacing 6 m 기준 이웃 환경 영역까지 침범해 물리 폭주의 원인이 되었다.
    #   dummy_base 의 prismatic x/y 관절값이 곧 env 원점 기준 베이스 변위다.
    base_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={
            "bounds": (-2.5, 2.5),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["dummy_base_prismatic_.*"]),
        },
    )


@configclass
class CurriculumCfg:
    """커리큘럼 — 어느 정도 배운 뒤 페널티를 강화해 동작을 매끄럽게 만든다."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.01, "num_steps": 12000}
    )
    arm_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "arm_joint_vel", "weight": -0.001, "num_steps": 12000}
    )


##
# 환경 설정
##


@configclass
class RidgebackFrankaReachEnvCfg(ManagerBasedRLEnvCfg):
    """모바일 매니퓰레이터 EE pose 추종 환경."""

    # env_spacing: 베이스 이탈 한계가 ±2.5 m 이므로 반경 2.5 + 로봇 크기 0.5 = 3.0 m.
    #   [v2] 6.0 -> 8.0 (반칸 4.0 m > 3.0 m) 으로 이웃 env 침범을 구조적으로 차단한다.
    scene: MobileReachSceneCfg = MobileReachSceneCfg(num_envs=4096, env_spacing=8.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # 제어 주기: sim 60 Hz / decimation 2 -> 정책 30 Hz
        self.decimation = 2
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        # [v12] 15.0 -> 21.0.
        #   목표 재샘플 주기가 7 초인데 에피소드가 15 초라 15 = 7+7+1 로 나누어떨어지지 않았다.
        #   매 에피소드 마지막 목표에 1 초만 주어져 도달이 원천적으로 불가능했고,
        #   그 구간이 학습 신호에 잡음으로 섞였다. 21 = 7*3 으로 맞춘다.
        self.episode_length_s = 21.0
        self.viewer.eye = (5.0, 5.0, 4.0)

        # [v12] 팔 액추에이터 damping 40 -> 80.
        #   stiffness=800, damping=40 은 감쇠비가 낮아(under-damped) 진동이 잘 죽지 않는다.
        #   IsaacLab 의 FRANKA_PANDA_HIGH_PD_CFG 도 damping 80 을 쓴다.
        #   보상(가속도 페널티)과 물리(감쇠) 두 층위에서 상보적으로 떨림을 억제한다.
        self.scene.robot.actuators["panda_shoulder"].damping = 80.0
        self.scene.robot.actuators["panda_forearm"].damping = 80.0


@configclass
class RidgebackFrankaReachEnvCfg_PLAY(RidgebackFrankaReachEnvCfg):
    """학습 결과 확인용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 6.0
        self.observations.policy.enable_corruption = False
