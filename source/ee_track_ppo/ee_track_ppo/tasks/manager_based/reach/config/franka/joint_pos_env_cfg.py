# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""고정형 매니퓰레이터(Franka Panda)의 EE pose 추종 태스크.

Isaac Lab 내장 ``ReachEnvCfg`` (source/isaaclab_tasks/.../manipulation/reach/reach_env_cfg.py)
를 그대로 상속하고, "로봇 / EE 링크 이름 / 액션" 3가지만 바꿔 끼운다.

나중에 Fairino FR3 로 교체할 때 손볼 곳은 아래 [SWAP] 주석 4곳뿐이다.
"""

import math

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

# 이 프로젝트의 커스텀 보상 (자세용 tanh 성형 보상)
import ee_track_ppo.tasks.manager_based.reach.mdp as ee_mdp

##
# 사전 정의된 로봇 설정
##
from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaReachEnvCfg(ReachEnvCfg):
    """Franka Panda 로 EE pose 를 추종하는 환경 설정."""

    def __post_init__(self):
        # 부모(ReachEnvCfg)의 기본값 먼저 적용
        super().__post_init__()

        # [SWAP 1] 로봇 교체 -------------------------------------------------
        # FR3 로 바꿀 때: 직접 만든 FR3_CFG(ArticulationCfg) 를 여기에 넣는다.
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # [SWAP 2] EE(엔드이펙터) 링크 이름 -----------------------------------
        # 보상 3개가 모두 이 링크의 world pose 를 본다.
        ee_body = "panda_hand"
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [ee_body]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [ee_body]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [ee_body]

        # [SWAP 3] 액션: 팔 관절 위치 제어 -------------------------------------
        # scale=0.5, use_default_offset=True -> 정책 출력이 "기본자세 기준 상대 오프셋"이 된다.
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.5,
            use_default_offset=True,
        )

        # [SWAP 4] 목표 pose 커맨드 -------------------------------------------
        # 주의: UniformPoseCommand 는 목표를 "로봇 base(root) 프레임" 기준으로 샘플링한다.
        #       ranges 는 팔이 실제로 도달 가능한 작업공간 안에 있어야 한다.
        self.commands.ee_pose.body_name = ee_body
        # Franka 는 EE 축이 z 방향이라 pitch 를 pi 로 고정한다 (FR3 는 실제 축 방향 확인 필요).
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)

        # ---------------------------------------------------------------
        # 보상 튜닝 이력 (실측 근거)
        #
        # [v0] baseline 2026-07-27_10-38-04 (1000 iter)
        #   position_error    0.270 -> 0.027 (it284 최저) -> 0.060 (it999)   ... 후퇴
        #   orientation_error 1.662 -> 0.154 (it489 최저) -> 0.196 (it999)   ... 정체
        #   Curriculum/action_rate 가 it187 에서 -0.0001 -> -0.005 (50배)로 점프
        #
        # [v1] ori_w04 2026-07-27_11-06-48 (1000 iter)
        #   자세 -0.4 / 자세 tanh 0.2(std 0.3) / 위치 tanh std 0.05 로 변경
        #   orientation_error 0.196 -> 0.024  ... 8배 개선 (성공)
        #   position_error    0.060 -> 0.385  ... 6배 악화 (실패)
        #   원인: 위치 tanh std 를 0.05 로 좁힌 탓에 오차 0.385 에서
        #        1-tanh(0.385/0.05) ~ 0 -> 위치 학습 신호가 소멸.
        #        로그상 Episode_Reward/..._position_tracking_fine_grained = 0.0000,
        #        자세 tanh 만 0.1736 을 벌어 정책이 자세만 최적화함.
        #
        # [v2] pos_fix_v2 2026-07-27_11-30-59 (1000 iter)
        #   위치 보상을 v0 설정으로 원복 + 가중치 강화, 자세는 절반으로 완화
        #   position_error    0.385 -> 0.107 (최고 0.088)  ... 회복했으나 v0(0.025)에 못 미침
        #   orientation_error 0.196 -> 0.057 (최고 0.050)  ... v0 대비 3.4배 개선
        #   두 지표 모두 it999 까지 미세하게 하강하나 기울기가 거의 0 -> 수렴
        #
        # [v3] v3_product 2026-07-27_11-50-40 (1000 iter)  ==> 실패, 되돌림
        #   곱셈형 결합 보상 + 정밀 tanh 사다리 추가, L2 절반으로.
        #   position_error    0.107 -> 0.220  (악화)
        #   orientation_error 0.057 -> 0.043  (소폭 개선)
        #   실패 원인: weight 는 위치 우선(0.4 vs 0.25)으로 줬는데 실현 보상은 자세가 5배였다.
        #     자세 precise std=0.1rad 인데 실제 오차가 0.043 -> 이미 구간 안 -> 0.088 을 거저 획득
        #     위치 precise std=0.03m  인데 실제 오차가 0.220 -> 구간 밖   -> 0.003 밖에 못 얻음
        #   교훈: 보상 배분을 결정하는 것은 weight 가 아니라 "std 대비 도달 가능한 오차"다.
        #   v3 설정은 아래 (e) 블록에 주석으로 보존.
        #
        # [v4] 현재: 보상은 가장 균형이 좋았던 v2 로 되돌리고,
        #      대신 **관측과 네트워크 용량**을 바꾼다 (변수를 하나만 바꿔 효과를 분리).
        #   근거: 보상만 4번 조정했으나 항상 한쪽을 희생하는 패턴이 반복됨.
        #        정책이 자기 EE pose 를 관측하지 못해 FK 를 암묵 학습해야 하는 것이 병목으로 판단.
        #        -> ObservationsCfg 에 현재 EE pose + 목표와의 오차 벡터 추가 (아래)
        #        -> agents/rsl_rl_ppo_cfg.py 의 네트워크 [64,64] -> [256,128,64]
        # ---------------------------------------------------------------

        # (a) L2 페널티: v2 값 유지
        self.rewards.end_effector_position_tracking.weight = -0.2
        self.rewards.end_effector_orientation_tracking.weight = -0.2

        # (b) 개별 성형 보상 (접근 단계 담당) - v2 값 유지
        #     (RewardManager 는 cfg 의 __dict__ 를 순회하므로 여기서 항을 새로 추가해도 등록된다)
        robot_ee = SceneEntityCfg("robot", body_names=[ee_body])

        # [v6] 위치 가중치 강화 0.2 -> 0.35 (아래 v6 주석 참고)
        self.rewards.end_effector_position_tracking_fine_grained.params["std"] = 0.1
        self.rewards.end_effector_position_tracking_fine_grained.weight = 0.35

        self.rewards.end_effector_orientation_tracking_fine_grained = RewTerm(
            func=ee_mdp.orientation_command_error_tanh,
            weight=0.1,
            params={"asset_cfg": robot_ee, "std": 0.3, "command_name": "ee_pose"},
        )

        # (c) ★ 관측 추가 - 이번 버전의 핵심.
        #     기존 관측은 joint_pos / joint_vel / 목표pose / last_action 뿐이라
        #     정책이 "지금 내 EE 가 어디 있는지"를 못 본다 (FK 를 암묵 학습해야 함).
        #     현재 EE pose 와 목표까지의 오차 벡터를 직접 넣어준다. 32 -> 45 차원.
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

        # ---------------------------------------------------------------
        # [v6] 실험 5회로 확인된 법칙: "위치:자세 실현보상 비율"이 결과를 거의 그대로 결정한다.
        #        v3  1 : 5.2  -> pos 0.220 / ori 0.043
        #        v2  1 : 1.1  -> pos 0.107 / ori 0.057
        #        v5  1.05: 1  -> pos 0.064 / ori 0.048
        #        v4  2.6 : 1  -> pos 0.011 / ori 0.087
        #      v5 는 떨림 억제(손목 속도 페널티 + 커리큘럼 복원)에 성공했으나
        #      (action_rate 실현값 -0.0109 -> -0.0032 로 동작이 실제로 부드러워짐)
        #      자세 precise 항이 0.068 을 벌어 비율을 1:1 로 뒤집는 바람에 위치가 무너졌다.
        #      => 떨림 억제 조치는 전부 유지하고 보상 비율만 위치 우세(약 2:1)로 되돌린다.
        #         위치 fine 0.2 -> 0.35 / 자세 precise 0.15 -> 0.08
        #
        # [v5] 도달 후 자세 떨림(chattering) 억제
        #   v4 실측: position 0.0109 (역대 최고) / orientation 0.0873
        #   GUI 관찰: 목표 근처까지는 잘 가는데 도달 후 자세만 흔들림 (위치는 안정)
        #   진단: play.py 는 결정론적 정책이므로 탐색 노이즈가 아니라 정책의 실제 행동.
        #        위치는 std 0.1 에 오차 0.011 -> 보상 기울기 가파름 -> 꽉 붙잡음
        #        자세는 std 0.3 에 오차 0.087 -> 상대적으로 평평 -> 붙잡을 이유가 약함
        #        => 7DOF 여유자유도(null space)에서 손목이 배회하는 것으로 판단.
        #        게다가 Franka 손목 panda_joint[5-7] 은 effort 12(어깨 87), damping 4 로
        #        원래 떨기 쉬운 관절이다.
        # ---------------------------------------------------------------

        # (e-1) 자세 정밀 tanh: 평평했던 정밀 구간에 기울기를 공급한다.
        #       std=0.05 는 v5 시점 오차 0.087 기준 "아직 도달 못한 칸"이라 공짜 보상이 아니다.
        #       (v3 실패는 std=0.1 이 이미 도달한 구간이라 거저 주는 보상이었기 때문)
        #
        # [v6] weight 0.15 -> 0.08. v5 에서 이 항이 0.068 을 벌어들이며
        #      위치:자세 실현보상 비율을 2.6:1 -> 1.05:1 로 뒤집어 위치를 망가뜨렸다.
        # [v7] 0.08 -> 0.15 (v5 값으로 복귀). v6 이 위치 0.004m 로 필요 이상 좋아져
        #      (비율 3.3:1) 자세에 여유를 내줄 수 있게 되었다. 위치 fine 은 v6 의 0.35 유지.
        #      => v5(1.05:1) 와 v6(3.3:1) 사이인 약 2:1 을 노린다.
        self.rewards.end_effector_orientation_tracking_precise = RewTerm(
            func=ee_mdp.orientation_command_error_tanh,
            weight=0.15,
            params={"asset_cfg": robot_ee, "std": 0.05, "command_name": "ee_pose"},
        )

        # (e-2) 손목 관절만 속도 페널티: 떨리는 곳을 직접 감쇠한다.
        #       전체 관절에 걸면 접근 속도까지 느려지므로 panda_joint[5-7] 만 타깃.
        self.rewards.wrist_joint_vel = RewTerm(
            func=mdp.joint_vel_l2,
            weight=-0.005,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint[5-7]"])},
        )

        # (e-3) [v3 실패 설정 보존] 곱셈형 결합 보상 + 정밀 tanh 사다리.
        #     아이디어 자체(곱셈형)는 의도대로 작동했으나(0.0376 획득),
        #     개별 자세 항이 거저 벌어들이는 0.167 에 묻혔다.
        #     다시 시도할 경우 std 를 "현재 달성 오차와 같은 난이도"로 맞춰야 한다.
        #     예) 자세 precise std 0.1 -> 0.03, 위치 precise std 0.03 -> 0.06
        # self.rewards.end_effector_position_tracking_precise = RewTerm(
        #     func=mdp.position_command_error_tanh,
        #     weight=0.2,
        #     params={"asset_cfg": robot_ee, "std": 0.03, "command_name": "ee_pose"},
        # )
        # self.rewards.end_effector_orientation_tracking_precise = RewTerm(
        #     func=ee_mdp.orientation_command_error_tanh,
        #     weight=0.15,
        #     params={"asset_cfg": robot_ee, "std": 0.1, "command_name": "ee_pose"},
        # )
        # self.rewards.ee_pose_tracking_joint = RewTerm(
        #     func=ee_mdp.pose_command_error_tanh_product,
        #     weight=0.3,
        #     params={"asset_cfg": robot_ee, "pos_std": 0.1, "ori_std": 0.3, "command_name": "ee_pose"},
        # )
        # self.rewards.ee_pose_tracking_joint_precise = RewTerm(
        #     func=ee_mdp.pose_command_error_tanh_product,
        #     weight=0.5,
        #     params={"asset_cfg": robot_ee, "pos_std": 0.03, "ori_std": 0.1, "command_name": "ee_pose"},
        # )

        # (f) 커리큘럼: v0 의 -0.005 는 과해서 정밀도를 깎았고(2.7cm -> 6cm),
        #     v2~v4 의 -0.001 은 너무 약해 떨림을 못 막았다. v5 에서 중간값으로 조정.
        #     관측이 좋아져 정밀도 여유가 생겼으므로 감당 가능하다고 판단.
        self.curriculum.action_rate.params["weight"] = -0.003  # v0 -0.005 / v2~v4 -0.001
        self.curriculum.joint_vel.params["weight"] = -0.001  # v0 -0.001 / v2~v4 -0.0005
        self.curriculum.action_rate.params["num_steps"] = 12000  # 기존 4500 (더 늦게 적용)
        self.curriculum.joint_vel.params["num_steps"] = 12000


@configclass
class FrankaReachEnvCfg_PLAY(FrankaReachEnvCfg):
    """학습된 정책 시각화(play)용 - 환경 수를 줄이고 관측 노이즈를 끈다."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
