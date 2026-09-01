# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMR + FR3 의 **EE 궤적 추종** 태스크.

정지 목표 추종(``Reach-AMR-FR3-PPO-v0``)과의 차이
--------------------------------------------------
실제 운용은 상위 제어기가 주는 **연속 EE 궤적**을 따라가는 것이다.
기존 태스크는 목표가 8 초마다 계단식으로 점프해서, 정책이 "가서 멈춰라"를 배운다.

측정 근거 (``scripts/measure_settling.py``, mm_v1 정책)::

    목표 근처 시정수 tau ~= 0.5 s,  정상상태 바닥 0.008 m
    추종 오차 ~= 궤적 속도 x tau + 바닥
      5 cm/s ->  3.3 cm      10 cm/s ->  5.8 cm
     20 cm/s -> 11 cm        30 cm/s -> 16 cm

목표가 30 cm/s 까지이므로 순수 피드백으로는 16 cm 오차가 남는다.
그래서 **목표 속도를 관측에 넣어 피드포워드**를 확보한다.

바뀌는 것은 세 가지뿐이고 보상·액션·종료·로봇은 그대로 재사용한다.

  1. 커맨드   : UniformPoseWorldCommand -> MovingPoseWorldCommand (연속 이동)
  2. 관측     : ee_target_vel_b (6차원) 추가  -> 관측 42 -> 48 차원
  3. 커리큘럼 : 목표 속력을 0 -> 0.30 m/s 로 단계적 상향

왜 속력 커리큘럼인가
--------------------
처음부터 30 cm/s 로 학습하면 "따라잡을 수 없는 목표"가 계속 주어져 학습 신호가 나빠진다.
정지에 가까운 목표부터 시작해 추종을 익힌 뒤 속력을 올리는 것이 안정적이다.
(고정팔 v1 에서 tanh std 를 처음부터 좁게 잡아 학습 신호가 소멸했던 것과 같은 종류의 함정)
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import ee_track_ppo.tasks.manager_based.reach.mdp as mdp

from ee_track_ppo.tasks.manager_based.reach.config.amr_fr3.mobile_reach_env_cfg import (  # isort: skip
    AMR_FR3_EE_BODY,
    AmrFr3ReachEnvCfg,
    GOAL_RANGE_XY,
    GOAL_Z_RANGE,
)


##
# 궤적 속력 커리큘럼 단계 [m/s]
##
SPEED_STAGES = [
    (0, (0.00, 0.05)),
    (14_400, (0.00, 0.12)),
    (33_600, (0.05, 0.22)),
    (57_600, (0.05, 0.30)),
]
"""(env step 수, 속력 범위). 목표 최대 30 cm/s 까지 단계적으로 올린다.

**단위 주의**: ``env.common_step_counter`` 는 ``env.step()`` 호출당 **1** 증가한다
(env 개수와 무관하다). 따라서

    1 iteration 당 증가량 = num_steps_per_env = 48

이 태스크의 임계값을 iteration 으로 환산하면::

    14,400 -> it  300      33,600 -> it  700      57,600 -> it 1200

2500 iteration 학습에서 마지막 단계(30 cm/s)에 it1200 에 진입해 1300 iteration 을
최대 속력으로 학습하게 된다.

처음에 30k/80k/150k 로 잡았다가 고쳤다. 그 값이면 최종 단계가 it3125 라
2500 iteration 안에는 **진입조차 못 했다**. num_steps_per_env 를 바꾸면
이 값들도 함께 조정해야 한다.
"""


def _ramp_speed(env, env_ids, data, stages):
    """common_step_counter 에 따라 속력 범위를 단계적으로 넓힌다."""
    new = None
    for step_thr, rng in stages:
        if env.common_step_counter >= step_thr:
            new = rng
    if new is None or tuple(data) == tuple(new):
        return mdp.modify_term_cfg.NO_CHANGE
    return new


@configclass
class TrajCurriculumCfg:
    """궤적 속력 커리큘럼 + 기존 페널티 커리큘럼."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.01, "num_steps": 12000}
    )
    arm_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "arm_joint_vel", "weight": -0.001, "num_steps": 12000}
    )
    target_speed = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.ee_pose.speed_range",
            "modify_fn": _ramp_speed,
            "modify_params": {"stages": SPEED_STAGES},
        },
    )


@configclass
class AmrFr3TrajEnvCfg(AmrFr3ReachEnvCfg):
    """AMR + FR3 로 **움직이는 EE 목표**를 추종하는 환경."""

    curriculum: TrajCurriculumCfg = TrajCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # [1] 커맨드: 계단식 점프 -> 연속 이동
        #     resampling_time_range 를 에피소드보다 길게 둬서 중간 점프를 없앤다.
        #     (부모 클래스의 _resample_command 는 에피소드 리셋 시 초기화 용도로만 쓰인다)
        self.commands.ee_pose = mdp.MovingPoseWorldCommandCfg(
            asset_name="robot",
            body_name=AMR_FR3_EE_BODY,
            resampling_time_range=(1.0e6, 1.0e6),
            debug_vis=True,
            ranges=mdp.MovingPoseWorldCommandCfg.Ranges(
                pos_x=(-GOAL_RANGE_XY, GOAL_RANGE_XY),
                pos_y=(-GOAL_RANGE_XY, GOAL_RANGE_XY),
                pos_z=GOAL_Z_RANGE,
                roll=(0.0, 0.0),
                pitch=(math.pi, math.pi),
                yaw=(-math.pi, math.pi),
            ),
            speed_range=SPEED_STAGES[0][1],
            ang_speed_range=(0.0, 0.3),
            waypoint_tolerance=0.05,
            angle_tolerance=0.1,
        )

        # [1-b] 자세 tanh 사다리에 **중간 칸** 추가.
        #
        #   traj_v1 실측: orientation_error 0.105 rad (정지 목표 태스크는 0.0165).
        #   목표 자세가 계속 회전하니 어려워진 것이고, 보상 비율 자체는 2.87:1 로 양호했다.
        #   문제는 그 오차 지점에서 **두 자세 항 모두 기울기를 못 낸다**는 것이었다::
        #
        #       fine    (std 0.3 ) : 1-tanh(0.105/0.3 ) = 0.66  -> 거의 포화
        #       precise (std 0.05) : 1-tanh(0.105/0.05) = 0.03  -> 구간 밖
        #
        #   tanh 성형보상의 기울기는 (오차 ~ std) 에서 최대이므로,
        #   현재 오차와 같은 크기의 std=0.12 칸을 끼워 넣는다.
        #   가중치를 올리는 것보다 정확한 처방이다 (가중치는 이미 균형점에 있다).
        self.rewards.ee_orientation_mid = RewTerm(
            func=mdp.orientation_command_error_tanh_w,
            weight=0.12,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[AMR_FR3_EE_BODY]),
                "std": 0.12,
                "command_name": "ee_pose",
            },
        )

        # [2] 관측: 목표 속도 추가 (피드포워드)
        self.observations.policy.target_vel = ObsTerm(
            func=mdp.ee_target_vel_b,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[AMR_FR3_EE_BODY]),
                "command_name": "ee_pose",
            },
            noise=Unoise(n_min=-0.005, n_max=0.005),
        )


@configclass
class AmrFr3TrajEnvCfg_PLAY(AmrFr3TrajEnvCfg):
    """학습 결과 확인용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        # PLAY 에서는 커리큘럼이 돌지 않으므로 최종 속력 범위를 직접 지정한다
        self.commands.ee_pose.speed_range = SPEED_STAGES[-1][1]
