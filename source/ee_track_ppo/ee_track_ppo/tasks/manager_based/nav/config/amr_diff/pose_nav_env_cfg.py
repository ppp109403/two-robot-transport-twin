# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""차동구동 AMR 의 목표 pose 도달 태스크 — 로봇별 설정.

기본 태스크 정의는 ``nav/pose_nav_env_cfg.py`` 에 있고, 여기서는 로봇에 의존하는
값만 채운다. 실제 설계 로봇 URDF 로 교체할 때 손대는 곳은 두 군데뿐이다.

1. ``ee_track_ppo/assets/amr_diff_drive.py`` — USD 경로, 조인트 이름, 휠 반경/윤거
2. 이 파일의 속도/가속 한계 (실측값으로)

지금 들어 있는 로봇은 메쉬 없이 기본 도형으로 만든 **플레이스홀더**다.
태스크 배관(관측/보상/액션)을 검증하는 용도이며, 제원은 일반적인 중형 AMR 급이다.
"""

from isaaclab.utils import configclass

from ee_track_ppo.assets.amr_diff_drive import (  # isort: skip
    AMR_BASE_BODY,
    AMR_DIFF_DRIVE_CFG,
    AMR_LEFT_WHEEL_JOINT,
    AMR_RIGHT_WHEEL_JOINT,
    WHEEL_BASE,
    WHEEL_RADIUS,
)
from ee_track_ppo.tasks.manager_based.nav.pose_nav_env_cfg import AmrPoseNavEnvCfg  # isort: skip

##
# 실로봇 실측값 — 여기가 sim2real 의 핵심 파라미터다
##

MAX_LIN_VEL = 0.5
"""최대 전진속도 [m/s]. 실로봇에서 안전하게 낼 수 있는 값으로 맞춰야 한다.
sim 에서 더 빠르게 학습하면 실로봇이 따라오지 못해 정책이 예상한 이동량이 나오지 않는다."""

MAX_ANG_VEL = 1.0
"""최대 요레이트 [rad/s]."""

MAX_REVERSE_VEL = 0.2
"""최대 후진속도 [m/s] (양수로 지정). 후방 센서가 없으므로 전진의 40% 로 제한한다."""

MAX_LIN_ACCEL = 0.6
"""전진 가감속 한계 [m/s^2]. 55 kg 차량이 0.83 초에 최대속도에 도달하는 램프."""

MAX_ANG_ACCEL = 1.5
"""요레이트 가감속 한계 [rad/s^2]."""

##
# 트위스트 이득 랜덤화
#
#   이 로봇의 sim 실현율 실측: 전진 0.99 / 제자리 선회 0.73.
#   선회 0.73 은 마찰·모터토크·캐스터 기하를 다 바꿔도 움직이지 않는 값이고
#   (근거는 assets/data/amr_diff_drive/urdf/amr_isaac.urdf 주석 참고)
#   실로봇 값은 바닥 재질과 캐스터 마모에 따라 또 달라진다.
#
#   그래서 이득 범위를 "실현율이 1.0 을 중심으로 퍼지도록" 잡는다.
#     전진: 이득 0.95~1.15 -> 실현율 0.94~1.14
#     선회: 이득 1.15~1.65 -> 실현율 0.84~1.20   (1/0.73 = 1.37 이 실현율 1.0)
#   정책은 이 폭 안에서 폐루프로 보정하는 법을 배우게 된다.
##

LIN_GAIN_RANGE = (0.95, 1.15)
"""전진속도 이득 랜덤화 범위."""

ANG_GAIN_RANGE = (1.15, 1.65)
"""요레이트 이득 랜덤화 범위. 선회 실현율 편차가 크므로 전진보다 넓다."""

##
# 태스크 규격 — 최대속도에 맞춰 "도달 가능한" 목표 범위와 시간을 잡는다
#
#   목표 재샘플 주기 10 초 동안 0.5 m/s 로 갈 수 있는 거리는 5 m 다.
#   여기서 초기 선회(최악 pi rad / 1.0 rad/s = 3.1 초)와 도착 후 방향 정렬 시간을 빼면
#   실제로 주행에 쓸 수 있는 시간은 약 6 초 = 3 m 다.
#   목표 범위 ±2.0 m 의 최대 거리는 대각선 2.83 m 이므로 여유가 남는다.
#
#   범위를 이보다 키우면 "시간 안에 도달 불가능한 목표"가 섞여 들어가고,
#   그 에피소드의 학습 신호는 잡음이 된다.
##

GOAL_RANGE = 2.0
"""목표 샘플링 범위 [m]. env 원점 기준 ±GOAL_RANGE 정사각형."""

GOAL_PERIOD = 10.0
"""목표 재샘플 주기 [s]."""

EPISODE_LENGTH = 30.0
"""에피소드 길이 [s]. GOAL_PERIOD 의 정수배여야 마지막 목표에도 도달 시간이 주어진다."""

BOUNDS_RADIUS = 3.5
"""이탈 종료 반경 [m]. 목표 범위 대각선(2.83) 보다 크게."""

ENV_SPACING = 8.0
"""env 간격 [m]. 반칸(4.0) > BOUNDS_RADIUS(3.5) 여야 이웃 env 침범이 불가능하다."""

HEADING_GATE_DIST = 0.8
"""방향 보상 전환 거리 [m].

차체 외접원 반경이 0.375 m 이므로 기본값 0.5 m 는 "거의 목표에 닿은 뒤"에야
yaw 정렬로 넘어가게 되어 늦다. 차체 크기의 2배 정도로 잡는다.
"""


@configclass
class AmrDiffPoseNavEnvCfg(AmrPoseNavEnvCfg):
    """플레이스홀더 차동구동 AMR 의 목표 pose 도달 환경."""

    def __post_init__(self):
        super().__post_init__()

        # --- 로봇 ---
        self.scene.robot = AMR_DIFF_DRIVE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # --- 액션: 차동구동 역기구학 파라미터 ---
        self.actions.drive.left_wheel_joint_names = [AMR_LEFT_WHEEL_JOINT]
        self.actions.drive.right_wheel_joint_names = [AMR_RIGHT_WHEEL_JOINT]
        self.actions.drive.wheel_radius = WHEEL_RADIUS
        self.actions.drive.wheel_base = WHEEL_BASE
        self.actions.drive.max_lin_vel = MAX_LIN_VEL
        self.actions.drive.max_ang_vel = MAX_ANG_VEL
        self.actions.drive.max_reverse_vel = MAX_REVERSE_VEL
        self.actions.drive.max_lin_accel = MAX_LIN_ACCEL
        self.actions.drive.max_ang_accel = MAX_ANG_ACCEL
        self.actions.drive.lin_gain_range = LIN_GAIN_RANGE
        self.actions.drive.ang_gain_range = ANG_GAIN_RANGE

        # --- 적재 하중 랜덤화는 베이스 링크에만 적용한다 ---
        #     바퀴 질량까지 스케일하면 회전 관성이 함께 변해 구동 특성이 왜곡된다.
        self.events.base_mass.params["asset_cfg"].body_names = [AMR_BASE_BODY]

        # --- 태스크 규격: 최대속도에 맞춰 도달 가능한 범위/시간으로 조정 ---
        self.commands.goal_pose.ranges.pos_x = (-GOAL_RANGE, GOAL_RANGE)
        self.commands.goal_pose.ranges.pos_y = (-GOAL_RANGE, GOAL_RANGE)
        self.commands.goal_pose.resampling_time_range = (GOAL_PERIOD, GOAL_PERIOD)
        self.episode_length_s = EPISODE_LENGTH
        self.terminations.out_of_bounds.params["max_dist"] = BOUNDS_RADIUS
        self.scene.env_spacing = ENV_SPACING
        self.rewards.heading.params["gate_dist"] = HEADING_GATE_DIST

        # 리셋 시 초기 위치 산포. 목표 범위보다 작게 두어 목표가 항상 주행 거리 안에 있게 한다.
        self.events.reset_base.params["pose_range"]["x"] = (-0.8, 0.8)
        self.events.reset_base.params["pose_range"]["y"] = (-0.8, 0.8)

        # 카메라를 로봇 크기에 맞춰 뒤로 뺀다
        self.viewer.eye = (4.0, 4.0, 3.0)


@configclass
class AmrDiffPoseNavEnvCfg_PLAY(AmrDiffPoseNavEnvCfg):
    """학습 결과 확인용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 25
        self.scene.env_spacing = 10.0
        # 관측 노이즈(Unoise)만 끈다. AMCL 모사(지연/갱신주기/점프)는 그대로 두어야
        # 실로봇과 같은 조건에서 성능을 본다.
        self.observations.policy.enable_corruption = False
