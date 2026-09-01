# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""차동구동 AMR 의 목표 pose 도달 태스크 — 로봇 무관(asset-agnostic) 기본 설정.

로봇에 의존하는 값은 모두 ``MISSING`` 으로 두고, 로봇별 설정
(``config/<robot>/pose_nav_env_cfg.py``)에서 채운다. 채워야 하는 것은 다음뿐이다.

* ``scene.robot``                       : :class:`ArticulationCfg`
* ``actions.drive.{wheel_radius, wheel_base, left_/right_wheel_joint_names}``
* 필요하면 속도/가속 한계와 목표 허용오차

태스크 정의
-----------
평지에서 env 원점 기준으로 무작위 목표 (x, y, yaw) 를 주고, 로봇이 그 pose 로
이동해 방향까지 맞추게 한다. 에피소드 동안 목표를 여러 번 재샘플하므로
"도달 -> 유지 -> 다음 목표로 이동"이 한 에피소드에서 반복 학습된다.

정책 입출력
-----------
* 입력 (8차원): AMCL 모사 pose 로 계산한 목표 상대량 5 + 바디 트위스트 2 ... 는 7,
  여기에 이전 액션 2 를 더해 9차원이다. 자세한 구성은 :class:`ObservationsCfg` 참고.
* 출력 (2차원): (v, ω) — 실로봇의 ``/cmd_vel`` 과 동일한 의미.

주파수 설계
-----------
정책 25 Hz (sim 100 Hz / decimation 4). 실로봇의 ``/cmd_vel`` 발행 주기와
AMCL 갱신 주기(10~20 Hz) 사이에 맞춘 값이다. 정책을 실로봇보다 훨씬 빠르게
학습시키면 "실로봇에서는 낼 수 없는 빠른 피드백"에 의존하는 정책이 나온다.
"""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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

import ee_track_ppo.tasks.manager_based.nav.mdp as mdp

##
# 태스크 규격 — 여기 숫자들이 서로 맞물려 있으므로 함께 바꿀 것
##

GOAL_RANGE = 3.5
"""목표 샘플링 범위 [m]. env 원점 기준 ±GOAL_RANGE 정사각형."""

BOUNDS_RADIUS = 5.0
"""이탈 종료 반경 [m]. 목표 범위(대각선 4.95)보다 크게 잡아야 도달 가능한 목표가 잘리지 않는다."""

ENV_SPACING = 12.0
"""env 간격 [m]. 반칸(6.0) > BOUNDS_RADIUS(5.0) 이어야 이웃 env 침범이 구조적으로 불가능하다."""

POS_TOL = 0.05
"""성공 판정 위치 허용오차 [m]."""

YAW_TOL = 5.0 * math.pi / 180.0
"""성공 판정 방향 허용오차 [rad] (5도)."""

HEADING_GATE_DIST = 0.5
"""방향 보상 전환 거리 [m]. 이보다 멀면 "목표 바라보기", 가까우면 "목표 yaw 맞추기"."""


##
# 씬
##


@configclass
class AmrNavSceneCfg(InteractiveSceneCfg):
    """평지 + 차동구동 AMR."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    robot: ArticulationCfg = MISSING

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP
##


@configclass
class CommandsCfg:
    """목표 2D pose 생성기."""

    # Isaac Lab 내장 UniformPose2dCommand 를 그대로 쓴다.
    # 이 커맨드는 목표를 **env 원점 기준**으로 샘플링하고 월드 좌표로 보관하므로
    # (pos_command_w / heading_command_w) 로봇이 움직여도 목표가 따라오지 않는다.
    # (root 프레임에서 샘플링하는 UniformPoseCommand 를 쓰면 목표가 로봇에 붙어
    #  따라다녀서 주행할 이유가 사라진다 — ridgeback 태스크에서 겪은 함정이다)
    goal_pose = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        # simple_heading=True 면 목표 방향이 "목표를 바라보는 방향"으로 자동 설정된다.
        # 그러면 방향 정렬이 주행 방향과 항상 일치해서 태스크가 쉬워진다.
        # 우리는 위치와 방향을 독립적으로 요구하고 싶으므로 False 로 두고 무작위 샘플링한다.
        simple_heading=False,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-GOAL_RANGE, GOAL_RANGE),
            pos_y=(-GOAL_RANGE, GOAL_RANGE),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """액션: 바디 트위스트 (v, ω) 2차원."""

    drive: mdp.DiffDriveActionCfg = mdp.DiffDriveActionCfg(
        asset_name="robot",
        # 로봇별 설정에서 채운다
        left_wheel_joint_names=MISSING,
        right_wheel_joint_names=MISSING,
        wheel_radius=MISSING,
        wheel_base=MISSING,
    )


@configclass
class ObservationsCfg:
    """관측 — 실로봇에서 계산 가능한 값만 담는다."""

    @configclass
    class PolicyCfg(ObsGroup):
        # [5] AMCL 모사 pose 로 계산한 목표 상대량 (ρ, cos/sin bearing, cos/sin yaw오차).
        #     노이즈/지연/갱신주기/점프가 이 term 내부에서 처리되므로
        #     여기에 Unoise 를 **추가로 얹지 않는다** (이중 노이즈가 된다).
        amcl_goal = ObsTerm(
            func=mdp.amcl_goal_polar,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "goal_pose",
                # AMCL 정상 상태 추정 오차. 2D 라이다 + 좋은 맵 기준 보수적인 값
                "pos_noise_std": 0.02,
                "yaw_noise_std": 0.02,
                # 스캔 매칭 주기 20 Hz (정책 25 Hz 보다 느리다 -> 값이 유지되는 구간이 생긴다)
                "update_period": 0.05,
                # 스캔 취득 ~ 퍼블리시 지연
                "latency": 0.06,
                # 파티클 필터 재수렴 도약. 학습 중 드물게 노출시켜 강건성을 만든다
                "jump_prob": 0.002,
                "jump_pos_std": 0.10,
                "jump_yaw_std": 0.10,
                "max_range": 12.0,
            },
        )
        # [2] 휠 엔코더 오도메트리 트위스트 (v, ω)
        base_twist = ObsTerm(
            func=mdp.base_twist_2d,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # [2] 직전에 퍼블리시한 /cmd_vel
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """리셋 이벤트 + 도메인 랜덤화."""

    # --- 시작 시 1회: sim2real 을 위한 물성 랜덤화 -------------------------
    # 바퀴 마찰. 매끄러운 에폭시 바닥 ~ 거친 콘크리트 범위를 덮는다.
    # 이것이 없으면 정책이 특정 마찰계수에 맞춘 가속 프로파일을 학습한다.
    wheel_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    # 적재 하중 변화. 실제 AMR 은 짐을 싣고 내린다.
    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "mass_distribution_params": (0.8, 1.4),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    # --- 리셋 ---------------------------------------------------------------
    # 루트 pose 무작위화. yaw 를 전 범위로 뿌리는 것이 중요하다.
    #   yaw 를 0 으로 고정하면 "목표가 대체로 앞쪽에 있는" 편향된 분포만 보게 되어
    #   제자리 선회를 제대로 배우지 못한다.
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {},
        },
    )
    # 바퀴 관절 상태 초기화. 안 하면 이전 에피소드의 바퀴 회전속도가 이어진다.
    reset_wheels = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """보상.

    설계 원칙은 **스텝 보상 총합을 양수로 유지**하는 것이다.
    그러면 "오래 살수록 이득"이 되어 정책이 일부러 종료를 유발할 유인이 사라진다.
    (ridgeback 태스크에서 스텝 보상이 음수일 때 정책이 의도적 이탈 종료를 학습했다.
     종료로 아끼는 페널티가 종료 페널티의 40 배였다)

    그래서 태스크 항은 모두 [0, 1] 범위의 양수 보상이고, 페널티는 작게 유지한다.
    """

    # --- 태스크 보상 -------------------------------------------------------
    # 접근 속도. 주행 구간의 유일한 조밀 신호다 (potential-based shaping).
    progress = RewTerm(
        func=mdp.position_progress,
        weight=1.0,
        params={"command_name": "goal_pose", "asset_cfg": SceneEntityCfg("robot")},
    )
    # 거리 tanh 사다리: coarse 는 먼 거리, fine 은 마지막 수십 cm 담당
    position_coarse = RewTerm(
        func=mdp.position_error_tanh,
        weight=0.5,
        params={"std": 2.0, "command_name": "goal_pose", "asset_cfg": SceneEntityCfg("robot")},
    )
    position_fine = RewTerm(
        func=mdp.position_error_tanh,
        weight=0.5,
        params={"std": 0.3, "command_name": "goal_pose", "asset_cfg": SceneEntityCfg("robot")},
    )
    # 거리에 따라 기준이 바뀌는 방향 보상 (멀면 목표 바라보기 / 가까우면 목표 yaw)
    heading = RewTerm(
        func=mdp.heading_alignment,
        weight=0.5,
        params={
            "gate_dist": HEADING_GATE_DIST,
            "command_name": "goal_pose",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # 위치와 방향을 **곱**으로 묶은 항. 한쪽만 챙기는 것을 막는다
    pose_precise = RewTerm(
        func=mdp.goal_pose_tanh_product,
        weight=1.0,
        params={
            "pos_std": 0.2,
            "yaw_std": 0.3,
            "command_name": "goal_pose",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # 허용오차 안에 있으면 계속 받는 보상. 이 항의 평균이 곧 성공률이다
    at_goal = RewTerm(
        func=mdp.at_goal,
        weight=2.0,
        params={
            "pos_tol": POS_TOL,
            "yaw_tol": YAW_TOL,
            "command_name": "goal_pose",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # 생존 보너스 (스텝 보상 양수 유지의 보조 장치)
    alive = RewTerm(func=mdp.is_alive, weight=0.2)

    # --- 페널티 -----------------------------------------------------------
    # 이탈/전복 종료. 생존 보너스 상실분과 합쳐 종료가 확실히 손해가 되게 만든다
    terminated = RewTerm(func=mdp.is_terminated, weight=-50.0)
    # 명령 진동 억제. 초기에는 작게 두고 커리큘럼으로 강화한다
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    # 불필요한 선회 억제
    ang_vel = RewTerm(
        func=mdp.ang_vel_z_l2,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # 후진 비용 (금지가 아니라 비용)
    reverse = RewTerm(
        func=mdp.reverse_motion_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 위치/방향 오차 자체는 별도 로깅 항을 두지 않는다.
    # 커맨드 term 이 ``Metrics/goal_pose/error_pos_2d`` 와 ``.../error_heading`` 으로
    # 이미 텐서보드에 남기고 있고, 그것이 학습 상태를 판단하는 1차 지표다.
    # (성공률은 ``Episode_Reward/at_goal`` 로 읽는다)


@configclass
class TerminationsCfg:
    """종료 조건.

    성공 종료가 없는 것은 의도된 설계다 (rewards 모듈 설명 참고).
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    out_of_bounds = DoneTerm(
        func=mdp.root_out_of_env_bounds,
        params={"max_dist": BOUNDS_RADIUS, "asset_cfg": SceneEntityCfg("robot")},
    )

    # 전복. 평지라 정상적으로는 일어나지 않지만, 학습 초기 폭주를 잘라낸다
    flipped = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.7, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class CurriculumCfg:
    """커리큘럼 — 주행을 익힌 뒤에 매끄러움을 요구한다.

    처음부터 페널티를 크게 걸면 "아무것도 하지 않는 것"이 국소 최적해가 된다.
    """

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.02, "num_steps": 12000},
    )
    ang_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "ang_vel", "weight": -0.05, "num_steps": 12000},
    )


##
# 환경
##


@configclass
class AmrPoseNavEnvCfg(ManagerBasedRLEnvCfg):
    """차동구동 AMR 목표 pose 도달 환경 (로봇 무관 기본형)."""

    scene: AmrNavSceneCfg = AmrNavSceneCfg(num_envs=4096, env_spacing=ENV_SPACING)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # 정책 25 Hz (sim 100 Hz / decimation 4)
        self.decimation = 4
        self.sim.dt = 1.0 / 100.0
        self.sim.render_interval = self.decimation
        # 목표 재샘플 8 초 * 3 = 24 초. 나누어떨어지게 맞춰야 마지막 목표에도
        # 도달할 시간이 주어진다 (ridgeback 에서 15초/7초로 어긋나 있었다)
        self.episode_length_s = 24.0
        self.viewer.eye = (6.0, 6.0, 5.0)
