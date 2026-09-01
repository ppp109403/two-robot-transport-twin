# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""자체 설계 AMR + Fairino FR3 모바일 매니퓰레이터의 EE pose 추종 태스크.

목표는 env 원점 기준으로 뿌려지고 **월드에 고정**된다. 목표 범위가 팔 리치(약 0.72 m)
보다 훨씬 넓으므로 팔만으로는 도달할 수 없고 **베이스가 주행해야** 한다.

앞선 두 태스크에서 검증된 것을 그대로 조합한 것이다.
  * 커맨드/보상 : ``Reach-RidgebackFranka-PPO-v0`` 의 월드 고정 pose 스택
  * 베이스 액션 : ``Nav-AMR-Diff-PPO-v0`` 의 :class:`DiffDriveAction` (v, ω)
  * 로봇        : :mod:`~ee_track_ppo.assets.amr_fr3`

ridgeback 과 다른 점
--------------------
1. 베이스가 **논홀로노믹**이다. ridgeback 은 홀로노믹 planar 3-DOF 더미 조인트라
   옆으로 미끄러질 수 있었지만, 차동구동은 횡방향으로 못 움직인다.
   -> 목표 쪽으로 가려면 먼저 몸을 돌려야 하고, 그만큼 태스크가 어렵다.
2. 베이스 액션이 관절 속도가 아니라 **바디 트위스트 (v, ω)** 다.
   실로봇의 ``/cmd_vel`` 과 같은 인터페이스라 정책을 그대로 옮길 수 있다.
3. 팔이 **6축**이다 (ridgeback 의 Franka 는 7축). 여유자유도가 없어 IK 해가 이산적이고,
   자세까지 맞추려면 베이스 위치가 더 중요해진다.
4. EE 가 툴 플랜지가 아니라 **그리퍼 TCP**(``tool0``)다. 파이퍼 2핑거 그리퍼가 달려 있고
   플랜지에서 TCP 까지 266 mm 나간다. 그리퍼 조인트는 정책이 제어하지 않고
   액추에이터가 닫힘 자세를 유지한다 (이 태스크는 파지가 아니라 pose 추종이다).

관측의 베이스 pose 출처에 대해
------------------------------
목표가 **맵 프레임**으로 주어지므로, 그것을 로봇 프레임으로 옮기려면 로봇이 자기
위치를 알아야 한다. 시뮬은 참값을 알지만 실로봇은 AMCL 추정값뿐이고 수 cm 가 틀린다.
그 오차는 EE 목표에 그대로 더해지므로 **맵 기준 EE 정확도의 상한이 AMCL 정확도**다.

2026-08-04 부터 :class:`~...reach.mdp.UniformPoseWorldAmclCommand` 로
**AMCL 을 제대로 모델링**한다 (갱신주기·지연·노이즈·점프). 그 전에는 ``pose_command``
관측에 균등노이즈 ±0.02 를 얹은 1차 근사였는데, 그것으로는 AMCL 이 튀거나 갱신이
늦을 때의 거동을 배우지 못한다.

무엇이 참값이고 무엇이 추정값인지::

    pose_command_w   참값. 월드에 고정된 진짜 목표
    pose_command_b   **추정값 기준**. 정책이 보는 것 (관측)
    metrics          참값 기준. 실제로 얼마나 틀렸는지 (평가)

정책은 추정값을 보고 움직이지만 평가는 참값으로 받는다. 그래서 지표에 **줄일 수 없는
바닥**이 남고, 그 바닥이 곧 로컬라이제이션 오차다. 그것이 정직한 성능 예측이다.

.. warning::
   AMCL 파라미터는 아직 **실측이 아니다** (``AMCL_*`` 상수 참고).
   3 cm 정확도 목표의 성패가 그 값들에 걸려 있다.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
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

# 이 태스크의 MDP term 은 두 모듈에서 온다.
#   reach.mdp : 월드 고정 pose 커맨드/보상/관측 (+ Isaac Lab 기본 term 전체)
#   nav.mdp   : 차동구동 액션과 베이스 전용 보상/종료
import ee_track_ppo.tasks.manager_based.reach.mdp as mdp  # isort: skip
from ee_track_ppo.tasks.manager_based.nav.mdp import (  # isort: skip
    DiffDriveActionCfg,
    ang_vel_z_l2,
    base_twist_2d,
    reverse_motion_l2,
    root_out_of_env_bounds,
)
from ee_track_ppo.assets.amr_fr3 import (  # isort: skip
    AMR_FR3_ARM_JOINTS,
    AMR_FR3_BASE_BODY,
    AMR_FR3_CFG,
    AMR_FR3_EE_BODY,
    AMR_FR3_LEFT_WHEEL_JOINT,
    AMR_FR3_RIGHT_WHEEL_JOINT,
    AMR_FR3_WHEEL_BASE,
    AMR_FR3_WHEEL_RADIUS,
    AMR_FR3_WRIST_JOINTS,
)

##
# 태스크 규격
#
#   최대속도 0.5 m/s 로 8 초에 갈 수 있는 거리는 4 m 지만, 초기 선회(최악 pi rad /
#   1.0 rad/s = 3.1 초)와 도착 후 팔 자세 정리 시간을 빼면 실제 주행 시간은 약 4 초 = 2 m 다.
#   목표 범위 ±1.2 m 의 최대 거리는 대각선 1.70 m 이고, 여기서 팔 리치 0.72 m 를 빼면
#   베이스가 실제로 이동해야 하는 거리는 최대 약 1.0 m 다. 충분히 도달 가능하다.
#
#   범위를 이보다 키우면 시간 안에 도달 불가능한 목표가 섞이고, 그 에피소드의
#   학습 신호는 잡음이 된다.
##

GOAL_RANGE_XY = 1.2
"""목표 xy 샘플링 범위 [m]. env 원점 기준."""

GOAL_Z_RANGE = (0.45, 0.60)
"""목표 TCP 높이 범위 [m, 지면 기준]. **도달률 실측으로 정한 값이다.**

한때 ``(0.45, 0.95)`` 였는데 **상단 절반이 도달 불가능**했다. 2026-07-31 실측
(pitch=180도 고정, yaw 전범위, 방위각 자유 = 베이스 선회 가정, rho 0.20~0.50)::

    지면 높이       도달률
    0.45 ~ 0.50    96.30%
    0.50 ~ 0.55    95.78%
    0.55 ~ 0.60    95.15%     <- 여기까지가 상한
    0.60 ~ 0.65    86.45%
    0.65 ~ 0.70    63.48%
    0.70 ~ 0.80    20.47%
    0.80 ~ 0.95     0.00%     <- 기존 설정의 상단 30% 가 전부 여기였다

**도달 불가능한 목표가 섞이면 위치 보상이 원리적으로 포화하지 못해 학습 신호가
소멸한다** (franka v1 에서 겪은 함정, `check_reach_box.py` 참고). 그 에피소드의
경사는 잡음이 되고, 정책은 "어차피 못 가는 목표" 를 평균적으로 포기하는 쪽으로 학습한다.

왜 이렇게 낮은가 — 그리퍼가 TCP 를 플랜지에서 **266 mm 아래**로 끌어내리기 때문이다.
같은 팔이 플랜지 기준으로는 지면 0.565~0.885 m 를 커버했다 (v4c). 자세히는
`handoff/BASELINE_INTEGRATED.md` §8 참고.
"""

GOAL_PERIOD = 8.0
"""목표 재샘플 주기 [s]."""

EPISODE_LENGTH = 24.0
"""에피소드 길이 [s]. GOAL_PERIOD 의 정수배여야 마지막 목표에도 도달 시간이 주어진다."""

BOUNDS_RADIUS = 3.0
"""베이스 이탈 종료 반경 [m]."""

ENV_SPACING = 10.0
"""env 격자 간격 [m].

**이웃 env 침범 방지 조건**: 반칸(ENV_SPACING/2) > BOUNDS_RADIUS + 로봇 최대 반경

  BOUNDS_RADIUS      3.0 m   (베이스가 원점에서 이만큼 벗어나면 종료)
  로봇 최대 반경    ~1.1 m   (차체 반길이 ~0.4 + 팔 리치 0.72)
  필요 반칸          4.1 m

8.0 이면 반칸이 정확히 4.0 m 라 **여유가 없다**. 이탈 한계 지점에서 팔을 이웃 쪽으로
뻗으면 이웃 칸에 닿는다. Ridgeback 태스크에서 이 침범이 물리 폭주를 일으킨 전례가 있다.
10.0 으로 두면 반칸 5.0 m 로 0.9 m 여유가 생긴다.
(BOUNDS_RADIUS 를 줄이는 방법도 있으나 그건 MDP 자체를 바꾸므로 간격만 넓힌다)
"""

# 베이스 주행 한계 (AMR 단독 태스크와 같은 값)
MAX_LIN_VEL = 0.5
MAX_ANG_VEL = 1.0
MAX_REVERSE_VEL = 0.2
MAX_LIN_ACCEL = 0.6
MAX_ANG_ACCEL = 1.5
LIN_GAIN_RANGE = (0.95, 1.15)
ANG_GAIN_RANGE = (1.15, 1.65)


##
# 씬
##


@configclass
class AmrFr3ReachSceneCfg(InteractiveSceneCfg):
    """평지 + 모바일 매니퓰레이터."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    robot = AMR_FR3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP
##


##
# AMCL 모델 — 실기 실측으로 대체해야 하는 값들
#
#   목표가 **맵 프레임**으로 주어지는 운용에서는 로컬라이제이션 오차가 EE 목표에
#   그대로 더해진다. 차체가 2 cm 틀린 곳에 있다고 믿으면 팔이 아무리 정확해도
#   EE 는 2 cm 틀린 곳으로 간다. **맵 기준 EE 정확도의 상한이 AMCL 정확도**다.
#
#   아래 기본값은 ``nav`` 태스크의 모델에서 가져온 것이고 **실측이 아니다.**
#   3 cm 목표의 성패가 이 숫자들에 걸려 있으므로 실기에서 재야 한다::
#
#       1) 한 지점에 세워두고 /amcl_pose 를 1~2분 기록  -> 노이즈, 발행주기, 튐
#       2) 같은 지점으로 내비 5~10회 반복 도착 후 기록   -> 재현성 (실질 오차)
#       3) /amcl_pose 의 covariance                     -> AMCL 자신의 불확실성
#
#   목표를 "로봇을 그 자리로 데려가 AMCL 값을 기록" 하는 방식으로 지정하면
#   **일정한 편향이 목표와 실행에 똑같이 들어가 상쇄된다.** 그 경우 2) 의 재현성만
#   문제가 되므로 훨씬 유리하다.
##

AMCL_POS_NOISE = 0.005
"""정상상태 위치 추정 오차 [m]. **실기 로그 실측 기반** (2026-08-04)."""

AMCL_YAW_NOISE = 0.004
"""정상상태 yaw 추정 오차 [rad]. **실기 로그 실측 기반**.

거리에 비례해 위치 오차로 번진다. 목표가 base 에서 0.5 m 면 2 mm 다.
"""

AMCL_UPDATE_PERIOD = 0.18
"""갱신 주기 [s]. **실측 중앙값 0.180 s (5.6 Hz)**.

nav2 설정이 ``update_min_d: 0.25`` / ``update_min_a: 0.2`` 라, 로봇이 25 cm 를
움직이거나 11.5도 돌아야 필터가 돈다. 그 사이에는 EKF 오도메트리(``odometry/filtered``)
로만 pose 를 이어간다. 최대 간격은 3.8 초까지 관측됐다.
"""

AMCL_LATENCY = 0.06
"""스캔 취득 -> 매칭 -> 퍼블리시 지연 [s]. **로그로는 알 수 없어 추정값 유지.**"""

AMCL_JUMP_PROB = 0.002
"""스텝당 점프 확률. 파티클 필터 재수렴 시의 불연속 도약.

실측에서 최대 86.8 mm 도약이 관측됐으므로 유지한다.
"""

@configclass
class CommandsCfg:
    """목표 EE pose 생성기 (월드 고정, **AMCL 추정 pose 기준**)."""

    ee_pose = mdp.UniformPoseWorldAmclCommandCfg(
        asset_name="robot",
        body_name=AMR_FR3_EE_BODY,
        resampling_time_range=(GOAL_PERIOD, GOAL_PERIOD),
        debug_vis=True,
        pos_noise_std=AMCL_POS_NOISE,
        yaw_noise_std=AMCL_YAW_NOISE,
        update_period=AMCL_UPDATE_PERIOD,
        latency=AMCL_LATENCY,
        jump_prob=AMCL_JUMP_PROB,
        ranges=mdp.UniformPoseWorldAmclCommandCfg.Ranges(
            pos_x=(-GOAL_RANGE_XY, GOAL_RANGE_XY),
            pos_y=(-GOAL_RANGE_XY, GOAL_RANGE_XY),
            pos_z=GOAL_Z_RANGE,
            roll=(0.0, 0.0),
            # EE(tool0)의 z축은 그리퍼가 뻗는 방향이라 플랜지 z축과 같다.
            # 따라서 pitch=pi 가 "그리퍼가 아래를 향함"이고, 파지 작업의 표준 접근 자세다.
            pitch=(math.pi, math.pi),
            yaw=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """액션: 팔(관절 위치 6) + 베이스(바디 트위스트 2) = 8차원."""

    # 팔: 기본자세 기준 상대 오프셋
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=AMR_FR3_ARM_JOINTS,
        scale=0.5,
        use_default_offset=True,
    )

    # 베이스: (v, ω). 실로봇의 /cmd_vel 과 같은 의미다.
    base_action = DiffDriveActionCfg(
        asset_name="robot",
        left_wheel_joint_names=[AMR_FR3_LEFT_WHEEL_JOINT],
        right_wheel_joint_names=[AMR_FR3_RIGHT_WHEEL_JOINT],
        wheel_radius=AMR_FR3_WHEEL_RADIUS,
        wheel_base=AMR_FR3_WHEEL_BASE,
        max_lin_vel=MAX_LIN_VEL,
        max_ang_vel=MAX_ANG_VEL,
        max_reverse_vel=MAX_REVERSE_VEL,
        max_lin_accel=MAX_LIN_ACCEL,
        max_ang_accel=MAX_ANG_ACCEL,
        lin_gain_range=LIN_GAIN_RANGE,
        ang_gain_range=ANG_GAIN_RANGE,
    )


@configclass
class ObservationsCfg:
    """관측."""

    @configclass
    class PolicyCfg(ObsGroup):
        # --- 팔 상태 [12] ---
        #   ★ 반드시 팔 조인트만 골라야 한다.
        #     구동륜/캐스터는 continuous 조인트라 관절각이 무한정 누적된다.
        #     그대로 넣으면 관측이 발산하고 정규화도 의미를 잃는다.
        arm_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=AMR_FR3_ARM_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        arm_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=AMR_FR3_ARM_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        # --- 베이스 상태 [2] ---
        #   실로봇에서는 휠 엔코더 오도메트리(/odom)의 트위스트에 대응한다.
        base_twist = ObsTerm(
            func=base_twist_2d,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        # --- 목표 [7] ---
        #   root 프레임 기준 목표 pose (xyz + quat).
        #   노이즈는 실로봇의 로컬라이제이션 오차를 1차 근사한 것이다
        #   (AMCL 위치 ~2 cm / 방향 ~0.02 rad).
        #   **노이즈를 0.02 -> 0.002 로 줄였다** (2026-08-04).
        #   기존 0.02 는 로컬라이제이션 오차를 관측 노이즈로 **근사**한 값이었다.
        #   이제 커맨드 term(UniformPoseWorldAmclCommand)이 AMCL 을 제대로 모델링하므로
        #   (갱신주기·지연·노이즈·점프) 여기에 또 얹으면 **이중 계상**이 된다.
        #   남긴 0.002 는 수치·양자화 수준의 잔여 노이즈다.
        pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "ee_pose"},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )

        # --- EE 상태 [7 + 6] ---
        #   정책이 자기 EE 위치를 직접 보지 못하면 순기구학을 신경망 안에서
        #   암묵 학습해야 하고, 그것이 정밀도의 병목이 된다.
        #   (franka 실험 v3->v4 에서 이 두 항으로 위치 오차 0.220 -> 0.011)
        ee_pose = ObsTerm(
            func=mdp.ee_pose_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[AMR_FR3_EE_BODY])},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        ee_pose_error = ObsTerm(
            func=mdp.ee_pose_error_b,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[AMR_FR3_EE_BODY]),
                "command_name": "ee_pose",
            },
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )

        # --- 이전 액션 [8] ---
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """리셋 이벤트 + 도메인 랜덤화."""

    # --- reset: 중력 배치 ---
    #   씬 전역 중력은 0 이다 (__post_init__ 참고). 실기 컨트롤러가 팔 중력을 보상하므로
    #   정책 입장의 팔은 무중력이어야 하고, 팔 단독 태스크(v3~v4)도 그렇게 학습했다.
    #   차체는 바퀴로 바닥을 딛어야 주행하므로 여기서만 중력을 되살린다.
    chassis_gravity = EventTerm(
        func=mdp.gravity_on_bodies_only,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "base_link",
                    "wheel_left_link",
                    "wheel_right_link",
                    "caster_.*_link",
                ],
            ),
        },
    )

    # --- startup: sim2real 물성 랜덤화 ---
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
    # 적재 하중은 베이스 링크에만. 팔 링크 질량까지 바꾸면 팔 제어 특성이 흔들린다.
    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[AMR_FR3_BASE_BODY]),
            "mass_distribution_params": (0.8, 1.4),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    # --- reset ---
    #   루트 pose 무작위화. yaw 를 전 범위로 뿌려야 제자리 선회를 제대로 배운다.
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
        },
    )
    # 팔 초기자세 산포. 루트가 실제 바디(base_link)이고 루트 pose 를 따로 리셋하므로
    # 팔을 흔들어도 베이스 위치가 어긋나지 않는다 (ridgeback 이 겪었던 제약이 없다).
    #
    #   **``by_scale`` 이 아니라 ``by_offset`` 이어야 한다.**
    #   ``reset_joints_by_scale`` 은 기본값에 배율을 곱하므로 기본값이 0 인 관절은
    #   0 x 0.7~1.3 = 0 이 되어 **영원히 랜덤화되지 않는다.** 기본자세는
    #   j1=0, j6=0 이라 정확히 그 둘이 항상 같은 값으로 시작했다.
    #   j6 는 감김 문제의 당사자이고 j1 은 방위각을 담당하는 관절이라, 초기 산포가
    #   없으면 정책이 특정 진입 방향만 보게 된다.
    #   (`CLOSEOUT_ARM_ONLY.md` §4-B 부수 발견)
    #   **j2/j3 는 좁게, 나머지는 넓게** 준다. 균등하게 ±0.45 를 주면 어깨·팔꿈치가
    #   차체와 겹친 자세로 놓여 PhysX 가 관통을 밀어내고, 그 반동으로 **차체가 튀어
    #   오른다.** 공중에서는 기울기가 잡히지 않아 그대로 넘어간다.
    #
    #   실측 (2026-08-04, scripts/diagnose_flip.py + 리셋 자세 충돌 검사)::
    #
    #       리셋 산포 조합                    자기충돌률
    #       전체 ±0.45                        3.47%      <- 기존
    #       전체 ±0.20                        0.00%
    #       j2/j3 ±0.25, 나머지 ±0.60         0.00%      <- 채택. 산포는 오히려 넓다
    #
    #       전복 22 회 분석: 5~20 스텝에 50%, 20~100 스텝에 45%
    #                       전복 시점 차체 높이 0.1245 m (정상 0.0100)
    #                       기울기 13 -> 16 -> 21 -> 26 -> 32 -> 38도 (스텝당 5~6도)
    #
    #   전복률이 학습 내내 1.3% 로 **일정**했던 것이 이것으로 설명된다 —
    #   정책이 배워서 피할 수 있는 것이 아니다.
    reset_arm_shoulder = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.25, 0.25),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["j[2-3]"]),
        },
    )
    reset_arm_rest = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.60, 0.60),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["j1", "j[4-6]"]),
        },
    )
    # 구동륜/캐스터는 0 으로. 안 하면 이전 에피소드의 회전속도가 이어진다.
    reset_wheels = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["wheel_.*", "caster_.*"]),
        },
    )


@configclass
class RewardsCfg:
    """보상 — ridgeback 태스크에서 수렴을 확인한 구성 그대로.

    설계 원칙은 **스텝 보상 총합을 양수로 유지**하는 것이다. 그러면 오래 살수록
    이득이라 정책이 일부러 종료를 유발할 유인이 사라진다.
    """

    # --- 위치 ---
    # coarse: 먼 거리 담당. 전 구간에서 "다가갈수록 커지는" 단조 신호를 만든다.
    ee_position_coarse = RewTerm(
        func=mdp.position_command_error_tanh_w,
        weight=0.4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[AMR_FR3_EE_BODY]), "std": 1.0, "command_name": "ee_pose"},
    )
    # 접근 속도 (potential-based shaping). 베이스를 쓰게 만드는 핵심 항이다.
    #   정적 보상만으로는 "수 초간 주행해야 비로소 가까워지는" 구간에 신호가 없어
    #   정책이 "베이스를 안 쓰고 팔만 뻗는" 국소 최적해에 갇힌다 (ridgeback v8 실측).
    ee_position_progress = RewTerm(
        func=mdp.position_command_progress_w,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[AMR_FR3_EE_BODY]), "command_name": "ee_pose"},
    )
    # fine: 마지막 수 cm
    ee_position_fine = RewTerm(
        func=mdp.position_command_error_tanh_w,
        weight=0.35,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[AMR_FR3_EE_BODY]), "std": 0.1, "command_name": "ee_pose"},
    )
    # precise: 운용 오차 규모(수 cm) 전용 칸.
    #
    #   **이 칸이 없어서 위치가 역주행했다.** `full_260730` (3000 it) 실측::
    #
    #       구간           위치 [mm]   자세 [deg]
    #       it1000-1499     24.37       7.62
    #       it2500-2999     33.25       3.83     <- 위치는 나빠지고 자세만 좋아진다
    #
    #   보상 항별로 보면 정책이 **의도대로 최적화**하고 있었다. 같은 구간에서
    #   자세 3 칸 합이 +0.0365 오르고 위치 3 칸 합이 -0.0109 떨어졌다. 순이득이라
    #   위치를 팔아 자세를 산 것이다. 목적함수가 그렇게 생겼던 게 문제다.
    #
    #   원인은 **사다리 깊이의 비대칭**이다::
    #
    #       위치 사다리   std 1.0 -> 0.1 에서 끝난다
    #       자세 사다리   coarse -> 0.3 -> 0.05
    #
    #   오차 33 mm 에서 위치의 가장 좁은 칸은 e/std = 0.33 이라 기울기 정점을 지나
    #   약해지는 구간인데, 자세는 3.83도에서 e/std = 1.34 로 정점 근처다.
    #   **자세 사다리만 운용 규모에 닿아 있어** 정책이 계속 자세를 산다.
    #
    #   팔 단독 체인은 이 칸을 갖고 있었다 (`fairino_fr3_v2` 의 `_add_position_precise`,
    #   std 0.01~0.02 / w 0.15~0.20). 그 설정이 실기 1.0 mm 를 냈다.
    #   여기서는 베이스 이동 오차가 더해지므로 std 를 조금 크게 잡는다.
    #
    #   tanh 사다리는 **칸을 더하지 빼지 않는다** — 기존 std 0.1 칸은 그대로 둔다.
    #   빼면 큰 오차 구간의 신호가 사라져 초기 수렴이 무너진다 (v3b 붕괴 사례).
    ee_position_precise = RewTerm(
        func=mdp.position_command_error_tanh_w,
        weight=0.25,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[AMR_FR3_EE_BODY]), "std": 0.03, "command_name": "ee_pose"},
    )

    # --- 자세 ---
    ee_orientation = RewTerm(
        func=mdp.orientation_command_error_w,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[AMR_FR3_EE_BODY]), "command_name": "ee_pose"},
    )
    ee_orientation_fine = RewTerm(
        func=mdp.orientation_command_error_tanh_w,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[AMR_FR3_EE_BODY]), "std": 0.3, "command_name": "ee_pose"},
    )
    ee_orientation_precise = RewTerm(
        func=mdp.orientation_command_error_tanh_w,
        weight=0.15,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[AMR_FR3_EE_BODY]), "std": 0.05, "command_name": "ee_pose"},
    )

    # --- 생존 / 종료 ---
    alive = RewTerm(func=mdp.is_alive, weight=0.2)
    terminated = RewTerm(func=mdp.is_terminated, weight=-50.0)

    # --- 팔 정규화 ---
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    arm_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=AMR_FR3_ARM_JOINTS)},
    )
    # 손목 속도: 도달 후 떨림을 잡는 항
    wrist_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=AMR_FR3_WRIST_JOINTS)},
    )
    # 관절 가속도: action_rate 가 못 잡는 미세 고주파 떨림 전용.
    #   1차 차분은 매 스텝 부호가 바뀌는 저진폭 진동에 거의 안 걸리지만,
    #   2차 미분은 진폭 A / 주파수 f 진동에서 A*f^2 에 비례해 고주파에 제곱으로 민감하다.
    arm_joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=AMR_FR3_ARM_JOINTS)},
    )

    # --- 베이스 정규화 ---
    #   도달 후 베이스가 목표 주변에서 진동하는 것을 억제한다.
    base_ang_vel = RewTerm(
        func=ang_vel_z_l2,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    #   후진은 금지가 아니라 비용 (후방 센서가 없는 실로봇 특성)
    base_reverse = RewTerm(
        func=reverse_motion_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """종료 조건."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_out_of_bounds = DoneTerm(
        func=root_out_of_env_bounds,
        params={"max_dist": BOUNDS_RADIUS, "asset_cfg": SceneEntityCfg("robot")},
    )

    # 전복. 팔이 앞쪽 위에 붙어 질량중심이 높으므로 평지라도 넣어 둔다.
    # 전복. **``mdp.bad_orientation`` 을 쓰면 안 된다** — 그쪽은 투영 중력 벡터로
    # 기울기를 재는데 이 태스크는 sim.gravity=0 이라 그 벡터가 영벡터가 되어
    # 항상 참이 된다 (2026-07-31: 모든 에피소드가 1 스텝 만에 종료돼 3000 it 이 통째로
    # 날아갔다). 루트 자세에서 직접 재는 term 을 쓴다.
    flipped = DoneTerm(
        func=mdp.base_tilt_exceeds,
        params={"limit_angle": 0.7, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class CurriculumCfg:
    """커리큘럼 — 도달을 익힌 뒤에 매끄러움을 요구한다.

    처음부터 페널티를 크게 걸면 "아무것도 하지 않는 것"이 국소 최적해가 된다.
    """

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.01, "num_steps": 12000}
    )
    arm_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "arm_joint_vel", "weight": -0.001, "num_steps": 12000}
    )


##
# 환경
##


@configclass
class AmrFr3ReachEnvCfg(ManagerBasedRLEnvCfg):
    """AMR + FR3 모바일 매니퓰레이터 EE pose 추종 환경."""

    scene: AmrFr3ReachSceneCfg = AmrFr3ReachSceneCfg(num_envs=4096, env_spacing=ENV_SPACING)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # 정책 **30 Hz** (sim 120 Hz / decimation 4).
        #   원래 25 Hz (sim 100 / dec 4) 였다. AMR 단독 태스크의 /cmd_vel 주기에 맞춘
        #   값이었는데, **팔 쪽 검증 자산이 전부 30 Hz 기준**이라 그쪽으로 통일한다::
        #
        #       alpha / follow_gain   30 Hz 조건에서 측정 (서보 200 / 정책 30 = 6.667)
        #       지연 주입 72 ms       정책 스텝 단위로 환산된 값
        #       실측 시정수 0.16 s     30 Hz 정책으로 낸 값
        #       보상 사다리 std        30 Hz 에서의 오차 규모에 맞춰 조정된 값
        #
        #   25 Hz 로 두면 한 주기가 33% 길어져 alpha 가 달라지고 follow_gain 역산의
        #   분모도 6.667 -> 8 로 바뀐다. 같은 alpha 를 가정해도 0.0377 -> 0.0316 이다.
        #   차체와 팔이 같은 주기로 돌 이유는 없으므로, 검증된 쪽에 맞추는 것이 싸다.
        #
        #   물리는 100 Hz 가 아니라 **120 Hz** 로 올렸다. 정책 주기를 1/30 s 로 맞추면서
        #   차체 바퀴/캐스터 접촉 해상도를 유지하기 위해서다 (팔 단독은 60 Hz 였다).
        self.decimation = 4
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation

        # 전역 중력 0. 실기 컨트롤러가 팔 중력을 보상하므로 정책 입장의 팔은 무중력이
        # 맞다 (v3 에서 확립, 실기 편향 12.1 mm 해소). 차체 중력은 events 의
        # chassis_gravity 가 외력으로 되살린다.
        self.sim.gravity = (0.0, 0.0, 0.0)
        self.episode_length_s = EPISODE_LENGTH
        self.viewer.eye = (4.0, 4.0, 3.0)

        # 자기충돌을 켠다. **끄고 학습하면 팔이 차체를 통과한다.**
        #   에셋 기본값(amr_fr3.py)은 False 라 여기서 덮어써야 한다. 팔 단독 v4a 에서
        #   같은 조치를 했고(ladder_env_cfg), 통합 모델은 차체까지 있어 더 중요하다.
        #
        #   왜 중요한가 (2026-07-31 측정): 확정 껍질에서 IK 해의 **1.3%** 가 팔-차체 충돌이다.
        #   끄면 그 자세들이 도달 가능으로 학습되는데, 학습 곡선도 시뮬 지표도 정상으로
        #   보이므로 **지표로는 발견되지 않는다.** 실기에 올려야 부딪힌다.
        #   v3b 가 충돌을 무시한 껍질(60 L)로 실기 폴트를 낸 것과 같은 계열이다.
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True


@configclass
class AmrFr3ReachEnvCfg_PLAY(AmrFr3ReachEnvCfg):
    """학습 결과 확인용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        # [수정] 6.0 -> ENV_SPACING(10.0).
        #   PLAY 도 BOUNDS_RADIUS(3.0) + 로봇 반경(~1.1) = 4.1 m 가 필요한데
        #   6.0 이면 반칸이 3.0 m 라 이웃 침범이 가능했다. 학습과 같은 간격을 쓴다.
        self.scene.env_spacing = ENV_SPACING
        self.observations.policy.enable_corruption = False
