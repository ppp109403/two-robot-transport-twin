# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""협력 운송 — **움직이는 EE 목표 궤적 추종** (STAGE4).

무엇이 달라지는가
-----------------
:class:`AmrFr3ReachEnvCfg` (통합 14.92 mm) 는 목표가 **월드에 고정**이고 8 초마다
순간이동한다. 정책이 배운 것은 "차로 가서 -> 멈추고 -> 팔로 정밀하게" 다.

협력 운송은 정지 구간이 없다. 상위 계획기가 EE 궤적을 뿌려 주고, 로봇은 0.2 m/s 로
달리는 내내 정밀도를 유지해야 한다. **정지가 없다는 것이 핵심 차이다.**

시나리오 전제 (2026-08-04 확정)
--------------------------------
::

    운송 속도      0.2 m/s 목표
    상대 로봇      같은 AMR+FR3.  **로봇간 직접 통신 없음**
    물체          없음 또는 고무줄처럼 변형되는 것 -> 폐쇄 사슬 결합은 아직 무시
    목표 공급      위에서 각자의 EE 궤적을 뿌려 준다

통신도 강결합도 없으므로 문제는 **"맵 프레임에서 움직이는 EE 목표를 혼자 따라가기"**
로 환원된다. 상대 로봇도 물체도 이 환경에 들어오지 않는다.

세 가지 변경
------------
::

    [램프]        MovingPoseWorldAmclCommand    목표가 월드에서 등속 이동
    [피드포워드]   ee_target_vel_b               목표 속도를 관측에 추가
    [rot6d]       pose_command_6d / ee_pose_6d_b 팔 정책과 자세 표현 통일

셋을 한 번에 바꾸는 이유는 서로 얽혀 있어서다. 램프만 넣으면 피드포워드가 없어
구조적으로 뒤처지고, 피드포워드만 넣으면 목표가 안 움직여 ``target_vel`` 이 항상
0 이라 배울 것이 없다. **램프와 피드포워드는 한 쌍이다.**

rot6d 는 독립이지만 어차피 재학습하므로 같이 얹는다. 팔 정책(arm_tool0)이 이미
rot6d 라, 안 맞추면 실기 브리지에 자세 변환이 **두 벌** 살아 있게 된다.

.. warning::
   **기존 파일은 수정하지 않는다.** :mod:`mobile_reach_env_cfg` 는 통합 14.92 mm 를
   낸 검증된 경로이고, 비교 기준으로 그대로 남겨야 한다. 여기서는 상속만 한다.

관측 배치 — 52 차원
--------------------
::

    [ 0: 6]  arm_joint_pos      j1..j6 상대각
    [ 6:12]  arm_joint_vel      j1..j6
    [12:14]  base_twist         [v_x, omega_z]   실기 /odom 의 twist
    [14:23]  pose_command       목표 xyz 3 + rot6d 6      <- quat 4 에서 바뀜
    [23:32]  ee_pose            현재 EE xyz 3 + rot6d 6   <- quat 4 에서 바뀜
    [32:38]  ee_pose_error      위치오차 3 + 자세오차 3
    [38:46]  last_action        8
    [46:52]  target_vel         목표 선속도 3 + 각속도 3   <- 신규 (피드포워드)

통합 42 차원과 **차원도 순서도 다르다.** 실기 브리지는 이 표를 따라야 한다.

.. warning::
   이 표는 **예상**이다. 확정값은 ``python scripts/dump_obs_layout.py`` 로 뽑아
   대조할 것. 관측 순서는 ``ObservationManager`` 가 인스턴스 ``__dict__`` 삽입
   순서로 정하므로, ``__post_init__`` 에서 붙인 항은 맨 뒤로 간다.

지연 예산 — 회전이 최악이다
---------------------------
목표 속도 0.2 m/s 가 그대로 팔의 부담이 되지는 않는다. 차체가 따라가면 base
프레임에서 목표는 거의 정지해 있고, 팔이 감당하는 것은 잔차다. 그 잔차를 지배하는
것은 **차체 회전**이다 (``omega x r``, r ~= 0.5 m)::

    선회반경 2.0 m  ->  0.05 m/s  ->  지연  8 mm
    선회반경 1.0 m  ->  0.10 m/s  ->  지연 16 mm
    선회반경 0.5 m  ->  0.20 m/s  ->  지연 32 mm      <- 3 cm 초과

학습이 끝나면 **3 cm 를 지키는 최소 선회반경**이 결과로 나온다. 그것이 상위 궤적
생성기가 지켜야 할 제약이 된다.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import ee_track_ppo.tasks.manager_based.reach.mdp as mdp  # isort: skip
from ee_track_ppo.assets.amr_fr3 import AMR_FR3_EE_BODY  # isort: skip
from .mobile_reach_env_cfg import GOAL_RANGE_XY, GOAL_Z_RANGE, AmrFr3ReachEnvCfg  # isort: skip

##
# 궤적 파라미터
##

TARGET_SPEED_RANGE = (0.0, 0.30)
"""목표의 이동 속력 범위 [m/s].

운용 목표는 0.2 다. 위아래로 넓게 잡는 이유는 **속도별 오차 곡선을 뽑기 위해서**다.
0.2 만 학습하면 "0.2 에서 몇 mm" 만 알 수 있고, 상위 궤적 생성기가 속도를 정할 때
쓸 근거가 안 나온다. 0 을 포함시켜 정지 목표(= 기존 태스크)도 계속 연습시킨다.
"""

TARGET_ANG_SPEED_RANGE = (0.0, 0.40)
"""목표 **자세**의 회전 각속도 범위 [rad/s].

.. note::
   이것은 **차체 선회율이 아니다.** 목표 pose 의 자세가 얼마나 빨리 도느냐다.
   두 가지가 EE 를 흔드는 경로는 서로 다르다::

       목표 자세 회전   손목(j4~j6)이 따라 돌아야 한다.  이 파라미터가 정한다
       차체 선회        base 프레임에서 목표 위치가 omega x r 로 쓸린다.
                        정책이 주행하며 **저절로 생기는** 것이라 여기서 못 정한다

   0.40 rad/s 는 손목 속도상한(3.2 rad/s)에 한참 못 미치므로 자세 추종 자체는
   여유가 있다. 넓게 잡는 이유는 자세가 정지한 목표만 보면 손목 피드포워드를
   배우지 못하기 때문이다.

   선회반경 제약(위 지연 예산)은 이 값이 아니라 **학습 후 재생에서 측정**해야
   나온다. 차체를 얼마나 돌릴지는 정책이 정한다.
"""

WAYPOINT_TOLERANCE = 0.05
"""이 거리 안에 들어오면 새 waypoint 를 뽑는다 [m]."""

ANGLE_TOLERANCE = 0.10
"""이 각도 안에 들어오면 새 waypoint 를 뽑는다 [rad]."""

TRAJ_PERIOD = 60.0
"""커맨드 재샘플링 주기 [s].

에피소드보다 길게 두어 **중간 점프가 없게** 한다. 목표는 waypoint 를 향해 연속
이동하므로 재샘플링은 에피소드 리셋 초기화 용도로만 쓰인다.
"""


@configclass
class AmrFr3TrajEnvCfg(AmrFr3ReachEnvCfg):
    """움직이는 EE 목표를 추종하는 통합 태스크."""

    def __post_init__(self):
        super().__post_init__()

        # --- [램프] 목표가 월드에서 연속 이동한다 ---
        #   기존 UniformPoseWorldAmclCommandCfg 의 AMCL 값을 그대로 옮긴다.
        #   (실기 로그 실측치. 두 곳이 어긋나면 안 된다)
        old = self.commands.ee_pose
        self.commands.ee_pose = mdp.MovingPoseWorldAmclCommandCfg(
            asset_name=old.asset_name,
            body_name=old.body_name,
            resampling_time_range=(TRAJ_PERIOD, TRAJ_PERIOD),
            debug_vis=old.debug_vis,
            ranges=mdp.MovingPoseWorldAmclCommandCfg.Ranges(
                pos_x=(-GOAL_RANGE_XY, GOAL_RANGE_XY),
                pos_y=(-GOAL_RANGE_XY, GOAL_RANGE_XY),
                pos_z=GOAL_Z_RANGE,
                roll=old.ranges.roll,
                pitch=old.ranges.pitch,
                yaw=old.ranges.yaw,
            ),
            speed_range=TARGET_SPEED_RANGE,
            ang_speed_range=TARGET_ANG_SPEED_RANGE,
            waypoint_tolerance=WAYPOINT_TOLERANCE,
            angle_tolerance=ANGLE_TOLERANCE,
            pos_noise_std=old.pos_noise_std,
            yaw_noise_std=old.yaw_noise_std,
            update_period=old.update_period,
            latency=old.latency,
            jump_prob=old.jump_prob,
            jump_pos_std=old.jump_pos_std,
            jump_yaw_std=old.jump_yaw_std,
        )

        # --- [rot6d] 자세 표현을 팔 정책과 통일한다 ---
        #   쿼터니언은 pitch=pi 에서 w=0 이라 부호 이중성이 특이면에 걸린다.
        #   실제로 사고가 났던 항이다 (ROT6D_CONTRACT.md).
        self.observations.policy.pose_command = ObsTerm(
            func=mdp.pose_command_6d,
            params={"command_name": "ee_pose"},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        self.observations.policy.ee_pose = ObsTerm(
            func=mdp.ee_pose_6d_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[AMR_FR3_EE_BODY])},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )

        # --- [피드포워드] 목표 속도를 관측에 넣는다 ---
        #   오차만 보는 정책은 순수 피드백이라 움직이는 기준에 대해
        #   '속도 x 시정수' 만큼 구조적으로 뒤처진다 (시정수 0.159 s 실측).
        #   목표가 어디로 얼마나 빨리 가는지를 알려주면 미리 움직일 수 있다.
        #
        #   .. note::
        #      여기서 붙이면 관측 **맨 뒤**로 간다 (ObservationManager 가 인스턴스
        #      ``__dict__`` 삽입 순서를 따르므로). 순서를 억지로 바꾸지 않는다 —
        #      학습에는 영향이 없고, 실기 브리지는 실제 순서를 따르면 된다.
        #      확정 배치는 ``scripts/dump_obs_layout.py`` 로 뽑아 문서에 박는다.
        self.observations.policy.target_vel = ObsTerm(
            func=mdp.ee_target_vel_b,
            params={
                "command_name": "ee_pose",
                "asset_cfg": SceneEntityCfg("robot", body_names=[AMR_FR3_EE_BODY]),
            },
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )


@configclass
class AmrFr3TrajEnvCfg_PLAY(AmrFr3TrajEnvCfg):
    """재생·평가용. 도메인 랜덤화와 관측 노이즈를 끈다."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v2 — 액션 지연 주입 (VERIFY_TRAJ.md 대응)
##

DELAY_STEPS = (4, 14)
"""액션 지연 [물리 스텝]. sim dt = 1/120 s 이므로 33 ~ 117 ms.

팔 단독 사슬은 ``MIN/MAX_DELAY_STEPS = 2/7`` 을 쓰는데 그쪽은 sim 60 Hz 다.
**같은 시간**이 되도록 두 배로 환산했다. 스텝 수를 그대로 베끼면 지연이 절반이 된다.
"""


@configclass
class AmrFr3TrajDelayEnvCfg(AmrFr3TrajEnvCfg):
    """v1 에 **액션 지연**을 넣는다. 실기 발산의 근본 대응이다.

    무엇이 잘못됐었나
    -----------------
    ``traj_v1`` 은 실기에서 루프이득 **1.155** 로 발산했다 (`handoff/VERIFY_TRAJ.md`).
    정지 목표에서 EE 가 p-p 30 mm 배회했고, 큰 이동 중 j1 지령이 -3.2 (정상 ±1) 까지
    튀어 카메라가 로봇에 접촉했다.

    원인은 **통합 태스크에 액션 지연이 없다**는 것이다::

        팔 단독 사슬   FairinoFR3Sim2RealEnvCfg 가 DelayedJointPositionActionCfg 를 건다
                      -> V3 -> V4c -> AmrFr3ArmEnvCfg      실기 루프이득 -0.742 (안정)

        통합 사슬      mobile_reach_env_cfg 의 ActionsCfg 가 일반 JointPositionActionCfg
                      -> AmrFr3TrajEnvCfg                  실기 루프이득 -1.155 (발산)

    ``mdp/actions.py`` 의 주석이 이 결과를 그대로 예언하고 있다::

        시뮬에 지연이 없으면 정책은 "내 명령은 즉시 실행된다"를 전제로 학습한다.
        그 정책은 72 ms 지연 앞에서 반드시 발산한다.

    .. note::
       ``mobile_reach_env_cfg.__post_init__`` 주석에는 30 Hz 를 택한 근거로
       "지연 주입 72 ms" 가 적혀 있다. **작성자는 지연이 걸려 있다고 믿었는데
       ``ActionsCfg`` 가 일반 액션으로 덮고 있었다.** 상속만 보고 액션을 확인하지
       않으면 이런 누락을 놓친다.

    왜 지연이 이득을 낮추는가
    -------------------------
    지연이 없으면 "세게 밀수록 오차가 빨리 준다" 가 항상 참이라 PPO 는 이득을 계속
    올린다. 지연이 있으면 세게 민 결과가 늦게 오므로 **오버슈트·진동이 보상에서
    깎인다.** 그래서 정책이 스스로 이득을 낮춘다.

    판정
    ----
    재학습 후 ``scripts/loop_gain_sim.py`` 로 먼저 거른다. 실기에 올리기 전에
    시뮬에서 판정할 수 있다는 것이 이번 사이클의 소득이다::

        arm_tool0 (실기 안정, -0.742)   j4열 0.585,  스펙트럴 반경 0.601
        traj_v1   (실기 발산, -1.155)   j4열 1.916,  스펙트럴 반경 5.416
        목표                            arm_tool0 수준까지
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action = mdp.DelayedJointPositionActionCfg(
            asset_name="robot",
            joint_names=self.actions.arm_action.joint_names,
            scale=self.actions.arm_action.scale,
            use_default_offset=self.actions.arm_action.use_default_offset,
            min_delay=DELAY_STEPS[0],
            max_delay=DELAY_STEPS[1],
        )


@configclass
class AmrFr3TrajDelayEnvCfg_PLAY(AmrFr3TrajDelayEnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v3 — 새 하드웨어 기준선 (페이로드 + TCP 60 cm + j6 감김 여유)
##

PAYLOAD_USD = "amr_fr3_payload/usd/amr_fr3_payload.usd"
"""배터리·모터드라이버가 상판에 얹힌 모델. ``assets/data`` 기준 상대경로."""

GOAL_Z_RANGE_V3 = (0.55, 0.65)
"""EE 목표 높이 [m], map 기준. 기존 (0.45, 0.60) 에서 올렸다.

근거 (`scripts/check_shell_collision.py --asset payload`, 표본 4000)::

    rho 0.20~0.44,  z(팔base) 0.155~0.255  = TCP 지면 0.55~0.65
    도달 98.83%,  충돌 0.00%,  최악구역 94.04%,  부피 32.2 L

**대가는 바깥 반경이다.** 팔 길이가 정해져 있어 위로 쓰는 만큼 옆으로 못 간다::

    TCP 지면 0.50 -> rho 0.55 까지
    TCP 지면 0.60 -> rho 0.50 까지
    TCP 지면 0.70 -> rho 0.40 까지

부채꼴을 좁히는 것은 도달성에 **아무 도움이 안 된다** (az 0 과 az 120 의 도달
범위가 동일하다). j1 회전은 이 예산에 들어가지 않기 때문이다.

.. note::
   이 태스크는 껍질을 직접 쓰지 않는다. 목표는 월드 +-1.2 m 상자에서 뽑히고 로봇이
   주행해서 간다. 위 rho 는 **정책이 지켜야 할 자연스러운 스탠드오프**이고, 실측
   중앙값이 팔 base 기준 rho 0.33 이라 0.44 안에 편하게 들어간다.
"""


J6_RESET = 2.60
"""j6 리셋 랜덤화 폭 [rad] = +-149 도.

실기 발산 띠(j6 -90 ~ -150 도)를 덮으면서 관절한계(+-175 도)와 26 도 여유를 남긴다.
기본자세 j6 = 0 이므로 +-2.60 이 그대로 -149 ~ +149 도가 된다.
"""


@configclass
class AmrFr3TrajV3EnvCfg(AmrFr3TrajDelayEnvCfg):
    """새 하드웨어 기준선.

    바꾸는 것 셋
    ------------
    ::

        [페이로드]   배터리 8.0 kg + 모터드라이버 0.5 kg  (총 89.87 -> 98.37 kg)
        [높이]       EE 목표 0.45~0.60 -> 0.55~0.65 m
        [j6]         손목 감김 여유를 관측에 추가

    액션 지연은 :class:`AmrFr3TrajDelayEnvCfg` 에서 그대로 물려받는다 — 그것이
    실기 발산의 근본 대응이다 (`handoff/traj_v2/README.md`).

    페이로드가 도달성을 해치지 않는다
    ---------------------------------
    배터리 윗면이 팔 마운트보다 65 mm 높아 팔이 뒤로 돌면 그 위를 지난다. 걱정했지만
    **실측 충돌률이 0.00 ~ 0.10 %** 였다 (표본 4000 x 5 후보). 작업공간을 줄일 필요가 없다.

    병목은 충돌이 아니라 **도달성**이고, 대가는 바깥 반경이다.

    j6 감김
    -------
    :func:`~...mdp.wrap_margin_named` 를 쓴다. 인덱스 하드코딩판(:func:`wrap_margin`)은
    팔 단독 에셋 전용이라 통합 에셋에서는 캐스터를 가리킨다.

    .. note::
       **관측에 정보가 있어도 정책이 그걸 쓰는 게 보상에 유리해야 배운다.**
       이것만으로 부족하면 한계 접근 벌점이나 초기 j6 랜덤화를 얹어야 한다
       (RETRAIN_V5 §2-A). 단일 변수로 하나씩 붙이는 중이다.
    """

    def __post_init__(self):
        super().__post_init__()

        # --- [페이로드] 에셋 교체 ---
        #   링크·조인트 이름은 기존과 같고 링크 2 개(battery, motor_driver)만 는다.
        #   따라서 액추에이터·초기자세 정규식이 그대로 맞는다.
        import os

        from ee_track_ppo.assets import ASSETS_DATA_DIR  # noqa: PLC0415

        usd = os.path.join(ASSETS_DATA_DIR, PAYLOAD_USD)
        if not os.path.exists(usd):
            raise FileNotFoundError("페이로드 USD 가 없다: %s" % usd)
        self.scene.robot.spawn.usd_path = usd

        # --- [높이] EE 목표를 올린다 ---
        self.commands.ee_pose.ranges.pos_z = GOAL_Z_RANGE_V3

        # --- [j6] 손목 감김 여유 ---
        self.observations.policy.wrap_margin = ObsTerm(
            func=mdp.wrap_margin_named,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["j6", "j4"])},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )

        # --- [j6] 감긴 상태에서 시작하는 **경험**을 준다 ---
        #   **관측만으로는 부족하다.** 기존 리셋은 j6 을 +-0.60 rad(+-34 도) 로만 흔든다.
        #   실기가 보고한 발산 띠는 **j6 -90 ~ -150 도** 라 학습 분포 밖이고, 정책은
        #   그 상태를 한 번도 본 적이 없다. wrap_margin 이 여유를 알려줘도 겪어본 적
        #   없는 상태에서 그 정보를 쓸 줄은 모른다 (RETRAIN_V5 §2-A(c) 가 같은 말을 한다).
        #
        #   시뮬 루프이득도 같은 띠에서 오른다 (``scripts/loop_gain_sim.py``, traj_v2)::
        #
        #       j6    0 도  ->  최대이득 0.562
        #       j6  -90 도  ->  0.956          <- 실기 발산 띠
        #       j6 -180 도  ->  1.545
        #
        #   도구가 20..30 % 과소평가하므로 -90 도의 0.956 은 실기 실측 1.28 에 대응한다.
        #   **도달성 문제가 아니라 루프이득 문제다.**
        #
        #   j6 만 따로 넓히고 j1/j4/j5 는 그대로 둔다. 리셋 자기충돌은 어깨(j2/j3)에서
        #   나므로 손목 롤을 넓히는 것은 위험이 낮지만, 재학습 후 전복률로 확인할 것.
        self.events.reset_arm_rest.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["j1", "j[4-5]"]
        )
        self.events.reset_j6 = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-J6_RESET, J6_RESET),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=["j6"]),
            },
        )


@configclass
class AmrFr3TrajV3EnvCfg_PLAY(AmrFr3TrajV3EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v4 — 파지 상태 운송 (목표가 현재 EE 에서 출발)
##


@configclass
class AmrFr3TrajV4EnvCfg(AmrFr3TrajV3EnvCfg):
    """**물체를 이미 잡고 있는** 조건으로 바꾼다.

    무엇이 문제였나 (GUI 관찰, 2026-08-05)
    --------------------------------------
    v3 까지는 목표가 월드 상자에서 무작위로 뽑혀 **멀리 순간이동**했다. 그래서
    에피소드의 상당 부분이 "멀리서 접근하는" 구간이고, 자세도 크게 달라서 팔이
    따라잡는 데 시간이 걸린다. 사용자 관찰: *"타겟 생성이 너무 멀거나 다른 자세로
    되고 움직이니까, 팔이 따라가려면 시간이 좀 걸려."*

    그 구간에서 정책은 **큰 오차를 빠르게 줄이는 공격적인 이득**을 배운다. 그 습성이
    정밀 구간까지 따라오면 떨림이 된다 — 같은 관찰에서 흔들림도 보고됐고, 시뮬
    루프이득의 원소 최대가 1 을 넘은 것과 방향이 맞는다.

    실제 시나리오는 접근 구간이 없다
    --------------------------------
    연구는 **물체를 파지한 상태로 운송**하는 것을 먼저 한다::

        시작    EE 가 이미 물체 파지점에 있다   ->  오차 0 에서 출발
        이후    물체가 움직이는 만큼 목표가 연속 이동

    ``init_at_ee=True`` 가 그 조건이다. 목표가 멀리 못 가는 것은 아니다 — waypoint 를
    향해 최대 0.3 m/s 로 계속 움직이므로 24 초에 최대 7 m 를 간다. **순간이동만 없앤다.**

    기대 효과
    ---------
    ::

        접근 구간이 사라짐   ->  정책이 추종에 특화된다
        오차가 작게 유지됨   ->  보상 기울기가 정밀 구간에 머문다
        공격적 이득의 유인   ->  줄어든다 (떨림·루프이득 개선 기대)

    .. note::
       이 변경만으로 떨림이 안 잡히면 액션 변화율 벌점을 올리는 것이 다음 수단이다.
       한 번에 하나씩 바꾼다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.ee_pose.init_at_ee = True


@configclass
class AmrFr3TrajV4EnvCfg_PLAY(AmrFr3TrajV4EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v5 — 목표 높이만 되돌린 대조군 (루프이득 원인 규명)
##


@configclass
class AmrFr3TrajV5EnvCfg(AmrFr3TrajV4EnvCfg):
    """v4 에서 **목표 높이만** 예전 값으로 되돌린다. 나머지는 전부 같다.

    왜 필요한가 — 변수 분리
    -----------------------
    시뮬 루프이득이 버전을 거치며 나빠졌다 (공정한 자세에서 전체 최대)::

        v2 (페이로드X, 목표 0.45~0.60)             0.996
        v3 (페이로드O, 목표 0.55~0.65, +j6/wrap)   1.249
        v4 (v3 + 파지조건)                          1.938

    v2 -> v3 에서 이미 나빠졌고, v3 에서 바꾼 넷 중 **목표 높이**가 가장 의심된다.

    TCP 지면 60 cm 는 도달 반경이 0.50 -> 0.44 로 줄어드는 지점이다. 팔이 작업공간
    **가장자리**에서 일하게 되고, 거기서는 야코비안이 나빠져 작은 EE 이동에 큰 관절
    이동이 필요하다. 그것이 루프이득으로 나타난다.

    ``GOAL_Z_RANGE`` 하나만 되돌리고 학습량도 v4 와 같게 맞추면, 차이가 그대로
    **높이의 대가**다.

    .. note::
       원인으로 확인되면 TCP 목표를 55 cm 쯤으로 낮추는 선택지가 생긴다
       (도달률 100 %, rho 0.55 까지). 60 cm 를 꼭 써야 한다면 루프이득을 낮추는
       다른 수단(액션 변화율 벌점 등)을 따로 써야 한다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.ee_pose.ranges.pos_z = GOAL_Z_RANGE   # (0.45, 0.60)


@configclass
class AmrFr3TrajV5EnvCfg_PLAY(AmrFr3TrajV5EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v6 — [가-1] 차체 궤적 추종 (재학습 B)
##

BASE_REF_STD = 0.15
"""차체 참조 추종 보상의 ``std`` [m]. 이 근처에서 기울기가 최대다.

목표는 차체 편차를 **실측 24 cm -> 10 cm 이하**로 줄이는 것이라 그 언저리로 잡는다.
"""

BASE_REF_WEIGHT = 0.15
"""차체 참조 추종 가중치.

**작게 시작한다.** 주 목적은 EE 추종(합산 가중치 약 1.5)이고 차체는 부차 목표다.
크게 주면 EE 가 희생된다. 1/10 정도에서 출발해 결과를 보고 조정한다.

이것이 [가-1]의 **유일한 설계 변수**다.
"""


@configclass
class AmrFr3TrajV6EnvCfg(AmrFr3TrajV5EnvCfg):
    """플래너의 **차체 궤적도 참조로 받아** 같이 추종한다 ([가-1]).

    왜 필요한가
    -----------
    EE 궤적만 따라가면 차체는 정책이 아무 데나 세운다. 플래너는 **자기가 계획한 차체
    자리**에서 SDF 여유를 하드 부등식으로 확보하는데, 우리가 다른 자리에 서면
    그 보장이 넘어오지 않는다.

    실측(``scripts/preview_plan.py``)::

        차체가 계획과 벌어진 거리   평균 0.324 m,  중앙 0.240,  95% 0.424
        -> 이걸 덮으려면 base_margin 을 0.05 -> 0.47 로 부풀려야 한다

    0.47 m 면 좁은 통로를 못 지난다. 편차를 줄이는 편이 낫다.

    덤으로 항목 6(베이스 일정 속도)이 해결된다
    ------------------------------------------
    플래너 차체 경로는 ``a_max 0.80`` / ``alpha_max 2.00`` 제한으로 이미 매끄럽다.
    **페널티로 매끄러움을 유도하는 것보다 이미 매끄러운 궤적을 따르는 것이 낫다.**

    무엇이 늘어나나 (54 -> 60)
    --------------------------
    ::

        base_ref_error   4    dx, dy (차체 프레임), sin(dtheta), cos(dtheta)   피드백
        base_ref_twist   2    v_ref, omega_ref                                피드포워드

    EE 쪽에서 검증된 구조 그대로다 — 오차만 주면 한 박자 늦고, 속도만 주면 표류한다.

    시뮬에서 참조를 어떻게 만드나
    -----------------------------
    실기에서는 플래너가 주지만 시뮬에는 없다. EE 목표에서 유도하되 **스탠드오프를
    랜덤화**한다 (거리 0.15~0.45 m, 방위 180+-60 도). 고정하면 정책이 외우고 참조를
    읽지 않는다. 자세한 것은 :meth:`MovingPoseWorldAmclCommand._update_base_ref`.

    판정
    ----
    ::

        차체 편차   중앙 24 cm -> **10 cm 이하**      <- 이게 핵심 성과 지표
        EE 정확도   25 mm 이내 (v5 는 17.2)
        루프이득     v5 와 비슷하거나 낫게
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.ee_pose.base_ref = True

        self.observations.policy.base_ref_error = ObsTerm(
            func=mdp.base_ref_error_b,
            params={"command_name": "ee_pose"},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        self.observations.policy.base_ref_twist = ObsTerm(
            func=mdp.base_ref_twist,
            params={"command_name": "ee_pose"},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        self.rewards.base_ref_tracking = RewTerm(
            func=mdp.base_ref_error_tanh,
            weight=BASE_REF_WEIGHT,
            params={"command_name": "ee_pose", "std": BASE_REF_STD},
        )


@configclass
class AmrFr3TrajV6EnvCfg_PLAY(AmrFr3TrajV6EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v7 — v2 기준선 + 페이로드 + 차체 평활 + 손목 되돌림
##

BASE_ACTION_RATE_W = -0.05
"""차체 액션 변화율 벌점.

기존 통합 ``action_rate`` 는 -0.0001 로 8 개를 한 덩어리로 본다. 그것을 올리면
팔의 정밀 추종까지 둔해지므로, **차체 두 성분만** 따로 500 배 세게 벌한다.

실기 관찰 (2026-08-06, v2): *"베이스가 과하게 움직이는 경향 빼곤 나쁘지 않다."*
팔은 문제가 없었다 — 정지유지 p-p 0.01 mm, 순회 정착오차 2.8 mm.
"""

BASE_ANG_VEL_W = -0.04
"""차체 각속도 벌점. 기존 -0.01 에서 4 배.

차동구동은 방향을 바꾸려면 제자리 회전을 해야 해서, 과한 움직임이 주로 선회로 나온다.
"""

WRIST_UNWIND_W = 0.10
WRIST_UNWIND_ERR_STD = 0.05
"""손목 되돌림 보상. EE 오차가 ``err_std`` 안일 때만 j6 를 중앙으로 당긴다.

로봇 PC 요청 (2026-08-05): *"정책이 손목을 스스로 푸는 항을 넣어 주세요.
지금은 한 번 감기면 못 풉니다."*

가중치를 작게 두는 이유는 **추종이 주 목적**이기 때문이다. 조건부라 오차가 클 때는
자동으로 0 이 되지만, 그래도 정밀 구간에서 경쟁한다.
"""


@configclass
class AmrFr3TrajV7EnvCfg(AmrFr3TrajDelayEnvCfg):
    """**v2 기준선**에 세 가지만 얹는다.

    왜 v2 로 돌아가나
    -----------------
    v2 는 이 프로젝트에서 **실기 검증을 통과한 유일한 통합 정책**이다::

        루프이득 0.561,  정지유지 p-p 0.01 mm,  0.10 m/s 순회 5/5,  정착오차 2.8 mm

    v3~v6 은 여러 변수를 한꺼번에 바꿔 원인 추적에 시간을 썼고, v6 은 차체 참조가
    액션 한계주기를 만들었다 (v5 진폭 0.000 -> v6 0.538, j6 -120 도 부근).

    그래서 검증된 기준선으로 돌아가 **실기에서 실제로 나온 문제**만 고친다.

    무엇을 얹나
    -----------
    ::

        [페이로드]   배터리 8.0 + 드라이버 0.5 kg      실물에 이미 올라가 있다
        [차체 평활]  차체 액션 변화율·각속도 벌점        "베이스가 과하게 움직인다"
        [손목]       조건부 되돌림 보상                 "한 번 감기면 못 푼다"

    무엇을 안 얹나
    --------------
    ::

        차체 참조 [가-1]   v6 의 한계주기 원인.  대책이 서기 전까지 보류
        wrap_margin 관측    효과가 확인된 적이 없다.  관측 차원을 v2 와 같게 유지
        j6 리셋 +-149 도    근거가 철회됐다 (로봇 PC, 도구 버그)
        TCP 60 cm          v2 범위 0.45~0.60 이 55 cm 를 포함한다.  변수를 줄인다

    **관측이 52 차원 그대로다.** 실기 브리지를 안 고쳐도 되고, v2 검증 자산이
    그대로 유효하다.
    """

    def __post_init__(self):
        super().__post_init__()

        # --- [페이로드] 에셋 교체 ---
        import os

        from ee_track_ppo.assets import ASSETS_DATA_DIR  # noqa: PLC0415

        usd = os.path.join(ASSETS_DATA_DIR, PAYLOAD_USD)
        if not os.path.exists(usd):
            raise FileNotFoundError("페이로드 USD 가 없다: %s" % usd)
        self.scene.robot.spawn.usd_path = usd

        # --- [차체 평활] ---
        self.rewards.base_action_rate = RewTerm(
            func=mdp.base_action_rate_l2, weight=BASE_ACTION_RATE_W
        )
        self.rewards.base_ang_vel.weight = BASE_ANG_VEL_W

        # --- [손목] 조건부 되돌림 ---
        self.rewards.wrist_unwind = RewTerm(
            func=mdp.wrist_unwind,
            weight=WRIST_UNWIND_W,
            params={
                "command_name": "ee_pose",
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["j6"], body_names=[AMR_FR3_EE_BODY]
                ),
                "err_std": WRIST_UNWIND_ERR_STD,
            },
        )


@configclass
class AmrFr3TrajV7EnvCfg_PLAY(AmrFr3TrajV7EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v8 — 실제 MPC 궤적으로 학습 (EE + 차체 참조)
##

PLAN_DATASET = "data/plans.npz"
"""``scripts/tools/gen_plan_dataset.py`` 가 만든 실제 플래너 궤적 묶음."""


@configclass
class AmrFr3TrajV8EnvCfg(AmrFr3TrajV7EnvCfg):
    """**진짜 MPC 궤적**을 학습에 쓴다.

    왜 v6 의 기하 유도가 실패했나
    -----------------------------
    v6 은 차체 참조를 EE 목표에서 기하로 만들었다 (거리 0.15~0.45 m 가정).
    실제 플래너 궤적을 150 개 뽑아 재보니::

        EE-차체 거리   중앙 0.462 m,  5% 0.339,  95% 0.551

    **가정한 범위와 거의 안 겹친다.** 그래서 정책이 배운 관계가 실제와 달랐고,
    실기 궤적 재생에서 차체 편차가 0.163 m(합성) -> 0.527 m(실제) 로 3 배가 됐다.

    무엇이 달라지나
    ---------------
    ``plan_dataset`` 을 주면 EE 목표와 차체 참조를 **둘 다 데이터에서** 읽는다::

        기존   waypoint 랜덤 직진 + 기하 유도 차체 참조
        v8     플래너가 실제로 푼 궤적 (장애물 회피가 반영된 EE-차체 관계)

    에피소드마다 궤적 하나를 뽑고 시작 시점도 흔든다 — 궤적 수가 수십 개뿐이라
    앞부분만 반복 학습하는 것을 막기 위해서다.

    한계
    ----
    시뮬에는 장애물이 없다. 궤적의 **모양**만 배우지 "왜 그렇게 돌아가는지" 는 못 배운다.
    그래도 목적은 "플래너가 주는 차체 참조를 따라가기" 이므로 이걸로 충분하다.

    v7 에서 물려받는 것
    -------------------
    페이로드, 차체 평활 벌점, 손목 되돌림, 액션 지연. 관측은 v7 의 52 차원에
    차체 참조 6 개가 붙어 **58 차원**이 된다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.ee_pose.plan_dataset = PLAN_DATASET
        self.commands.ee_pose.base_ref = True

        self.observations.policy.base_ref_error = ObsTerm(
            func=mdp.base_ref_error_b,
            params={"command_name": "ee_pose"},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        self.observations.policy.base_ref_twist = ObsTerm(
            func=mdp.base_ref_twist,
            params={"command_name": "ee_pose"},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        self.rewards.base_ref_tracking = RewTerm(
            func=mdp.base_ref_error_tanh,
            weight=0.30,
            params={"command_name": "ee_pose", "std": 0.15},
        )


@configclass
class AmrFr3TrajV8EnvCfg_PLAY(AmrFr3TrajV8EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v9 — 차체 평활 벌점 제거 + 참조 가중치 상향
##


@configclass
class AmrFr3TrajV9EnvCfg(AmrFr3TrajV8EnvCfg):
    """v8 에서 **차체 평활 벌점을 빼고** 참조 추종을 강화한다.

    v8 이 왜 실패했나
    -----------------
    실제 MPC 궤적으로 학습했는데도 차체 편차가 안 줄었고(0.527 -> 0.554 m),
    **전진/후진 반전이 초당 7.9 회**로 폭증했다 (v2 는 0.24). 차체 선속이 0.025 m/s 로
    사실상 제자리에서 떨었다.

    원인은 두 항의 충돌이다::

        차체 평활 벌점 (v7)   "액션을 바꾸지 마라"
        차체 참조 추종 (v8)   "저기로 가라"
        -> 움직이지 않는 것이 이득.  미세하게 떨며 제자리

    v7 에서도 조짐이 있었다 (반전 +29 %). v8 에서 참조 보상이 더해지며 터졌다.

    무엇을 고치나
    -------------
    **평활 벌점 자체가 잘못된 접근이었다.** 플래너 차체 경로는 ``a_max 0.80`` /
    ``alpha_max 2.00`` 제한으로 이미 매끄럽다. **잘 따라가면 평활은 공짜로 얻어진다.**
    그런데 벌점을 따로 걸어 서로 싸우게 만들었다.

    그래서::

        차체 평활 벌점   -0.05 -> 0 (제거)
        참조 추종        0.30 -> 0.60

    EE 가 9.99 mm 로 목표(30) 대비 여유가 크므로 참조 가중치를 올릴 여력이 있다.

    .. note::
       ``base_ang_vel`` 은 v7 에서 -0.01 -> -0.04 로 올려 둔 것을 **원래대로 되돌린다.**
       같은 이유다 — 참조를 따라가려면 돌아야 하는데 도는 것을 벌하면 충돌한다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.base_action_rate = None
        self.rewards.base_ang_vel.weight = -0.01
        self.rewards.base_ref_tracking.weight = 0.60


@configclass
class AmrFr3TrajV9EnvCfg_PLAY(AmrFr3TrajV9EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v10 — 차체 참조에 방위 추가 + 가중치 완화
##


@configclass
class AmrFr3TrajV10EnvCfg(AmrFr3TrajV9EnvCfg):
    """차체 **방위** 보상을 추가하고 위치 가중치를 낮춘다.

    v9 에서 드러난 것
    -----------------
    ::

        base_ref_error       0.33 -> 0.025 m     위치는 잘 따라감
        base_ref_yaw_error   1.54 ~ 1.78 rad     **전혀 안 따라감**
        orientation_error    0.166 rad           EE 자세가 v8(0.05) 대비 나빠짐

    방위를 안 따라간 이유는 단순하다 — **보상에 없었다.** 관측과 지표에는 넣었는데
    :func:`base_ref_error_tanh` 는 위치 거리만 본다. 평균 |오차| 1.6 rad 은
    균등랜덤 yaw 의 기대값 pi/2 = 1.57 과 거의 같아, 리셋 랜덤값 그대로였다.

    그리고 위치 가중치 0.60 이 **EE 자세를 희생시켰다.** EE 위치는 여유가 있었지만
    자세가 0.05 -> 0.166 rad 로 3 배가 됐다.

    무엇을 바꾸나
    -------------
    ::

        차체 위치 추종   0.60 -> 0.35      나머지가 희생되지 않게
        차체 방위 추종   없음 -> 0.20      새로 추가.  위치보다 작게

    방위를 작게 두는 이유는 **우리 팔이 뒤를 보기 때문**이다. 플래너는 EE 가 앞에
    있다고 가정하므로(``ee_forward_min``), 참조 방위를 그대로 따르면 팔이 불리해질 수
    있다. 작게 걸고 결과를 본다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.base_ref_tracking.weight = 0.35
        self.rewards.base_ref_yaw = RewTerm(
            func=mdp.base_ref_yaw_tanh,
            weight=0.20,
            params={"command_name": "ee_pose", "std": 0.5},
        )


@configclass
class AmrFr3TrajV10EnvCfg_PLAY(AmrFr3TrajV10EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v11 — j1 리셋을 전 범위로 (앞쪽 배치를 학습에서 겪게)
##

J1_RESET = 2.967
"""j1 리셋 랜덤화 폭 [rad] = +-170 도. 관절한계 +-175 와 5 도 여유.

왜 넓히나
---------
데이터셋(실제 MPC 궤적)은 EE 를 **차체 전방 +0.407 m** 에 둔다 (전방 비율 95.1 %).
그런데 리셋은 팔을 항상 기본자세(뒤쪽) 근처에서 시작시킨다 — j1 이 +-34 도뿐이었다.
**목표는 앞인데 시작은 늘 뒤**라, 정책이 앞쪽으로 크게 도는 것을 충분히 겪지 못했다.

도달 자체는 문제가 없다. 시드를 여러 개 준 IK 스윕에서 **방위각과 무관하게 동일**했다::

    TCP 지면 0.50~0.60   az 0/45/90/135/180/-135/-90/-45  전부 100 %
    az 180 (앞) 에서 j1 = +163 도 — 한계 +-175 안

.. note::
   초판에서 "앞쪽 도달률 37 %" 라고 적고 팔 180 도 재장착까지 검토했는데,
   그것은 **시드 하나짜리 IK 의 허상**이었다. j1 은 수직축 회전이라 작업공간이
   회전 대칭이다. 재장착은 필요 없다.

   다만 az -160 도 부근에 좁은 사각지대가 있다 (j1 이 +-175 를 넘어야 하는 구간).
"""


@configclass
class AmrFr3TrajV11EnvCfg(AmrFr3TrajV10EnvCfg):
    """j1 리셋을 **+-170 도**로 넓힌다. 나머지는 v10 그대로.

    v10 까지의 재생 결과 (도구 수정 후, 실제 MPC 계획 재생)::

        v6 (기하 유도)      EE   9.14 mm,  자세  3.72 도,  차체 편차 0.508 m
        v8 (MPC 데이터셋)   EE 118.20 mm,  자세 23.51 도,  차체 편차 0.365 m
        v9 (가중치 0.60)    EE 186.24 mm,  자세 108.0 도,  차체 편차 0.459 m

    데이터셋을 쓰면 EE 가 13 배 나빠진다. 기구학적 불가능이 아니라 **학습 분포 문제**로
    본다 — 목표는 앞쪽인데 시작 자세가 늘 뒤쪽이었다.
    """

    def __post_init__(self):
        super().__post_init__()
        # j1 만 따로 넓힌다. j4/j5 는 그대로 (+-0.60).
        self.events.reset_arm_rest.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["j[4-5]"]
        )
        self.events.reset_j1 = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-J1_RESET, J1_RESET),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=["j1"]),
            },
        )


@configclass
class AmrFr3TrajV11EnvCfg_PLAY(AmrFr3TrajV11EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v12 — 실측 통계를 맞춘 **생성 궤적**으로 학습, 판정은 실제 궤적으로
##

PLAN_SYNTH = "data/plans_synth.npz"
"""``scripts/tools/gen_plan_synth.py`` 가 만든 궤적 4000 개 x 321 점 (48 초).

실제 플래너 궤적 122 개의 통계(속력·곡률·EE 배치)를 맞추되, **덜 모인 구간을
일부러 넓혔다**::

                        생성      실측(122개)
    EE 가 차체 뒤       20.7 %     4.9 %
    |요차이| > 60 도    25.9 %    13.1 %
    속력 > 0.15 m/s     38.8 %     1.6 %

넓힌 이유는 플래너가 그런 궤적을 **만들 수 있는데** 우리 수집이 놓쳤기 때문이다.
``plan_real.json`` 이 바로 그런 경우다 (EE 가 뒤 -0.115 m, 요차이 +82 도).

길이가 48 초인 이유
-------------------
에피소드가 24 초이고 시작 시점을 궤적 길이의 절반까지 흔든다. 궤적이 짧으면
에피소드 후반이 **끝점에 멈춘 목표**가 되어 추종이 아니라 정지유지를 학습한다.
``dur >= 2 x episode`` 여야 그 구간이 안 생긴다.
"""


@configclass
class AmrFr3TrajV12EnvCfg(AmrFr3TrajV11EnvCfg):
    """v11 과 보상·관측은 같고, **학습 궤적만 생성본**으로 바꾼다.

    왜 바꾸나
    ---------
    실제 궤적 122 개로 학습한 v8/v9/v11 은 전부 실제 계획 재생에서 무너졌다::

        v6 (기하)  10.8 mm     v8 118 mm     v9 186 mm     v11 217 mm

    개수가 적어 **외웠다**. 생성기는 무제한이라 그 실패 방식이 없다.

    판정 방법
    ---------
    학습에 생성 궤적만 쓰므로 ``data/plans.npz`` 의 **실제 122 개 전부가
    held-out** 이다. ``scripts/eval_plans.py`` 로 잰다::

        python scripts/eval_plans.py --load_run <run> --task Reach-AMR-FR3-TrajV12-Play-v0 \\
               --test_n 24 --json <plan_real.json>

    .. note::
       생성기가 자기 자신을 검증하지 않게 하는 것이 요점이다. 학습은 생성본,
       판정은 실측본으로 나눠야 "통계를 맞춘 것"이 실제로 통했는지 알 수 있다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.ee_pose.plan_dataset = PLAN_SYNTH
        # 생성기가 이미 속도 범위를 덮으므로 배속 랜덤화는 끈다 (두 번 흔들 이유가 없다).
        self.commands.ee_pose.plan_time_scale = (1.0, 1.0)
        # 전부 데이터셋을 쓴다. 기하 합성 경로는 EE-차체 관계가 실제와 달라
        # (거리 0.15~0.45, 방위 뒤쪽) 이제 섞을 이유가 없다.
        self.commands.ee_pose.plan_mix = 1.0
        # 생성본은 학습 전용이라 따로 뺄 것이 없다 — 실측본 전체가 시험용이다.
        self.commands.ee_pose.plan_test_n = 0


@configclass
class AmrFr3TrajV12EnvCfg_PLAY(AmrFr3TrajV12EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v13 — v12 에 **높이 랜덤화**를 넣는다
##

PLAN_GOAL_Z_RANGE_V13 = (0.45, 0.65)
"""데이터셋 재생 시 EE 목표 높이 범위 [m].

v12 는 0.55 한 점만 봐서 5 cm 벗어나면 오차가 5 배가 됐다 (13.9 -> 72.6 mm).
v6 는 0.45~0.60 을 학습해서 높이에 강했다. 그 성질을 되찾는다.
"""


@configclass
class AmrFr3TrajV13EnvCfg(AmrFr3TrajV12EnvCfg):
    """v12 + 목표 높이 랜덤화. 나머지는 그대로.

    v12 실측 (실제 플래너 궤적 25 개, 높이 0.55)::

        EE 13.9 mm   자세 10.5 도   차체 편차 0.240 m   3cm 통과 84 %

    비교 대상 v6 (같은 시험대, 같은 높이)::

        EE 10.9 mm   자세  3.3 도   차체 편차 0.488 m   3cm 통과 88 %

    노리는 것은 **v12 의 차체 추종 + v6 의 강건함**이다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.ee_pose.plan_goal_z_range = PLAN_GOAL_Z_RANGE_V13


@configclass
class AmrFr3TrajV13EnvCfg_PLAY(AmrFr3TrajV13EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v14 — 실기 v12 결과에 대한 답
##

PLAN_SEEDED = "data/plans_seeded.npz"
"""``scripts/tools/gen_plan_seeded.py`` — 실측 (v, omega) 를 이어붙여 만든 궤적.

첫 생성기는 주변분포만 맞췄고, 실기에서 **속도가 출렁인다**는 지적을 받았다.
재보니 크기는 맞는데 부드럽기가 달랐다::

                 |a| 95%    |j| 95%
    실측 MPC      0.0357     0.014
    OU 생성       0.0544     0.257     <- 저크 18 배
    씨앗 생성     0.0331     0.028     <- 가속은 실측과 일치
"""

PLAN_GOAL_Z_RANGE_V14 = (0.50, 0.60)
"""EE 목표 높이 [m]. 실기 기준이 0.55 로 정해졌고, 그 언저리만 흔든다.

v12 는 0.55 한 점만 봐서 0.60 에서 오차가 5 배였다. 좁게라도 흔들어 두면
TCP 캘리브레이션 오차나 물체 두께 변화를 흡수한다.
"""

BASE_REF_W_V14 = 0.50
"""차체 위치 추종 가중치. v12 의 0.35 에서 올린다 — 실기에서 "추종 능력 부족"."""

BASE_REF_TWIST_W = 0.25
BASE_REF_TWIST_V_STD = 0.05
BASE_REF_TWIST_W_STD = 0.15
"""차체 **속도** 추종. v12 에 없던 항이다.

기존 보상은 위치와 방위만 봐서, 정책이 "결국 그 자리에 있기만 하면 된다" 를 배웠다.
가는 방식이 제멋대로가 되어 실기에서 빨랐다 느렸다 했다. 참조 속도를 직접 따라가면
참조가 MPC 의 매끄러운 프로파일이므로 **평활과 추종이 같이 해결된다.**
"""

BASE_ACTION_RATE_W_V14 = -0.03
"""차체 액션 변화율 감쇠. **v9 에서 껐던 것을 되살린다.**

v9 가 끈 이유는 차체 추종을 방해한다는 것이었는데, 그때는 속도 추종 항이 없어서
감쇠가 유일한 평활 수단이었다. 이제 속도 추종이 있으므로 감쇠는 약하게만 둔다.
"""


@configclass
class AmrFr3TrajV14EnvCfg(AmrFr3TrajV12EnvCfg):
    """실기 v12 의 세 가지 지적에 각각 대응한다.

    실기 보고 (2026-08-06, v12)::

        EE 좌표 추종        매우 우수      <- 건드리지 않는다
        1. 차체 속력 출렁임             -> 씨앗 생성기(저크 18배 -> 2배) + 속도 추종 + 액션 감쇠
        2. 차체 참조 추종 부족          -> 가중치 0.35 -> 0.50 + 속도 추종 항
        3. EE 자세 오차                 -> 생성 요차이를 실측 범위로 되돌림 (+-90 -> +-65 도)

    .. note::
       EE 위치 추종이 우수하다는 것은 **건드릴 이유가 없다**는 뜻이다. 위치 관련
       보상과 ``end_effector_orientation_tracking_precise`` 는 그대로 둔다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.ee_pose.plan_dataset = PLAN_SEEDED
        self.commands.ee_pose.plan_goal_z_range = PLAN_GOAL_Z_RANGE_V14
        self.rewards.base_ref_tracking.weight = BASE_REF_W_V14
        self.rewards.base_ref_twist = RewTerm(
            func=mdp.base_ref_twist_tanh,
            weight=BASE_REF_TWIST_W,
            params={
                "command_name": "ee_pose",
                "v_std": BASE_REF_TWIST_V_STD,
                "w_std": BASE_REF_TWIST_W_STD,
            },
        )
        self.rewards.base_action_rate = RewTerm(
            func=mdp.base_action_rate_l2, weight=BASE_ACTION_RATE_W_V14
        )


@configclass
class AmrFr3TrajV14EnvCfg_PLAY(AmrFr3TrajV14EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v15 — v14 와 보상은 같고, **데이터가 고쳐졌다**
##


@configclass
class AmrFr3TrajV15EnvCfg(AmrFr3TrajV14EnvCfg):
    """v14 를 버리고 다시 — 궤적 생성기의 결함을 고친 판.

    무엇이 잘못됐었나
    -----------------
    v14 데이터는 **차체 프레임에서 EE 가 거의 안 움직였다**::

                      상대속력 중앙   상대요 95%
        실측 MPC       0.0577         0.0644
        v14 데이터     0.0028         0.0218      <- 20 배 작음

    원인은 변동을 빚던 필터였다. 1 차 필터를 두 번 걸면서 진폭이 **1/67 로** 죽었다
    (25 도를 요청했는데 실제 0.37 도). 그래서 정책은 "팔을 거의 안 쓰고 차체만 따라가는
    법" 을 배웠고, 학습 중 자세오차가 56 -> 78 도로 **거꾸로 갔다.**

    무엇을 고쳤나
    -------------
    상대운동도 **실측을 이어붙인다** (차체 (v, omega) 와 같은 방식). 수준과 변동을
    나눠, 변동만 이어붙이고 수준은 궤적당 하나 뽑는다 — 값 그대로 이으면 경계에서
    크게 튀고(상대요 7 배), 증분으로 이으면 흘러내린다(거리 0.455 -> 0.216).

    고친 뒤::

        씨앗 생성      상대속력 0.0258/0.1259   상대요 95% 0.0606   거리 0.433

    자세오차에 대해 확인한 것
    -------------------------
    ``scripts/diag_orientation.py`` 로 v12 를 스텝 단위로 캐서 가설 넷을 **전부 반증**했다::

        도달 껍질 경계    상관 +0.05    아니다
        목표 측면 성분    상관 -0.26    반대다 (측면일수록 오히려 좋다)
        목표 속력         상관 +0.09    약하다
        손목 한계 근접    99.6 % 가 한계에서 멀다.  아니다

    남은 것은 **차체-참조 거리 (상관 +0.507)** 하나다. 차체가 참조에서 벗어날 때
    자세가 무너진다 — v14 의 차체 추종 강화(가중치 0.50 + 속도 추종)가 맞는 방향인데,
    데이터가 나빠서 확인을 못 했다.
    """


@configclass
class AmrFr3TrajV15EnvCfg_PLAY(AmrFr3TrajV15EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v16 — 차체 추종에 **불감대**를 준다
##

BASE_BAND_M = 0.15
BASE_BAND_RAD = 0.35
"""차체가 참조에서 벗어나도 벌하지 않는 범위 [m], [rad] (약 20 도).

왜 이 값인가
------------
v6(자유) 0.488 m, v12(가중치 0.35) 0.240 m, v15(꽉 붙듦) 0.062 m 였고 자세오차는
각각 3.3 / 10.5 / 53.4 도였다. **0.15 m 는 v15 보다는 자유롭고 v12 보다는 좁다** —
자세를 만들 여지는 주되 플래너 여유는 지키는 지점을 노린다.

플래너 ``base_margin`` 은 이 밴드만큼만 키우면 된다 (0.05 + 0.15 = 0.20 m).
v6 은 0.71 m 가 필요했다.
"""


@configclass
class AmrFr3TrajV16EnvCfg(AmrFr3TrajV15EnvCfg):
    """차체 추종을 **불감대 + 한계**로 바꾼다. 데이터·나머지 보상은 v15 그대로.

    v15 가 밝힌 것::

        차체를 참조에 꽉 붙들면 (편차 0.062 m) 자세오차가 53.4 도로 무너진다.
        차체가 벗어나는 것은 **자세를 맞추려는 시도**였다.

    그래서 "정확히 따라가라" 를 "이 반경 밖으로 나가지 마라" 로 바꾼다. 플래너가
    보장하는 것도 정확한 위치가 아니라 **장애물 여유**이므로, 밴드가 문제의 본래
    형태에 더 맞다.

    속도 추종(``base_ref_twist``)은 **그대로 둔다.** 그것은 위치를 구속하지 않고
    출렁임만 잡는 항이라, v15 에서 실기 지적 1 번을 해결한 공로가 있다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.base_ref_tracking = RewTerm(
            func=mdp.base_ref_error_band,
            weight=BASE_REF_W_V14,
            params={"command_name": "ee_pose", "band": BASE_BAND_M, "std": 0.25},
        )
        self.rewards.base_ref_yaw = RewTerm(
            func=mdp.base_ref_yaw_band,
            weight=0.20,
            params={"command_name": "ee_pose", "band": BASE_BAND_RAD, "std": 0.5},
        )


@configclass
class AmrFr3TrajV16EnvCfg_PLAY(AmrFr3TrajV16EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v17 — 자세 사다리의 빠진 칸 + 밴드 안에서도 약한 추종
##

ORI_MID_STD = 0.12
ORI_MID_W = 0.15
"""자세 사다리에 **빠져 있던 칸** (std, weight).

왜 필요한가
-----------
자세 사다리가 ``std 0.3 -> std 0.05`` 로 건너뛴다. 지금 막혀 있는 0.175 rad 에서::

    std 0.3    e/std = 0.58    기울기가 절반으로 약해진 구간
    std 0.05   e/std = 3.5     완전 포화 — 신호가 없다

**0.1~0.2 rad 구간에 기울기를 주는 칸이 없다.** 정확히 정체된 지점이다.

같은 진단을 위치 사다리에서 이미 한 번 했고 (``mobile_reach_env_cfg`` 의
``ee_position_precise`` 주석), 칸을 더해서 뚫었다. tanh 사다리는 **칸을 더하지
빼지 않는다** — 기존 칸은 그대로 둔다.
"""

BASE_SOFT_W = 0.15
BASE_SOFT_STD = 0.30
"""밴드 **안에서도** 약하게 당기는 항 (weight, std).

밴드만 있으면 안쪽이 평평해서 정책이 밴드 가장자리에 머물러도 손해가 없다
(v16 실측 편차 0.189 m, 밴드 0.15 바로 밖). 약한 전 구간 항을 더해 안쪽에서도
조금씩 당기되, 자세를 만들 자유는 남긴다.

.. note::
   v15 처럼 **강하게** 당기면 자세가 무너진다 (편차 0.062 m, 자세 53 도).
   가중치를 밴드 항(0.50)의 1/3 로 두는 이유다.
"""


@configclass
class AmrFr3TrajV17EnvCfg(AmrFr3TrajV16EnvCfg):
    """v16 에 두 가지를 더한다. 데이터는 그대로.

    실제 궤적 24 개 평가 (높이 0.55)::

        v6   EE 10.9 mm  자세  3.3 도  차체 0.488 m   88 % 통과
        v12  EE 13.9     자세 10.5     차체 0.240     84 %
        v16  EE 37.3     자세 23.1     차체 0.189     33 %
        v15  EE 38.8     자세 53.4     차체 0.062     29 %

    차체를 조일수록 자세가 무너지는 것이 일관되게 보인다. v17 은 **자세 쪽에
    기울기를 주는 칸을 더해** 그 맞바꿈의 위치 자체를 옮기려는 시도다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.ee_orientation_mid = RewTerm(
            func=mdp.orientation_command_error_tanh_w,
            weight=ORI_MID_W,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[AMR_FR3_EE_BODY]),
                "std": ORI_MID_STD,
                "command_name": "ee_pose",
            },
        )
        self.rewards.base_ref_soft = RewTerm(
            func=mdp.base_ref_error_tanh,
            weight=BASE_SOFT_W,
            params={"command_name": "ee_pose", "std": BASE_SOFT_STD},
        )


@configclass
class AmrFr3TrajV17EnvCfg_PLAY(AmrFr3TrajV17EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v18 — j6 가 한계에 박히는 것을 막는다
##

WRIST_UNWIND_W_V18 = 0.25
"""손목 감김 억제 가중치. v7 이후 0.10 을 **2.5 배로** 올린다.

왜 필요한가
-----------
v17 진단 (실제 궤적 24 개, ``scripts/diag_orientation.py``)::

            |j6| 중앙   |j6| 95%    손목여유 < 0.2 인 표본
    v12      17.5 도     71.5 도      0.4 %
    v17      27.3       **175.0**    14.5 %          <- 한계가 +-175 도

    손목여유 0.10~0.20 구간의 자세오차 평균 **79.2 도**
    손목여유 0.20 초과 구간은                 8.5 도

v17 은 요를 잘 맞추게 되면서(자세 중앙 3.0 도) **j6 를 계속 같은 방향으로 감는다.**
그러다 한계에 박히면 자세가 통째로 무너진다 — 롤·요 오차 최대가 179 도로, 손목이
못 풀려 뒤집힌 모습이다.

24 개 중 17 개는 자세 10 도 이하인데 나머지가 32~113 도인 것이 이 구조다.
실기에서 "잘 맞출 때는 잘 맞추는데 크게 틀어진다" 고 하신 것과 같다.
"""


@configclass
class AmrFr3TrajV18EnvCfg(AmrFr3TrajV17EnvCfg):
    """v17 + **손목 여유를 보고, 감김을 더 억제한다**. 관측이 58 -> 60 이 된다.

    .. warning::
       **관측 차원이 바뀐다.** 실기 브리지의 ``ee_obs.py`` 를 60 차원판으로 갈아야
       한다. 배치는 v6 과 같다 (``wrap_margin`` 이 [52:54], base_ref 가 [54:58]).

    v3 에서 넣었다가 v7 에서 뺐던 관측이다. 뺀 이유는 "효과가 확인된 적이 없다" 였고
    그때는 맞았다 — v12 는 j6 가 한계 근처에 간 표본이 0.4 % 뿐이라 볼 일이 없었다.
    v17 에서 처음으로 **14.5 %** 가 되면서 필요해졌다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.wrap_margin = ObsTerm(
            func=mdp.wrap_margin_named,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["j6", "j4"])},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        self.rewards.wrist_unwind.weight = WRIST_UNWIND_W_V18


@configclass
class AmrFr3TrajV18EnvCfg_PLAY(AmrFr3TrajV18EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v19 — 실기 v17 결과에 대한 네 가지 대응
##

BASE_MAX_LIN_V19 = 0.2
BASE_MAX_ANG_V19 = 0.5
BASE_MAX_REV_V19 = 0.1
"""차체 행동 공간 [m/s], [rad/s]. v2 이후 0.5 / 1.0 이던 것을 좁힌다.

왜 좁히나
---------
행동은 **배율**이다 (``v = action * max_lin_vel``). 그런데 실제로 내야 하는 값은::

    참조 차체 속력   중앙 0.063 m/s   95% 0.114      (범위 0.5 의 12 %)
    참조 각속도      95% 0.058 rad/s                 (범위 1.0 의 6 %)

정책이 출력 범위의 6~12 % 지점에서 미세조정을 해야 한다. 신경망 출력이 그 좁은
구간에서 매끄럽지 않아 **지그재그**로 나타난다. 범위를 좁히면 같은 출력 정밀도로
2~2.5 배 곱게 낼 수 있다.

.. warning::
   **실기 브리지의 상수와 반드시 같아야 한다.** ``policy_runner_traj.py`` 의
   ``MAX_LIN_VEL`` / ``MAX_ANG_VEL`` 이 학습값과 다르면 그 비만큼 **게인이 어긋난다.**
   실기에서 ``-p max_lin_vel:=0.15`` 로 돌리고 있었다면 학습(0.5) 대비 3.3 배
   어긋난 셈이고, 되먹임으로 보상하려다 진동이 난다.
"""

ARM_ACTION_RATE_V19 = -0.03
"""팔 액션 변화율 벌점. 커리큘럼 최종값 -0.01 에서 3 배로 올린다.

실기 v17: **손이 떨린다.** 루프이득이 중앙 1.051 / 최대 2.089 / 1 초과 60 % 로
v12(0.953 / 40 %) 보다 나쁘다. 액션 변화율 벌점은 루프이득을 직접 누르는 항이다.

.. note::
   너무 크면 추종이 굼떠진다. v2 가 실기 루프이득 0.561 로 안정적이었으니 그 방향으로
   3 배만 올려 본다.
"""

BASE_YAW_SOFT_W = 0.10
BASE_YAW_SOFT_STD = 0.40
"""차체 **방위**를 밴드 안에서도 약하게 당기는 항.

v17 은 방위에 +-20 도 불감대만 있어 그 안에서는 만점이라 기울기가 없었다. 차체가
밴드 안을 자유롭게 흔들 수 있었고 그것이 **지그재그**의 한 축이다. 위치에 넣은
``base_ref_soft`` 와 같은 구조를 방위에도 넣는다.
"""

BASE_BAND_M_V19 = 0.12
BASE_SOFT_W_V19 = 0.20
"""차체 위치 불감대와 약한 추종 항. v17 의 0.15 / 0.15 에서 조인다.

v17 실측 편차 0.126 m 로 밴드(0.15) 안쪽에 있었다. 조금 더 붙이려면 밴드를 줄이고
안쪽 기울기를 키우는 두 가지를 같이 해야 한다. **v15 처럼 세게 하면 자세가 무너지므로**
(편차 0.062 m, 자세 53 도) 한 번에 조금씩만 옮긴다.
"""


@configclass
class AmrFr3TrajV19EnvCfg(AmrFr3TrajV17EnvCfg):
    """실기 v17 의 네 가지 요구.

    ::

        1. 90 도 문제        PLAN_YAW_OFFSET 로 해결 (commands.py).  데이터·평가 전부 반영
        2. 지그재그          차체 행동공간 축소 + 방위 밴드 안쪽 기울기
        3. 손 떨림           팔 액션 변화율 벌점 3 배
        4. 차체 위치 추종     밴드 0.15 -> 0.12,  약한 항 0.15 -> 0.20

    .. warning::
       **배포 시 브리지 상수를 같이 바꿔야 한다.** ``MAX_LIN_VEL 0.5 -> 0.2``,
       ``MAX_ANG_VEL 1.0 -> 0.5``, 그리고 목표 요에 **-90 도**. 셋 중 하나라도
       빠지면 에러 없이 어긋난다.
    """

    def __post_init__(self):
        super().__post_init__()
        # 2. 지그재그 — 행동 공간
        self.actions.base_action.max_lin_vel = BASE_MAX_LIN_V19
        self.actions.base_action.max_ang_vel = BASE_MAX_ANG_V19
        self.actions.base_action.max_reverse_vel = BASE_MAX_REV_V19
        # 2. 지그재그 — 방위 밴드 안쪽에도 기울기
        self.rewards.base_ref_yaw_soft = RewTerm(
            func=mdp.base_ref_yaw_tanh,
            weight=BASE_YAW_SOFT_W,
            params={"command_name": "ee_pose", "std": BASE_YAW_SOFT_STD},
        )
        # 3. 손 떨림 — 액션 변화율
        self.rewards.action_rate.weight = ARM_ACTION_RATE_V19
        self.curriculum.action_rate.params["weight"] = ARM_ACTION_RATE_V19
        # 4. 차체 위치 추종
        self.rewards.base_ref_tracking.params["band"] = BASE_BAND_M_V19
        self.rewards.base_ref_soft.weight = BASE_SOFT_W_V19


@configclass
class AmrFr3TrajV19EnvCfg_PLAY(AmrFr3TrajV19EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v20 — v19 에서 팔을 얼려버린 액션 벌점을 되돌린다
##

ARM_ACTION_RATE_V20 = -0.015
"""팔 액션 변화율 벌점. v19 의 -0.03 이 **팔을 멈춰 세웠다.**

v19 (700 it) 와 v17 (700 it) 보상 내역::

                          v19        v17
    ee_position_coarse    0.2619     0.3812
    ee_position_fine      0.0329     0.2486      <- 7.6 배 차이
    action_rate          -0.0336    -0.0108

움직여 위치를 맞추는 이득보다 움직임 벌점이 커져, **가만히 있는 것이 총점이 높아졌다.**
그리고 90 도 보정 뒤에는 목표 자세가 기본자세와 같은 방향이라 **안 움직여도 자세 점수가
공짜로** 나온다 (``ee_orientation_fine`` 0.127). 정책은 얼어붙는 쪽을 골랐다.

기본자세 tool0 은 차체 기준 (-0.137, -0.102, 0.539) — **모터드라이버 바로 위**다.
그래서 GUI 에서 "손끝을 모터드라이버에 박고 멈춘" 것으로 보였다.

.. warning::
   **항 하나를 올릴 때는 경쟁하는 항과의 균형을 같이 봐야 한다.** v18 에서
   ``wrist_unwind`` 0.10 -> 0.25 로 올려 자세 보상을 눌러버린 것과 같은 실수를
   v19 에서 반복했다. 학습 200 it 쯤에서 ``ee_position_fine`` 이 v17 수준(0.2 이상)에
   오는지 반드시 확인할 것.
"""


@configclass
class AmrFr3TrajV20EnvCfg(AmrFr3TrajV19EnvCfg):
    """v19 에서 ``action_rate`` 만 되돌린다. 90 도 보정과 나머지 세 변경은 유지.

    유지하는 것::

        90 도 규약      검증 통과 (막대 0 도 == 기본자세, 회전 차이 0.000 도)
                        IK 로도 보정 전후 동등 (성공률 93.1 -> 94.4 %)
        행동공간 축소    0.2 / 0.5.   포화 지표로 병목 여부를 사후 확인
        차체 방위 기울기  밴드 안쪽에도 약한 항
        차체 위치        밴드 0.12, 약한 항 0.20
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate.weight = ARM_ACTION_RATE_V20
        self.curriculum.action_rate.params["weight"] = ARM_ACTION_RATE_V20


@configclass
class AmrFr3TrajV20EnvCfg_PLAY(AmrFr3TrajV20EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v21 — 도달 보정을 빼고, 액션 벌점을 v17 로 되돌린다
##


@configclass
class AmrFr3TrajV21EnvCfg(AmrFr3TrajV19EnvCfg):
    """v19/v20 에서 팔이 주기적으로 무너진 두 원인을 뺀다.

    무엇이 원인이었나
    -----------------
    **[1] 도달 보정** — 못 닿는 목표를 "가장 가까운 닿는 점" 으로 당겼는데, 그 자리가
    IK 경계라 연속된 두 시점의 해가 **다른 분기로 튀었다**::

        한 스텝 관절변화 최대   보정 있음 303.5 도   ->   보정 없음 24.0 도

    v17 까지는 토막이 짧아 그 자리를 스쳐 지나갔고, v19 에서 토막을 6.8~9 초로 늘리며
    오래 머물러 드러났다. GUI 관찰 "따라가다 갑자기 끼듯이 무너진다" 가 이것이다.

    **[2] 액션 변화율 벌점** — v19 의 -0.03 은 팔을 얼려버렸다 (``ee_position_fine``
    0.033 vs v17 의 0.249). v20 의 -0.015 도 마찬가지였다. v17 의 -0.01 로 되돌린다.

    부호(+-90 도) 는 원인이 아니었다
    --------------------------------
    여러 시드로 풀어 |j6| 최소 해를 고르면 세 규약의 손목 부담이 거의 같다
    (필요 |j6| 중앙 48.6 / 49.6 / 42.2 도). 워밍스타트 IK 로 재던 초판 측정이
    분기 운에 흔들려 두 번 반대 결론을 냈던 것이다.

    유지하는 것
    -----------
    ::

        90 도 규약        PLAN_YAW_OFFSET = -pi/2.  실물 막대 정렬 (실기 관찰의 근본 수정)
        행동공간 축소      0.2 / 0.5.  포화 지표로 사후 확인
        차체 방위 기울기   밴드 안쪽에도 약한 항
        차체 위치         밴드 0.12, 약한 항 0.20
    """

    def __post_init__(self):
        super().__post_init__()
        # v17 수준으로 되돌린다. 손 떨림은 다른 항으로 접근할 것.
        self.rewards.action_rate.weight = -0.01
        self.curriculum.action_rate.params["weight"] = -0.01


@configclass
class AmrFr3TrajV21EnvCfg_PLAY(AmrFr3TrajV21EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v22 — 원인 분리: v17 에 **90 도 규약만** 더한다
##


@configclass
class AmrFr3TrajV22EnvCfg(AmrFr3TrajV17EnvCfg):
    """v17 과 설정이 **완전히 같다.** 달라지는 것은 ``PLAN_YAW_OFFSET`` 뿐이다.

    왜 이렇게 하나
    --------------
    v19~v21 에서 EE 위치가 400 mm 근처에 멈춰 수렴하지 않았다. 한 판에 여러 개를
    바꿔서 **원인을 못 갈랐다**::

        [a] 90 도 규약          목표 자세가 전부 90 도 회전
        [b] 행동공간 0.5 -> 0.2  차체 여유가 줄어 EE 목표를 못 쫓아가나
        [c] 새 데이터            토막 45~60 (v17 은 12~40)

    v21 에서 도달 보정과 액션 벌점을 되돌려도 그대로였으므로 그 둘은 원인이 아니다.
    남은 셋을 **하나씩** 지운다. 이 판은 [a] 만 켠 것이다.

    판정
    ----
    ::

        it 200 에서 EE 가 v17 수준(111 mm, fine 0.097) 이면   -> [a] 무죄
        400 mm 근처에 머물면                                   -> 90 도 규약이 원인

    데이터도 v17 과 같게 다시 만들었다 (SEG 12~40, 도달 보정 켬).
    """


@configclass
class AmrFr3TrajV22EnvCfg_PLAY(AmrFr3TrajV22EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v23 — 원인 분리 [b]: v22 에 **행동공간 축소만** 더한다
##


@configclass
class AmrFr3TrajV23EnvCfg(AmrFr3TrajV22EnvCfg):
    """v22 (= v17 + 90 도) 에 차체 행동공간 축소만 더한다.

    검정하려는 것
    -------------
    v19~v21 이 EE 400 mm 에서 막힌 원인 후보 [b]. 행동은 **배율**이라
    ``v = action * max_lin_vel`` 이고, 상한을 좁히면 해상도는 좋아지지만 **여유가 준다**::

        참조 차체 속력 95 %   0.128 m/s
        상한 0.2 -> 여유 1.6 배     상한 0.5 -> 여유 3.9 배

    지그재그를 줄이려고 넣은 변경인데, 차체가 EE 목표를 잡을 자리로 갈 여력을
    뺏었을 수 있다. v19/v21 모두 **차체 편차는 좋은데(0.086~0.117 m) EE 만 막혔다** —
    "참조는 따라가지만 EE 를 잡을 위치로는 못 간다" 와 맞는 모습이다.

    판정
    ----
    ::

        v22 처럼 수렴하면   -> [b] 무죄.  남은 것은 [c] 새 데이터
        400 mm 에서 막히면  -> 행동공간 축소가 원인.  지그재그는 다른 방법으로
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.base_action.max_lin_vel = BASE_MAX_LIN_V19
        self.actions.base_action.max_ang_vel = BASE_MAX_ANG_V19
        self.actions.base_action.max_reverse_vel = BASE_MAX_REV_V19


@configclass
class AmrFr3TrajV23EnvCfg_PLAY(AmrFr3TrajV23EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v23 — 원인 분리 [b]: v22 에 **행동공간 축소만** 더한다
##


@configclass
class AmrFr3TrajV23EnvCfg(AmrFr3TrajV22EnvCfg):
    """v22 (= v17 + 90 도) 에 차체 행동공간 축소만 더한다.

    검정하려는 것
    -------------
    v19~v21 이 EE 400 mm 에서 막힌 원인 후보 [b]. 행동은 **배율**이라
    ``v = action * max_lin_vel`` 이고, 상한을 좁히면 해상도는 좋아지지만 여유가 준다::

        참조 차체 속력 95 %   0.128 m/s
        상한 0.2 -> 여유 1.6 배      상한 0.5 -> 여유 3.9 배

    지그재그를 줄이려고 넣은 변경인데, 차체가 EE 목표를 잡을 자리로 갈 여력을
    뺏었을 수 있다. v19/v21 모두 **차체 편차는 좋은데(0.086~0.117 m) EE 만 막혔다** —
    "참조는 따라가지만 EE 를 잡을 위치로는 못 간다" 와 맞는 모습이다.

    판정
    ----
    ::

        v22 처럼 수렴하면   -> [b] 무죄.  남은 것은 [c] 새 데이터 (토막 45~60)
        400 mm 에서 막히면  -> 행동공간 축소가 원인.  지그재그는 다른 방법으로
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.base_action.max_lin_vel = BASE_MAX_LIN_V19
        self.actions.base_action.max_ang_vel = BASE_MAX_ANG_V19
        self.actions.base_action.max_reverse_vel = BASE_MAX_REV_V19


@configclass
class AmrFr3TrajV23EnvCfg_PLAY(AmrFr3TrajV23EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v24 — v17 + 90 도 + **자세 사다리의 큰 오차 칸**
##

ORI_COARSE_STD = 0.80
ORI_COARSE_W = 0.20
"""자세 사다리에서 **큰 오차 구간**을 담당할 칸 (std, weight).

왜 필요한가
-----------
자세오차가 1.2 rad 근처일 때 기존 칸이 전부 포화한다::

    std 0.30 (w 0.20)   e/std =  4.2   포화
    std 0.12 (w 0.15)   e/std = 10.6   포화
    std 0.05 (w 0.15)   e/std = 25     포화
    선형   (w -0.05)                   이것만 작동.  기울기가 매우 약하다

**0.5 ~ 1.5 rad 를 끌어내릴 항이 없다.** v22 는 이 구간에서 위치 보상에 밀려
자세가 0.67 -> 1.28 rad 로 거꾸로 올라갔다 (it 150 -> 250).

같은 진단을 이미 두 번 했다 — 위치 사다리(``ee_position_precise``) 와 v17 의
``ee_orientation_mid`` (std 0.12). 둘 다 칸을 더해서 뚫었다.
``std 0.8`` 에서 1.2 rad 는 e/std = 1.5 로 기울기가 살아 있는 구간이다.
"""


@configclass
class AmrFr3TrajV24EnvCfg(AmrFr3TrajV22EnvCfg):
    """v22 (= v17 + 90 도) 에 자세 큰오차 칸만 더한다.

    v19 가 자세는 예쁘게 내려갔던 것(0.85 -> 0.11 rad) 은 사실 **팔이 얼어붙어
    기본자세를 유지한 덕**이었다 — 90 도 보정 뒤 목표 자세가 기본자세와 같은 방향이라
    가만히 있어도 자세 점수가 나온다. 위치(``ee_position_fine`` 0.033) 가 그 증거다.
    그래서 v19 의 자세 곡선을 목표로 삼으면 안 된다.

    이 판은 **위치를 v17 처럼 수렴시키면서 자세도 같이 내리는** 것이 목표다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.ee_orientation_coarse = RewTerm(
            func=mdp.orientation_command_error_tanh_w,
            weight=ORI_COARSE_W,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[AMR_FR3_EE_BODY]),
                "std": ORI_COARSE_STD,
                "command_name": "ee_pose",
            },
        )


@configclass
class AmrFr3TrajV24EnvCfg_PLAY(AmrFr3TrajV24EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v25 — v24 에서 이어받아 자세 비중을 낮춘다 (위치를 뚫으려고)
##


@configclass
class AmrFr3TrajV25EnvCfg(AmrFr3TrajV24EnvCfg):
    """v24 와 같고 ``ee_orientation_coarse`` 가중치만 절반.

    왜
    --
    v24 는 자세를 목표 이상으로 잘 잡았지만(0.066 rad = 3.8 도, 요청 0.1 미만)
    위치가 v17 보다 뒤진다 (1000 it 기준 118 mm vs v17 약 21 mm). 자세 항이 넷이 되어
    총점에서 차지하는 비중이 커진 탓으로 본다::

        자세 항 합 (coarse 0.20 + fine 0.20 + mid 0.15 + precise 0.15)  약 0.55
        위치 항 합                                                       그보다 작다

    자세에 여유가 있으므로 큰오차 칸을 절반으로 줄여 위치 쪽에 총점을 돌려준다.

    이어받기로 돌린다
    -----------------
    v24 체크포인트에서 이어받으면 400~1000 이터레이션어치를 안 버린다::

        agent.resume=true agent.load_run=<v24 run> agent.load_checkpoint=model_1000.pt

    .. note::
       보상 가중치가 바뀌므로 **critic 이 낡은 기준으로 학습돼 있다.** 처음 수십
       이터레이션의 지표는 못 믿는다. 보통 100 it 안에 회복한다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.ee_orientation_coarse.weight = ORI_COARSE_W * 0.5


@configclass
class AmrFr3TrajV25EnvCfg_PLAY(AmrFr3TrajV25EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v26 — v24 에서 이어받아 차체 추종을 되살리고 EE 위치를 조금 더 민다
##

BASE_BAND_M_V26 = 0.10
BASE_YAW_BAND_V26 = 0.25
"""차체 위치·방위 불감대 [m], [rad] (방위 약 14 도). v17 의 0.15 / 0.35 에서 좁힌다.

왜 가중치가 아니라 밴드인가
---------------------------
v24 는 차체 편차가 600 it 에서 0.077 m 로 바닥을 찍고 1200 it 에 0.101 m 로 되돌아갔다.
원인은 **밴드 안에서 벌점이 거의 안 늘기 때문**이다::

    편차 m   밴드항   약한항    합       기울기/cm
    0.077    0.500    0.111    0.611    -0.0047
    0.101    0.500    0.102    0.602    -0.0045     <- 3 cm 벌어져도 0.013 밖에 안 잃는다
    0.150    0.500    0.081    0.581    -0.0239     <- 경계에서 5 배 급증

3 cm 를 EE 위치에서 버는 이득이 더 커서 정책이 차체를 파는 것이 합리적이었다.

밴드를 0.10 으로 좁히면 **지금 서 있는 자리(0.101)** 에서 기울기가 5.5 배가 된다
(-0.0045 -> -0.0245 /cm). 가중치를 올리지 않아도 다시 당겨지고, 0.10 안쪽으로
들어가면 다시 평평해져 자세·EE 를 해치지 않는다.

차체 항의 가중치 합은 이미 약 0.83 으로 자세(0.55)·위치(0.50) 보다 크다. 더 올리면
균형이 무너진다 — v18(wrist_unwind)·v19(action_rate) 에서 두 번 겪은 실패다.
"""

EE_FINE_W_V26 = 0.45
"""EE 위치 fine 칸 가중치. **기본값 0.35** 에서 소폭 인상 (약 1.3 배).

v24 는 1200 it 에서 EE 105 mm 인데 v17 은 400 it 에 33 mm 였다. 자세 칸을 넷으로
늘리면서 위치가 밀렸다. 자세는 0.062 rad 로 목표(0.1) 를 크게 밑돌아 여유가 있다.
"""


@configclass
class AmrFr3TrajV26EnvCfg(AmrFr3TrajV24EnvCfg):
    """v24 이어받기용. 밴드를 좁히고 EE fine 을 조금 올린다.

    이어받아 돌린다 (처음부터 하면 600 mm 에서 다시 내려와야 해 1 시간 넘게 걸린다)::

        --resume --load_run <v24 run> --checkpoint <절대경로>/model_1300.pt

    .. warning::
       ``agent.resume=true`` 같은 hydra 형식은 **조용히 무시된다** (v25 가 그래서
       처음부터 돌았다). ``--resume`` / ``--load_run`` / ``--checkpoint`` 를 쓰고,
       checkpoint 는 **파일 절대경로**여야 한다.

    .. note::
       보상이 바뀌므로 critic 이 잠깐 낡는다. 초반 50~100 it 지표는 못 믿는다.
       밴드만 좁히는 작은 변경이라 흔들림도 작을 것으로 본다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.base_ref_tracking.params["band"] = BASE_BAND_M_V26
        self.rewards.base_ref_yaw.params["band"] = BASE_YAW_BAND_V26
        self.rewards.ee_position_fine.weight = EE_FINE_W_V26


@configclass
class AmrFr3TrajV26EnvCfg_PLAY(AmrFr3TrajV26EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v27 — v24 에서 이어받아 **자세 여유를 위치로 돌린다**
##

ORI_COARSE_W_V27 = 0.10
ORI_MID_W_V27 = 0.10
EE_FINE_W_V27 = 0.45
"""자세 두 칸을 절반으로, EE 위치 fine 을 0.35 -> 0.45.

왜
--
v24 는 실제 궤적 24 개에서 **자세 2.89 도 (중앙), 차체 0.074 m** 로 지금까지 최고인데
**EE 위치가 112 mm** 로 가장 나쁘다. 자세 칸을 넷으로 늘리며 위치가 밀린 것이다::

    자세 항 합   coarse 0.20 + fine 0.20 + mid 0.15 + precise 0.15 = 0.70
    위치 항 합   coarse + fine 0.35 + precise 0.25 + progress      = 약 0.50

자세는 목표(0.1 rad) 의 절반인 0.05 rad 까지 내려와 **여유가 크다.** 그 여유를 위치로
돌린다.

.. note::
   ``ee_orientation_precise`` (std 0.05) 는 건드리지 않는다 — 사용자 지시.
   큰/중간 칸만 줄인다.

무엇을 안 바꾸나
----------------
v26 에서 밴드까지 같이 좁혔다가 진전이 없었다. 차체를 조이면 EE 를 잡을 자유가 줄어
**이번 목적(위치 개선) 과 반대**로 작용한다. 밴드는 v24 값 (0.15 / 0.35) 그대로 둔다.
"""


@configclass
class AmrFr3TrajV27EnvCfg(AmrFr3TrajV24EnvCfg):
    """v24 이어받기. 자세 두 칸을 줄이고 위치 fine 을 올린다.

    이어받아 돌린다::

        --resume --load_run <v24 run> --checkpoint model_1300.pt

    .. warning::
       ``--checkpoint`` 는 **파일명**이다 (run 폴더 안에서 패턴으로 찾는다).
       절대경로를 주면 "No checkpoints match" 로 죽는다. ``play.py`` 의 같은 이름
       인자는 반대로 **절대경로**를 받는다 — 헷갈리기 쉬운 자리다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.ee_orientation_coarse.weight = ORI_COARSE_W_V27
        self.rewards.ee_orientation_mid.weight = ORI_MID_W_V27
        self.rewards.ee_position_fine.weight = EE_FINE_W_V27


@configclass
class AmrFr3TrajV27EnvCfg_PLAY(AmrFr3TrajV27EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v28 — v22 기반, 자세 역행만 막는다 (약한 큰오차 칸)
##

ORI_COARSE_W_V28 = 0.06
"""자세 큰오차 칸의 **약한** 가중치. v24 의 0.20 을 0.3 배로.

왜 이 값인가
------------
v22 (= v17 + 90 도) 와 v24 (= v22 + 큰오차 칸 0.20) 를 같은 이터레이션에서 비교하면
정확히 반대다::

            it 200      it 250      it 300
    v17     111.6 mm    72.8        54.1      자세 1.125 -> 0.954 -> 0.856
    v22     117.3       82.3        58.5      자세 1.148 -> 1.276 -> 1.279   <- 역행
    v24     292.2      246.3       236.7      자세 0.221 -> 0.177 -> 0.151   <- 자세는 좋고 위치가 죽음

**90 도 규약은 EE 에 영향이 없다** (v22 가 v17 을 그대로 따라간다). EE 를 망친 것은
큰오차 칸의 가중치 0.20 이다.

v22 의 역행 원인은 **큰 오차에서 자세를 포기하는 것이 거의 공짜**라는 데 있다.
1.2 rad 에서는 tanh 칸 셋(std 0.30/0.12/0.05) 이 전부 포화하고 약한 선형항만 남는다.
그래서 위치를 사려고 자세를 판다.

0.06 은 "포기가 공짜가 되지 않을 만큼만" 이다. 0.20 이 위치를 4 배 나쁘게 만들었으므로
0.3 배로 시작한다.
"""


@configclass
class AmrFr3TrajV28EnvCfg(AmrFr3TrajV22EnvCfg):
    """v22 (= v17 + 90 도) 에 **약한** 자세 큰오차 칸만 더한다. 처음부터 3000 학습.

    ``ee_orientation_precise`` (std 0.05) 는 건드리지 않는다 — 사용자 지시.

    판정
    ----
    ::

        EE      v22/v17 궤도(300 it 에 55~60 mm) 를 유지하는가
        자세    v22 처럼 250 이후 오르지 않고 v17 처럼 꺾이는가
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.ee_orientation_coarse = RewTerm(
            func=mdp.orientation_command_error_tanh_w,
            weight=ORI_COARSE_W_V28,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[AMR_FR3_EE_BODY]),
                "std": ORI_COARSE_STD,
                "command_name": "ee_pose",
            },
        )


@configclass
class AmrFr3TrajV28EnvCfg_PLAY(AmrFr3TrajV28EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v29 — v2 계열로 되돌아간다: 차체 참조를 버리고 EE 정밀도를 되찾는다
##

BASE_ANG_ACCEL_W = -0.30
BASE_ACTION_RATE_W_V29 = -0.06
BASE_ANG_VEL_W_V29 = -0.03
BASE_REVERSE_W_V29 = -0.08
"""차체 안정 항 넷. **MPC 참조를 쓰지 않고** 지그재그만 직접 잡는다.

왜 참조를 버리나
----------------
같은 시험대(실제 플래너 궤적 24 개, 높이 0.55) 에서::

                  EE 평균/중앙    자세 평균/중앙   차체 평균    3cm 통과
    v2  (참조 없음)  **8.6 / 8.0**  **2.18/1.21**   0.437       **92 %**
    v12 (0.35)       13.9  / -      10.5 / -        0.240        84 %
    v17 (밴드)       31.4 / 34.2    17.2 / 3.00     0.126        33 %
    v28 (밴드)       41.7 / 31.8     5.18/ 2.08     0.130        25 %
    v24 (밴드+칸)   105.1 /112.7     3.48/ 2.89     0.117        17 %

**차체를 조일수록 EE 가 나빠지는 것이 일관된다.** 차체 참조를 얻는 대가가 EE 5 배였다.
차동구동은 옆으로 못 가므로, 팔이 좋은 자세를 만들려면 **차체가 자리를 옮겨야** 한다.
참조가 그것을 금지한다.

무엇으로 대신하나
-----------------
::

    base_ang_accel   각속도의 **한 스텝 변화량 제곱**.  방향을 바꾸는 행위만 벌한다.
                     (step_dt 로 나누면 900 배가 되어 다른 항을 압도한다 — 실측 -1.02)
    base_action_rate 차체 지령 변화율.  평활
    base_ang_vel     각속도 크기.  불필요한 선회 억제
    base_reverse     후진 비용.  갈팡질팡 억제

각가속도가 핵심이다 — 지그재그는 각속도의 **부호가 자주 뒤집히는 것**이지 크기가 큰
것이 아니다. 각속도 자체를 크게 벌하면 필요한 선회까지 굼떠진다.

운용상 대가
-----------
차체가 계획 경로에서 벌어진다 (v2 실측 중앙 0.447 m, 95 % 약 0.55). 플래너
``base_margin`` 을 **0.60 m 이상**으로 잡아야 장애물 여유 보장이 넘어온다.
"""


@configclass
class AmrFr3TrajV29EnvCfg(AmrFr3TrajV7EnvCfg):
    """v7 (= v2 + 페이로드 + 차체 감쇠) 에 나머지 넷을 한 번에 얹는다.

    ::

        1. 90 도 규약        PLAN_YAW_OFFSET 이 전역이라 자동 적용
        2. 최신 URDF         v7 이 이미 페이로드 에셋을 쓴다
        3. 학습 궤적          실측 (v, omega) 이어붙이기 생성본.  **EE 목표만** 쓴다
        4. 목표 높이          0.50 ~ 0.60 (실기 기준 0.55 +- 5 cm)
        5. 차체 안정          MPC 참조 대신 각가속도·변화율·후진 벌점

    관측은 **52 차원**이다 (차체 참조 없음). 실기 브리지는 v2 시절 것을 그대로 쓰면
    되고 ``/base_ref_*`` 토픽이 필요 없다. 다만 **목표 요에 -90 도 보정은 필요하다.**

    .. note::
       한 번에 다섯 가지를 바꾼다. 원인 분리를 포기하는 대신 시간을 아끼는 선택이며,
       실패하면 하나씩 되돌린다 (사용자 합의).
    """

    def __post_init__(self):
        super().__post_init__()
        # [3] 학습 궤적 — 데이터셋에서 EE 목표만 가져온다 (base_ref 는 꺼져 있음)
        self.commands.ee_pose.plan_dataset = PLAN_SEEDED
        self.commands.ee_pose.plan_mix = 1.0
        self.commands.ee_pose.plan_test_n = 0
        self.commands.ee_pose.plan_time_scale = (1.0, 1.0)
        # [4] 목표 높이
        self.commands.ee_pose.plan_goal_z_range = PLAN_GOAL_Z_RANGE_V14
        # [5] 차체 안정 — 참조 없이
        self.rewards.base_ang_accel = RewTerm(
            func=mdp.base_ang_accel_l2, weight=BASE_ANG_ACCEL_W
        )
        self.rewards.base_action_rate.weight = BASE_ACTION_RATE_W_V29
        self.rewards.base_ang_vel.weight = BASE_ANG_VEL_W_V29
        self.rewards.base_reverse.weight = BASE_REVERSE_W_V29


@configclass
class AmrFr3TrajV29EnvCfg_PLAY(AmrFr3TrajV29EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


##
# v30 — v29 의 실기 지적 둘을 고친다 (그리퍼 방향, 추종 뒤쳐짐)
##


@configclass
class AmrFr3TrajV30EnvCfg(AmrFr3TrajV29EnvCfg):
    """v29 와 설정은 같다. 바뀐 것은 **전역 상수와 학습 데이터**다.

    실기 v29 지적 (2026-08-08)
    --------------------------
    ::

        1. EE 타겟에 뒤쳐지며 따라간다
        2. j6 가 반대로 90 도 틀렸다
           필요:  어깨 - 그리퍼 - 물체 - 그리퍼 - 어깨   (팔이 물체와 안 부딪힘)
           현재:  그리퍼 - 어깨 - 물체 - 어깨 - 그리퍼
        3. 차체 지그재그 (학습 이슈가 아닐 수 있음 — 관제 파라미터로 해결됐던 전례)

    무엇을 고쳤나
    -------------
    **[2] 부호** — ``PLAN_YAW_OFFSET`` 을 -pi/2 -> **+pi/2**. 막대 축만 보면 +-90 도가
    같지만 **그리퍼 몸통이 어느 쪽에 오는지가 반대**다. 기구학적 부담은 비슷하다
    (필요 |j6| 중앙 49.6 도, >170 도 1.58 %).

    **[1] 속도** — 학습 궤적을 실기 운전조건에 맞게 빠르게 만들었다::

                      차체 v 중앙/95%    EE 속력 95%
        v29 데이터     0.061 / 0.128     0.19
        v30 데이터     0.104 / 0.206     0.26
        실측 122 개    0.056 / 0.118     0.147      <- 관제 속도를 올리기 **전** 값

    관제에서 한계속도를 올려 "갔다 멈췄다" 를 고치셨는데 학습 데이터는 옛 저속 그대로였다.
    목표가 학습보다 빠르면 정책은 구조적으로 뒤쳐진다.

    **[3] 은 건드리지 않는다** — v29 의 차체 안정 항(각가속도 등) 을 그대로 두고,
    관제 파라미터(``reach_max`` 0.75 -> 0.60, ``w_max``) 쪽 가능성을 먼저 본다.
    """


@configclass
class AmrFr3TrajV30EnvCfg_PLAY(AmrFr3TrajV30EnvCfg):
    """재생·평가용."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False
