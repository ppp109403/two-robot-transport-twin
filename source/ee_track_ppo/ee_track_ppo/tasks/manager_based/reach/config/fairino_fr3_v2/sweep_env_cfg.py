# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FR3 v2 액추에이터 모델 위에서 **보상만** 바꿔 보는 스윕 (v2a ~ v2e).

전제
----
v2 본 학습(2000 iter, 4096 envs)이 끝났고 결과는::

    마지막 200 it 평균 : position 9.5 mm / orientation 0.0522 rad (2.99 deg)
    it1200 이후 800 iteration 동안 10.3 -> 9.5 mm (8%)  -> 사실상 정체

지연·DR 은 **다섯 갈래 모두 v2 와 동일**하게 고정한다. 이번에 움직이는 변수는
보상 하나뿐이라, v2 를 기준선으로 두면 귀속이 깨끗하다.

왜 위치 사다리에 칸이 비었다고 보는가 (진단)
--------------------------------------------
v2 종료 시점 실현 보상::

    pos L2              -0.0038
    pos fine (std 0.1)   0.2986        <- 위치 신호의 거의 전부
    ori L2              -0.0520
    ori fine (std 0.3)   0.0727
    ori precise(std .05) 0.0296        <- 건드리지 않음
    wrist_joint_vel     -0.0117

위치:자세 실현 비율은 5.9 : 1 로 이미 위치가 압도적인데도 위치가 9.5 mm 에서 멈췄다.
가중치가 부족한 게 아니라 **그 오차 지점에 기울기가 없다**는 뜻이다.

    1 - tanh(0.0095 / 0.1) = 0.905     -> 90% 포화

tanh 성형보상의 기울기는 ``sech^2(e/s)/s`` 라 ``e ~ s`` 에서 최대다. 현재 오차 9.5 mm
기준으로 비교하면::

    std 0.1  : sech^2(0.095)/0.1 = 9.9
    std 0.02 : sech^2(0.475)/0.02 = 39.7      (4.0 배)
    std 0.01 : sech^2(0.95) /0.01 = 46.6      (4.7 배)

자세 쪽은 반대로 이미 제자리다. 오차 0.052 rad 에 precise 의 std 가 0.05 라
``e ~ s`` 조건을 정확히 만족한다. 그래서 이 항은 **가중치도 std 도 건드리지 않는다.**

원칙: 좁은 칸은 **더하는 것**이지 넓은 칸을 **바꾸는 것**이 아니다
------------------------------------------------------------------
franka v1 에서 위치 tanh std 를 0.1 -> 0.05 로 *교체*했다가 학습 신호가 통째로
소멸했다 (그때 오차 0.385 m 에서 ``1-tanh(0.385/0.05) ~= 0``). 넓은 칸은 멀리서
끌어오는 역할이라 반드시 남겨야 한다. v2a/v2b/v2e 는 전부 **추가**다.

다섯 갈래
---------
기준선(v2) 대비 **한 번에 하나씩만** 바꾼다. v2e 만 예외로 a+c 를 겹쳐서
두 조치가 서로 보태지는지(아니면 겹치는지) 본다.

    v2a  + 위치 precise 칸 (std 0.01, w 0.15)      <- 진단상 본命
    v2b  + 위치 precise 칸 (std 0.02, w 0.20)      <- 0.01 이 너무 좁을 때의 대비
    v2c  위치 fine 가중치 0.35 -> 0.50             <- 새 항 없이 손잡이만
    v2d  wrist_joint_vel -0.005 -> -0.002          <- 손목 억제 완화
    v2e  v2a + v2c                                 <- 겹치는지 확인

v2d 의 근거: 손목 속도 페널티가 -0.0117 을 먹고 있는데 이는 자세 신호 총합(0.0503)의
23% 다. 이 항은 원래 franka 의 7DOF 여유자유도 배회를 잡으려던 것이고 6DOF FR3 에는
그 진단이 성립하지 않는다(v1 주석에 이미 의문이 적혀 있다). 게다가 이제 지연이
들어가 폐루프에 감쇠가 생겼으므로 손목을 이만큼 억제할 이유가 더 줄었다.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp

from ee_track_ppo.assets.fairino_fr3 import FAIRINO_FR3_EE_BODY  # isort: skip
from .sim2real_env_cfg import FairinoFR3Sim2RealEnvCfg  # isort: skip


def _add_position_precise(cfg: FairinoFR3Sim2RealEnvCfg, std: float, weight: float) -> None:
    """위치 tanh 사다리에 좁은 칸을 **추가**한다 (기존 std 0.1 칸은 그대로 둔다)."""
    cfg.rewards.end_effector_position_tracking_precise = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=weight,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[FAIRINO_FR3_EE_BODY]),
            "std": std,
            "command_name": "ee_pose",
        },
    )


@configclass
class FR3SweepAEnvCfg(FairinoFR3Sim2RealEnvCfg):
    """v2a: 위치 precise 칸 추가 (std 0.01, w 0.15).

    현재 오차 9.5 mm 와 std 를 맞춰 기울기가 최대인 지점에 칸을 놓는다.
    """

    def __post_init__(self):
        super().__post_init__()
        _add_position_precise(self, std=0.01, weight=0.15)


@configclass
class FR3SweepBEnvCfg(FairinoFR3Sim2RealEnvCfg):
    """v2b: 위치 precise 칸 추가 (std 0.02, w 0.20).

    v2a 보다 넓다. std 0.01 이 좁아서 학습 초반(오차가 아직 클 때) 이 칸이 죽어
    있는 시간이 길면 v2b 가 유리할 수 있다. 가중치를 조금 더 준 것은 std 가 넓어
    단위 오차당 기울기가 낮은 것을 보상하기 위해서다.
    """

    def __post_init__(self):
        super().__post_init__()
        _add_position_precise(self, std=0.02, weight=0.20)


@configclass
class FR3SweepCEnvCfg(FairinoFR3Sim2RealEnvCfg):
    """v2c: 위치 fine 가중치만 0.35 -> 0.50.

    새 항 없이 기존 손잡이만 돌린다. franka v7 실험에서 "이 영역의 실질적 손잡이는
    자세가 아니라 위치 가중치" 라는 결론이 나왔던 조치다.
    다만 std 0.1 칸이 이미 90% 포화라 **기울기가 아니라 크기만 커질** 위험이 있다.
    그걸 확인하는 게 이 갈래의 목적이다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.end_effector_position_tracking_fine_grained.weight = 0.50


@configclass
class FR3SweepDEnvCfg(FairinoFR3Sim2RealEnvCfg):
    """v2d: 손목 속도 페널티 -0.005 -> -0.002.

    다른 넷과 달리 **자세**를 겨냥한 갈래다. 위치는 v2 와 비슷하게 나오고 자세만
    좋아지면 성공, 대신 실기에서 떨림이 늘어날 위험이 있으니 그때는 채택하지 않는다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.wrist_joint_vel.weight = -0.002


@configclass
class FR3SweepEEnvCfg(FairinoFR3Sim2RealEnvCfg):
    """v2e: v2a + v2c 를 겹친다.

    a 와 c 가 각각 효과가 있다면 이 갈래가 가장 좋아야 하고, 겹쳐서 위치 보상이
    과해지면 자세가 밀려나면서 오히려 나빠질 것이다. 둘 중 뭔지 보려고 넣었다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.end_effector_position_tracking_fine_grained.weight = 0.50
        _add_position_precise(self, std=0.01, weight=0.15)


@configclass
class FR3SweepFEnvCfg(FR3SweepEEnvCfg):
    """v2f: v2e 위에 칸을 하나 더 (std 0.004, w 0.15).

    스윕이 끝난 뒤 그 결과를 보고 **추가로 넣은 갈래**다 (a~e 와 달리 사후 설계).

    근거: v2e 가 3000 iteration 에서 4.0 mm 로 정체했다. 같은 진단을 한 칸 더 적용하면
    이번에 비는 곳은 4 mm 대다::

        std 0.01  : 1 - tanh(0.004/0.01 ) = 0.62   아직 포화는 아니다
        std 0.004 : 1 - tanh(0.004/0.004) = 0.24   e ~ std 로 기울기 최대

    v2e 가 정체한 이유가 "사다리에 칸이 또 비어서" 라면 이 갈래가 더 내려가고,
    "액추에이터 모델(지연 33~117 ms + alpha 0.15~0.35)이 만든 물리적 바닥" 이라면
    v2e 와 같은 자리에 멈춘다. **둘 중 무엇인지 가르는 게 목적**이라, 안 내려가도
    그 자체로 답이 된다.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.end_effector_position_tracking_ultra = RewTerm(
            func=mdp.position_command_error_tanh,
            weight=0.15,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[FAIRINO_FR3_EE_BODY]),
                "std": 0.004,
                "command_name": "ee_pose",
            },
        )


##
# PLAY 변형 (export 용)
##


@configclass
class FR3SweepAEnvCfg_PLAY(FR3SweepAEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FR3SweepBEnvCfg_PLAY(FR3SweepBEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FR3SweepCEnvCfg_PLAY(FR3SweepCEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FR3SweepDEnvCfg_PLAY(FR3SweepDEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FR3SweepEEnvCfg_PLAY(FR3SweepEEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FR3SweepFEnvCfg_PLAY(FR3SweepFEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
