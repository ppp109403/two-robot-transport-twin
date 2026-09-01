# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""차동구동 액션 term.

정책 출력은 2차원 **바디 트위스트** (전진속도 v, 요레이트 ω) 이고,
이 term 이 차동구동 역기구학으로 좌/우 휠 각속도 목표로 변환한다.

왜 휠 속도가 아니라 (v, ω) 를 정책 출력으로 두는가
----------------------------------------------------
실로봇 인터페이스가 ROS2 ``geometry_msgs/Twist`` (``/cmd_vel``) 다.
정책이 곧 ``/cmd_vel`` 퍼블리셔가 되므로 배포 시 변환 계층이 필요 없다.

그리고 차동구동 로봇끼리 **다른 것은 사실상 두 숫자**(휠 반경, 윤거)뿐이다.
그 두 값은 (v, ω) -> 휠 각속도 변환 상수로만 등장하므로,
정책을 (v, ω) 공간에서 학습해 두면 로봇을 바꿔도 정책을 그대로 옮길 수 있다.
반대로 정책이 휠 토크/휠 속도를 직접 내면 관성·마찰·기어비 오차가
정책 파라미터에 흡수되어 URDF 정확도가 곧 sim2real 격차가 된다.

그럼에도 **실제 휠 조인트를 구동**하는 이유
--------------------------------------------
Isaac Lab 내장 :class:`~isaaclab.envs.mdp.NonHolonomicAction` 은
더미 planar 조인트(prismatic x/y + revolute z)를 직접 속도 제어한다.
간단하고 안정적이지만 접지·슬립·가속 한계가 전혀 재현되지 않는다.
(명령한 속도가 그대로 실현된다 — 즉 완벽한 로봇이다)

여기서는 실제 바퀴를 돌려 접지 마찰로 로봇을 밀어낸다. 그래서
  * 급가속 시 휠 슬립
  * 회전 시 캐스터 저항
  * 하중/질량중심에 따른 거동
이 물리적으로 재현되고, 정책이 그 한계를 감안한 명령을 배우게 된다.

가감속 한계
-----------
실제 모터 드라이버는 명령을 즉시 따르지 못하고 램프를 탄다.
그 램프를 여기서 명시적으로 걸어 둔다 (``max_lin_accel`` / ``max_ang_accel``).
이것이 없으면 정책이 스텝마다 부호가 바뀌는 명령을 내도 이득을 볼 수 있어
실로봇에서 재현되지 않는 고주파 진동 정책이 나온다.

트위스트 이득 랜덤화 (``lin_gain_range`` / ``ang_gain_range``)
--------------------------------------------------------------
"명령한 (v, ω) 가 실제로 얼마나 실현되는가"는 sim 에서 정확히 맞출 수 없는 값이다.
휠 반경 공차, 감속비 오차, 타이어 마모, 접지 마찰, 캐스터 저항, 모터 토크 한계가
모두 여기에 섞여 들어간다. 특히 **제자리 선회**는 캐스터 스크럽 저항 때문에
실현율이 구조적으로 1.0 보다 낮고, 그 값은 바닥 재질과 캐스터 상태에 따라 변한다.

실측(이 로봇): 전진 실현율 0.99 / 제자리 선회 실현율 0.73.

선회 실현율 0.73 은 다음을 다 시도해도 움직이지 않았다.
  * 마찰 고정 0.8 / 1.0 / 1.2 / 3.0     -> 0.69 ~ 0.73 (3.0 에서 오히려 악화)
  * 모터 토크 한계 15 / 40 / 60 N*m     -> 소수 3자리까지 동일 (60 에서 56 N*m 인가)
  * 캐스터 콜리전 원기둥 -> 구           -> 0.694 -> 0.731
  * 캐스터 높이 2 mm 조정 (하중 배분)    -> 무변화
즉 시뮬레이터에서 임의로 맞출 수 있는 값이 아니고, 실로봇의 값은 바닥 재질과
캐스터 마모에 따라 또 달라진다.

이걸 억지로 1.0 에 맞추는 것은 틀린 접근이다 (실로봇에서 어떤 값일지 모르므로).
대신 **env 마다 이득을 무작위로 뽑아** 정책이 이득 오차에 강건해지게 만든다.
정책은 관측으로 자기 (v, ω) 와 목표 상대 pose 를 모두 보고 있으므로,
이득이 흔들려도 폐루프로 보정하는 법을 배울 수 있다 — 그 능력을 학습으로 강제한다.

이득 범위는 **실현율**이 1.0 을 중심으로 퍼지도록 잡는다.
선회는 실현율이 0.73 이므로 이득 1/0.73 = 1.37 이 실현율 1.0 에 해당한다.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DiffDriveAction(ActionTerm):
    r"""2차원 액션 (v, ω) 을 좌/우 휠 각속도 목표로 변환하는 액션 term.

    .. math::

        \dot{\phi}_L = \frac{v - \omega b / 2}{r}, \qquad
        \dot{\phi}_R = \frac{v + \omega b / 2}{r}

    :math:`r` 은 휠 반경, :math:`b` 는 윤거(좌우 휠 간격)다.

    처리 순서는 다음과 같다.

    1. 정책 출력(무계 가우시안)에 최대 속도를 곱하고 물리 한계로 클리핑
    2. 이전 스텝 명령 기준으로 가감속 램프 적용
    3. 차동구동 역기구학으로 휠 각속도 계산
    4. 휠 각속도 한계로 클리핑

    4번에서 클리핑하면 (v, ω) 조합이 미세하게 변형되지만, 실제 드라이버도
    같은 방식으로 포화하므로 의도한 동작이다.
    """

    cfg: DiffDriveActionCfg
    """액션 term 설정."""
    _asset: Articulation
    """액션을 적용할 articulation."""

    def __init__(self, cfg: DiffDriveActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # 좌/우 휠 조인트 해석. 한쪽에 바퀴가 여러 개여도(스키드 스티어) 된다.
        self._left_ids, self._left_names = self._asset.find_joints(cfg.left_wheel_joint_names)
        self._right_ids, self._right_names = self._asset.find_joints(cfg.right_wheel_joint_names)
        if len(self._left_ids) == 0:
            raise ValueError(f"좌측 휠 조인트를 찾지 못했다: {cfg.left_wheel_joint_names}")
        if len(self._right_ids) == 0:
            raise ValueError(f"우측 휠 조인트를 찾지 못했다: {cfg.right_wheel_joint_names}")
        if len(self._left_ids) != len(self._right_ids):
            raise ValueError(
                "좌/우 휠 조인트 개수가 다르다:"
                f" 좌 {self._left_names} vs 우 {self._right_names}"
            )

        # 좌 -> 우 순서로 이어붙인 인덱스. set_joint_velocity_target 에 한 번에 넣는다.
        self._joint_ids = list(self._left_ids) + list(self._right_ids)
        self._num_per_side = len(self._left_ids)

        # 버퍼
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        # _processed_actions = 램프까지 적용된 실제 (v, ω) 명령. 다음 스텝 램프의 기준이 된다.
        self._processed_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._wheel_vel_target = torch.zeros(self.num_envs, 2 * self._num_per_side, device=self.device)

        # 정책 스텝당 허용 변화량
        self._max_dv = cfg.max_lin_accel * self._env.step_dt
        self._max_dw = cfg.max_ang_accel * self._env.step_dt

        # 휠 각속도 한계. 지정하지 않으면 (v, ω) 한계에서 유도한다.
        #   이득 랜덤화 상한까지 낼 수 있어야 하므로 최대 이득을 곱해 둔다.
        if cfg.max_wheel_vel is None:
            max_gain = max(cfg.lin_gain_range[1], cfg.ang_gain_range[1], 1.0)
            self._max_wheel_vel = (
                max_gain * (cfg.max_lin_vel + cfg.max_ang_vel * cfg.wheel_base * 0.5) / cfg.wheel_radius
            )
        else:
            self._max_wheel_vel = cfg.max_wheel_vel

        # env 별 트위스트 이득 (리셋 때 재샘플)
        self._lin_gain = torch.ones(self.num_envs, device=self.device)
        self._ang_gain = torch.ones(self.num_envs, device=self.device)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        """정책이 낸 원본 액션. (num_envs, 2)"""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """물리 단위로 변환되고 램프까지 적용된 (v [m/s], ω [rad/s]). (num_envs, 2)"""
        return self._processed_actions

    @property
    def wheel_velocity_target(self) -> torch.Tensor:
        """휠 각속도 목표 [rad/s]. 좌측 먼저, 그다음 우측."""
        return self._wheel_vel_target

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        # 1. 물리 단위로 스케일 + 클리핑
        v_cmd = torch.clamp(
            actions[:, 0] * self.cfg.max_lin_vel, min=-self.cfg.max_reverse_vel, max=self.cfg.max_lin_vel
        )
        w_cmd = torch.clamp(actions[:, 1] * self.cfg.max_ang_vel, min=-self.cfg.max_ang_vel, max=self.cfg.max_ang_vel)

        # 2. 가감속 램프 (이전 명령 기준)
        v_prev = self._processed_actions[:, 0]
        w_prev = self._processed_actions[:, 1]
        v = v_prev + torch.clamp(v_cmd - v_prev, min=-self._max_dv, max=self._max_dv)
        w = w_prev + torch.clamp(w_cmd - w_prev, min=-self._max_dw, max=self._max_dw)
        self._processed_actions[:, 0] = v
        self._processed_actions[:, 1] = w

        # 3. 차동구동 역기구학. env 별 이득을 여기서 곱한다.
        #    휠 목표에 이득을 곱하면 같은 정책 출력에 대해 로봇이 이득배로 움직인다.
        #    즉 정책 관점에서 "액추에이터 이득이 env 마다 다른" 상황이 된다.
        half_track = self.cfg.wheel_base * 0.5
        v_g = v * self._lin_gain
        w_g = w * self._ang_gain
        left = (v_g - w_g * half_track) / self.cfg.wheel_radius
        right = (v_g + w_g * half_track) / self.cfg.wheel_radius

        # 4. 휠 각속도 한계
        left = torch.clamp(left, min=-self._max_wheel_vel, max=self._max_wheel_vel)
        right = torch.clamp(right, min=-self._max_wheel_vel, max=self._max_wheel_vel)

        # 같은 쪽 바퀴에는 같은 목표를 준다
        self._wheel_vel_target[:, : self._num_per_side] = left.unsqueeze(1)
        self._wheel_vel_target[:, self._num_per_side :] = right.unsqueeze(1)

    def apply_actions(self):
        self._asset.set_joint_velocity_target(self._wheel_vel_target, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
            num = self.num_envs
        else:
            num = len(env_ids)
        self._raw_actions[env_ids] = 0.0
        # 램프 기준값도 0 으로. 안 하면 리셋 직후 이전 에피소드의 속도에서 이어진다.
        self._processed_actions[env_ids] = 0.0
        self._wheel_vel_target[env_ids] = 0.0

        # 트위스트 이득 재샘플
        lo, hi = self.cfg.lin_gain_range
        if lo != 1.0 or hi != 1.0:
            self._lin_gain[env_ids] = torch.empty(num, device=self.device).uniform_(lo, hi)
        lo, hi = self.cfg.ang_gain_range
        if lo != 1.0 or hi != 1.0:
            self._ang_gain[env_ids] = torch.empty(num, device=self.device).uniform_(lo, hi)


@configclass
class DiffDriveActionCfg(ActionTermCfg):
    """:class:`DiffDriveAction` 설정."""

    class_type: type[ActionTerm] = DiffDriveAction

    left_wheel_joint_names: list[str] = MISSING
    """좌측 구동륜 조인트 이름 (정규식 가능). 스키드 스티어면 여러 개."""

    right_wheel_joint_names: list[str] = MISSING
    """우측 구동륜 조인트 이름. 좌측과 개수가 같아야 한다."""

    wheel_radius: float = MISSING
    """구동륜 반경 [m]."""

    wheel_base: float = MISSING
    """좌우 구동륜 간격(윤거) [m]."""

    max_lin_vel: float = 0.8
    """최대 전진속도 [m/s]. 정책 출력 1.0 이 이 값에 대응한다."""

    max_ang_vel: float = 1.5
    """최대 요레이트 [rad/s]."""

    max_reverse_vel: float = 0.3
    """최대 후진속도 [m/s] (양수로 지정). 실로봇은 보통 후방 센서가 없어 후진을 제한한다."""

    max_lin_accel: float = 1.0
    """전진 가감속 한계 [m/s^2]. 정책 스텝당 변화량으로 환산해 램프를 건다."""

    max_ang_accel: float = 3.0
    """요레이트 가감속 한계 [rad/s^2]."""

    max_wheel_vel: float | None = None
    """휠 각속도 한계 [rad/s]. ``None`` 이면 (v, ω) 한계와 최대 이득에서 유도한다."""

    lin_gain_range: tuple[float, float] = (1.0, 1.0)
    """전진속도 이득 랜덤화 범위. env 마다 리셋 때 균등 샘플링한다.

    ``(1.0, 1.0)`` 이면 랜덤화하지 않는다. 휠 반경 공차·감속비 오차·타이어 마모를 덮는다.
    """

    ang_gain_range: tuple[float, float] = (1.0, 1.0)
    """요레이트 이득 랜덤화 범위.

    제자리 선회 실현율은 캐스터 스크럽 저항과 모터 토크 한계에 좌우되어
    sim 에서 실로봇 값을 맞출 수 없다. 전진보다 넓게 잡는 것이 맞다.
    """
