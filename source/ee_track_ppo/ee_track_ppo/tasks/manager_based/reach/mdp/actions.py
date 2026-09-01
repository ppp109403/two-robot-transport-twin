# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""sim-to-real 용 액션 term.

왜 필요한가 (RETRAIN_V2 §1-B)
-----------------------------
실기 FR3 는 정책이 낸 관절 목표가 실제로 반영되기까지 **72 ms** 가 걸린다.
정책 주기가 33 ms 이므로 약 2.2 주기 분량이다.
이 지연은 FAIRINO 컨트롤러 내부 ServoJ 파이프라인에서 나오므로 우리가 줄일 수 없다
(상태 읽기는 XML-RPC 직접 폴링으로 1.5 ms, 추론은 1 ms 미만까지 이미 내려갔다).

시뮬에 지연이 없으면 정책은 "내 명령은 즉시 실행된다"를 전제로 학습한다.
그 정책은 72 ms 지연 앞에서 반드시 발산한다.

실측 근거: 실기에서 follow_gain 을 0.05 로 올려 alpha 를 0.29 로 만든 적이 있다.
시뮬의 alpha 0.23 과 사실상 같은 영역인데도 **108 mm 로 발산**했다.
=> alpha 만 맞추는 것으로는 부족하고 (alpha, 지연) **쌍**이 맞아야 한다.

왜 DelayedPDActuator 가 아니라 액션 버퍼인가
--------------------------------------------
Isaac Lab 의 :class:`DelayedPDActuator` 도 같은 일을 하지만 **explicit 액추에이터**라
물리 경로가 바뀌고 ``effort_limit`` 이 실제로 걸리기 시작한다.
이번 사이클의 방침은 "시뮬의 강성·감쇠·보상을 전부 그대로 두고 실기를 시뮬에 맞춘다"
이므로 물리 경로까지 바꿀 이유가 없다. ImplicitActuatorCfg 를 유지한 채
**설정점이 액추에이터에 도달하는 시각만** 늦춘다.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions import joint_actions
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DelayedJointPositionAction(joint_actions.JointPositionAction):
    """관절 위치 설정점을 **N 물리 스텝**만큼 늦춰서 적용하는 액션 term.

    지연 단위가 정책 스텝이 아니라 **물리 스텝**인 이유는 분해능이다.
    정책 30 Hz 기준으로는 2 주기(66 ms) 아니면 3 주기(100 ms) 밖에 못 고르지만,
    물리 60 Hz 기준이면 16.7 ms 단위라 실측치 72 ms 를 감쌀 수 있다::

        50 ms -> 3 스텝      72 ms -> 4.3 스텝      100 ms -> 6 스텝

    지연량은 env 마다 다르게 뽑고 **에피소드 리셋마다 다시 뽑는다**
    (도메인 랜덤화. 하나의 정책이 50~100 ms 전 구간에서 동작해야 한다).
    """

    cfg: DelayedJointPositionActionCfg

    def __init__(self, cfg: DelayedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        if not (0 <= cfg.min_delay <= cfg.max_delay):
            raise ValueError(f"min_delay/max_delay 가 잘못됐다: {cfg.min_delay}, {cfg.max_delay}")

        self._buf_len = cfg.max_delay + 1
        # (buf_len, num_envs, action_dim) 링 버퍼. 물리 스텝마다 한 칸씩 돈다.
        self._buffer = torch.zeros(self._buf_len, self.num_envs, self.action_dim, device=self.device)
        # env 별 지연량 [물리 스텝]
        self._delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._head = 0
        self._env_idx = torch.arange(self.num_envs, device=self.device)
        # 첫 리셋 전에도 버퍼가 유효해야 한다 (기본 관절자세로 채움)
        self._prime_buffer(None)

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = self._env_idx if env_ids is None else torch.as_tensor(env_ids, device=self.device).long()
        # 지연량 재추첨
        self._delay[ids] = torch.randint(
            self.cfg.min_delay, self.cfg.max_delay + 1, (len(ids),), device=self.device
        )
        # 리셋된 env 의 버퍼는 이전 에피소드의 명령을 들고 있으면 안 된다.
        self._prime_buffer(ids)

    def apply_actions(self):
        # 이번 물리 스텝의 명령을 버퍼에 넣고, delay 스텝 전의 명령을 꺼내 적용한다.
        self._buffer[self._head] = self._processed_actions
        read = (self._head - self._delay) % self._buf_len
        delayed = self._buffer[read, self._env_idx]
        self._asset.set_joint_position_target(delayed, joint_ids=self._joint_ids)
        self._head = (self._head + 1) % self._buf_len

    """
    Helpers.
    """

    def _prime_buffer(self, env_ids: torch.Tensor | None) -> None:
        """버퍼 전체를 "현재 기본 관절자세" 로 채운다.

        비워두면(0 으로) 에피소드 시작 직후 delay 스텝 동안 관절 목표가 0 rad 로 튄다.
        이는 실기에 없는 현상이고 학습 초반을 크게 망친다.
        """
        default = self._asset.data.default_joint_pos[:, self._joint_ids]
        if env_ids is None:
            self._buffer[:] = default.unsqueeze(0)
        else:
            self._buffer[:, env_ids] = default[env_ids].unsqueeze(0)


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """:class:`DelayedJointPositionAction` 설정.

    ``JointPositionActionCfg`` 를 그대로 상속하므로 ``scale`` / ``use_default_offset`` 등
    v1 과 동일한 값을 쓸 수 있다. 지연 관련 필드 두 개만 추가된다.
    """

    class_type: type[ActionTerm] = DelayedJointPositionAction

    min_delay: int = MISSING
    """지연 하한 [물리 스텝]. sim dt=1/60 기준 3 = 50 ms."""

    max_delay: int = MISSING
    """지연 상한 [물리 스텝]. sim dt=1/60 기준 6 = 100 ms."""
