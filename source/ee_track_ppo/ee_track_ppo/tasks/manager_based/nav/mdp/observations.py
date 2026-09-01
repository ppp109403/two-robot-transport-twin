# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMR pose 도달 태스크용 관측 함수.

설계 원칙: 관측은 **실로봇에서 계산 가능한 값**만 담는다
--------------------------------------------------------
실로봇에서 정책 입력은 ROS2 Nav2 스택의 산출물이다.

============================  ==========================================
관측                          실로봇 출처
============================  ==========================================
로봇 pose (map 프레임)        AMCL (``/amcl_pose``) — 2D 라이다 + 휠 오도메트리
목표 pose (map 프레임)        상위 계획기가 지정 (정확한 값)
바디 트위스트 (v, ω)          휠 엔코더 오도메트리 (``/odom``)
이전 액션                     정책이 직전에 퍼블리시한 ``/cmd_vel``
============================  ==========================================

절대좌표를 그대로 넣지 않는 이유
--------------------------------
AMCL 은 map 프레임의 절대 pose (x, y, yaw) 를 준다. 이것을 관측에 **날것으로**
넣으면 정책이 "학습 시 목표가 뿌려졌던 좌표 영역"에 과적합된다.
실제 맵의 원점 위치·크기·좌표 부호는 학습 환경과 다르므로 그대로 옮겨지지 않는다.

그래서 AMCL pose 와 목표 pose 로부터 **베이스 프레임 기준 상대량**을 만들어 넣는다.
이렇게 하면 관측이 평행이동/회전 불변이 되어 맵 어디서든 같은 정책이 동작한다.
sim 과 실로봇이 완전히 같은 전처리 수식을 쓰므로 배포 시 재구현 위험도 없다.

.. math::

    \\rho = \\lVert p_{goal} - \\hat{p} \\rVert, \\quad
    \\alpha = \\mathrm{atan2}(\\Delta y, \\Delta x) - \\hat{\\theta}, \\quad
    \\theta_e = \\theta_{goal} - \\hat{\\theta}

각도는 :math:`(\\cos, \\sin)` 쌍으로 넣는다. :math:`\\pm\\pi` 경계에서 불연속인
스칼라 각도를 그대로 넣으면 그 지점에서 정책 출력이 튄다.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class amcl_goal_polar(ManagerTermBase):
    """AMCL 로컬라이제이션을 모사해 목표까지의 상대량을 관측으로 낸다. (num_envs, 5)

    출력은 ``[ρ, cos α, sin α, cos θe, sin θe]`` 다.

    - ``ρ``  : 목표까지 거리 [m] (``max_range`` 로 상한)
    - ``α``  : 베이스 프레임에서 목표를 바라보는 방향 (bearing)
    - ``θe`` : 목표 yaw 와 추정 yaw 의 차이

    AMCL 특성 모사
    --------------
    단순 가우시안 노이즈만 넣으면 실로봇 격차를 못 메운다. AMCL 의 실제 성질은
    "느리게 갱신되고, 갱신 사이에는 값이 고정되고, 지연이 있고, 가끔 튄다" 이다.
    네 가지를 모두 재현한다.

    1. **갱신 주기** (``update_period``): 스캔 매칭 주기(보통 10~20 Hz)로만 값이 바뀐다.
       정책 주기가 더 빠르면 그 사이에는 이전 추정값이 유지된다(zero-order hold).
    2. **지연** (``latency``): 스캔 취득 -> 매칭 -> 퍼블리시까지의 시간.
       추정값은 "과거의 참값"에 노이즈를 얹은 것이다.
    3. **노이즈** (``pos_noise_std`` / ``yaw_noise_std``): 정상 상태 추정 오차.
    4. **점프** (``jump_prob``): 파티클 필터가 재수렴할 때의 불연속 도약.
       실로봇에서 정책이 가장 크게 흔들리는 순간이라 반드시 학습에 노출해야 한다.

    .. note::
        이 term 은 내부 상태(지연 버퍼, 갱신 타이머)를 들고 있으므로
        **관측 그룹 하나에만** 등록해야 한다. 두 그룹에 넣으면 한 스텝에
        두 번 호출되어 내부 시계가 두 배로 흐른다.
    """

    cfg: ObservationTermCfg

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._dt = env.step_dt
        # 지연 버퍼 길이는 생성 시점에 고정된다. latency 만 예외적으로 cfg.params 에서
        # 직접 읽는다 (버퍼 크기를 정해야 하므로 매 스텝 인자로 받을 수 없다).
        latency: float = cfg.params.get("latency", 0.06)
        self._hist_len = int(round(latency / self._dt)) + 1

        # (hist_len, num_envs, 3) = x, y, yaw 의 링 버퍼
        self._hist = torch.zeros(self._hist_len, self.num_envs, 3, device=self.device)
        self._head = 0
        # 현재 유지 중인 AMCL 추정값
        self._estimate = torch.zeros(self.num_envs, 3, device=self.device)
        self._time_since_update = torch.zeros(self.num_envs, device=self.device)
        # 리셋된 env 는 다음 호출에서 버퍼를 참값으로 채운다
        self._needs_fill = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # 여기서 참값을 읽지 않고 플래그만 세운다.
        # 리셋 이벤트와 관측 매니저 리셋의 호출 순서에 의존하지 않기 위해서다
        # (실제 채우기는 다음 __call__ 에서, 즉 모든 리셋이 끝난 뒤에 한다).
        if env_ids is None:
            self._needs_fill[:] = True
        else:
            self._needs_fill[env_ids] = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        pos_noise_std: float = 0.02,
        yaw_noise_std: float = 0.02,
        update_period: float = 0.05,
        latency: float = 0.06,  # noqa: ARG002 - 생성 시점에 버퍼 크기로 반영된다
        jump_prob: float = 0.0,
        jump_pos_std: float = 0.10,
        jump_yaw_std: float = 0.10,
        max_range: float = 12.0,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]

        # --- 참값 (월드 프레임). 목표도 월드라 env 원점 오프셋은 상쇄된다 ---
        truth = torch.cat([asset.data.root_pos_w[:, :2], asset.data.heading_w.unsqueeze(1)], dim=1)

        # --- 리셋된 env: 지연 버퍼 전체를 현재 참값으로 채운다 ---
        if self._needs_fill.any():
            fill = self._needs_fill
            self._hist[:, fill, :] = truth[fill]
            self._estimate[fill] = truth[fill]
            self._time_since_update[fill] = 0.0
            self._needs_fill[fill] = False

        # --- 지연 버퍼 갱신: 현재 참값을 쓰고, 가장 오래된 값을 읽는다 ---
        self._hist[self._head] = truth
        oldest = (self._head + 1) % self._hist_len
        delayed = self._hist[oldest]
        self._head = oldest

        # --- AMCL 갱신 타이밍 ---
        self._time_since_update += self._dt
        due = self._time_since_update >= update_period

        noise_scale = torch.tensor([pos_noise_std, pos_noise_std, yaw_noise_std], device=self.device)
        candidate = delayed + torch.randn_like(delayed) * noise_scale
        if jump_prob > 0.0:
            jump_scale = torch.tensor([jump_pos_std, jump_pos_std, jump_yaw_std], device=self.device)
            jump_mask = torch.rand(self.num_envs, 1, device=self.device) < jump_prob
            candidate = candidate + jump_mask * torch.randn_like(candidate) * jump_scale

        self._estimate = torch.where(due.unsqueeze(1), candidate, self._estimate)
        self._time_since_update = torch.where(due, torch.zeros_like(self._time_since_update), self._time_since_update)

        # --- 추정 pose + 목표 pose -> 베이스 프레임 상대량 ---
        command = env.command_manager.get_term(command_name)
        goal_xy = command.pos_command_w[:, :2]
        goal_yaw = command.heading_command_w

        delta = goal_xy - self._estimate[:, :2]
        est_yaw = self._estimate[:, 2]
        rho = torch.norm(delta, dim=1).clamp(max=max_range)
        bearing = wrap_to_pi(torch.atan2(delta[:, 1], delta[:, 0]) - est_yaw)
        yaw_error = wrap_to_pi(goal_yaw - est_yaw)

        return torch.stack(
            [rho, torch.cos(bearing), torch.sin(bearing), torch.cos(yaw_error), torch.sin(yaw_error)], dim=1
        )

    @property
    def estimate(self) -> torch.Tensor:
        """현재 AMCL 추정값 (x, y, yaw). 디버깅/로깅용. (num_envs, 3)"""
        return self._estimate


def base_twist_2d(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """베이스 프레임 트위스트 ``[v_x, ω_z]``. (num_envs, 2)

    실로봇에서는 휠 엔코더 오도메트리(``/odom`` 의 twist)에 대응한다.
    차동구동은 :math:`v_y = 0` 이고 롤/피치 각속도는 태스크와 무관하므로 2차원만 쓴다.

    엔코더 오도메트리의 트위스트는 오차가 매우 작으므로(적분되는 pose 와 달리
    누적 드리프트가 없다) 참값에 작은 노이즈만 얹으면 충분하다.
    노이즈는 호출부의 ``noise`` 설정으로 준다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.stack([asset.data.root_lin_vel_b[:, 0], asset.data.root_ang_vel_b[:, 2]], dim=1)


def localization_error(
    env: ManagerBasedRLEnv,
    obs_term_name: str = "amcl_goal",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """AMCL 추정 오차 ``[위치오차, |yaw 오차|]``. (num_envs, 2)

    학습에는 쓰지 않는다. AMCL 모사가 의도한 크기로 동작하는지 확인하기 위한
    진단용 함수다 (관측 그룹에 넣지 말고 ``check_diff_drive.py`` 처럼 직접 호출할 것).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    term = _find_obs_term(env, obs_term_name)
    est = term.estimate
    pos_err = torch.norm(est[:, :2] - asset.data.root_pos_w[:, :2], dim=1)
    yaw_err = torch.abs(wrap_to_pi(est[:, 2] - asset.data.heading_w))
    return torch.stack([pos_err, yaw_err], dim=1)


def _find_obs_term(env: ManagerBasedRLEnv, term_name: str) -> amcl_goal_polar:
    """이름으로 :class:`amcl_goal_polar` term 인스턴스를 찾는다.

    ``ObservationManager`` 에는 term 인스턴스를 돌려주는 공개 API 가 없어
    (``RewardManager.get_term_cfg`` 에 해당하는 것이 없다) 내부 목록을 참조한다.
    진단 경로에서만 쓰므로 감수한다.
    """
    manager = env.observation_manager
    for group_name, term_names in manager.active_terms.items():
        if term_name in term_names:
            term = manager._group_obs_term_cfgs[group_name][term_names.index(term_name)].func
            if not isinstance(term, amcl_goal_polar):
                raise TypeError(f"'{term_name}' 은 amcl_goal_polar term 이 아니다: {type(term)}")
            return term
    raise KeyError(f"관측 term '{term_name}' 을 찾지 못했다. 등록된 term: {manager.active_terms}")
