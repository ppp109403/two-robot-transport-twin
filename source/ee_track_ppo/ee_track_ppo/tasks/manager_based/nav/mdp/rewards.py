# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMR 목표 pose 도달 태스크용 보상 함수.

보상은 **참값(ground truth)** 을 쓴다. 관측은 AMCL 추정값을 쓰지만 보상은 다르다.
보상은 학습 신호이지 정책 입력이 아니므로, 노이즈를 얹으면 학습 신호만 나빠진다.
"노이즈 있는 추정으로 참값 목표를 달성하라"가 정확한 문제 정의다.

차동구동 pose 도달의 핵심 난점: 두 목표가 서로 충돌한다
--------------------------------------------------------
목표는 (1) 목표 위치로 이동, (2) 목표 방향 정렬 두 가지다.
그런데 차동구동은 횡방향으로 움직일 수 없어서, 이동하려면 **먼저 목표를 향해
몸을 돌려야** 한다. 즉 주행 중에 요구되는 방향(목표를 바라보는 방향)과
최종적으로 요구되는 방향(목표 yaw)이 다르다.

이 둘을 구분하지 않고 "목표 yaw 정렬" 하나만 보상하면 전형적인 병리가 나온다.
목표 yaw 를 유지하려는 힘과 목표 쪽으로 가려는 힘이 상충해서,
로봇이 목표 주변을 원주 운동하거나 제자리에서 방향만 맞추고 멈춘다.

그래서 :func:`heading_alignment` 은 거리에 따라 두 기준을 매끄럽게 섞는다.
멀면 "목표를 바라보기", 가까우면 "목표 yaw 맞추기" 다.

성공 시 종료를 넣지 않은 이유
-----------------------------
스텝 보상이 양수인 설계(생존 보너스 포함)에서 성공 시 에피소드를 끝내면
**성공이 손해**가 된다. 남은 스텝의 보상을 포기하는 것이기 때문이다.
ridgeback 태스크에서 반대 방향의 같은 병리(일부러 이탈 종료)를 실측했다.

여기서는 성공해도 종료하지 않고, 목표를 주기적으로 재샘플한다.
그러면 "도달 후 유지"까지 학습되는데, 도킹·정위치 정지가 필요한 실제 용도에
정확히 부합한다.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _goal_w(env: ManagerBasedRLEnv, command_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """커맨드 term 이 들고 있는 월드 좌표 목표 (xy, yaw)."""
    command = env.command_manager.get_term(command_name)
    return command.pos_command_w[:, :2], command.heading_command_w


def _errors(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(거리, bearing 오차, yaw 오차) 를 한 번에 계산한다."""
    asset: Articulation = env.scene[asset_cfg.name]
    goal_xy, goal_yaw = _goal_w(env, command_name)

    delta = goal_xy - asset.data.root_pos_w[:, :2]
    heading = asset.data.heading_w
    distance = torch.norm(delta, dim=1)
    bearing = wrap_to_pi(torch.atan2(delta[:, 1], delta[:, 0]) - heading)
    yaw_error = wrap_to_pi(goal_yaw - heading)
    return distance, bearing, yaw_error


def position_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """거리 오차를 tanh 커널로 변환한 양의 보상. ``1 - tanh(ρ / std)``

    ``std`` 를 크게 잡은 항(coarse)과 작게 잡은 항(fine)을 함께 쓰면
    먼 거리와 마지막 수 cm 양쪽에서 기울기가 살아 있다.
    """
    distance, _, _ = _errors(env, command_name, asset_cfg)
    return 1.0 - torch.tanh(distance / std)


def position_progress(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """목표를 향해 다가가는 속도 [m/s].

    거리의 시간미분(-dρ/dt)과 같으므로 potential-based shaping 이다.
    최적 정책을 바꾸지 않으면서 주행 구간의 학습 신호를 조밀하게 만든다.

    정적 보상("가까이 있으면 보상")만으로는 수 초간 주행해야 비로소 가까워지는
    모바일 로봇에서 신호가 너무 희박하다. ridgeback 태스크에서 이 항을 넣기 전까지
    900 iteration 동안 정체했다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    goal_xy, _ = _goal_w(env, command_name)

    delta = goal_xy - asset.data.root_pos_w[:, :2]
    distance = torch.norm(delta, dim=1, keepdim=True).clamp(min=1e-4)
    direction = delta / distance
    return torch.sum(asset.data.root_lin_vel_w[:, :2] * direction, dim=1)


def heading_alignment(
    env: ManagerBasedRLEnv,
    gate_dist: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """거리에 따라 기준이 바뀌는 방향 정렬 보상. 값 범위 [0, 1].

    .. math::

        w = \\tanh(\\rho / d_{gate}), \\qquad
        r = w \\cdot \\frac{1 + \\cos\\alpha}{2} + (1 - w) \\cdot \\frac{1 + \\cos\\theta_e}{2}

    - 멀 때(:math:`w \\to 1`): 목표를 **바라보도록** 정렬 (bearing :math:`\\alpha`)
    - 가까울 때(:math:`w \\to 0`): 목표 **yaw 에** 정렬 (:math:`\\theta_e`)

    거리로 부드럽게 섞기 때문에 전환 지점에서 보상이 튀지 않는다.
    하드 스위치(``if ρ > d``)를 쓰면 경계에서 보상이 불연속이 되고,
    정책이 경계 근처를 진동하며 두 보상을 번갈아 먹는 현상이 생긴다.

    ``cos`` 을 그대로 쓰지 않고 ``(1 + cos)/2`` 로 [0, 1] 로 옮기는 이유는
    스텝 보상 총합을 양수로 유지하기 위해서다. 스텝 보상이 음수가 되면
    정책이 에피소드를 일찍 끝내는 것(이탈/전복)에서 이득을 본다.
    """
    distance, bearing, yaw_error = _errors(env, command_name, asset_cfg)
    weight = torch.tanh(distance / gate_dist)
    face_goal = 0.5 * (1.0 + torch.cos(bearing))
    match_goal_yaw = 0.5 * (1.0 + torch.cos(yaw_error))
    return weight * face_goal + (1.0 - weight) * match_goal_yaw


def goal_pose_tanh_product(
    env: ManagerBasedRLEnv,
    pos_std: float,
    yaw_std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """위치와 방향 tanh 보상의 **곱**. ``(1 - tanh(ρ/pos_std)) * (1 - tanh(|θe|/yaw_std))``

    더하기가 아니라 곱인 이유: 합으로 두면 한쪽을 포기하고 다른 쪽만 키워도
    총합이 오르기 때문에 트레이드오프가 정책에게 이득으로 남는다.
    곱은 **둘 다 좋아야만** 값이 커지므로 "정확한 pose 도달"을 직접 표현한다.
    (FR3 reach 실험에서 합 형태가 자세만 챙기고 위치를 0.06 -> 0.385 m 로 붕괴시킴)
    """
    distance, _, yaw_error = _errors(env, command_name, asset_cfg)
    pos_term = 1.0 - torch.tanh(distance / pos_std)
    yaw_term = 1.0 - torch.tanh(torch.abs(yaw_error) / yaw_std)
    return pos_term * yaw_term


def at_goal(
    env: ManagerBasedRLEnv,
    pos_tol: float,
    yaw_tol: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """목표 허용오차 안에 들어와 있으면 1, 아니면 0.

    성공 판정을 그대로 보상으로 쓴다. 종료시키지 않으므로 "머물러 있는 동안"
    계속 받는 보상이 되어 정위치 유지까지 학습된다.
    성공률 로깅도 이 항의 평균값으로 읽을 수 있다.
    """
    distance, _, yaw_error = _errors(env, command_name, asset_cfg)
    return ((distance < pos_tol) & (torch.abs(yaw_error) < yaw_tol)).float()


def ang_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """요레이트 제곱 페널티. 불필요한 선회와 도달 후 떨림을 억제한다."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])


def reverse_motion_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """후진 속도에만 걸리는 제곱 페널티.

    실제 AMR 은 후방 센서가 없거나 시야가 좁아 후진 주행을 꺼린다.
    완전히 금지하면 좁은 공간에서 자세를 맞출 방법이 없어지므로,
    금지하지 않고 비용만 부과한다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 0].clamp(max=0.0))
