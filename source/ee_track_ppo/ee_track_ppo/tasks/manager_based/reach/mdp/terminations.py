# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""reach 태스크용 종료 term."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_tilt_exceeds(
    env: "ManagerBasedRLEnv",
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """차체가 ``limit_angle`` [rad] 이상 기울면 종료한다.

    왜 ``mdp.bad_orientation`` 을 못 쓰는가
    --------------------------------------
    Isaac Lab 의 :func:`bad_orientation` 은 **투영 중력 벡터**로 기울기를 잰다::

        torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle

    이 태스크는 ``sim.gravity = (0, 0, 0)`` 이다 (실기 컨트롤러가 팔 중력을 보상하므로
    정책 입장의 팔은 무중력이어야 한다 — `BASELINE_INTEGRATED.md` §6-C). 그러면
    ``projected_gravity_b`` 가 **영벡터**가 되어::

        acos(-0) = pi/2 = 1.5708 rad  >  limit 0.7   ->  항상 참

    **모든 에피소드가 1 스텝 만에 전복 판정으로 종료된다.** 2026-07-31 에 실제로
    이 조합으로 3000 iteration 을 돌렸고, 위치오차가 1005 mm 에 고정된 채 학습이
    전혀 진행되지 않았다 (``Episode_Termination/flipped`` 100%,
    ``mean_episode_length`` 1.0). 학습 곡선의 손실은 정상적으로 움직여서 로그만
    봐서는 알아채기 어렵다.

    그래서 중력에 의존하지 않고 **루트 자세에서 직접** 기울기를 잰다. 바디 z 축과
    월드 z 축 사이 각이 곧 기울기다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # 회전행렬의 3열이 바디 z 축을 월드에서 본 것이다.
    body_z = matrix_from_quat(asset.data.root_quat_w)[:, :, 2]
    return torch.acos(body_z[:, 2].clamp(-1.0, 1.0)) > limit_angle
