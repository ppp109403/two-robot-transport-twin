# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMR 목표 pose 도달 태스크용 종료 조건.

성공 종료는 의도적으로 제공하지 않는다. 이유는
:mod:`~ee_track_ppo.tasks.manager_based.nav.mdp.rewards` 모듈 설명 참고.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_out_of_env_bounds(
    env: ManagerBasedRLEnv, max_dist: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """env 원점에서 xy 거리가 ``max_dist`` 를 넘으면 종료.

    학습 초기에는 액션 노이즈로 인한 랜덤워크로 로봇이 멀리 배회한다.
    그대로 두면 이웃 env 영역을 침범해 물리 폭주의 원인이 되므로 잘라낸다.
    (ridgeback 태스크에서 실제로 겪었다)

    ``max_dist`` 는 목표 샘플링 범위보다 충분히 크게, ``env_spacing`` 의 절반보다는
    작게 잡아야 한다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    offset = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return torch.norm(offset, dim=1) > max_dist
