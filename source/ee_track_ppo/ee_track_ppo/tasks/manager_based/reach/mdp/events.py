# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""sim-to-real 용 이벤트(도메인 랜덤화) term."""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING, Literal

from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_gains_iso_damping(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    stiffness_scale_range: tuple[float, float],
    distribution: Literal["uniform", "log_uniform"] = "log_uniform",
) -> None:
    r"""강성을 랜덤화하되 **감쇠비를 유지**하도록 감쇠를 sqrt 로 함께 스케일한다.

    왜 기본 ``randomize_actuator_gains`` 를 안 쓰는가
    ------------------------------------------------
    기본 term 은 stiffness 와 damping 을 **독립**으로 뽑는다. 그러면 감쇠비가 망가진다.

    .. math::

        \zeta = \frac{k_d}{2\sqrt{k_p I}}
        \quad\Rightarrow\quad
        k_p \to s\,k_p,\; k_d \to s\,k_d \;\;\text{이면}\;\; \zeta \to \sqrt{s}\,\zeta_0

    ``s = 0.4`` 이면 :math:`\zeta = 0.63\,\zeta_0` 로 **부족감쇠**가 되어 시뮬 관절 자체가
    링잉을 시작한다. 이건 실기에 **존재하지 않는 현상**이다. 없는 것을 견디도록
    학습시키면 정책만 과보수적으로 만든다.

    그래서 스케일 :math:`s` 를 하나만 뽑아::

        k_p -> s * k_p
        k_d -> sqrt(s) * k_d      ->   zeta 유지

    로 걸어서, 응답의 **속도**(고유진동수 :math:`\omega \propto \sqrt{k_p}`)만 흔들고
    **형상**(감쇠비)은 고정한다.

    alpha 환산 (kp 400 / kd 40 기준 실측 alpha = 0.23)::

        kp 0.4 배  ->  alpha 약 0.15
        kp 1.0 배  ->  alpha 0.23        <- 측정값
        kp 2.5 배  ->  alpha 약 0.35

    구현 참고
    ---------
    ``asset_cfg`` 는 어떤 asset 인지만 고르는 데 쓰고, 스케일은 그 asset 의 **모든
    액추에이터 그룹 전체**에 걸린다. FR3 는 액추에이터 그룹이 ``arm`` 하나뿐이라
    (= j1~j6 전부) 이 단순화가 안전하다. 관절을 골라서 랜덤화해야 하는 로봇에
    이 함수를 재사용하려면 인덱싱을 추가해야 한다.

    암시적(implicit) 액추에이터의 gain 은 CPU 텐서를 거쳐 sim 에 써야 하므로
    이 term 은 ``mode="startup"`` 으로만 쓴다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    lo, hi = stiffness_scale_range
    if lo <= 0.0 or hi < lo:
        raise ValueError(f"stiffness_scale_range 가 잘못됐다: {stiffness_scale_range}")

    # env 당 스케일 하나. (n, 1) 로 두면 그 env 의 모든 관절에 같은 배율이 걸린다.
    n = len(env_ids)
    if distribution == "log_uniform":
        s = torch.empty(n, 1, device=asset.device).uniform_(math.log(lo), math.log(hi)).exp()
    else:
        s = torch.empty(n, 1, device=asset.device).uniform_(lo, hi)
    s_sqrt = s.sqrt()

    for actuator in asset.actuators.values():
        jids = actuator.joint_indices
        # 기본값에서 출발한다 (현재값에 곱하면 배율이 누적된다)
        kp = asset.data.default_joint_stiffness[env_ids][:, jids].clone() * s
        kd = asset.data.default_joint_damping[env_ids][:, jids].clone() * s_sqrt

        actuator.stiffness[env_ids] = kp
        actuator.damping[env_ids] = kd
        if isinstance(actuator, ImplicitActuator):
            asset.write_joint_stiffness_to_sim(kp, joint_ids=jids, env_ids=env_ids)
            asset.write_joint_damping_to_sim(kd, joint_ids=jids, env_ids=env_ids)


def gravity_on_bodies_only(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
) -> None:
    """전역 중력이 0 인 씬에서 **지정한 링크에만** 중력을 되살린다 (``mode="reset"``).

    왜 이 방향인가
    --------------
    실기 FAIRINO 컨트롤러는 **자체 중력보상**을 한다. 그래서 정책 입장의 팔은 중력이
    없는 것처럼 거동하는 것이 맞다. 팔 단독 태스크는 v3 부터 ``sim.gravity=0`` 으로
    이걸 구현했고 실기 편향 12.1 mm 가 그때 해소됐다.

    통합 모델은 차체가 바퀴로 바닥을 딛어야 주행이 되므로 차체에는 중력이 필요하다.
    선택지가 둘인데 **오차가 어디에 남느냐**가 다르다::

        중력 -9.81 + 팔에 상향 외력   팔에 잔차가 남는다  <- 실측 j2 3.35 -> 2.49도, 26% 만 상쇄
        중력 0 + 차체에 하향 외력      차체에 잔차가 남는다  <- 이쪽을 쓴다

    외력은 링크 원점에 걸리는데 중력은 질량중심에 걸리므로 둘의 차이가 잔여 토크를
    만든다. 팔에 남으면 관절 편향이 되어 그대로 실기 오차가 되지만, 차체에 남으면
    바퀴에 눌린 강체라 무해하다. 그리고 이 배치는 팔 쪽이 **검증된 팔 단독 조건과
    정확히 동일**해진다는 장점이 있다.

    .. warning::
       **툴 하중이 실기 컨트롤러에 등록되어 있어야 이 모델이 맞다.** FR3 웹앱의 툴/하중에
       그리퍼 질량(약 1.2 kg)이 들어 있지 않으면 실기만 처지고 시뮬은 안 처져서
       반대 방향의 격차가 생긴다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ids = asset_cfg.body_ids
    if ids is None or isinstance(ids, slice):
        raise ValueError("gravity_on_bodies_only: body_names 로 대상 링크를 지정해야 한다.")
    ids = list(ids)

    g = float(torch.tensor(env.sim.cfg.gravity).norm())
    if g > 1e-6:
        raise ValueError(
            "gravity_on_bodies_only 는 sim.gravity=(0,0,0) 전제다. 현재 %s"
            % (env.sim.cfg.gravity,)
        )
    g = 9.81

    if env_ids is None:
        env_ids = torch.arange(asset.num_instances, device=asset.device)

    masses = asset.root_physx_view.get_masses().to(asset.device)[env_ids][:, ids]
    forces = torch.zeros(len(env_ids), len(ids), 3, device=asset.device)
    forces[..., 2] = -masses * g          # 월드 -z 로 무게만큼 누른다
    asset.set_external_force_and_torque(
        forces=forces,
        torques=torch.zeros_like(forces),
        body_ids=ids,
        env_ids=env_ids,
        is_global=True,
    )
