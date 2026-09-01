# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""월드 고정 목표(:class:`UniformPoseWorldCommand`) 전용 보상 함수.

Isaac Lab 내장 reach 보상(``isaaclab_tasks...reach.mdp.rewards``)은
커맨드가 base 프레임이라는 전제로 ``des_pos_w = root_pose ⊕ command`` 를 매번 다시 계산한다.
우리 커맨드는 이미 월드 좌표(``pose_command_w``)를 들고 있으므로 그걸 그대로 쓴다.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _desired_pose_w(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """커맨드 term 이 들고 있는 월드 좌표 목표 pose (num_envs, 7)."""
    return env.command_manager.get_term(command_name).pose_command_w


def position_command_error_w(env: "ManagerBasedRLEnv", command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """EE 위치 오차 L2 (월드 고정 목표 기준)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    des_pos_w = _desired_pose_w(env, command_name)[:, :3]
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh_w(
    env: "ManagerBasedRLEnv", std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """EE 위치 오차를 tanh 커널로 변환한 보상 (월드 고정 목표 기준).

    ``std`` 가 클수록 먼 거리에서도 기울기가 살아있다.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    des_pos_w = _desired_pose_w(env, command_name)[:, :3]
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error_w(env: "ManagerBasedRLEnv", command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """EE 자세 오차 (월드 고정 목표 기준)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    des_quat_w = _desired_pose_w(env, command_name)[:, 3:]
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def orientation_command_error_tanh(
    env: "ManagerBasedRLEnv", std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """EE 자세 오차를 tanh 커널로 변환한 보상 (고정형 팔용, base 프레임 커맨드 기준).

    IsaacLab 내장 reach 의 위치용 tanh 보상과 같은 형태를 자세에 적용한 것.
    오차가 0 에 가까울수록 1 에 수렴하는 **양의 보상**이라, 목표 근처에서 자세를 다듬을 유인이 생긴다.
    ``std`` 가 작을수록 좁고 날카로운 보상이 된다.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_w = quat_mul(asset.data.root_quat_w, command[:, 3:7])
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    error = quat_error_magnitude(curr_quat_w, des_quat_w)
    return 1 - torch.tanh(error / std)


def position_command_progress_w(
    env: "ManagerBasedRLEnv", command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """목표를 향해 다가가는 **속도**에 대한 보상 (m/s). 월드 고정 목표 기준.

    왜 필요한가
    -----------
    기존 보상은 전부 정적이다 — "목표에 가까이 **있으면**" 보상을 준다.
    "목표 쪽으로 **이동하면**" 주는 보상이 없다.

    고정형 팔은 관절만 움직이면 즉시 가까워지므로 이것으로 충분했다.
    하지만 모바일은 베이스가 수 초간 주행해야 비로소 가까워지고, 주행 중에는
    아무 이득이 없으면서 이탈 종료 위험만 진다. 그래서 정책이
    "베이스를 쓰지 않고 팔만 뻗는" 국소 최적해에 갇힌다.
    (v8 실측: it599 에서 position_error 0.56 m 로 수렴 후 900 iteration 동안 개선 없음.
     팔 리치 0.85 m / 목표 평균거리 1.2 m 로 계산한 "베이스 미사용" 예상치와 일치)

    구현
    ----
    EE 의 월드 속도를 목표 방향 단위벡터에 투영한다::

        progress = v_EE . (target - EE) / |target - EE|

    다가가면 양수, 멀어지면 음수다. 이는 거리의 시간미분(-d|e|/dt)과 같으므로
    potential-based shaping 에 해당한다 — **최적 정책을 바꾸지 않으면서**
    학습 신호만 조밀하게 만들어 주는, 이론적으로 안전한 형태다.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]  # type: ignore
    des_pos_w = _desired_pose_w(env, command_name)[:, :3]
    curr_pos_w = asset.data.body_pos_w[:, body_id]
    curr_vel_w = asset.data.body_lin_vel_w[:, body_id]

    to_target = des_pos_w - curr_pos_w
    distance = torch.norm(to_target, dim=1, keepdim=True).clamp(min=1e-4)
    direction = to_target / distance
    return torch.sum(curr_vel_w * direction, dim=1)


def orientation_command_error_tanh_w(
    env: "ManagerBasedRLEnv", std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """EE 자세 오차를 tanh 커널로 변환한 보상 (월드 고정 목표 기준).

    :func:`orientation_command_error_tanh` 의 월드 프레임 버전.
    모바일 매니퓰레이터처럼 root 가 움직이는 로봇에 쓴다.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    des_quat_w = _desired_pose_w(env, command_name)[:, 3:]
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    error = quat_error_magnitude(curr_quat_w, des_quat_w)
    return 1 - torch.tanh(error / std)


def pose_command_error_tanh_product(
    env: "ManagerBasedRLEnv",
    pos_std: float,
    ori_std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """위치와 자세 tanh 보상을 **곱**으로 결합한 보상 (고정형 팔용, base 프레임 커맨드 기준).

    왜 곱인가
    ---------
    위치 보상과 자세 보상을 **더하면** 한쪽을 포기하고 다른 쪽만 키워도 총합이 오른다.
    즉 트레이드오프가 정책에게 이득으로 남는다. (실측: ori_w04 런에서 자세 보상만 챙기고
    위치 오차가 0.06 -> 0.385 m 로 붕괴)

    곱으로 만들면 한쪽이 나쁠 때 전체가 0 에 가까워지므로 **둘 다 좋아야만** 보상을 받는다::

        reward = (1 - tanh(pos_err / pos_std)) * (1 - tanh(ori_err / ori_std))

    ``pos_std`` / ``ori_std`` 를 작게 잡을수록 "정밀하게 둘 다 맞췄을 때만" 주는 보상이 된다.
    넓은 값 하나 + 좁은 값 하나를 함께 쓰면(사다리 구성) 접근 단계와 정밀 단계 모두에서
    기울기가 살아 있다.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # 위치 오차 (목표는 base 프레임이므로 월드로 변환해서 비교)
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, command[:, :3])
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    pos_error = torch.norm(curr_pos_w - des_pos_w, dim=1)

    # 자세 오차
    des_quat_w = quat_mul(asset.data.root_quat_w, command[:, 3:7])
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    ori_error = quat_error_magnitude(curr_quat_w, des_quat_w)

    return (1 - torch.tanh(pos_error / pos_std)) * (1 - torch.tanh(ori_error / ori_std))



def base_ref_error_tanh(env: "ManagerBasedRLEnv", command_name: str, std: float) -> torch.Tensor:
    """차체가 플래너 참조를 얼마나 잘 따라가는가. ``tanh`` 사다리 한 칸.

    **가중치를 작게 둘 것.** 주 목적은 EE 추종이고, 차체 추종은 장애물 보장을 넘겨받기
    위한 부차 목표다. 크게 주면 EE 가 희생된다.

    ``std`` 는 "이 정도 오차에서 기울기가 최대" 를 뜻한다. 목표가 편차를 24 cm 에서
    10 cm 이하로 줄이는 것이므로 그 언저리로 잡는다.
    """
    term = env.command_manager.get_term(command_name)
    ref = getattr(term, "base_ref_w", None)
    if ref is None:
        return torch.zeros(env.num_envs, device=env.device)
    asset = env.scene["robot"]
    d = torch.norm(ref[:, :2] - asset.data.root_pos_w[:, :2], dim=1)
    return 1.0 - torch.tanh(d / std)


def base_ref_yaw_tanh(env: "ManagerBasedRLEnv", command_name: str, std: float) -> torch.Tensor:
    """차체 **방위**가 참조를 얼마나 따라가는가. (num_envs,)

    왜 따로 필요한가
    ----------------
    :func:`base_ref_error_tanh` 는 **위치 거리만** 본다. 그래서 v9 에서 위치는
    0.33 -> 0.025 m 로 잘 따라갔는데 **방위는 전혀 안 따라갔다**::

        base_ref_yaw_error   1.54 ~ 1.78 rad 로 학습 내내 그대로

    평균 |오차| 1.6 rad 은 균등랜덤 yaw 의 기대값 pi/2 = 1.57 과 거의 같다.
    즉 **리셋 랜덤값 그대로**이고 추종을 아예 안 한 것이다. 보상이 없었으니 당연하다.

    왜 방위가 중요한가
    ------------------
    차체 발자국이 0.80 x 0.55 로 정사각이 아니다. 같은 위치라도 방위가 다르면
    점유 영역이 달라져 플래너의 SDF 여유가 그대로 넘어오지 않는다.

    .. note::
       다만 **우리 팔은 뒤를 본다.** 플래너는 EE 가 앞에 있다고 가정하므로
       (``ee_forward_min``), 참조 방위를 그대로 따르면 팔이 불리해질 수 있다.
       가중치를 위치보다 작게 두고 결과를 본다.
    """
    term = env.command_manager.get_term(command_name)
    ref = getattr(term, "base_ref_w", None)
    if ref is None:
        return torch.zeros(env.num_envs, device=env.device)
    asset = env.scene["robot"]
    d = ref[:, 2] - asset.data.heading_w
    err = torch.atan2(torch.sin(d), torch.cos(d)).abs()
    return 1.0 - torch.tanh(err / std)


def base_action_rate_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """**차체 액션**의 변화율만 벌한다. (num_envs,)

    왜 팔과 나눠서 다루나
    ---------------------
    기존 ``action_rate_l2`` 는 8 개를 한 덩어리로 본다. 그러면 가중치를 올릴 때
    팔의 정밀 추종까지 같이 둔해진다. 실기에서 나온 문제는 **차체가 과하게 움직인다**
    는 것이고(2026-08-06, v2 실기), 팔은 문제가 없었다 — 정지유지 p-p 0.01 mm,
    순회 정착오차 2.8 mm.

    그래서 차체 두 성분만 따로 벌한다.

    .. note::
       차체 참조 추종([가-1], v6)으로도 같은 목적을 달성할 수 있지만, 그쪽은
       j6 -120 도 부근에서 액션 한계주기를 만들었다 (v5 진폭 0.000 -> v6 0.538).
       원인이 차체 참조로 확정됐으므로 이번에는 **직접 벌점** 쪽으로 간다.
    """
    return torch.sum(torch.square(env.action_manager.action[:, 6:8] - env.action_manager.prev_action[:, 6:8]), dim=1)


def wrist_unwind(env: "ManagerBasedRLEnv", command_name: str, asset_cfg: SceneEntityCfg,
                 err_std: float, joint_name: str = "j6") -> torch.Tensor:
    """EE 오차가 **작을 때만** 손목을 중앙으로 되돌린다. (num_envs,)

    왜 필요한가
    -----------
    실기 관찰 (로봇 PC, 2026-08-05): 목표 자세가 map 고정이라 차체 yaw 를 팔이
    되돌려야 하는데, 여유가 없으면 **j6 가 보상의 약 45 % 를 혼자 먹는다.**
    여유가 있으면 정책이 j1/j5 로 분산시킨다 (상관 +0.138 대 +0.981).

    **한 번 감기면 스스로 못 푼다.** 추종만 보상하므로 풀 이유가 없기 때문이다.

    왜 조건부인가
    -------------
    무조건 당기면 추종 정확도를 상시 깎는다. EE 오차가 클 때는 추종이 우선이고,
    **여유가 생겼을 때만** 되돌리게 한다::

        보상 = (오차가 작은 정도) x (j6 가 중앙에 가까운 정도)

    둘 다 ``tanh`` 로 눌러 한쪽이 0 이면 보상이 0 이 된다.

    .. note::
       조건부의 약점은 **계속 바쁘면 영영 못 푸는 것**이다. 운송처럼 오차가 늘 작은
       운용에서는 문제가 안 되지만, 그렇지 않으면 무조건 판으로 바꿔야 한다.
    """
    asset = env.scene[asset_cfg.name]
    idx = asset_cfg.joint_ids[0]
    lim = asset.data.joint_pos_limits[:, idx, :]
    mid = 0.5 * (lim[:, 0] + lim[:, 1])
    half = 0.5 * (lim[:, 1] - lim[:, 0]).clamp_min(1e-6)
    centered = 1.0 - torch.tanh(((asset.data.joint_pos[:, idx] - mid) / half).abs() / 0.3)

    # EE 오차는 **월드 프레임**에서 잰다. 커맨드 term 이 월드 목표를 들고 있으므로
    # 그걸 그대로 쓰면 된다 (이 모듈의 다른 보상들과 같은 방식).
    des_pos_w = _desired_pose_w(env, command_name)[:, :3]
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    err = torch.norm(curr_pos_w - des_pos_w, dim=1)
    near = 1.0 - torch.tanh(err / err_std)
    return near * centered


def base_ref_twist_tanh(env: "ManagerBasedRLEnv", command_name: str,
                        v_std: float, w_std: float) -> torch.Tensor:
    """차체 **속도**가 참조 속도를 따라가는가 (v, omega). (num_envs,)

    왜 필요한가
    -----------
    실기 v12 관찰 (2026-08-06)::

        "베이스 속력이 일정하지 않고 빨랐다 느렸다 함"
        "베이스 MPC 레퍼런스를 추종하는 능력이 부족함"

    기존 보상은 :func:`base_ref_error_tanh` (위치) 와 :func:`base_ref_yaw_tanh` (방위)
    **둘 다 자세만** 본다. 속도는 관측(``base_ref_twist``)에 들어가는데 보상에는 없었다.
    그래서 정책은 "결국 그 자리에 있기만 하면 된다" 를 배웠고, 가는 방식은 제멋대로가
    됐다 — 늦었으면 빨리 가고 앞섰으면 서는 식이다.

    참조 속도를 직접 따라가면 **위치 추종과 속도 평활이 같이 해결된다.**
    참조 자체가 MPC 가 푼 매끄러운 프로파일이기 때문이다.

    .. note::
       ``base_ang_vel`` 같은 **감쇠 항과 다르다.** 감쇠는 "덜 움직여라" 인데 이것은
       "참조만큼 움직여라" 다. 감쇠만 걸면 참조가 빠를 때도 굼떠진다.
    """
    term = env.command_manager.get_term(command_name)
    ref = getattr(term, "base_ref_twist", None)
    if ref is None:
        return torch.zeros(env.num_envs, device=env.device)
    d = env.scene["robot"].data
    ev = torch.abs(d.root_lin_vel_b[:, 0] - ref[:, 0])
    ew = torch.abs(d.root_ang_vel_b[:, 2] - ref[:, 1])
    return 0.5 * (1.0 - torch.tanh(ev / v_std)) + 0.5 * (1.0 - torch.tanh(ew / w_std))


def base_ref_error_band(env: "ManagerBasedRLEnv", command_name: str,
                        band: float, std: float) -> torch.Tensor:
    """차체가 참조에서 **밴드 밖으로** 나갈 때만 벌한다 (불감대 있는 추종).

    왜 밴드인가
    -----------
    참조를 정확히 따라가게 하면 **자세를 맞출 자유도가 사라진다.** 실측::

        차체 가중치        차체 편차    EE 자세오차
        v6  0.15          0.488 m      3.3 도
        v12 0.35          0.240        10.5
        v15 0.50 + 속도   0.062        53.4      <- 차체는 13 배 좋아지고 자세는 16 배 나빠짐

    차체가 참조에서 벗어나는 것은 **자세를 맞추려는 시도**였다. 그것을 막으니 자세가
    무너졌다. 차동구동이라 옆으로 못 가므로, 팔이 못 만드는 자세는 차체가 자리를
    옮겨 만들어야 한다.

    그리고 애초에 필요한 것은 **정확한 추종이 아니라 한계**다. 플래너가 보장하는 것은
    장애물 여유(``base_margin``)이고, 그건 "이 반경 안에 있으면 된다" 는 조건이다.
    밴드 안에서는 보상을 만점으로 주어 정책이 자유롭게 쓰게 한다.

    Args:
        band: 이 거리 안에서는 벌하지 않는다 [m]. 플래너 여유에서 온다.
        std: 밴드 밖에서 얼마나 빨리 벌점이 커지는가 [m].
    """
    term = env.command_manager.get_term(command_name)
    ref = getattr(term, "base_ref_w", None)
    if ref is None:
        return torch.zeros(env.num_envs, device=env.device)
    d = torch.norm(ref[:, :2] - env.scene["robot"].data.root_pos_w[:, :2], dim=1)
    return 1.0 - torch.tanh((d - band).clamp_min(0.0) / std)


def base_ref_yaw_band(env: "ManagerBasedRLEnv", command_name: str,
                      band: float, std: float) -> torch.Tensor:
    """차체 **방위**도 같은 이유로 불감대를 준다.

    ``band`` [rad] 안에서는 만점. 방위는 위치보다 더 여유를 줘도 되는데, 차동구동은
    제자리 회전이 가능해서 언제든 되돌릴 수 있기 때문이다.
    """
    term = env.command_manager.get_term(command_name)
    ref = getattr(term, "base_ref_w", None)
    if ref is None:
        return torch.zeros(env.num_envs, device=env.device)
    # ``_wrap_pi`` 는 commands.py 에 있고 여기엔 없다. 같은 파일의 base_ref_yaw_tanh 과
    # 같은 방식으로 감는다 — 남의 모듈 함수를 끌어오면 두 벌이 어긋날 위험이 있다.
    d = ref[:, 2] - env.scene["robot"].data.heading_w
    e = torch.atan2(torch.sin(d), torch.cos(d)).abs()
    return 1.0 - torch.tanh((e - band).clamp_min(0.0) / std)


def base_ang_accel_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """차체 **각가속도**를 벌한다 — 참조 없이 지그재그만 직접 잡는다. (num_envs,)

    왜 이 항인가
    ------------
    지그재그는 차체가 **좌우로 방향을 자주 바꾸는 것**이다. 각속도의 크기가 아니라
    **부호가 자주 뒤집히는 것**이 문제이므로, 각속도 자체(``base_ang_vel``) 를 벌하면
    필요한 선회까지 굼떠진다.

    각가속도는 "방향을 바꾸는 행위" 만 벌한다. 일정한 속도로 도는 것은 공짜다::

        일정 선회      omega 일정  ->  각가속도 0     벌점 없음
        지그재그       omega 부호 반전  ->  각가속도 큼   벌점

    .. note::
       MPC 차체 참조를 쓰지 않는 판(v2 계열) 을 위한 항이다. 참조가 있으면
       :func:`base_ref_twist_tanh` 가 같은 일을 더 직접적으로 하지만, 그것은
       **차체 위치까지 구속**해 EE 정밀도를 크게 희생시킨다 (실측: EE 8.6 -> 41.7 mm).
    """
    d = env.scene["robot"].data
    prev = getattr(env, "_prev_base_w", None)
    w = d.root_ang_vel_b[:, 2]
    if prev is None or prev.shape != w.shape:
        env._prev_base_w = w.clone()
        return torch.zeros_like(w)
    # **step_dt 로 나누지 않는다.** 나누면 (dt=0.033) 값이 30 배가 되고 제곱하면
    # 900 배라, 다른 항을 압도한다 (실측: 다른 항이 0.1 대일 때 이 항만 -1.02).
    # 여기서 재는 것은 "한 스텝에 각속도가 얼마나 바뀌었나" 이고, 그것으로 충분하다.
    a = w - prev
    env._prev_base_w = w.clone()
    return torch.square(a)
