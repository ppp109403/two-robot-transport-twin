# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""EE pose 추종 태스크용 관측 함수.

왜 필요한가
-----------
Isaac Lab 내장 reach 태스크의 관측은 ``joint_pos_rel``, ``joint_vel_rel``,
``generated_commands``(목표 pose), ``last_action`` 뿐이다.
즉 **정책이 자기 엔드이펙터가 지금 어디에 있는지를 직접 보지 못한다.**
관절각으로부터 순기구학(FK)을 신경망 안에서 암묵적으로 학습해야 하며,
그 근사 오차가 "위치와 자세를 동시에 정밀하게" 맞추는 데 병목이 된다.

실측 근거 (franka_reach 실험 4회):
  v0 위치 0.060 / 자세 0.196   (자세를 포기)
  v1 위치 0.385 / 자세 0.024   (위치를 포기)
  v2 위치 0.107 / 자세 0.057   (가장 균형)
  v3 위치 0.220 / 자세 0.043   (보상 재배분 실패)
보상 배분을 어떻게 바꿔도 한쪽을 희생하는 패턴이 반복되었다.
=> 보상 문제가 아니라 관측(관측 가능성) 문제로 판단.

여기서는 목표와 현재 EE 사이의 **오차 벡터를 직접** 관측에 넣어준다.
오차는 원리상 (관절각 + 목표)로부터 유도 가능하지만, 그 유도(FK + 쿼터니언 연산)를
정책이 배우지 않아도 되게 만들어 주는 것이 핵심이다.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    axis_angle_from_quat,
    matrix_from_quat,
    quat_conjugate,
    quat_mul,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


QUAT_HEMISPHERE_REF = (0.0, 0.0, 1.0, 0.0)
"""반구 정규화 기준 쿼터니언 (wxyz) = roll 0, pitch pi, yaw 0.

**배포 쪽(``fr3_rl/policy_runner.py``)과 반드시 같은 값이어야 한다.**

왜 성분 우선순위(w -> x -> y -> z) 가 아니라 기준 자세와의 내적인가
-------------------------------------------------------------------
이 태스크의 커맨드는 ``pitch = pi`` 고정이라 목표 쿼터니언이 전부 **w = 0** 인
특이면 위에만 있다. 실기에서 이 w 는 +-1e-17 수준의 부동소수점 잡음이 되고,
w 부호로 반구를 정하면 목표와 현재가 **서로 다른 반구로 갈라진다**.

이 사고가 실제로 났다. 오차항은 axis-angle 이라 정상적으로 0 을 가리켜서
**어떤 오차 지표로도 안 잡혔고**, 정책이 학습 중 본 적 없는 부호의 입력을 받아
j6 액션이 1.849 로 포화하면서 명령이 기본자세에서 53 도 튀었다.
(실기 수정 후: 명령 편차 52.98 deg -> 4.31 deg)

학습 커맨드는 ``q = (0, -sin(psi/2), cos(psi/2), 0)`` 이라 이 기준과의 내적이
정확히 ``cos(psi/2)`` 다. yaw 범위 [-pi, pi] 전 구간에서 ``cos(psi/2) >= 0.0008`` 이므로
잡음(1e-17)보다 훨씬 커서 부호가 결정적으로 정해진다.

``commands`` 쪽 ``make_quat_unique`` 는 False 로 둔다 - 커맨드는 이미 y >= 0 반구에만
생성되므로 이 기준과 일치한다.
"""


def _hemisphere_normalize(quat: torch.Tensor) -> torch.Tensor:
    """쿼터니언을 기준 자세와 같은 반구로 보낸다. 자세한 이유는 :data:`QUAT_HEMISPHERE_REF`."""
    ref = torch.tensor(QUAT_HEMISPHERE_REF, device=quat.device, dtype=quat.dtype)
    sign = torch.where((quat @ ref) < 0.0, -1.0, 1.0).unsqueeze(-1)
    return quat * sign


def ee_pose_b(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """현재 EE pose 를 로봇 root 프레임으로 표현한 값. (num_envs, 7) = xyz + quat(wxyz).

    월드 절대좌표가 아니라 root 프레임이라야 env 마다 원점이 달라도 일반화된다.

    쿼터니언은 :func:`_hemisphere_normalize` 로 기준 반구에 맞춘다.
    ``subtract_frame_transforms`` 는 부호를 정규화하지 않는 raw 값을 내놓는데,
    배포 쪽은 정규화하므로 그대로 두면 **시뮬과 실기의 규약이 원리적으로 어긋난다**.
    지금까지 문제가 없었던 것은 PhysX 가 일관된 반구를 내놓았기 때문일 뿐이고,
    보장된 동작이 아니다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]  # type: ignore
    pos_b, quat_b = subtract_frame_transforms(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        asset.data.body_pos_w[:, body_id],
        asset.data.body_quat_w[:, body_id],
    )
    return torch.cat([pos_b, _hemisphere_normalize(quat_b)], dim=-1)


def ee_pose_error_b(env: "ManagerBasedRLEnv", command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """목표 pose 와 현재 EE pose 의 오차. (num_envs, 6) = 위치오차 3 + 자세오차(axis-angle) 3.

    - 위치 오차: root 프레임에서 (목표 - 현재) 벡터. 크기와 방향을 그대로 담는다.
    - 자세 오차: 목표 회전과 현재 회전의 상대 회전을 axis-angle 3벡터로 변환.
      쿼터니언(4차원, 부호 이중성 있음)보다 학습에 쓰기 좋다. 크기가 곧 회전 오차(rad)다.

    커맨드가 root 프레임 기준(``UniformPoseCommand``)이든 월드 고정 목표를 root 프레임으로
    변환해 주는 것(``UniformPoseWorldCommand``)이든, 둘 다 ``command`` 는 root 프레임이므로
    고정형 팔과 모바일 매니퓰레이터 양쪽에 그대로 쓸 수 있다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]  # type: ignore
    command = env.command_manager.get_command(command_name)

    curr_pos_b, curr_quat_b = subtract_frame_transforms(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        asset.data.body_pos_w[:, body_id],
        asset.data.body_quat_w[:, body_id],
    )
    pos_error_b = command[:, :3] - curr_pos_b
    quat_error = quat_mul(command[:, 3:7], quat_conjugate(curr_quat_b))
    ori_error_b = axis_angle_from_quat(quat_error)
    return torch.cat([pos_error_b, ori_error_b], dim=-1)


##
# 6D 회전 표현 (RETRAIN_V2 §1-C)
##


def rot6d_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """쿼터니언(wxyz) -> 회전행렬의 **첫 두 열**. (N, 4) -> (N, 6)

    왜 쿼터니언을 버리는가
    ----------------------
    이 태스크의 커맨드는 ``pitch = pi`` 로 고정이라 목표 쿼터니언이 전부 **w = 0** 인
    특이면 위에만 존재한다. 실기에서는 이 w 가 +-1e-17 수준의 부동소수점 잡음이 되고,
    부호 정규화(q 와 -q 는 같은 회전)가 목표와 현재를 **서로 다른 반구로 보내는** 사고가 났다.

    이 사고가 무서운 이유는 **어떤 오차 지표로도 잡히지 않았다**는 점이다.
    오차항은 axis-angle 이라 정상적으로 0 을 가리켰다. 그런데 정책은 학습 중 본 적 없는
    부호의 입력을 받아 j6 액션이 1.849 로 포화했고 명령이 기본자세에서 53 도 튀었다.

    회전행렬은 SO(3) -> R^9 의 **일대일** 사상이라 부호 이중성이 원천적으로 없다.
    첫 두 열만 쓰는 이유는 세 번째 열이 외적으로 복원 가능해 중복이기 때문이다
    (Zhou et al. 2019, "On the Continuity of Rotation Representations").

    레이아웃(배포 쪽과 반드시 일치시켜야 함)::

        [ R00, R10, R20,   R01, R11, R21 ]
          <-- 1열 -->      <-- 2열 -->
    """
    rot = matrix_from_quat(quat)  # (N, 3, 3)
    return torch.cat([rot[:, :, 0], rot[:, :, 1]], dim=-1)


def ee_pose_6d_b(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """현재 EE pose (root 프레임), 회전은 6D. (num_envs, 9) = xyz 3 + rot6d 6.

    :func:`ee_pose_b` 의 쿼터니언 버전을 대체한다. 자세한 이유는 :func:`rot6d_from_quat`.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]  # type: ignore
    pos_b, quat_b = subtract_frame_transforms(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        asset.data.body_pos_w[:, body_id],
        asset.data.body_quat_w[:, body_id],
    )
    return torch.cat([pos_b, rot6d_from_quat(quat_b)], dim=-1)


def pose_command_6d(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """목표 pose 커맨드, 회전은 6D. (num_envs, 9) = xyz 3 + rot6d 6.

    Isaac Lab 기본 ``generated_commands`` (7차원, pos + quat) 를 대체한다.
    **부호 사고가 실제로 난 곳이 바로 이 항이다** (pitch=pi -> w=0 특이면).
    """
    command = env.command_manager.get_command(command_name)
    return torch.cat([command[:, :3], rot6d_from_quat(command[:, 3:7])], dim=-1)


def ee_target_vel_b(env: "ManagerBasedRLEnv", command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """목표 pose 의 **속도**를 로봇 root 프레임으로 표현한 값. (num_envs, 6) = 선속도 3 + 각속도 3.

    왜 필요한가 (피드포워드)
    ------------------------
    오차만 보는 정책은 순수 피드백 제어기라, 움직이는 기준을 따라갈 때
    ``추종 오차 ~= 궤적 속도 x 시정수`` 만큼 구조적으로 뒤처진다.
    (AMR+FR3 실측 시정수 tau ~= 0.5 s -> 30 cm/s 궤적에서 15 cm 지연)

    목표가 **어느 방향으로 얼마나 빨리 가는지**를 관측에 넣어 주면 정책이
    "목표가 갈 곳으로 미리" 움직일 수 있어 이 지연이 크게 줄어든다.

    커맨드 term 이 ``target_lin_vel_w`` / ``target_ang_vel_w`` 를 제공해야 한다
    (:class:`~ee_track_ppo.tasks.manager_based.reach.mdp.MovingPoseWorldCommand`).
    정지 목표 커맨드에는 이 속성이 없으므로 0 을 돌려준다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    term = env.command_manager.get_term(command_name)
    lin_w = getattr(term, "target_lin_vel_w", None)
    if lin_w is None:
        return torch.zeros(env.num_envs, 6, device=env.device)
    ang_w = term.target_ang_vel_w
    # 월드 -> root 프레임 (회전만 적용. 속도는 위치 오프셋과 무관)
    from isaaclab.utils.math import quat_apply_inverse

    q = asset.data.root_quat_w
    return torch.cat([quat_apply_inverse(q, lin_w), quat_apply_inverse(q, ang_w)], dim=-1)


def target_lin_vel_b(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """목표의 **선속도**를 커맨드 프레임(=root/base) 그대로 돌려준다. (num_envs, 3)

    왜 ``ee_target_vel_b`` 를 안 쓰나
    ---------------------------------
    그 함수는 월드 프레임 속도를 받아 root 프레임으로 변환한다. 그런데
    :class:`CylindricalRampCommand` 는 **처음부터 base 프레임**으로 속도를 만들므로
    변환을 또 하면 틀린다 (FR3 는 base 가 z축 180도 돌아 있어 부호가 뒤집힌다).

    왜 필요한가 (피드포워드)
    ------------------------
    오차만 보는 정책은 순수 피드백이라 움직이는 기준을 따라갈 때 ``속도 x 시정수`` 만큼
    구조적으로 뒤처진다 (실기 시정수 0.160 s). 목표가 **어느 방향으로 얼마나 빨리 가는지**
    를 관측에 넣어 주면 정책이 미리 움직일 수 있다.

    램프 모드가 아닌 커맨드에는 이 속성이 없으므로 0 을 돌려준다.
    """
    term = env.command_manager.get_term(command_name)
    v = getattr(term, "target_lin_vel_b", None)
    if v is None:
        return torch.zeros(env.num_envs, 3, device=env.device)
    return v


def wrap_margin(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """손목 관절(j6, j4)이 자기 범위 안에서 **어디에 있는지**. (num_envs, 2), 0=하한 1=상한.

    왜 필요한가 — j6 감김 (RETRAIN_V5 §2-A)
    ---------------------------------------
    실기와 시뮬 양쪽에서 같은 실패를 확인했다. 껍질 중앙에서 ``yaw +90`` 목표를
    ``yaw 0`` 자세에서 접근하면 자세오차가 **20.8~21.2 deg** 로 남는다.
    다른 진입 방향에서는 1.08 deg 로 통과한다.

    ==================  ============  ==============
    진입                 j6 [deg]      자세오차
    yaw 0 (실패)          -175.0        20.82 deg
    그 외 4 방향          +164.4         1.08 deg
    ==================  ============  ==============

    정확해는 ``j6 = +164.4`` 로 **존재**하지만 j6 범위의 반대편이라 360 도 돌아야 한다.
    문제는 관측의 ``ee_pose_error`` axis-angle 이 **최단 회전**을 준다는 것이다.
    그것을 따르면 j6 이 한계(-175)로 간다. 정책은 ``joint_pos_rel`` 로 자기 j6 위치를
    보긴 하지만, **"지금 최단 회전을 따르면 한계에 부딪힌다"** 를 미리 알 방법이 없다.
    부딪힌 뒤에는 오차가 그대로 남아 결정론적 고정점이 된다 (독립 측정 2 회 동일).

    그래서 한계까지의 여유를 명시적으로 넣는다. 손목만 넣는 이유는 감김이 손목 현상이고,
    어깨(j1~j3)는 이 태스크에서 한계에 닿지 않기 때문이다.

    한계
    ----
    **관측에 정보가 있어도 정책이 그것을 쓰는 것이 보상에 유리해야 배운다.**
    이것만으로 부족하면 한계 접근 벌점(RETRAIN_V5 §2-A(b))이나 초기 자세
    랜덤화(§2-A(c))를 얹어야 한다. 그래서 단일 변수로 하나씩 붙인다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    lim = asset.data.joint_pos_limits  # (num_envs, num_joints, 2)
    q = asset.data.joint_pos
    idx = [5, 3]  # j6, j4  (감김이 일어나는 관절)
    lo = lim[:, idx, 0]
    hi = lim[:, idx, 1]
    return ((q[:, idx] - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)


def wrap_margin_named(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """:func:`wrap_margin` 의 **이름 기반** 판. (num_envs, len(joint_names))

    왜 따로 만드는가
    ----------------
    :func:`wrap_margin` 은 관절 인덱스를 ``idx = [5, 3]`` 으로 **하드코딩**한다.
    팔 단독 에셋(FR3 6 관절)에서는 그것이 j6, j4 가 맞다.

    **통합 에셋에서는 틀린다.** 구동륜 2 + 캐스터 8 + 팔 6 + 손가락 2 라
    5 번과 3 번은 캐스터다. 그대로 쓰면 아무 의미 없는 값을 관측에 넣게 되고,
    **에러 없이 조용히** 그렇게 된다.

    그래서 ``asset_cfg.joint_names`` 로 지정한 관절만 본다::

        SceneEntityCfg("robot", joint_names=["j6", "j4"])

    의미는 같다 — 0 = 하한, 1 = 상한, 0.5 = 범위 중앙.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    idx = asset_cfg.joint_ids
    lim = asset.data.joint_pos_limits
    q = asset.data.joint_pos
    lo = lim[:, idx, 0]
    hi = lim[:, idx, 1]
    return ((q[:, idx] - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)


def base_ref_error_b(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """플래너 차체 참조와 현재 차체의 **오차**. (num_envs, 4) = dx, dy, sin(dtheta), cos(dtheta)

    왜 필요한가 ([가-1])
    --------------------
    EE 궤적만 따라가면 차체는 정책이 아무 데나 세운다. 그런데 플래너는 **자기가 계획한
    차체 자리**에서 SDF 여유를 확보한다. 우리가 다른 자리에 서면 그 보장이 안 넘어온다.

    실측(``scripts/preview_plan.py``): 차체가 계획과 **중앙 24 cm** 벌어졌다.
    플래너 기본 여유 ``base_margin 0.05`` 로는 전혀 못 덮는다.

    왜 sin/cos 인가
    ---------------
    ``dtheta`` 를 각도 그대로 넣으면 +-pi 경계에서 튄다. rot6d 를 쓰는 것과 같은 이유다.
    1 차원 더 쓰고 불연속을 없앤다.

    프레임
    ------
    **차체 프레임**이다. 그리고 현재 차체 pose 는 **AMCL 추정값**을 쓴다 — EE 목표와
    같은 처리라야 실기와 조건이 맞는다.
    """
    term = env.command_manager.get_term(command_name)
    ref = getattr(term, "base_ref_w", None)
    if ref is None:
        return torch.zeros(env.num_envs, 4, device=env.device)
    est = term.amcl_estimate                      # (N, 3) = x, y, yaw
    d = ref[:, :2] - est[:, :2]
    c, s = torch.cos(est[:, 2]), torch.sin(est[:, 2])
    dx = c * d[:, 0] + s * d[:, 1]
    dy = -s * d[:, 0] + c * d[:, 1]
    dth = ref[:, 2] - est[:, 2]
    return torch.stack([dx, dy, torch.sin(dth), torch.cos(dth)], dim=1)


def base_ref_twist(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """플래너 차체 참조의 **속도** (v, omega). (num_envs, 2)

    피드포워드다. 오차만 보면 항상 한 박자 늦는다 — EE 쪽에서 이미 확인된 구조로,
    ``target_vel`` 을 막으면 운용 구간에서 오차가 2.5 배가 됐다.

    유니사이클이라 이미 차체 프레임 값이다. 변환이 필요 없다.
    """
    term = env.command_manager.get_term(command_name)
    t = getattr(term, "base_ref_twist", None)
    if t is None:
        return torch.zeros(env.num_envs, 2, device=env.device)
    return t
