# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""모바일 매니퓰레이터용 커맨드 term.

왜 필요한가
-----------
Isaac Lab 기본 ``UniformPoseCommand`` 는 목표 pose 를 **로봇 root 프레임 기준**으로 샘플링한다
(``pose_command_b``). 매 스텝 ``pose_command_w = root_pose ⊕ pose_command_b`` 로 월드 좌표를 만든다.

고정형 팔은 root 가 안 움직이므로 이게 곧 월드 고정 목표와 같다.
하지만 **모바일 매니퓰레이터는 root 가 움직인다** (Ridgeback 실측: ``is_fixed_base=False``,
베이스 조인트 구동 시 ``root_pos_w`` 가 실제로 이동). 그러면 목표가 로봇에 붙어 따라다니므로
베이스가 주행해도 오차가 줄지 않고, 결국 **베이스를 움직일 이유가 사라진다.**

그래서 목표를 **env 원점(월드) 기준으로 고정**시키는 term 이 필요하다.
구현은 Isaac Lab 의 ``UniformPose2dCommand`` (navigation 용, env_origins 기준 샘플링)와
``UniformPoseCommand`` (6D pose)를 합친 형태다.

동작
----
- ``pose_command_w`` : env 원점 기준으로 샘플링된 **월드 고정** 목표 (보상/메트릭/시각화가 이걸 씀)
- ``pose_command_b`` : 매 스텝 월드 목표를 로봇 root 프레임으로 변환한 값 (**관측**이 이걸 씀)

관측을 base 프레임으로 주는 이유: 정책이 "내 기준 목표가 어디인지"를 보게 하려고.
월드 절대좌표를 그대로 주면 env 마다 원점이 달라 일반화가 안 된다.
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import MISSING

from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    compute_pose_error,
    euler_xyz_from_quat,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
    quat_unique,
    subtract_frame_transforms,
)


class UniformPoseWorldCommand(UniformPoseCommand):
    """env 원점 기준으로 목표 EE pose 를 샘플링하는 커맨드 (= 월드에 고정된 목표)."""

    cfg: UniformPoseWorldCommandCfg

    def _resample_command(self, env_ids: Sequence[int]):
        # env 원점을 기준으로 월드 좌표에서 직접 샘플링
        origins = self._env.scene.env_origins[env_ids]
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_w[env_ids, 0] = origins[:, 0] + r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_w[env_ids, 1] = origins[:, 1] + r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_w[env_ids, 2] = origins[:, 2] + r.uniform_(*self.cfg.ranges.pos_z)
        # 자세
        euler_angles = torch.zeros_like(self.pose_command_w[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        self.pose_command_w[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _update_command(self):
        # 매 스텝: 월드 고정 목표를 현재 로봇 root 프레임으로 변환 -> 관측(command)에 쓰인다
        self.pose_command_b[:, :3], self.pose_command_b[:, 3:] = subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
        )

    def _update_metrics(self):
        # 부모는 pose_command_b 로부터 월드 목표를 재계산하지만, 우리는 이미 월드 목표를 들고 있다
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)


@configclass
class UniformPoseWorldCommandCfg(UniformPoseCommandCfg):
    """:class:`UniformPoseWorldCommand` 설정.

    ``ranges`` 의 의미가 부모와 다르다: **로봇 base 기준이 아니라 env 원점 기준**이다.
    """

    class_type: type = UniformPoseWorldCommand


class MovingPoseWorldCommand(UniformPoseWorldCommand):
    """월드(env 원점) 프레임에서 **연속적으로 움직이는** EE 목표 pose.

    왜 필요한가
    -----------
    :class:`UniformPoseWorldCommand` 는 일정 주기마다 목표를 **계단식으로 점프**시킨다.
    그러나 실제 운용은 상위 제어기가 주는 **연속 궤적**을 추종하는 것이다.

    정지 목표로 학습한 정책은 "가서 멈춰라"를 배우고, 목표 속도를 관측하지 못하므로
    순수 피드백 제어기가 된다. 그러면 움직이는 기준에 대해 정상상태 추종 오차가 남는다::

        추종 오차 ~= (궤적 속도) x (시정수)

    실측(AMR+FR3 mm_v1, scripts/measure_settling.py): 시정수 tau ~= 0.5 s.
    따라서 30 cm/s 궤적이면 오차가 15 cm 까지 벌어진다. 목표 속도를 관측에 넣어
    **피드포워드**를 확보해야 이 지연을 줄일 수 있다.

    동작
    ----
    목표는 waypoint 를 향해 일정 속력으로 직진하고, 도달하면 새 waypoint 를 뽑는다.
    자세도 목표 자세를 향해 각속도 제한을 걸고 회전한다. 결과적으로 목표 pose 가
    **매 스텝 연속적으로** 변한다.

    - ``pose_command_w``  : 월드 좌표 목표 (보상/메트릭/마커가 사용)
    - ``pose_command_b``  : root 프레임으로 변환한 값 (관측)
    - ``target_lin_vel_w``: 목표의 월드 선속도. 관측(피드포워드)에 쓰인다
    - ``target_ang_vel_w``: 목표의 월드 각속도

    ``resampling_time_range`` 는 에피소드보다 길게 두어 중간 점프가 없게 한다.
    (부모의 ``_resample_command`` 는 에피소드 리셋 시 초기화 용도로만 쓰인다)
    """

    cfg: "MovingPoseWorldCommandCfg"

    def __init__(self, cfg: "MovingPoseWorldCommandCfg", env):
        super().__init__(cfg, env)
        self.waypoint_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.waypoint_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.waypoint_quat_w[:, 0] = 1.0
        self.speed = torch.zeros(self.num_envs, device=self.device)
        self.ang_speed = torch.zeros(self.num_envs, device=self.device)
        self.target_lin_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_ang_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["target_speed"] = torch.zeros(self.num_envs, device=self.device)

    def _sample_pose_w(self, env_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """env 원점 기준으로 pose 하나를 균등 샘플링해서 월드 좌표로 돌려준다."""
        origins = self._env.scene.env_origins[env_ids]
        r = torch.empty(len(env_ids), device=self.device)
        pos = torch.zeros(len(env_ids), 3, device=self.device)
        pos[:, 0] = origins[:, 0] + r.uniform_(*self.cfg.ranges.pos_x)
        pos[:, 1] = origins[:, 1] + r.uniform_(*self.cfg.ranges.pos_y)
        pos[:, 2] = origins[:, 2] + r.uniform_(*self.cfg.ranges.pos_z)
        euler = torch.zeros(len(env_ids), 3, device=self.device)
        euler[:, 0].uniform_(*self.cfg.ranges.roll)
        euler[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        return pos, (quat_unique(quat) if self.cfg.make_quat_unique else quat)

    def _new_pos_waypoint(self, env_ids: Sequence[int]):
        """새 **위치** waypoint 와 이동 속력을 뽑는다."""
        pos, _ = self._sample_pose_w(env_ids)
        self.waypoint_w[env_ids] = pos
        self.speed[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.speed_range)

    def _new_ang_waypoint(self, env_ids: Sequence[int]):
        """새 **자세** waypoint 와 회전 속력을 뽑는다."""
        _, quat = self._sample_pose_w(env_ids)
        self.waypoint_quat_w[env_ids] = quat
        self.ang_speed[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.ang_speed_range)

    def _new_waypoint(self, env_ids: Sequence[int]):
        """위치·자세 waypoint 를 모두 새로 뽑는다 (에피소드 리셋용)."""
        self._new_pos_waypoint(env_ids)
        self._new_ang_waypoint(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        # 에피소드 리셋 시: 목표 pose 를 새로 놓고 첫 waypoint 를 뽑는다
        pos, quat = self._sample_pose_w(env_ids)
        self.pose_command_w[env_ids, :3] = pos
        self.pose_command_w[env_ids, 3:] = quat
        self._new_waypoint(env_ids)

    def _update_command(self):
        dt = self._env.step_dt

        # --- 위치: waypoint 를 향해 일정 속력으로 직진 ---
        to_wp = self.waypoint_w - self.pose_command_w[:, :3]
        dist = torch.norm(to_wp, dim=1, keepdim=True)
        direction = to_wp / dist.clamp(min=1e-6)
        step_len = (self.speed * dt).unsqueeze(-1)
        move = torch.minimum(step_len, dist) * direction
        self.pose_command_w[:, :3] += move
        self.target_lin_vel_w = move / dt

        # --- 자세: 목표 자세를 향해 각속도 제한을 걸고 회전 ---
        q_err = quat_mul(self.waypoint_quat_w, quat_conjugate(self.pose_command_w[:, 3:]))
        aa = axis_angle_from_quat(q_err)
        angle = torch.norm(aa, dim=1, keepdim=True)
        max_step = (self.ang_speed * dt).unsqueeze(-1)
        aa_step = aa / angle.clamp(min=1e-6) * torch.minimum(max_step, angle)
        self.pose_command_w[:, 3:] = quat_unique(
            quat_mul(quat_from_angle_axis_vec(aa_step), self.pose_command_w[:, 3:])
        )
        self.target_ang_vel_w = aa_step / dt

        # --- waypoint 도달 시 새 목표 ---
        #   [중요] 위치와 자세를 **분리**해서 판정한다.
        #   AND 로 묶으면(초기 구현) 위치가 먼저 도착한 뒤 자세가 맞을 때까지
        #   목표가 **제자리에 멈춘 채 회전만** 한다. 그 동안 선속도가 0 이므로
        #   평균 목표 속력이 회전 대기시간에 지배되어 포화된다::
        #
        #       평균 속력 = D / (D/v + dtheta/omega)   ->  v 를 키워도 D*omega/dtheta 로 수렴
        #
        #   실측(traj_v1): 속력 커리큘럼을 3단계까지 올렸는데 target_speed 가
        #   0.022 -> 0.036 -> 0.068 에서 포화했고, 4단계에서는 아예 안 올랐다.
        #   position_error 도 8.0 mm 로 **정지 목표 태스크(9.5 mm)와 거의 같아서**
        #   목표가 대부분의 시간 정지해 있었음이 확인되었다.
        reached_pos = dist.squeeze(-1) < self.cfg.waypoint_tolerance
        if reached_pos.any():
            self._new_pos_waypoint(reached_pos.nonzero().flatten())
        reached_ang = angle.squeeze(-1) < self.cfg.angle_tolerance
        if reached_ang.any():
            self._new_ang_waypoint(reached_ang.nonzero().flatten())

        # --- 관측용: 월드 목표를 root 프레임으로 ---
        self.pose_command_b[:, :3], self.pose_command_b[:, 3:] = subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
        )
        self.metrics["target_speed"] = torch.norm(self.target_lin_vel_w, dim=1)


PLAN_YAW_OFFSET = +math.pi / 2.0
"""플래너가 주는 **막대 방향** -> 우리 tool0 목표 요 로 바꿀 때 더하는 값 [rad].

왜 필요한가
-----------
2026-08-07 실기: **그리퍼가 막대를 잡았을 때 막대 방향이 기준과 90 도 어긋났다.**
정책이 오차 0 이라고 믿는 상태에서 실물 오차가 90 도였다.

원인은 규약이다::

    플래너      ee_yaw = obj_yaw          **막대 축 방향** (two_robot_nlp.py:363)
    URDF        손가락이 tool0 **x축**으로 여닫힘 (finger axis = 1 0 0)
                -> 잡힌 막대는 tool0 **y축**에 놓인다
    우리 규약   Rz(gamma) @ Ry(pi) 의 x열 = [-cos g, -sin g, 0]
                -> gamma 를 **손가락 방향**에 걸고 있었다

기본자세(j6=0) FK 로 검산하면::

    tool0 y축 -> [1, 0, 0] = +x   (막대가 +x. 실기 확인과 일치)
    tool0 x축 -> [0, 1, 0] = +y   (손가락 여닫힘. 실기 확인과 일치)

**부호는 +90 도다.** 막대 축만 보면 +-90 도가 같지만, **그리퍼 몸통이 어느 쪽에
오는지가 반대**다. 실기 관찰 (2026-08-08)::

    필요한 배치   어깨 - 그리퍼 - 물체 - 그리퍼 - 어깨    (팔이 물체와 안 부딪힘)
    -90 도 결과   그리퍼 - 어깨 - 물체 - 어깨 - 그리퍼    <- 뒤집힘

기구학적 부담은 어느 쪽이든 비슷하다. 여러 시드로
풀어 **|j6| 가 가장 작은 해**를 고르는 방식으로 재면 손목 부담이 거의 같다::

                     도달률   필요 |j6| 중앙   >170 도
    보정 없음         82.3 %    48.6 도        0.91 %
    -90 도            82.6      49.6           0.61 %      <- 채택
    +90 도            81.5      42.2           0.80 %

.. warning::
   **워밍스타트 IK 로는 부호를 고르면 안 된다.** 직전 해에서 이어 푸는 방식은 어느
   분기에 떨어지느냐에 따라 값이 크게 달라져, 같은 질문에 두 번 반대 답을 냈다
   (-90 이 4.3 % / +90 이 0.2 % -> 데이터를 바꾸니 +90 이 6.6 %). 분기 운을 없애려면
   **여러 시드로 풀고 목적에 맞는 해를 고른 뒤** 비교해야 한다.

   v19/v20 에서 팔이 주기적으로 무너진 진짜 원인은 부호가 아니라 **도달 보정**이었다
   (한 스텝 관절변화 최대 303 도 -> 보정을 끄면 24 도).

.. warning::
   **이 상수를 쓰는 곳이 여러 군데다.** 학습(:meth:`_update_from_plan`) 과 평가
   스크립트(``eval_plans`` / ``diag_orientation`` / ``eval_base_speed`` / ``preview_plan``),
   그리고 실기 브리지가 전부 같은 값을 써야 한다. 한 곳만 고치면 **에러 없이**
   학습과 평가가 90 도 어긋난다. 평가 스크립트는 이 상수를 import 해서 쓴다.
"""


def _wrap_pi(x: torch.Tensor) -> torch.Tensor:
    """각도를 -pi..pi 로 감싼다."""
    return torch.atan2(torch.sin(x), torch.cos(x))


def quat_from_angle_axis_vec(aa: torch.Tensor) -> torch.Tensor:
    """axis-angle 3벡터 -> 쿼터니언 (wxyz). 각이 0 에 가까우면 항등 쿼터니언."""
    angle = torch.norm(aa, dim=1, keepdim=True)
    axis = aa / angle.clamp(min=1e-6)
    half = 0.5 * angle
    q = torch.cat([torch.cos(half), axis * torch.sin(half)], dim=1)
    return torch.where(angle < 1e-6, torch.tensor([1.0, 0.0, 0.0, 0.0], device=aa.device).expand_as(q), q)


@configclass
class MovingPoseWorldCommandCfg(UniformPoseWorldCommandCfg):
    """:class:`MovingPoseWorldCommand` 설정."""

    class_type: type = MovingPoseWorldCommand

    speed_range: tuple[float, float] = (0.0, 0.05)
    """목표의 이동 속력 범위 [m/s]. 커리큘럼으로 넓혀 간다."""

    ang_speed_range: tuple[float, float] = (0.0, 0.2)
    """목표의 회전 각속도 범위 [rad/s]."""

    waypoint_tolerance: float = 0.05
    """이 거리 안에 들어오면 새 waypoint 를 뽑는다 [m]."""

    angle_tolerance: float = 0.1
    """이 각도 안에 들어오면 새 waypoint 를 뽑는다 [rad]."""


##
# 원통 껍질 목표 커맨드 (RETRAIN_V4 §2-A)
##


class CylindricalPoseCommand(UniformPoseCommand):
    """목표 위치를 **원통 껍질**에서 뽑는다. 자세는 부모(균등 오일러)와 동일하다.

    왜 박스가 아니라 껍질인가
    -------------------------
    팔의 도달 영역은 base 중심 **구각**이다. 링크 길이 합(0.28+0.24+0.102+0.102 = 0.72 m)이
    바깥 반경을, 관절 한계가 안쪽과 각도 범위를 정한다. 축정렬 박스를 그 안에 넣으면
    **대각선 모서리가 항상 껍질을 뚫는다.**

    v3d 에서 실측으로 확인했다 - 후보 상자를 어떻게 바꿔도 예외 없이 `멀고·옆으로·높은`
    구석에서 도달률이 무너졌고, 37.6 L 이 축정렬 박스의 한계였다. 그 방향이 특별히
    나빠서가 아니라 박스 대각선이 가장 먼 지점이기 때문이다.

    반경으로 다시 읽으면 수평거리 0.26~0.50 구간이 z 0.18~0.54 전 구간에서 90~100% 인
    **띠(annulus)** 다. 같은 도달률에서 부피가 1.6~2 배가 된다.

    좌표 규약
    ---------
    FR3 는 URDF 상 ``q=0`` 에서 팔이 base 기준 **-x** 를 향한다 (``assets/fairino_fr3.py``).
    그래서 azimuth 를 **base -x 축 기준**으로 재고, 위치는 이렇게 만든다::

        x = -rho * cos(azimuth)
        y = +rho * sin(azimuth)
        z =  z

    azimuth = 0 이 팔이 곧게 뻗은 방향이고, +-azimuth_range 로 좌우로 벌어진다.
    기존 박스 커맨드와 **같은 base 프레임**을 쓰므로 관측·보상은 손댈 필요가 없다.

    면적 균등 샘플링
    ----------------
    ``rho`` 를 그냥 균등하게 뽑으면 **안쪽이 과대표집**된다 (고리의 면적이 rho 에 비례하므로).
    ``rho = sqrt(uniform(r1^2, r2^2))`` 로 뽑아야 공간에 고르게 퍼진다.
    이걸 틀리면 정책이 안쪽만 잘하고 바깥이 나빠진다.
    """

    cfg: "CylindricalPoseCommandCfg"

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        r = torch.empty(n, device=self.device)

        # -- 위치: 원통 껍질에서 면적 균등으로
        #    껍질은 **팔 base 축** 기준으로 정의되고, 커맨드는 로봇 root 프레임이다.
        #    통합 모델은 둘이 다르므로 origin_offset 으로 옮긴다 (팔 단독은 (0,0,0)).
        ox, oy, oz = self.cfg.origin_offset
        r1, r2 = self.cfg.ranges.radius
        rho = r.uniform_(r1 * r1, r2 * r2).sqrt()
        azim = torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.azimuth)
        self.pose_command_b[env_ids, 0] = -rho * torch.cos(azim) + ox
        self.pose_command_b[env_ids, 1] = rho * torch.sin(azim) + oy
        self.pose_command_b[env_ids, 2] = torch.empty(n, device=self.device).uniform_(
            *self.cfg.ranges.pos_z
        ) + oz

        # -- 자세
        euler_angles = torch.zeros(n, 3, device=self.device)
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        if self.cfg.yaw_relative_to_radial:
            # yaw 를 **목표점의 base 프레임 방위각** 기준으로 잡는다.
            #   목표점이 (-rho cos az, rho sin az) 이므로 그 방위각은 atan2(y, x) = pi - az 다.
            #   ranges.yaw 는 그 기준으로부터의 **오프셋**이 된다.
            euler_angles[:, 2] += torch.pi - azim
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat


@configclass
class CylindricalPoseCommandCfg(UniformPoseCommandCfg):
    """:class:`CylindricalPoseCommand` 설정.

    ``ranges.pos_x`` / ``pos_y`` 는 **쓰이지 않는다** (radius / azimuth 가 대체한다).
    부모 클래스의 필드라 남아 있지만 무시된다.
    """

    class_type: type = CylindricalPoseCommand

    origin_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """껍질 중심축을 로봇 root 프레임에서 어디에 둘지 [m].

    **커맨드는 로봇 root 프레임인데 껍질은 팔 base 축 기준으로 정의된다.** 팔 단독
    태스크는 root 가 곧 팔 base 라 (0,0,0) 이면 되지만, 통합 모델은 팔이 차체 위에
    올라가 있어 둘이 다르다::

        팔 단독 (fairino_fr3)   root = 팔 base          -> (0, 0, 0)
        통합    (amr_fr3)       root = base_link        -> (0.205, 0, 0.385)

    이걸 빠뜨리면 껍질이 차체 중심을 축으로 잡혀 **팔이 닿지 않는 영역**이 섞인다.
    방위각과 반경은 이 축을 기준으로 재므로 단순 평행이동으로 처리된다.
    """

    yaw_relative_to_radial: bool = False
    """목표 yaw 를 **반경방향 기준**으로 잡을지. True 면 ``ranges.yaw`` 가 오프셋이 된다.

    **``j6`` 감김을 없애는 설정이다** (`BASELINE_INTEGRATED.md` §4).

    base 프레임에 yaw 를 고정하면, 방위각이 넓을 때 ``j1`` 이 목표를 따라 돌고 ``j6`` 가
    그만큼 되돌려야 해서 관절 한계까지 밀린다. 정확해가 관절범위 반대편에 생기는데
    관측의 축-각 오차는 **최단 회전**만 주므로 정책이 한계 쪽으로 밀어붙이고, 실기에서
    자세오차 21.23 도가 났다 (v4c). ``yaw`` **범위를 좁혀도 해결되지 않는다** —
    원인이 범위가 아니라 방위각 폭이기 때문이다 (오히려 9.15% -> 11.82% 로 악화).

    확정 껍질(rho 0.20~0.50, z 0.05~0.22, ±135도, pitch 180±10도) 에서 측정한 ``j6`` 여유::

        yaw 정의              도달률    여유 하위10%   한계15도이내   j6 사용범위
        고정 ±172 (기존)     98.47%       16.6도        9.15%    -175 ~ +175도
        반경방향 ±90         99.61%       30.9도        2.53%     -39 ~ +171도
        반경방향 ±45         99.34%       66.1도        0.00%       1 ~ +125도
        반경방향 ±20         99.23%       85.2도        0.00%      27 ~  +98도

    ``±45 도`` 에서 한계 근접이 0% 가 되고 도달률도 오른다. 물리적으로는 **그리퍼가 팔이
    뻗는 방향을 기준으로 물체에 접근**하는 것이고, 파지 작업의 자연스러운 자세다.

    커버리지 손실은 없다 — 차체가 회전하므로 물체가 어떤 방향이든 AMR 이 몸을 돌려
    맞춘다. 논홀로노믹이라 어차피 목표 쪽으로 선회해야 하므로 자연스러운 분업이다.
    """

    @configclass
    class Ranges(UniformPoseCommandCfg.Ranges):
        """부모의 pos_z / roll / pitch / yaw 를 그대로 쓰고 radius / azimuth 를 추가한다."""

        radius: tuple[float, float] = MISSING
        """base 축으로부터의 수평거리 [m]. 면적 균등으로 샘플된다."""

        azimuth: tuple[float, float] = MISSING
        """base **-x 축 기준** 방위각 [rad]. 0 이 팔이 곧게 뻗은 방향이다."""

    ranges: Ranges = MISSING


class CylindricalRampCommand(CylindricalPoseCommand):
    """원통 껍질 안에서 목표가 **연속 이동**하는 커맨드 (RETRAIN_V4 §2-C).

    왜 필요한가
    -----------
    지금까지 학습은 전부 **계단 목표**였다 (``resampling_time_range=(4,4)``, 4 초마다 순간이동).
    정책이 배우는 것은 "큰 오차를 보고 가서 멈춰라" 다.

    그런데 베이스가 붙으면 목표가 base 프레임에서 **계속 흐른다.** 순수 피드백 제어기는
    움직이는 기준을 따라갈 때 구조적으로 뒤처진다::

        추종 오차 ~= 목표 속도 x 시정수

    실기 실측 시정수가 **0.160 s** 이고 속도에 정확히 비례했다 (10/30/50 mm/s 에서
    비율 0.95/0.98/0.99). 베이스가 0.2 m/s 로 움직이면 지연이 32 mm 가 되는데,
    현재 정확도 2.8 mm 의 11 배다.

    계단과 램프를 **섞는** 이유
    ---------------------------
    전부 램프로 바꾸면 "멀리서 접근하는" 큰 오차 구간을 학습하지 못해 초기 수렴이 나빠진다.
    ``ramp_prob`` 로 비율을 정한다 (지시서 제안은 절반).

    경계 처리
    ---------
    램프가 껍질 밖으로 나가면 안 된다. 원통 좌표 기저에서 **위반한 성분만 반사**한다::

        e_r = (-cos az,  sin az, 0)      반경 방향
        e_t = ( sin az,  cos az, 0)      접선 방향
        e_z = (0, 0, 1)

    rho 경계를 넘으면 v_r 을 뒤집고, azimuth 경계면 v_t, z 경계면 v_z 를 뒤집는다.
    그 뒤 위치를 경계 안으로 클램프한다. 반사이므로 속력이 유지되고 목표가 껍질 안에 머문다.

    피드포워드 관측
    ---------------
    ``target_lin_vel_b`` 속성을 노출한다. 정책이 **목표가 어디로 갈지** 직접 보게 해야
    지연이 줄어든다 - 오차만 보면 순수 피드백이라 구조적 지연이 남는다.
    (AMR+FR3 궤적 추종에서 확인한 교훈. ``ee_target_vel_b`` 참고)

    자세는 램프 중에도 고정이다. 베이스 **회전**까지 다루려면 각속도도 흘려야 하는데,
    이번 사이클은 병진에 집중한다.
    """

    cfg: "CylindricalRampCommandCfg"

    def __init__(self, cfg: "CylindricalRampCommandCfg", env):
        super().__init__(cfg, env)
        self.target_lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_ang_vel_b = torch.zeros(self.num_envs, 3, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        n = len(env_ids)
        # 이 env 가 램프인가 (아니면 계단 = 속도 0)
        is_ramp = torch.rand(n, device=self.device) < self.cfg.ramp_prob
        speed = torch.empty(n, device=self.device).uniform_(*self.cfg.speed_range)
        speed = torch.where(is_ramp, speed, torch.zeros_like(speed))
        # 등방 방향 (구면 균등)
        d = torch.randn(n, 3, device=self.device)
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.target_lin_vel_b[env_ids] = d * speed.unsqueeze(-1)

    def _update_command(self):
        v = self.target_lin_vel_b
        active = v.norm(dim=-1) > 1e-6
        if not active.any():
            return
        dt = self._env.step_dt
        # 껍질 축 기준으로 옮겨서 계산한다 (origin_offset 참고). 팔 단독은 0 이라 무영향이다.
        off = torch.tensor(self.cfg.origin_offset, device=self.device)
        p = self.pose_command_b[:, :3] + v * dt - off

        # 원통 좌표
        rho = p[:, :2].norm(dim=-1).clamp_min(1e-6)
        az = torch.atan2(p[:, 1], -p[:, 0])
        z = p[:, 2]
        r1, r2 = self.cfg.ranges.radius
        a1, a2 = self.cfg.ranges.azimuth
        z1, z2 = self.cfg.ranges.pos_z

        # 위반 성분만 반사
        c, s = torch.cos(az), torch.sin(az)
        e_r = torch.stack([-c, s, torch.zeros_like(c)], dim=-1)
        e_t = torch.stack([s, c, torch.zeros_like(c)], dim=-1)
        v_r = (v * e_r).sum(-1)
        v_t = (v * e_t).sum(-1)
        v_z = v[:, 2]
        flip_r = (rho < r1) | (rho > r2)
        flip_t = (az < a1) | (az > a2)
        flip_z = (z < z1) | (z > z2)
        v_r = torch.where(flip_r, -v_r, v_r)
        v_t = torch.where(flip_t, -v_t, v_t)
        v_z = torch.where(flip_z, -v_z, v_z)
        new_v = e_r * v_r.unsqueeze(-1) + e_t * v_t.unsqueeze(-1)
        new_v[:, 2] = v_z
        self.target_lin_vel_b = torch.where(active.unsqueeze(-1), new_v, v)

        # 위치를 경계 안으로 클램프
        rho_c = rho.clamp(r1, r2)
        az_c = az.clamp(a1, a2)
        z_c = z.clamp(z1, z2)
        p_c = torch.stack([-rho_c * torch.cos(az_c), rho_c * torch.sin(az_c), z_c], dim=-1) + off
        self.pose_command_b[:, :3] = torch.where(active.unsqueeze(-1), p_c, self.pose_command_b[:, :3])


@configclass
class CylindricalRampCommandCfg(CylindricalPoseCommandCfg):
    """:class:`CylindricalRampCommand` 설정."""

    class_type: type = CylindricalRampCommand

    ramp_prob: float = 0.5
    """램프가 될 확률. 나머지는 계단(속도 0)이다.

    1.0 으로 두면 큰 오차 구간을 학습하지 못해 초기 수렴이 나빠질 수 있다 (지시서 §2-C).
    """

    speed_range: tuple[float, float] = (0.02, 0.20)
    """목표 속력 [m/s]. 예상 베이스 속도 범위."""


class AmclPoseEstimator:
    """AMCL 로 추정한 root pose 를 만들어 주는 도구.

    왜 커맨드 클래스에서 분리했는가
    -------------------------------
    이동 목표(:class:`MovingPoseWorldAmclCommand`)가 정지 목표
    (:class:`UniformPoseWorldAmclCommand`)와 **같은** AMCL 모델을 써야 한다.

    .. warning::
       **지금 이 모델은 두 벌 존재한다.** :class:`UniformPoseWorldAmclCommand` 는
       자기 안에 같은 로직을 직접 들고 있고, 이 클래스를 쓰지 않는다.
       그쪽은 통합 14.92 mm 를 낸 **검증된 경로라 동결**했기 때문이다.

       따라서 **AMCL 모델을 고칠 때는 반드시 두 곳을 같이 고쳐야 한다.**
       한쪽만 고치는 사고는 이 저장소에서 실제로 여러 번 났다 (EE 리타깃을
       보상에만 적용하고 관측·종료에 빠뜨린 건 등). 이동 목표 쪽이 검증되면
       정지 목표 쪽도 이 클래스를 쓰도록 합쳐서 두 벌을 없앨 것.

    모델링하는 것 넷
    ----------------
    단순 가우시안으로는 실기 격차를 못 메운다. 실제 성질은 "느리게 갱신되고,
    갱신 사이에는 값이 고정되고, 지연이 있고, 가끔 튄다" 다::

        update_period   스캔 매칭 주기. 그 사이에는 이전 추정값 유지 (zero-order hold)
        latency         스캔 취득 -> 매칭 -> 퍼블리시. 추정값은 '과거의 참값' 이다
        noise           정상상태 추정 오차
        jump_prob       파티클 필터 재수렴 시의 불연속 도약.
                        실기에서 정책이 가장 크게 흔들리는 순간이라 반드시 노출해야 한다

    .. note::
       nav2 의 ``update_min_d: 0.25`` 때문에 **실기 AMCL 은 로봇이 움직여야 돈다.**
       여기서는 시간 주기로 근사한다. 실측한 0.18 s 는 주행 중 값이므로, 정지
       구간이 긴 태스크에서는 낙관적인 모델이다. 협력 운송처럼 계속 움직이는
       운용에서는 이 근사가 오히려 잘 맞는다.
    """

    def __init__(self, num_envs: int, device, dt: float, cfg):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self._dt = dt
        self._hist_len = int(round(cfg.latency / dt)) + 1
        # (hist_len, N, 3) = x, y, yaw 링 버퍼
        self._hist = torch.zeros(self._hist_len, num_envs, 3, device=device)
        self._head = 0
        self._estimate = torch.zeros(num_envs, 3, device=device)
        self._t_since = torch.zeros(num_envs, device=device)
        self._needs_fill = torch.ones(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids=None):
        # 여기서 참값을 읽지 않고 플래그만 세운다 — 리셋 순서에 의존하지 않기 위해서다.
        if env_ids is None:
            self._needs_fill[:] = True
        else:
            self._needs_fill[env_ids] = True

    @property
    def estimate(self) -> torch.Tensor:
        """현재 AMCL 추정값 (x, y, yaw). 로깅/디버깅용."""
        return self._estimate

    def __call__(self, robot) -> tuple[torch.Tensor, torch.Tensor]:
        """추정 root 위치 (N,3) 와 자세 쿼터니언 (N,4). 롤/피치는 참값을 쓴다.

        AMCL 은 2D 로컬라이제이션이라 x/y/yaw 만 추정한다. 높이와 기울기는
        오도메트리/IMU 로 훨씬 정확히 알기 때문에 참값을 그대로 둔다.
        """
        c = self.cfg
        truth = torch.cat([robot.data.root_pos_w[:, :2], robot.data.heading_w.unsqueeze(1)], dim=1)
        if self._needs_fill.any():
            f = self._needs_fill
            self._hist[:, f, :] = truth[f]
            self._estimate[f] = truth[f]
            self._t_since[f] = 0.0
            self._needs_fill[f] = False

        # 지연 버퍼: 현재를 쓰고 가장 오래된 것을 읽는다
        self._hist[self._head] = truth
        oldest = (self._head + 1) % self._hist_len
        delayed = self._hist[oldest]
        self._head = oldest

        self._t_since += self._dt
        due = self._t_since >= c.update_period
        scale = torch.tensor([c.pos_noise_std, c.pos_noise_std, c.yaw_noise_std], device=self.device)
        cand = delayed + torch.randn_like(delayed) * scale
        if c.jump_prob > 0.0:
            js = torch.tensor([c.jump_pos_std, c.jump_pos_std, c.jump_yaw_std], device=self.device)
            m = torch.rand(self.num_envs, 1, device=self.device) < c.jump_prob
            cand = cand + m * torch.randn_like(cand) * js
        self._estimate = torch.where(due.unsqueeze(1), cand, self._estimate)
        self._t_since = torch.where(due, torch.zeros_like(self._t_since), self._t_since)

        pos = robot.data.root_pos_w.clone()
        pos[:, :2] = self._estimate[:, :2]
        # 참값 자세에서 yaw 만 추정값으로 갈아끼운다
        r, p, _ = euler_xyz_from_quat(robot.data.root_quat_w)
        quat = quat_from_euler_xyz(r, p, self._estimate[:, 2])
        return pos, quat


class UniformPoseWorldAmclCommand(UniformPoseWorldCommand):
    """월드 고정 EE 목표를 **AMCL 추정 pose 기준**으로 베이스 프레임에 변환한다.

    왜 필요한가
    -----------
    부모 :class:`UniformPoseWorldCommand` 는 목표를 로봇 root 프레임으로 옮길 때
    **참값 root pose** 를 쓴다. 시뮬은 차체가 어디 있는지 완벽히 알기 때문이다.
    실로봇에는 그런 것이 없다 — 그 자리에 AMCL 추정값이 들어가고 수 cm 가 틀린다.

    목표가 **맵 프레임**으로 주어지는 운용에서는 이 오차가 EE 목표에 그대로 더해진다.
    차체가 2 cm 틀린 곳에 있다고 믿으면 팔이 아무리 정확해도 EE 는 2 cm 틀린 곳으로 간다.
    **맵 프레임 EE 정확도의 상한이 로컬라이제이션 정확도**다.

    그래서 정책은 "내가 어디 있는지 정확히 모른다" 는 조건에서 학습되어야 한다.
    관측 노이즈로 근사하면 정상상태 오차만 흉내 내고, AMCL 이 튀거나 갱신이 늦을 때의
    거동은 배우지 못한다.

    무엇이 참값이고 무엇이 추정값인가
    ----------------------------------
    ==================== ================================================
    ``pose_command_w``   참값. 월드에 고정된 진짜 목표
    ``pose_command_b``   **추정값 기준**. 정책이 보는 것 (관측)
    ``metrics``          참값 기준. 실제로 얼마나 틀렸는지 (보상/평가)
    ==================== ================================================

    정책은 추정값을 보고 움직이지만 평가는 참값으로 받는다. 그래서 학습되는 것은
    "추정 오차 아래에서 최선을 다하는 것" 이고, 지표에는 **줄일 수 없는 바닥**이 남는다.
    그 바닥이 곧 로컬라이제이션 오차이며, 그것이 정직한 성능 예측이다.

    AMCL 특성 (nav 의 ``amcl_goal_polar`` 와 같은 모델)
    ---------------------------------------------------
    단순 가우시안으로는 실기 격차를 못 메운다. 실제 성질은 "느리게 갱신되고, 갱신
    사이에는 값이 고정되고, 지연이 있고, 가끔 튄다" 다. 넷을 모두 재현한다::

        update_period   스캔 매칭 주기. 그 사이에는 이전 추정값 유지 (zero-order hold)
        latency         스캔 취득 -> 매칭 -> 퍼블리시. 추정값은 '과거의 참값' 이다
        noise           정상상태 추정 오차
        jump_prob       파티클 필터 재수렴 시의 불연속 도약.
                        실기에서 정책이 가장 크게 흔들리는 순간이라 반드시 노출해야 한다

    .. note::
       yaw 추정 오차는 **거리에 비례해 위치 오차로 번진다.** 목표가 2 m 앞이면
       yaw 0.02 rad 는 4 cm 다. 위치 노이즈보다 이쪽이 지배적일 수 있다.
    """

    cfg: "UniformPoseWorldAmclCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        dt = env.step_dt
        self._dt = dt
        self._hist_len = int(round(cfg.latency / dt)) + 1
        # (hist_len, N, 3) = x, y, yaw 링 버퍼
        self._hist = torch.zeros(self._hist_len, self.num_envs, 3, device=self.device)
        self._head = 0
        self._estimate = torch.zeros(self.num_envs, 3, device=self.device)
        self._t_since = torch.zeros(self.num_envs, device=self.device)
        self._needs_fill = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None):
        # 여기서 참값을 읽지 않고 플래그만 세운다 — 리셋 순서에 의존하지 않기 위해서다.
        if env_ids is None:
            self._needs_fill[:] = True
        else:
            self._needs_fill[env_ids] = True
        return super().reset(env_ids)

    def _amcl_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """추정 root 위치 (N,3) 와 자세 쿼터니언 (N,4). 롤/피치는 참값을 쓴다.

        AMCL 은 2D 로컬라이제이션이라 x/y/yaw 만 추정한다. 높이와 기울기는
        오도메트리/IMU 로 훨씬 정확히 알기 때문에 참값을 그대로 둔다.
        """
        c = self.cfg
        truth = torch.cat(
            [self.robot.data.root_pos_w[:, :2], self.robot.data.heading_w.unsqueeze(1)], dim=1
        )
        if self._needs_fill.any():
            f = self._needs_fill
            self._hist[:, f, :] = truth[f]
            self._estimate[f] = truth[f]
            self._t_since[f] = 0.0
            self._needs_fill[f] = False

        # 지연 버퍼: 현재를 쓰고 가장 오래된 것을 읽는다
        self._hist[self._head] = truth
        oldest = (self._head + 1) % self._hist_len
        delayed = self._hist[oldest]
        self._head = oldest

        self._t_since += self._dt
        due = self._t_since >= c.update_period
        scale = torch.tensor(
            [c.pos_noise_std, c.pos_noise_std, c.yaw_noise_std], device=self.device
        )
        cand = delayed + torch.randn_like(delayed) * scale
        if c.jump_prob > 0.0:
            js = torch.tensor([c.jump_pos_std, c.jump_pos_std, c.jump_yaw_std], device=self.device)
            m = torch.rand(self.num_envs, 1, device=self.device) < c.jump_prob
            cand = cand + m * torch.randn_like(cand) * js
        self._estimate = torch.where(due.unsqueeze(1), cand, self._estimate)
        self._t_since = torch.where(due, torch.zeros_like(self._t_since), self._t_since)

        pos = self.robot.data.root_pos_w.clone()
        pos[:, :2] = self._estimate[:, :2]
        # 참값 자세에서 yaw 만 추정값으로 갈아끼운다
        r, p, _ = euler_xyz_from_quat(self.robot.data.root_quat_w)
        quat = quat_from_euler_xyz(r, p, self._estimate[:, 2])
        return pos, quat

    def _update_command(self):
        est_pos, est_quat = self._amcl_pose()
        self.pose_command_b[:, :3], self.pose_command_b[:, 3:] = subtract_frame_transforms(
            est_pos, est_quat, self.pose_command_w[:, :3], self.pose_command_w[:, 3:]
        )

    @property
    def amcl_estimate(self) -> torch.Tensor:
        """현재 AMCL 추정값 (x, y, yaw). 로깅/디버깅용."""
        return self._estimate


@configclass
class UniformPoseWorldAmclCommandCfg(UniformPoseWorldCommandCfg):
    """:class:`UniformPoseWorldAmclCommand` 설정.

    기본값은 ``nav`` 태스크의 AMCL 모델과 같은 값이다. 실기 AMCL 을 실측하면
    그 값으로 바꿔야 한다.
    """

    class_type: type = UniformPoseWorldAmclCommand

    pos_noise_std: float = 0.02
    """정상상태 위치 추정 오차 [m]."""

    yaw_noise_std: float = 0.02
    """정상상태 yaw 추정 오차 [rad]. 목표가 2 m 앞이면 위치로 4 cm 가 된다."""

    update_period: float = 0.05
    """AMCL 갱신 주기 [s]. 그 사이에는 이전 추정값이 유지된다."""

    latency: float = 0.06
    """스캔 취득 -> 퍼블리시 지연 [s]. 버퍼 길이가 생성 시점에 고정되므로 런타임 변경 불가."""

    jump_prob: float = 0.0
    """스텝당 점프 확률. 0 이면 점프 없음. 실기 노출을 원하면 0.001~0.01 정도."""

    jump_pos_std: float = 0.10
    jump_yaw_std: float = 0.10


##
# 협력 운송 — 움직이는 월드 목표 + AMCL (STAGE4)
##


class MovingPoseWorldAmclCommand(MovingPoseWorldCommand):
    """월드에서 **연속 이동**하는 EE 목표를, **AMCL 추정 pose 기준**으로 베이스 프레임에 준다.

    두 기능의 결합이다::

        MovingPoseWorldCommand      목표가 waypoint 를 향해 등속 이동 (+ 자세 회전)
        AmclPoseEstimator           로봇이 자기 위치를 모른다는 조건

    왜 필요한가 — 협력 운송 [가]
    -----------------------------
    상위 계획기가 각 로봇에 **EE 궤적을 뿌려 주고** 로봇끼리는 통신하지 않는다.
    그러면 각 로봇의 문제는 "맵 프레임에서 움직이는 EE 목표를 따라가기" 로 환원된다.
    상대 로봇도, 물체도 이 커맨드에 들어오지 않는다.

    지금까지의 통합 학습은 목표가 **월드에 고정**이라 정책이 "차로 가서 멈추고 팔로
    정밀하게" 를 배웠다. 운송은 정지 구간이 없다. 0.2 m/s 로 달리는 내내 정밀도를
    유지해야 하고, 그건 다른 기술이다.

    지연 예산
    ---------
    순수 피드백 제어기는 움직이는 기준에 대해 구조적으로 뒤처진다::

        추종 오차 ~= 목표속도 x 시정수        시정수 0.159 s (실기 실측)

    다만 **목표 속도가 그대로 팔의 부담이 되지는 않는다.** 차체가 따라가면 base
    프레임에서 목표는 거의 정지해 있다. 팔이 감당하는 것은 잔차이고, 그 잔차를
    지배하는 것은 **차체 회전**이다::

        base 프레임 목표 속도 ~= omega x r        r ~= 0.5 m (EE 의 base 원점 거리)

        선회반경 2.0 m  ->  omega 0.10 rad/s  ->  0.05 m/s  ->  지연  8 mm
        선회반경 1.0 m  ->  omega 0.20 rad/s  ->  0.10 m/s  ->  지연 16 mm
        선회반경 0.5 m  ->  omega 0.40 rad/s  ->  0.20 m/s  ->  지연 32 mm   <- 3 cm 초과

    그래서 :func:`~..observations.ee_target_vel_b` 피드포워드가 선택이 아니라 필수다.
    학습이 끝나면 **3 cm 를 지키는 최소 선회반경**이 결과로 나오고, 그것이 상위
    궤적 생성기가 지켜야 할 제약이 된다.

    참값과 추정값 (부모와 같은 규약)
    ---------------------------------
    ==================== ================================================
    ``pose_command_w``   참값. 월드에서 움직이는 진짜 목표
    ``pose_command_b``   **추정값 기준**. 정책이 보는 것 (관측)
    ``metrics``          참값 기준. 실제로 얼마나 틀렸는지 (보상/평가)
    ==================== ================================================
    """

    cfg: "MovingPoseWorldAmclCommandCfg"

    def __init__(self, cfg: "MovingPoseWorldAmclCommandCfg", env):
        super().__init__(cfg, env)
        self._amcl = AmclPoseEstimator(self.num_envs, self.device, env.step_dt, cfg)
        self._needs_init = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # --- 재생 훅 (scripts/preview_plan.py) ---
        #   여기에 (N,7) pose 와 (N,3)+(N,3) 속도를 써 넣으면 내부 이동 대신 그것을 쓴다.
        #   실기 플래너가 푼 궤적을 **학습된 정책으로 폐루프 재생**해서, 실기에 올리기
        #   전에 "이 궤적을 우리가 따라갈 수 있는가" 를 시뮬에서 판정하려는 것이다.
        #   None 이면 평소대로 동작한다 — 학습 경로에는 영향이 없다.
        self.replay_pose_w: torch.Tensor | None = None
        self.replay_lin_vel_w: torch.Tensor | None = None
        self.replay_ang_vel_w: torch.Tensor | None = None
        self.replay_base_ref_w: torch.Tensor | None = None
        self.replay_base_ref_twist: torch.Tensor | None = None

        # --- 차체 참조 궤적 ([가-1]) ---
        #   실기에서는 플래너가 /base_pose, /base_twist 로 준다. 시뮬에는 없으므로
        #   EE 목표에서 유도한다.  (x, y, yaw) 와 (v, omega).
        self.base_ref_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_ref_twist = torch.zeros(self.num_envs, 2, device=self.device)
        self._standoff = torch.zeros(self.num_envs, 2, device=self.device)   # 거리, 방위
        self._prev_ref_yaw = torch.zeros(self.num_envs, device=self.device)

        # --- 실제 MPC 궤적 데이터셋 ---
        #   기하 유도 대신 **플래너가 실제로 푼 궤적**을 쓴다.
        #   v6 이 기하로 만들었다가 실제와 3 배 어긋났다 (편차 0.163 vs 0.527 m).
        #   실측 EE-차체 거리는 중앙 0.462 m (5% 0.339, 95% 0.551) 로,
        #   v6 이 가정한 0.15~0.45 와 거의 안 겹친다.
        self._ds_ee = self._ds_base = None
        self._ds_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ds_t0 = torch.zeros(self.num_envs, device=self.device)
        self._ep_t = torch.zeros(self.num_envs, device=self.device)
        if cfg.plan_dataset:
            import numpy as _np
            import os as _os
            if not _os.path.exists(cfg.plan_dataset):
                raise FileNotFoundError("궤적 데이터셋이 없다: %s" % cfg.plan_dataset)
            d = _np.load(cfg.plan_dataset)
            ee_all = torch.as_tensor(d["ee"], dtype=torch.float32, device=self.device)
            bs_all = torch.as_tensor(d["base"], dtype=torch.float32, device=self.device)
            # **뒤쪽 plan_test_n 개는 학습에서 뺀다** — 시험대(eval_plans.py)가 쓴다.
            #   안 빼면 "학습한 궤적에서 잘한다" 를 성능이라 착각한다. v8/v9/v11 판정이
            #   계획 하나로만 이뤄진 것과 같은 종류의 실수다.
            n_test = max(int(cfg.plan_test_n), 0)
            if n_test >= ee_all.shape[0]:
                raise ValueError("plan_test_n %d 이 궤적 수 %d 이상이다"
                                 % (n_test, ee_all.shape[0]))
            self._ds_ee = ee_all[: ee_all.shape[0] - n_test] if n_test else ee_all
            self._ds_base = bs_all[: bs_all.shape[0] - n_test] if n_test else bs_all
            print("[MovingPoseWorldAmclCommand] MPC 궤적 %d 개 x %d 점 적재 "
                  "(전체 %d, 시험용 제외 %d)"
                  % (self._ds_ee.shape[0], self._ds_ee.shape[1], ee_all.shape[0], n_test))
            print("[MovingPoseWorldAmclCommand] 재생배속 %.2f~%.2f, 데이터셋 사용 비율 %.2f"
                  % (*cfg.plan_time_scale, cfg.plan_mix))
        self._ds_b0 = torch.zeros(self.num_envs, 3, device=self.device)
        self._ds_goal_z = torch.full((self.num_envs,), cfg.plan_goal_z, device=self.device)
        self._ds_scale = torch.ones(self.num_envs, device=self.device)
        self._use_ds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 텐서보드용 지표 — 보상이 아니라 **미터 단위 편차**를 직접 본다.
        #   보상만 보면 "가중치를 바꿨을 때 실제로 나아졌는지" 를 알 수 없다.
        self.metrics["base_ref_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["base_ref_yaw_error"] = torch.zeros(self.num_envs, device=self.device)
        self._prev_ee_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._prev_base_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._prev_base_yaw = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids=None):
        self._amcl.reset(env_ids)
        ids = env_ids if env_ids is not None else torch.arange(self.num_envs, device=self.device)
        # **둘 다 준비한다.** 섞어 쓰므로 어느 쪽이 될지 리셋 시점에 정해지고,
        # 기하 쪽 스탠드오프는 그때 이미 뽑혀 있어야 한다.
        if self._ds_ee is not None:
            self._resample_plan(ids)
        if self.cfg.base_ref:
            self._resample_base_ref(ids)
        if self.cfg.init_at_ee:
            # 여기서 EE 를 읽지 않고 플래그만 세운다 — 리셋 순서에 의존하지 않기 위해서다.
            # (AmclPoseEstimator 와 같은 이유. 이벤트 매니저가 관절을 리셋한 직후라
            #  articulation 데이터가 아직 갱신 전일 수 있다)
            if env_ids is None:
                self._needs_init[:] = True
            else:
                self._needs_init[env_ids] = True
        return super().reset(env_ids)

    def _init_at_ee(self):
        """목표를 **현재 EE 위치에서** 출발시킨다 (물체를 이미 잡고 있는 상태).

        왜 필요한가
        -----------
        기존은 목표를 월드 상자에서 무작위로 뽑아 **멀리 떨어진 곳에 순간이동**시켰다.
        그러면 에피소드의 상당 부분이 "멀리서 접근하는" 구간이 되고, 정책은 큰 오차를
        빠르게 줄이는 **공격적인 이득**을 그 구간에서 배운다. 그 습성이 정밀 구간까지
        따라와 떨림으로 나타난다 (GUI 관찰, 2026-08-05).

        실제 연구 시나리오는 다르다. **물체를 이미 파지한 채로 운송**하므로::

            시작    EE 가 이미 물체 파지점에 있다  ->  오차 0 에서 출발
            이후    물체가 움직이는 만큼 목표가 연속 이동

        접근 구간이 아예 없다. 그 조건으로 학습해야 추종에 특화된다.

        목표가 멀리 못 가는 것은 아니다 — waypoint 를 향해 최대 0.3 m/s 로 계속
        움직이므로 24 초 에피소드에서 최대 7 m 를 이동한다. **순간이동만 없앤 것이다.**
        """
        f = self._needs_init
        if not f.any():
            return
        self.pose_command_w[f, :3] = self.robot.data.body_pos_w[f, self.body_idx]
        self.pose_command_w[f, 3:] = quat_unique(self.robot.data.body_quat_w[f, self.body_idx])
        self._new_waypoint(f.nonzero().flatten())
        self._needs_init[f] = False

    def _update_command(self):
        if self.replay_pose_w is not None:
            # **재생이 데이터셋보다 우선한다.** 초판은 반대라 preview_plan.py 가
            # 계획 파일을 넣어도 데이터셋 궤적이 재생돼 숫자가 무의미했다.
            # 재생 모드: 궤적을 그대로 쓰고 내부 이동은 건너뛴다.
            self.pose_command_w[:] = self.replay_pose_w
            if self.replay_lin_vel_w is not None:
                self.target_lin_vel_w = self.replay_lin_vel_w
            if self.replay_ang_vel_w is not None:
                self.target_ang_vel_w = self.replay_ang_vel_w
            self.metrics["target_speed"] = torch.norm(self.target_lin_vel_w, dim=1)
            if self.replay_base_ref_w is not None:
                self.base_ref_w[:] = self.replay_base_ref_w
            if self.replay_base_ref_twist is not None:
                self.base_ref_twist[:] = self.replay_base_ref_twist
        else:
            # **기하 합성을 먼저 돌리고, 데이터셋 env 만 덮어쓴다.**
            #   순서가 반대면 부모가 전체 env 를 다시 써 버려 데이터셋이 지워진다.
            #   부모 로직을 복사해 마스킹하는 대신 한 번 더 도는 쪽을 택했다 —
            #   중복 계산은 싸지만, 갈라진 두 벌이 어긋나면 조용히 틀린다.
            if self.cfg.init_at_ee:
                self._init_at_ee()
            # 부모가 목표를 전진시키고 **참값** 기준으로 pose_command_b 를 채운다.
            super()._update_command()
            if self.cfg.base_ref:
                self._update_base_ref()
            if self._ds_ee is not None:
                # MPC 궤적 데이터셋 — EE 목표와 차체 참조를 둘 다 여기서 덮어쓴다
                self._update_from_plan()
        # 그 위에 AMCL 추정 기준으로 덮어쓴다 (정책이 보는 값).
        #   부모의 변환을 한 번 버리는 셈이라 약간 낭비지만, 부모의 이동 로직을
        #   복사하지 않는 편이 어긋날 위험이 훨씬 작다.
        est_pos, est_quat = self._amcl(self.robot)
        self.pose_command_b[:, :3], self.pose_command_b[:, 3:] = subtract_frame_transforms(
            est_pos, est_quat, self.pose_command_w[:, :3], self.pose_command_w[:, 3:]
        )
        # .. warning::
        #    여기에 있던 ``if base_ref and _ds_ee is None: _update_base_ref()`` 를 위로
        #    옮겼다. 이 자리에 있으면 **재생 모드에서 주입한 차체 참조를 덮어썼다.**
        #    데이터셋이 있는 v8/v9/v11 은 조건에 걸려 살아남았지만, v6 은 매 스텝
        #    기하 참조로 되돌려졌다 — 즉 v6 만 학습과 같은(유리한) 입력을 받고
        #    나머지는 실제 계획의 참조를 받는, **불공평한 비교**였다.

    def _update_metrics(self):
        super()._update_metrics()
        if self.cfg.base_ref:
            r = self.robot.data
            self.metrics["base_ref_error"] = torch.norm(
                self.base_ref_w[:, :2] - r.root_pos_w[:, :2], dim=1)
            self.metrics["base_ref_yaw_error"] = _wrap_pi(
                (self.base_ref_w[:, 2] - r.heading_w).unsqueeze(1)).squeeze(1).abs()

    def _update_base_ref(self):
        """EE 목표에서 **차체 참조 궤적**을 유도한다 ([가-1]).

        왜 유도하나
        -----------
        실기에서는 플래너가 ``/base_pose`` / ``/base_twist`` 를 준다 — 장애물을 피해
        푼 결과다. 시뮬에는 플래너가 없으므로 만들어야 하는데, **아무 데나 두면
        EE 목표와 모순**된다 (그 자리에서 EE 에 닿지 않는다).

        플래너 구조를 흉내낸다. 플래너도 차체를 "reach 고리 안" 에 배치한다::

            차체 참조 = EE 목표 − R(진행방향) * 스탠드오프

        스탠드오프를 반드시 랜덤화한다
        ------------------------------
        **고정하면 정책이 그 값을 외우고 참조를 안 읽는다.** 그러면 관측을 넣은 의미가
        없다. 에피소드마다 거리와 방위를 새로 뽑아, 참조를 읽어야만 맞출 수 있게 한다::

            거리   0.15 ~ 0.45 m      실측 자연 스탠드오프 0.244 를 감싼다
            방위   180 +- 60 도       팔이 뒤를 보므로 목표는 차체 뒤쪽에 온다

        방위 기준이 180 도인 이유는 실측이다 — 정책이 목표를 차체 기준 뒤쪽
        (x 중앙 -0.122 m) 에 두고 선다.

        진행 방향
        ---------
        차체 참조의 yaw 는 **목표의 진행 방향**으로 둔다. 그래야 차체가 전진으로
        움직인다 (후진 상한 0.2 m/s 에 걸리지 않는다). 목표가 거의 멈춰 있으면
        직전 방향을 유지한다.
        """
        v = self.target_lin_vel_w[:, :2]
        speed = torch.norm(v, dim=1)
        moving = speed > 1e-3
        yaw = torch.where(moving, torch.atan2(v[:, 1], v[:, 0]), self._prev_ref_yaw)
        d, phi = self._standoff[:, 0], self._standoff[:, 1]
        # 목표를 차체 프레임에서 (d, phi) 에 두려면 차체는 그만큼 반대편에 있어야 한다
        off = torch.stack([d * torch.cos(phi), d * torch.sin(phi)], dim=1)
        c, s = torch.cos(yaw), torch.sin(yaw)
        world_off = torch.stack([c * off[:, 0] - s * off[:, 1], s * off[:, 0] + c * off[:, 1]], dim=1)
        self.base_ref_w[:, :2] = self.pose_command_w[:, :2] - world_off
        self.base_ref_w[:, 2] = yaw
        dyaw = torch.atan2(torch.sin(yaw - self._prev_ref_yaw), torch.cos(yaw - self._prev_ref_yaw))
        self.base_ref_twist[:, 0] = speed
        self.base_ref_twist[:, 1] = dyaw / self._env.step_dt
        self._prev_ref_yaw = yaw

    def _resample_plan(self, env_ids):
        """에피소드마다 **MPC 궤적 하나**를 뽑고 시작 시점을 무작위로 정한다.

        시작 시점을 흔드는 이유는 궤적 개수가 수십 개뿐이라 앞부분만 반복 학습하는
        것을 막기 위해서다. 에피소드(24 s)가 궤적보다 길면 마지막 점에서 유지된다.
        """
        n = int(env_ids.numel()) if hasattr(env_ids, "numel") else len(env_ids)
        idx = torch.randint(0, self._ds_ee.shape[0], (n,), device=self.device)
        self._ds_idx[env_ids] = idx
        # 길이는 **궤적마다 다르다.** 초판은 ``self._ds_ee[0, -1, 0]`` 하나로 전부를
        # 대신했는데, 실측 데이터셋은 전부 9 초라 안 드러났을 뿐이다.
        dur = self._ds_ee[idx, -1, 0]
        room = (dur - self._env.max_episode_length_s).clamp_min(0.0)
        self._ds_t0[env_ids] = torch.rand(n, device=self.device) * room
        self._ep_t[env_ids] = 0.0
        # --- 좌표 기준을 **t0 시점**으로 잡는다 ---
        #
        # .. warning::
        #    기준을 궤적 **시작점**으로 두면, t0 를 흔든 만큼 참조가 로봇보다 앞서
        #    시작한다. 로봇은 env 원점에 스폰되기 때문이다. 궤적이 길수록 심하다::
        #
        #        실측 122 개 (9 초)    앞선 거리 ~0.25 m   -> 안 드러남
        #        생성본 (105 초)       앞선 거리 중앙 3.06 m, 최대 5.95 m
        #
        #    그 상태로 돌린 v12 첫 판은 19 it 에서 위치오차 2.17 m 였다. 추종이 아니라
        #    **멀리 있는 목표로 달려가기**를 학습한다.
        self._ds_b0[env_ids] = self._sample_base_at(idx, self._ds_t0[env_ids])
        # --- 재생 배속 ---
        #   시간축만 늘려 속도 분포를 넓힌다 (궤적을 새로 풀지 않고).
        lo, hi = self.cfg.plan_time_scale
        self._ds_scale[env_ids] = lo + (hi - lo) * torch.rand(n, device=self.device)
        # --- 데이터셋 / 기하 합성 섞기 ---
        #   에피소드마다 어느 쪽을 쓸지 정한다. ``plan_mix`` 가 1.0 이면 전부 데이터셋.
        self._use_ds[env_ids] = torch.rand(n, device=self.device) < self.cfg.plan_mix
        # 목표 높이 — 범위가 주어지면 에피소드마다 새로 뽑는다.
        r = self.cfg.plan_goal_z_range
        if r is not None:
            self._ds_goal_z[env_ids] = r[0] + (r[1] - r[0]) * torch.rand(n, device=self.device)

    def _sample_base_at(self, idx, t):
        """궤적 ``idx`` 의 시각 ``t`` 에서 차체 자세 (x, y, yaw) 를 선형보간한다."""
        B = self._ds_base[idx]
        ts = B[:, :, 0].contiguous()
        i = torch.searchsorted(ts, t.unsqueeze(1)).squeeze(1).clamp(1, ts.shape[1] - 1)
        ar = torch.arange(len(idx), device=self.device)
        t0, t1 = ts[ar, i - 1], ts[ar, i]
        f = ((t - t0) / (t1 - t0).clamp_min(1e-6)).clamp(0.0, 1.0).unsqueeze(1)
        b0, b1 = B[ar, i - 1, 1:], B[ar, i, 1:]
        return b0 + f * torch.cat([b1[:, :2] - b0[:, :2],
                                   _wrap_pi(b1[:, 2:3] - b0[:, 2:3])], dim=1)

    def _update_from_plan(self):
        """데이터셋 궤적에서 EE 목표와 차체 참조를 읽는다 (선형 보간)."""
        self._ep_t += self._env.step_dt
        # 배속을 **경과 시간에만** 곱한다. 시작 시점 t0 까지 곱하면 배속이 바뀔 때마다
        # 궤적의 다른 부분에서 출발하게 돼 두 랜덤화가 얽힌다.
        t = self._ep_t * self._ds_scale + self._ds_t0
        E = self._ds_ee[self._ds_idx]      # (N, T, 4) = t, x, y, yaw
        B = self._ds_base[self._ds_idx]
        ts = E[:, :, 0]
        # 각 env 에서 t 가 들어가는 구간을 찾는다
        i = torch.searchsorted(ts.contiguous(), t.unsqueeze(1)).squeeze(1)
        i = i.clamp(1, ts.shape[1] - 1)
        ar = torch.arange(self.num_envs, device=self.device)
        t0, t1 = ts[ar, i - 1], ts[ar, i]
        f = ((t - t0) / (t1 - t0).clamp_min(1e-6)).clamp(0.0, 1.0).unsqueeze(1)

        e0, e1 = E[ar, i - 1, 1:], E[ar, i, 1:]
        b0, b1 = B[ar, i - 1, 1:], B[ar, i, 1:]
        ee = e0 + f * torch.cat([e1[:, :2] - e0[:, :2],
                                 _wrap_pi(e1[:, 2:3] - e0[:, 2:3])], dim=1)
        bs = b0 + f * torch.cat([b1[:, :2] - b0[:, :2],
                                 _wrap_pi(b1[:, 2:3] - b0[:, 2:3])], dim=1)

        # env 원점 기준으로 옮긴다 — 궤적은 map 절대좌표라 그대로 쓰면 env 밖으로 나간다.
        # 궤적 **시작 차체 pose** 를 원점에 맞추면 로봇과의 상대 기하가 보존된다.
        # 기준은 **t0 시점의 차체 자세**다 (궤적 시작점이 아니다 — _resample_plan 참고).
        b_start = self._ds_b0
        c, s_ = torch.cos(-b_start[:, 2]), torch.sin(-b_start[:, 2])
        def _rel(p):
            d = p[:, :2] - b_start[:, :2]
            return torch.stack([c * d[:, 0] - s_ * d[:, 1], s_ * d[:, 0] + c * d[:, 1]], dim=1)
        org = self._env.scene.env_origins
        ee_xy, bs_xy = _rel(ee) + org[:, :2], _rel(bs) + org[:, :2]

        # --- 여기부터는 **데이터셋을 쓰는 env 에만** 쓴다 ---
        #   나머지 env 는 기하 합성 경로가 이미 채워 놨다. 마스크 없이 통째로 쓰면
        #   섞기(plan_mix)가 무의미해진다.
        m = self._use_ds
        m1 = m.unsqueeze(1)

        dt = self._env.step_dt
        # 데이터셋의 요는 **막대 방향**이다. tool0 목표 요로 바꾸려면 90 도를 뺀다.
        yaw = ee[:, 2] - b_start[:, 2] + PLAN_YAW_OFFSET
        z = torch.zeros_like(yaw)
        quat = quat_unique(quat_from_euler_xyz(z, z + torch.pi, yaw))
        self.pose_command_w[:, :2] = torch.where(m1, ee_xy, self.pose_command_w[:, :2])
        self.pose_command_w[:, 2] = torch.where(
            m, org[:, 2] + self._ds_goal_z, self.pose_command_w[:, 2])
        self.pose_command_w[:, 3:] = torch.where(m1, quat, self.pose_command_w[:, 3:])

        started = (self._ep_t > dt * 1.5)
        v = torch.where((started & m).unsqueeze(1), (ee_xy - self._prev_ee_xy) / dt,
                        torch.zeros_like(ee_xy))
        self.target_lin_vel_w[:, :2] = torch.where(m1, v, self.target_lin_vel_w[:, :2])
        self._prev_ee_xy = ee_xy.clone()
        self.metrics["target_speed"] = torch.norm(self.target_lin_vel_w, dim=1)

        bs_yaw = bs[:, 2] - b_start[:, 2]
        vb = torch.norm((bs_xy - self._prev_base_xy) / dt, dim=1)
        wb = _wrap_pi((bs_yaw - self._prev_base_yaw).unsqueeze(1)).squeeze(1) / dt
        self.base_ref_w[:, :2] = torch.where(m1, bs_xy, self.base_ref_w[:, :2])
        self.base_ref_w[:, 2] = torch.where(m, bs_yaw, self.base_ref_w[:, 2])
        self.base_ref_twist[:, 0] = torch.where(
            m, torch.where(started, vb, torch.zeros_like(vb)), self.base_ref_twist[:, 0])
        self.base_ref_twist[:, 1] = torch.where(
            m, torch.where(started, wb, torch.zeros_like(wb)), self.base_ref_twist[:, 1])
        self._prev_base_xy = bs_xy.clone()
        # 캐시는 **데이터셋 값**으로 둔다. base_ref_w 를 그대로 받으면 기하 env 의
        # 값이 섞여 들어가, 그 env 가 나중에 데이터셋으로 바뀔 때 각속도가 튄다.
        self._prev_base_yaw = bs_yaw.clone()

    def _resample_base_ref(self, env_ids):
        """에피소드마다 스탠드오프를 새로 뽑는다 (참조를 외우지 못하게)."""
        n = len(env_ids) if hasattr(env_ids, "__len__") else int(env_ids.numel())
        r = torch.empty(n, device=self.device)
        self._standoff[env_ids, 0] = r.uniform_(*self.cfg.standoff_range)
        self._standoff[env_ids, 1] = torch.pi + torch.empty(
            n, device=self.device).uniform_(-self.cfg.standoff_bearing, self.cfg.standoff_bearing)
        self._prev_ref_yaw[env_ids] = self.robot.data.heading_w[env_ids]

    @property
    def amcl_estimate(self) -> torch.Tensor:
        """현재 AMCL 추정값 (x, y, yaw). 로깅/디버깅용."""
        return self._amcl.estimate


@configclass
class MovingPoseWorldAmclCommandCfg(MovingPoseWorldCommandCfg):
    """:class:`MovingPoseWorldAmclCommand` 설정.

    AMCL 기본값은 **실기 로그 실측치**다 (``scripts/analyze_amcl_logs.py``,
    ``~/ros2_ws/src/AFL_nav2/params`` + ``~/ros2_ws/src/visualizer/data``).
    :class:`UniformPoseWorldAmclCommandCfg` 의 값과 같아야 한다 — 한쪽만 고치지 말 것.
    """

    class_type: type = MovingPoseWorldAmclCommand

    plan_dataset: str = ""
    """실제 MPC 궤적 데이터셋 경로 (``scripts/tools/gen_plan_dataset.py`` 산출물).

    비어 있으면 기존대로 waypoint 랜덤 궤적 + 기하 유도 차체 참조를 쓴다.
    지정하면 **EE 목표와 차체 참조를 둘 다 그 데이터에서** 가져온다.

    왜 필요한가::

        v6 은 차체 참조를 EE 목표에서 기하로 유도했다 (거리 0.15~0.45 m 가정)
        실측 플래너 궤적의 EE-차체 거리는 중앙 0.462 m (5% 0.339, 95% 0.551)
        -> 거의 안 겹친다.  그래서 학습한 관계가 실제와 달랐다
           (차체 편차 합성 0.163 vs 실제 0.527 m)
    """

    plan_goal_z: float = 0.55
    """데이터셋 재생 시 EE 목표 높이 [m], env 원점 기준. 플래너는 2D 라 여기서 얹는다."""

    plan_goal_z_range: tuple[float, float] | None = None
    """데이터셋 재생 시 EE 목표 높이를 **에피소드마다** 뽑을 범위 [m].

    ``None`` 이면 :attr:`plan_goal_z` 고정값을 쓴다 (v8~v12 동작).

    .. warning::
       v12 는 0.55 m **한 점만** 학습해서 높이에 취약하다::

           0.55 m   EE 13.9 mm      0.60 m   EE 72.6 mm

       5 cm 벗어나니 5 배다. 실기에서 목표 높이를 정확히 맞춰야 하는 운용 제약이
       되고, 평가할 때도 높이를 안 맞추면 **정책이 망가진 것으로 오판**한다
       (실제로 그렇게 잘못 판정했다).
    """

    plan_test_n: int = 0
    """데이터셋 **뒤에서** 이만큼을 학습에서 뺀다 (시험 전용, held-out).

    ``scripts/eval_plans.py --test_n`` 과 **같은 값**을 써야 한다. 다르면 시험 궤적이
    학습에 섞여 들어가 "외운 것"을 성능으로 착각한다.

    0 이면 전부 학습에 쓴다 — 기존 v8/v9/v11 동작이다.
    """

    plan_time_scale: tuple[float, float] = (1.0, 1.0)
    """데이터셋 재생 배속 범위. 에피소드마다 하나 뽑는다.

    데이터셋 차체 속력이 **중앙 0.067 m/s, 95 % 0.118** 인데 목표는 0.2 m/s 다.
    궤적을 새로 풀지 않고 시간축만 줄여 속도 분포를 넓힌다.

    .. note::
       배속을 올리면 EE 목표 속도도 같이 오른다. 차체 속도 한계를 넘는 배속은
       "따라갈 수 없는 목표" 라 학습을 망친다. 3.0 을 넘기지 말 것.
    """

    plan_mix: float = 1.0
    """에피소드가 **데이터셋**을 쓸 확률. 나머지는 기하 합성 궤적을 쓴다.

    1.0 이면 데이터셋만 (v8/v9/v11 동작), 0.0 이면 기하만 (v6 동작).

    왜 섞나
    -------
    데이터셋만으로 학습하면 122 개를 외워 **일반화가 무너진다.** 같은 계획 재생::

        v6  기하만        EE  10.8 mm
        v8  데이터셋만    EE 118.2 mm
        v9  데이터셋만    EE 186.2 mm
        v11 데이터셋만    EE 216.7 mm

    데이터셋이 안 덮는 배치가 원인이다 (EE 뒤쪽 3.3 %, |EE요-차체요| > 60 도 12.3 %).
    """

    base_ref: bool = False
    """차체 참조 궤적을 만들지 ([가-1]).

    ``True`` 면 ``base_ref_w`` (x, y, yaw) 와 ``base_ref_twist`` (v, omega) 를 채운다.
    실기에서는 플래너가 주는 값에 해당한다.
    """

    standoff_range: tuple[float, float] = (0.15, 0.45)
    """차체 참조가 EE 목표에서 떨어지는 거리 [m]. 실측 자연 스탠드오프 0.244 를 감싼다.

    **반드시 랜덤이어야 한다.** 고정하면 정책이 외우고 참조를 안 읽는다.
    """

    standoff_bearing: float = 1.047
    """스탠드오프 방위의 흔들림 폭 [rad] = +-60 도. 기준은 180 도(차체 뒤쪽)다.

    팔이 뒤를 보므로 정책은 목표를 차체 뒤에 둔다 (실측 x 중앙 -0.122 m).
    """

    init_at_ee: bool = False
    """목표를 **현재 EE 위치에서** 출발시킬지.

    ``True`` 면 물체를 이미 파지한 채 운송하는 조건이 된다 (오차 0 에서 출발).
    ``False`` 면 월드 상자에서 무작위로 뽑아 멀리 순간이동시킨다 (접근 구간 포함).

    ``ranges.pos_*`` 는 ``True`` 일 때 **waypoint 샘플링에만** 쓰인다.
    """

    pos_noise_std: float = 0.02
    """정상상태 위치 추정 오차 [m]."""

    yaw_noise_std: float = 0.02
    """정상상태 yaw 추정 오차 [rad]. 목표가 2 m 앞이면 위치로 4 cm 가 된다."""

    update_period: float = 0.05
    """AMCL 갱신 주기 [s]. 그 사이에는 이전 추정값이 유지된다."""

    latency: float = 0.06
    """스캔 취득 -> 퍼블리시 지연 [s]. 버퍼 길이가 생성 시점에 고정되므로 런타임 변경 불가."""

    jump_prob: float = 0.0
    """스텝당 점프 확률. 0 이면 점프 없음."""

    jump_pos_std: float = 0.10
    jump_yaw_std: float = 0.10
