"""팔 역기구학 추종기 — Isaac Lab ``DifferentialIKController`` 래퍼.

왜 Isaac Lab 것을 쓰나
----------------------
이 저장소에는 이미 numpy 역기구학이 있다 (``scripts/chain_tool0.py``,
``ik_batch.py``). 그런데 그것은 **URDF 를 따로 파싱한 두 번째 기구학 모델**이다.
시뮬 안에서 제어할 때 그걸 쓰면, 모델이 갈라지는 순간 원인 불명의 오차가 생긴다 —
이 저장소가 반복해서 당한 사고가 정확히 그것이다.

그래서 **시뮬이 실제로 쓰는 관절 상태와 PhysX 자코비안**으로 푼다. 모델이 하나뿐이라
대조할 상대가 없다.

무엇을 푸는가
-------------
**가중** 감쇠 최소자승(DLS) 미분 IK 한 스텝::

    W = diag(w_pos·I₃, w_rot·I₃)
    Δq = (WJ)ᵀ((WJ)(WJ)ᵀ + λ²I)⁻¹ · W·e        e = [위치오차(3), 자세오차(3)]
    q_des = q + clip(Δq, ±max_step)

제어주기마다 한 스텝만 돈다 (Isaac Lab 표준 패턴). 30 Hz 에서 목표가 천천히 움직이므로
사실상 **작업공간 비례 제어기**로 동작한다. ``hold`` 국면에서 목표가 고정된 동안
수렴하므로 ``run`` 시작 시점에는 이미 붙어 있다.

왜 가중치가 필요한가 — 단위가 섞인다
------------------------------------
Isaac Lab 의 ``DifferentialIKController`` 는 오차를 ``cat(위치[m], 자세[rad])`` 로
쌓아 **같은 가중치**로 푼다. 그런데 1 rad = 57 도이고 1 m 는 이 팔에서 거의 전체
리치다. 즉 **자세 60 도(1.05 rad)가 위치 1 m 와 맞먹는다.**

실측으로 이게 물렸다 (2026-09-06). 목표가 리치 경계에 있는 궤적에서 팔이 자세를
맞추려다 **일자로 뻗은 채 위치 28 cm 를 포기**했다 (j3 +0.5 도, 특이점). 관절 한계에
걸린 것이 아니라 최소자승이 그렇게 타협한 것이다.

그래서 ``w_rot`` 을 두어 "자세 1 rad 을 위치 몇 m 로 칠 것인가" 를 고를 수 있게 한다.
``w_rot=1.0`` 이면 Isaac Lab 기본과 같다.

.. note::
   오차 계산 자체는 Isaac Lab 의 ``compute_pose_error`` 를 그대로 쓴다 (쿼터니언
   부호 처리가 들어 있다). 푸는 부분만 가중치를 받는 형태로 바꾼 것이다.

자코비안 색인 — 떠 있는 베이스라 다르다
---------------------------------------
고정 베이스면 자코비안에 베이스 행이 없어 ``body_idx − 1`` 이지만, 우리 로봇은
**떠 있는 베이스(mobile)** 라 행은 ``body_idx`` 그대로이고 **열이 앞에서 6 칸
밀린다** (베이스 6 자유도). Isaac Lab 의 ``DifferentialInverseKinematicsAction`` 이
쓰는 규칙과 같게 맞췄다.

프레임
------
컨트롤러는 **base(root) 프레임**에서 푼다. 그래서:

- 현재 EE pose 는 팔 FK 로 얻는다 (참값 root 기준). 실기의 관절 엔코더에 해당한다.
- 목표 pose 는 월드에서 오므로 **추정 base pose** 로 base 프레임에 옮긴다.
  실기에서 AMCL 오차가 들어오는 통로가 정확히 여기 하나뿐이고, 관측을 만드는
  ``episode._obs`` 도 같은 규약이다 (``ee_pose`` 는 참값, ``pose_command`` 만 추정값).
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.utils.math import (compute_pose_error, matrix_from_quat, quat_inv,
                                 subtract_frame_transforms)


class ArmIK:
    """로봇 한 대의 팔을 EE 목표로 끌고 가는 미분 IK. 관절 **목표각**을 낸다.

    Args:
        robot: ``Articulation``
        body_id: EE 바디 인덱스 (tool0)
        joint_ids: 팔 관절 인덱스 6 개
        lam: DLS 감쇠. 크면 특이점 근처에서 안정하지만 느리다
        w_rot: 자세 오차 가중치. "자세 1 rad 을 위치 몇 m 로 칠 것인가".
            1.0 이면 Isaac Lab 기본과 같다. 목표가 리치 경계에 있는 궤적에서는
            낮춰야 위치를 지킨다 (모듈 docstring 참고)
        max_step: 한 제어주기당 관절 변화 상한 [rad]. 특이점에서 튀는 것을 막는다
        clamp_limits: 관절 소프트 한계로 목표를 자른다. 실기 컨트롤러가 하는 일이다
    """

    def __init__(self, robot, body_id: int, joint_ids, *, lam: float = 0.05,
                 w_rot: float = 0.35, max_step: float = 0.15,
                 clamp_limits: bool = True, device: str = "cuda"):
        self.robot, self.body_id = robot, body_id
        self.joint_ids = list(joint_ids)
        self.device = device
        self.clamp_limits = clamp_limits
        self.lam = float(lam)
        self.w_rot = float(w_rot)
        self.max_step = float(max_step)
        self._eye6 = torch.eye(6, device=device).unsqueeze(0)

        # 떠 있는 베이스 -> 행은 그대로, 열은 6 칸 밀린다
        if robot.is_fixed_base:
            self._jac_body = body_id - 1
            self._jac_joints = self.joint_ids
        else:
            self._jac_body = body_id
            self._jac_joints = [i + 6 for i in self.joint_ids]

        lim = robot.data.soft_joint_pos_limits[0, self.joint_ids]      # (6, 2)
        self._q_lo = lim[:, 0].clone()
        self._q_hi = lim[:, 1].clone()

    # ------------------------------------------------------------------ 상태
    def ee_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        """현재 EE pose, **base 프레임**. 팔 FK 라 로컬라이제이션과 무관하다."""
        d = self.robot.data
        return subtract_frame_transforms(
            d.root_pos_w[:1], d.root_quat_w[:1],
            d.body_pos_w[:1, self.body_id], d.body_quat_w[:1, self.body_id])

    def jacobian_b(self) -> torch.Tensor:
        """EE 기하 자코비안, base 프레임. (1, 6, 6)"""
        jac = self.robot.root_physx_view.get_jacobians()[
            :1, self._jac_body, :, self._jac_joints]
        R = matrix_from_quat(quat_inv(self.robot.data.root_quat_w[:1]))
        jac = jac.clone()
        jac[:, :3, :] = torch.bmm(R, jac[:, :3, :])
        jac[:, 3:, :] = torch.bmm(R, jac[:, 3:, :])
        return jac

    # -------------------------------------------------------------------- 풀기
    def solve(self, tgt_pos_w: np.ndarray, tgt_quat_w: np.ndarray,
              base_pos_w: np.ndarray, base_quat_w: np.ndarray) -> dict:
        """월드 목표 -> 관절 목표각.

        Args:
            tgt_pos_w / tgt_quat_w: 월드 EE 목표 (wxyz)
            base_pos_w / base_quat_w: **추정** base pose. 목표를 base 프레임으로
                옮길 때만 쓴다 (실기의 AMCL 자리)

        Returns:
            ``q_des`` (6,) 관절 목표각과 진단값.
        """
        dev = self.device
        t_p = torch.as_tensor(np.asarray(tgt_pos_w, dtype=np.float32),
                              device=dev).view(1, 3)
        t_q = torch.as_tensor(np.asarray(tgt_quat_w, dtype=np.float32),
                              device=dev).view(1, 4)
        b_p = torch.as_tensor(np.asarray(base_pos_w, dtype=np.float32),
                              device=dev).view(1, 3)
        b_q = torch.as_tensor(np.asarray(base_quat_w, dtype=np.float32),
                              device=dev).view(1, 4)

        # 목표를 base 프레임으로 (추정 base 기준)
        tgt_p_b, tgt_q_b = subtract_frame_transforms(b_p, b_q, t_p, t_q)
        ee_p_b, ee_q_b = self.ee_pose_b()
        q_now = self.robot.data.joint_pos[:1, self.joint_ids]

        # 오차는 Isaac Lab 것을 그대로 (쿼터니언 부호 처리 포함), 푸는 것만 가중 DLS
        pos_err, rot_err = compute_pose_error(
            ee_p_b, ee_q_b, tgt_p_b, tgt_q_b, rot_error_type="axis_angle")
        res_pos = float(torch.norm(pos_err))
        res_ang = float(torch.norm(rot_err))

        e = torch.cat([pos_err, rot_err * self.w_rot], dim=1).unsqueeze(-1)  # (1,6,1)
        J = self.jacobian_b()
        J = torch.cat([J[:, :3, :], J[:, 3:, :] * self.w_rot], dim=1)
        JT = J.transpose(1, 2)
        dq = JT @ torch.linalg.solve(J @ JT + (self.lam ** 2) * self._eye6, e)
        dq = dq.squeeze(-1).clamp(-self.max_step, self.max_step)
        q_des = q_now + dq

        if self.clamp_limits:
            q_des = torch.max(torch.min(q_des, self._q_hi), self._q_lo)

        return {
            "q_des": q_des[0].detach().cpu().numpy().astype(np.float64),
            "res_pos": res_pos,          # base 프레임 위치 잔차 [m]
            "res_ang": res_ang,          # 자세 잔차 [rad]
            "reach": float(torch.norm(tgt_p_b[:, :2])),   # 차체 중심에서 목표까지 [m]
            # 한계에 붙었는가 — 하나라도 걸리면 True. and/or 를 섞으면 우선순위 때문에
            # (A and B) or C 로 읽히므로 괄호를 명시한다.
            "clamped": bool(self.clamp_limits and bool(
                (q_des[0] <= self._q_lo + 1e-6).any()
                or (q_des[0] >= self._q_hi - 1e-6).any())),
        }

    def reset(self) -> None:
        """상태 없는 제어기라 할 일이 없다. 인터페이스만 맞춘다."""


def target_world_from_stream(tgt: dict) -> tuple[np.ndarray, np.ndarray]:
    """``TargetStream.step()`` 산출물에서 월드 EE 목표를 꺼낸다.

    정책 경로와 **같은 목표**를 쓰는 것이 이 비교의 전제다. 여기서 목표를 다시
    만들면 "제어기가 더 쉬운 목표를 받았다" 는 반론을 막을 수 없다.
    """
    return np.asarray(tgt["pos_w"], dtype=np.float64), np.asarray(tgt["quat_w"],
                                                                  dtype=np.float64)


__all__ = ["ArmIK", "target_world_from_stream"]
