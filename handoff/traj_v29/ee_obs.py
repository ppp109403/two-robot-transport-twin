"""궤적 추종 정책(52차원)의 **관측 조립** — 실기용 순수 numpy 구현.

왜 별도 모듈인가
----------------
관측 벡터를 손으로 조립하는 것이 이 배포에서 가장 위험한 부분이다. 한 칸만
어긋나도 **에러 없이 조용히 이상한 값**을 내고, 그게 가장 찾기 어려운 버그다.

실제로 이 프로젝트에서 팔 정책(42)과 통합 정책(42)이 **차원은 같은데 내용도
순서도 전혀 다른** 상태가 있었다. 차원 검사로는 안 걸린다.

그래서 조립 로직을 노드에서 떼어내 **단독으로 자가검증 가능**하게 만든다::

    python3 ee_obs.py        # 시뮬과 맞춘 기준값으로 자체 검사

시뮬 쪽 원본
------------
::

    ee_track_ppo/tasks/manager_based/reach/mdp/observations.py
    ee_track_ppo/tasks/manager_based/reach/config/amr_fr3/traj_env_cfg.py

배치가 바뀌면 ``python scripts/dump_obs_layout.py --task Reach-AMR-FR3-Traj-v0``
로 다시 뽑아 이 파일의 :data:`LAYOUT` 과 대조할 것.

쿼터니언 규약
-------------
**wxyz** (Isaac Lab 규약). ROS 는 **xyzw** 이므로 경계에서 반드시 바꿔야 한다.
:func:`quat_from_ros` / :func:`quat_to_ros` 를 쓸 것.
"""

from __future__ import annotations

import numpy as np

OBS_DIM = 52
ACTION_DIM = 8

LAYOUT = [
    ("arm_joint_pos", 6),  # [ 0: 6] j1..j6 기본자세 기준 상대각 [rad]
    ("arm_joint_vel", 6),  # [ 6:12] j1..j6 [rad/s]
    ("base_twist", 2),     # [12:14] [v_x, omega_z]  /odom 의 twist
    ("pose_command", 9),   # [14:23] 목표 xyz 3 + rot6d 6   (base 프레임)
    ("ee_pose", 9),        # [23:32] 현재 EE xyz 3 + rot6d 6 (base 프레임)
    ("ee_pose_error", 6),  # [32:38] 위치오차 3 + 자세오차 axis-angle 3
    ("actions", 8),        # [38:46] 직전 스텝의 raw action
    ("target_vel", 6),     # [46:52] 목표 선속도 3 + 각속도 3 (base 프레임)
]
"""관측 배치. ``scripts/dump_obs_layout.py`` 실측값 (2026-08-04)."""


def slices() -> dict[str, slice]:
    """항목 이름 -> 구간. 디버깅·검증용."""
    out, i = {}, 0
    for name, n in LAYOUT:
        out[name] = slice(i, i + n)
        i += n
    return out


##
# 쿼터니언 (wxyz)
##


def quat_from_ros(x: float, y: float, z: float, w: float) -> np.ndarray:
    """ROS(xyzw) -> Isaac Lab(wxyz)."""
    return np.array([w, x, y, z], dtype=np.float64)


def quat_to_ros(q: np.ndarray) -> tuple[float, float, float, float]:
    """Isaac Lab(wxyz) -> ROS(xyzw)."""
    return float(q[1]), float(q[2]), float(q[3]), float(q[0])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """단위 쿼터니언(wxyz) -> 3x3 회전행렬."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return matrix_from_quat(q) @ v


def quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return matrix_from_quat(q).T @ v


def quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw * 0.5), 0.0, 0.0, np.sin(yaw * 0.5)])


def yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


##
# 회전 표현
##


def rot6d_from_quat(q: np.ndarray) -> np.ndarray:
    """쿼터니언(wxyz) -> 회전행렬의 **첫 두 열**. (4,) -> (6,)

    레이아웃 (시뮬과 반드시 일치)::

        [ R00, R10, R20,   R01, R11, R21 ]
          <-- 1열 -->      <-- 2열 -->

    왜 쿼터니언을 안 쓰는가
    -----------------------
    이 태스크의 목표 자세는 ``pitch = pi`` 근처(손을 아래로)라 쿼터니언이 전부
    **w ~= 0** 인 특이면 위에 있다. 실기에서 이 w 는 부동소수점 잡음이 되고,
    부호 정규화(q 와 -q 는 같은 회전)가 목표와 현재를 **서로 다른 반구로 보내는**
    사고가 실제로 났다.

    무서운 점은 **어떤 오차 지표로도 안 잡혔다**는 것이다. 오차항은 axis-angle 이라
    정상적으로 0 을 가리켰는데, 정책은 본 적 없는 부호의 입력을 받아 j6 액션이
    포화했고 명령이 기본자세에서 53 도 튀었다.

    회전행렬은 SO(3) -> R^9 의 일대일 사상이라 부호 이중성이 원천적으로 없다.
    """
    r = matrix_from_quat(q)
    return np.concatenate([r[:, 0], r[:, 1]])


def axis_angle_from_quat(q: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """쿼터니언(wxyz) -> axis-angle 3벡터. 크기가 곧 회전각 [rad].

    Isaac Lab ``axis_angle_from_quat`` 과 같은 식이다 (w 를 양수로 정규화한 뒤
    ``sin(half)/angle`` 로 나눈다). 작은 각에서 테일러 전개로 넘어가는 분기까지 맞췄다.
    """
    q = np.asarray(q, dtype=np.float64)
    if q[0] < 0.0:
        q = -q
    mag = np.linalg.norm(q[1:])
    half_angle = np.arctan2(mag, q[0])
    angle = 2.0 * half_angle
    if abs(angle) > eps:
        s = np.sin(half_angle) / angle
    else:
        s = 0.5 - angle * angle / 48.0
    return q[1:4] / s


def subtract_frame_transforms(
    t01: np.ndarray, q01: np.ndarray, t02: np.ndarray, q02: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """프레임 0 기준의 두 pose 로부터 **1 기준 2 의 pose** 를 구한다.

    ``t12 = R01^T (t02 - t01)``,  ``q12 = q01^-1 * q02``
    """
    q10 = quat_conj(q01)
    return quat_apply(q10, t02 - t01), quat_mul(q10, q02)


##
# 관측 조립
##


def build_obs(
    arm_joint_pos_rel: np.ndarray,
    arm_joint_vel: np.ndarray,
    base_lin_x: float,
    base_ang_z: float,
    base_pos_w: np.ndarray,
    base_quat_w: np.ndarray,
    ee_pos_w: np.ndarray,
    ee_quat_w: np.ndarray,
    goal_pos_w: np.ndarray,
    goal_quat_w: np.ndarray,
    goal_lin_vel_w: np.ndarray,
    goal_ang_vel_w: np.ndarray,
    last_action: np.ndarray,
) -> np.ndarray:
    """52 차원 관측을 만든다.

    좌표계
    ------
    ==================== ==========================================================
    ``*_w``              **map(월드) 프레임**
    ``base_pos_w``       AMCL 이 추정한 ``base_link`` 위치. **참값이 아니라 추정값**
    ``ee_pos_w``         현재 EE(tool0) 의 map 프레임 pose
    출력 관측             전부 **base_link 프레임**
    ==================== ==========================================================

    .. warning::
       ``base_*_w`` 에 넣는 것은 **AMCL 추정값**이다. 정책은 "내가 어디 있는지
       정확히 모른다" 는 조건으로 학습됐고, 시뮬도 추정값을 관측에 준다.
       여기에 다른 (더 정확한) 값을 넣으면 학습 조건과 어긋난다.

    .. note::
       ``ee_pos_w`` 를 만들 때 쓰는 base pose 도 같은 AMCL 추정값이어야
       ``ee_pose`` 가 base 프레임에서 정확해진다. 팔 FK 로 base 기준 EE 를 직접
       구할 수 있으면 그쪽이 낫다 — AMCL 오차가 두 번 들어가지 않는다.

    .. note::
       시뮬의 ``target_vel`` 은 **참값 yaw** 로 회전시킨다 (``ee_target_vel_b``).
       실기에는 참값이 없으니 AMCL yaw 를 쓴다. yaw 오차 0.02 rad 는 속도 방향을
       약 1.1 도 틀리게 하므로 영향은 작다.
    """
    base_pos_w = np.asarray(base_pos_w, dtype=np.float64)
    base_quat_w = np.asarray(base_quat_w, dtype=np.float64)

    # 목표를 base 프레임으로
    cmd_pos_b, cmd_quat_b = subtract_frame_transforms(
        base_pos_w, base_quat_w, np.asarray(goal_pos_w, dtype=np.float64), np.asarray(goal_quat_w, dtype=np.float64)
    )
    # 현재 EE 를 base 프레임으로
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        base_pos_w, base_quat_w, np.asarray(ee_pos_w, dtype=np.float64), np.asarray(ee_quat_w, dtype=np.float64)
    )
    # 오차 — 위치는 단순 차, 자세는 상대회전의 axis-angle
    pos_err = cmd_pos_b - ee_pos_b
    ori_err = axis_angle_from_quat(quat_mul(cmd_quat_b, quat_conj(ee_quat_b)))
    # 목표 속도를 base 프레임으로 (회전만 적용)
    lin_b = quat_apply_inverse(base_quat_w, np.asarray(goal_lin_vel_w, dtype=np.float64))
    ang_b = quat_apply_inverse(base_quat_w, np.asarray(goal_ang_vel_w, dtype=np.float64))

    obs = np.concatenate([
        np.asarray(arm_joint_pos_rel, dtype=np.float64).reshape(6),
        np.asarray(arm_joint_vel, dtype=np.float64).reshape(6),
        np.array([base_lin_x, base_ang_z], dtype=np.float64),
        cmd_pos_b, rot6d_from_quat(cmd_quat_b),
        ee_pos_b, rot6d_from_quat(ee_quat_b),
        pos_err, ori_err,
        np.asarray(last_action, dtype=np.float64).reshape(ACTION_DIM),
        lin_b, ang_b,
    ])
    assert obs.shape == (OBS_DIM,), "관측 차원이 %d (기대 %d)" % (obs.shape[0], OBS_DIM)
    return obs.astype(np.float32)


##
# 자가검증
##


def _self_test():
    """``ROT6D_CONTRACT.md`` 의 기준값과 대조한다."""
    ok = True

    def check(tag, got, want, tol=1e-5):
        nonlocal ok
        good = np.allclose(got, want, atol=tol)
        ok &= good
        print("  %-28s %s" % (tag, "OK" if good else "실패"))
        if not good:
            print("     got  ", np.round(got, 6))
            print("     want ", np.round(want, 6))

    print("rot6d — ROT6D_CONTRACT.md 기준값")
    check("단위", rot6d_from_quat(np.array([1.0, 0, 0, 0])),
          [1, 0, 0, 0, 1, 0])
    check("pitch=pi, yaw=0", rot6d_from_quat(np.array([0.0, 0.0, 1.0, 0.0])),
          [-1, 0, 0, 0, 1, 0])
    check("pitch=pi, yaw=+90", rot6d_from_quat(np.array([0.0, -0.707107, 0.707107, 0.0])),
          [0, -1, 0, -1, 0, 0])
    check("pitch=pi-0.5, yaw=-90",
          rot6d_from_quat(np.array([0.174941, 0.685125, 0.685125, -0.174941])),
          [0, 0.877583, -0.479426, 1, 0, 0])
    check("임의값", rot6d_from_quat(np.array([0.5, 0.5, 0.5, 0.5])),
          [0, 1, 0, 0, 0, 1])

    print("부호 이중성 — q 와 -q 가 같은 6D 를 내는가")
    q = np.array([0.174941, 0.685125, 0.685125, -0.174941])
    check("q vs -q", rot6d_from_quat(q), rot6d_from_quat(-q))

    print("프레임 변환 왕복")
    bq = quat_from_yaw(0.7)
    bp = np.array([1.0, 2.0, 0.0])
    tp = np.array([1.5, 2.5, 0.6])
    tq = quat_mul(quat_from_yaw(0.2), np.array([0.0, 0.0, 1.0, 0.0]))
    rp, rq = subtract_frame_transforms(bp, bq, tp, tq)
    back_p = bp + quat_apply(bq, rp)
    check("위치 왕복", back_p, tp)
    check("자세 왕복", matrix_from_quat(quat_mul(bq, rq)), matrix_from_quat(tq))

    print("axis-angle")
    check("무회전", axis_angle_from_quat(np.array([1.0, 0, 0, 0])), [0, 0, 0])
    check("z 축 0.3 rad", axis_angle_from_quat(quat_from_yaw(0.3)), [0, 0, 0.3])

    print("관측 조립")
    obs = build_obs(
        arm_joint_pos_rel=np.zeros(6), arm_joint_vel=np.zeros(6),
        base_lin_x=0.2, base_ang_z=0.1,
        base_pos_w=np.array([1.0, 2.0, 0.0]), base_quat_w=quat_from_yaw(0.5),
        ee_pos_w=np.array([1.3, 2.1, 0.5]), ee_quat_w=np.array([0.0, 0.0, 1.0, 0.0]),
        goal_pos_w=np.array([1.4, 2.2, 0.5]), goal_quat_w=np.array([0.0, 0.0, 1.0, 0.0]),
        goal_lin_vel_w=np.array([0.2, 0.0, 0.0]), goal_ang_vel_w=np.zeros(3),
        last_action=np.zeros(8),
    )
    sl = slices()
    check("차원", [obs.shape[0]], [OBS_DIM])
    check("base_twist", obs[sl["base_twist"]], [0.2, 0.1])
    # 목표와 EE 의 자세가 같으므로 자세오차는 0
    check("자세오차 0", obs[sl["ee_pose_error"]][3:], [0, 0, 0])
    # 위치오차는 목표-EE 를 base 로 돌린 것
    check("위치오차", obs[sl["ee_pose_error"]][:3],
          quat_apply_inverse(quat_from_yaw(0.5), np.array([0.1, 0.1, 0.0])))

    print()
    print("전체", "통과" if ok else "실패 — 배포 금지")
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if _self_test() else 1)
