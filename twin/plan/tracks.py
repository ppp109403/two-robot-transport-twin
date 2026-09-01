"""로봇 한 대가 따라갈 **독립 궤적** 하나.

왜 별도 개념인가
----------------
초판은 협력 운송 계획(물체 + base_A/B + ee_A/ee_B)에서 두 로봇의 목표를 **유도**했다.
그건 편대를 검증하는 구성이고, 편대는 한 대가 계속 후진해야 해서 v29 학습 분포 밖이다
(README 의 Phase 1 결과 참고).

여기서 원하는 것은 그게 아니다 — **두 대를 한 씬에 올려 각자 자기 궤적을 자기 정책으로
추종**하는 것이다. 그러면 두 로봇은 서로 독립이고, 각자에게 줄 궤적은
**학습이 실제로 본 분포**에서 가져오는 것이 맞다.

``data/plans.npz`` 가 정확히 그것이다
-------------------------------------
실기 MPC 주행 122 개, 9 초씩. v29 가 뒤 24 개(held-out)에서 EE 중앙 9.5 mm 를 낸
바로 그 데이터다.

.. warning::
   **절대좌표는 munji 맵과 무관하다.** 좌표 범위가 맵과 겹쳐 보이지만, 실제로는
   전 점(100%)이 픽셀 205(미지) 위에 있고 매핑된 자유공간까지 중앙 12.8 m 떨어져 있다.
   학습은 궤적을 항상 시작 차체 기준으로 상대화해 썼으므로(``to_start_relative``)
   절대좌표는 아무도 쓴 적이 없다.

   그래서 twin 은 :func:`relativize` 로 모양만 가져오고 :func:`place_at` 으로
   **맵의 자유공간에 놓는다** (자리는 ``world.spawn_points`` 가 고른다).

그래서 이 궤적으로 twin 을 돌리면 두 가지를 한 번에 얻는다::

    [검증]  twin 이 9.5 mm 를 재현하면 씬·관측·액션·보간 전 경로가 종단 검증된다
    [신규]  거기에 **실제 3D 맵과 두 번째 로봇**을 더했을 때 무엇이 달라지는지
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class Track:
    """로봇 한 대의 궤적. 둘 다 (T, 4) = t, x, y, yaw, **map 절대좌표**.

    ``ee`` 의 yaw 는 학습 데이터와 같은 의미 = **막대(도구) 방향**이다. tool0 목표 요로
    바꾸는 보정(v29: -pi/2)은 :class:`~plan.plan_to_target.TargetStream` 이 한다.
    """

    name: str
    ee: np.ndarray
    base: np.ndarray

    @property
    def duration(self) -> float:
        return float(self.ee[-1, 0] - self.ee[0, 0])

    def bbox(self) -> tuple[float, float, float, float]:
        xy = np.vstack([self.ee[:, 1:3], self.base[:, 1:3]])
        return (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max())

    def centre(self) -> np.ndarray:
        x0, x1, y0, y1 = self.bbox()
        return np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])


def load_npz(path: str, index: int) -> Track:
    d = np.load(path)
    E, B = d["ee"].astype(np.float64), d["base"].astype(np.float64)
    if not (0 <= index < len(E)):
        raise IndexError("plans.npz 에 %d 번이 없다 (0..%d)" % (index, len(E) - 1))
    return Track("ds[%d]" % index, E[index], B[index])


def load_csv_side(path: str, side: str) -> Track:
    """협력 운송 계획 CSV 에서 **한쪽만** 떼어낸다.

    편대를 일부러 검증할 때만 쓴다. 기본 경로가 아니다 — B 쪽은 계속 후진하는
    역할이라 v29 분포 밖이다.
    """
    from plan.plan_sampler import Plan
    p = Plan.from_csv_dir(path)
    if side == "a":
        ee = np.column_stack([p.t, p.ee1_xy_csv, p.obj_yaw])
        base = np.column_stack([p.t, p.base1_xy, p.base1_yaw])
    else:
        ee = np.column_stack([p.t, p.ee2_xy_csv, p.obj_yaw + np.pi])
        base = np.column_stack([p.t, p.base2_xy, p.base2_yaw])
    return Track("%s:%s" % (os.path.basename(path), side), ee, base)


def parse(spec: str) -> Track:
    """``npz:<i>`` 또는 ``npz:<path>:<i>`` 또는 ``csv:<dir>:<a|b>``."""
    parts = spec.split(":")
    if parts[0] == "npz":
        if len(parts) == 2:
            return load_npz("data/plans.npz", int(parts[1]))
        return load_npz(parts[1], int(parts[2]))
    if parts[0] == "csv":
        return load_csv_side(parts[1], parts[2])
    raise ValueError("궤적 지정이 이상하다: %r (npz:98 / csv:<dir>:a)" % spec)


def separation(a: Track, b: Track) -> float:
    """두 궤적이 **가장 가까워지는** 거리. 두 대를 같이 올릴 수 있는지 판정한다.

    시각마다 비교하는 것이 아니라 전 구간 대 전 구간의 최소값을 본다 — 두 로봇은
    서로 독립이라 같은 시각에 같은 자리에 있으라는 법이 없고, 한쪽이 늦어지면
    다른 시각의 상대와 만난다.
    """
    p, q = a.base[:, 1:3], b.base[:, 1:3]
    d = np.linalg.norm(p[:, None, :] - q[None, :, :], axis=2)
    return float(d.min())


def pick_pair(path: str, lo: int, hi: int, min_sep: float = 4.0):
    """``[lo, hi)`` 안에서 **충분히 떨어진** 두 궤적을 고른다.

    가까우면 두 로봇이 서로 부딪혀서, 재는 것이 추종 성능이 아니라 충돌이 된다.
    """
    d = np.load(path)
    n = len(d["ee"])
    lo, hi = max(0, lo), min(n, hi)
    best = None
    for i in range(lo, hi):
        ti = load_npz(path, i)
        for j in range(i + 1, hi):
            s = separation(ti, load_npz(path, j))
            if s >= min_sep:
                return i, j, s
            if best is None or s > best[2]:
                best = (i, j, s)
    return best


def relativize(tr: Track) -> Track:
    """궤적을 **자기 시작 차체 자세 기준**으로 옮긴다 (시작 = 원점, 헤딩 0).

    학습이 하는 것과 같다 (``eval_plans.to_start_relative``,
    ``_update_from_plan`` 의 ``b_start``). 그래서 ``plans.npz`` 의 절대좌표는
    애초에 학습에 쓰인 적이 없다 — 실제로 이 데이터의 절대좌표는 munji 맵의
    자유공간과 무관하다 (전 점이 미지영역 위, 자유공간까지 중앙 12.8 m).

    twin 도 같은 규약을 따라야 한다: 모양만 가져오고 **놓을 자리는 맵에서 고른다.**
    """
    bx, by, bpsi = tr.base[0, 1], tr.base[0, 2], tr.base[0, 3]
    c, s = np.cos(-bpsi), np.sin(-bpsi)
    R = np.array([[c, -s], [s, c]])
    out = []
    for A in (tr.ee, tr.base):
        A = A.copy()
        A[:, 1:3] = (R @ (A[:, 1:3] - np.array([bx, by])).T).T
        A[:, 3] -= bpsi
        out.append(A)
    return Track(tr.name, out[0], out[1])


def place_at(tr: Track, x: float, y: float, yaw: float) -> Track:
    """상대화된 궤적을 맵의 ``(x, y, yaw)`` 에 놓는다."""
    rel = relativize(tr)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    out = []
    for A in (rel.ee, rel.base):
        A = A.copy()
        A[:, 1:3] = (R @ A[:, 1:3].T).T + np.array([x, y])
        A[:, 3] += yaw
        out.append(A)
    return Track("%s@(%.1f,%.1f)" % (tr.name, x, y), out[0], out[1])
