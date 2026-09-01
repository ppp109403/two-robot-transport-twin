"""MPC 계획을 제어주기로 되살리는 보간기 — 플래너 노드와 **같은 규약**.

왜 별도 모듈인가
----------------
플래너는 dt=0.15 s (6.7 Hz) 로 궤적을 풀고, 정책은 30 Hz 로 돈다. 그 사이를 어떻게
메우느냐가 결과를 바꾼다. 그리고 틀리는 방식이 조용하다.

가장 큰 함정은 **두 EE 를 각각 보간하는 것**이다::

    틀림   ee_a(s) = lerp(ee_a),  ee_b(s) = lerp(ee_b)
    맞음   물체 pose 를 보간한 뒤 EE 를 **재유도**한다

막대는 강체라 두 끝점 사이 거리가 항상 obj_len 이어야 한다. 그런데 회전하는 강체의
두 점을 각각 선형보간하면 그 사이 **현(chord)이 짧아진다**. 즉 보간기가 막대를
줄여 놓고, 그 오차가 나중에 "정책이 막대 길이를 못 지킨다" 로 둔갑한다.

플래너 노드도 같은 이유로 같은 선택을 해 두었다
(``two_robot_nlp_node._sample_step``, md5 ``b1b11a24e7c53b3bfe345ac4472ed722``)::

    # Interpolate the *object* pose and rederive the grasp points, rather
    # than interpolating the two EEs independently -- lerping two points of
    # a rotating rigid body shortens the chord between them.

이 모듈은 그 함수의 시뮬 쪽 짝이다. 어긋나면 ``twin/tools/check_plan_sampler.py`` 가 잡는다.

요는 언랩되어 있다
------------------
NLP 이 내놓는 yaw 계열은 전부 언랩된 연속값이라 그냥 lerp 하면 된다. CSV 로 오갈 때도
**절대 wrap 하지 말 것** — wrap 하면 ±pi 를 넘는 순간 보간이 반대로 돈다.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass

import numpy as np

#: 플래너 노드가 EE 를 물체 pose 에서 재유도할 때 쓰는 규약.
#: ``ee_a = obj - half``, ``ee_b = obj + half``, ``half = (obj_len/2)*[cos, sin](obj_yaw)``
#: 그리고 ``ee_yaw_a = obj_yaw``, ``ee_yaw_b = obj_yaw + pi`` (두 EE 가 마주 본다).
#:
#: 2026-09-01 갱신: b1b11a24 -> 7707316e. 노드 파일은 바뀌었지만 ``_sample_step`` 은
#: 한 줄도 안 바뀌었다 (전문 대조 완료). md5 는 파일 전체를 보는 거친 감시선이라
#: 무관한 수정에도 걸린다 — 걸리면 **함수를 직접 대조한 뒤** 이 값을 갱신할 것.
PLANNER_NODE_MD5 = "7707316ea744e41aa8cf77bd4d7fecdd"

CSV_FILES = {
    "object": "object.csv",
    "base1": "base_A.csv",
    "base2": "base_B.csv",
    "ee1": "ee_A.csv",
    "ee2": "ee_B.csv",
}


def _read_csv(path: str) -> np.ndarray:
    """``t,x,y,yaw`` CSV -> (N, 4). yaw 는 **언랩된 채로** 둔다."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    head, body = rows[0], rows[1:]
    if [c.strip() for c in head] != ["t", "x", "y", "yaw"]:
        raise ValueError("%s: 헤더가 t,x,y,yaw 가 아니다 -> %s" % (path, head))
    return np.array([[float(c) for c in r] for r in body], dtype=np.float64)


@dataclass
class Plan:
    """플래너가 푼 한 판. 배열은 전부 길이 N+1."""

    dt: float
    t: np.ndarray
    obj_xy: np.ndarray
    obj_yaw: np.ndarray
    base1_xy: np.ndarray
    base1_yaw: np.ndarray
    base2_xy: np.ndarray
    base2_yaw: np.ndarray
    obj_len: float
    #: CSV 에 들어 있던 EE. **대조용으로만** 쓴다 — 재생에는 재유도한 값을 쓴다.
    ee1_xy_csv: np.ndarray | None = None
    ee2_xy_csv: np.ndarray | None = None

    @property
    def N(self) -> int:
        return len(self.t) - 1

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    # ------------------------------------------------------------------ 입력
    @classmethod
    def from_csv_dir(cls, path: str, obj_len: float | None = None) -> "Plan":
        """``two_robot_nlp_node`` 의 ``csv_dir`` 산출물을 읽는다.

        ``obj_len`` 을 주지 않으면 CSV 의 EE 두 열 사이 거리에서 **추정**한다.
        추정치와 실제 설정이 다르면 막대 길이 오차가 통째로 offset 으로 실리므로,
        운용할 때는 명시적으로 넘기는 쪽이 안전하다.
        """
        d = {k: _read_csv(os.path.join(path, f)) for k, f in CSV_FILES.items()}
        n = {k: len(v) for k, v in d.items()}
        if len(set(n.values())) != 1:
            raise ValueError("CSV 길이가 서로 다르다: %s" % n)

        t = d["object"][:, 0]
        dts = np.diff(t)
        dt = float(np.median(dts)) if len(dts) else 0.0
        if len(dts) and float(np.max(np.abs(dts - dt))) > 1e-6:
            raise ValueError("시간 간격이 일정하지 않다 (max 편차 %.3e s)"
                             % float(np.max(np.abs(dts - dt))))

        ee1, ee2 = d["ee1"][:, 1:3], d["ee2"][:, 1:3]
        if obj_len is None:
            obj_len = float(np.median(np.linalg.norm(ee2 - ee1, axis=1)))

        return cls(
            dt=dt, t=t,
            obj_xy=d["object"][:, 1:3], obj_yaw=d["object"][:, 3],
            base1_xy=d["base1"][:, 1:3], base1_yaw=d["base1"][:, 3],
            base2_xy=d["base2"][:, 1:3], base2_yaw=d["base2"][:, 3],
            obj_len=float(obj_len), ee1_xy_csv=ee1, ee2_xy_csv=ee2,
        )

    # ---------------------------------------------------------------- 보간
    def sample_step(self, s: float) -> dict:
        """분수 스텝 인덱스 ``s`` 에서의 계획 상태.

        ``two_robot_nlp_node._sample_step`` 과 **같은 식**이어야 한다.
        차이가 나면 시뮬이 관제와 다른 목표를 재생하게 된다.
        """
        n = self.N
        s = min(max(float(s), 0.0), float(n))
        k = int(min(s, n - 1e-9))
        a = s - k
        k1 = min(k + 1, n)

        def lerp(arr):
            return (1.0 - a) * arr[k] + a * arr[k1]

        po = lerp(self.obj_xy)
        tho = float(lerp(self.obj_yaw))
        # ★ EE 는 보간하지 않고 물체 pose 에서 재유도한다 (모듈 docstring 참고).
        half = 0.5 * self.obj_len * np.array([math.cos(tho), math.sin(tho)])

        return {
            "s": s,
            "t": s * self.dt,
            "obj": po,
            "obj_yaw": tho,
            "ee": {"a": po - half, "b": po + half},
            "ee_yaw": {"a": tho, "b": tho + math.pi},
            "base": {"a": lerp(self.base1_xy), "b": lerp(self.base2_xy)},
            "base_yaw": {"a": float(lerp(self.base1_yaw)),
                         "b": float(lerp(self.base2_yaw))},
            "finished": s >= n,
        }

    def sample_time(self, t: float) -> dict:
        """계획 시작 후 경과 ``t`` 초에서의 상태."""
        return self.sample_step(t / self.dt)

    def resample(self, rate_hz: float) -> list[dict]:
        """제어주기로 통째로 되살린다. 마지막 스텝을 **포함**한다."""
        step_dt = 1.0 / float(rate_hz)
        n = int(math.floor(self.duration / step_dt)) + 1
        return [self.sample_time(i * step_dt) for i in range(n)]


def naive_ee_lerp(plan: Plan, s: float) -> tuple[np.ndarray, np.ndarray]:
    """**틀린 방식** — 두 EE 를 각각 보간한다. 게이트가 오차를 재기 위해서만 존재한다."""
    if plan.ee1_xy_csv is None or plan.ee2_xy_csv is None:
        raise ValueError("CSV EE 열이 없다")
    n = plan.N
    s = min(max(float(s), 0.0), float(n))
    k = int(min(s, n - 1e-9))
    a = s - k
    k1 = min(k + 1, n)
    e1 = (1.0 - a) * plan.ee1_xy_csv[k] + a * plan.ee1_xy_csv[k1]
    e2 = (1.0 - a) * plan.ee2_xy_csv[k] + a * plan.ee2_xy_csv[k1]
    return e1, e2


#: 합성 계획 체제. (obj_v_max, obj_w_max) — EE 속력 ~= v + (obj_len/2)*w 이고
#: 막대가 2 m 라 계수가 1.0 이다.
REGIMES = {
    "v29": (0.12, 0.08),        # twin/configs/policy_v29.yaml 의 권장값
    "node_default": (0.60, 0.60),   # two_robot_nlp_node 기본값
}


def synthetic_plan(n: int = 40, dt: float = 0.15, obj_len: float = 2.0,
                   turn_rate: float = 0.08, speed: float = 0.12) -> Plan:
    """게이트용 합성 계획 — **회전하면서 전진**한다.

    회전이 있어야 EE 개별보간의 현 축소가 드러난다. 직진만 하는 계획으로는
    두 방식이 똑같이 나와서 검사가 통과해 버린다.

    기본값은 v29 권장 프로파일이다 (:data:`REGIMES`). 노드 기본값처럼 빠른 체제에서는
    같은 보간 실수가 훨씬 크게 나타나므로 게이트가 두 체제를 모두 잰다.
    """
    t = np.arange(n + 1, dtype=np.float64) * dt
    yaw = turn_rate * t
    # 가로 흔들림도 회전율에 맞춰 줄인다 — 안 그러면 저속 체제에서 EE 속력이
    # 회전이 아니라 이 항에 지배당해 체제 비교가 무의미해진다.
    xy = np.stack([speed * t, (0.6 * turn_rate) * np.sin(0.4 * t)], axis=1)
    half = 0.5 * obj_len * np.stack([np.cos(yaw), np.sin(yaw)], axis=1)
    off = np.stack([np.cos(yaw + math.pi / 2), np.sin(yaw + math.pi / 2)], axis=1)
    return Plan(
        dt=dt, t=t, obj_xy=xy, obj_yaw=yaw,
        base1_xy=xy - half * 1.35 + 0.05 * off, base1_yaw=yaw,
        base2_xy=xy + half * 1.35 - 0.05 * off, base2_yaw=yaw + math.pi,
        obj_len=obj_len, ee1_xy_csv=xy - half, ee2_xy_csv=xy + half,
    )
