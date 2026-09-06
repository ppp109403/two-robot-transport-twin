"""차체 궤적 추종기 — 고전 유니사이클 제어기.

왜 이게 필요한가
----------------
RL 정책 없이 "**MPC 계획이 애초에 추종 가능한가**" 를 먼저 확인하려면, 무난하고
검증된 제어기로 한 번 돌려 봐야 한다. 여기서 못 따라가면 정책을 학습시킬 이유가 없다.

어떤 제어기인가
---------------
Kanayama 궤적 추종 (1990) 의 실용 변형이다. 피드포워드(참조 속도) + 차체 프레임
오차 피드백::

    e_x   =  cos(θ)(x_r − x) + sin(θ)(y_r − y)      전진 방향 오차
    e_y   = −sin(θ)(x_r − x) + cos(θ)(y_r − y)      횡방향 오차
    e_θ   =  wrap(θ_r − θ)                          방위 오차

    v = v_r·cos(e_θ) + k_x·e_x
    ω = ω_r + v_r·k_y·e_y + k_θ·sin(e_θ)

**원형과 다른 점 하나:** 원래 Kanayama 는 마지막 항이 ``v_r·k_θ·sin(e_θ)`` 라서
``v_r → 0`` 이면 방위 보정이 통째로 사라진다. 계획이 정지하거나 방향을 바꾸는
구간(우리 계획은 한쪽이 주행의 86 % 를 후진한다)에서 그대로 쓰면 방위가 표류한다.
그래서 방위 항만 ``v_r`` 을 곱하지 않는다. 횡방향 항은 곱한 채로 둔다 — 정지 중에
횡오차를 방위로 바꾸려 들면 제자리에서 빙빙 돈다.

왜 pure pursuit 이 아닌가
-------------------------
pure pursuit 은 **경로**를 따라간다. 우리 계획은 시각이 붙은 **궤적**이고
(``t,x,y,yaw``), 확인하려는 것이 "제 시각에 그 자리에 있을 수 있는가" 다.
경로만 맞고 시각이 밀리면 EE 목표가 팔 리치 밖으로 빠져나가므로 그것까지 봐야 한다.

참조 속도는 왜 직접 미분하나
----------------------------
``TargetStream`` 이 주는 ``base_ref_twist`` 의 속력은 **부호 없는 노름**이다
(58 차원 관측이 그렇게 학습됐다). 후진 구간에서 부호가 사라지면 피드포워드가
정반대로 들어가므로, 여기서는 참조 방위에 투영해 **부호 있는** v_r 을 만든다.
"""

from __future__ import annotations

import math

import numpy as np


def wrap_pi(a: float) -> float:
    """(-pi, pi]. 요 **차분**에만 쓴다."""
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


class UnicycleTracker:
    """차체 참조 (x, y, yaw) 를 추종해 (v, ω) 를 낸다. 로봇 한 대분.

    Args:
        dt: 제어주기 [s]
        k_x: 전진 오차 이득. 크면 시각을 맞추려 급가속한다
        k_y: 횡방향 오차 이득. 이것이 "경로로 되돌아오는" 힘이다
        k_theta: 방위 오차 이득
        v_max / v_min / w_max: 하드웨어 한계 [m/s, rad/s]. ``DiffDriveActionCfg``
            에서 읽어 넘긴다 — 여기 손으로 적으면 두 벌째가 생긴다
    """

    def __init__(self, dt: float, k_x: float = 1.5, k_y: float = 6.0,
                 k_theta: float = 2.5, v_max: float = 0.5, v_min: float = -0.2,
                 w_max: float = 1.0):
        self.dt = float(dt)
        self.k_x, self.k_y, self.k_th = float(k_x), float(k_y), float(k_theta)
        self.v_max, self.v_min, self.w_max = float(v_max), float(v_min), float(w_max)
        self._prev_ref: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_ref = None

    def ref_twist(self, ref: np.ndarray) -> tuple[float, float]:
        """참조의 **부호 있는** (v_r, ω_r). 첫 호출은 (0, 0)."""
        ref = np.asarray(ref, dtype=np.float64)
        if self._prev_ref is None:
            self._prev_ref = ref.copy()
            return 0.0, 0.0
        d = ref[:2] - self._prev_ref[:2]
        # 참조 방위에 투영 -> 후진이면 음수가 된다
        v_r = float(d[0] * math.cos(ref[2]) + d[1] * math.sin(ref[2])) / self.dt
        w_r = wrap_pi(ref[2] - self._prev_ref[2]) / self.dt
        self._prev_ref = ref.copy()
        return v_r, w_r

    def step(self, pose: np.ndarray, ref: np.ndarray) -> dict:
        """한 스텝. ``pose`` 와 ``ref`` 는 둘 다 월드 (x, y, yaw).

        ``pose`` 는 **추정 자세**를 넣는다 (실기는 AMCL 밖에 없다). 참값을 넣으면
        실기가 못 가진 정보로 제어하는 셈이 된다.

        Returns:
            v, w (한계 적용 전/후), 오차 성분, 포화 여부.
        """
        pose = np.asarray(pose, dtype=np.float64)
        ref = np.asarray(ref, dtype=np.float64)
        v_r, w_r = self.ref_twist(ref)

        c, s = math.cos(pose[2]), math.sin(pose[2])
        dx, dy = ref[0] - pose[0], ref[1] - pose[1]
        e_x = c * dx + s * dy
        e_y = -s * dx + c * dy
        e_th = wrap_pi(ref[2] - pose[2])

        v_raw = v_r * math.cos(e_th) + self.k_x * e_x
        # 방위 항에 v_r 을 곱하지 않는다 (모듈 docstring 참고)
        w_raw = w_r + v_r * self.k_y * e_y + self.k_th * math.sin(e_th)

        v = float(np.clip(v_raw, self.v_min, self.v_max))
        w = float(np.clip(w_raw, -self.w_max, self.w_max))
        return {
            "v": v, "w": w, "v_raw": v_raw, "w_raw": w_raw,
            "v_ref": v_r, "w_ref": w_r,
            "e_x": e_x, "e_y": e_y, "e_th": e_th,
            "e_pos": float(math.hypot(dx, dy)),
            "sat": bool(v != v_raw or w != w_raw),
        }
