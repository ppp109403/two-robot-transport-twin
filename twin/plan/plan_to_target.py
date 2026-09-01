"""계획(t, x, y, yaw) -> 정책이 받는 EE 목표. 학습 커맨드 term 과 **같은 규약**.

무엇을 옮기는가
---------------
학습에서 이 일을 하는 것은 ``MovingPoseWorldAmclCommand._update_from_plan`` 이다.
twin 은 매니저 기반 커맨드 term 을 안 쓰고 직접 목표를 만들기 때문에, 그 변환을
여기서 다시 해야 한다. 옮겨야 하는 것은 여섯 가지고 **전부 조용히 틀릴 수 있다**::

    1 시작 차체 기준 상대화   궤적은 map 절대좌표다. 그대로 쓰면 로봇과의 상대 기하가 깨진다
    2 요 보정                 데이터셋의 요는 **막대 방향**이다. tool0 목표 요가 아니다
    3 피치 = pi 고정          손을 아래로. 학습은 정확히 pi 고정이라 기운 접근은 학습에 없다
    4 목표 높이               map 기준 고정값 (v29 는 0.55)
    5 선속도 피드포워드 xy   (ee_xy - prev_ee_xy) / step_dt
    6 나머지 속도 채널       아래 참고 — **학습·실기·twin 이 서로 다르다**

속도 피드포워드 채널의 어긋남 (2026-08-12, G3 가 찾음)
-------------------------------------------------------
1~5 는 세 경로가 정확히 일치한다 (실측 1e-6). 그런데 ``target_vel`` 의 나머지 성분은
**세 곳이 전부 다른 값을 쓰고 있다**::

    학습 env    _update_command 가 기하 합성 경로를 **먼저 전체 env 에 돌리고**,
                _update_from_plan 이 그 위에 pose 와 lin_vel[:2] 만 덮어쓴다.
                따라서 계획 env 의 lin_vel[2] 와 ang_vel 은 **기하 합성 경로의 잔여값**이다.
                계획과 아무 상관 없다. 실측 ang 중앙 0.215 rad/s.

    실기 브리지  plan_adapter.py 는 계획에서 유도한다.
                lin xy = 수치미분(+1차 저역통과), lin z = 0, ang z = 목표 요의 변화율

    twin        여기서 고른다 (:class:`PlanTargetCfg` 의 ``ff_mode``)

학습 쪽 잔여값은 계획과 무상관한 잡음이므로 정책이 **무시하도록 학습됐을 가능성이 높다**.
실기가 (다른 값을 넣는데도) 예측대로 동작해 온 것이 그 방증이다. 다만 확인된 적은 없다.

그래서 twin 의 기본값은 ``ff_mode="planner"`` — **실기와 같은 값**이다. twin 의 목적이
실기 거동 예측이므로 학습 잔여값이 아니라 실기를 따라가는 것이 맞다.
Phase 1 에서 ``zero`` 와 비교해 이 채널의 민감도를 직접 재면 위 가설이 검증된다.

요 보정 값을 여기서 정하지 않는 이유
------------------------------------
``PLAN_YAW_OFFSET`` 은 전역 상수인데 판마다 다르다 (v29 는 -pi/2, v30 은 +pi/2).
repo 에서 import 하면 v29 를 돌리면서 v30 값을 쓰게 되고, 에러 없이 180 도 틀어진
목표가 나간다. 그래서 **호출자가 config 에서 읽어 넘겨야 한다.**
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def wrap_pi(a):
    """(-pi, pi]. 요 **차분**에만 쓴다 — 절대 요는 언랩된 채로 두어야 한다."""
    return (np.asarray(a) + math.pi) % (2.0 * math.pi) - math.pi


def quat_pitch_pi(yaw: float | np.ndarray) -> np.ndarray:
    """roll 0, pitch pi, yaw 인 쿼터니언 (wxyz). Isaac Lab 규약.

    ``quat_from_euler_xyz(0, pi, yaw)`` 를 전개한 닫힌 형태다::

        (w, x, y, z) = (0, -sin(yaw/2), cos(yaw/2), 0)

    ``scripts/eval_plans.py`` 의 ``quat_from_yaw_down`` 과 같은 식이다.

    .. note::
       w 가 정확히 0 인 특이면이라 부호(q vs -q)가 부동소수점에 좌우된다. 정책은
       rot6d 로 받으므로 부호에 불변이지만, **쿼터니언끼리 직접 비교하면 안 된다.**
       비교는 회전행렬이나 rot6d 로 할 것.
    """
    y = np.asarray(yaw, dtype=np.float64)
    h = y * 0.5
    z = np.zeros_like(y)
    return np.stack([z, -np.sin(h), np.cos(h), z], axis=-1)


def sample_traj(traj: np.ndarray, t: float | np.ndarray):
    """(T, 4) = t,x,y,yaw 궤적을 시각 ``t`` 에서 선형보간. -> (..., 3)

    ``_update_from_plan`` 과 같이 **요는 차분을 wrap 해서** 보간한다. 그냥 lerp 하면
    ±pi 를 넘는 순간 반대로 돈다.
    """
    ts = traj[:, 0]
    t = np.asarray(t, dtype=np.float64)
    i = np.clip(np.searchsorted(ts, t), 1, len(ts) - 1)
    t0, t1 = ts[i - 1], ts[i]
    f = np.clip((t - t0) / np.maximum(t1 - t0, 1e-6), 0.0, 1.0)[..., None]
    p0, p1 = traj[i - 1, 1:], traj[i, 1:]
    d = np.concatenate([p1[..., :2] - p0[..., :2],
                        wrap_pi(p1[..., 2:3] - p0[..., 2:3])], axis=-1)
    return p0 + f * d


@dataclass
class PlanTargetCfg:
    """정책 판마다 달라지는 값. `twin/configs/policy_*.yaml` 에서 읽어 채운다."""

    yaw_offset: float           # v29: -pi/2
    goal_z: float               # v29: 0.55 (map 기준)
    step_dt: float              # 제어주기 [s]. 30 Hz -> 1/30
    #: 배속·시작시각. twin 재생은 1.0 / 0.0 이지만 게이트가 학습 랜덤화를 재현할 때 쓴다.
    time_scale: float = 1.0
    t0: float = 0.0
    #: 속도 피드포워드 규약. 모듈 docstring 의 "어긋남" 절을 먼저 읽을 것.
    #:
    #:   ``planner``  실기 ``plan_adapter.py`` 와 같다 (기본값).
    #:                lin xy = 차분, lin z = 0, ang z = 목표 요 변화율
    #:   ``zero``     ang 과 lin z 를 0 으로 둔다. 민감도 비교용
    ff_mode: str = "planner"
    #: 실기 브리지의 1 차 저역통과 계수. ``plan_adapter`` 는 100 Hz 스트림에서
    #: 이 값을 쓴다 — twin 은 30 Hz 라 같은 alpha 가 같은 차단주파수를 뜻하지 않는다.
    #: 1.0 이면 필터 없음(순수 차분).
    ff_alpha: float = 1.0


class TargetStream:
    """계획을 제어주기로 흘려 보내며 EE 목표 pose 와 피드포워드 속도를 만든다.

    상태를 갖는 이유는 선속도가 **직전 목표와의 차분**이기 때문이다. 학습이 그렇게
    만들었으므로 (해석적 미분이 아니라) 여기서도 같은 방식이어야 한다.
    """

    def __init__(self, ee_traj: np.ndarray, base_traj: np.ndarray, cfg: PlanTargetCfg,
                 b_start: np.ndarray | None = None, origin: np.ndarray | None = None):
        """
        Args:
            ee_traj:   (T, 4) t, x, y, yaw.  yaw 는 **막대 방향** (tool0 목표 요가 아님)
            base_traj: (T, 4) 차체 참조. 상대화 기준과 차체 편차 지표에 쓴다
            b_start:   상대화 기준 차체 자세 (x, y, yaw). 없으면 ``t0`` 시점 값을 쓴다.
                       ★ 궤적 첫 점이 아니라 **t0 시점** 이다 (학습 ``_resample_plan`` 규약)
            origin:    씬에서의 원점 오프셋 (x, y, z). 없으면 0
        """
        self.ee, self.base, self.cfg = ee_traj, base_traj, cfg
        self.origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=np.float64)
        self.b_start = (np.asarray(b_start, dtype=np.float64) if b_start is not None
                        else sample_traj(base_traj, cfg.t0))
        if cfg.ff_mode not in ("planner", "zero"):
            raise ValueError("ff_mode 는 planner | zero 중 하나다: %r" % cfg.ff_mode)
        self._ep_t = 0.0
        self._prev_ee_xy: np.ndarray | None = None
        self._prev_yaw: float | None = None
        self._vel = np.zeros(3)      # 저역통과 상태 [vx, vy, wz]
        self._prev_ref: np.ndarray | None = None

    # -------------------------------------------------------------- 상대화
    def _rel(self, p_xy: np.ndarray) -> np.ndarray:
        """map 절대 xy -> 시작 차체 기준 -> 씬 원점 기준."""
        c, s = math.cos(-self.b_start[2]), math.sin(-self.b_start[2])
        d = np.asarray(p_xy, dtype=np.float64) - self.b_start[:2]
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]]) + self.origin[:2]

    def base_ref(self, ep_t: float | None = None) -> np.ndarray:
        """그 시각의 차체 참조 (x, y, yaw), 씬 좌표. 차체 편차 지표용."""
        t = self._ep_t if ep_t is None else ep_t
        b = sample_traj(self.base, t * self.cfg.time_scale + self.cfg.t0)
        return np.array([*self._rel(b[:2]), b[2] - self.b_start[2]])

    # ---------------------------------------------------------------- 진행
    def step(self) -> dict:
        """한 제어주기 전진하고 그 시점의 목표를 낸다.

        학습과 순서를 맞춘다: ``_ep_t`` 를 **먼저** 더하고 그 시각을 쓴다.
        """
        c = self.cfg
        self._ep_t += c.step_dt
        t = self._ep_t * c.time_scale + c.t0
        ee = sample_traj(self.ee, t)
        ee_xy = self._rel(ee[:2])

        # 데이터셋의 요는 막대 방향이다. tool0 목표 요로 바꾸려면 판별 보정을 더한다.
        yaw = float(ee[2] - self.b_start[2] + c.yaw_offset)
        pos = np.array([ee_xy[0], ee_xy[1], self.origin[2] + c.goal_z])
        quat = quat_pitch_pi(yaw)

        # 피드포워드. started 게이트까지 학습과 같게 — 첫 스텝의 (0 - ee) 로
        # 말도 안 되는 속도가 튀는 것을 막는 장치다.
        lin, ang = np.zeros(3), np.zeros(3)
        started = self._ep_t > c.step_dt * 1.5
        if started and self._prev_ee_xy is not None:
            raw = np.array([
                *((ee_xy - self._prev_ee_xy) / c.step_dt),
                wrap_pi(yaw - self._prev_yaw) / c.step_dt if self._prev_yaw is not None else 0.0,
            ])
            self._vel = (1.0 - c.ff_alpha) * self._vel + c.ff_alpha * raw
            lin[:2] = self._vel[:2]
            if c.ff_mode == "planner":
                ang[2] = self._vel[2]
        self._prev_ee_xy = ee_xy.copy()
        self._prev_yaw = yaw

        # 차체 참조 트위스트 — 58 차원 관측(v12/v28)이 쓴다.
        # 학습의 _update_from_plan 과 같은 식: 속력은 **부호 없는 노름**이고
        # 각속도는 wrap 한 차분이며, 초반 1.5 스텝은 0 이다.
        ref = self.base_ref()
        ref_tw = np.zeros(2)
        if self._ep_t > c.step_dt * 1.5 and self._prev_ref is not None:
            ref_tw[0] = float(np.linalg.norm(ref[:2] - self._prev_ref[:2]) / c.step_dt)
            ref_tw[1] = float(wrap_pi(ref[2] - self._prev_ref[2]) / c.step_dt)
        self._prev_ref = ref.copy()

        return {
            "t": self._ep_t,
            "plan_t": t,
            "pos_w": pos,
            "quat_w": quat,
            "yaw": yaw,
            "lin_vel_w": lin,       # z 는 항상 0 (실기 브리지와 같다)
            "ang_vel_w": ang,       # ff_mode 에 따라 0 또는 목표 요 변화율
            "base_ref": ref,
            "base_ref_twist": ref_tw,
            "finished": t >= self.ee[-1, 0],
        }
