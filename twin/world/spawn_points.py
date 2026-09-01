"""궤적을 놓을 자리를 맵에서 고른다.

왜 필요한가
-----------
``data/plans.npz`` 의 절대좌표는 munji 맵과 무관하다. 전 점(100%)이 픽셀 205(미지)
위에 있고 매핑된 자유공간까지 중앙 12.8 m 떨어져 있다. 그런데 학습은 궤적을 항상
**시작 차체 기준으로 상대화**해서 썼으므로 (``eval_plans.to_start_relative``),
절대좌표는 애초에 아무도 쓰지 않았다.

그래서 twin 도 같은 규약을 쓴다 — 궤적은 **모양만** 가져오고, 놓을 자리는 맵의
자유공간에서 고른다. 그것이 학습 조건과 같으면서 맵을 진짜로 쓰는 유일한 방법이다.

무엇을 최대화하나
-----------------
자리 하나를 고를 때 중심점의 여유거리만 보면 안 된다. 궤적이 최대 1.2 x 3.1 m 를
훑으므로, **궤적 전체를 놓아 보고 그 중 최소 여유**를 본다. 요도 같이 돌려 가며
가장 좋은 조합을 고른다.
"""

from __future__ import annotations

import math
import os

import numpy as np
from scipy import ndimage as ndi

from world.map_extrude import classify, read_map_yaml, read_pgm

DEFAULT_MAP_YAML = "handoff/map_munji_3f_2025_wide/munji_3f_2025_wide.yaml"


class MapClearance:
    """맵의 여유거리 조회기. 압출 USD 와 **같은 마스크**를 쓴다."""

    def __init__(self, map_yaml: str = DEFAULT_MAP_YAML, open_px: int = 1):
        meta = read_map_yaml(map_yaml)
        img = read_pgm(os.path.join(os.path.dirname(map_yaml), str(meta["image"])))
        free, wall, unk = classify(img)
        # map_extrude 기본값과 동일: close_px=0, open_px=1, 벽은 절대 침식하지 않음
        obst = ndi.binary_opening(wall | unk, np.ones((open_px * 2 + 1,) * 2)) | wall
        self.res = float(meta["resolution"])
        self.ox, self.oy = float(meta["origin"][0]), float(meta["origin"][1])
        self.h, self.w = img.shape
        self.clear = ndi.distance_transform_edt(~obst, sampling=self.res)
        self.free = ~obst

    def at(self, xy: np.ndarray) -> np.ndarray:
        """(N,2) map 좌표 -> 여유거리. 맵 밖은 0."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        fx = (xy[:, 0] - self.ox) / self.res
        fy = (xy[:, 1] - self.oy) / self.res
        ix = np.floor(fx).astype(int)
        iy = self.h - 1 - np.floor(fy).astype(int)      # pgm 행 0 = 맵 위쪽
        ok = (ix >= 0) & (ix < self.w) & (iy >= 0) & (iy < self.h)
        out = np.zeros(len(xy))
        out[ok] = self.clear[iy[ok], ix[ok]]
        return out

    def track_clearance(self, tr, x: float, y: float, yaw: float) -> float:
        """궤적을 (x,y,yaw) 에 놓았을 때 **차체·EE 전 구간의 최소 여유**."""
        from plan.tracks import place_at
        p = place_at(tr, x, y, yaw)
        return float(min(self.at(p.base[:, 1:3]).min(), self.at(p.ee[:, 1:3]).min()))


def candidates(mc: MapClearance, min_clear: float, stride: int = 12) -> np.ndarray:
    """여유가 충분한 자리 후보 (N,2). ``stride`` 로 성기게 훑는다."""
    m = mc.free & (mc.clear >= min_clear)
    iy, ix = np.nonzero(m)
    sel = (ix % stride == 0) & (iy % stride == 0)
    ix, iy = ix[sel], iy[sel]
    return np.stack([mc.ox + (ix + 0.5) * mc.res,
                     mc.oy + (mc.h - 1 - iy + 0.5) * mc.res], axis=1)


def best_pose(mc: MapClearance, tr, cand: np.ndarray, n_yaw: int = 12,
              exclude: list | None = None, min_sep: float = 4.0,
              max_sep: float | None = None):
    """후보들 중 **궤적 전체의 최소 여유가 가장 큰** (x, y, yaw).

    ``max_sep`` 은 GUI 로 볼 때 쓴다. 여유만 최대화하면 두 로봇이 맵 양 끝(48 m)에
    떨어져 한 화면에 안 들어온다.
    """
    best = (None, -1e9)
    for x, y in cand:
        if exclude:
            ds = [math.hypot(x - ex, y - ey) for ex, ey in exclude]
            if any(v < min_sep for v in ds):
                continue
            if max_sep is not None and min(ds) > max_sep:
                continue
        for yaw in np.linspace(-math.pi, math.pi, n_yaw, endpoint=False):
            c = mc.track_clearance(tr, float(x), float(y), float(yaw))
            if c > best[1]:
                best = ((float(x), float(y), float(yaw)), c)
    return best


def pick_two(mc: MapClearance, tr_a, tr_b, min_clear: float = 0.5,
             min_sep: float = 4.0, stride: int = 12, max_sep: float | None = None):
    """두 로봇의 스폰 자세를 고른다. 서로 ``min_sep`` 이상 떨어뜨린다."""
    cand = candidates(mc, min_clear, stride)
    if len(cand) == 0:
        raise SystemExit("여유 %.2f m 이상인 자리가 없다" % min_clear)
    pa, ca = best_pose(mc, tr_a, cand)
    if pa is None:
        raise SystemExit("A 자리를 못 찾았다")
    pb, cb = best_pose(mc, tr_b, cand, exclude=[(pa[0], pa[1])],
                       min_sep=min_sep, max_sep=max_sep)
    if pb is None:
        raise SystemExit("B 자리를 못 찾았다 (min_sep %.1f / max_sep %s 완화 필요)"
                         % (min_sep, max_sep))
    return (pa, ca), (pb, cb)
