"""A* over the signed distance field, used only to seed the NLP.

The path this produces is never executed.  Its job is to pick the homotopy
class (which side of each obstacle to pass) and to give IPOPT an initial guess
that is already roughly collision free -- a gradient-based solver can do
neither of those on its own.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1), (1, 0), (1, 1)]


def _snap_to_free(free: np.ndarray, ix: int, iy: int, max_radius_cells: int = 200):
    """Nudge a cell to the closest free one; RViz clicks land in walls a lot."""
    nx, ny = free.shape
    if free[ix, iy]:
        return ix, iy
    for r in range(1, max_radius_cells + 1):
        x0, x1 = max(0, ix - r), min(nx - 1, ix + r)
        y0, y1 = max(0, iy - r), min(ny - 1, iy + r)
        best, best_d2 = None, None
        for jx in range(x0, x1 + 1):
            for jy in (y0, y1):
                if free[jx, jy]:
                    d2 = (jx - ix) ** 2 + (jy - iy) ** 2
                    if best_d2 is None or d2 < best_d2:
                        best, best_d2 = (jx, jy), d2
        for jy in range(y0, y1 + 1):
            for jx in (x0, x1):
                if free[jx, jy]:
                    d2 = (jx - ix) ** 2 + (jy - iy) ** 2
                    if best_d2 is None or d2 < best_d2:
                        best, best_d2 = (jx, jy), d2
        if best is not None:
            return best
    return None


def astar(sdf, start_xy, goal_xy, clearance: float,
          prefer_clearance: float = 0.0, clearance_weight: float = 0.0,
          max_expansions: int = 4_000_000):
    """Returns an (M, 2) world-frame path, or None if unreachable."""
    free = sdf.sdf >= clearance
    nx, ny = free.shape
    res = sdf.res

    # Extra per-cell cost for hugging obstacles.  Cheap way to bias the seed
    # towards the middle of corridors without changing what is *feasible*.
    if clearance_weight > 0.0 and prefer_clearance > 0.0:
        pen = clearance_weight * np.maximum(0.0, prefer_clearance - sdf.sdf)
    else:
        pen = np.zeros_like(sdf.sdf)

    six_a, siy_a = sdf.world_to_idx(start_xy)
    gix_a, giy_a = sdf.world_to_idx(goal_xy)

    s = _snap_to_free(free, int(six_a[0]), int(siy_a[0]))
    g = _snap_to_free(free, int(gix_a[0]), int(giy_a[0]))
    if s is None or g is None:
        return None
    six, siy = s
    gix, giy = g

    start_flat = six + nx * siy
    goal_flat = gix + nx * giy

    gscore = np.full(nx * ny, np.inf, dtype=np.float64)
    came = np.full(nx * ny, -1, dtype=np.int64)
    closed = np.zeros(nx * ny, dtype=bool)
    pen_flat = pen.ravel(order='F')
    free_flat = free.ravel(order='F')

    gscore[start_flat] = 0.0
    h0 = math.hypot(gix - six, giy - siy) * res
    heap = [(h0, start_flat)]
    expansions = 0

    while heap:
        _, cur = heapq.heappop(heap)
        if closed[cur]:
            continue
        closed[cur] = True
        if cur == goal_flat:
            break
        expansions += 1
        if expansions > max_expansions:
            return None

        cix = cur % nx
        ciy = cur // nx
        gc = gscore[cur]
        for dx, dy in _NEIGHBOURS:
            jx = cix + dx
            jy = ciy + dy
            if jx < 0 or jy < 0 or jx >= nx or jy >= ny:
                continue
            nxt = jx + nx * jy
            if closed[nxt] or not free_flat[nxt]:
                continue
            step = res * (1.4142135623730951 if (dx and dy) else 1.0)
            cand = gc + step * (1.0 + pen_flat[nxt])
            if cand < gscore[nxt]:
                gscore[nxt] = cand
                came[nxt] = cur
                f = cand + math.hypot(gix - jx, giy - jy) * res
                heapq.heappush(heap, (f, nxt))

    if not closed[goal_flat]:
        return None

    idx = []
    cur = goal_flat
    while cur != -1:
        idx.append(cur)
        if cur == start_flat:
            break
        cur = came[cur]
    idx.reverse()
    ixs = np.array([i % nx for i in idx])
    iys = np.array([i // nx for i in idx])
    return sdf.idx_to_world(ixs, iys)


def segment_clear(sdf, p0, p1, clearance: float) -> bool:
    n = max(2, int(np.linalg.norm(np.asarray(p1) - np.asarray(p0)) / (0.5 * sdf.res)) + 1)
    ts = np.linspace(0.0, 1.0, n)[:, None]
    pts = np.asarray(p0)[None, :] * (1 - ts) + np.asarray(p1)[None, :] * ts
    return bool(np.all(sdf.value(pts) >= clearance))


def shortcut(path, sdf, clearance: float):
    """Greedy line-of-sight shortcutting; removes the 8-connected staircase."""
    path = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if len(path) < 3:
        return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not segment_clear(sdf, path[i], path[j], clearance):
            j -= 1
        out.append(path[j])
        i = j
    return np.asarray(out)


def path_length(path) -> float:
    p = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if len(p) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def resample_by_arclength(path, n_points: int):
    """Uniform arc-length resample to exactly ``n_points`` samples."""
    p = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if len(p) == 1:
        return np.repeat(p, n_points, axis=0)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-9])
    p = p[keep]
    if len(p) < 2:
        return np.repeat(p[:1], n_points, axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    target = np.linspace(0.0, s[-1], n_points)
    return np.stack([np.interp(target, s, p[:, 0]),
                     np.interp(target, s, p[:, 1])], axis=1)


def tangent_yaw(path) -> np.ndarray:
    """Unwrapped heading of a polyline, one value per sample."""
    p = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if len(p) < 2:
        return np.zeros(len(p))
    d = np.diff(p, axis=0)
    yaw = np.arctan2(d[:, 1], d[:, 0])
    yaw = np.concatenate([yaw, yaw[-1:]])
    return np.unwrap(yaw)
