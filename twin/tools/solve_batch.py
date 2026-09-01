"""여러 경로를 연달아 풀어 **실패율**을 잰다.

왜 필요한가
-----------
"순간이동이 자주 나온다" 를 고치려면 먼저 얼마나 자주인지 알아야 한다. 한 경로만
보고 고치면 그 경로에만 맞춘 변경이 되고, 이 저장소는 이미 그 실수를 두 번 했다
(``eval_plans.py`` 주석의 "표본 하나로 판정하지 않는다").

무엇을 세나
-----------
::

    성립       IPOPT 수렴 + verify 통과
    미수렴     IPOPT 가 못 찾음 -> 마지막 반복치.  이게 순간이동의 정체다
    불연속     한 스텝 이동량이 v_max*dt 를 넘음 (순간이동이 실제로 일어난 판)
    충돌       수렴했는데 여유가 음수

사용법
------
::

    <ros2 conda python> twin/tools/solve_batch.py --n 6 --report /tmp/batch.txt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "world"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "planner"))

import solve_plan as sp  # noqa: E402
from mobile_manipulator_trajectory.sdf_utils import SignedDistanceField  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=6, help="풀 경로 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obj-len", dest="obj_len", type=float, default=1.5)
    ap.add_argument("--margin", type=float, default=0.18)
    ap.add_argument("--dist", type=float, default=6.0, help="경로 목표 길이 [m]")
    ap.add_argument("--report", default="")
    a = ap.parse_args()

    lines = []

    def say(m=""):
        print(m, flush=True)
        lines.append(m)

    data, meta, (free, obst) = sp.build_grid(
        "handoff/map_munji_3f_2025_wide/munji_3f_2025_wide.yaml")
    res = float(meta["resolution"])
    origin = meta["origin"]
    sdf = SignedDistanceField.from_occupancy(
        data, origin=(origin[0], origin[1]), res=res,
        unknown_is_occupied=True, smooth_sigma_cells=1.0)

    from scipy import ndimage as ndi
    lab, n = ndi.label(free, structure=np.ones((3, 3)))
    sz = ndi.sum(free, lab, range(1, n + 1))
    big = lab == (int(np.argmax(sz)) + 1)

    geom0 = sp.make_geom(a.margin, 0.05, a.obj_len, sp.N_OBJ_DISKS)
    adm, yawmap, slackmap = sp.bar_admissible(sdf, big, res, origin, geom0)
    ys, xs = np.nonzero(adm)
    rng = np.random.default_rng(a.seed)

    # 인자 객체를 solve_once 가 기대하는 모양으로 만든다
    class A:
        pass
    args = A()
    for k, v in (("dt", 0.15), ("v_nom", 0.10), ("horizon_slack", 1.25),
                 ("n_settle", 12), ("max_steps", 700), ("astar_clearance", 0.35),
                 ("astar_prefer", 0.90), ("astar_w", 0.60), ("sdf_crop_margin", 3.0),
                 ("sdf_max_cells", 400_000), ("max_iter", 3000),
                 ("obj_margin", 0.05), ("obj_len", a.obj_len),
                 ("n_obj_disks", sp.N_OBJ_DISKS),
                 ("w_base_ref", 3.0), ("w_base_yaw_ref", 0.0), ("w_base_vel", 8.0),
                 ("w_ref", 20.0), ("w_collision", 1.0e6)):
        setattr(args, k, v)
    sp.args = args

    say("=" * 78)
    say("배치 — %d 경로, obj_len %.2f, base_margin %.2f" % (a.n, a.obj_len, a.margin))
    say("=" * 78)
    say("  %-4s %8s %8s %9s %10s %10s %12s  %s"
        % ("#", "길이m", "IPOPT s", "최소여유", "목표오차", "동역학", "최대스텝", "판정"))
    say("-" * 78)

    tally = {"성립": 0, "미수렴": 0, "불연속": 0, "충돌": 0, "슬랙사용": 0,
             "A*없음": 0, "경계": 0}
    t_all = time.time()
    tried = 0
    made = 0
    while made < a.n and tried < a.n * 8:
        tried += 1
        i, j = rng.integers(0, len(xs), 2)
        p0 = np.array([origin[0] + (xs[i] + 0.5) * res, origin[1] + (ys[i] + 0.5) * res,
                       yawmap[ys[i], xs[i]]])
        p1 = np.array([origin[0] + (xs[j] + 0.5) * res, origin[1] + (ys[j] + 0.5) * res,
                       yawmap[ys[j], xs[j]]])
        d = float(np.linalg.norm(p1[:2] - p0[:2]))
        if not (a.dist * 0.7 <= d <= a.dist * 1.3):
            continue
        made += 1
        r = sp.solve_once(sdf, p0, p1, a.margin, args, lambda *x, **k: None)
        if r["stage"] == "astar":
            tally["A*없음"] += 1
            say("  %-4d %8.2f %8s %9s %10s %10s %12s  A* 경로 없음" % (made, d, "-", "-", "-", "-", "-"))
            continue
        if r["stage"] == "boundary":
            tally["경계"] += 1
            say("  %-4d %8.2f %8s %9s %10s %10s %12s  %s" % (made, d, "-", "-", "-", "-", "-", r["msg"]))
            continue
        rep = r["rep"]
        c = rep["continuity"]
        mx = max(c["base1"]["max_step_m"], c["base2"]["max_step_m"])
        slack = r["res"]["collision_slack_max"] if r.get("res") is not None else 0.0
        if not r.get("success"):
            v = "미수렴"
        elif slack > 1e-3:
            v = "슬랙사용"
        elif not rep["continuity"]["continuous"]:
            v = "불연속"
        elif not rep["collision_free"]:
            v = "충돌"
        else:
            v = "성립"
        tally[v] += 1
        say("  %-4d %8.2f %8.1f %+9.3f %10.3f %10.1e %12.4f  %-8s 슬랙 %.4f"
            % (made, d, r["ipopt_s"], rep["worst_clearance"], r["goal_err"],
               rep["worst_dynamics_residual"], mx, v, slack))
    say("-" * 78)
    tot = sum(tally.values())
    say("  %d 경로, %.0f s" % (tot, time.time() - t_all))
    say("  " + "   ".join("%s %d (%.0f%%)" % (k, v, 100.0 * v / max(1, tot))
                          for k, v in tally.items() if v))
    ok = tally["성립"]
    say("  성립률 %.0f%%" % (100.0 * ok / max(1, tot)))
    if a.report:
        open(a.report, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
