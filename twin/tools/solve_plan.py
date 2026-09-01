"""MPC 플래너를 **RViz 없이** 돌려 계획 CSV 를 뽑고, margin 별 성립 여부를 잰다.

왜 필요한가
-----------
``twin/world/map_extrude.py`` 의 여유거리 분석은 "필요 여유를 확보한 셀이 연결되어
있는가" 만 본다. 그것은 필요조건이지 충분조건이 아니다 — 실제로는 NLP 가 막대 회전,
리치 구간, 로봇-로봇 거리까지 동시에 만족해야 한다. **정말 풀리는지는 풀어 봐야 안다.**

그리고 Phase 1 의 시험 궤적이 어차피 필요하다.

실행 환경이 다르다
------------------
플래너는 CasADi 를 쓰는데 Isaac Lab conda env(``ppo_afl``) 에는 없다. 그래서 이
스크립트만 **casadi 가 설치된 다른 파이썬**으로 돈다 (개발 기계에서는 ``ros2``
conda env, casadi 3.7.2)::

    $TWIN_SOLVER_PY twin/tools/solve_plan.py ...

Isaac 은 전혀 안 쓰므로 아무 파이썬이나 상관없다 — ``casadi``, ``numpy``, ``scipy``
셋만 있으면 된다. rclpy 도 안 쓴다 (``solve_transport`` 는 순수 CasADi/numpy).

rclpy 는 안 쓴다. ``two_robot_nlp.solve_transport`` 는 순수 CasADi/numpy 다.

맵 변환에 대하여
----------------
``map_server`` 가 발행하는 ``/map`` 을 흉내내지 **않는다.** 이 맵은 free_thresh 0.25
때문에 미지(205)가 자유공간으로 발행되고, 그러면 ``unknown_is_occupied`` 가 무효가 되어
A* 가 건물 바깥을 가로지른다 (twin/README_KR.md 참고). 여기서는 원본 pgm 을
OccupancyGrid 규약(0 자유 / 100 점유 / -1 미지)으로 **정직하게** 옮긴다.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
from scipy import ndimage as ndi

_HERE = os.path.dirname(os.path.abspath(__file__))
#: 플래너 ROS 2 패키지. 이 repo 안에 있으므로 colcon 빌드 없이 import 된다
#: (``solve_transport`` 는 순수 CasADi/numpy 라 rclpy 가 필요 없다).
sys.path.insert(0, os.path.join(_HERE, "..", "..", "planner"))
sys.path.insert(0, os.path.join(_HERE, "..", "world"))

from map_extrude import classify, read_map_yaml, read_pgm  # noqa: E402
from robot_geom import BASE_LEN, BASE_WID, N_BASE_DISKS, N_OBJ_DISKS, OBJ_WID  # noqa: E402

#: 2D 라이다 맵의 벽 과대묘사 보정 [m]. 0.05 = 격자 한 칸.
MAP_INFLATION = 0.05
from mobile_manipulator_trajectory import grid_astar as ga  # noqa: E402
from mobile_manipulator_trajectory.sdf_utils import SignedDistanceField  # noqa: E402
from mobile_manipulator_trajectory.two_robot_nlp import (  # noqa: E402
    Geometry, Limits, SolverOpts, Weights, pose_clearance, solve_transport, verify)

LINES: list[str] = []


def say(m: str = "") -> None:
    print(m, flush=True)
    LINES.append(m)


def build_grid(map_yaml: str):
    """원본 pgm -> OccupancyGrid 규약 (0 자유 / 100 점유 / -1 미지)."""
    meta = read_map_yaml(map_yaml)
    img = read_pgm(os.path.join(os.path.dirname(map_yaml), str(meta["image"])))
    free, wall, unk = classify(img)
    data = np.full(img.shape, -1, dtype=np.int16)
    data[free] = 0
    data[wall] = 100
    # pgm 은 첫 행이 맵의 위쪽. OccupancyGrid 는 첫 행이 origin 쪽이므로 뒤집는다.
    return data[::-1].copy(), meta, (free[::-1].copy(), (wall | unk)[::-1].copy())


def bar_admissible(sdf, free_big, res, origin, geom, n_yaw: int = 36):
    """막대가 **어떤 요로든** 들어가는 자리의 마스크와 그때의 요.

    점 여유거리로 거르면 안 된다. 2 m 막대는 중심에서 ±0.8 m 까지 디스크가 뻗으므로,
    중심이 아무리 넓어도 양 끝이 벽에 걸리면 시작·목표 자세가 **하드 제약 위반**이 되어
    IPOPT 가 한 판을 통째로 태운 뒤에야 알려 준다.
    """
    ys, xs = np.nonzero(free_big)
    px = origin[0] + (xs + 0.5) * res
    py = origin[1] + (ys + 0.5) * res
    offs = -geom.obj_len / 2 + (2 * np.arange(geom.n_obj_disks) + 1) * (
        geom.obj_len / 2 / geom.n_obj_disks)
    r_need = float(np.hypot(geom.obj_wid / 2, (geom.obj_len / 2) / geom.n_obj_disks))

    best = np.full(len(xs), -1e9)
    best_yaw_ = np.zeros(len(xs))
    for yaw in np.linspace(0.0, math.pi, n_yaw, endpoint=False):   # 막대는 180도 대칭
        ux, uy = math.cos(yaw), math.sin(yaw)
        worst = np.full(len(xs), 1e9)
        for o in offs:
            v = sdf.value(np.stack([px + o * ux, py + o * uy], axis=1))
            worst = np.minimum(worst, v)
        upd = worst > best
        best[upd] = worst[upd]
        best_yaw_[upd] = yaw

    slack = best - r_need - geom.obj_margin
    mask = np.zeros(free_big.shape, dtype=bool)
    mask[ys[slack >= 0.0], xs[slack >= 0.0]] = True
    yawmap = np.zeros(free_big.shape)
    yawmap[ys, xs] = best_yaw_
    slackmap = np.full(free_big.shape, -1e9)
    slackmap[ys, xs] = slack
    return mask, yawmap, slackmap


def geodesic(comp, seed_yx):
    """``comp`` 안에서만 퍼지는 BFS. 벽을 뚫고 재면 통로를 안 지나는 쌍이 나온다."""
    d = np.full(comp.shape, np.inf)
    d[seed_yx] = 0.0
    frontier = [seed_yx]
    nbr = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while frontier:
        nxt = []
        for (y, x) in frontier:
            dc = d[y, x]
            for dy, dx in nbr:
                ny, nx_ = y + dy, x + dx
                if not comp[ny, nx_]:
                    continue
                nd = dc + (1.4142135 if dy and dx else 1.0)
                if nd < d[ny, nx_]:
                    d[ny, nx_] = nd
                    nxt.append((ny, nx_))
        frontier = nxt
    return d


def pick_endpoints(sdf, free_big, obst, res, origin, geom, target_m, say):
    """막대가 들어가는 자리들 중, **측지거리가 target_m 에 가장 가까운** 쌍.

    최원점을 쓰지 않는 이유는 예산이다. v29 프로파일(obj_v_max 0.12)에서 한 판이
    갈 수 있는 거리는 ``max_steps`` 에 묶여 있어서, 건물을 가로지르는 71 m 짜리
    쌍을 고르면 지평선이 잘려 계획이 목표에 못 닿는다.
    """
    clear = ndi.distance_transform_edt(~obst, sampling=res)
    adm, yawmap, slackmap = bar_admissible(sdf, free_big, res, origin, geom)
    lab, n = ndi.label(adm, structure=np.ones((3, 3)))
    if n == 0:
        return None, None
    sizes = ndi.sum(adm, lab, range(1, n + 1))
    comp = lab == (int(np.argmax(sizes)) + 1)
    say("  막대가 들어가는 자리 %.1f m^2 중 최대성분 %.1f m^2 (%d 성분)"
        % (adm.sum() * res * res, sizes.max() * res * res, n))

    # 넓은 곳에서 출발한다 — 시작 자세가 빡빡하면 초기 실현가능성부터 어렵다.
    cand = np.where(comp, clear, -1.0)
    a = np.unravel_index(int(np.argmax(cand)), cand.shape)
    d = geodesic(comp, a) * res
    d[~comp] = np.inf
    b = np.unravel_index(int(np.argmin(np.abs(d - target_m))), d.shape)
    say("  끝점 측지거리 %.1f m (목표 %.1f m, 직선 %.1f m)"
        % (d[b], target_m, res * math.hypot(float(b[0] - a[0]), float(b[1] - a[1]))))

    def pose(yx):
        return np.array([origin[0] + (yx[1] + 0.5) * res,
                         origin[1] + (yx[0] + 0.5) * res, yawmap[yx]]), slackmap[yx]

    pa, sa = pose(a)
    pb, sb = pose(b)
    say("  start (%.2f, %.2f, %+.1f deg) 막대여유 %+.3f m"
        % (pa[0], pa[1], math.degrees(pa[2]), sa))
    say("  goal  (%.2f, %.2f, %+.1f deg) 막대여유 %+.3f m"
        % (pb[0], pb[1], math.degrees(pb[2]), sb))
    return pa, pb


def best_yaw(sdf, xy, geom):
    """그 자리에서 막대가 가장 잘 들어가는 요를 고른다."""
    best, best_s = 0.0, -1e9
    for yaw in np.linspace(-math.pi, math.pi, 73)[:-1]:
        s = pose_clearance(sdf, np.array([xy[0], xy[1], yaw]),
                           geom.obj_len / 2, geom.obj_wid / 2,
                           geom.n_obj_disks, geom.obj_margin)
        if s > best_s:
            best, best_s = float(yaw), float(s)
    return best, best_s


def make_geom(base_margin: float, obj_margin: float = 0.05,
              obj_len: float = 2.00, n_obj_disks: int = 5) -> Geometry:
    """v29 프로파일 (twin/configs/policy_v29.yaml) + 주어진 여유.

    ``obj_margin`` 이 EE 여유를 정한다. 막대 끝 디스크 중심이 ±0.8 인데 EE 는
    ±1.0 이라 0.2 m 바깥이므로::

        EE 맵여유 = (0.25 + obj_margin) - 0.2 = 0.05 + obj_margin

    기본 0.05 면 EE 가 벽에서 **0.10 m** 밖에 안 떨어진다 (실측 일치).
    그리고 팔·그리퍼는 충돌 모델에 아예 없다.
    """
    return Geometry(base_len=BASE_LEN, base_wid=BASE_WID, n_base_disks=N_BASE_DISKS,
                    base_margin=base_margin,
                    obj_len=obj_len, obj_wid=OBJ_WID, n_obj_disks=n_obj_disks,
                    obj_margin=obj_margin,
                    reach_min=0.15, reach_max=0.60, ee_forward_min=-0.30,
                    map_inflation=MAP_INFLATION)


def make_limits() -> Limits:
    return Limits(v_max=0.30, v_min=-0.20, w_max=0.60, a_max=0.80, alpha_max=2.00,
                  obj_v_max=0.12, obj_w_max=0.08, ee_v_max=0.20, robot_robot_min=0.90)


def solve_once(sdf_full, start_obj, goal_obj, base_margin, args, say) -> dict:
    """노드 ``_plan`` 과 같은 순서로 한 판 푼다."""
    geom, lim = make_geom(base_margin, args.obj_margin, args.obj_len, args.n_obj_disks), make_limits()
    wts = Weights(ref=args.w_ref, base_ref=args.w_base_ref,
                  base_yaw_ref=args.w_base_yaw_ref, base_vel=args.w_base_vel,
                  collision=args.w_collision)
    opts = SolverOpts(max_iter=args.max_iter, print_level=0,
                      hessian="exact", sdf_interp="bspline")
    dt, n_settle = args.dt, args.n_settle
    out = {"margin": base_margin}

    t0 = time.time()
    raw = ga.astar(sdf_full, start_obj[:2], goal_obj[:2],
                   clearance=args.astar_clearance,
                   prefer_clearance=args.astar_prefer, clearance_weight=args.astar_w)
    if raw is None:
        out.update(stage="astar", ok=False, msg="A* 경로 없음 (clearance %.2f)"
                   % args.astar_clearance)
        return out
    seed = ga.shortcut(raw, sdf_full, args.astar_clearance)
    length = ga.path_length(seed)
    out["astar_len"] = length
    out["astar_s"] = time.time() - t0

    n_move = int(math.ceil(args.horizon_slack * length / max(1e-6, args.v_nom * dt)))
    n_cap = args.max_steps - n_settle
    n_move = int(np.clip(n_move, 20, n_cap))
    ref_move = ga.resample_by_arclength(seed, n_move + 1)
    ref_move[0], ref_move[-1] = start_obj[:2], goal_obj[:2]

    period = math.pi
    th = ga.tangent_yaw(ref_move)
    th = th + period * round((start_obj[2] - th[0]) / period)
    th = th + (start_obj[2] - th[0])
    goal_yaw_eff = goal_obj[2] + period * round((th[-1] - goal_obj[2]) / period)
    n_blend = max(5, n_move // 4)
    ramp = np.linspace(0.0, 1.0, n_blend)
    th[-n_blend:] = (1 - ramp) * th[-n_blend:] + ramp * goal_yaw_eff

    ref_xy = np.vstack([ref_move, np.repeat(goal_obj[:2][None, :], n_settle, axis=0)])
    yaw_guess = np.concatenate([th, np.full(n_settle, goal_yaw_eff)])
    goal_eff = np.array([goal_obj[0], goal_obj[1], goal_yaw_eff])
    out["N"] = len(ref_xy) - 1

    stand = geom.obj_len / 2 + geom.reach_max + max(geom.base_len, geom.base_wid)
    sdf = sdf_full.crop(ref_xy, margin=args.sdf_crop_margin + stand)
    ds = 1
    if sdf.nx * sdf.ny > args.sdf_max_cells:
        ds = int(math.ceil(math.sqrt(sdf.nx * sdf.ny / args.sdf_max_cells)))
    sdf = sdf.downsample(ds)
    out["sdf"] = (sdf.nx, sdf.ny, sdf.res)

    for tag, pose in (("start", start_obj), ("goal", goal_eff)):
        slack = pose_clearance(sdf, pose, geom.obj_len / 2, geom.obj_wid / 2,
                               geom.n_obj_disks, geom.obj_margin)
        if slack < 0.0:
            out.update(stage="boundary", ok=False,
                       msg="%s 물체 자세가 %.3f m 부족" % (tag, -slack))
            return out

    t0 = time.time()
    try:
        res = solve_transport(sdf, ref_xy, yaw_guess, start_obj, None, goal_eff,
                              dt, geom, lim, wts, opts, log=lambda *a, **k: None)
    except Exception as exc:  # noqa: BLE001
        out.update(stage="ipopt", ok=False, msg="예외: %s" % exc)
        return out
    out["ipopt_s"] = time.time() - t0
    out["success"] = bool(res["success"])
    out["iters"] = res["iters"]

    rep = verify(sdf, res, geom, lim)
    out["rep"] = rep
    out["goal_err"] = float(np.linalg.norm(res["obj_xy"][-1] - goal_obj[:2]))
    out["res"] = res
    out.update(stage="done", ok=bool(res["success"]) and bool(rep["executable"]),
               msg="")
    return out


def export_csv(res, out_dir: str) -> None:
    import csv
    os.makedirs(out_dir, exist_ok=True)
    rows = {"object": (res["obj_xy"], res["obj_yaw"]),
            "base_A": (res["base1_xy"], res["base1_yaw"]),
            "base_B": (res["base2_xy"], res["base2_yaw"]),
            "ee_A": (res["ee1_xy"], res["ee1_yaw"]),
            "ee_B": (res["ee2_xy"], res["ee2_yaw"])}
    for name, (xy, yaw) in rows.items():
        with open(os.path.join(out_dir, "%s.csv" % name), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "x", "y", "yaw"])
            for t, (x, y), th in zip(res["t"], xy, yaw):
                w.writerow(["%.4f" % t, "%.5f" % x, "%.5f" % y, "%.6f" % th])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="handoff/map_munji_3f_2025_wide/munji_3f_2025_wide.yaml")
    ap.add_argument("--margins", type=float, nargs="*",
                    default=[0.05, 0.20, 0.30, 0.50, 0.70])
    ap.add_argument("--out-dir", dest="out_dir", default="twin/data/plans")
    ap.add_argument("--start", type=float, nargs=3, default=None)
    ap.add_argument("--goal", type=float, nargs=3, default=None)
    ap.add_argument("--dt", type=float, default=0.15)
    ap.add_argument("--v-nom", dest="v_nom", type=float, default=0.10)
    ap.add_argument("--horizon-slack", dest="horizon_slack", type=float, default=1.25)
    ap.add_argument("--n-settle", dest="n_settle", type=int, default=12)
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=700)
    ap.add_argument("--astar-clearance", dest="astar_clearance", type=float, default=0.35)
    ap.add_argument("--astar-prefer", dest="astar_prefer", type=float, default=0.90)
    ap.add_argument("--astar-w", dest="astar_w", type=float, default=0.60)
    ap.add_argument("--sdf-crop-margin", dest="sdf_crop_margin", type=float, default=3.0)
    ap.add_argument("--sdf-max-cells", dest="sdf_max_cells", type=int, default=400_000)
    ap.add_argument("--max-iter", dest="max_iter", type=int, default=3000)
    ap.add_argument("--target-dist", dest="target_dist", type=float, default=0.0,
                    help="끝점 사이 측지거리 목표 [m]. 0 이면 지평선 예산의 85%%")
    ap.add_argument("--obj-margin", dest="obj_margin", type=float, default=0.05,
                    help="막대 여유. EE 맵여유 = 0.05 + 이 값")
    ap.add_argument("--obj-len", dest="obj_len", type=float, default=2.00,
                    help="막대 길이. 짧으면 통과 가능 면적이 늘어 base_margin 을 더 줄 수 있다")
    ap.add_argument("--n-obj-disks", dest="n_obj_disks", type=int, default=N_OBJ_DISKS,
                    help="막대 원판 수. 늘리면 EE 여유가 는다 (L=1.5 기준 5->10 이면 0.112->0.143)")
    ap.add_argument("--w-collision", dest="w_collision", type=float, default=5000.0,
                    help="충돌 슬랙 벌점. 다른 항을 압도해야 슬랙이 최후수단이 된다")
    ap.add_argument("--w-ref", dest="w_ref", type=float, default=20.0,
                    help="물체가 계획 경로(A* 시드)를 따라가는 가중치")
    ap.add_argument("--w-base-ref", dest="w_base_ref", type=float, default=3.0)
    ap.add_argument("--w-base-yaw-ref", dest="w_base_yaw_ref", type=float, default=3.0)
    ap.add_argument("--w-base-vel", dest="w_base_vel", type=float, default=2.0)
    ap.add_argument("--tag", default="", help="출력 디렉토리 접미사")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    data, meta, (free, obst) = build_grid(args.map)
    res_m = float(meta["resolution"])
    origin = meta["origin"]

    say("=" * 78)
    say("MPC 플래너 헤드리스 — margin 별 성립 여부")
    say("=" * 78)
    t0 = time.time()
    sdf_full = SignedDistanceField.from_occupancy(
        data, origin=(origin[0], origin[1]), res=res_m,
        unknown_is_occupied=True, smooth_sigma_cells=1.0)
    say("  SDF  %dx%d @ %.3f m  범위 %.2f..%.2f m  (%.1fs)"
        % (sdf_full.nx, sdf_full.ny, sdf_full.res,
           sdf_full.sdf.min(), sdf_full.sdf.max(), time.time() - t0))

    lab, n = ndi.label(free, structure=np.ones((3, 3)))
    sizes = ndi.sum(free, lab, range(1, n + 1))
    free_big = lab == (int(np.argmax(sizes)) + 1)

    geom0 = make_geom(0.05, args.obj_margin, args.obj_len, args.n_obj_disks)
    # 한 판의 이동 예산. obj_v_max 가 낮은 v29 프로파일에서는 이것이 곧 계획 길이 상한이다.
    budget_m = (args.max_steps - args.n_settle) * args.dt * args.v_nom / args.horizon_slack
    say("  지평선 예산  max_steps %d, dt %.2f, v_nom %.2f -> 한 판 최대 %.1f m"
        % (args.max_steps, args.dt, args.v_nom, budget_m))
    if args.start and args.goal:
        start_obj, goal_obj = np.array(args.start), np.array(args.goal)
    else:
        target = args.target_dist if args.target_dist > 0 else budget_m * 0.85
        start_obj, goal_obj = pick_endpoints(
            sdf_full, free_big, obst, res_m, origin, geom0, target, say)
        if start_obj is None:
            raise SystemExit("막대가 들어가는 자리를 못 찾았다")

    say("")
    say("  %-8s %-9s %6s %8s %9s %9s %10s   %s"
        % ("margin", "단계", "N", "IPOPT s", "최소여유", "목표오차", "동역학", "판정"))
    say("-" * 78)
    for mg in args.margins:
        r = solve_once(sdf_full, start_obj, goal_obj, mg, args, say)
        if r["stage"] in ("astar", "boundary"):
            say("  %-8.2f %-9s %6s %8s %9s %9s %10s   %s"
                % (mg, r["stage"], "-", "-", "-", "-", "-", r["msg"]))
            continue
        rep = r.get("rep")
        wc = rep["worst_clearance"] if rep else float("nan")
        dyn = rep["worst_dynamics_residual"] if rep else float("nan")
        verdict = ("성립" if r["ok"] else
                   ("IPOPT 미수렴" if not r.get("success") else
                    "충돌" if rep and not rep["collision_free"] else "동역학 불가"))
        say("  %-8.2f %-9s %6d %8.1f %+9.3f %9.3f %10.1e   %s"
            % (mg, r["stage"], r["N"], r.get("ipopt_s", 0.0), wc,
               r.get("goal_err", float("nan")), dyn, verdict))
        res = r.get("res")
        if res is not None:
            say("           슬랙 최대 %.4f m (%d 샘플)   물체-시드 편차 평균 %.3f 최대 %.3f m"
                % (res["collision_slack_max"], res["collision_slack_n"],
                   res["ref_dev_mean"], res["ref_dev_max"]))
        if r["ok"]:
            d = os.path.join(args.out_dir, "L%03d_base%03d_obj%03d%s"
                             % (int(round(args.obj_len * 100)),
                                int(round(mg * 100)),
                                int(round(args.obj_margin * 100)),
                                ("_" + args.tag) if args.tag else ""))
            export_csv(r["res"], d)
            say("           -> CSV %s" % d)
    say("-" * 78)

    if args.report:
        with open(args.report, "w") as fh:
            fh.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
